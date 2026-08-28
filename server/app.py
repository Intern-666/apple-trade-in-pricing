# ============================================================
# APPLE TRADE-IN VALUATION API
# CUSTOMER-FACING API
# ============================================================

import joblib
import pandas as pd
import numpy as np
import sys
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional, cast
from pydantic import BaseModel
from datetime import datetime
from internal.tradein_fallback import TradeInFallback
from internal.sheets_sync import SheetsSync


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# app.py's own directory -- internal/ (including the service
# account credentials file) lives alongside app.py, one level
# below BASE_DIR.
APP_DIR = Path(__file__).resolve().parent

ASSETS_DIR = BASE_DIR / "assets"

DATA_FILE = BASE_DIR / "data" / "master_msrp.csv"

MODEL_FILE = BASE_DIR / "model" / "model_v2.pkl"

FITTED_CURVES_FILE = BASE_DIR / "data" / "fitted_curves.csv"

# ------------------------------------------------------------
# Google Sheets sync configuration.
#
# The service account key file is never committed to source
# control -- it must be placed manually at this path on the
# server. If it's missing, Sheets sync is disabled and the app
# still runs normally against the local CSV (see SheetsSync).
# ------------------------------------------------------------

SHEETS_SERVICE_ACCOUNT_FILE = (
    os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_FILE",
        str(BASE_DIR / "internal" / "service_account.json")
    )
)

SHEETS_SERVICE_ACCOUNT_JSON = os.getenv(
    "GOOGLE_SERVICE_ACCOUNT_JSON"
)

SHEETS_SPREADSHEET_ID = (
    "1TzySGhtEs-ptmzLHNxcJ5q_lQ9nGr6HDofy0IL7G1vs"
)

SHEETS_WORKSHEET_NAME = "Cleaned Master"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Apple Trade-In Valuation API"
)

app.mount(
    "/assets",
    StaticFiles(directory=ASSETS_DIR),
    name="assets",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING APPLE TRADE-IN DATA")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print(f"Rows loaded: {len(df)}")
print(f"Master dataset: {DATA_FILE.name}")

model = joblib.load(MODEL_FILE)

print(f"Model loaded: {MODEL_FILE.name}")

# ============================================================
# LOAD DEPRECIATION FALLBACK
# ============================================================

fallback = TradeInFallback(
    str(FITTED_CURVES_FILE),
    raw_data_path=str(DATA_FILE),
)

print(
    f"Depreciation curves loaded: "
    f"{FITTED_CURVES_FILE.name}"
)

# ============================================================
# GOOGLE SHEETS SYNC
# ============================================================

sheets_sync = SheetsSync(
    service_account_file=str(SHEETS_SERVICE_ACCOUNT_FILE),
    service_account_json=SHEETS_SERVICE_ACCOUNT_JSON,
    spreadsheet_id=SHEETS_SPREADSHEET_ID,
    worksheet_name=SHEETS_WORKSHEET_NAME,
)

if sheets_sync.is_available:

    print(
        "Google Sheets sync ready -> "
        f"worksheet '{SHEETS_WORKSHEET_NAME}'"
    )

else:

    print(
        "Google Sheets sync UNAVAILABLE -- admin writes will "
        "still save to the local CSV, but will not be mirrored "
        "to Google Sheets until this is resolved."
    )


# ============================================================
# CLEAN STORAGE
# ============================================================

def storage_to_gb(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if "tb" in value:

        numbers = [
            x for x in value.replace(",", "").split()
            if x.replace(".", "", 1).isdigit()
        ]

        if numbers:
            return float(numbers[0]) * 1024

    if "gb" in value:

        numbers = [
            x for x in value.replace(",", "").split()
            if x.replace(".", "", 1).isdigit()
        ]

        if numbers:
            return float(numbers[0])

    try:
        return float(value)

    except (ValueError, TypeError):
        return np.nan


df["Storage (GB)"] = (
    df["Storage (GB)"]
    .apply(storage_to_gb)
)


# ============================================================
# CLEAN TARGET
# ============================================================

df["Max. Trade-In Value (RM)"] = pd.to_numeric(
    df["Max. Trade-In Value (RM)"],
    errors="coerce"
)

# Keep records with missing trade-in values.
# Admin needs these records so they can later be modified.
df = df.copy()


# ============================================================
# CLEAN TEXT COLUMNS
# ============================================================

TEXT_COLUMNS = [
    "Device",
    "Sub-device",
    "Standardized Model",
    "Provider",
    "Storage Type",
    "Connectivity",
    "Chipset",
    "Mac Generation"
]


for col in TEXT_COLUMNS:

    if col in df.columns:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )


# ============================================================
# BUILD DEVICE MODEL MAP
# ============================================================

device_model_map = {}


for (
    device,
    sub_device,
    model_name
), group in df.groupby(
    [
        "Device",
        "Sub-device",
        "Standardized Model"
    ]
):

    if device not in device_model_map:
        device_model_map[device] = {}

    if sub_device not in device_model_map[device]:
        device_model_map[device][sub_device] = {}

    storages = (
        group["Storage (GB)"]
        .dropna()
        .unique()
        .tolist()
    )

    storages = sorted(storages)

    clean_storages = [
        int(x) if float(x).is_integer() else float(x)
        for x in storages
    ]

    device_model_map[
        device
    ][
        sub_device
    ][
        model_name
    ] = clean_storages


print(
    f"Devices available: {len(device_model_map)}"
)

print("=" * 70)


# ============================================================
# AVAILABLE MODELS
# ============================================================

@app.get("/available-models")
def get_models():

    return device_model_map


# ============================================================
# REQUEST MODEL
# ============================================================

class DeviceInput(BaseModel):
    Device: str
    SubDevice: str
    Model: str
    Storage: float | None = None


# ============================================================
# EXACT DEVICE MEDIAN VALUATION
# ============================================================

@app.post("/predict")
def predict_price(item: DeviceInput):

    print("\n" + "=" * 70)
    print("CUSTOMER VALUATION REQUEST")
    print("=" * 70)

    print(f"Device     : {item.Device}")
    print(f"Sub-Device : {item.SubDevice}")
    print(f"Model      : {item.Model}")
    print(f"Storage    : {item.Storage} GB")


    # ========================================================
    # AIRPODS
    # AirPods do not have storage
    # ========================================================

    if item.Device == "AirPods":

        matches = df[
            (df["Device"] == item.Device)
            &
            (df["Sub-device"] == item.SubDevice)
            &
            (df["Standardized Model"] == item.Model)
        ].copy()


    # ========================================================
    # ALL OTHER DEVICES
    # Storage is required
    # ========================================================

    else:

        if item.Storage is None:

            print("Storage is required for this device.")

            return {
                "status": "unresolved",
                "estimated_value": None,
                "message": (
                    "Storage is required for this device."
                )
            }


        matches = df[
            (df["Device"] == item.Device)
            &
            (df["Sub-device"] == item.SubDevice)
            &
            (df["Standardized Model"] == item.Model)
            &
            (
                np.isclose(
                    df["Storage (GB)"],
                    item.Storage,
                    equal_nan=False
                )
            )
        ].copy()


    # ========================================================
    # NO MATCH
    # ========================================================

    if matches.empty:

        print("No exact database match.")

        return {
            "status": "unresolved",
            "estimated_value": None,
            "message": (
                "This exact device configuration "
                "is not available in the database."
            )
        }


    # ========================================================
    # MEDIAN TRADE-IN VALUE
    # ========================================================

    median_price = (
        matches["Max. Trade-In Value (RM)"]
        .median()
    )


    # ========================================================
    # SUPPORTING INFORMATION
    # ========================================================

    provider_count = (
        matches["Provider"]
        .nunique()
    )

    record_count = len(matches)


    print(
        f"Matching records : {record_count}"
    )

    print(
        f"Providers         : {provider_count}"
    )

    print(
        f"Median value      : RM {median_price:,.2f}"
    )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "status": "resolved",

        "estimated_value": round(
            float(median_price),
            2
        ),

        "method": "exact_configuration_median",

        "matching_records": record_count,

        "provider_count": provider_count
    }

# ============================================================
# ADMIN ADD REQUEST
# ============================================================

class AdminAddDevice(BaseModel):

    Device: str
    SubDevice: str
    Model: str

    Provider: Optional[str] = None

    MSRP: float

    TradeInValue: Optional[float] = None

    Storage: Optional[float] = None
    StorageType: Optional[str] = None

    ModelYear: Optional[int] = None

    Chipset: Optional[str] = None
    Connectivity: Optional[str] = None

    ScreenSize: Optional[str] = None

    Material: Optional[str] = None
    CaseSize: Optional[str] = None

    ChargingMethod: Optional[str] = None

    RAMConfigurations: Optional[str] = None
    RAMMin: Optional[float] = None

    MacStorageCategory: Optional[str] = None
    MacGeneration: Optional[str] = None

# ============================================================
# ADMIN — ADD DEVICE
# ============================================================

@app.post("/admin/add")
def admin_add_device(item: AdminAddDevice):

    global df

    print("\n" + "=" * 70)
    print("ADMIN — ADD DEVICE")
    print("=" * 70)

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    device = item.Device.strip()
    sub_device = item.SubDevice.strip()
    model_name = item.Model.strip()

    if not device:
        return {
            "status": "error",
            "message": "Device is required."
        }

    if not sub_device:
        return {
            "status": "error",
            "message": "Sub-device is required."
        }

    if not model_name:
        return {
            "status": "error",
            "message": "Model is required."
        }

    # ========================================================
    # MSRP
    # ========================================================

    if item.MSRP < 0:

        return {
            "status": "error",
            "message": "MSRP cannot be negative."
        }

    msrp = float(item.MSRP)

    # ========================================================
    # PROVIDER
    # ========================================================

    provider = (
        item.Provider.strip()
        if item.Provider
        else "Unknown"
    )

    # ========================================================
    # TRADE-IN VALUE
    # ========================================================

    if item.TradeInValue is None:

        trade_in_value = np.nan
        price_status = "N/A"

    else:

        if item.TradeInValue < 0:

            return {
                "status": "error",
                "message":
                    "Trade-in value cannot be negative."
            }

        trade_in_value = float(item.TradeInValue)
        price_status = "confirmed"

    # ========================================================
    # STORAGE
    # ========================================================

    storage = item.Storage

    storage_type = (
        item.StorageType.strip()
        if item.StorageType
        else np.nan
    )

    # AirPods do not have storage

    if device == "AirPods":

        storage = np.nan
        storage_type = np.nan

    # ========================================================
    # DEVICE-SPECIFIC VALUES
    # ========================================================

    chipset = (
        item.Chipset.strip()
        if item.Chipset
        else np.nan
    )

    connectivity = (
        item.Connectivity.strip()
        if item.Connectivity
        else np.nan
    )

    screen_size = (
        item.ScreenSize.strip()
        if item.ScreenSize
        else np.nan
    )

    material = (
        item.Material.strip()
        if item.Material
        else np.nan
    )

    case_size = (
        item.CaseSize.strip()
        if item.CaseSize
        else np.nan
    )

    charging_method = (
        item.ChargingMethod.strip()
        if item.ChargingMethod
        else np.nan
    )

    ram_configurations = (
        item.RAMConfigurations.strip()
        if item.RAMConfigurations
        else np.nan
    )

    # ========================================================
    # NON-MAC RAM
    # ========================================================

    if device != "Mac":

        ram_configurations = np.nan
        ram_min = np.nan

    else:

        if (
            item.RAMMin is not None
            and item.RAMMin < 0
        ):

            return {
                "status": "error",
                "message": "RAM cannot be negative."
            }

        ram_min = item.RAMMin

    # ========================================================
    # MAC-SPECIFIC
    # ========================================================

    mac_storage_category = (
        item.MacStorageCategory.strip()
        if item.MacStorageCategory
        else np.nan
    )

    mac_generation = (
        item.MacGeneration.strip()
        if item.MacGeneration
        else np.nan
    )

    if device != "Mac":

        mac_storage_category = np.nan
        mac_generation = np.nan

    # ========================================================
    # MODEL YEAR
    # ========================================================

    model_year = item.ModelYear

    if model_year is not None:

        if model_year < 1976:

            return {
                "status": "error",
                "message": "Invalid Apple model year."
            }

    # ========================================================
    # CREATE NEW ROW
    # ========================================================

    new_row = {

        "Provider":
            provider,

        "Device":
            device,

        "Sub-device":
            sub_device,

        "Standardized Model":
            model_name,

        "MSRP":
            msrp,

        "Specification":
            np.nan,

        "Storage (GB)":
            storage,

        "Storage Type":
            storage_type,

        "Connectivity":
            connectivity,

        "Material":
            material,

        "Max. Trade-In Value (RM)":
            trade_in_value,

        "Model_Year":
            model_year,

        "Chipset":
            chipset,

        "Screen Size":
            screen_size,

        "Case Size":
            case_size,

        "Charging Method":
            charging_method,

        "RAM Configurations (GB)":
            ram_configurations,

        "RAM Min (GB)":
            ram_min,

        "Mac Storage Category":
            mac_storage_category,

        "Mac Generation":
            mac_generation
    }

    # ========================================================
    # APPEND TO DATAFRAME
    # ========================================================

    df = pd.concat(
        [
            df,
            pd.DataFrame([new_row])
        ],
        ignore_index=True
    )

    # Save updated master dataset
    df.to_csv(
        DATA_FILE,
        index=False
    )

    # ========================================================
    # SYNC TO GOOGLE SHEETS (best-effort, non-blocking)
    # ========================================================

    sync_result = sheets_sync.sync_dataset(df)

    if not sync_result.success:

        print(
            "WARNING: Google Sheets sync failed after ADD: "
            f"{sync_result.error}"
        )

    # ========================================================
    # LOG
    # ========================================================

    print(
        f"Device       : {device}"
    )

    print(
        f"Sub-device   : {sub_device}"
    )

    print(
        f"Model        : {model_name}"
    )

    print(
        f"Provider     : {provider}"
    )

    print(
        f"MSRP         : RM {msrp:,.2f}"
    )

    print(
        f"Trade-in     : "
        f"{'N/A' if pd.isna(trade_in_value) else f'RM {trade_in_value:,.2f}'}"
    )

    print(
        "Device added successfully."
    )

    print("=" * 70)
    
    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "status":
            "success",

        "message":
            "Device added successfully.",

        "price_status":
            price_status,

        "model":
            model_name,

        "msrp":
            msrp,

        "trade_in_value":
            (
                None
                if pd.isna(trade_in_value)
                else trade_in_value
            ),

        "sheets_sync":
            sync_result.as_dict()

    }

# ============================================================
# ADMIN — TRADE-IN DATA STATUS
# ============================================================

@app.get("/admin/status")
def admin_status():

    total_records = len(df)

    missing_trade_in = (
        df["Max. Trade-In Value (RM)"]
        .isna()
        .sum()
    )

    confirmed_trade_in = (
        total_records
        -
        missing_trade_in
    )

    if total_records > 0:

        missing_percentage = (
            missing_trade_in
            /
            total_records
            *
            100
        )

    else:

        missing_percentage = 0

    return {

        "total_records":
            int(total_records),

        "confirmed_trade_in":
            int(confirmed_trade_in),

        "missing_trade_in":
            int(missing_trade_in),

        "missing_percentage":
            round(
                float(missing_percentage),
                2
            )

    }

# ============================================================
# ADMIN — AVAILABLE MODELS
# ============================================================

@app.get("/admin/models")
def admin_models():

    models = {}

    for device, group in df.groupby("Device"):

        values = (
            group["Standardized Model"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        models[device] = sorted(values)

    return models

# ============================================================
# ADMIN — FIND RECORDS
# ============================================================

@app.get("/admin/records")
def admin_records(
    device: str,
    model: str
):

    matches = df[
        (df["Device"] == device)
        &
        (df["Standardized Model"] == model)
    ]


    records = []


    for index, row in matches.iterrows():

        record_id = int(cast(int, index))


        # ----------------------------------------------------
        # TRADE-IN VALUE
        # ----------------------------------------------------

        value = row["Max. Trade-In Value (RM)"]

        if pd.isna(value):
            value = None

        else:
            value = float(value)


        # ----------------------------------------------------
        # STORAGE
        # ----------------------------------------------------

        storage = row["Storage (GB)"]

        if pd.isna(storage):
            storage = None

        else:
            storage = float(storage)


        # ----------------------------------------------------
        # HELPER
        # ----------------------------------------------------

        def clean_value(column):

            if column not in row.index:
                return None

            value = row[column]

            if pd.isna(value):
                return None

            return str(value).strip()


        # ----------------------------------------------------
        # BUILD RECORD
        # ----------------------------------------------------

        records.append({

            "id":
                record_id,

            "model":
                clean_value(
                    "Standardized Model"
                ),

            "storage":
                storage,

            "storage_type":
                clean_value(
                    "Storage Type"
                ),

            "connectivity":
                clean_value(
                    "Connectivity"
                ),

            "material":
                clean_value(
                    "Material"
                ),

            "ram":
                (
                    None
                    if (
                        "RAM Min (GB)" not in row.index
                        or pd.isna(row["RAM Min (GB)"])
                    )
                    else (
                        int(row["RAM Min (GB)"])
                        if float(row["RAM Min (GB)"]).is_integer()
                        else float(row["RAM Min (GB)"])
                    )
                ),

            "chipset":
                clean_value(
                    "Chipset"
                ),

            "mac_generation":
                clean_value(
                    "Mac Generation"
                ),

            "provider":
                clean_value(
                    "Provider"
                ),

            "msrp":
                (
                    None
                    if pd.isna(row["MSRP"])
                    else float(row["MSRP"])
                ),

            "trade_in_value":
                value

        })


    return records

# ============================================================
# ADMIN MODIFY REQUEST
# ============================================================

class AdminModifyDevice(BaseModel):

    id: int

    Provider: Optional[str] = None

    MSRP: Optional[float] = None

    TradeInValue: Optional[float] = None

@app.post("/admin/modify")
def admin_modify_device(
    item: AdminModifyDevice
):

    global df

    print("\n" + "=" * 70)
    print("ADMIN — MODIFY DEVICE")
    print("=" * 70)

    # ========================================================
    # CHECK RECORD
    # ========================================================

    if item.id < 0 or item.id >= len(df):

        return {
            "status": "error",
            "message": "Record not found."
        }

    # ========================================================
    # VALIDATE MSRP
    # ========================================================

    if (
        item.MSRP is not None
        and item.MSRP < 0
    ):

        return {
            "status": "error",
            "message": "MSRP cannot be negative."
        }

    # ========================================================
    # VALIDATE TRADE-IN VALUE
    # ========================================================

    if (
        item.TradeInValue is not None
        and item.TradeInValue < 0
    ):

        return {
            "status": "error",
            "message":
                "Trade-in value cannot be negative."
        }

    # ========================================================
    # PROVIDER
    # ========================================================

    if item.Provider is not None:

        provider = item.Provider.strip()

        if not provider:

            return {
                "status": "error",
                "message": "Provider cannot be empty."
            }

        old_provider = df.at[
            item.id,
            "Provider"
        ]

        df.at[
            item.id,
            "Provider"
        ] = provider

        print(
            f"Provider updated: "
            f"{old_provider} → {provider}"
        )

    # ========================================================
    # MSRP
    # ========================================================

    if item.MSRP is not None:

        old_msrp = df.at[
            item.id,
            "MSRP"
        ]

        df.at[
            item.id,
            "MSRP"
        ] = float(item.MSRP)

        print(
            f"MSRP updated: "
            f"{'N/A' if pd.isna(old_msrp) else f'RM {old_msrp:,.2f}'}"
            f" → RM {item.MSRP:,.2f}"
        )

    # ========================================================
    # TRADE-IN VALUE
    # ========================================================

    if item.TradeInValue is not None:

        old_value = df.at[
            item.id,
            "Max. Trade-In Value (RM)"
        ]

        df.at[
            item.id,
            "Max. Trade-In Value (RM)"
        ] = float(item.TradeInValue)

        print(
            f"Trade-in value updated: "
            f"RM {item.TradeInValue:,.2f}"
        )

    else:

        old_value = df.at[
            item.id,
            "Max. Trade-In Value (RM)"
        ]

        df.at[
            item.id,
            "Max. Trade-In Value (RM)"
        ] = np.nan

        print(
            "Trade-in value changed to N/A."
        )

    # ========================================================
    # SAVE MASTER DATASET
    # ========================================================

    df.to_csv(
        DATA_FILE,
        index=False
    )

    # ========================================================
    # SYNC TO GOOGLE SHEETS (best-effort, non-blocking)
    # ========================================================

    sync_result = sheets_sync.sync_dataset(df)

    if not sync_result.success:

        print(
            "WARNING: Google Sheets sync failed after MODIFY: "
            f"{sync_result.error}"
        )

    # ========================================================
    # LOG
    # ========================================================

    print(
        f"Record index : {item.id}"
    )

    print(
        "Record updated successfully."
    )

    print("=" * 70)

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "status":
            "success",

        "message":
            "Record updated successfully.",

        "id":
            item.id,

        "provider":
            (
                str(df.at[item.id, "Provider"])
                if not pd.isna(df.at[item.id, "Provider"])
                else None
            ),

        "msrp":
            (
                None
                if pd.isna(df.at[item.id, "MSRP"])
                else float(df.at[item.id, "MSRP"])
            ),

        "trade_in_value":
            (
                None
                if pd.isna(
                    df.at[
                        item.id,
                        "Max. Trade-In Value (RM)"
                    ]
                )
                else float(
                    df.at[
                        item.id,
                        "Max. Trade-In Value (RM)"
                    ]
                )
            ),

        "sheets_sync":
            sync_result.as_dict()

    }

# ============================================================
# ADMIN — DELETE DEVICE
# ============================================================

@app.post("/admin/delete")
def admin_delete_device(
    item: dict
):

    global df

    print("\n" + "=" * 70)
    print("ADMIN — DELETE DEVICE")
    print("=" * 70)


    # --------------------------------------------------------
    # GET RECORD ID
    # --------------------------------------------------------

    record_id = item.get("id")


    if record_id is None:

        return {
            "status": "error",
            "message": "Record ID is required."
        }


    try:

        record_id = int(record_id)

    except (ValueError, TypeError):

        return {
            "status": "error",
            "message": "Invalid record ID."
        }


    # --------------------------------------------------------
    # CHECK RECORD EXISTS
    # --------------------------------------------------------

    if record_id not in df.index:

        return {
            "status": "error",
            "message": "Record not found."
        }


    # --------------------------------------------------------
    # GET RECORD BEFORE DELETING
    # --------------------------------------------------------

    deleted_row = df.loc[record_id].copy()


    print(
        f"Deleting record index: {record_id}"
    )

    print(
        f"Model: "
        f"{deleted_row['Standardized Model']}"
    )

    print(
        f"Trade-In Value: "
        f"{deleted_row['Max. Trade-In Value (RM)']}"
    )


    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

    df = df.drop(
        index=record_id
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # SAVE MASTER CLEAN
    # --------------------------------------------------------

    df.to_csv(
        DATA_FILE,
        index=False
    )


    # --------------------------------------------------------
    # SYNC TO GOOGLE SHEETS (best-effort, non-blocking)
    # --------------------------------------------------------

    sync_result = sheets_sync.sync_dataset(df)

    if not sync_result.success:

        print(
            "WARNING: Google Sheets sync failed after DELETE: "
            f"{sync_result.error}"
        )


    print(
        "Record deleted successfully."
    )

    print(
        f"Updated dataset rows: {len(df)}"
    )

    print("=" * 70)


    return {
        "status": "success",
        "message": "Record deleted successfully.",
        "sheets_sync": sync_result.as_dict()
    }

# ============================================================
# ADMIN — FORECAST TRADE-IN VALUE
# ============================================================

@app.post("/admin/forecast")
def admin_forecast(item: dict):

    try:

        # ----------------------------------------------------
        # CURRENT YEAR
        # ----------------------------------------------------

        current_year = datetime.now().year


        # ----------------------------------------------------
        # GET INPUT
        # ----------------------------------------------------

        record_id = item.get("record_id")
        forecast_until = item.get("forecast_until")


        if record_id is None:

            return {
                "status": "error",
                "message": "Record ID is required."
            }


        if forecast_until is None:

            return {
                "status": "error",
                "message": "Forecast end year is required."
            }


        record_id = int(record_id)
        forecast_until = int(forecast_until)


        # ----------------------------------------------------
        # VALIDATE YEAR
        # ----------------------------------------------------

        if forecast_until < current_year:

            return {
                "status": "error",
                "message":
                    f"Forecast year must be "
                    f"{current_year} or later."
            }


        if record_id not in df.index:

            return {
                "status": "error",
                "message": "Record not found."
            }


        # ----------------------------------------------------
        # GET BASE RECORD
        # ----------------------------------------------------

        base = df.loc[record_id].copy()


        device = str(
            base["Device"]
        )

        sub_device = str(
            base["Sub-device"]
        )

        model_name = str(
            base["Standardized Model"]
        )

        provider = str(
            base["Provider"]
        )

        model_year_value = base["Model_Year"]


        # ----------------------------------------------------
        # VALIDATE MODEL YEAR
        # ----------------------------------------------------

        if pd.isna(model_year_value):

            return {
                "status": "unresolved",
                "message":
                    "Model year is unavailable for "
                    "this device."
            }


        model_year = int(
            float(model_year_value)
        )


        # ----------------------------------------------------
        # MSRP
        #
        # MSRP is stored directly in the master dataset.
        # Each record carries its own MSRP.
        # ----------------------------------------------------

        msrp_value = base["MSRP"]

        if pd.isna(msrp_value):

            return {
                "status":
                    "unresolved",

                "message":
                    "MSRP is unavailable for this device."
            }


        msrp = float(msrp_value)

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("ADMIN — DEPRECIATION FORECAST")
        print("=" * 70)

        print(
            f"Device       : {device}"
        )

        print(
            f"Sub-device   : {sub_device}"
        )

        print(
            f"Model        : {model_name}"
        )

        print(
            f"Provider     : {provider}"
        )

        print(
            f"Model Year   : {model_year}"
        )

        print(
            f"MSRP         : RM {msrp:,.2f}"
        )

        print(
            f"Forecast     : {current_year} → "
            f"{forecast_until}"
        )


        # ----------------------------------------------------
        # GENERATE FORECAST
        #
        # The forecast must never evaluate a year before the
        # device's own Model Year, since device_age would be
        # negative and there is no meaningful trade-in value
        # before a device has been released. This matters for
        # devices with a Model Year later than the current year
        # (e.g. a not-yet-released model added in advance) --
        # in that case the forecast starts at the device's
        # release year instead of the server's current year.
        # ----------------------------------------------------

        forecast_start = max(current_year, model_year)

        if forecast_start > forecast_until:

            return {
                "status": "error",
                "message":
                    "Forecast end year must be "
                    f"{forecast_start} or later, since "
                    f"this device's Model Year is {model_year}."
            }

        results = []

        previous_value = None


        for year in range(
            forecast_start,
            forecast_until + 1
        ):

            # ------------------------------------------------
            # DEVICE AGE
            # ------------------------------------------------

            device_age = (
                year
                -
                model_year
            )


            # ------------------------------------------------
            # AGE 0
            #
            # Device is in its model year.
            # Use MSRP as the baseline value.
            # No depreciation curve is required.
            # ------------------------------------------------

            if device_age == 0:

                prediction = msrp

                print(
                    f"Forecast {year}: "
                    f"RM {prediction:,.2f} | "
                    f"Tier: launch_price | "
                    f"Curve: MSRP baseline"
                )


            # ------------------------------------------------
            # AGE 1+
            #
            # Use depreciation fallback.
            # ------------------------------------------------

            else:

                print("\n" + "=" * 70)
                print("DEBUG — FALLBACK OBJECT")
                print("=" * 70)

                print("Fallback class:")
                print(type(fallback))

                print("Fallback module:")
                print(type(fallback).__module__)

                print("Fallback file:")
                print(sys.modules[type(fallback).__module__].__file__)

                print("Reference year:")
                print(year)

                print("Device age:")
                print(device_age)

                print("Device:")
                print(device)

                print("Sub-device:")
                print(sub_device)

                print("Provider:")
                print(provider)

                print("MSRP:")
                print(msrp)

                fallback_result = fallback.predict(

                    device=device,

                    sub_device=sub_device,

                    provider=provider,

                    msrp=msrp,

                    model_year=model_year,

                    reference_year=year

                )

                print(
                    f"DEBUG FORECAST RESULT: "
                    f"year={year}, "
                    f"age={device_age}, "
                    f"tier={fallback_result.matched_tier}, "
                    f"form={fallback_result.form}, "
                    f"retention={fallback_result.predicted_retention}, "
                    f"value={fallback_result.predicted_value}, "
                    f"analogs={fallback_result.analog_models_used}"
                )


                # ------------------------------------------------
                # UNRESOLVED
                # ------------------------------------------------

                if (
                    fallback_result.predicted_value
                    is None
                ):

                    return {

                        "status":
                            "unresolved",

                        "message":
                            (
                                "No suitable depreciation curve "
                                "is available for this device."
                            ),

                        "confidence_flag":
                            fallback_result.confidence_flag

                    }


                print(
                    f"Forecast {year}: "
                    f"RM {fallback_result.predicted_value:,.2f} | "
                    f"Tier: {fallback_result.matched_tier} | "
                    f"Curve: {fallback_result.form}"
                )


                prediction = float(
                    fallback_result.predicted_value
                )


            # ------------------------------------------------
            # CHANGE
            # ------------------------------------------------

            if previous_value is None:

                change = None

                change_percent = None

            else:

                change = (
                    prediction
                    -
                    previous_value
                )


                if previous_value != 0:

                    change_percent = (
                        change
                        /
                        previous_value
                    ) * 100

                else:

                    change_percent = None


            # ------------------------------------------------
            # DEVICE AGE
            # ------------------------------------------------

            device_age = (
                year
                -
                model_year
            )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            results.append({

                "year":
                    year,

                "device_age":
                    device_age,

                "estimated_trade_in":
                    round(
                        prediction,
                        2
                    ),

                "change":
                    (
                        None
                        if change is None
                        else round(
                            change,
                            2
                        )
                    ),

                "change_percent":
                    (
                        None
                        if change_percent is None
                        else round(
                            change_percent,
                            2
                        )
                    )

            })


            previous_value = prediction


        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {

            "status":
                "success",

            "model":
                model_name,

            "device":
                device,

            "sub_device":
                sub_device,

            "provider":
                provider,

            "model_year":
                model_year,

            "msrp":
                round(
                    msrp,
                    2
                ),

            "current_year":
                current_year,

            "forecast_until":
                forecast_until,

            "method":
                "depreciation_curve_with_msrp_baseline",

            "results":
                results

        }


    except Exception as e:

        print(
            "\nFORECAST ERROR:"
        )

        print(
            repr(e)
        )


        return {

            "status":
                "error",

            "message":
                str(e)

        }

# ============================================================
# CUSTOMER FRONTEND
# ============================================================

@app.get("/")
def customer_frontend():

    return FileResponse(
        ASSETS_DIR / "index.html"
    )


# ============================================================
# ADMIN FRONTEND
# ============================================================

@app.get("/admin")
def admin_frontend():

    return FileResponse(
        ASSETS_DIR / "admin.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "online",
    }