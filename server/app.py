# ============================================================
# APPLE TRADE-IN VALUATION API
# CUSTOMER-FACING API
# ============================================================

import pandas as pd
import numpy as np
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional, cast, List
from pydantic import BaseModel
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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

FITTED_CURVES_FILE = BASE_DIR / "data" / "fitted_curves.csv"

# ------------------------------------------------------------
# Google Sheets sync configuration.
#
# The service account key file is never committed to source
# control -- it must be placed manually at this path on the
# server. If it's missing, Sheets sync is disabled and the app
# still runs normally against the local CSV (see SheetsSync).
# ------------------------------------------------------------

SHEETS_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", str(BASE_DIR / "internal" / "service_account.json"))

SHEETS_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

SHEETS_SPREADSHEET_ID = "1TzySGhtEs-ptmzLHNxcJ5q_lQ9nGr6HDofy0IL7G1vs"

SHEETS_WORKSHEET_NAME = "Cleaned Master"

CUSTOMER_SHEETS_WORKSHEET_NAME = "Customer Data"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Apple Trade-In Valuation API")

app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ============================================================
# DATA CLEANING + MAP BUILDING (REUSABLE)
#
# Extracted into functions so the exact same cleaning/build
# logic can run both at startup (from the local CSV) and later
# on a refresh (from a Google Sheets read) -- one code path,
# two possible sources.
# ============================================================


def storage_to_gb(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if "tb" in value:

        numbers = [x for x in value.replace(",", "").split() if x.replace(".", "", 1).isdigit()]

        if numbers:
            return float(numbers[0]) * 1024

    if "gb" in value:

        numbers = [x for x in value.replace(",", "").split() if x.replace(".", "", 1).isdigit()]

        if numbers:
            return float(numbers[0])

    try:
        return float(value)

    except (ValueError, TypeError):
        return np.nan


TEXT_COLUMNS = ["Device", "Sub-device", "Standardized Model", "Provider", "Storage Type", "Connectivity", "Chipset"]


def clean_dataset(raw_df):
    """
    Normalize the master dataset consistently regardless of source.

    Applies:
    - Storage (GB) parsing
    - Numeric coercion for Max. Trade-In Value (RM)
    - Text normalization for known text columns

    Records with missing trade-in values are intentionally retained
    because Admin may need to edit them later.
    """

    cleaned = raw_df.copy()

    # ------------------------------------------------------------
    # STORAGE
    # ------------------------------------------------------------

    if "Storage (GB)" in cleaned.columns:
        cleaned["Storage (GB)"] = cleaned["Storage (GB)"].apply(storage_to_gb)

    # ------------------------------------------------------------
    # TRADE-IN VALUE
    # ------------------------------------------------------------

    if "Max. Trade-In Value (RM)" in cleaned.columns:
        cleaned["Max. Trade-In Value (RM)"] = pd.to_numeric(cleaned["Max. Trade-In Value (RM)"], errors="coerce")

    # ------------------------------------------------------------
    # TEXT NORMALIZATION
    # ------------------------------------------------------------

    for col in TEXT_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("Unknown").astype(str).str.strip()

    return cleaned


def build_device_maps(cleaned_df):
    """
    Build device_model_map and device_config_map from a cleaned
    dataframe.

    Returns:
        model_map:
            Device -> Sub-device -> Model -> Storage options

        config_map:
            Device -> Sub-device -> Model ->
            Storage Type / Connectivity options
    """

    model_map = {}
    config_map = {}

    for (device, sub_device, model_name), group in cleaned_df.groupby(["Device", "Sub-device", "Standardized Model"]):

        model_map.setdefault(device, {})
        model_map[device].setdefault(sub_device, {})

        # --------------------------------------------------------
        # STORAGE OPTIONS
        # --------------------------------------------------------

        storages = group["Storage (GB)"].dropna().unique().tolist()

        storages = sorted(storages)

        clean_storages = [int(x) if float(x).is_integer() else float(x) for x in storages]

        model_map[device][sub_device][model_name] = clean_storages

        # --------------------------------------------------------
        # CONFIGURATION OPTIONS
        # --------------------------------------------------------

        config_map.setdefault(device, {})
        config_map[device].setdefault(sub_device, {})

        storage_types = []

        if "Storage Type" in group.columns:
            storage_types = sorted(t for t in group["Storage Type"].dropna().unique().tolist() if t and t != "Unknown")

        connectivity_options = []

        if "Connectivity" in group.columns:
            connectivity_options = sorted(c for c in group["Connectivity"].dropna().unique().tolist() if c and c != "Unknown")

        config_map[device][sub_device][model_name] = {"storageTypes": storage_types, "connectivity": connectivity_options}

    return model_map, config_map


def build_model_number_map(model_number_df):
    """
    Build a lookup map for device model numbers.

    Returns:
        Device -> Sub-device -> Model Number -> Model names

    Model numbers without a valid value are ignored.
    """

    model_number_map = {}

    for (device, sub_device, model_number, model_name), group in model_number_df.groupby(["Device", "Sub-device", "Model Number", "Standardized Model"]):

        if pd.isna(model_number):
            continue

        model_number = str(model_number).strip()

        if not model_number:
            continue

        device = str(device).strip()
        sub_device = str(sub_device).strip()
        model_name = str(model_name).strip()

        model_number_map.setdefault(device, {})
        model_number_map[device].setdefault(sub_device, {})
        model_number_map[device][sub_device].setdefault(model_number, [])

        if model_name not in model_number_map[device][sub_device][model_number]:
            model_number_map[device][sub_device][model_number].append(model_name)

    return model_number_map


# ============================================================
# LOAD DATA (INITIAL, FROM LOCAL CSV)
# ============================================================

print("=" * 70)
print("LOADING APPLE TRADE-IN DATA")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print(f"Rows loaded: {len(df)}")
print(f"Master dataset: {DATA_FILE.name}")

df = clean_dataset(df)

device_model_map, device_config_map = build_device_maps(df)
model_number_map = build_model_number_map(df)

# ============================================================
# LOAD DEPRECIATION FALLBACK
# ============================================================

fallback = TradeInFallback(str(FITTED_CURVES_FILE), raw_data_path=str(DATA_FILE))

print(f"Depreciation curves loaded: " f"{FITTED_CURVES_FILE.name}")

# ============================================================
# GOOGLE SHEETS SYNC
# ============================================================

sheets_sync = SheetsSync(service_account_file=str(SHEETS_SERVICE_ACCOUNT_FILE), service_account_json=SHEETS_SERVICE_ACCOUNT_JSON, spreadsheet_id=SHEETS_SPREADSHEET_ID, worksheet_name=SHEETS_WORKSHEET_NAME)

if sheets_sync.is_available:

    print("Google Sheets sync ready -> " f"worksheet '{SHEETS_WORKSHEET_NAME}'")

else:

    print("Google Sheets sync UNAVAILABLE -- admin writes will " "still save to the local CSV, but will not be mirrored " "to Google Sheets until this is resolved.")

# ============================================================
# CUSTOMER GOOGLE SHEETS SYNC
# ============================================================

customer_sheets_sync = SheetsSync(service_account_file=str(SHEETS_SERVICE_ACCOUNT_FILE), service_account_json=SHEETS_SERVICE_ACCOUNT_JSON, spreadsheet_id=SHEETS_SPREADSHEET_ID, worksheet_name=CUSTOMER_SHEETS_WORKSHEET_NAME)

if customer_sheets_sync.is_available:

    print("Customer Google Sheets sync ready -> " f"worksheet '{CUSTOMER_SHEETS_WORKSHEET_NAME}'")

else:

    print("Customer Google Sheets sync UNAVAILABLE.")

print(f"Devices available: {len(device_model_map)}")

print("=" * 70)


# ============================================================
# LIVE REFRESH FROM GOOGLE SHEETS
#
# "Cleaned Master" (the sheet `sheets_sync` points at) is the
# intended source of truth -- rows deleted or edited directly in
# the Sheet must be reflected here, not just admin-tool writes.
#
# A refresh is attempted at most once per REFRESH_INTERVAL: cheap
# on every request when the cache is still fresh (a timestamp
# comparison), and a real Sheets read only every 10 minutes at
# most. If a refresh attempt fails for any reason (network, auth,
# quota, malformed data), the existing in-memory df/maps are kept
# untouched and used as-is -- this must never take the app down
# or serve empty data because of a transient Sheets issue.
# ============================================================

REFRESH_INTERVAL_SECONDS = 10 * 60

# Deliberately set in the past (further back than the interval
# itself) so the very first call to refresh_data_if_stale() after
# server startup always counts as stale and checks Sheets right
# away -- a restart should reflect the latest Sheets state
# immediately, not after waiting a full 10 minutes.
_last_refresh_at = datetime.now() - timedelta(seconds=REFRESH_INTERVAL_SECONDS + 1)


def write_csv_atomically(raw_df, destination_path):
    """
    Writes `raw_df` to `destination_path` safely: writes to a temp
    file in the same directory first, then atomically renames it
    over the real path. This avoids ever leaving master_msrp.csv
    half-written (e.g. if the process is killed mid-write) --
    since this file is read again on every server boot, a
    corrupted or truncated copy would take the whole app down.

    Returns True on success, False on any failure (never raises --
    a failed write-back must not affect the in-memory refresh that
    already succeeded by the time this is called).
    """

    temp_path = destination_path.with_suffix(destination_path.suffix + ".tmp")

    try:

        raw_df.to_csv(temp_path, index=False)

        os.replace(temp_path, destination_path)

        return True

    except Exception as exc:

        print("WARNING: Failed to write refreshed data back to " f"{destination_path.name}, local fallback file is " f"now stale until the next successful refresh: {exc}")

        # Best-effort cleanup of the temp file if it was created
        # but the rename itself failed.
        try:

            if temp_path.exists():
                temp_path.unlink()

        except Exception:
            pass

        return False


def refresh_data_if_stale():

    global df, device_model_map, device_config_map, model_number_map, _last_refresh_at

    seconds_since_refresh = (datetime.now() - _last_refresh_at).total_seconds()

    if seconds_since_refresh < REFRESH_INTERVAL_SECONDS:
        return

    # Mark the attempt time regardless of outcome, so a failed
    # fetch doesn't retry on every single request until the next
    # interval -- it still waits the full interval before trying
    # again, matching the fail-safe design of SheetsSync itself.
    _last_refresh_at = datetime.now()

    if not sheets_sync.is_available:
        return

    fetch_result = sheets_sync.fetch_dataset()

    if not fetch_result.success:

        print("WARNING: Sheets refresh failed, keeping existing " f"in-memory data: {fetch_result.error}")

        return

    try:

        refreshed_df = clean_dataset(fetch_result.dataframe)

        refreshed_model_map, refreshed_config_map = build_device_maps(refreshed_df)

        refreshed_model_number_map = build_model_number_map(refreshed_df)

    except Exception as exc:

        # A malformed Sheet (missing column, bad header, etc.)
        # must not corrupt the currently-working in-memory data.
        print("WARNING: Sheets refresh fetched data but it failed " f"to clean/build correctly, keeping existing " f"in-memory data: {exc}")

        return

    df = refreshed_df
    device_model_map = refreshed_model_map
    device_config_map = refreshed_config_map
    model_number_map = refreshed_model_number_map

    bump_data_version()

    # ----------------------------------------------------------
    # WRITE BACK TO master_msrp.csv
    #
    # Keeps the local CSV (the boot-time fallback) in sync with
    # whatever Sheets currently has, so a server restart boots
    # from the last-known-good Sheets state rather than a
    # potentially old/stale CSV snapshot. Writes the RAW fetched
    # frame (before clean_dataset()'s transforms), so the CSV
    # stays a faithful mirror of the Sheet's actual content --
    # cleaning still happens fresh on every load regardless.
    #
    # This is a separate, independent step from the in-memory
    # update above: if this write fails, df/the maps in memory
    # are already correctly updated and stay that way -- only
    # the on-disk fallback file remains stale until the next
    # successful refresh tries again.
    # ----------------------------------------------------------

    write_csv_atomically(fetch_result.dataframe, DATA_FILE)

    print("Sheets refresh applied -- " f"{fetch_result.rows_fetched} rows, " f"{len(device_model_map)} devices")


# ------------------------------------------------------------
# DATA VERSION STAMP
#
# Distinct from _last_refresh_at, which updates on every refresh
# ATTEMPT (including failed ones, to avoid hammering Sheets after
# an error). This stamp only changes when df is ACTUALLY replaced
# -- a successful Sheets refresh, or an admin add/modify/delete.
#
# Admin's modify/delete requests carry the stamp they saw when
# they loaded /admin/records; the backend rejects the write if it
# no longer matches, since that means the row positions they're
# relying on may no longer point at the same records.
# ------------------------------------------------------------

_data_version = datetime.now().isoformat()


def force_refresh_from_sheets():
    """
    Force an immediate Google Sheets check.

    Updates the backend dataframe and data version only when
    the Sheet contents actually differ from the current dataframe.
    """
    global df, device_model_map, device_config_map, model_number_map, _last_refresh_at

    if not sheets_sync.is_available:
        print("Admin force refresh failed: Google Sheets sync is not available.")
        return False

    try:
        fetch_result = sheets_sync.fetch_dataset()

        if not fetch_result.success or fetch_result.dataframe is None:
            print(f"Admin force refresh failed: {fetch_result.error}")
            return False

        refreshed_df = clean_dataset(fetch_result.dataframe)

        if refreshed_df.empty:
            print("Admin force refresh returned an empty dataset.")
            return False

        # Compare the actual dataset contents.
        current = df.reset_index(drop=True).fillna("").astype(str)
        refreshed = refreshed_df.reset_index(drop=True).fillna("").astype(str)

        data_changed = not current.equals(refreshed)

        # Build the refreshed maps locally first.
        # Global state is only updated after all refreshed data is ready.
        refreshed_model_map, refreshed_config_map = build_device_maps(refreshed_df)
        refreshed_model_number_map = build_model_number_map(refreshed_df)

        # Apply the refreshed state atomically.
        df = refreshed_df
        device_model_map = refreshed_model_map
        device_config_map = refreshed_config_map
        model_number_map = refreshed_model_number_map
        _last_refresh_at = datetime.now()

        if data_changed:
            bump_data_version()
            print(f"Admin force refresh detected a data change: " f"{len(df)} rows loaded.")
        else:
            print(f"Admin force refresh: no data changes detected " f"({len(df)} rows).")

        write_csv_atomically(fetch_result.dataframe, DATA_FILE)

        return True

    except Exception as exc:
        print(f"Admin force refresh error: {exc}")
        return False


def bump_data_version():

    global _data_version

    _data_version = datetime.now().isoformat()


# The startup refresh above may have already replaced df once --
# stamp that as the initial version so admin's very first page
# load already reflects it correctly.
bump_data_version()


# ------------------------------------------------------------
# STARTUP REFRESH
#
# Check Sheets once immediately at boot, rather than waiting for
# the first incoming request. This way, a server restart always
# reflects the latest Sheets state right away -- not "eventually,
# whenever the first customer happens to hit an endpoint".
#
# Uses the same fail-safe function as every later check: if this
# fails (Sheets down, credentials bad, etc.), the app still boots
# normally and simply keeps serving from the CSV-loaded data,
# exactly as before this feature existed.
# ------------------------------------------------------------

print("Checking Google Sheets for the latest data at startup...")

refresh_data_if_stale()


# ============================================================
# AVAILABLE MODELS
# ============================================================


@app.get("/available-models")
def get_models():

    refresh_data_if_stale()

    return device_model_map


# ============================================================
# MODEL CONFIGURATION
# (Storage Type + Connectivity, additive -- does not change
# the /available-models response shape above)
# ============================================================


@app.get("/model-configuration")
def get_model_configuration():

    refresh_data_if_stale()

    return device_config_map


# ============================================================
# MODEL NUMBERS
# (Exposes the existing model_number_map for lookup/filtering.
# Does not change how the map is built.)
# ============================================================


@app.get("/model-numbers")
def get_model_numbers():

    refresh_data_if_stale()

    return model_number_map


# ============================================================
# REQUEST MODEL
# ============================================================


class DeviceInput(BaseModel):
    Device: str
    SubDevice: str
    Model: str
    Storage: float | None = None
    StorageType: str | None = None
    Connectivity: str | None = None


# ============================================================
# EXACT DEVICE MEDIAN VALUATION
# ============================================================


@app.post("/predict")
def predict_price(item: DeviceInput):

    refresh_data_if_stale()

    print("\n" + "=" * 70)
    print("CUSTOMER VALUATION REQUEST")
    print("=" * 70)

    print(f"Device     : {item.Device}")
    print(f"Sub-Device : {item.SubDevice}")
    print(f"Model      : {item.Model}")
    print(f"Storage    : {item.Storage} GB")
    print(f"StorageType: {item.StorageType}")
    print(f"Connectivity: {item.Connectivity}")

    # ========================================================
    # AIRPODS
    # AirPods do not have storage
    # ========================================================

    if item.Device == "AirPods":

        matches = df[(df["Device"] == item.Device) & (df["Sub-device"] == item.SubDevice) & (df["Standardized Model"] == item.Model)].copy()

    # ========================================================
    # ALL OTHER DEVICES
    # Storage is required
    # ========================================================

    else:

        if item.Storage is None:

            print("Storage is required for this device.")

            return {"status": "unresolved", "estimated_value": None, "message": ("Storage is required for this device.")}

        matches = df[(df["Device"] == item.Device) & (df["Sub-device"] == item.SubDevice) & (df["Standardized Model"] == item.Model) & (np.isclose(df["Storage (GB)"], item.Storage, equal_nan=False))].copy()

        # ----------------------------------------------------
        # OPTIONAL CONFIGURATION FILTERS
        #
        # Only applied when the client actually sends a value --
        # devices without Storage Type / Connectivity in the
        # dataset (e.g. iPhone) never send these, so their
        # matching behaves exactly as before this change.
        # ----------------------------------------------------

        if item.StorageType:

            matches = matches[matches["Storage Type"] == item.StorageType]

        if item.Connectivity:

            matches = matches[matches["Connectivity"] == item.Connectivity]

    # ========================================================
    # NO MATCH
    # ========================================================

    if matches.empty:

        print("No exact database match.")

        return {"status": "unresolved", "estimated_value": None, "message": ("This exact device configuration " "is not available in the database.")}

    # ========================================================
    # MEDIAN TRADE-IN VALUE
    # ========================================================

    median_price = matches["Max. Trade-In Value (RM)"].median()

    if pd.isna(median_price):

        print("No trade-in values recorded for matching " "records -- treating as unresolved.")

        return {"status": "unresolved", "estimated_value": None, "message": ("This exact device configuration does not " "have a trade-in value on record yet.")}

    # ========================================================
    # SUPPORTING INFORMATION
    # ========================================================

    provider_count = matches["Provider"].nunique()

    record_count = len(matches)

    print(f"Matching records : {record_count}")

    print(f"Providers         : {provider_count}")

    print(f"Median value      : RM {median_price:,.2f}")

    # ========================================================
    # RETURN
    # ========================================================

    return {"status": "resolved", "estimated_value": round(float(median_price), 2), "method": "exact_configuration_median", "matching_records": record_count, "provider_count": provider_count}


# ============================================================
# CUSTOMER TRADE-IN RECORD
# ============================================================


class CustomerTradeInRecord(BaseModel):

    customer: dict

    device: dict

    valuation: dict

    createdAt: str


# ============================================================
# CUSTOMER — SAVE TRADE-IN RECORD
# ============================================================


@app.post("/customer/trade-in")
def save_customer_trade_in(item: CustomerTradeInRecord):

    print("\n" + "=" * 70)
    print("CUSTOMER — TRADE-IN RECORD")
    print("=" * 70)

    customer = item.customer
    device = item.device
    valuation = item.valuation

    # --------------------------------------------------------
    # CUSTOMER DETAILS
    # --------------------------------------------------------

    customer_name = str(customer.get("name", "")).strip()

    customer_phone = str(customer.get("phone", "")).strip()

    customer_email = str(customer.get("email", "")).strip()

    preferred_contact = str(customer.get("preferredContact", "")).strip()

    if not customer_name:
        return {"status": "error", "message": "Customer name is required."}

    if not customer_phone:
        return {"status": "error", "message": "Customer phone is required."}

    if not customer_email:
        return {"status": "error", "message": "Customer email is required."}

    if not preferred_contact:
        return {"status": "error", "message": "Preferred contact method is required."}

    # --------------------------------------------------------
    # DEVICE DETAILS
    # --------------------------------------------------------

    device_name = device.get("device")
    sub_device = device.get("subDevice")
    model_name = device.get("model")

    if not device_name:
        return {"status": "error", "message": "Device is required."}

    if not sub_device:
        return {"status": "error", "message": "Sub-device is required."}

    if not model_name:
        return {"status": "error", "message": "Model is required."}

    # --------------------------------------------------------
    # VALUATION DETAILS
    # --------------------------------------------------------

    market_value = valuation.get("marketValue")
    final_value = valuation.get("finalValue")

    # --------------------------------------------------------
    # GOOGLE SHEETS ROW
    # --------------------------------------------------------

    customer_record = {"Timestamp": datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).strftime("%d %b %Y, %I:%M %p"), "Customer Name": customer_name, "Phone": customer_phone, "Email": customer_email, "Preferred Contact": preferred_contact, "Device": device_name, "Sub-device": sub_device, "Model": model_name, "Model Number": device.get("modelNumber") or "N/A", "Storage (GB)": device.get("storage"), "Storage Type": device.get("storageType") or "N/A", "Connectivity": device.get("connectivity") or "N/A", "Market Value (RM)": market_value, "Final Trade-In Value (RM)": final_value}

    # --------------------------------------------------------
    # SAVE TO CUSTOMER SHEET
    # --------------------------------------------------------

    sync_result = customer_sheets_sync.append_record(customer_record)

    # --------------------------------------------------------
    # LOG RESULT
    # --------------------------------------------------------

    print(f"Customer : {customer_name}")
    print(f"Device   : {device_name}")
    print(f"Model    : {model_name}")

    if final_value is not None:
        print(f"Final    : RM {final_value:,.2f}")
    else:
        print("Final    : N/A")

    print("Sheets   : " + ("SAVED" if sync_result.success else "FAILED"))

    if sync_result.error:
        print(f"Sheets error: {sync_result.error}")

    print("=" * 70)

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    if not sync_result.success:

        return {"status": "error", "message": ("Your trade-in details could not be saved. " "Please try again."), "customer": customer_name, "model": model_name, "sheets": sync_result.as_dict()}

    return {"status": "success", "message": "Trade-in record received successfully.", "customer": customer_name, "model": model_name, "sheets": sync_result.as_dict()}


# ============================================================
# ADMIN ADD REQUEST
# ============================================================


class AdminAddDevice(BaseModel):

    Device: str
    SubDevice: str
    Model: str

    ModelNumber: Optional[str] = None

    Provider: Optional[str] = None

    MSRP: float

    TradeInValue: Optional[float] = None

    Storage: Optional[float] = None
    StorageType: Optional[str] = None

    ModelYear: Optional[int] = None

    Chipset: Optional[str] = None
    Connectivity: Optional[str] = None

    Material: Optional[str] = None
    CaseSize: Optional[int] = None

    ChargingMethod: Optional[str] = None


# ============================================================
# ADMIN — ADD DEVICE
# ============================================================


@app.post("/admin/add")
def admin_add_device(item: AdminAddDevice):

    global df

    print("\n" + "=" * 70)
    print("ADMIN — ADD DEVICE")
    print("=" * 70)

    if not force_refresh_from_sheets():
        raise HTTPException(status_code=503, detail="Unable to refresh data from Google Sheets. Operation cancelled.")

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    device = item.Device.strip()
    sub_device = item.SubDevice.strip()
    model_name = item.Model.strip()

    model_number = item.ModelNumber.strip() if item.ModelNumber else np.nan

    if not device:
        raise HTTPException(status_code=400, detail="Device is required.")

    if not sub_device:
        raise HTTPException(status_code=400, detail="Sub-device is required.")

    if not model_name:
        raise HTTPException(status_code=400, detail="Model is required.")

    charging_method = item.ChargingMethod.strip() if item.ChargingMethod else np.nan

    if device == "AirPods" and pd.isna(charging_method):

        raise HTTPException(status_code=400, detail="Charging method is required.")

    # ========================================================
    # Retail Price (MSRP)
    # ========================================================

    if item.MSRP < 0:

        raise HTTPException(status_code=400, detail="Retail price cannot be negative.")

    msrp = float(item.MSRP)

    # ========================================================
    # PROVIDER
    # ========================================================

    provider = item.Provider.strip() if item.Provider else "Unknown"

    # ========================================================
    # TRADE-IN VALUE
    # ========================================================

    if item.TradeInValue is None:

        trade_in_value = np.nan
        price_status = "N/A"

    else:

        if item.TradeInValue < 0:

            raise HTTPException(status_code=400, detail="Trade-in value cannot be negative.")

        trade_in_value = float(item.TradeInValue)
        price_status = "confirmed"

    # ========================================================
    # STORAGE
    # ========================================================

    storage = item.Storage

    storage_type = item.StorageType.strip() if item.StorageType else np.nan

    if device != "AirPods" and storage is None:

        raise HTTPException(status_code=400, detail="Storage is required.")

    if device == "Mac" and pd.isna(storage_type):

        raise HTTPException(status_code=400, detail="Storage type is required.")

    # AirPods do not have storage

    if device == "AirPods":

        storage = np.nan
        storage_type = np.nan

    # ========================================================
    # DEVICE-SPECIFIC VALUES
    # ========================================================

    chipset = item.Chipset.strip() if item.Chipset else np.nan

    if device in ("iPhone", "iPad", "Mac") and pd.isna(chipset):

        raise HTTPException(status_code=400, detail="Chipset is required.")

    connectivity = item.Connectivity.strip() if item.Connectivity else np.nan

    if device in ("iPad", "Apple Watch") and pd.isna(connectivity):

        raise HTTPException(status_code=400, detail="Connectivity is required.")

    material = item.Material.strip() if item.Material else np.nan

    if device == "Apple Watch" and pd.isna(material):

        raise HTTPException(status_code=400, detail="Material is required.")

    case_size = int(item.CaseSize) if item.CaseSize is not None else np.nan

    if device == "Apple Watch" and item.CaseSize is None:

        raise HTTPException(status_code=400, detail="Case size is required.")

    charging_method = item.ChargingMethod.strip() if item.ChargingMethod else np.nan

    # ========================================================
    # MODEL YEAR
    # ========================================================

    model_year = item.ModelYear

    if model_year is None:

        raise HTTPException(status_code=400, detail="Model year is required.")

    if model_year < 1976:

        raise HTTPException(status_code=400, detail="Invalid Apple model year.")

    # ========================================================
    # CREATE NEW ROW
    # ========================================================

    new_row = {"Provider": provider, "Device": device, "Sub-device": sub_device, "Standardized Model": model_name, "Model Number": model_number, "Retail Price": msrp, "Storage (GB)": storage, "Storage Type": storage_type, "Connectivity": connectivity, "Material": (item.Material.strip() if item.Material else np.nan), "Max. Trade-In Value (RM)": trade_in_value, "Model_Year": model_year, "Chipset": chipset, "Case Size": case_size, "Charging Method": charging_method}

    # ========================================================
    # APPEND TO DATAFRAME
    # ========================================================

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save updated master dataset
    df.to_csv(DATA_FILE, index=False)

    # ========================================================
    # SYNC TO GOOGLE SHEETS (best-effort, non-blocking)
    # ========================================================

    bump_data_version()

    sync_result = sheets_sync.sync_dataset(df)

    if not sync_result.success:

        print("WARNING: Google Sheets sync failed after ADD: " f"{sync_result.error}")

    # ========================================================
    # LOG
    # ========================================================

    print(f"Device       : {device}")

    print(f"Sub-device   : {sub_device}")

    print(f"Model        : {model_name}")

    print(f"Provider     : {provider}")

    print(f"Retail Price : RM {msrp:,.2f}")

    print(f"Trade-in     : " f"{'N/A' if pd.isna(trade_in_value) else f'RM {trade_in_value:,.2f}'}")

    print("Device added successfully.")

    print("=" * 70)

    # ========================================================
    # RESPONSE
    # ========================================================

    return {"status": "success", "message": "Device added successfully.", "price_status": price_status, "model": model_name, "msrp": msrp, "trade_in_value": (None if pd.isna(trade_in_value) else trade_in_value), "sheets_sync": sync_result.as_dict()}


# ============================================================
# ADMIN — DATA STATUS
# ============================================================


@app.get("/admin/status")
def admin_status():
    refresh_data_if_stale()

    print(df["Device"].value_counts(dropna=False))
    # --------------------------------------------------------
    # DEVICE-SPECIFIC INTENTIONAL MISSING FIELDS
    # --------------------------------------------------------

    excluded_fields = {"iPhone": {"Connectivity", "Material", "Case Size", "Charging Method", "Storage Type"}, "iPad": {"Storage Type", "Material", "Case Size", "Charging Method"}, "Mac": {"Connectivity", "Material", "Case Size", "Charging Method"}, "Apple Watch": {"Storage Type", "Charging Method"}, "AirPods": {"Storage (GB)", "Storage Type", "Connectivity", "Material", "Case Size"}}

    # --------------------------------------------------------
    # COLUMNS ACTUALLY USED BY THE APPLICATION
    # --------------------------------------------------------

    fields = ["Provider", "Device", "Sub-device", "Standardized Model", "Retail Price", "Storage (GB)", "Storage Type", "Connectivity", "Material", "Max. Trade-In Value (RM)", "Model_Year", "Chipset", "Case Size", "Charging Method"]

    # --------------------------------------------------------
    # MISSING VALUE CHECK
    # --------------------------------------------------------

    def is_missing(value):

        if pd.isna(value):
            return True

        if isinstance(value, str):

            value = value.strip().lower()

            return value in {"", "n/a", "na", "Unknown", "none", "nan"}

        return False

    categories = []

    # --------------------------------------------------------
    # CHECK EACH COLUMN
    # --------------------------------------------------------

    for field in fields:

        if field not in df.columns:
            continue

        affected_records = []

        for index, row in df.iterrows():

            device = None if pd.isna(row["Device"]) else str(row["Device"]).strip()

            # ------------------------------------------------
            # SKIP INTENTIONALLY MISSING FIELDS
            # ------------------------------------------------

            if device in excluded_fields and field in excluded_fields[device]:

                continue

            # ------------------------------------------------
            # CHECK VALUE
            # ------------------------------------------------

            if not is_missing(row[field]):
                continue

            if field == "Charging Method":
                print("CHARGING METHOD MISSING:", device, "|", row["Charging Method"])

            affected_records.append(
                {
                    "id": int(cast(int, index)),
                    "device": device,
                    "sub_device": (None if pd.isna(row["Sub-device"]) else str(row["Sub-device"]).strip()),
                    "model": (None if pd.isna(row["Standardized Model"]) else str(row["Standardized Model"]).strip()),
                    "provider": (None if pd.isna(row["Provider"]) else str(row["Provider"]).strip()),
                    "storage": (None if pd.isna(row["Storage (GB)"]) else float(row["Storage (GB)"])),
                    "storage_type": (None if pd.isna(row["Storage Type"]) else str(row["Storage Type"]).strip()),
                    "connectivity": (None if pd.isna(row["Connectivity"]) else str(row["Connectivity"]).strip()),
                    "material": (None if pd.isna(row["Material"]) else str(row["Material"]).strip()),
                    "chipset": (None if pd.isna(row["Chipset"]) else str(row["Chipset"]).strip()),
                    "case_size": (None if pd.isna(row["Case Size"]) else str(row["Case Size"]).strip()),
                    "charging_method": (None if pd.isna(row["Charging Method"]) else str(row["Charging Method"]).strip()),
                    "trade_in_value": (None if pd.isna(row["Max. Trade-In Value (RM)"]) else float(row["Max. Trade-In Value (RM)"])),
                }
            )

        # ----------------------------------------------------
        # ONLY CREATE CATEGORY IF SOMETHING IS MISSING
        # ----------------------------------------------------

        if affected_records:

            categories.append({"field": field, "count": len(affected_records), "records": affected_records})

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_missing = sum(category["count"] for category in categories)

    return {"total_records": int(len(df)), "categories": categories, "total_missing": int(total_missing)}


# ============================================================
# ADMIN — DATA VERSION (lightweight staleness check)
#
# The admin UI polls this on an interval to detect when the
# underlying data has changed (e.g. someone edited the Sheet
# directly, or another admin session made a change) so it can
# refresh itself automatically instead of relying on a manual
# "Refresh Data" button.
#
# Deliberately does none of the heavier work /admin/status does
# (iterating every row to find missing fields) -- it just runs the
# same throttled staleness check every other admin endpoint runs,
# then reports the current stamp. Cheap enough to poll frequently.
# ============================================================


@app.get("/admin/data-version")
def admin_data_version():
    refresh_data_if_stale()

    return {"dataVersion": _data_version}


# ============================================================
# ADMIN — AVAILABLE MODELS
# ============================================================


@app.get("/admin/models")
def admin_models():
    refresh_data_if_stale()

    models = {}

    for device, group in df.groupby("Device"):

        values = group["Standardized Model"].dropna().astype(str).unique().tolist()

        models[device] = sorted(values)

    return models


# ============================================================
# ADMIN — FORECAST MODEL HIERARCHY
# ============================================================


@app.get("/admin/forecast-models")
def admin_forecast_models():
    """
    Returns the Device -> Sub-device -> [Models] hierarchy used by
    the Forecast page's cascading pickers:

    {
        "iPhone": {
            "Standard": ["iPhone 13", "iPhone 14", ...],
            "Pro": ["iPhone 13 Pro", "iPhone 14 Pro", ...]
        },
        ...
    }
    """
    refresh_data_if_stale()

    hierarchy = {}

    for (device, sub_device), group in df.groupby(["Device", "Sub-device"]):
        if pd.isna(device) or pd.isna(sub_device):
            continue

        device_str = str(device)
        sub_device_str = str(sub_device)

        if device_str not in hierarchy:
            hierarchy[device_str] = {}

        models = group["Standardized Model"].dropna().astype(str).unique().tolist()

        hierarchy[device_str][sub_device_str] = sorted(models)

    return hierarchy


# ============================================================
# ADMIN — FIND RECORDS
# ============================================================


@app.get("/admin/records")
def admin_records(device: str, model: str, sub_device: Optional[str] = None):

    refresh_data_if_stale()

    matches = df[(df["Device"] == device) & (df["Standardized Model"] == model)]

    if sub_device:
        matches = matches[matches["Sub-device"] == sub_device]

    # ----------------------------------------------------
    # HELPERS
    # ----------------------------------------------------

    def clean_value(column, row):

        if column not in row.index:
            return None

        value = row[column]

        if pd.isna(value):
            return None

        return str(value).strip()

    def clean_json_value(value):

        if isinstance(value, float) and np.isnan(value):
            return None

        if isinstance(value, dict):
            return {key: clean_json_value(val) for key, val in value.items()}

        if isinstance(value, list):
            return [clean_json_value(item) for item in value]

        return value

    records = []

    for index, row in matches.iterrows():

        record_id = int(cast(int, index))

        # ------------------------------------------------
        # TRADE-IN VALUE
        # ------------------------------------------------

        value = row["Max. Trade-In Value (RM)"]

        if pd.isna(value):
            value = None
        else:
            value = float(value)

        # ------------------------------------------------
        # STORAGE
        # ------------------------------------------------

        storage = row["Storage (GB)"]

        if pd.isna(storage):
            storage = None
        else:
            storage = float(storage)

        # ------------------------------------------------
        # BUILD RECORD
        # ------------------------------------------------

        records.append({"id": record_id, "model": clean_value("Standardized Model", row), "sub_device": clean_value("Sub-device", row), "storage": storage, "storage_type": clean_value("Storage Type", row), "connectivity": clean_value("Connectivity", row), "material": clean_value("Material", row), "chipset": clean_value("Chipset", row), "provider": clean_value("Provider", row), "msrp": (None if pd.isna(row["Retail Price"]) else float(row["Retail Price"])), "trade_in_value": value})

    records = clean_json_value(records)

    return JSONResponse(content=records, headers={"X-Data-Version": _data_version})


# ============================================================
# ADMIN MODIFY REQUEST
# ============================================================


class AdminModifyDevice(BaseModel):

    id: int

    dataVersion: Optional[str] = None

    Provider: Optional[str] = None

    MSRP: Optional[float] = None

    TradeInValue: Optional[float] = None


@app.post("/admin/modify")
def admin_modify_device(item: AdminModifyDevice):

    global df

    print("\n" + "=" * 70)
    print("ADMIN — MODIFY DEVICE")
    print("=" * 70)

    if not force_refresh_from_sheets():
        raise HTTPException(status_code=503, detail="Unable to refresh data from Google Sheets. Operation cancelled.")

    # ========================================================
    # CHECK RECORD
    # ========================================================

    if item.id < 0 or item.id >= len(df):

        return {"status": "error", "message": "Record not found."}

    # ========================================================
    # CHECK DATA VERSION
    #
    # item.id is a POSITIONAL row index into df. If the
    # underlying data was reloaded (Sheets refresh, or another
    # admin's write) since this admin last loaded /admin/records,
    # that same index may now point at a completely different
    # record. Reject rather than silently write to the wrong row.
    # ========================================================

    if item.dataVersion is not None and item.dataVersion != _data_version:

        raise HTTPException(status_code=409, detail=("The underlying data has changed since you " "loaded this record. Reloading..."))

    # ========================================================
    # VALIDATE RETAIL PRICE (MSRP)
    # ========================================================

    if item.MSRP is not None and item.MSRP < 0:

        return {"status": "error", "message": "Retail price cannot be negative."}

    # ========================================================
    # VALIDATE TRADE-IN VALUE
    # ========================================================

    if item.TradeInValue is not None and item.TradeInValue < 0:

        return {"status": "error", "message": "Trade-in value cannot be negative."}

    # ========================================================
    # PROVIDER
    # ========================================================

    if item.Provider is not None:

        provider = item.Provider.strip()

        if not provider:

            return {"status": "error", "message": "Provider cannot be empty."}

        old_provider = df.at[item.id, "Provider"]

        df.at[item.id, "Provider"] = provider

        print(f"Provider updated: " f"{old_provider} → {provider}")

    # ========================================================
    # RETAIL PRICE (MSRP)
    # ========================================================

    if item.MSRP is not None:

        old_msrp = df.at[item.id, "Retail Price"]

        df.at[item.id, "Retail Price"] = float(item.MSRP)

        print(f"Retail Price updated: " f"{'N/A' if pd.isna(old_msrp) else f'RM {old_msrp:,.2f}'}" f" → RM {item.MSRP:,.2f}")

    # ========================================================
    # TRADE-IN VALUE
    # ========================================================

    if item.TradeInValue is not None:

        old_value = df.at[item.id, "Max. Trade-In Value (RM)"]

        df.at[item.id, "Max. Trade-In Value (RM)"] = float(item.TradeInValue)

        print(f"Trade-in value updated: " f"RM {item.TradeInValue:,.2f}")

    else:

        old_value = df.at[item.id, "Max. Trade-In Value (RM)"]

        df.at[item.id, "Max. Trade-In Value (RM)"] = np.nan

        print("Trade-in value changed to N/A.")

    # ========================================================
    # SAVE MASTER DATASET
    # ========================================================

    df.to_csv(DATA_FILE, index=False)

    # ========================================================
    # SYNC TO GOOGLE SHEETS (best-effort, non-blocking)
    # ========================================================

    bump_data_version()

    sync_result = sheets_sync.sync_dataset(df)

    if not sync_result.success:

        print("WARNING: Google Sheets sync failed after MODIFY: " f"{sync_result.error}")

    # ========================================================
    # LOG
    # ========================================================

    print(f"Record index : {item.id}")

    print("Record updated successfully.")

    print("=" * 70)

    # ========================================================
    # RESPONSE
    # ========================================================

    return {"status": "success", "message": "Record updated successfully.", "id": item.id, "provider": (str(df.at[item.id, "Provider"]) if not pd.isna(df.at[item.id, "Provider"]) else None), "msrp": (None if pd.isna(df.at[item.id, "Retail Price"]) else float(df.at[item.id, "Retail Price"])), "trade_in_value": (None if pd.isna(df.at[item.id, "Max. Trade-In Value (RM)"]) else float(df.at[item.id, "Max. Trade-In Value (RM)"])), "sheets_sync": sync_result.as_dict()}


# ============================================================
# ADMIN — DELETE DEVICE
# ============================================================


@app.post("/admin/delete")
def admin_delete_device(item: dict):

    global df

    print("\n" + "=" * 70)
    print("ADMIN — DELETE DEVICE")
    print("=" * 70)

    if not force_refresh_from_sheets():
        raise HTTPException(status_code=503, detail="Unable to refresh data from Google Sheets. Operation cancelled.")

    # --------------------------------------------------------
    # GET RECORD ID
    # --------------------------------------------------------

    record_id = item.get("id")

    if record_id is None:

        return {"status": "error", "message": "Record ID is required."}

    try:

        record_id = int(record_id)

    except (ValueError, TypeError):

        return {"status": "error", "message": "Invalid record ID."}

    # --------------------------------------------------------
    # CHECK RECORD EXISTS
    # --------------------------------------------------------

    if record_id not in df.index:

        return {"status": "error", "message": "Record not found."}

    # --------------------------------------------------------
    # CHECK DATA VERSION
    #
    # Same reasoning as /admin/modify: record_id is a POSITIONAL
    # row index, which can silently point at a different record
    # if df was reloaded since this admin last loaded the record
    # list. Reject rather than risk deleting the wrong row.
    # --------------------------------------------------------

    submitted_version = item.get("dataVersion")

    if submitted_version is not None and submitted_version != _data_version:

        raise HTTPException(status_code=409, detail=("The underlying data has changed since you " "loaded this record. Reloading..."))

    # --------------------------------------------------------
    # GET RECORD BEFORE DELETING
    # --------------------------------------------------------

    deleted_row = df.loc[record_id].copy()

    print(f"Deleting record index: {record_id}")

    print(f"Model: " f"{deleted_row['Standardized Model']}")

    print(f"Trade-In Value: " f"{deleted_row['Max. Trade-In Value (RM)']}")

    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

    df = df.drop(index=record_id).reset_index(drop=True)

    # --------------------------------------------------------
    # SAVE MASTER CLEAN
    # --------------------------------------------------------

    df.to_csv(DATA_FILE, index=False)

    # --------------------------------------------------------
    # SYNC TO GOOGLE SHEETS (best-effort, non-blocking)
    # --------------------------------------------------------

    bump_data_version()

    sync_result = sheets_sync.sync_dataset(df)

    if not sync_result.success:

        print("WARNING: Google Sheets sync failed after DELETE: " f"{sync_result.error}")

    print("Record deleted successfully.")

    print(f"Updated dataset rows: {len(df)}")

    print("=" * 70)

    return {"status": "success", "message": "Record deleted successfully.", "sheets_sync": sync_result.as_dict()}


# ============================================================
# ADMIN — CUSTOMER DATA
#
# Unlike the master dataset, Customer Data has no in-memory
# dataframe -- it's append-only from the customer-facing flow and
# is read/deleted directly against the "Customer Data" worksheet
# via customer_sheets_sync.
#
# Condition information is never collected in the customer trade-in
# flow above, and is stripped again here defensively so it can
# never surface through this endpoint even if a column by that name
# were ever added to the sheet directly.
# ============================================================

CONDITION_FIELD_MARKER = "condition"


@app.get("/admin/customers")
def admin_list_customers():

    result = customer_sheets_sync.list_records()

    if not result.success:
        raise HTTPException(status_code=503, detail=result.error or "Unable to load customer data.")

    customers = []

    for record in result.records:

        fields = {
            key: value
            for key, value in record["fields"].items()
            if CONDITION_FIELD_MARKER not in key.strip().lower()
        }

        customers.append({"row": record["row"], "fields": fields})

    return {"status": "success", "customers": customers}


class CustomerDeleteRequest(BaseModel):

    rows: List[dict]


@app.post("/admin/customers/delete")
def admin_delete_customers(item: CustomerDeleteRequest):

    print("\n" + "=" * 70)
    print("ADMIN — DELETE CUSTOMER RECORDS")
    print("=" * 70)

    if not item.rows:
        return {"status": "error", "message": "No records selected."}

    result = customer_sheets_sync.delete_rows(item.rows)

    if not result.success:
        raise HTTPException(status_code=503, detail=result.error or "Unable to delete customer records.")

    print(f"Deleted rows  : {result.deleted_rows}")
    print(f"Skipped rows  : {result.skipped_rows}")
    print("=" * 70)

    deleted_count = len(result.deleted_rows)
    skipped_count = len(result.skipped_rows)

    # --------------------------------------------------------
    # ALL SELECTED ROWS WERE STALE
    #
    # Nothing was deleted -- the same "data changed since you
    # loaded it" situation /admin/delete signals with a 409, so
    # it's signaled the same way here rather than as a 200 with
    # status: error, which this codebase reserves for bad input.
    # --------------------------------------------------------

    if skipped_count and not deleted_count:

        raise HTTPException(
            status_code=409,
            detail=(
                "The selected record(s) changed since the list was "
                "loaded. Reloading..."
            ),
        )

    # --------------------------------------------------------
    # PARTIAL SUCCESS
    #
    # Some rows were deleted, some were stale. /admin/delete has
    # no analog for this -- it only ever touches one record, so
    # it's always all-or-nothing. Here real mutation did happen,
    # so this can't collapse into a single exception; it stays a
    # 200 with its own "partial" status.
    # --------------------------------------------------------

    if skipped_count:

        return {
            "status": "partial",
            "message": (
                f"{deleted_count} record(s) deleted. {skipped_count} record(s) "
                "were skipped because their data changed since the list was "
                "loaded. Reloading..."
            ),
            "deleted_rows": result.deleted_rows,
            "skipped_rows": result.skipped_rows,
        }

    return {
        "status": "success",
        "message": f"{deleted_count} record(s) deleted.",
        "deleted_rows": result.deleted_rows,
    }


# ============================================================
# ADMIN — FORECAST TRADE-IN VALUE
# ============================================================


@app.post("/admin/forecast")
def admin_forecast(item: dict):
    refresh_data_if_stale()

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

            return {"status": "error", "message": "Record ID is required."}

        if forecast_until is None:

            return {"status": "error", "message": "Forecast end year is required."}

        record_id = int(record_id)
        forecast_until = int(forecast_until)

        # ----------------------------------------------------
        # VALIDATE YEAR
        # ----------------------------------------------------

        if forecast_until < current_year:

            return {"status": "error", "message": f"Forecast year must be " f"{current_year} or later."}

        if record_id not in df.index:

            return {"status": "error", "message": "Record not found."}

        # ----------------------------------------------------
        # GET BASE RECORD
        # ----------------------------------------------------

        base = df.loc[record_id].copy()

        device = str(base["Device"])

        sub_device = str(base["Sub-device"])

        model_name = str(base["Standardized Model"])

        provider = str(base["Provider"])

        model_year_value = base["Model_Year"]

        # ----------------------------------------------------
        # VALIDATE MODEL YEAR
        # ----------------------------------------------------

        if pd.isna(model_year_value):

            return {"status": "unresolved", "message": "Model year is unavailable for " "this device."}

        model_year = int(float(model_year_value))

        # ----------------------------------------------------
        # RETAIL PRICE (MSRP)
        #
        # Retail Price is stored directly in the master dataset.
        # Each record carries its own Retail Price.
        # ----------------------------------------------------

        msrp_value = base["Retail Price"]

        if pd.isna(msrp_value):

            return {"status": "unresolved", "message": "Retail price is unavailable for this device."}

        msrp = float(msrp_value)

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("ADMIN — DEPRECIATION FORECAST")
        print("=" * 70)

        print(f"Device       : {device}")

        print(f"Sub-device   : {sub_device}")

        print(f"Model        : {model_name}")

        print(f"Provider     : {provider}")

        print(f"Model Year   : {model_year}")

        print(f"Retail Price : RM {msrp:,.2f}")

        print(f"Forecast     : {model_year} → " f"{forecast_until}")

        # ----------------------------------------------------
        # TIMELINE ANCHORED TO DEVICE RELEASE YEAR
        #
        # The forecast always starts from the device's own
        # Model Year (not the server's current year), so admin
        # sees the device's full modelled history from release
        # through to the requested end year, not just the
        # forward-looking portion. There is no meaningful
        # trade-in value before a device has been released, so
        # the timeline can never start earlier than model_year.
        # ----------------------------------------------------

        forecast_start = model_year

        if forecast_start > forecast_until:

            return {"status": "error", "message": "Forecast end year must be " f"{forecast_start} or later, since " f"this device's Model Year is {model_year}."}

        results = []

        previous_value = None

        # ----------------------------------------------------
        # ACTUAL OBSERVATION YEAR
        #
        # If this exact record has a real, recorded trade-in
        # value, that value is ground truth for its collection
        # year and takes priority over the fitted curve for
        # that one year -- the curve still generates every
        # other year in the timeline.
        # ----------------------------------------------------

        ACTUAL_OBSERVATION_YEAR = 2026

        actual_trade_in_raw = base["Max. Trade-In Value (RM)"]

        has_actual_trade_in = not pd.isna(actual_trade_in_raw)

        for year in range(forecast_start, forecast_until + 1):

            # ------------------------------------------------
            # DEVICE AGE
            # ------------------------------------------------

            device_age = year - model_year

            # ------------------------------------------------
            # ACTUAL OBSERVATION (PRIORITY)
            #
            # This exact record's own recorded trade-in value,
            # used for its collection year instead of the
            # fitted curve, since real data is ground truth
            # where it exists.
            # ------------------------------------------------

            if year == ACTUAL_OBSERVATION_YEAR and has_actual_trade_in:

                prediction = float(actual_trade_in_raw)

                point_type = "actual_observation"

                print(f"Forecast {year}: " f"RM {prediction:,.2f} | " f"Tier: actual_observation | " f"Curve: recorded value")

            # ------------------------------------------------
            # AGE 0
            #
            # Device is in its model year.
            # Use Retail Price as the baseline value.
            # No depreciation curve is required.
            # ------------------------------------------------

            elif device_age == 0:

                prediction = msrp

                point_type = "msrp_baseline"

                print(f"Forecast {year}: " f"RM {prediction:,.2f} | " f"Tier: launch_price | " f"Curve: Retail price baseline")

            # ------------------------------------------------
            # AGE 1+
            #
            # Use depreciation fallback.
            # ------------------------------------------------

            else:

                fallback_result = fallback.predict(device=device, sub_device=sub_device, provider=provider, msrp=msrp, model_year=model_year, reference_year=year)

                # ------------------------------------------------
                # UNRESOLVED
                # ------------------------------------------------

                if fallback_result.predicted_value is None:

                    return {"status": "unresolved", "message": ("No suitable depreciation curve " "is available for this device."), "confidence_flag": fallback_result.confidence_flag}

                print(f"Forecast {year}: " f"RM {fallback_result.predicted_value:,.2f} | " f"Tier: {fallback_result.matched_tier} | " f"Curve: {fallback_result.form}")

                prediction = float(fallback_result.predicted_value)

                point_type = "curve_forecast"

            # ------------------------------------------------
            # CHANGE
            # ------------------------------------------------

            if previous_value is None:

                change = None

                change_percent = None

            else:

                change = prediction - previous_value

                if previous_value != 0:

                    change_percent = (change / previous_value) * 100

                else:

                    change_percent = None

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            results.append({"year": year, "device_age": device_age, "estimated_trade_in": round(prediction, 2), "change": (None if change is None else round(change, 2)), "change_percent": (None if change_percent is None else round(change_percent, 2)), "data_point_type": point_type})

            previous_value = prediction

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {"status": "success", "model": model_name, "device": device, "sub_device": sub_device, "provider": provider, "model_year": model_year, "msrp": round(msrp, 2), "current_year": current_year, "forecast_until": forecast_until, "method": "depreciation_curve_with_msrp_baseline", "results": results}

    except Exception as e:

        print("\nFORECAST ERROR:")

        print(repr(e))

        return {"status": "error", "message": str(e)}


# ============================================================
# CUSTOMER FRONTEND
# ============================================================


@app.get("/")
def customer_frontend():
    return FileResponse(ASSETS_DIR / "index.html")


@app.get("/customer-detail")
def customer_detail_page():
    return FileResponse(ASSETS_DIR / "customer-detail.html")


# ============================================================
# ADMIN FRONTEND
# ============================================================


@app.get("/admin")
def admin_frontend():
    return FileResponse(ASSETS_DIR / "admin.html")


# ============================================================
# ADMIN — DATA STATUS FRONTEND
# ============================================================


@app.get("/admin/status-page")
def admin_status_page():
    return FileResponse(ASSETS_DIR / "status.html")


@app.post("/admin/refresh")
def admin_refresh():
    try:
        refreshed = force_refresh_from_sheets()

        if not refreshed:
            raise HTTPException(status_code=503, detail="force_refresh_from_sheets() returned False. Check FastAPI terminal.")

        return {"success": True, "message": "Data refreshed successfully.", "dataVersion": _data_version, "rows": len(df)}

    except HTTPException:
        raise

    except Exception as exc:
        print(f"Admin refresh endpoint error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/health")
def health_check():

    return {"status": "online"}