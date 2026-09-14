# ============================================================
# APPLE TRADE-IN VALUATION API
# CUSTOMER-FACING API
# ============================================================

import pandas as pd
import numpy as np
import os
import io
import json
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
from internal.bulk_import import (
    analyze_upload,
    effective_record_to_row,
    apply_classified_rows,
    build_effective_record,
    normalize_master_model_number_column,
    json_safe,
    APPLY_NEW,
    APPLY_UPDATE,
    APPLY_SKIPPED_DUPLICATE,
    APPLY_QUEUED,
    APPLY_ERROR,
)


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

# Persistent home for Bulk Import rows classified as Needs Review
# or Invalid Data -- these are never written to Master, but must
# not be lost either (see admin_bulk_import_apply() /
# admin/review-queue/* below). A worksheet with this exact name
# must exist in the spreadsheet above (create it once, headers are
# written automatically on first use) for Sheets persistence to be
# active; until then this degrades the same way every other Sheets
# integration in this file does -- Admin Review Queue writes/reads
# fail with a clear 503 instead of silently losing data.
REVIEW_QUEUE_WORKSHEET_NAME = "Admin Review Queue"

# Column order used for every row written to the Review Queue
# worksheet. "Record JSON" carries the full proposed record
# (Effective Record values + provenance), the classification
# result that caused it to be queued, and the batch metadata it
# arrived with -- everything admin/review-queue/* needs to re-
# analyze and, eventually, apply the row. The other columns exist
# purely so the worksheet itself is readable/searchable directly
# in Google Sheets, mirroring the raw-data philosophy already used
# for "Cleaned Master".
REVIEW_QUEUE_COLUMNS = [
    "Queue ID",
    "Queued At",
    "Updated At",
    "Conflict Type",
    "Provider",
    "Device",
    "Sub-device",
    "Standardized Model",
    "Reasons",
    "Record JSON",
]

CONDITION_FIELD_MARKER = "condition"

VALID_DATE_MODES = {"collection_date", "price_last_updated"}


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


TEXT_COLUMNS = ["Device", "Sub-device", "Standardized Model", "Provider", "Storage Type", "Connectivity"]


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

    # ------------------------------------------------------------
    # COLLECTION DATE
    #
    # Metadata, not a matching/business field -- guarantee the
    # column exists so every downstream read/write path can rely on
    # it, but never invent a value for a row that doesn't have one.
    # ------------------------------------------------------------

    if "Collection Date" not in cleaned.columns:
        cleaned["Collection Date"] = np.nan

    # A column that is missing (backfilled as np.nan above) or that
    # happens to be entirely blank when read from CSV is inferred by
    # pandas as float64. Collection Date holds date *strings*, so a
    # float64 column here makes any later `df.at[i, "Collection Date"]
    # = "2026-09-01"` raise TypeError: Invalid value ... for dtype
    # 'float64'. Force object dtype unconditionally so every
    # downstream write path (Admin Modify, Bulk Import apply) can
    # safely store a string, regardless of how this column arrived.
    cleaned["Collection Date"] = cleaned["Collection Date"].astype(object)

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

# ============================================================
# ADMIN REVIEW QUEUE GOOGLE SHEETS SYNC
#
# Same access pattern as Customer Data: no in-memory dataframe and
# no local CSV mirror -- the worksheet itself is the single source
# of truth, read/written live via list_records() /
# append_generic_row() / delete_rows(). This is a deliberately
# different persistence strategy than Master's "CSV + full-
# overwrite Sheet sync" -- the Review Queue is admin-curated,
# individual rows are added/edited/removed one at a time, and
# there is no separate customer-facing or startup-time consumer
# that needs a local fallback copy of it the way Master's CSV
# provides for the pricing engine.
# ============================================================

review_queue_sheets_sync = SheetsSync(service_account_file=str(SHEETS_SERVICE_ACCOUNT_FILE), service_account_json=SHEETS_SERVICE_ACCOUNT_JSON, spreadsheet_id=SHEETS_SPREADSHEET_ID, worksheet_name=REVIEW_QUEUE_WORKSHEET_NAME)

if review_queue_sheets_sync.is_available:

    print("Admin Review Queue sync ready -> " f"worksheet '{REVIEW_QUEUE_WORKSHEET_NAME}'")

else:

    print("Admin Review Queue sync UNAVAILABLE -- Bulk Import rows that need review will fail to queue until this is resolved. " f"Create a worksheet named '{REVIEW_QUEUE_WORKSHEET_NAME}' in the spreadsheet to enable it.")

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

    Connectivity: Optional[str] = None

    Material: Optional[str] = None
    CaseSize: Optional[int] = None

    ChargingMethod: Optional[str] = None

    CollectionDate: Optional[str] = None


# ============================================================
# ADMIN — ADD DEVICE
# ============================================================


@app.post("/admin/add")
def admin_add_device(item: AdminAddDevice):

    global df, device_model_map, device_config_map, model_number_map

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
    # COLLECTION DATE
    # ========================================================

    collection_date = item.CollectionDate.strip() if item.CollectionDate else np.nan

    # ========================================================
    # CREATE NEW ROW
    # ========================================================

    new_row = {"Provider": provider, "Device": device, "Sub-device": sub_device, "Standardized Model": model_name, "Model Number": model_number, "Retail Price": msrp, "Storage (GB)": storage, "Storage Type": storage_type, "Connectivity": connectivity, "Material": (item.Material.strip() if item.Material else np.nan), "Max. Trade-In Value (RM)": trade_in_value, "Model_Year": model_year, "Case Size": case_size, "Charging Method": charging_method, "Collection Date": collection_date}

    # ========================================================
    # APPEND TO DATAFRAME
    # ========================================================

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df = normalize_master_model_number_column(df)

    device_model_map, device_config_map = build_device_maps(df)
    model_number_map = build_model_number_map(df)

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

    print(f"Collection Date : " f"{'N/A' if pd.isna(collection_date) else collection_date}")

    print("Device added successfully.")

    print("=" * 70)

    # ========================================================
    # RESPONSE
    # ========================================================

    return {"status": "success", "message": "Device added successfully.", "price_status": price_status, "model": model_name, "msrp": msrp, "trade_in_value": (None if pd.isna(trade_in_value) else trade_in_value), "collection_date": (None if pd.isna(collection_date) else collection_date), "sheets_sync": sync_result.as_dict()}


# ============================================================
# ADMIN — DATA STATUS
# ============================================================


@app.get("/admin/status")
def admin_status():
    refresh_data_if_stale()

    # --------------------------------------------------------
    # DEVICE-SPECIFIC INTENTIONAL MISSING FIELDS
    # --------------------------------------------------------

    excluded_fields = {"iPhone": {"Connectivity", "Material", "Case Size", "Charging Method", "Storage Type"}, "iPad": {"Storage Type", "Material", "Case Size", "Charging Method"}, "Mac": {"Connectivity", "Material", "Case Size", "Charging Method"}, "Apple Watch": {"Storage Type", "Charging Method"}, "AirPods": {"Storage (GB)", "Storage Type", "Connectivity", "Material", "Case Size"}}

    # --------------------------------------------------------
    # COLUMNS ACTUALLY USED BY THE APPLICATION
    # --------------------------------------------------------

    fields = ["Provider", "Device", "Sub-device", "Standardized Model", "Retail Price", "Storage (GB)", "Storage Type", "Connectivity", "Material", "Max. Trade-In Value (RM)", "Model_Year", "Case Size", "Charging Method"]

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

        records.append({"id": record_id, "model": clean_value("Standardized Model", row), "sub_device": clean_value("Sub-device", row), "storage": storage, "storage_type": clean_value("Storage Type", row), "connectivity": clean_value("Connectivity", row), "material": clean_value("Material", row), "provider": clean_value("Provider", row), "msrp": (None if pd.isna(row["Retail Price"]) else float(row["Retail Price"])), "trade_in_value": value, "collection_date": clean_value("Collection Date", row)})

    records = clean_json_value(records)

    return JSONResponse(content=records, headers={"X-Data-Version": _data_version})


# ============================================================
# ADMIN REVIEW QUEUE — HELPERS
#
# A queued row is persisted as one worksheet row: a handful of
# human-readable columns (see REVIEW_QUEUE_COLUMNS) plus one
# "Record JSON" column holding everything needed to redisplay,
# re-analyze, and eventually apply it -- the row's Effective
# Record (values + provenance), the classification result that
# caused it to be queued, and the date-mode/date-value metadata
# it arrived with.
#
# These helpers only ever build/read that row shape; they never
# touch Master and never decide whether a row should be queued --
# that decision is made once, in admin_bulk_import_apply() and
# admin_review_queue_apply(), from a classification already
# produced by analyze_upload().
# ============================================================


def _build_review_queue_record_json(row_result, effective_record, batch_meta):

    return json_safe({
        "classification": row_result.get("classification"),
        "conflict_type": row_result.get("conflict_type"),
        "reasons": row_result.get("reasons", []),
        "warnings": row_result.get("warnings", []),
        "unknown_fields": row_result.get("unknown_fields", []),
        "different_fields": row_result.get("different_fields", []),
        "match_index": row_result.get("match_index"),
        "matched_master": row_result.get("matched_master"),
        "what_this_means": row_result.get("what_this_means"),
        "effective_record": effective_record,
        "date_mode": (batch_meta or {}).get("date_mode"),
        "date_value": (batch_meta or {}).get("date_value"),
    })


def _build_review_queue_fields(queue_id, queued_at, updated_at, row_result, effective_record, batch_meta):

    values = (effective_record or {}).get("values", {}) or {}

    record_json = _build_review_queue_record_json(row_result, effective_record, batch_meta)

    def _display(value):
        return "" if value is None else str(value)

    return {
        "Queue ID": queue_id,
        "Queued At": queued_at,
        "Updated At": updated_at,
        "Conflict Type": _display(row_result.get("conflict_type")),
        "Provider": _display(values.get("Provider") or "Unknown"),
        "Device": _display(values.get("Device")),
        "Sub-device": _display(values.get("Sub-device")),
        "Standardized Model": _display(values.get("Standardized Model")),
        "Reasons": "; ".join(row_result.get("reasons") or []),
        "Record JSON": json.dumps(record_json),
    }


def _review_queue_item_from_sheet_record(record):
    """
    Convert one list_records() entry ({"row": int, "fields": {...}})
    into the JSON-friendly shape returned by the Admin Review Queue
    endpoints. Tolerates a corrupted/hand-edited "Record JSON" cell
    by falling back to empty classification metadata rather than
    raising -- a single bad row must never take down the whole
    queue listing.
    """

    fields = record.get("fields", {})

    raw_json = fields.get("Record JSON", "")

    try:
        record_json = json.loads(raw_json) if raw_json else {}
    except (TypeError, ValueError):
        record_json = {}

    # A row queued before this sanitization existed (or hand-edited
    # in the sheet) may still carry a literal NaN/Infinity token from
    # an un-sanitized write -- json.loads() happily parses those back
    # into real float('nan')/float('inf') values, which FastAPI's
    # JSONResponse (allow_nan=False) then refuses to serialize. Run
    # it through json_safe() so a legacy bad row degrades to null
    # instead of taking down the whole queue listing.
    record_json = json_safe(record_json)

    return {
        "row": record.get("row"),
        "queue_id": fields.get("Queue ID"),
        "queued_at": fields.get("Queued At"),
        "updated_at": fields.get("Updated At"),
        "conflict_type": fields.get("Conflict Type") or record_json.get("conflict_type"),
        "provider": fields.get("Provider"),
        "device": fields.get("Device"),
        "sub_device": fields.get("Sub-device"),
        "standardized_model": fields.get("Standardized Model"),
        "reasons": record_json.get("reasons", []),
        "warnings": record_json.get("warnings", []),
        "unknown_fields": record_json.get("unknown_fields", []),
        "different_fields": record_json.get("different_fields", []),
        "matched_master": record_json.get("matched_master"),
        "effective_record": record_json.get("effective_record", {}),
        "what_this_means": record_json.get("what_this_means"),
        "date_mode": record_json.get("date_mode"),
        "date_value": record_json.get("date_value"),
    }


def _find_review_queue_sheet_record(queue_id):
    """
    Look up a queued row's LIVE sheet row number + full field
    content by Queue ID. Returns None if not found. Raises
    HTTPException(503) if the Review Queue itself can't be read.

    Always re-fetched fresh (never cached) because the row number
    is only meaningful at the instant of this read -- see
    SheetsSync.delete_rows().
    """

    result = review_queue_sheets_sync.list_records()

    if not result.success:
        raise HTTPException(status_code=503, detail=result.error or "Unable to load the Admin Review Queue.")

    for record in result.records:
        if record.get("fields", {}).get("Queue ID") == queue_id:
            return record

    return None


def _queue_row_for_review(row_result, canonical_row, batch_meta):
    """
    Persist one classify_incoming_row() conflict outcome (Needs
    Review or Invalid Data) to the Admin Review Queue worksheet.

    Returns the SheetsSync SyncResult so the caller can decide how
    to report a failure -- this never raises, and never silently
    drops the row: a failed queue write is always surfaced back to
    the Admin rather than the row quietly disappearing.
    """

    # canonical_row already IS the effective/proposed record for
    # this row (post re-analysis) -- reuse analyze_upload()'s own
    # Effective Record builder so the queued record has the exact
    # same {values, provenance} shape the Admin preview already
    # knows how to render, with every field's provenance marked
    # "incoming" (nothing here is inherited a second time; that
    # inheritance already happened during re-analysis and is
    # baked into canonical_row).
    effective_record = build_effective_record(
        pre_inference_row=canonical_row,
        normalized_row=canonical_row,
        effective_row=canonical_row,
        inherited_fields=[],
        matched_master_row=None,
    )

    queue_id = uuid.uuid4().hex
    now = datetime.now().isoformat()

    queue_fields = _build_review_queue_fields(
        queue_id=queue_id,
        queued_at=now,
        updated_at=now,
        row_result=row_result,
        effective_record=effective_record,
        batch_meta=batch_meta,
    )

    sync_result = review_queue_sheets_sync.append_generic_row(queue_fields, REVIEW_QUEUE_COLUMNS)

    return queue_id, sync_result


def _recheck_review_queue_record(record, effective_record_override=None):
    """
    Re-analyze one Admin Review Queue row against the CURRENT
    Master dataset, persist the refreshed classification (and any
    admin edit) back to its worksheet row, and return the fresh
    classification result for the caller to act on.

    This is the shared core of both POST /admin/review-queue/{id}/
    recheck (report only) and POST /admin/review-queue/{id}/apply
    (report, then act on the result) -- an edit must never be lost
    just because the row still isn't valid yet, so both endpoints
    persist it the same way before doing anything else.

    Returns (row_result, canonical_row, stored_item, sheet_record)
    where sheet_record is the FRESH sheet row (row number + fields)
    for the just-persisted state, suitable for an immediate
    delete_rows() call by the caller.
    """

    stored_item = _review_queue_item_from_sheet_record(record)

    effective_record = effective_record_override or stored_item["effective_record"]

    proposed_row = effective_record_to_row(effective_record)

    proposed_df = pd.DataFrame([proposed_row])

    reanalysis = analyze_upload(raw_df=proposed_df, master_df=df.copy(), clean_fn=clean_dataset)

    row_result = reanalysis["rows"][0]
    canonical_row = reanalysis["canonical_df"].loc[0]

    # The re-analysis may itself enrich the record (e.g. Master-
    # inherited Retail Price) -- persist THAT, not the raw override,
    # so what's stored always matches what was actually classified.
    refreshed_effective_record = row_result["effective_record"]

    batch_meta = {"date_mode": stored_item.get("date_mode"), "date_value": stored_item.get("date_value")}

    unchanged = (
        effective_record_override is None
        and row_result.get("conflict_type") == stored_item.get("conflict_type")
        and row_result.get("reasons", []) == stored_item.get("reasons", [])
        and refreshed_effective_record == stored_item.get("effective_record", {})
    )

    if unchanged:
        return row_result, canonical_row, stored_item, record

    queue_fields = _build_review_queue_fields(
        queue_id=stored_item["queue_id"],
        queued_at=stored_item["queued_at"],
        updated_at=datetime.now().isoformat(),
        row_result=row_result,
        effective_record=refreshed_effective_record,
        batch_meta=batch_meta,
    )

    # 1. Safely APPEND the replacement record FIRST
    append_result = review_queue_sheets_sync.append_generic_row(queue_fields, REVIEW_QUEUE_COLUMNS)

    if not append_result.success:
        raise HTTPException(status_code=503, detail=append_result.error or "Unable to update the Admin Review Queue.")

    # 2. Only DELETE the old record if the append succeeded
    delete_result = review_queue_sheets_sync.delete_rows([{"row": record["row"], "fields": record["fields"]}])

    if not delete_result.success:
        raise HTTPException(status_code=503, detail=delete_result.error or "Unable to update the Admin Review Queue.")

    if not delete_result.deleted_rows:
        raise HTTPException(status_code=409, detail="This review queue item changed since it was loaded. Reloading...")

    fresh_record = _find_review_queue_sheet_record(stored_item["queue_id"])

    if fresh_record is None:
        raise HTTPException(status_code=503, detail="Lost track of the review queue item after updating it. Please reload the queue.")

    return row_result, canonical_row, stored_item, fresh_record


@app.post("/admin/bulk-import/preview")
def admin_bulk_import_preview(
    file: UploadFile = File(...),
    provider: str = Form("Unknown"),
    date_mode: str = Form("collection_date"),
    date_value: Optional[str] = Form(None),
):

    print("\n" + "=" * 70)
    print("ADMIN — BULK IMPORT PREVIEW")
    print("=" * 70)
    print(f"Provider: {provider!r}  |  Date mode: {date_mode!r} = {date_value!r}")

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    if date_mode not in VALID_DATE_MODES:
        raise HTTPException(
            status_code=400,
            detail="date_mode must be exactly one of 'collection_date' or 'price_last_updated'.",
        )

    try:
        raw_bytes = file.file.read()
        raw_df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if raw_df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows.")

    # Stamp Provider onto every row as an actual column -- overwrites
    # any pre-existing "Provider" column in the uploaded file, since
    # the admin's selection for this batch is authoritative.
    provider_value = (provider or "").strip() or "Unknown"
    raw_df["Provider"] = provider_value

    # Stamp Collection Date onto every row the same way -- this batch's
    # date (whichever of the two existing date_mode pickers the Admin
    # used) is the date associated with this provider's pricing data
    # for every row in the file. Reuses the existing date_mode/
    # date_value mechanism rather than adding a second date input;
    # blank/missing stays blank rather than inventing a date.
    collection_date_value = (date_value or "").strip()
    raw_df["Collection Date"] = collection_date_value or np.nan

    try:
        result = analyze_upload(
            raw_df=raw_df,
            master_df=df.copy(),
            clean_fn=clean_dataset,
        )
    except Exception as exc:
        print(f"Bulk import preview error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    # Metadata echo for the Preview modal's own display -- the actual
    # Collection Date value is now carried per-row via the stamped
    # column above (and from there through effective_record like any
    # other canonical field), not via this echo.
    result["date_mode"] = date_mode
    result["date_value"] = date_value

    result.pop("canonical_df", None)
    result.pop("master_norm_by_index", None)
    result.pop("master_device_by_index", None)
    result.pop("master_provider_by_index", None)
    result.pop("within_file_duplicates", None)

    return JSONResponse(content=result)


# ============================================================
# ADMIN BULK IMPORT — PREVIEW ROW RE-ANALYZE
#
# Supports the Admin Bulk Import Preview's Problematic Rows
# editor: after the Admin edits a Needs Review / Invalid Data
# row's proposed values, this re-classifies that ONE row against
# the CURRENT Master dataset so the Admin can see whether the
# edit resolved it before deciding to Apply / Resolve / Skip.
#
# This is a read-only counterpart to the Admin Review Queue's
# recheck endpoint (_recheck_review_queue_record below) and
# reuses the exact same building blocks (effective_record_to_row
# + analyze_upload) -- it is not a second classification system,
# just the existing one invoked one row at a time, before that
# row has been queued or applied anywhere. Nothing is persisted
# here: no CSV write, no Sheets write, no Review Queue entry.
# Resolving the row (Apply / Resolve-Skip) is a separate call to
# the existing /admin/bulk-import/apply endpoint, which already
# knows how to apply, skip, or -- if the row still isn't valid --
# queue it to the real Admin Review Queue.
# ============================================================


class BulkImportReanalyzeRowRequest(BaseModel):

    effective_record: Optional[dict] = None


@app.post("/admin/bulk-import/reanalyze-row")
def admin_bulk_import_reanalyze_row(item: BulkImportReanalyzeRowRequest):

    print("\n" + "=" * 70)
    print("ADMIN — BULK IMPORT PREVIEW ROW RE-ANALYZE")
    print("=" * 70)

    if not force_refresh_from_sheets():
        raise HTTPException(
            status_code=503,
            detail="Unable to refresh data from Google Sheets. Operation cancelled.",
        )

    proposed_row = effective_record_to_row(item.effective_record)

    if not proposed_row:
        raise HTTPException(
            status_code=400,
            detail="effective_record is required.",
        )

    proposed_df = pd.DataFrame([proposed_row])

    reanalysis = analyze_upload(
        raw_df=proposed_df,
        master_df=df.copy(),
        clean_fn=clean_dataset,
    )

    row_result = reanalysis["rows"][0]

    print(f"Classification: {row_result['classification']} / {row_result.get('conflict_type')}")
    print("=" * 70)

    return {
        "status": "success",
        "classification": row_result["classification"],
        "conflict_type": row_result.get("conflict_type"),
        "reasons": row_result.get("reasons", []),
        "warnings": row_result.get("warnings", []),
        "unknown_fields": row_result.get("unknown_fields", []),
        "different_fields": row_result.get("different_fields", []),
        "what_this_means": row_result.get("what_this_means"),
        "matched_master": row_result.get("matched_master"),
        "effective_record": row_result.get("effective_record"),
    }


# ============================================================
# ADMIN MODIFY REQUEST
# ============================================================


class AdminModifyDevice(BaseModel):

    id: int

    dataVersion: Optional[str] = None

    Provider: Optional[str] = None

    MSRP: Optional[float] = None

    TradeInValue: Optional[float] = None

    CollectionDate: Optional[str] = None


@app.post("/admin/modify")
def admin_modify_device(item: AdminModifyDevice):

    global df, device_model_map, device_config_map, model_number_map

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
    # COLLECTION DATE
    #
    # Only touched when explicitly provided, same as Provider above
    # -- editing Trade-In Value or MSRP alone must never clear an
    # existing Collection Date.
    # ========================================================

    if item.CollectionDate is not None:

        collection_date = item.CollectionDate.strip()

        old_collection_date = df.at[item.id, "Collection Date"] if "Collection Date" in df.columns else np.nan

        # Guard against a float64 Collection Date column (e.g. still
        # entirely blank on disk) rejecting a string value.
        if "Collection Date" in df.columns and df["Collection Date"].dtype != object:
            df["Collection Date"] = df["Collection Date"].astype(object)

        df.at[item.id, "Collection Date"] = collection_date if collection_date else np.nan

        print(f"Collection Date updated: " f"{old_collection_date} → {collection_date or 'N/A'}")

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

    df = normalize_master_model_number_column(df)

    device_model_map, device_config_map = build_device_maps(df)
    model_number_map = build_model_number_map(df)

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

    return {"status": "success", "message": "Record updated successfully.", "id": item.id, "provider": (str(df.at[item.id, "Provider"]) if not pd.isna(df.at[item.id, "Provider"]) else None), "msrp": (None if pd.isna(df.at[item.id, "Retail Price"]) else float(df.at[item.id, "Retail Price"])), "trade_in_value": (None if pd.isna(df.at[item.id, "Max. Trade-In Value (RM)"]) else float(df.at[item.id, "Max. Trade-In Value (RM)"])), "collection_date": (None if ("Collection Date" not in df.columns or pd.isna(df.at[item.id, "Collection Date"])) else str(df.at[item.id, "Collection Date"])), "sheets_sync": sync_result.as_dict()}


# ============================================================
# ADMIN — DELETE DEVICE
# ============================================================


@app.post("/admin/delete")
def admin_delete_device(item: dict):

    global df, device_model_map, device_config_map, model_number_map

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

    df = normalize_master_model_number_column(df)

    device_model_map, device_config_map = build_device_maps(df)
    model_number_map = build_model_number_map(df)

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

@app.post("/admin/bulk-import/apply")
def admin_bulk_import_apply(item: dict):

    global df, device_model_map, device_config_map, model_number_map

    print("\n" + "=" * 70)
    print("ADMIN — BULK IMPORT APPLY")
    print("=" * 70)

    # --------------------------------------------------------
    # REFRESH MASTER FROM GOOGLE SHEETS
    # --------------------------------------------------------

    if not force_refresh_from_sheets():
        raise HTTPException(
            status_code=503,
            detail="Unable to refresh data from Google Sheets. Operation cancelled."
        )

    # --------------------------------------------------------
    # CHECK REQUEST
    # --------------------------------------------------------

    if not item:
        return {
            "status": "error",
            "message": "Bulk import data is required."
        }

    print("Bulk import apply request received.")

    rows = item.get("rows", [])

    if not isinstance(rows, list):
        return {
            "status": "error",
            "message": "Invalid bulk import rows."
        }

    if not rows:
        return {
            "status": "success",
            "message": "No rows to import.",
            "master_rows": len(df),
        }

    batch_meta = {
        "date_mode": item.get("date_mode"),
        "date_value": item.get("date_value"),
    }

    proposed_rows = [
        effective_record_to_row(row.get("effective_record"))
        for row in rows
    ]

    print(f"Received {len(proposed_rows)} proposed import rows.")

    # --------------------------------------------------------
    # RE-ANALYZE ADMIN-EDITED PROPOSED ROWS AGAINST THE CURRENT
    # MASTER DATASET.
    #
    # This is the batch's single source of truth for what happens
    # next -- the classification the Admin saw in the preview modal
    # may be stale (Master can have changed since then), so nothing
    # is applied off the original preview result.
    # --------------------------------------------------------

    proposed_df = pd.DataFrame(proposed_rows)

    reanalysis = analyze_upload(
        raw_df=proposed_df,
        master_df=df.copy(),
        clean_fn=clean_dataset,
    )

    print("Re-analysis complete:", reanalysis["summary"])

    # --------------------------------------------------------
    # APPLY EVERY ROW INDEPENDENTLY.
    #
    # NEW / UPDATE rows are applied to Master; DUPLICATE rows are
    # skipped; NEEDS_REVIEW / INVALID_DATA rows are queued below --
    # none of these block one another.
    # --------------------------------------------------------

    apply_result = apply_classified_rows(
        reanalysis["canonical_df"],
        reanalysis["rows"],
        df,
    )

    outcomes = apply_result["outcomes"]

    # --------------------------------------------------------
    # PERSIST NEEDS_REVIEW / INVALID_DATA ROWS TO THE ADMIN REVIEW
    # QUEUE. A failure here is reported back, but never silently
    # drops the row, and never blocks the NEW/UPDATE rows below.
    # --------------------------------------------------------

    queued_ids = []
    queue_errors = []

    for outcome in outcomes:
        if outcome["action"] != APPLY_QUEUED:
            continue

        row_result = outcome["row_result"]
        canonical_row = reanalysis["canonical_df"].loc[outcome["row_index"]]

        queue_id, sync_result = _queue_row_for_review(
            row_result, canonical_row, batch_meta
        )

        if sync_result.success:
            queued_ids.append(queue_id)
        else:
            print(
                "WARNING: Failed to queue row "
                f"{outcome['row_index']} for review: {sync_result.error}"
            )
            queue_errors.append({
                "row_index": outcome["row_index"],
                "error": sync_result.error,
            })

    # --------------------------------------------------------
    # PERSIST MASTER CHANGES (only if anything actually changed).
    # --------------------------------------------------------

    changed = any(
        outcome["action"] in (APPLY_NEW, APPLY_UPDATE)
        for outcome in outcomes
    )

    sync_result = None

    if changed:
        df = apply_result["master_df"]

        df = normalize_master_model_number_column(df)

        device_model_map, device_config_map = build_device_maps(df)
        model_number_map = build_model_number_map(df)

        df.to_csv(DATA_FILE, index=False)

        bump_data_version()

        sync_result = sheets_sync.sync_dataset(df)

        if not sync_result.success:
            print(f"WARNING: Google Sheets sync failed after BULK IMPORT APPLY: {sync_result.error}")

    # --------------------------------------------------------
    # COUNTS + RESPONSE
    # --------------------------------------------------------

    counts = {
        "imported_new": 0,
        "imported_update": 0,
        "skipped_duplicate": 0,
        "queued_for_review": 0,
        "errors": 0,
    }

    for outcome in outcomes:
        if outcome["action"] == APPLY_NEW:
            counts["imported_new"] += 1
        elif outcome["action"] == APPLY_UPDATE:
            counts["imported_update"] += 1
        elif outcome["action"] == APPLY_SKIPPED_DUPLICATE:
            counts["skipped_duplicate"] += 1
        elif outcome["action"] == APPLY_QUEUED:
            counts["queued_for_review"] += 1
        elif outcome["action"] == APPLY_ERROR:
            counts["errors"] += 1

    print(f"Imported (new)      : {counts['imported_new']}")
    print(f"Imported (update)   : {counts['imported_update']}")
    print(f"Skipped (duplicate) : {counts['skipped_duplicate']}")
    print(f"Queued (review)     : {counts['queued_for_review']}")
    if queue_errors:
        print(f"Queue failures      : {len(queue_errors)}")
    print("=" * 70)

    message = (
        f"{counts['imported_new']} new record(s) added, "
        f"{counts['imported_update']} record(s) updated, "
        f"{counts['skipped_duplicate']} duplicate(s) skipped, "
        f"{counts['queued_for_review']} row(s) sent to the Admin Review Queue."
    )

    return {
        "status": "success" if not queue_errors else "partial",
        "message": message,
        "master_rows": len(df),
        **counts,
        "queued_ids": queued_ids,
        "queue_errors": queue_errors,
        "reanalysis_summary": reanalysis["summary"],
        "sheets_sync": sync_result.as_dict() if sync_result else None,
    }


# ============================================================
# ADMIN REVIEW QUEUE
#
# Rows Bulk Import could not import automatically (Needs Review /
# Invalid Data) land here instead of blocking the rest of their
# batch. An Admin can list them, edit the proposed record, re-check
# it against the CURRENT Master dataset without committing anything
# (recheck), and apply it once it resolves to New or Update (apply)
# -- or discard it outright.
#
# Persistence is the Review Queue worksheet itself (see
# review_queue_sheets_sync above) -- there is no in-memory
# dataframe and no local CSV mirror for this, the same design
# already used for Customer Data.
# ============================================================


@app.get("/admin/review-queue")
def admin_review_queue_list():

    result = review_queue_sheets_sync.list_records()

    if not result.success:
        raise HTTPException(status_code=503, detail=result.error or "Unable to load the Admin Review Queue.")

    items = [
        _review_queue_item_from_sheet_record(record)
        for record in result.records
    ]

    return {"status": "success", "items": items}


class ReviewQueueActionRequest(BaseModel):

    effective_record: Optional[dict] = None


@app.post("/admin/review-queue/{queue_id}/recheck")
def admin_review_queue_recheck(queue_id: str, item: ReviewQueueActionRequest):

    print("\n" + "=" * 70)
    print("ADMIN — REVIEW QUEUE RECHECK")
    print("=" * 70)

    if not force_refresh_from_sheets():
        raise HTTPException(status_code=503, detail="Unable to refresh data from Google Sheets. Operation cancelled.")

    record = _find_review_queue_sheet_record(queue_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Review queue item not found.")

    row_result, canonical_row, stored_item, fresh_record = _recheck_review_queue_record(
        record, effective_record_override=item.effective_record
    )

    print(f"Queue ID     : {queue_id}")
    print(f"Classification: {row_result['classification']} / {row_result.get('conflict_type')}")
    print("=" * 70)

    return {
        "status": "success",
        "classification": row_result["classification"],
        "conflict_type": row_result.get("conflict_type"),
        "reasons": row_result.get("reasons", []),
        "warnings": row_result.get("warnings", []),
        "unknown_fields": row_result.get("unknown_fields", []),
        "different_fields": row_result.get("different_fields", []),
        "what_this_means": row_result.get("what_this_means"),
        "matched_master": row_result.get("matched_master"),
        "effective_record": row_result.get("effective_record"),
        "item": _review_queue_item_from_sheet_record(fresh_record),
    }


@app.post("/admin/review-queue/{queue_id}/apply")
def admin_review_queue_apply(queue_id: str, item: ReviewQueueActionRequest):

    global df, device_model_map, device_config_map, model_number_map

    print("\n" + "=" * 70)
    print("ADMIN — REVIEW QUEUE APPLY")
    print("=" * 70)

    if not force_refresh_from_sheets():
        raise HTTPException(status_code=503, detail="Unable to refresh data from Google Sheets. Operation cancelled.")

    record = _find_review_queue_sheet_record(queue_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Review queue item not found.")

    row_result, canonical_row, stored_item, fresh_record = _recheck_review_queue_record(
        record, effective_record_override=item.effective_record
    )

    classification = row_result["classification"]
    conflict_type = row_result.get("conflict_type")

    # ----------------------------------------------------------
    # STILL NOT VALID -- leave it queued (already persisted above
    # with its refreshed reasons/effective record).
    # ----------------------------------------------------------

    if classification == "conflict" and conflict_type in ("needs_review", "invalid_data"):

        print(f"Queue ID {queue_id} still requires review: {row_result.get('reasons')}")
        print("=" * 70)

        return {
            "status": "still_needs_review",
            "classification": classification,
            "conflict_type": conflict_type,
            "reasons": row_result.get("reasons", []),
            "unknown_fields": row_result.get("unknown_fields", []),
            "different_fields": row_result.get("different_fields", []),
            "what_this_means": row_result.get("what_this_means"),
            "item": _review_queue_item_from_sheet_record(fresh_record),
        }

    # ----------------------------------------------------------
    # DUPLICATE -- already represented in Master; nothing to apply.
    # Discard the queue entry.
    # ----------------------------------------------------------

    if classification == "conflict" and conflict_type == "duplicate":

        delete_result = review_queue_sheets_sync.delete_rows(
            [{"row": fresh_record["row"], "fields": fresh_record["fields"]}]
        )

        if not delete_result.success:
            raise HTTPException(status_code=503, detail=delete_result.error or "Unable to update the Admin Review Queue.")

        print(f"Queue ID {queue_id} is now a duplicate of an existing Master row; discarded.")
        print("=" * 70)

        return {
            "status": "duplicate_discarded",
            "message": "This row already matches an existing Master record exactly, so it was discarded.",
        }

    # ----------------------------------------------------------
    # NEW / UPDATE -- apply to Master, then remove from the queue.
    # ----------------------------------------------------------

    single_canonical_df = pd.DataFrame([canonical_row])
    single_canonical_df.index = [0]

    apply_result = apply_classified_rows(single_canonical_df, [row_result], df)

    outcome = apply_result["outcomes"][0]

    if outcome["action"] == APPLY_ERROR:
        raise HTTPException(status_code=409, detail=outcome.get("error") or "Unable to apply this row.")

    df = apply_result["master_df"]

    df = normalize_master_model_number_column(df)

    device_model_map, device_config_map = build_device_maps(df)
    model_number_map = build_model_number_map(df)

    df.to_csv(DATA_FILE, index=False)

    bump_data_version()

    sync_result = sheets_sync.sync_dataset(df)

    if not sync_result.success:
        print(f"WARNING: Google Sheets sync failed after REVIEW QUEUE APPLY: {sync_result.error}")

    delete_result = review_queue_sheets_sync.delete_rows(
        [{"row": fresh_record["row"], "fields": fresh_record["fields"]}]
    )

    if not delete_result.success:
        print(f"WARNING: Failed to remove resolved item from the Admin Review Queue: {delete_result.error}")

    print(f"Queue ID {queue_id} resolved as {outcome['action']} and applied to Master.")
    print("=" * 70)

    return {
        "status": "success",
        "message": "Row applied to Master and removed from the Review Queue.",
        "action": outcome["action"],
        "master_rows": len(df),
        "sheets_sync": sync_result.as_dict(),
        "review_queue_removed": bool(delete_result.success and delete_result.deleted_rows),
    }


@app.post("/admin/review-queue/{queue_id}/discard")
def admin_review_queue_discard(queue_id: str):

    print("\n" + "=" * 70)
    print("ADMIN — REVIEW QUEUE DISCARD")
    print("=" * 70)

    record = _find_review_queue_sheet_record(queue_id)

    if record is None:
        raise HTTPException(status_code=404, detail="Review queue item not found.")

    delete_result = review_queue_sheets_sync.delete_rows(
        [{"row": record["row"], "fields": record["fields"]}]
    )

    if not delete_result.success:
        raise HTTPException(status_code=503, detail=delete_result.error or "Unable to update the Admin Review Queue.")

    if not delete_result.deleted_rows:
        raise HTTPException(status_code=409, detail="This review queue item changed since it was loaded. Reloading...")

    print(f"Queue ID {queue_id} discarded.")
    print("=" * 70)

    return {"status": "success", "message": "Review queue item discarded."}



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