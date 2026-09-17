import pandas as pd; import numpy as np; import os; import io; import json; import uuid; import cv2; import pdfplumber; import re; from thefuzz import fuzz; from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request; from fastapi.middleware.cors import CORSMiddleware; from fastapi.responses import FileResponse, JSONResponse, RedirectResponse; from fastapi.staticfiles import StaticFiles; from pathlib import Path; from typing import Optional, cast, List; from pydantic import BaseModel; from datetime import datetime, timedelta; from zoneinfo import ZoneInfo; from internal.tradein_fallback import TradeInFallback; from internal.sheets_sync import SheetsSync; from internal.bulk_import import analyze_upload, effective_record_to_row, apply_classified_rows, build_effective_record, normalize_master_model_number_column, json_safe, APPLY_NEW, APPLY_UPDATE, APPLY_SKIPPED_DUPLICATE, APPLY_QUEUED, APPLY_ERROR; BASE_DIR = Path(__file__).resolve().parent.parent; APP_DIR = Path(__file__).resolve().parent; ASSETS_DIR = BASE_DIR / 'assets'; DATA_FILE = BASE_DIR / 'data' / 'master_msrp.csv'; FITTED_CURVES_FILE = BASE_DIR / 'data' / 'fitted_curves.csv'; SHEETS_SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE', str(BASE_DIR / 'internal' / 'service_account.json')); SHEETS_SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON'); SHEETS_SPREADSHEET_ID = '1TzySGhtEs-ptmzLHNxcJ5q_lQ9nGr6HDofy0IL7G1vs'; SHEETS_WORKSHEET_NAME = 'Cleaned Master'; CUSTOMER_SHEETS_WORKSHEET_NAME = 'Customer Data'; REVIEW_QUEUE_WORKSHEET_NAME = 'Admin Review Queue'; REVIEW_QUEUE_COLUMNS = ['Queue ID', 'Queued At', 'Updated At', 'Conflict Type', 'Provider', 'Device', 'Sub-device', 'Standardized Model', 'Reasons', 'Record JSON']; CONDITION_FIELD_MARKER = 'condition'; VALID_DATE_MODES = {'collection_date', 'price_last_updated'}; CUSTOMER_DETAIL_COOKIE = 'customer_detail_ack'; app = FastAPI(title='Apple Trade-In Valuation API'); app.mount('/assets', StaticFiles(directory=ASSETS_DIR), name='assets'); app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
def storage_to_gb(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if 'tb' in value:
        numbers = [x for x in value.replace(',', '').split() if x.replace('.', '', 1).isdigit()]
        if numbers:
            return float(numbers[0]) * 1024
    if 'gb' in value:
        numbers = [x for x in value.replace(',', '').split() if x.replace('.', '', 1).isdigit()]
        if numbers:
            return float(numbers[0])
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan
TEXT_COLUMNS = ['Device', 'Sub-device', 'Standardized Model', 'Provider', 'Storage Type', 'Connectivity']
def clean_dataset(raw_df):
    cleaned = raw_df.copy()
    for col in cleaned.columns:
        if str(cleaned[col].dtype).startswith('string'):
            cleaned[col] = cleaned[col].astype(object)
    if 'Storage (GB)' in cleaned.columns:
        cleaned['Storage (GB)'] = cleaned['Storage (GB)'].apply(storage_to_gb)
    numeric_cols = ['Max. Trade-In Value (RM)', 'Retail Price', 'Model_Year', 'Model Year', 'Case Size']
    for col in numeric_cols:
        if col in cleaned.columns:
            if cleaned[col].dtype == object or str(cleaned[col].dtype).startswith('string'):
                cleaned[col] = cleaned[col].astype(str).str.replace('[Rr][Mm]|\\s|,', '', regex=True)
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
    for col in TEXT_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna('Unknown').astype(str).str.strip().astype(object)
    if 'Collection Date' not in cleaned.columns:
        cleaned['Collection Date'] = np.nan
    cleaned['Collection Date'] = cleaned['Collection Date'].astype(object); return cleaned
def build_device_maps(cleaned_df):
    model_map = {}; config_map = {}
    for (device, sub_device, model_name), group in cleaned_df.groupby(['Device', 'Sub-device', 'Standardized Model']):
        model_map.setdefault(device, {}); model_map[device].setdefault(sub_device, {}); storages = group['Storage (GB)'].dropna().unique().tolist(); storages = sorted(storages); clean_storages = [int(x) if float(x).is_integer() else float(x) for x in storages]; model_map[device][sub_device][model_name] = clean_storages; config_map.setdefault(device, {}); config_map[device].setdefault(sub_device, {}); storage_types = []
        if 'Storage Type' in group.columns:
            storage_types = sorted((t for t in group['Storage Type'].dropna().unique().tolist() if t and t != 'Unknown'))
        connectivity_options = []
        if 'Connectivity' in group.columns:
            connectivity_options = sorted((c for c in group['Connectivity'].dropna().unique().tolist() if c and c != 'Unknown'))
        config_map[device][sub_device][model_name] = {'storageTypes': storage_types, 'connectivity': connectivity_options}
    return (model_map, config_map)
def build_model_number_map(model_number_df):
    model_number_map = {}
    for (device, sub_device, model_number, model_name), group in model_number_df.groupby(['Device', 'Sub-device', 'Model Number', 'Standardized Model']):
        if pd.isna(model_number):
            continue
        model_number = str(model_number).strip()
        if not model_number:
            continue
        device = str(device).strip(); sub_device = str(sub_device).strip(); model_name = str(model_name).strip(); model_number_map.setdefault(device, {}); model_number_map[device].setdefault(sub_device, {}); model_number_map[device][sub_device].setdefault(model_number, [])
        if model_name not in model_number_map[device][sub_device][model_number]:
            model_number_map[device][sub_device][model_number].append(model_name)
    return model_number_map
print(f'Loading Apple Trade-In data from {DATA_FILE.name}...'); df = pd.read_csv(DATA_FILE); print(f'Loaded {len(df)} rows.'); df = clean_dataset(df); device_model_map, device_config_map = build_device_maps(df); model_number_map = build_model_number_map(df); fallback = TradeInFallback(str(FITTED_CURVES_FILE), raw_data_path=str(DATA_FILE)); print(f'Loaded depreciation curves from {FITTED_CURVES_FILE.name}.'); sheets_sync = SheetsSync(service_account_file=str(SHEETS_SERVICE_ACCOUNT_FILE), service_account_json=SHEETS_SERVICE_ACCOUNT_JSON, spreadsheet_id=SHEETS_SPREADSHEET_ID, worksheet_name=SHEETS_WORKSHEET_NAME)
if sheets_sync.is_available:
    print(f"Google Sheets sync is ready, using worksheet '{SHEETS_WORKSHEET_NAME}'.")
else:
    print("Google Sheets sync isn't available -- admin writes will still save to the local CSV, but won't be mirrored to Google Sheets until this is resolved.")
customer_sheets_sync = SheetsSync(service_account_file=str(SHEETS_SERVICE_ACCOUNT_FILE), service_account_json=SHEETS_SERVICE_ACCOUNT_JSON, spreadsheet_id=SHEETS_SPREADSHEET_ID, worksheet_name=CUSTOMER_SHEETS_WORKSHEET_NAME)
if customer_sheets_sync.is_available:
    print(f"Customer Google Sheets sync is ready, using worksheet '{CUSTOMER_SHEETS_WORKSHEET_NAME}'.")
else:
    print("Customer Google Sheets sync isn't available.")
review_queue_sheets_sync = SheetsSync(service_account_file=str(SHEETS_SERVICE_ACCOUNT_FILE), service_account_json=SHEETS_SERVICE_ACCOUNT_JSON, spreadsheet_id=SHEETS_SPREADSHEET_ID, worksheet_name=REVIEW_QUEUE_WORKSHEET_NAME)
if review_queue_sheets_sync.is_available:
    print(f"Admin Review Queue sync is ready, using worksheet '{REVIEW_QUEUE_WORKSHEET_NAME}'.")
else:
    print(f"Admin Review Queue sync isn't available -- rows that need review won't be queued until this is fixed. Create a worksheet named '{REVIEW_QUEUE_WORKSHEET_NAME}' in the spreadsheet to enable it.")
print(f'{len(device_model_map)} devices available.'); REFRESH_INTERVAL_SECONDS = 10 * 60; _last_refresh_at = datetime.now() - timedelta(seconds=REFRESH_INTERVAL_SECONDS + 1)
def write_csv_atomically(raw_df, destination_path):
    temp_path = destination_path.with_suffix(destination_path.suffix + '.tmp')
    try:
        raw_df.to_csv(temp_path, index=False); os.replace(temp_path, destination_path); return True
    except Exception as exc:
        print(f'Warning: Failed to write refreshed data back to {destination_path.name}, local fallback file is now stale until the next successful refresh: {exc}')
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return False
def refresh_data_if_stale():
    global df, device_model_map, device_config_map, model_number_map, _last_refresh_at; seconds_since_refresh = (datetime.now() - _last_refresh_at).total_seconds()
    if seconds_since_refresh < REFRESH_INTERVAL_SECONDS:
        return
    _last_refresh_at = datetime.now()
    if not sheets_sync.is_available:
        return
    fetch_result = sheets_sync.fetch_dataset()
    if not fetch_result.success:
        print(f'Warning: Sheets refresh failed, keeping existing in-memory data: {fetch_result.error}'); return
    try:
        refreshed_df = clean_dataset(fetch_result.dataframe); refreshed_model_map, refreshed_config_map = build_device_maps(refreshed_df); refreshed_model_number_map = build_model_number_map(refreshed_df)
    except Exception as exc:
        print(f'Warning: Sheets refresh fetched data but it failed to clean/build correctly, keeping existing in-memory data: {exc}'); return
    df = refreshed_df; device_model_map = refreshed_model_map; device_config_map = refreshed_config_map; model_number_map = refreshed_model_number_map; bump_data_version(); write_csv_atomically(fetch_result.dataframe, DATA_FILE); print(f'Sheets refresh applied -- {fetch_result.rows_fetched} rows, {len(device_model_map)} devices')
_data_version = datetime.now().isoformat()
def force_refresh_from_sheets():
    global df, device_model_map, device_config_map, model_number_map, _last_refresh_at
    if not sheets_sync.is_available:
        print('Admin force refresh failed: Google Sheets sync is not available.'); return False
    try:
        fetch_result = sheets_sync.fetch_dataset()
        if not fetch_result.success or fetch_result.dataframe is None:
            print(f'Admin force refresh failed: {fetch_result.error}'); return False
        refreshed_df = clean_dataset(fetch_result.dataframe)
        if refreshed_df.empty:
            print('Admin force refresh returned an empty dataset.'); return False
        current = df.reset_index(drop=True).fillna('').astype(str); refreshed = refreshed_df.reset_index(drop=True).fillna('').astype(str); data_changed = not current.equals(refreshed); refreshed_model_map, refreshed_config_map = build_device_maps(refreshed_df); refreshed_model_number_map = build_model_number_map(refreshed_df); df = refreshed_df; device_model_map = refreshed_model_map; device_config_map = refreshed_config_map; model_number_map = refreshed_model_number_map; _last_refresh_at = datetime.now()
        if data_changed:
            bump_data_version(); print(f'Admin force refresh detected a data change: {len(df)} rows loaded.')
        else:
            print(f'Admin force refresh: no data changes detected ({len(df)} rows).')
        write_csv_atomically(fetch_result.dataframe, DATA_FILE); return True
    except Exception as exc:
        print(f'Admin force refresh error: {exc}'); return False
def bump_data_version():
    global _data_version; _data_version = datetime.now().isoformat()
bump_data_version(); print('Checking Google Sheets for the latest data at startup...'); refresh_data_if_stale()
@app.get('/available-models')
def get_models():
    refresh_data_if_stale(); return device_model_map
@app.get('/model-configuration')
def get_model_configuration():
    refresh_data_if_stale(); return device_config_map
@app.get('/model-numbers')
def get_model_numbers():
    refresh_data_if_stale(); return model_number_map
class DeviceInput(BaseModel):
    Device: str; SubDevice: str; Model: str; Storage: float | None = None; StorageType: str | None = None; Connectivity: str | None = None
@app.post('/predict')
def predict_price(item: DeviceInput):
    refresh_data_if_stale()
    if item.Device == 'AirPods':
        matches = df[(df['Device'] == item.Device) & (df['Sub-device'] == item.SubDevice) & (df['Standardized Model'] == item.Model)].copy()
    else:
        if item.Storage is None:
            return {'status': 'unresolved', 'estimated_value': None, 'message': 'Storage is required for this device.'}
        matches = df[(df['Device'] == item.Device) & (df['Sub-device'] == item.SubDevice) & (df['Standardized Model'] == item.Model) & np.isclose(df['Storage (GB)'], item.Storage, equal_nan=False)].copy()
        if item.StorageType:
            matches = matches[matches['Storage Type'] == item.StorageType]
        if item.Connectivity:
            matches = matches[matches['Connectivity'] == item.Connectivity]
    if matches.empty:
        return {'status': 'unresolved', 'estimated_value': None, 'message': 'This exact device configuration is not available in the database.'}
    trade_in_values = matches['Max. Trade-In Value (RM)'].dropna()
    if trade_in_values.empty:
        return {'status': 'unresolved', 'estimated_value': None, 'message': 'This exact device configuration does not have a trade-in value on record yet.'}
    median_price = trade_in_values.median()
    if pd.isna(median_price):
        return {'status': 'unresolved', 'estimated_value': None, 'message': 'This exact device configuration does not have a trade-in value on record yet.'}
    provider_count = matches['Provider'].nunique(); record_count = len(matches); return {'status': 'resolved', 'estimated_value': round(float(median_price), 2), 'method': 'exact_configuration_median', 'matching_records': record_count, 'provider_count': provider_count}
class CustomerTradeInRecord(BaseModel):
    customer: dict; device: dict; valuation: dict; createdAt: str
@app.post('/customer/trade-in')
def save_customer_trade_in(item: CustomerTradeInRecord):
    customer = item.customer; device = item.device; valuation = item.valuation; customer_name = str(customer.get('name', '')).strip(); customer_phone = str(customer.get('phone', '')).strip(); customer_email = str(customer.get('email', '')).strip(); preferred_contact = str(customer.get('preferredContact', '')).strip()
    if not customer_name:
        return {'status': 'error', 'message': 'Customer name is required.'}
    if not customer_phone:
        return {'status': 'error', 'message': 'Customer phone is required.'}
    if not customer_email:
        return {'status': 'error', 'message': 'Customer email is required.'}
    if not preferred_contact:
        return {'status': 'error', 'message': 'Preferred contact method is required.'}
    device_name = device.get('device'); sub_device = device.get('subDevice'); model_name = device.get('model')
    if not device_name:
        return {'status': 'error', 'message': 'Device is required.'}
    if not sub_device:
        return {'status': 'error', 'message': 'Sub-device is required.'}
    if not model_name:
        return {'status': 'error', 'message': 'Model is required.'}
    market_value = valuation.get('marketValue'); final_value = valuation.get('finalValue'); customer_record = {'Timestamp': datetime.now(ZoneInfo('Asia/Kuala_Lumpur')).strftime('%d %b %Y, %I:%M %p'), 'Customer Name': customer_name, 'Phone': customer_phone, 'Email': customer_email, 'Preferred Contact': preferred_contact, 'Device': device_name, 'Sub-device': sub_device, 'Model': model_name, 'Model Number': device.get('modelNumber') or 'N/A', 'Storage (GB)': device.get('storage'), 'Storage Type': device.get('storageType') or 'N/A', 'Connectivity': device.get('connectivity') or 'N/A', 'Market Value (RM)': market_value, 'Final Trade-In Value (RM)': final_value}; sync_result = customer_sheets_sync.append_record(customer_record)
    if not sync_result.success:
        raise HTTPException(status_code=500, detail='The record was not saved because Google Sheets synchronization failed.')
    return {'status': 'success', 'message': 'Trade-in record received successfully.', 'customer': customer_name, 'model': model_name, 'sheets': sync_result.as_dict()}
class AdminAddDevice(BaseModel):
    Device: str; SubDevice: str; Model: str; ModelNumber: Optional[str] = None; Provider: Optional[str] = None; MSRP: float; TradeInValue: Optional[float] = None; Storage: Optional[float] = None; StorageType: Optional[str] = None; ModelYear: Optional[int] = None; Connectivity: Optional[str] = None; Material: Optional[str] = None; CaseSize: Optional[int] = None; ChargingMethod: Optional[str] = None; CollectionDate: Optional[str] = None
@app.post('/admin/add')
def admin_add_device(item: AdminAddDevice):
    global df, device_model_map, device_config_map, model_number_map
    if not force_refresh_from_sheets():
        raise HTTPException(status_code=503, detail='Unable to refresh data from Google Sheets. Operation cancelled.')
    device = item.Device.strip(); sub_device = item.SubDevice.strip(); model_name = item.Model.strip(); model_number = item.ModelNumber.strip() if item.ModelNumber else np.nan
    if not device:
        raise HTTPException(status_code=400, detail='Device is required.')
    if not sub_device:
        raise HTTPException(status_code=400, detail='Sub-device is required.')
    if not model_name:
        raise HTTPException(status_code=400, detail='Model is required.')
    charging_method = item.ChargingMethod.strip() if item.ChargingMethod else np.nan
    if device == 'AirPods' and pd.isna(charging_method):
        raise HTTPException(status_code=400, detail='Charging method is required.')
    if item.MSRP < 0:
        raise HTTPException(status_code=400, detail='Retail price cannot be negative.')
    msrp = float(item.MSRP); provider = item.Provider.strip() if item.Provider else 'Unknown'
    if item.TradeInValue is None:
        trade_in_value = np.nan; price_status = 'N/A'
    else:
        if item.TradeInValue < 0:
            raise HTTPException(status_code=400, detail='Trade-in value cannot be negative.')
        trade_in_value = float(item.TradeInValue); price_status = 'confirmed'
    storage = item.Storage; storage_type = item.StorageType.strip() if item.StorageType else np.nan
    if device != 'AirPods' and storage is None:
        raise HTTPException(status_code=400, detail='Storage is required.')
    if device == 'Mac' and pd.isna(storage_type):
        raise HTTPException(status_code=400, detail='Storage type is required.')
    if device == 'AirPods':
        storage = np.nan; storage_type = np.nan
    connectivity = item.Connectivity.strip() if item.Connectivity else np.nan
    if device in ('iPad', 'Apple Watch') and pd.isna(connectivity):
        raise HTTPException(status_code=400, detail='Connectivity is required.')
    material = item.Material.strip() if item.Material else np.nan
    if device == 'Apple Watch' and pd.isna(material):
        raise HTTPException(status_code=400, detail='Material is required.')
    case_size = int(item.CaseSize) if item.CaseSize is not None else np.nan
    if device == 'Apple Watch' and item.CaseSize is None:
        raise HTTPException(status_code=400, detail='Case size is required.')
    model_year = item.ModelYear
    if model_year is None:
        raise HTTPException(status_code=400, detail='Model year is required.')
    if model_year < 1976:
        raise HTTPException(status_code=400, detail='Invalid Apple model year.')
    collection_date = item.CollectionDate.strip() if item.CollectionDate else np.nan; new_row = {'Provider': provider, 'Device': device, 'Sub-device': sub_device, 'Standardized Model': model_name, 'Model Number': model_number, 'Retail Price': msrp, 'Storage (GB)': storage, 'Storage Type': storage_type, 'Connectivity': connectivity, 'Material': material, 'Max. Trade-In Value (RM)': trade_in_value, 'Model_Year': model_year, 'Case Size': case_size, 'Charging Method': charging_method, 'Collection Date': collection_date}; updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True); updated_df = normalize_master_model_number_column(updated_df); updated_model_map, updated_config_map = build_device_maps(updated_df); updated_model_number_map = build_model_number_map(updated_df); sync_result = sheets_sync.sync_dataset(updated_df)
    if not sync_result.success:
        return {'status': 'error', 'message': 'The device was not added because Google Sheets synchronization failed.', 'sheets_sync': sync_result.as_dict()}
    df = updated_df; device_model_map = updated_model_map; device_config_map = updated_config_map; model_number_map = updated_model_number_map; df.to_csv(DATA_FILE, index=False); bump_data_version(); return {'status': 'success', 'message': 'Device added successfully.', 'price_status': price_status, 'model': model_name, 'msrp': msrp, 'trade_in_value': None if pd.isna(trade_in_value) else trade_in_value, 'collection_date': None if pd.isna(collection_date) else collection_date, 'sheets_sync': sync_result.as_dict()}
@app.get('/admin/status')
def admin_status():
    refresh_data_if_stale(); excluded_fields = DEVICE_EXCLUDED_FIELDS; fields = ADMIN_EDITABLE_FIELDS
    def is_missing(value):
        if pd.isna(value):
            return True
        if isinstance(value, str):
            value = value.strip().lower(); return value in {'', 'n/a', 'na', 'unknown', 'none', 'nan'}
        return False
    categories = []
    for field in fields:
        if field not in df.columns:
            continue
        affected_records = []
        for index, row in df.iterrows():
            device = None if pd.isna(row['Device']) else str(row['Device']).strip()
            if device in excluded_fields and field in excluded_fields[device]:
                continue
            if not is_missing(row[field]):
                continue
            affected_records.append({'id': int(cast(int, index)), 'device': device, 'sub_device': None if pd.isna(row['Sub-device']) else str(row['Sub-device']).strip(), 'model': None if pd.isna(row['Standardized Model']) else str(row['Standardized Model']).strip(), 'provider': None if pd.isna(row['Provider']) else str(row['Provider']).strip(), 'storage': None if pd.isna(row['Storage (GB)']) else float(row['Storage (GB)']), 'storage_type': None if pd.isna(row['Storage Type']) else str(row['Storage Type']).strip(), 'connectivity': None if pd.isna(row['Connectivity']) else str(row['Connectivity']).strip(), 'material': None if pd.isna(row['Material']) else str(row['Material']).strip(), 'case_size': None if pd.isna(row['Case Size']) else str(row['Case Size']).strip(), 'charging_method': None if pd.isna(row['Charging Method']) else str(row['Charging Method']).strip(), 'trade_in_value': None if pd.isna(row['Max. Trade-In Value (RM)']) else float(row['Max. Trade-In Value (RM)'])})
        if affected_records:
            categories.append({'field': field, 'count': len(affected_records), 'records': affected_records})
    total_missing = sum((category['count'] for category in categories)); return {'total_records': int(len(df)), 'categories': categories, 'total_missing': int(total_missing)}
@app.get('/admin/data-version')
def admin_data_version():
    refresh_data_if_stale(); return {'dataVersion': _data_version}
@app.get('/admin/models')
def admin_models():
    refresh_data_if_stale(); models = {}
    for device, group in df.groupby('Device'):
        values = group['Standardized Model'].dropna().astype(str).unique().tolist(); models[device] = sorted(values)
    return models
@app.get('/admin/forecast-models')
def admin_forecast_models():
    refresh_data_if_stale(); hierarchy = {}
    for (device, sub_device), group in df.groupby(['Device', 'Sub-device']):
        if pd.isna(device) or pd.isna(sub_device):
            continue
        device_str = str(device); sub_device_str = str(sub_device)
        if device_str not in hierarchy:
            hierarchy[device_str] = {}
        models = group['Standardized Model'].dropna().astype(str).unique().tolist(); hierarchy[device_str][sub_device_str] = sorted(models)
    return hierarchy
@app.get('/admin/records')
def admin_records(device: str, model: str, sub_device: Optional[str]=None):
    matches = df[(df['Device'] == device) & (df['Standardized Model'] == model)]
    if sub_device:
        matches = matches[matches['Sub-device'] == sub_device]
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
    relevant_fields = get_relevant_admin_fields(device); records = []
    for index, row in matches.iterrows():
        record_id = int(cast(int, index)); trade_in_value = row['Max. Trade-In Value (RM)']; trade_in_value = None if pd.isna(trade_in_value) else float(trade_in_value); storage = row['Storage (GB)']; storage = None if pd.isna(storage) else float(storage); records.append({'id': record_id, 'device': clean_value('Device', row), 'model': clean_value('Standardized Model', row), 'sub_device': clean_value('Sub-device', row), 'provider': clean_value('Provider', row), 'msrp': None if pd.isna(row['Retail Price']) else float(row['Retail Price']), 'storage': storage, 'storage_type': clean_value('Storage Type', row), 'connectivity': clean_value('Connectivity', row), 'material': clean_value('Material', row), 'trade_in_value': trade_in_value, 'model_year': None if 'Model_Year' not in row.index or pd.isna(row['Model_Year']) else float(row['Model_Year']), 'case_size': clean_value('Case Size', row), 'charging_method': clean_value('Charging Method', row), 'collection_date': clean_value('Collection Date', row), 'relevant_fields': relevant_fields})
    records = clean_json_value(records); return JSONResponse(content=records, headers={'X-Data-Version': _data_version})
def _build_review_queue_record_json(row_result, effective_record, batch_meta):
    return json_safe({'classification': row_result.get('classification'), 'conflict_type': row_result.get('conflict_type'), 'reasons': row_result.get('reasons', []), 'warnings': row_result.get('warnings', []), 'unknown_fields': row_result.get('unknown_fields', []), 'different_fields': row_result.get('different_fields', []), 'match_index': row_result.get('match_index'), 'matched_master': row_result.get('matched_master'), 'what_this_means': row_result.get('what_this_means'), 'effective_record': effective_record, 'date_mode': (batch_meta or {}).get('date_mode'), 'date_value': (batch_meta or {}).get('date_value')})
def _build_review_queue_fields(queue_id, queued_at, updated_at, row_result, effective_record, batch_meta):
    values = (effective_record or {}).get('values', {}) or {}; record_json = _build_review_queue_record_json(row_result, effective_record, batch_meta)
    def _display(value):
        return '' if value is None else str(value)
    return {'Queue ID': queue_id, 'Queued At': queued_at, 'Updated At': updated_at, 'Conflict Type': _display(row_result.get('conflict_type')), 'Provider': _display(values.get('Provider') or 'Unknown'), 'Device': _display(values.get('Device')), 'Sub-device': _display(values.get('Sub-device')), 'Standardized Model': _display(values.get('Standardized Model')), 'Reasons': '; '.join(row_result.get('reasons') or []), 'Record JSON': json.dumps(record_json)}
def _review_queue_item_from_sheet_record(record):
    fields = record.get('fields', {}); raw_json = fields.get('Record JSON', '')
    try:
        record_json = json.loads(raw_json) if raw_json else {}
    except (TypeError, ValueError):
        record_json = {}
    record_json = json_safe(record_json); return {'row': record.get('row'), 'queue_id': fields.get('Queue ID'), 'queued_at': fields.get('Queued At'), 'updated_at': fields.get('Updated At'), 'conflict_type': fields.get('Conflict Type') or record_json.get('conflict_type'), 'provider': fields.get('Provider'), 'device': fields.get('Device'), 'sub_device': fields.get('Sub-device'), 'standardized_model': fields.get('Standardized Model'), 'reasons': record_json.get('reasons', []), 'warnings': record_json.get('warnings', []), 'unknown_fields': record_json.get('unknown_fields', []), 'different_fields': record_json.get('different_fields', []), 'matched_master': record_json.get('matched_master'), 'effective_record': record_json.get('effective_record', {}), 'what_this_means': record_json.get('what_this_means'), 'date_mode': record_json.get('date_mode'), 'date_value': record_json.get('date_value')}
def _find_review_queue_sheet_record(queue_id):
    result = review_queue_sheets_sync.list_records()
    if not result.success:
        raise HTTPException(status_code=503, detail=result.error or 'Unable to load the Admin Review Queue.')
    for record in result.records:
        if record.get('fields', {}).get('Queue ID') == queue_id:
            return record
    return None
def _queue_row_for_review(row_result, canonical_row, batch_meta):
    effective_record = build_effective_record(pre_inference_row=canonical_row, normalized_row=canonical_row, effective_row=canonical_row, inherited_fields=[], matched_master_row=None); queue_id = uuid.uuid4().hex; now = datetime.now().isoformat(); queue_fields = _build_review_queue_fields(queue_id=queue_id, queued_at=now, updated_at=now, row_result=row_result, effective_record=effective_record, batch_meta=batch_meta); sync_result = review_queue_sheets_sync.append_generic_row(queue_fields, REVIEW_QUEUE_COLUMNS); return (queue_id, sync_result)
def _recheck_review_queue_record(record, effective_record_override=None):
    stored_item = _review_queue_item_from_sheet_record(record); effective_record = effective_record_override or stored_item['effective_record']; proposed_row = effective_record_to_row(effective_record); proposed_df = pd.DataFrame([proposed_row]); reanalysis = analyze_upload(raw_df=proposed_df, master_df=df.copy(), clean_fn=clean_dataset); row_result = reanalysis['rows'][0]; canonical_row = reanalysis['canonical_df'].loc[0]; refreshed_effective_record = row_result['effective_record']; batch_meta = {'date_mode': stored_item.get('date_mode'), 'date_value': stored_item.get('date_value')}; unchanged = effective_record_override is None and row_result.get('conflict_type') == stored_item.get('conflict_type') and (row_result.get('reasons', []) == stored_item.get('reasons', [])) and (refreshed_effective_record == stored_item.get('effective_record', {}))
    if unchanged:
        return (row_result, canonical_row, stored_item, record)
    queue_fields = _build_review_queue_fields(queue_id=stored_item['queue_id'], queued_at=stored_item['queued_at'], updated_at=datetime.now().isoformat(), row_result=row_result, effective_record=refreshed_effective_record, batch_meta=batch_meta); append_result = review_queue_sheets_sync.append_generic_row(queue_fields, REVIEW_QUEUE_COLUMNS)
    if not append_result.success:
        raise HTTPException(status_code=503, detail=append_result.error or 'Unable to update the Admin Review Queue.')
    delete_result = review_queue_sheets_sync.delete_rows([{'row': record['row'], 'fields': record['fields']}])
    if not delete_result.success:
        raise HTTPException(status_code=503, detail=delete_result.error or 'Unable to update the Admin Review Queue.')
    if not delete_result.deleted_rows:
        raise HTTPException(status_code=409, detail='This review queue item changed since it was loaded. Reloading...')
    fresh_record = _find_review_queue_sheet_record(stored_item['queue_id'])
    if fresh_record is None:
        raise HTTPException(status_code=503, detail='Lost track of the review queue item after updating it. Please reload the queue.')
    return (row_result, canonical_row, stored_item, fresh_record)
ocr_reader = None
def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        import easyocr
        ocr_reader = easyocr.Reader(['en'], gpu=False, model_storage_directory="/tmp/easyocr_models")
    return ocr_reader
def validate_extraction_upload(file: UploadFile, ext: str):
    MAX_FILE_SIZE = 10 * 1024 * 1024; file.file.seek(0, 2); file_size = file.file.tell(); file.file.seek(0)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail='This file is too large to process. Please upload a file under 10MB.')
    allowed_extensions = ['.csv', '.xls', '.xlsx', '.pdf', '.jpg', '.jpeg', '.png']
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="We can't read this file type. Please upload a JPG, PNG, PDF, Excel, or CSV.")
def preprocess_image_for_ocr(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8); img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="We couldn't read this image. Please check the photo and try again.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); denoised = cv2.medianBlur(gray, 3); binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2); return binary
def extract_raw_text(file: UploadFile, ext: str) -> list:
    extracted_lines = []
    try:
        if ext == '.pdf':
            with pdfplumber.open(file.file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_lines.extend(text.split('\n'))
        elif ext in ['.jpg', '.jpeg', '.png']:
            if ocr_reader is None:
                raise HTTPException(status_code=500, detail='Image processing is currently unavailable. Please try again later.')
            image_bytes = file.file.read(); processed_img = preprocess_image_for_ocr(image_bytes); results =get_ocr_reader().readtext(processed_img, detail=1)
            for bbox, text, prob in results:
                if prob > 0.4:
                    extracted_lines.append(text)
    except HTTPException:
        raise
    except Exception as e:
        print(f'Extraction engine error: {e}'); raise HTTPException(status_code=500, detail='Something went wrong while reading this file. Please try again in a moment.')
    if not extracted_lines:
        raise HTTPException(status_code=400, detail="We couldn't find any text in this file. Please check the photo and try again.")
    return extracted_lines
@app.post('/admin/extract')
async def api_admin_extract(file: UploadFile=File(...)):
    ext = Path(file.filename).suffix.lower(); validate_extraction_upload(file, ext); refresh_data_if_stale(); apple_devices_master = df[['Device', 'Sub-device', 'Standardized Model']].drop_duplicates()
    if ext in ['.csv', '.xls', '.xlsx']:
        if ext == '.csv':
            df_upload = pd.read_csv(file.file, dtype=object)
        else:
            df_upload = pd.read_excel(file.file, dtype=object)
        candidates = ['model', 'model name', 'device', 'product', 'name', 'item', 'description']; model_col = None
        for col in df_upload.columns:
            if str(col).lower().strip() in candidates:
                model_col = col; break
        if not model_col:
            for col in df_upload.select_dtypes(include=['object', 'string']).columns:
                model_col = col; break
        if not model_col:
            raise HTTPException(status_code=400, detail='Could not identify a model name column in the spreadsheet.')
        extracted_rows = []
        for _, row in df_upload.iterrows():
            val = str(row[model_col])
            if len(val) < 4 or val.lower() == 'nan':
                continue
            best_match = None; highest_score = 0
            for _, master_row in apple_devices_master.iterrows():
                target = str(master_row['Standardized Model']); score = fuzz.token_set_ratio(val.lower(), target.lower())
                if score > 88 and score > highest_score:
                    highest_score = score; best_match = master_row
            if best_match is not None:
                row_dict = row.to_dict(); row_dict['Device'] = best_match['Device']; row_dict['Sub-device'] = best_match['Sub-device']; row_dict['Standardized Model'] = best_match['Standardized Model']; extracted_rows.append(row_dict)
        if not extracted_rows:
            raise HTTPException(status_code=400, detail="We couldn't find any recognized Apple devices in this file.")
        result_df = pd.DataFrame(extracted_rows)
    else:
        raw_lines = extract_raw_text(file, ext); extracted_data = []
        for line in raw_lines:
            if not isinstance(line, str) or len(line.strip()) < 4:
                continue
            best_match = None; highest_score = 0
            for _, master_row in apple_devices_master.iterrows():
                target = str(master_row['Standardized Model']); score = fuzz.token_set_ratio(line.lower(), target.lower())
                if score > 88 and score > highest_score:
                    highest_score = score; best_match = master_row
            if best_match is not None:
                storage = None; price = None; temp_line = line; storage_match = re.search('\\b(\\d+)\\s*(GB|TB)\\b', temp_line, re.IGNORECASE)
                if storage_match:
                    val = int(storage_match.group(1)); unit = storage_match.group(2).upper()
                    if unit == 'TB':
                        val *= 1024
                    storage = val; temp_line = temp_line[:storage_match.start()] + temp_line[storage_match.end():]
                price_match = re.search('(?:RM|Price:?|\\$)\\s*(\\d{1,5}(?:\\.\\d{2})?)', temp_line, re.IGNORECASE)
                if price_match:
                    price = float(price_match.group(1))
                else:
                    numbers = re.findall('\\b(\\d{3,5}(?:\\.\\d{2})?)\\b', temp_line)
                    if numbers:
                        price = float(numbers[-1])
                extracted_data.append({'Device': best_match['Device'], 'Sub-device': best_match['Sub-device'], 'Standardized Model': best_match['Standardized Model'], 'Storage (GB)': storage if storage else '', 'Retail Price': price if price else ''})
        if not extracted_data:
            raise HTTPException(status_code=400, detail="We couldn't find any recognized Apple devices in this file.")
        result_df = pd.DataFrame(extracted_data)
    csv_buffer = io.StringIO(); result_df.to_csv(csv_buffer, index=False); return JSONResponse(content={'status': 'success', 'csv_content': csv_buffer.getvalue()})
@app.post('/admin/bulk-import/preview')
def admin_bulk_import_preview(file: UploadFile=File(...), provider: str=Form('Unknown'), date_mode: str=Form('collection_date'), date_value: Optional[str]=Form(None)):
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail='Please upload a .csv file.')
    if date_mode not in VALID_DATE_MODES:
        raise HTTPException(status_code=400, detail="date_mode must be exactly one of 'collection_date' or 'price_last_updated'.")
    try:
        raw_bytes = file.file.read(); raw_df = pd.read_csv(io.BytesIO(raw_bytes), dtype=object)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Could not parse CSV: {exc}')
    if raw_df.empty:
        raise HTTPException(status_code=400, detail='Uploaded CSV has no rows.')
    provider_value = (provider or '').strip() or 'Unknown'; raw_df['Provider'] = provider_value; collection_date_value = (date_value or '').strip(); raw_df['Collection Date'] = collection_date_value or np.nan
    try:
        result = analyze_upload(raw_df=raw_df, master_df=df.copy(), clean_fn=clean_dataset)
    except Exception as exc:
        print(f'Bulk import preview failed: {exc}'); raise HTTPException(status_code=500, detail=str(exc))
    result['date_mode'] = date_mode; result['date_value'] = date_value; result.pop('canonical_df', None); result.pop('master_norm_by_index', None); result.pop('master_device_by_index', None); result.pop('master_provider_by_index', None); result.pop('within_file_duplicates', None); return JSONResponse(content=result)
class BulkImportReanalyzeRowRequest(BaseModel):
    effective_record: Optional[dict] = None
@app.post('/admin/bulk-import/reanalyze-row')
def admin_bulk_import_reanalyze_row(item: BulkImportReanalyzeRowRequest):
    if not force_refresh_from_sheets():
        raise HTTPException(status_code=503, detail='Unable to refresh data from Google Sheets. Operation cancelled.')
    proposed_row = effective_record_to_row(item.effective_record)
    if not proposed_row:
        raise HTTPException(status_code=400, detail='effective_record is required.')
    proposed_df = pd.DataFrame([proposed_row], dtype=object); reanalysis = analyze_upload(raw_df=proposed_df, master_df=df.copy(), clean_fn=clean_dataset); row_result = reanalysis['rows'][0]; return {'status': 'success', 'classification': row_result['classification'], 'conflict_type': row_result.get('conflict_type'), 'reasons': row_result.get('reasons', []), 'warnings': row_result.get('warnings', []), 'unknown_fields': row_result.get('unknown_fields', []), 'different_fields': row_result.get('different_fields', []), 'what_this_means': row_result.get('what_this_means'), 'matched_master': row_result.get('matched_master'), 'effective_record': row_result.get('effective_record')}
class AdminModifyDevice(BaseModel):
    id: int; dataVersion: Optional[str] = None; Provider: Optional[str] = None; Device: Optional[str] = None; SubDevice: Optional[str] = None; StandardizedModel: Optional[str] = None; MSRP: Optional[float] = None; Storage: Optional[float] = None; StorageType: Optional[str] = None; Connectivity: Optional[str] = None; Material: Optional[str] = None; TradeInValue: Optional[float] = None; ModelYear: Optional[float] = None; CaseSize: Optional[str] = None; ChargingMethod: Optional[str] = None; CollectionDate: Optional[str] = None
@app.post('/admin/modify')
def admin_modify_device(item: AdminModifyDevice):
    global df, device_model_map, device_config_map, model_number_map
    if not force_refresh_from_sheets():
        raise HTTPException(status_code=503, detail='Unable to refresh data from Google Sheets. Operation cancelled.')
    if item.id < 0 or item.id >= len(df):
        return {'status': 'error', 'message': 'Record not found.'}
    if item.dataVersion is not None and item.dataVersion != _data_version:
        raise HTTPException(status_code=409, detail='The underlying data has changed since you loaded this record. Reloading...')
    if item.MSRP is not None and item.MSRP < 0:
        return {'status': 'error', 'message': 'Retail price cannot be negative.'}
    if item.TradeInValue is not None and item.TradeInValue < 0:
        return {'status': 'error', 'message': 'Trade-in value cannot be negative.'}
    updated_df = df.copy()
    if item.Provider is not None:
        provider = item.Provider.strip()
        if not provider:
            return {'status': 'error', 'message': 'Provider cannot be empty.'}
        updated_df.at[item.id, 'Provider'] = provider
    if item.Device is not None:
        device = item.Device.strip()
        if not device:
            return {'status': 'error', 'message': 'Device cannot be empty.'}
        updated_df.at[item.id, 'Device'] = device
    if item.SubDevice is not None:
        sub_device = item.SubDevice.strip()
        if not sub_device:
            return {'status': 'error', 'message': 'Sub-device cannot be empty.'}
        updated_df.at[item.id, 'Sub-device'] = sub_device
    if item.StandardizedModel is not None:
        standardized_model = item.StandardizedModel.strip()
        if not standardized_model:
            return {'status': 'error', 'message': 'Standardized Model cannot be empty.'}
        updated_df.at[item.id, 'Standardized Model'] = standardized_model
    if item.CollectionDate is not None:
        collection_date = item.CollectionDate.strip()
        if 'Collection Date' in updated_df.columns and updated_df['Collection Date'].dtype != object:
            updated_df['Collection Date'] = updated_df['Collection Date'].astype(object)
        updated_df.at[item.id, 'Collection Date'] = collection_date if collection_date else np.nan
    if item.MSRP is not None:
        if item.MSRP < 0:
            raise HTTPException(status_code=400, detail='Retail price cannot be negative.')
        updated_df.at[item.id, 'Retail Price'] = float(item.MSRP)
    if item.Storage is not None:
        if item.Storage < 0:
            raise HTTPException(status_code=400, detail='Storage cannot be negative.')
        updated_df.at[item.id, 'Storage (GB)'] = float(item.Storage); updated_df.at[item.id, 'Storage (GB)'] = float(item.Storage)
    if item.StorageType is not None:
        updated_df.at[item.id, 'Storage Type'] = item.StorageType.strip()
    if item.Connectivity is not None:
        updated_df.at[item.id, 'Connectivity'] = item.Connectivity.strip()
    if item.Material is not None:
        updated_df.at[item.id, 'Material'] = item.Material.strip()
    if item.TradeInValue is not None:
        if item.TradeInValue < 0:
            raise HTTPException(status_code=400, detail='Trade-in value cannot be negative.')
        updated_df.at[item.id, 'Max. Trade-In Value (RM)'] = float(item.TradeInValue)
    if item.ModelYear is not None:
        if item.ModelYear < 0:
            raise HTTPException(status_code=400, detail='Model Year cannot be negative.')
        updated_df.at[item.id, 'Model_Year'] = float(item.ModelYear)
    if item.CaseSize is not None:
        updated_df.at[item.id, 'Case Size'] = item.CaseSize.strip()
    if item.ChargingMethod is not None:
        updated_df.at[item.id, 'Charging Method'] = item.ChargingMethod.strip()
    updated_df = normalize_master_model_number_column(updated_df); updated_model_map, updated_config_map = build_device_maps(updated_df); updated_model_number_map = build_model_number_map(updated_df); sync_result = sheets_sync.sync_dataset(updated_df)
    if not sync_result.success:
        return {'status': 'error', 'message': 'The record was not saved because Google Sheets synchronization failed.', 'sheets_sync': sync_result.as_dict()}
    df = updated_df; device_model_map = updated_model_map; device_config_map = updated_config_map; model_number_map = updated_model_number_map; df.to_csv(DATA_FILE, index=False); bump_data_version(); return {'status': 'success', 'message': 'Record updated successfully.', 'id': item.id, 'provider': str(df.at[item.id, 'Provider']) if not pd.isna(df.at[item.id, 'Provider']) else None, 'msrp': None if pd.isna(df.at[item.id, 'Retail Price']) else float(df.at[item.id, 'Retail Price']), 'trade_in_value': None if pd.isna(df.at[item.id, 'Max. Trade-In Value (RM)']) else float(df.at[item.id, 'Max. Trade-In Value (RM)']), 'collection_date': None if 'Collection Date' not in df.columns or pd.isna(df.at[item.id, 'Collection Date']) else str(df.at[item.id, 'Collection Date']), 'sheets_sync': sync_result.as_dict()}
@app.post('/admin/delete')
def admin_delete_device(item: dict):
    global df, device_model_map, device_config_map, model_number_map
    if not force_refresh_from_sheets():
        raise HTTPException(status_code=503, detail='Unable to refresh data from Google Sheets. Operation cancelled.')
    record_id = item.get('id')
    if record_id is None:
        return {'status': 'error', 'message': 'Record ID is required.'}
    try:
        record_id = int(record_id)
    except (ValueError, TypeError):
        return {'status': 'error', 'message': 'Invalid record ID.'}
    if record_id not in df.index:
        return {'status': 'error', 'message': 'Record not found.'}
    submitted_version = item.get('dataVersion')
    if submitted_version is not None and submitted_version != _data_version:
        raise HTTPException(status_code=409, detail='The underlying data has changed since you loaded this record. Reloading...')
    df = df.drop(index=record_id).reset_index(drop=True); df = normalize_master_model_number_column(df); device_model_map, device_config_map = build_device_maps(df); model_number_map = build_model_number_map(df); df.to_csv(DATA_FILE, index=False); bump_data_version(); sync_result = sheets_sync.sync_dataset(df); return {'status': 'success', 'message': 'Record deleted successfully.', 'sheets_sync': sync_result.as_dict()}
@app.post('/admin/bulk-import/apply')
def admin_bulk_import_apply(item: dict):
    global df, device_model_map, device_config_map, model_number_map
    if not force_refresh_from_sheets():
        raise HTTPException(status_code=503, detail='Unable to refresh data from Google Sheets. Operation cancelled.')
    if not item:
        return {'status': 'error', 'message': 'Bulk import data is required.'}
    rows = item.get('rows', [])
    if not isinstance(rows, list):
        return {'status': 'error', 'message': 'Invalid bulk import rows.'}
    if not rows:
        return {'status': 'success', 'message': 'No rows to import.', 'master_rows': len(df)}
    batch_meta = {'date_mode': item.get('date_mode'), 'date_value': item.get('date_value')}; proposed_rows = [effective_record_to_row(row.get('effective_record')) for row in rows]; proposed_df = pd.DataFrame(proposed_rows); reanalysis = analyze_upload(raw_df=proposed_df, master_df=df.copy(), clean_fn=clean_dataset); apply_result = apply_classified_rows(reanalysis['canonical_df'], reanalysis['rows'], df); outcomes = apply_result['outcomes']; queued_ids = []; queue_errors = []
    for outcome in outcomes:
        if outcome['action'] != APPLY_QUEUED:
            continue
        row_result = outcome['row_result']; canonical_row = reanalysis['canonical_df'].loc[outcome['row_index']]; queue_id, sync_result = _queue_row_for_review(row_result, canonical_row, batch_meta)
        if sync_result.success:
            queued_ids.append(queue_id)
        else:
            queue_errors.append({'row_index': outcome['row_index'], 'error': sync_result.error})
    changed = any((outcome['action'] in (APPLY_NEW, APPLY_UPDATE) for outcome in outcomes)); sync_result = None
    if changed:
        df = apply_result['master_df']; df = normalize_master_model_number_column(df); device_model_map, device_config_map = build_device_maps(df); model_number_map = build_model_number_map(df); df.to_csv(DATA_FILE, index=False); bump_data_version(); sync_result = sheets_sync.sync_dataset(df)
    counts = {'imported_new': 0, 'imported_update': 0, 'skipped_duplicate': 0, 'queued_for_review': 0, 'errors': 0}
    for outcome in outcomes:
        if outcome['action'] == APPLY_NEW:
            counts['imported_new'] += 1
        elif outcome['action'] == APPLY_UPDATE:
            counts['imported_update'] += 1
        elif outcome['action'] == APPLY_SKIPPED_DUPLICATE:
            counts['skipped_duplicate'] += 1
        elif outcome['action'] == APPLY_QUEUED:
            counts['queued_for_review'] += 1
        elif outcome['action'] == APPLY_ERROR:
            counts['errors'] += 1
    message = f'{counts['imported_new']} new record(s) added, {counts['imported_update']} record(s) updated, {counts['skipped_duplicate']} duplicate(s) skipped, {counts['queued_for_review']} row(s) sent to the Admin Review Queue.'; return {'status': 'success' if not queue_errors else 'partial', 'message': message, 'master_rows': len(df), **counts, 'queued_ids': queued_ids, 'queue_errors': queue_errors, 'reanalysis_summary': reanalysis['summary'], 'sheets_sync': sync_result.as_dict() if sync_result else None}
@app.get('/admin/review-queue')
def admin_review_queue_list():
    result = review_queue_sheets_sync.list_records()
    if not result.success:
        raise HTTPException(status_code=503, detail=result.error or 'Unable to load the Admin Review Queue.')
    items = [_review_queue_item_from_sheet_record(record) for record in result.records]; return {'status': 'success', 'items': items}
class ReviewQueueActionRequest(BaseModel):
    effective_record: Optional[dict] = None
@app.post('/admin/review-queue/{queue_id}/recheck')
def admin_review_queue_recheck(queue_id: str, item: ReviewQueueActionRequest):
    if not force_refresh_from_sheets():
        raise HTTPException(status_code=503, detail='Unable to refresh data from Google Sheets. Operation cancelled.')
    record = _find_review_queue_sheet_record(queue_id)
    if record is None:
        raise HTTPException(status_code=404, detail='Review queue item not found.')
    row_result, canonical_row, stored_item, fresh_record = _recheck_review_queue_record(record, effective_record_override=item.effective_record); return {'status': 'success', 'classification': row_result['classification'], 'conflict_type': row_result.get('conflict_type'), 'reasons': row_result.get('reasons', []), 'warnings': row_result.get('warnings', []), 'unknown_fields': row_result.get('unknown_fields', []), 'different_fields': row_result.get('different_fields', []), 'what_this_means': row_result.get('what_this_means'), 'matched_master': row_result.get('matched_master'), 'effective_record': row_result.get('effective_record'), 'item': _review_queue_item_from_sheet_record(fresh_record)}
@app.post('/admin/review-queue/{queue_id}/apply')
def admin_review_queue_apply(queue_id: str, item: ReviewQueueActionRequest):
    global df, device_model_map, device_config_map, model_number_map
    if not force_refresh_from_sheets():
        raise HTTPException(status_code=503, detail='Unable to refresh data from Google Sheets. Operation cancelled.')
    record = _find_review_queue_sheet_record(queue_id)
    if record is None:
        raise HTTPException(status_code=404, detail='Review queue item not found.')
    row_result, canonical_row, stored_item, fresh_record = _recheck_review_queue_record(record, effective_record_override=item.effective_record); classification = row_result['classification']; conflict_type = row_result.get('conflict_type')
    if classification == 'conflict' and conflict_type in ('needs_review', 'invalid_data'):
        return {'status': 'still_needs_review', 'classification': classification, 'conflict_type': conflict_type, 'reasons': row_result.get('reasons', []), 'unknown_fields': row_result.get('unknown_fields', []), 'different_fields': row_result.get('different_fields', []), 'what_this_means': row_result.get('what_this_means'), 'item': _review_queue_item_from_sheet_record(fresh_record)}
    if classification == 'conflict' and conflict_type == 'duplicate':
        delete_result = review_queue_sheets_sync.delete_rows([{'row': fresh_record['row'], 'fields': fresh_record['fields']}])
        if not delete_result.success:
            raise HTTPException(status_code=503, detail=delete_result.error or 'Unable to update the Admin Review Queue.')
        return {'status': 'duplicate_discarded', 'message': 'This row already matches an existing Master record exactly, so it was discarded.'}
    single_canonical_df = pd.DataFrame([canonical_row]); single_canonical_df.index = [0]; apply_result = apply_classified_rows(single_canonical_df, [row_result], df); outcome = apply_result['outcomes'][0]
    if outcome['action'] == APPLY_ERROR:
        raise HTTPException(status_code=409, detail=outcome.get('error') or 'Unable to apply this row.')
    df = apply_result['master_df']; df = normalize_master_model_number_column(df); device_model_map, device_config_map = build_device_maps(df); model_number_map = build_model_number_map(df); df.to_csv(DATA_FILE, index=False); bump_data_version(); sync_result = sheets_sync.sync_dataset(df); delete_result = review_queue_sheets_sync.delete_rows([{'row': fresh_record['row'], 'fields': fresh_record['fields']}]); return {'status': 'success', 'message': 'Row applied to Master and removed from the Review Queue.', 'action': outcome['action'], 'master_rows': len(df), 'sheets_sync': sync_result.as_dict(), 'review_queue_removed': bool(delete_result.success and delete_result.deleted_rows)}
@app.post('/admin/review-queue/{queue_id}/discard')
def admin_review_queue_discard(queue_id: str):
    record = _find_review_queue_sheet_record(queue_id)
    if record is None:
        raise HTTPException(status_code=404, detail='Review queue item not found.')
    delete_result = review_queue_sheets_sync.delete_rows([{'row': record['row'], 'fields': record['fields']}])
    if not delete_result.success:
        raise HTTPException(status_code=503, detail=delete_result.error or 'Unable to update the Admin Review Queue.')
    if not delete_result.deleted_rows:
        raise HTTPException(status_code=409, detail='This review queue item changed since it was loaded. Reloading...')
    return {'status': 'success', 'message': 'Review queue item discarded.'}
@app.get('/admin/customers')
def admin_list_customers():
    result = customer_sheets_sync.list_records()
    if not result.success:
        raise HTTPException(status_code=503, detail=result.error or 'Unable to load customer data.')
    customers = []
    for record in result.records:
        fields = {key: value for key, value in record['fields'].items() if CONDITION_FIELD_MARKER not in key.strip().lower()}; customers.append({'row': record['row'], 'fields': fields})
    return {'status': 'success', 'customers': customers}
class CustomerDeleteRequest(BaseModel):
    rows: List[dict]
@app.post('/admin/customers/delete')
def admin_delete_customers(item: CustomerDeleteRequest):
    if not item.rows:
        return {'status': 'error', 'message': 'No records selected.'}
    result = customer_sheets_sync.delete_rows(item.rows)
    if not result.success:
        raise HTTPException(status_code=503, detail=result.error or 'Unable to delete customer records.')
    deleted_count = len(result.deleted_rows); skipped_count = len(result.skipped_rows)
    if skipped_count and (not deleted_count):
        raise HTTPException(status_code=409, detail='The selected record(s) changed since the list was loaded. Reloading...')
    if skipped_count:
        return {'status': 'partial', 'message': f'{deleted_count} record(s) deleted. {skipped_count} record(s) were skipped because their data changed since the list was loaded. Reloading...', 'deleted_rows': result.deleted_rows, 'skipped_rows': result.skipped_rows}
    return {'status': 'success', 'message': f'{deleted_count} record(s) deleted.', 'deleted_rows': result.deleted_rows}
@app.post('/admin/forecast')
def admin_forecast(item: dict):
    refresh_data_if_stale()
    try:
        current_year = datetime.now().year; record_id = item.get('record_id'); forecast_until = item.get('forecast_until')
        if record_id is None:
            return {'status': 'error', 'message': 'Record ID is required.'}
        if forecast_until is None:
            return {'status': 'error', 'message': 'Forecast end year is required.'}
        record_id = int(record_id); forecast_until = int(forecast_until)
        if forecast_until < current_year:
            return {'status': 'error', 'message': f'Forecast year must be {current_year} or later.'}
        if record_id not in df.index:
            return {'status': 'error', 'message': 'Record not found.'}
        base = df.loc[record_id].copy(); device = str(base['Device']); sub_device = str(base['Sub-device']); model_name = str(base['Standardized Model']); provider = str(base['Provider']); model_year_value = base['Model_Year']
        if pd.isna(model_year_value):
            return {'status': 'unresolved', 'message': 'Model year is unavailable for this device.'}
        model_year = int(float(model_year_value)); msrp_value = base['Retail Price']
        if pd.isna(msrp_value):
            return {'status': 'unresolved', 'message': 'Retail price is unavailable for this device.'}
        msrp = float(msrp_value); forecast_start = model_year
        if forecast_start > forecast_until:
            return {'status': 'error', 'message': f"Forecast end year must be {forecast_start} or later, since this device's Model Year is {model_year}."}
        results = []; previous_value = None; ACTUAL_OBSERVATION_YEAR = 2026; actual_trade_in_raw = base['Max. Trade-In Value (RM)']; has_actual_trade_in = not pd.isna(actual_trade_in_raw)
        for year in range(forecast_start, forecast_until + 1):
            device_age = year - model_year
            if year == ACTUAL_OBSERVATION_YEAR and has_actual_trade_in:
                prediction = float(actual_trade_in_raw); point_type = 'actual_observation'
            elif device_age == 0:
                prediction = msrp; point_type = 'msrp_baseline'
            else:
                fallback_result = fallback.predict(device=device, sub_device=sub_device, provider=provider, msrp=msrp, model_year=model_year, reference_year=year)
                if fallback_result.predicted_value is None:
                    return {'status': 'unresolved', 'message': 'No suitable depreciation curve is available for this device.', 'confidence_flag': fallback_result.confidence_flag}
                prediction = float(fallback_result.predicted_value); point_type = 'curve_forecast'
            if previous_value is None:
                change = None; change_percent = None
            else:
                change = prediction - previous_value
                if previous_value != 0:
                    change_percent = change / previous_value * 100
                else:
                    change_percent = None
            results.append({'year': year, 'device_age': device_age, 'estimated_trade_in': round(prediction, 2), 'change': None if change is None else round(change, 2), 'change_percent': None if change_percent is None else round(change_percent, 2), 'data_point_type': point_type}); previous_value = prediction
        return {'status': 'success', 'model': model_name, 'device': device, 'sub_device': sub_device, 'provider': provider, 'model_year': model_year, 'msrp': round(msrp, 2), 'current_year': current_year, 'forecast_until': forecast_until, 'method': 'depreciation_curve_with_msrp_baseline', 'results': results}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
@app.get('/')
def customer_frontend(request: Request):
    if not request.cookies.get(CUSTOMER_DETAIL_COOKIE):
        return RedirectResponse(url='/customer-detail')
    return FileResponse(ASSETS_DIR / 'index.html')
@app.get('/customer-detail')
def customer_detail_page():
    return FileResponse(ASSETS_DIR / 'customer-detail.html')
@app.post('/customer-detail/complete')
def customer_detail_complete():
    # Marks this browser session as having filled in the customer-detail
    # form, so the '/' route above will let it through. This is what
    # actually gates the site server-side -- sessionStorage alone can't be
    # checked by the server, and it can't stop someone from requesting '/'
    # directly (curl, JS disabled, etc.), so this cookie is the real gate.
    response = JSONResponse(content={'status': 'success'})
    response.set_cookie(key=CUSTOMER_DETAIL_COOKIE, value='1', httponly=True, samesite='lax', path='/')
    return response
@app.get('/admin')
def admin_frontend():
    return FileResponse(ASSETS_DIR / 'admin.html')
DEVICE_EXCLUDED_FIELDS = {'iPhone': {'Connectivity', 'Material', 'Case Size', 'Charging Method', 'Storage Type'}, 'iPad': {'Storage Type', 'Material', 'Case Size', 'Charging Method'}, 'Mac': {'Connectivity', 'Material', 'Case Size', 'Charging Method'}, 'Apple Watch': {'Storage Type', 'Charging Method'}, 'AirPods': {'Storage (GB)', 'Storage Type', 'Connectivity', 'Material', 'Case Size'}}; ADMIN_EDITABLE_FIELDS = ['Provider', 'Device', 'Sub-device', 'Standardized Model', 'Retail Price', 'Storage (GB)', 'Storage Type', 'Connectivity', 'Material', 'Max. Trade-In Value (RM)', 'Model_Year', 'Case Size', 'Charging Method', 'Collection Date']
def get_relevant_admin_fields(device):
    excluded = DEVICE_EXCLUDED_FIELDS.get(device, set()); return [field for field in ADMIN_EDITABLE_FIELDS if field not in excluded]
@app.get('/admin/status-page')
def admin_status_page():
    return FileResponse(ASSETS_DIR / 'status.html')
@app.post('/admin/refresh')
def admin_refresh():
    try:
        refreshed = force_refresh_from_sheets()
        if not refreshed:
            raise HTTPException(status_code=503, detail='force_refresh_from_sheets() returned False. Check FastAPI terminal.')
        return {'success': True, 'message': 'Data refreshed successfully.', 'dataVersion': _data_version, 'rows': len(df)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
@app.get('/health')
def health_check():
    return {'status': 'online'}