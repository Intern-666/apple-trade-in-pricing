from dataclasses import dataclass; from pathlib import Path; from typing import Optional, List, Dict, Any; import json; import pandas as pd
@dataclass
class SyncResult:
    success: bool; rows_synced: Optional[int]; error: Optional[str]
    def as_dict(self):
        return {'success': self.success, 'rows_synced': self.rows_synced, 'error': self.error}
@dataclass
class FetchResult:
    success: bool; dataframe: Optional[pd.DataFrame]; rows_fetched: Optional[int]; error: Optional[str]
    def as_dict(self):
        return {'success': self.success, 'rows_fetched': self.rows_fetched, 'error': self.error}
@dataclass
class RecordsResult:
    success: bool; records: Optional[List[Dict[str, Any]]]; error: Optional[str]
    def as_dict(self):
        return {'success': self.success, 'records': self.records, 'error': self.error}
@dataclass
class DeleteRowsResult:
    success: bool; deleted_rows: Optional[List[int]]; skipped_rows: Optional[List[int]]; error: Optional[str]
    def as_dict(self):
        return {'success': self.success, 'deleted_rows': self.deleted_rows, 'skipped_rows': self.skipped_rows, 'error': self.error}
class SheetsSync:
    def __init__(self, service_account_file: str, spreadsheet_id: str, worksheet_name: str, service_account_json: Optional[str]=None):
        self.spreadsheet_id = spreadsheet_id; self.worksheet_name = worksheet_name; self._client = None; self._worksheet = None; self._init_error = None
        try:
            import gspread; from google.oauth2.service_account import Credentials; scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']
            if service_account_json:
                service_account_info = json.loads(service_account_json); credentials = Credentials.from_service_account_info(service_account_info, scopes=scopes)
            else:
                if not Path(service_account_file).exists():
                    raise FileNotFoundError(f"Service account credentials file not found at '{service_account_file}'. Google Sheets sync is disabled until this is provided.")
                credentials = Credentials.from_service_account_file(service_account_file, scopes=scopes)
            self._client = gspread.authorize(credentials); spreadsheet = self._client.open_by_key(self.spreadsheet_id); self._worksheet = spreadsheet.worksheet(self.worksheet_name)
        except Exception as exc:
            self._init_error = str(exc); print(f'Warning: Google Sheets sync could not be initialized: {self._init_error}')
    @property
    def is_available(self) -> bool:
        return self._worksheet is not None and self._init_error is None
    def sync_dataset(self, df: pd.DataFrame) -> SyncResult:
        if not self.is_available:
            return SyncResult(success=False, rows_synced=None, error=self._init_error or 'Google Sheets client is not initialized.')
        try:
            clean_df = df.where(pd.notna(df), ''); header = clean_df.columns.tolist(); values = clean_df.astype(str).values.tolist(); payload = [header] + values; existing_values = self._worksheet.get_all_values(); existing_row_count = len(existing_values); new_row_count = len(payload); self._worksheet.update(payload, 'A1', value_input_option='RAW')
            if existing_row_count > new_row_count:
                self._worksheet.batch_clear([f'A{new_row_count + 1}:ZZ{existing_row_count}'])
            return SyncResult(success=True, rows_synced=len(values), error=None)
        except Exception as exc:
            print(f'Warning: Google Sheets sync failed: {exc}'); return SyncResult(success=False, rows_synced=None, error=str(exc))
    def fetch_dataset(self) -> 'FetchResult':
        if not self.is_available:
            return FetchResult(success=False, dataframe=None, rows_fetched=None, error=self._init_error or 'Google Sheets client is not initialized.')
        try:
            records = self._worksheet.get_all_records()
            if not records:
                return FetchResult(success=False, dataframe=None, rows_fetched=0, error='Worksheet is empty or has no data rows below the header.')
            fetched_df = pd.DataFrame(records); return FetchResult(success=True, dataframe=fetched_df, rows_fetched=len(fetched_df), error=None)
        except Exception as exc:
            print(f'Warning: Google Sheets fetch failed: {exc}'); return FetchResult(success=False, dataframe=None, rows_fetched=None, error=str(exc))
    def append_record(self, record: dict) -> SyncResult:
        if not self.is_available:
            return SyncResult(success=False, rows_synced=0, error=self._init_error or 'Google Sheets sync is unavailable.')
        try:
            columns = ['Timestamp', 'Customer Name', 'Phone', 'Email', 'Preferred Contact', 'Device', 'Sub-device', 'Model', 'Model Number', 'Storage (GB)', 'Storage Type', 'Connectivity', 'Market Value (RM)', 'Final Trade-In Value (RM)']; existing_values = self._worksheet.get_all_values()
            if not existing_values or not any((str(cell).strip() for row in existing_values for cell in row)):
                self._worksheet.append_row(columns, value_input_option='USER_ENTERED')
            row = [record.get('Timestamp', ''), record.get('Customer Name', ''), "'" + str(record.get('Phone', '')), record.get('Email', ''), record.get('Preferred Contact', ''), record.get('Device', ''), record.get('Sub-device', ''), record.get('Model', ''), record.get('Model Number', ''), record.get('Storage (GB)', ''), record.get('Storage Type', ''), record.get('Connectivity', ''), record.get('Market Value (RM)', ''), record.get('Final Trade-In Value (RM)', '')]; self._worksheet.append_row(row, value_input_option='USER_ENTERED'); return SyncResult(success=True, rows_synced=1, error=None)
        except Exception as exc:
            return SyncResult(success=False, rows_synced=0, error=str(exc))
    def append_generic_row(self, fields: Dict[str, Any], columns: List[str]) -> SyncResult:
        if not self.is_available:
            return SyncResult(success=False, rows_synced=0, error=self._init_error or 'Google Sheets sync is unavailable.')
        try:
            existing_values = self._worksheet.get_all_values()
            if not existing_values or not any((str(cell).strip() for row in existing_values for cell in row)):
                self._worksheet.append_row(columns, value_input_option='USER_ENTERED')
            row = [str(fields.get(column, '')) for column in columns]; self._worksheet.append_row(row, value_input_option='USER_ENTERED'); return SyncResult(success=True, rows_synced=1, error=None)
        except Exception as exc:
            print(f'Warning: Google Sheets append_generic_row failed: {exc}'); return SyncResult(success=False, rows_synced=0, error=str(exc))
    def list_records(self) -> RecordsResult:
        if not self.is_available:
            return RecordsResult(success=False, records=None, error=self._init_error or 'Google Sheets client is not initialized.')
        try:
            values = self._worksheet.get_all_values()
            if not values:
                return RecordsResult(success=True, records=[], error=None)
            header = values[0]; data_rows = values[1:]; records = []
            for offset, row in enumerate(data_rows):
                padded = row + [''] * (len(header) - len(row)); row_number = offset + 2; records.append({'row': row_number, 'fields': dict(zip(header, padded[:len(header)]))})
            return RecordsResult(success=True, records=records, error=None)
        except Exception as exc:
            print(f'Warning: Google Sheets list_records failed: {exc}'); return RecordsResult(success=False, records=None, error=str(exc))
    def delete_rows(self, requested: List[Dict[str, Any]]) -> DeleteRowsResult:
        if not self.is_available:
            return DeleteRowsResult(success=False, deleted_rows=None, skipped_rows=None, error=self._init_error or 'Google Sheets client is not initialized.')
        if not requested:
            return DeleteRowsResult(success=True, deleted_rows=[], skipped_rows=[], error=None)
        try:
            values = self._worksheet.get_all_values(); header = values[0] if values else []; data_rows = values[1:] if values else []; confirmed_rows = []; skipped_rows = []
            for item in requested:
                row_number = item.get('row'); expected_fields = item.get('fields') or {}; index = row_number - 2 if row_number is not None else -1
                if row_number is None or index < 0 or index >= len(data_rows):
                    skipped_rows.append(row_number); continue
                current_row = data_rows[index]; padded = current_row + [''] * (len(header) - len(current_row)); current_fields = dict(zip(header, padded[:len(header)])); matches = all((str(current_fields.get(key, '')) == str(value) for key, value in expected_fields.items()))
                if matches:
                    confirmed_rows.append(row_number)
                else:
                    skipped_rows.append(row_number)
            for row_number in sorted(confirmed_rows, reverse=True):
                self._worksheet.delete_rows(row_number)
            return DeleteRowsResult(success=True, deleted_rows=sorted(confirmed_rows), skipped_rows=skipped_rows, error=None)
        except Exception as exc:
            print(f'Warning: Google Sheets delete_rows failed: {exc}'); return DeleteRowsResult(success=False, deleted_rows=None, skipped_rows=None, error=str(exc))