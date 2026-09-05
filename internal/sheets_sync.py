"""
sheets_sync.py

Mirrors the admin tool's master dataset to a Google Sheet after every
write (Add / Modify / Delete), so the company's Google Sheets-based
workflows always see the current data.

STRATEGY: FULL OVERWRITE
------------------------------------------------------------------
Rather than tracking per-row appends/updates/deletes against the
Sheet, every sync clears the target worksheet and rewrites it
entirely from the current in-memory dataframe. This is deliberately
simple and self-healing:

    - No incremental state to track or get out of sync.
    - A failed sync never leaves the Sheet half-updated in a way
      that compounds -- the next successful sync fully corrects it.
    - The Sheet is always either "fully current" or "stale from one
      failed attempt", never "partially and confusingly wrong".

The tradeoff is a slower per-sync cost (rewriting the whole sheet),
which is acceptable here because this is triggered by low-frequency
admin actions (Add/Modify/Delete), not high-frequency writes.

FAIL-SAFE BY DESIGN
------------------------------------------------------------------
A Sheets sync failure (network issue, auth problem, quota, etc.)
must NEVER block or roll back the underlying CSV write that already
succeeded. Callers should treat sync_dataset() as best-effort: call
it after the CSV write is already durable, check the returned
SyncResult, and surface a warning to the admin UI if it failed --
but the admin action itself should still be reported as successful.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

import pandas as pd


# ------------------------------------------------------------------
# Result object
# ------------------------------------------------------------------

@dataclass
class SyncResult:

    success: bool
    rows_synced: Optional[int]
    error: Optional[str]

    def as_dict(self):

        return {
            "success": self.success,
            "rows_synced": self.rows_synced,
            "error": self.error,
        }


@dataclass
class FetchResult:

    success: bool
    dataframe: Optional[pd.DataFrame]
    rows_fetched: Optional[int]
    error: Optional[str]

    def as_dict(self):

        return {
            "success": self.success,
            "rows_fetched": self.rows_fetched,
            "error": self.error,
        }


@dataclass
class RecordsResult:
    """
    Result of reading a worksheet's raw rows, each paired with its
    live 1-indexed sheet row number (row 1 is the header, so the
    first data row is row 2).

    Row numbers are only meaningful at the instant of this read --
    a row can shift if the sheet changes between calls. Callers
    that need to act on a specific row later (e.g. deletion) must
    re-verify the row's contents at that time rather than trusting
    the row number alone; see SheetsSync.delete_rows().
    """

    success: bool
    records: Optional[List[Dict[str, Any]]]
    error: Optional[str]

    def as_dict(self):

        return {
            "success": self.success,
            "records": self.records,
            "error": self.error,
        }


@dataclass
class DeleteRowsResult:
    """
    Result of a verified row deletion. `deleted_rows` lists the row
    numbers actually removed; `skipped_rows` lists row numbers that
    were requested but not deleted because their live content no
    longer matched what the caller expected (edited or already
    removed since it was last read).
    """

    success: bool
    deleted_rows: Optional[List[int]]
    skipped_rows: Optional[List[int]]
    error: Optional[str]

    def as_dict(self):

        return {
            "success": self.success,
            "deleted_rows": self.deleted_rows,
            "skipped_rows": self.skipped_rows,
            "error": self.error,
        }


# ------------------------------------------------------------------
# Sync client
# ------------------------------------------------------------------

class SheetsSync:
    """
    Wraps a gspread client authorized via a service account, scoped
    to a single spreadsheet + worksheet, and exposes one operation:
    full-overwrite sync of a dataframe into that worksheet.

    Credentials and connection are established once at construction
    (app startup); sync_dataset() can then be called cheaply on every
    admin write.
    """

    def __init__(
        self,
        service_account_file: str,
        spreadsheet_id: str,
        worksheet_name: str,
        service_account_json: Optional[str] = None,
    ):

        self.spreadsheet_id = spreadsheet_id
        self.worksheet_name = worksheet_name

        self._client = None
        self._worksheet = None
        self._init_error = None

        try:

            import gspread
            from google.oauth2.service_account import Credentials

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.file",
            ]

            if service_account_json:


                service_account_info = json.loads(
                    service_account_json
                )

                credentials = Credentials.from_service_account_info(
                    service_account_info,
                    scopes=scopes,
                )

            else:

                if not Path(service_account_file).exists():

                    raise FileNotFoundError(
                        "Service account credentials file not found "
                        f"at '{service_account_file}'. Google Sheets "
                        "sync is disabled until this is provided."
                    )

                credentials = Credentials.from_service_account_file(
                    service_account_file,
                    scopes=scopes,
                )

            self._client = gspread.authorize(credentials)

            spreadsheet = self._client.open_by_key(
                self.spreadsheet_id
            )

            self._worksheet = spreadsheet.worksheet(
                self.worksheet_name
            )

        except Exception as exc:

            # ------------------------------------------------------
            # Do not raise on startup. If credentials/sheet access
            # are misconfigured, the admin tool should still boot
            # and function normally against the local CSV -- Sheets
            # sync simply stays disabled and every sync_dataset()
            # call will report a clear error instead of crashing
            # the app.
            # ------------------------------------------------------

            self._init_error = str(exc)

            print(
                "WARNING: Google Sheets sync could not be "
                f"initialized: {self._init_error}"
            )

    @property
    def is_available(self) -> bool:

        return (
            self._worksheet is not None
            and self._init_error is None
        )

    def sync_dataset(
        self,
        df: pd.DataFrame,
    ) -> SyncResult:
        """
        Fully overwrite the target worksheet with the contents of
        `df`: clears existing content, writes the header row, then
        writes all data rows.

        NaN / None values are written as empty strings, since Sheets
        has no native concept of a missing value and an empty cell
        is the closest equivalent.

        This never raises -- any failure is captured in the
        returned SyncResult so the caller can log/report it without
        the admin write itself failing.
        """

        if not self.is_available:

            return SyncResult(
                success=False,
                rows_synced=None,
                error=(
                    self._init_error
                    or "Google Sheets client is not initialized."
                ),
            )

        try:

            # Replace NaN/None with empty string for Sheets, and
            # make sure everything is a plain string/number gspread
            # can serialize (avoids surprises with numpy dtypes).
            clean_df = df.where(
                pd.notna(df),
                "",
            )

            header = clean_df.columns.tolist()

            values = clean_df.astype(str).values.tolist()

            self._worksheet.clear()

            self._worksheet.update(
                [header] + values,
                value_input_option="RAW",
            )

            return SyncResult(
                success=True,
                rows_synced=len(values),
                error=None,
            )

        except Exception as exc:

            print(
                f"WARNING: Google Sheets sync failed: {exc}"
            )

            return SyncResult(
                success=False,
                rows_synced=None,
                error=str(exc),
            )

    def fetch_dataset(self) -> "FetchResult":
        """
        Reads the target worksheet's current contents back into a
        DataFrame -- the read-side counterpart to sync_dataset().

        This is what makes the Sheet the actual source of truth: a
        row deleted directly in the Sheet (rather than through the
        admin tool) is reflected here, since this reads whatever is
        currently in the worksheet, not whatever sync_dataset() last
        wrote.

        Never raises -- any failure (auth, network, empty sheet,
        malformed data) is captured in the returned FetchResult so
        the caller can fall back to its last-known-good dataframe
        instead of crashing or serving empty data.
        """

        if not self.is_available:

            return FetchResult(
                success=False,
                dataframe=None,
                rows_fetched=None,
                error=(
                    self._init_error
                    or "Google Sheets client is not initialized."
                ),
            )

        try:

            records = self._worksheet.get_all_records()

            if not records:

                return FetchResult(
                    success=False,
                    dataframe=None,
                    rows_fetched=0,
                    error=(
                        "Worksheet is empty or has no data rows "
                        "below the header."
                    ),
                )

            fetched_df = pd.DataFrame(records)

            return FetchResult(
                success=True,
                dataframe=fetched_df,
                rows_fetched=len(fetched_df),
                error=None,
            )

        except Exception as exc:

            print(
                f"WARNING: Google Sheets fetch failed: {exc}"
            )

            return FetchResult(
                success=False,
                dataframe=None,
                rows_fetched=None,
                error=str(exc),
            )

    def append_record(self, record: dict) -> SyncResult:

        if not self.is_available:
            return SyncResult(
                success=False,
                rows_synced=0,
                error=self._init_error
                or "Google Sheets sync is unavailable."
            )

        try:
            # Expected column order for customer trade-in records
            columns = [
                "Timestamp",
                "Customer Name",
                "Phone",
                "Email",
                "Preferred Contact",
                "Device",
                "Sub-device",
                "Model",
                "Model Number",
                "Storage (GB)",
                "Storage Type",
                "Connectivity",
                "Market Value (RM)",
                "Final Trade-In Value (RM)"
            ]

            existing_values = self._worksheet.get_all_values()

            # Create headers if the worksheet has no actual data.
            if not existing_values or not any(
                str(cell).strip()
                for row in existing_values
                for cell in row
            ):
                self._worksheet.append_row(
                    columns,
                    value_input_option="USER_ENTERED"
                )

            # Keep values in the exact same order as the headers.
            row = [
                record.get("Timestamp", ""),
                record.get("Customer Name", ""),
                "'" + str(record.get("Phone", "")),
                record.get("Email", ""),
                record.get("Preferred Contact", ""),
                record.get("Device", ""),
                record.get("Sub-device", ""),
                record.get("Model", ""),
                record.get("Model Number", ""),
                record.get("Storage (GB)", ""),
                record.get("Storage Type", ""),
                record.get("Connectivity", ""),
                record.get("Market Value (RM)", ""),
                record.get("Final Trade-In Value (RM)", "")
            ]

            self._worksheet.append_row(
                row,
                value_input_option="USER_ENTERED"
            )

            return SyncResult(
                success=True,
                rows_synced=1,
                error=None
            )

        except Exception as exc:
            return SyncResult(
                success=False,
                rows_synced=0,
                error=str(exc)
            )

    def list_records(self) -> RecordsResult:
        """
        Reads the worksheet's raw rows and returns each data row
        paired with its live 1-indexed sheet row number.

        Uses raw get_all_values() rather than get_all_records() so
        an empty worksheet (no data rows yet) is a normal, valid
        state -- returned as an empty list, not an error -- which
        matters for Customer Data since a fresh deployment starts
        with none.

        Never raises -- any failure is captured in the returned
        RecordsResult.
        """

        if not self.is_available:

            return RecordsResult(
                success=False,
                records=None,
                error=(
                    self._init_error
                    or "Google Sheets client is not initialized."
                ),
            )

        try:

            values = self._worksheet.get_all_values()

            if not values:

                return RecordsResult(success=True, records=[], error=None)

            header = values[0]
            data_rows = values[1:]

            records = []

            for offset, row in enumerate(data_rows):

                # Sheets trims trailing empty cells, so a short row
                # is padded out to the header's width rather than
                # left ragged -- otherwise fields would silently
                # shift left for rows with empty trailing columns.
                padded = row + [""] * (len(header) - len(row))

                row_number = offset + 2  # +1 for 1-index, +1 for header row

                records.append({
                    "row": row_number,
                    "fields": dict(zip(header, padded[:len(header)])),
                })

            return RecordsResult(success=True, records=records, error=None)

        except Exception as exc:

            print(f"WARNING: Google Sheets list_records failed: {exc}")

            return RecordsResult(success=False, records=None, error=str(exc))

    def delete_rows(self, requested: List[Dict[str, Any]]) -> DeleteRowsResult:
        """
        Deletes specific rows from the worksheet, identified by
        1-indexed row number AND verified against their full field
        content immediately before deletion.

        `requested` is a list of {"row": int, "fields": {header:
        value, ...}} -- normally exactly the items previously
        returned by list_records(). A row number alone is not a
        safe identifier (rows shift if the sheet is edited between
        listing and deleting), and neither is a single field like
        name (two customers can share one). So every requested row
        is re-checked against a fresh read of the sheet, comparing
        every column, and is only deleted if the whole row still
        matches what the caller expects.

        Rows that no longer match (edited, or already removed) are
        reported back as skipped rather than deleted, so the caller
        can tell the admin to refresh and retry instead of silently
        deleting the wrong record.

        Deletions happen bottom-up (highest row number first) so
        that removing one row never shifts the row numbers of the
        others still pending in this same call.

        Never raises -- any failure is captured in the returned
        DeleteRowsResult.
        """

        if not self.is_available:

            return DeleteRowsResult(
                success=False,
                deleted_rows=None,
                skipped_rows=None,
                error=(
                    self._init_error
                    or "Google Sheets client is not initialized."
                ),
            )

        if not requested:

            return DeleteRowsResult(
                success=True,
                deleted_rows=[],
                skipped_rows=[],
                error=None,
            )

        try:

            values = self._worksheet.get_all_values()

            header = values[0] if values else []
            data_rows = values[1:] if values else []

            confirmed_rows = []
            skipped_rows = []

            for item in requested:

                row_number = item.get("row")
                expected_fields = item.get("fields") or {}

                index = (row_number - 2) if row_number is not None else -1

                if row_number is None or index < 0 or index >= len(data_rows):

                    skipped_rows.append(row_number)
                    continue

                current_row = data_rows[index]

                padded = current_row + [""] * (len(header) - len(current_row))

                current_fields = dict(zip(header, padded[:len(header)]))

                matches = all(
                    str(current_fields.get(key, "")) == str(value)
                    for key, value in expected_fields.items()
                )

                if matches:
                    confirmed_rows.append(row_number)
                else:
                    skipped_rows.append(row_number)

            for row_number in sorted(confirmed_rows, reverse=True):

                self._worksheet.delete_rows(row_number)

            return DeleteRowsResult(
                success=True,
                deleted_rows=sorted(confirmed_rows),
                skipped_rows=skipped_rows,
                error=None,
            )

        except Exception as exc:

            print(f"WARNING: Google Sheets delete_rows failed: {exc}")

            return DeleteRowsResult(
                success=False,
                deleted_rows=None,
                skipped_rows=None,
                error=str(exc),
            )