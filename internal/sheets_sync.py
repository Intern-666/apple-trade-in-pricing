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
from typing import Optional
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
                "Model Number",
                "Model",
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
                record.get("Model Number", ""),
                record.get("Model", ""),
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