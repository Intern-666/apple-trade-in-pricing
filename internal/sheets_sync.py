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

                credentials_info = json.loads(
                    service_account_json
                )

                credentials = Credentials.from_service_account_info(
                    credentials_info,
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
