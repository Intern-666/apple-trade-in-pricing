"""
Automated tests for the Admin Bulk Import Apply + Review Queue workflow.
"""
import sys
import json
import unittest
from unittest.mock import patch
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "internal"))
sys.path.insert(0, str(PROJECT_ROOT / "server"))

from internal import bulk_import as bi
from internal.sheets_sync import SyncResult, DeleteRowsResult, RecordsResult

# Import the FastAPI app directly for integration testing
from server import app as server_app
from fastapi.testclient import TestClient

class TestBulkImportApplyLogic(unittest.TestCase):
    """
    Pure unit tests testing the core application of rules directly 
    via apply_classified_rows without HTTP overhead.
    """
    def setUp(self):
        self.master_columns = bi.CANONICAL_FIELDS
        self.master_df = pd.DataFrame([
            {
                "Provider": "TestProv",
                "Device": "iPhone",
                "Sub-device": "Standard",
                "Standardized Model": "iPhone 10",
                "Model Number": "A1000",
                "Retail Price": 1000.0,
                "Storage (GB)": 64.0,
                "Max. Trade-In Value (RM)": 500.0,
                "Model_Year": 2020,
                "Storage Type": "Unknown",
                "Connectivity": "Unknown",
                "Material": "Unknown",
                "Chipset": "A10",
                "Case Size": "Unknown",
                "Charging Method": "Unknown"
            }
        ], columns=self.master_columns)

    def test_1_new_rows_are_applied(self):
        canonical_df = pd.DataFrame([
            {
                "Provider": "TestProv",
                "Device": "iPhone",
                "Sub-device": "Standard",
                "Standardized Model": "iPhone 11",
                "Model Number": "A1100",
                "Retail Price": 1200.0,
                "Storage (GB)": 64.0,
                "Max. Trade-In Value (RM)": 600.0,
                "Model_Year": 2021,
                "Storage Type": "Unknown",
                "Connectivity": "Unknown",
                "Material": "Unknown",
                "Chipset": "A11",
                "Case Size": "Unknown",
                "Charging Method": "Unknown"
            }
        ], columns=self.master_columns)

        row_results = [{
            "row_index": 0,
            "classification": "new",
            "conflict_type": None,
        }]

        result = bi.apply_classified_rows(canonical_df, row_results, self.master_df)
        outcomes = result["outcomes"]
        new_master = result["master_df"]

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["action"], bi.APPLY_NEW)
        self.assertEqual(len(new_master), 2)
        # Ensure existing row is untouched
        self.assertEqual(new_master.iloc[0]["Standardized Model"], "iPhone 10")
        self.assertEqual(new_master.iloc[1]["Standardized Model"], "iPhone 11")

    def test_2_update_rows_are_applied(self):
        # Same configuration, but Trade-In Value changed from 500.0 to 650.0
        canonical_df = pd.DataFrame([
            {
                "Provider": "TestProv",
                "Device": "iPhone",
                "Sub-device": "Standard",
                "Standardized Model": "iPhone 10",
                "Model Number": "A1000",
                "Retail Price": 1000.0,
                "Storage (GB)": 64.0,
                "Max. Trade-In Value (RM)": 650.0, # Updated
                "Model_Year": 2020,
                "Storage Type": "Unknown",
                "Connectivity": "Unknown",
                "Material": "Unknown",
                "Chipset": "A10",
                "Case Size": "Unknown",
                "Charging Method": "Unknown"
            }
        ], columns=self.master_columns)

        row_results = [{
            "row_index": 0,
            "classification": "update",
            "conflict_type": None,
            "match_index": 0,
            "model_number_update_required": False
        }]

        result = bi.apply_classified_rows(canonical_df, row_results, self.master_df)
        outcomes = result["outcomes"]
        new_master = result["master_df"]

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["action"], bi.APPLY_UPDATE)
        self.assertEqual(len(new_master), 1)
        # Retail price should not be modified, but Trade-in value must be
        self.assertEqual(new_master.iloc[0]["Max. Trade-In Value (RM)"], 650.0)

    def test_3_duplicate_rows_are_skipped(self):
        canonical_df = self.master_df.copy()

        row_results = [{
            "row_index": 0,
            "classification": "conflict",
            "conflict_type": "duplicate",
            "match_index": 0,
        }]

        result = bi.apply_classified_rows(canonical_df, row_results, self.master_df)
        outcomes = result["outcomes"]
        new_master = result["master_df"]

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["action"], bi.APPLY_SKIPPED_DUPLICATE)
        self.assertEqual(len(new_master), 1)
        self.assertTrue(new_master.equals(self.master_df))

    def test_4_needs_review_rows_are_queued(self):
        canonical_df = pd.DataFrame([
            {
                "Provider": "Unknown", # Unknown provider triggers Needs Review
                "Device": "iPhone",
                "Sub-device": "Standard",
                "Standardized Model": "iPhone 10",
                "Model Number": "A1000",
                "Retail Price": 1000.0,
                "Storage (GB)": 64.0,
                "Max. Trade-In Value (RM)": 500.0,
                "Model_Year": 2020,
                "Storage Type": "Unknown",
                "Connectivity": "Unknown",
                "Material": "Unknown",
                "Chipset": "A10",
                "Case Size": "Unknown",
                "Charging Method": "Unknown"
            }
        ], columns=self.master_columns)

        row_results = [{
            "row_index": 0,
            "classification": "conflict",
            "conflict_type": "needs_review",
            "match_index": 0,
        }]

        result = bi.apply_classified_rows(canonical_df, row_results, self.master_df)
        outcomes = result["outcomes"]
        new_master = result["master_df"]

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["action"], bi.APPLY_QUEUED)
        self.assertEqual(len(new_master), 1)
        self.assertTrue(new_master.equals(self.master_df))
        self.assertIn("row_result", outcomes[0])


class TestBulkImportApplyEndpoint(unittest.TestCase):
    """
    Integration tests using FastAPI TestClient to test endpoint 
    routing, queue persistence, and global map synchronization.
    """
    def setUp(self):
        self.client = TestClient(server_app.app)
        
        # SAVE ORIGINAL GLOBAL STATE
        self.original_df = server_app.df.copy() if server_app.df is not None else None
        self.original_model_map = server_app.device_model_map
        self.original_config_map = server_app.device_config_map
        self.original_number_map = server_app.model_number_map

        # Mock out file system and external sheet dependencies 
        self.patcher_refresh = patch("server.app.force_refresh_from_sheets", return_value=True)
        self.patcher_refresh.start()

        self.patcher_write_csv = patch("server.app.write_csv_atomically", return_value=True)
        self.patcher_write_csv.start()

        # Seed the global dataframe isolated from real data
        server_app.df = pd.DataFrame([
            {
                "Provider": "TestProv",
                "Device": "iPhone",
                "Sub-device": "Standard",
                "Standardized Model": "iPhone Update",
                "Model Number": "A0002",
                "Retail Price": 1000.0,
                "Storage (GB)": 128.0,
                "Max. Trade-In Value (RM)": 500.0,
                "Model_Year": 2020,
                "Storage Type": "Unknown",
                "Connectivity": "Unknown",
                "Material": "Unknown",
                "Chipset": "A10",
                "Case Size": "Unknown",
                "Charging Method": "Unknown"
            },
            {
                "Provider": "TestProv",
                "Device": "iPhone",
                "Sub-device": "Standard",
                "Standardized Model": "iPhone Dup",
                "Model Number": "A0003",
                "Retail Price": 1000.0,
                "Storage (GB)": 128.0,
                "Max. Trade-In Value (RM)": 500.0,
                "Model_Year": 2020,
                "Storage Type": "Unknown",
                "Connectivity": "Unknown",
                "Material": "Unknown",
                "Chipset": "A10",
                "Case Size": "Unknown",
                "Charging Method": "Unknown"
            },
        ], columns=bi.CANONICAL_FIELDS)

        # Rebuild lookup maps so the endpoint recognizes the seeded dataframe
        server_app.device_model_map, server_app.device_config_map = server_app.build_device_maps(server_app.df)
        server_app.model_number_map = server_app.build_model_number_map(server_app.df)

    def tearDown(self):
        patch.stopall()
        # RESTORE ORIGINAL GLOBAL STATE
        server_app.df = self.original_df
        server_app.device_model_map = self.original_model_map
        server_app.device_config_map = self.original_config_map
        server_app.model_number_map = self.original_number_map

    @patch("server.app.sheets_sync.sync_dataset")
    @patch("server.app.review_queue_sheets_sync.append_generic_row")
    def test_5_valid_rows_applied_with_needs_review(self, mock_queue_append, mock_sheets_sync):
        mock_queue_append.return_value = SyncResult(success=True, rows_synced=1, error=None)
        mock_sheets_sync.return_value = SyncResult(success=True, rows_synced=1, error=None)

        def make_row(model, price, provider="TestProv"):
            return {
                "effective_record": {
                    "values": {
                        "Provider": provider,
                        "Device": "iPhone",
                        "Sub-device": "Standard",
                        "Standardized Model": model,
                        "Model Number": "A9999" if model == "iPhone New" else ("A0002" if model == "iPhone Update" else "A0003"),
                        "Retail Price": 1000.0,
                        "Storage (GB)": 128.0,
                        "Max. Trade-In Value (RM)": price,
                        "Model_Year": 2020,
                        "Storage Type": "Unknown",
                        "Connectivity": "Unknown",
                        "Material": "Unknown",
                        "Chipset": "A10",
                        "Case Size": "Unknown",
                        "Charging Method": "Unknown"
                    }
                }
            }

        payload = {
            "date_mode": "collection_date",
            "date_value": "2026-01-01",
            "rows": [
                make_row("iPhone New", 800.0), # NEW
                make_row("iPhone Update", 600.0), # UPDATE
                make_row("iPhone Dup", 500.0), # DUPLICATE
                make_row("iPhone Dup", 400.0, provider="Unknown") # NEEDS_REVIEW
            ]
        }

        response = self.client.post("/admin/bulk-import/apply", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["imported_new"], 1)
        self.assertEqual(data["imported_update"], 1)
        self.assertEqual(data["skipped_duplicate"], 1)
        self.assertEqual(data["queued_for_review"], 1)

        # Assert Master was updated (base 2 rows + 1 new row = 3 rows)
        self.assertEqual(len(server_app.df), 3)

        # Assert NEW was added
        self.assertTrue("iPhone New" in server_app.df["Standardized Model"].values)

        # Assert UPDATE was applied accurately
        updated_row = server_app.df[server_app.df["Standardized Model"] == "iPhone Update"].iloc[0]
        self.assertEqual(updated_row["Max. Trade-In Value (RM)"], 600.0)

        # Assert Queue append was called exactly once for the Review row
        mock_queue_append.assert_called_once()
        queued_args = mock_queue_append.call_args[0][0]
        self.assertIn("iPhone Dup", queued_args["Standardized Model"])

    @patch("server.app.sheets_sync.sync_dataset")
    @patch("server.app.review_queue_sheets_sync.append_generic_row")
    def test_6_lookup_maps_are_rebuilt_after_apply(self, mock_queue_append, mock_sheets_sync):
        mock_queue_append.return_value = SyncResult(success=True, rows_synced=1, error=None)
        mock_sheets_sync.return_value = SyncResult(success=True, rows_synced=1, error=None)

        new_model_name = "iPhone 15 Ultra"
        payload = {
            "rows": [
                {
                    "effective_record": {
                        "values": {
                            "Provider": "TestProv",
                            "Device": "iPhone",
                            "Sub-device": "Pro Max",
                            "Standardized Model": new_model_name,
                            "Model Number": "A9999",
                            "Retail Price": 5000.0,
                            "Storage (GB)": 512.0,
                            "Max. Trade-In Value (RM)": 4000.0,
                            "Model_Year": 2024,
                            "Storage Type": "Unknown",
                            "Connectivity": "Unknown",
                            "Material": "Unknown",
                            "Chipset": "A17",
                            "Case Size": "Unknown",
                            "Charging Method": "Unknown"
                        }
                    }
                }
            ]
        }

        # Pre-check: model is strictly not in the map
        self.assertNotIn(new_model_name, server_app.device_model_map.get("iPhone", {}).get("Pro Max", {}))

        response = self.client.post("/admin/bulk-import/apply", json=payload)
        self.assertEqual(response.status_code, 200)

        # Post-check: model correctly rebuilt and inserted into global map state
        self.assertIn(new_model_name, server_app.device_model_map["iPhone"]["Pro Max"])
        self.assertIn(512, server_app.device_model_map["iPhone"]["Pro Max"][new_model_name])
        self.assertIn("A9999", server_app.model_number_map["iPhone"]["Pro Max"])
        self.assertIn(new_model_name, server_app.device_config_map["iPhone"]["Pro Max"])

    @patch("server.app.review_queue_sheets_sync.list_records")
    @patch("server.app.review_queue_sheets_sync.delete_rows")
    @patch("server.app.review_queue_sheets_sync.append_generic_row")
    def test_7_review_queue_recheck_append_fails_prevents_deletion(
        self, mock_append, mock_delete, mock_list
    ):
        # Setup a fake existing queue record
        mock_list.return_value = RecordsResult(
            success=True,
            error=None,
            records=[{
                "row": 2,
                "fields": {
                    "Queue ID": "fake-queue-id",
                    "Device": "iPhone",
                    "Standardized Model": "iPhone Review",
                    "Record JSON": "{}"
                }
            }]
        )

        # Simulate APPEND FAILURE
        mock_append.return_value = SyncResult(success=False, rows_synced=0, error="Google Sheets API Outage")

        payload = {
            "effective_record": {
                "values": {
                    "Provider": "TestProv",
                    "Device": "iPhone",
                    "Sub-device": "Standard",
                    "Standardized Model": "iPhone Review",
                    "Model Number": "A0004",
                    "Retail Price": 1000.0,
                    "Storage (GB)": 128.0,
                    "Max. Trade-In Value (RM)": 500.0,
                    "Model_Year": 2020,
                    "Storage Type": "Unknown",
                    "Connectivity": "Unknown",
                    "Material": "Unknown",
                    "Chipset": "A10",
                    "Case Size": "Unknown",
                    "Charging Method": "Unknown"
                }
            }
        }

        # Issue recheck attempt
        response = self.client.post("/admin/review-queue/fake-queue-id/recheck", json=payload)

        # Assert 503 error surfaces to Admin
        self.assertEqual(response.status_code, 503)

        # Critical: Assert delete was NEVER called to prevent data loss
        mock_delete.assert_not_called()

        # --- Test SUCCESS PATH ---
        mock_append.return_value = SyncResult(success=True, rows_synced=1, error=None)
        mock_delete.return_value = DeleteRowsResult(success=True, deleted_rows=[2], skipped_rows=[], error=None)

        # Fake the find_record lookup so it successfully returns the newly appended row
        def side_effect_list_records():
            return RecordsResult(
                success=True,
                error=None,
                records=[{
                    "row": 3,
                    "fields": {
                        "Queue ID": "fake-queue-id",
                        "Device": "iPhone",
                        "Standardized Model": "iPhone Review",
                        "Record JSON": "{}"
                    }
                }]
            )
        mock_list.side_effect = [
            mock_list.return_value, # First call for finding the old record
            side_effect_list_records() # Second call inside _find_review_queue_sheet_record after append completes
        ]

        response = self.client.post("/admin/review-queue/fake-queue-id/recheck", json=payload)

        # Assert full success execution
        self.assertEqual(response.status_code, 200)
        mock_append.assert_called()
        mock_delete.assert_called_once()

if __name__ == '__main__':
    unittest.main()