"""
Regression tests for internal/bulk_import.py.

These tests exercise the CURRENT implementation and CURRENT result
structure directly:

    classification: "new" | "update" | "conflict"
    conflict_type:  None | "duplicate" | "needs_review" | "invalid_data"
                    (set only when classification == "conflict")

    reasons, warnings, match, match_index, multi_match,
    unknown_fields, different_fields, model_number_flag,
    model_numbers_to_append, model_number_update_required

`multi_match` is internal metadata only and is never itself a
classification.

The old test file imported a nonexistent `server.app` package and
asserted an obsolete API (`column_mapping`, `display_classification`,
`master_classification`, `existing`/`multi_match` as classifications,
etc.). None of that infrastructure exists in the current project, so
this file replaces it rather than patching around it.

Run with:
    python3 -m unittest internal.test_bulk_import -v
or directly:
    python3 internal/test_bulk_import.py
"""

import sys
import unittest
from unittest import mock
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "internal"))

import pandas as pd

try:
    from internal import bulk_import as bi
except ImportError:
    import bulk_import as bi

DATA_DIR = PROJECT_ROOT / "data"


def _master_row(
    provider="CompAsia",
    device="iPhone",
    sub_device="Standard",
    model_number="A2111, A2221, A2223",
    model="iPhone 11",
    retail_price=3599.0,
    storage=128.0,
    storage_type="Unknown",
    connectivity="Unknown",
    material=None,
    trade_in=390.0,
    model_year=2019.0,
    case_size=None,
    charging_method=None,
    collection_date=None,
):
    """Build one Master-schema row dict (CANONICAL_FIELDS order)."""
    return {
        "Provider": provider,
        "Device": device,
        "Sub-device": sub_device,
        "Model Number": model_number,
        "Standardized Model": model,
        "Retail Price": retail_price,
        "Storage (GB)": storage,
        "Storage Type": storage_type,
        "Connectivity": connectivity,
        "Material": material,
        "Max. Trade-In Value (RM)": trade_in,
        "Model_Year": model_year,
        "Case Size": case_size,
        "Charging Method": charging_method,
        "Collection Date": collection_date,
    }


def _df(rows):
    return pd.DataFrame(rows, columns=bi.CANONICAL_FIELDS)


def _assert_valid_result_shape(testcase, row):
    """
    Shared invariant: classification is always one of the three
    top-level values, conflict_type is populated if and only if
    classification == "conflict", and multi_match never leaks out
    as a classification or conflict_type.
    """
    testcase.assertIn(
        row["classification"], {"new", "update", "conflict"}
    )
    testcase.assertNotIn(
        row["classification"], {"multi_match"}
    )

    if row["classification"] == "conflict":
        testcase.assertIn(
            row["conflict_type"],
            {"duplicate", "needs_review", "invalid_data"},
        )
    else:
        testcase.assertIsNone(row["conflict_type"])


class TestFixtureFiles(unittest.TestCase):
    """Sanity checks that the fixtures exist and load as expected."""

    def test_fixture_files_exist(self):
        self.assertTrue(
            (DATA_DIR / "test_incoming.csv").exists(),
            "data/test_incoming.csv is required for these tests.",
        )
        self.assertTrue(
            (DATA_DIR / "master_msrp.csv").exists(),
            "data/master_msrp.csv is required for these tests.",
        )

    def test_master_has_expected_row_count(self):
        master_df = pd.read_csv(DATA_DIR / "master_msrp.csv")
        # The real Master dataset is documented as ~4,190 rows.
        self.assertGreater(len(master_df), 4000)

    def test_incoming_fixture_has_eighteen_rows(self):
        incoming_df = pd.read_csv(DATA_DIR / "test_incoming.csv")
        self.assertEqual(len(incoming_df), 18)


class TestAnalyzeUploadAgainstRealFixtures(unittest.TestCase):
    """
    Exercises analyze_upload() against the real 18-row fixture and
    the real ~4,190-row Master file, using the CURRENT result
    structure.

    The 18-row fixture represents six conceptual cases repeated
    across iPhone / iPad / Mac rows:

        rows 0-2   fabricated models with no Master counterpart
                   -> new
        rows 3-5   existing configuration for that same Provider
                   -> row 3 is an exact duplicate, row 4 has only
                   its Trade-In Value changed (-> update), row 5
                   is a genuinely new Provider (-> new); see below
        rows 6-8   exact Provider + configuration match
                   -> conflict / duplicate
        rows 9-11  Unknown Provider
                   -> conflict / needs_review
        rows 12-14 Unknown relevant value (storage/connectivity/
                   storage type)
                   -> conflict / needs_review
        rows 15-17 malformed supplied value
                   -> conflict / invalid_data
    """

    @classmethod
    def setUpClass(cls):
        cls.master_df = pd.read_csv(DATA_DIR / "master_msrp.csv")
        cls.incoming_df = pd.read_csv(DATA_DIR / "test_incoming.csv")
        cls.result = bi.analyze_upload(
            raw_df=cls.incoming_df,
            master_df=cls.master_df,
            clean_fn=None,
        )
        cls.rows = cls.result["rows"]

    def test_result_shape(self):
        self.assertEqual(len(self.rows), len(self.incoming_df))
        for row in self.rows:
            _assert_valid_result_shape(self, row)

    def test_current_api_keys_present(self):
        # Guards against silently regressing to an older API shape.
        self.assertIn("mapping", self.result)
        self.assertIn("unmapped_columns", self.result)
        self.assertIn("missing_required", self.result)
        for row in self.rows:
            for key in (
                "classification",
                "conflict_type",
                "reasons",
                "match_index",
                "multi_match",
                "unknown_fields",
                "different_fields",
                "model_number_flag",
                "model_numbers_to_append",
                "model_number_update_required",
            ):
                self.assertIn(key, row)

    def test_fabricated_models_are_new(self):
        for index in (0, 1, 2):
            with self.subTest(index=index):
                self.assertEqual(
                    self.rows[index]["classification"], "new"
                )
                self.assertIsNone(self.rows[index]["conflict_type"])

    def test_row_3_machines_exact_match_is_duplicate(self):
        # Row 3 (Machines iPhone 11) has an EXACT Machines
        # counterpart in Master (this is the confirmed
        # Provider-selection bug fixture) so it must resolve as an
        # exact duplicate against that row, not "new" against an
        # unrelated Provider's row.
        self.assertEqual(self.rows[3]["classification"], "conflict")
        self.assertEqual(self.rows[3]["conflict_type"], "duplicate")

    def test_row_4_switch_trade_in_only_change_is_update(self):
        # Row 4 (Switch iPad 10) matches Switch's OWN existing
        # Master row once Provider-aware candidate selection picks
        # the correct reference: same Provider, same Apple
        # configuration, same Retail Price, only Trade-In Value
        # differs -> update, not new.
        row = self.rows[4]
        self.assertEqual(row["classification"], "update")
        self.assertIsNone(row["conflict_type"])
        self.assertEqual(
            row["different_fields"], ["Max. Trade-In Value (RM)"]
        )

    def test_row_5_mac_city_new_provider_is_new(self):
        # Row 5 (Mac City iMac) has no prior Mac City counterpart
        # at all for this configuration -> a genuinely new Provider,
        # regardless of pricing.
        row = self.rows[5]
        self.assertEqual(row["classification"], "new")
        self.assertIsNone(row["conflict_type"])
        self.assertIn("Provider", row["different_fields"])

    def test_exact_matches_are_duplicate(self):
        for index in (6, 7, 8):
            with self.subTest(index=index):
                self.assertEqual(
                    self.rows[index]["classification"], "conflict"
                )
                self.assertEqual(
                    self.rows[index]["conflict_type"], "duplicate"
                )
                self.assertEqual(self.rows[index]["unknown_fields"], [])
                self.assertEqual(self.rows[index]["different_fields"], [])

    def test_unknown_provider_is_needs_review(self):
        for index in (9, 10, 11):
            with self.subTest(index=index):
                self.assertEqual(
                    self.rows[index]["classification"], "conflict"
                )
                self.assertEqual(
                    self.rows[index]["conflict_type"], "needs_review"
                )
                self.assertIn("Provider", self.rows[index]["unknown_fields"])

    def test_unknown_relevant_value_is_needs_review(self):
        for index in (12, 13, 14):
            with self.subTest(index=index):
                self.assertEqual(
                    self.rows[index]["classification"], "conflict"
                )
                self.assertEqual(
                    self.rows[index]["conflict_type"], "needs_review"
                )
                self.assertTrue(self.rows[index]["unknown_fields"])

    def test_malformed_value_is_invalid_data(self):
        for index in (15, 16, 17):
            with self.subTest(index=index):
                self.assertEqual(
                    self.rows[index]["classification"], "conflict"
                )
                self.assertEqual(
                    self.rows[index]["conflict_type"], "invalid_data"
                )

    def test_within_file_duplicate_flag_well_formed(self):
        for row in self.rows:
            self.assertIn("within_file_duplicate", row)
            self.assertIsInstance(row["within_file_duplicate"], bool)


class TestProviderAwareCandidateSelection(unittest.TestCase):
    """
    Regression test for the confirmed production bug:

        choose_reference_candidate() must prefer a same-Provider
        candidate over Master file order.

    Reproduces the reported scenario directly: a CompAsia row with
    the exact same configuration/specs sits EARLIER in the Master
    file than the correct Machines row. Before the fix, the
    incoming Machines row was incorrectly matched to the CompAsia
    row (wrong Provider) and misclassified as "new" instead of
    "conflict" / "duplicate".
    """

    def setUp(self):
        # CompAsia (index 0) intentionally precedes Machines
        # (index 1) in Master file order, and shares the exact same
        # non-Provider specs, to reproduce the reported bug.
        self.master_df = _df(
            [
                _master_row(provider="CompAsia"),
                _master_row(provider="Machines"),
            ]
        )

    def test_same_provider_candidate_is_preferred(self):
        incoming_df = _df([_master_row(provider="Machines")])

        result = bi.analyze_upload(
            raw_df=incoming_df,
            master_df=self.master_df,
            clean_fn=None,
        )

        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "duplicate")
        self.assertEqual(row["match_index"], 1)  # the Machines row
        self.assertEqual(row["different_fields"], [])
        self.assertEqual(row["unknown_fields"], [])

    def test_fallback_to_file_order_when_no_provider_match(self):
        # A brand-new Provider with no counterpart in Master at all
        # should still fall back to the first suitable candidate
        # (existing behavior) rather than failing to match.
        incoming_df = _df([_master_row(provider="NewProvider")])

        result = bi.analyze_upload(
            raw_df=incoming_df,
            master_df=self.master_df,
            clean_fn=None,
        )

        row = result["rows"][0]

        self.assertEqual(row["match_index"], 0)  # first candidate
        self.assertEqual(row["classification"], "new")
        self.assertIsNone(row["conflict_type"])
        self.assertIn("Provider", row["different_fields"])

    def test_choose_reference_candidate_prefers_provider_directly(self):
        # Unit-level check, independent of the full pipeline: given
        # multiple candidates, the same-Provider one must win even
        # when it is not first in the list.
        candidates = [
            {"index": 0, "norm": {"provider": "CompAsia"}, "confidence": 0.9},
            {"index": 1, "norm": {"provider": "Machines"}, "confidence": 0.5},
        ]

        chosen = bi.choose_reference_candidate(candidates, "Machines")
        self.assertEqual(chosen["index"], 1)

    def test_choose_reference_candidate_prefers_best_score_within_provider(self):
        # When multiple same-Provider candidates exist, the highest
        # -confidence one should be selected, not simply the first
        # same-Provider row encountered in file order.
        candidates = [
            {"index": 0, "norm": {"provider": "Machines"}, "confidence": 0.4},
            {"index": 1, "norm": {"provider": "Machines"}, "confidence": 0.95},
        ]

        chosen = bi.choose_reference_candidate(candidates, "Machines")
        self.assertEqual(chosen["index"], 1)

    def test_choose_reference_candidate_no_candidates(self):
        self.assertIsNone(bi.choose_reference_candidate([], "Machines"))

    def test_choose_reference_candidate_unknown_incoming_provider(self):
        candidates = [
            {"index": 0, "norm": {"provider": "CompAsia"}, "confidence": 0.9},
        ]
        # An Unknown incoming Provider can't be "preferred" against;
        # fall back to file order.
        chosen = bi.choose_reference_candidate(candidates, "Unknown")
        self.assertEqual(chosen["index"], 0)


class TestModelNumberAppend(unittest.TestCase):
    """
    Model Number is special: a new Model Number on an otherwise
    identical row should be appended to the existing Master row
    (classification "update"), not treated as a brand-new Master
    row and not treated as an Admin conflict. An overlapping Model
    Number on an otherwise identical row is simply an ordinary
    conflict / duplicate.
    """

    def test_new_model_number_is_flagged_for_append(self):
        master_df = _df(
            [_master_row(model_number="A2111, A2221, A2223")]
        )
        incoming_df = _df([_master_row(model_number="A9999")])

        result = bi.analyze_upload(
            raw_df=incoming_df,
            master_df=master_df,
            clean_fn=None,
        )

        row = result["rows"][0]

        self.assertEqual(row["classification"], "update")
        self.assertIsNone(row["conflict_type"])
        self.assertEqual(row["model_number_flag"], "append")
        self.assertTrue(row["model_number_update_required"])
        self.assertEqual(row["model_numbers_to_append"], ["A9999"])

    def test_overlapping_model_number_is_a_plain_match(self):
        # Incoming model number "A2111" is a subset of the Master
        # row's existing "A2111, A2221, A2223" -> already
        # represented, ordinary conflict/duplicate, nothing to
        # append.
        master_df = _df(
            [_master_row(model_number="A2111, A2221, A2223")]
        )
        incoming_df = _df([_master_row(model_number="A2111")])

        result = bi.analyze_upload(
            raw_df=incoming_df,
            master_df=master_df,
            clean_fn=None,
        )

        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "duplicate")
        self.assertEqual(row["model_number_flag"], "match")

    def test_append_uses_provider_aware_candidate(self):
        # The append target must be the correct Provider's Master
        # row, not just the first Master row that happens to match
        # the configuration.
        master_df = _df(
            [
                _master_row(
                    provider="CompAsia",
                    model_number="A2111, A2221, A2223",
                ),
                _master_row(
                    provider="Machines",
                    model_number="A2111, A2221, A2223",
                ),
            ]
        )
        incoming_df = _df(
            [_master_row(provider="Machines", model_number="A9999")]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df,
            master_df=master_df,
            clean_fn=None,
        )

        row = result["rows"][0]

        self.assertEqual(row["classification"], "update")
        self.assertIsNone(row["conflict_type"])
        self.assertEqual(row["model_number_flag"], "append")
        self.assertEqual(row["match_index"], 1)  # the Machines row


class TestUpdateAndConflictClassification(unittest.TestCase):
    """
    Same-Provider, same-Apple-configuration Retail Price / Trade-In
    Value handling.

        1. No relevant changes                        -> conflict / duplicate
        2. Trade-In Value changed (only)               -> update
        3. Other mutable value(s) changed, including
           Trade-In                                    -> update
        4. Retail Price changed                        -> conflict / needs_review
        5. Retail Price AND Trade-In changed            -> conflict / needs_review
           (Retail Price takes precedence)
        6. Different valid Provider, regardless of
           pricing                                     -> new
        7. Unknown Provider / Unknown relevant value    -> conflict / needs_review
        8. Malformed supplied value                     -> conflict / invalid_data
    """

    # ---- Rule 1: exact duplicate --------------------------------

    def test_exact_duplicate_is_conflict_duplicate(self):
        master_df = _df([_master_row()])
        incoming_df = _df([_master_row()])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "duplicate")
        self.assertEqual(row["different_fields"], [])
        self.assertEqual(row["unknown_fields"], [])

    # ---- Rule 2: Trade-In-only change ---------------------------

    def test_trade_in_only_change_is_update(self):
        master_df = _df([_master_row(trade_in=390.0)])
        incoming_df = _df([_master_row(trade_in=420.0)])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "update")
        self.assertIsNone(row["conflict_type"])
        self.assertEqual(
            row["different_fields"], ["Max. Trade-In Value (RM)"]
        )

    # ---- Rule 3: multiple mutable values, including Trade-In ----

    def test_multiple_mutable_values_including_trade_in_is_update(self):
        # The current canonical schema only exposes Retail Price
        # and Trade-In Value as fields that can genuinely differ
        # once Provider and the full Apple configuration already
        # match (every other comparable field uses the identical
        # normalized equality check in both the coarse
        # configuration match and the full row comparison, so they
        # can never disagree between the two stages; Chipset is a
        # documented "ghost field" that never participates in the
        # comparison at all). To exercise the general "any set of
        # non-Retail-Price mutable fields -> update" rule (as
        # opposed to the Trade-In-specific case above), this test
        # calls classify_incoming_row() directly and patches
        # compare_master_relevant_fields() to report a second,
        # hypothetical mutable field changing alongside Trade-In.
        # This is a unit-level test of the branching logic, not a
        # claim that "Chipset" differences are reachable today.
        master_df = _df([_master_row()])
        incoming_df = _df([_master_row()])

        (
            master_norm_by_index,
            master_device_by_index,
            master_provider_by_index,
        ) = bi.build_master_indexes(master_df)

        incoming_norm = bi.normalize_row_for_matching(
            incoming_df.iloc[0]
        )

        fake_full_comparison = {
            "exact": False,
            "unknown_fields": [],
            "different_fields": ["Max. Trade-In Value (RM)", "Storage Type"],
        }

        with mock.patch.object(
            bi,
            "compare_master_relevant_fields",
            return_value=fake_full_comparison,
        ):
            result = bi.classify_incoming_row(
                incoming_row_canonical=incoming_df.iloc[0],
                incoming_norm=incoming_norm,
                master_df=master_df,
                master_norm_by_index=master_norm_by_index,
                master_device_by_index=master_device_by_index,
                master_provider_by_index=master_provider_by_index,
            )

        self.assertEqual(result["classification"], "update")
        self.assertIsNone(result["conflict_type"])

    # ---- Rule 4: Retail Price change -----------------------------

    def test_retail_price_change_is_conflict_needs_review(self):
        master_df = _df([_master_row(retail_price=3599.0)])
        incoming_df = _df([_master_row(retail_price=3699.0)])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "needs_review")
        self.assertIn("Retail Price", row["different_fields"])

    # ---- Rule 5: Retail Price AND Trade-In change ----------------

    def test_retail_price_and_trade_in_change_is_conflict_needs_review(self):
        master_df = _df(
            [_master_row(retail_price=3599.0, trade_in=390.0)]
        )
        incoming_df = _df(
            [_master_row(retail_price=3699.0, trade_in=420.0)]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        # Retail Price takes precedence: still needs_review, not
        # update, even though Trade-In changed too.
        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "needs_review")
        self.assertIn("Retail Price", row["different_fields"])
        self.assertIn(
            "Max. Trade-In Value (RM)", row["different_fields"]
        )

    # ---- Rule 6: different Provider, regardless of pricing -------

    def test_different_provider_with_different_pricing_is_new(self):
        master_df = _df(
            [
                _master_row(
                    provider="CompAsia",
                    retail_price=3599.0,
                    trade_in=390.0,
                )
            ]
        )
        incoming_df = _df(
            [
                _master_row(
                    provider="Machines",
                    retail_price=3699.0,
                    trade_in=420.0,
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "new")
        self.assertIsNone(row["conflict_type"])
        self.assertIn("Provider", row["different_fields"])

    # ---- Rule 7: Unknown Provider ---------------------------------

    def test_unknown_provider_is_conflict_needs_review(self):
        master_df = _df([_master_row(provider="CompAsia")])
        incoming_df = _df([_master_row(provider=None)])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "needs_review")
        self.assertIn("Provider", row["unknown_fields"])

    # ---- Rule 7: Unknown relevant value ----------------------------

    def test_unknown_relevant_value_is_conflict_needs_review(self):
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="iMac",
                    model="iMac 27-inch",
                    storage_type="SSD",
                )
            ]
        )
        incoming_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="iMac",
                    model="iMac 27-inch",
                    storage_type=None,
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "needs_review")
        self.assertTrue(row["unknown_fields"])

    # ---- Rule 8: malformed Retail Price ----------------------------

    def test_malformed_retail_price_is_conflict_invalid_data(self):
        master_df = _df([_master_row()])
        incoming_df = _df([_master_row(retail_price="RM 3599")])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "invalid_data")

    # ---- Model Number overlap/append, revisited under this rule set

    def test_model_number_overlap_is_conflict_duplicate(self):
        master_df = _df(
            [_master_row(model_number="A2111, A2221, A2223")]
        )
        incoming_df = _df([_master_row(model_number="A2221")])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "duplicate")
        self.assertEqual(row["model_number_flag"], "match")

    def test_model_number_append_is_update(self):
        master_df = _df(
            [_master_row(model_number="A2111, A2221, A2223")]
        )
        incoming_df = _df([_master_row(model_number="A9999")])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "update")
        self.assertIsNone(row["conflict_type"])
        self.assertEqual(row["model_number_flag"], "append")


class TestModelIdentityConflict(unittest.TestCase):
    """
    Latent issue #1: model_identity_conflict() did not cross-check
    purely numeric generation tokens (e.g. "99") against alphanumeric
    model designation tokens (e.g. "16e") on the other side, letting
    fabricated generations slip past the coarse candidate threshold.
    """

    def test_disjoint_numeric_generation_vs_designation_conflicts(self):
        self.assertTrue(
            bi.model_identity_conflict("iPhone 99", "iPhone 16e")
        )

    def test_bare_generation_vs_matching_designation_base_not_forced_conflict(self):
        # "iPhone 16" vs "iPhone 16e" share the same numeric base;
        # per the established guidance this is deliberately NOT
        # forced into a hard identity conflict at this coarse layer
        # (that would incorrectly assume numeric vs alphanumeric
        # always means a different model). Finer-grained exact
        # matching is left to the row-level comparison.
        self.assertFalse(
            bi.model_identity_conflict("iPhone 16", "iPhone 16e")
        )

    def test_matching_designation_no_conflict(self):
        self.assertFalse(
            bi.model_identity_conflict("iPhone 16e", "iPhone 16e")
        )

    def test_disjoint_designations_conflict(self):
        self.assertTrue(
            bi.model_identity_conflict("iPhone 16e", "iPhone 17e")
        )

    def test_different_product_line_conflicts(self):
        self.assertTrue(
            bi.model_identity_conflict("iPhone 15", "iPad 15")
        )

    def test_same_model_no_conflict(self):
        self.assertFalse(
            bi.model_identity_conflict("iPhone 11", "iPhone 11")
        )


class TestBlankModelNumberEquality(unittest.TestCase):
    """
    Latent issue #2: _values_equal() returns None (Unknown) for
    Model Number when either side's normalized Model Number set is
    empty, rather than True.

    Decision: this is correct and intentionally left unchanged.
    Every other optional field already follows exactly this pattern
    (both sides blank -> Unknown -> needs_review, not an automatic
    match), because a blank value never establishes that two rows
    are actually the same - it just means the information wasn't
    supplied. Special-casing Model Number to treat "both blank" as
    equal would make it inconsistent with how every other field
    (Storage Type, Connectivity, Material, ...) is handled, and
    would silently let two otherwise-unverified rows collapse into
    a duplicate.
    """

    def test_blank_model_number_both_sides_is_unknown_not_equal(self):
        master_df = _df([_master_row(model_number=None)])
        incoming_df = _df([_master_row(model_number=None)])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "needs_review")
        self.assertIn("Model Number", row["unknown_fields"])

    def test_this_matches_behavior_of_other_optional_fields(self):
        # Same shape of test, but for Storage Type (applicable to
        # Mac, unlike iPhone), to show Model Number is not being
        # singled out - both-blank is Unknown for every optional
        # field that actually applies to the device.
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="iMac",
                    model="iMac 27-inch",
                    storage_type=None,
                )
            ]
        )
        incoming_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="iMac",
                    model="iMac 27-inch",
                    storage_type=None,
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "needs_review")
        self.assertIn("Storage Type", row["unknown_fields"])


class TestWithinFileDuplicates(unittest.TestCase):
    """Second/subsequent identical rows in the same upload."""

    def test_second_identical_row_is_flagged_duplicate(self):
        master_df = _df([])  # empty Master -> both rows would be "new"
        incoming_df = _df(
            [
                _master_row(provider="CompAsia"),
                _master_row(provider="CompAsia"),
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        rows = result["rows"]

        # Both rows belong to the same within-file duplicate group,
        # but only the second (non-first) occurrence is actually
        # classified as a duplicate and withheld from Master.
        self.assertEqual(rows[0]["classification"], "new")
        self.assertIsNone(rows[0]["conflict_type"])
        self.assertTrue(rows[0]["within_file_duplicate"])
        self.assertIn(0, rows[0]["within_file_duplicate_group"])

        self.assertEqual(rows[1]["classification"], "conflict")
        self.assertEqual(rows[1]["conflict_type"], "duplicate")
        self.assertTrue(rows[1]["within_file_duplicate"])


class TestSubDeviceInference(unittest.TestCase):
    """
    Regression coverage for a confirmed production bug: Sub-device
    inference was gated entirely on Device being blank, so any row
    that already supplied an explicit Device (the overwhelming
    majority of real uploads) never even attempted Sub-device
    inference -- regardless of whether Sub-device itself was blank.
    This made Sub-device come back "Unknown" for essentially every
    real-world row, forcing conflict/needs_review far more than the
    data actually warranted.

    A second, previously-unreachable bug was exposed by fixing the
    first one: build_row_search_text() called `.values()` on its
    argument, which is a dict method, not a pandas Series one (the
    real call site always passes a Series). This line was never
    executed before because the caller only reached it when Device
    was blank, which never happened for rows that already supplied
    Device.
    """

    def _master_df_with_iphone_variants(self):
        return _df(
            [
                _master_row(
                    sub_device="Standard", model="iPhone 13"
                ),
                _master_row(
                    sub_device="Mini", model="iPhone 13 Mini"
                ),
            ]
        )

    def test_sub_device_is_inferred_even_when_device_is_explicit(self):
        # Device is already explicit ("iPhone"); Sub-device is
        # blank. Before the fix, the entire inference block was
        # skipped because it only ran when Device was blank.
        master_df = self._master_df_with_iphone_variants()
        incoming_df = _df(
            [
                _master_row(
                    sub_device=None, model="iPhone 13 Mini"
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertNotIn("Sub-device", row["unknown_fields"])
        # Sub-device is inferred as the comparison-safe "mini", then
        # display-canonicalized to Master's own casing ("Mini") by
        # canonicalize_incoming_row_display_values().
        self.assertEqual(
            result["canonical_df"].iloc[0]["Sub-device"], "Mini"
        )

    def test_explicit_device_is_never_overwritten_by_inference(self):
        # Even though Sub-device inference now also runs for rows
        # with an explicit Device, that Device value must never be
        # silently replaced.
        master_df = self._master_df_with_iphone_variants()
        incoming_df = _df(
            [
                _master_row(
                    device="iPhone",
                    sub_device=None,
                    model="iPhone 13 Mini",
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )

        self.assertEqual(
            result["canonical_df"].iloc[0]["Device"], "iPhone"
        )

    def test_specificity_rule_resolves_word_subset_ambiguity(self):
        # "Pro" is a substring token of "Pro Max" too, so a model
        # name containing "Pro Max" matches both "Pro" and "Pro Max"
        # in the vocabulary. Since "Pro"'s words ({"pro"}) are a
        # strict subset of "Pro Max"'s words ({"pro", "max"}), the
        # Group B specificity fix drops "Pro" and keeps "Pro Max" as
        # the unique most-specific match, so Sub-device now resolves
        # correctly instead of staying Unknown. (Previously this
        # test asserted the pre-fix conservative fallback; that
        # premise no longer holds now that the ambiguity is
        # resolvable.)
        master_df = _df(
            [
                _master_row(
                    sub_device="Pro", model="iPhone 13 Pro"
                ),
                _master_row(
                    sub_device="Pro Max",
                    model="iPhone 13 Pro Max",
                ),
            ]
        )
        incoming_df = _df(
            [
                _master_row(
                    sub_device=None, model="iPhone 13 Pro Max"
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertNotIn("Sub-device", row["unknown_fields"])
        # Sub-device is inferred as the comparison-safe "pro max",
        # then display-canonicalized to Master's own casing
        # ("Pro Max") by canonicalize_incoming_row_display_values().
        self.assertEqual(
            result["canonical_df"].iloc[0]["Sub-device"], "Pro Max"
        )

    def test_genuine_tie_between_equally_specific_candidates_stays_unknown(
        self,
    ):
        # Two candidates of equal specificity, neither a subset of
        # the other ("special edition" vs "deluxe edition" -- same
        # word count, no containment relationship) must still fall
        # back to Unknown. The specificity fix only ever resolves a
        # candidate that strictly subsumes another; it must never
        # guess between two unrelated, equally-specific candidates.
        master_df = _df(
            [
                _master_row(
                    sub_device="Special Edition",
                    model="iPhone Special Edition",
                ),
                _master_row(
                    sub_device="Deluxe Edition",
                    model="iPhone Deluxe Edition",
                ),
            ]
        )
        incoming_df = _df(
            [
                _master_row(
                    sub_device=None,
                    model="iPhone Special Edition Deluxe Edition",
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertIn("Sub-device", row["unknown_fields"])
        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "needs_review")

    def test_build_row_search_text_accepts_a_pandas_series(self):
        # Direct regression test for the `.values()` crash: the
        # real call site always passes a pandas Series, never a
        # plain dict.
        row = pd.Series(
            {
                "Device": "iPhone",
                "Standardized Model": "iPhone 13 Mini",
                "Sub-device": None,
            }
        )
        text = bi.build_row_search_text(row)
        self.assertIn("mini", text)


class TestModelNumberSubDeviceInference(unittest.TestCase):
    """
    Coverage for the Model Number -> Sub-device inference that
    supplements the generic keyword-based Sub-device inference for
    iPhone rows with no qualifier word at all (e.g. plain "iPhone
    17", with nothing for the keyword matcher to latch onto).

    This must only ever trust Master Model Number -> Sub-device
    relationships that look like genuine Apple regulatory model
    numbers, never overwrite an explicit incoming Sub-device, and
    back off to Unknown on any ambiguity -- it must never guess
    "no qualifier means Standard".
    """

    def test_legitimate_iphone_model_number_infers_sub_device(self):
        # Master default fixture is exactly this shape: iPhone,
        # Standard, "A2111, A2221, A2223", "iPhone 11".
        master_df = _df([_master_row()])
        incoming_df = _df(
            [
                _master_row(
                    sub_device=None,
                    model_number="A2111 / A2223 / A2221",
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertNotIn("Sub-device", row["unknown_fields"])
        self.assertEqual(
            result["canonical_df"].iloc[0]["Sub-device"], "Standard"
        )

    def test_garbage_model_number_is_not_trusted(self):
        # "Test123" is one of Master's known fictional/test rows.
        # It must never be trusted as an inference source, even
        # though the incoming row matches it exactly.
        master_df = _df(
            [
                _master_row(
                    model_number="Test123",
                    model="testing iphone",
                )
            ]
        )
        incoming_df = _df(
            [
                _master_row(
                    sub_device=None,
                    model_number="Test123",
                    model="testing iphone",
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertIn("Sub-device", row["unknown_fields"])
        self.assertTrue(
            bi._is_blank(
                result["canonical_df"].iloc[0]["Sub-device"]
            )
        )

    def test_ambiguous_model_number_mapping_stays_unknown(self):
        # The same Model Number maps to two different Sub-devices
        # across Master rows -- the mapping must be treated as
        # untrustworthy rather than guessed.
        master_df = _df(
            [
                _master_row(
                    model_number="A1234",
                    sub_device="Standard",
                    model="iPhone Ambiguous",
                ),
                _master_row(
                    model_number="A1234",
                    sub_device="Pro",
                    model="iPhone Ambiguous",
                ),
            ]
        )
        incoming_df = _df(
            [
                _master_row(
                    sub_device=None,
                    model_number="A1234",
                    model="iPhone Ambiguous",
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )

        self.assertTrue(
            bi._is_blank(
                result["canonical_df"].iloc[0]["Sub-device"]
            )
        )

    def test_explicit_incoming_sub_device_is_never_overwritten(self):
        # Master's trusted mapping for this Model Number says
        # "Standard", but the incoming row explicitly states "Pro".
        # The explicit value must win.
        master_df = _df([_master_row()])
        incoming_df = _df(
            [
                _master_row(
                    sub_device="Pro",
                    model_number="A2111 / A2223 / A2221",
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )

        self.assertEqual(
            result["canonical_df"].iloc[0]["Sub-device"], "Pro"
        )


class TestIncomingRowDisplayCanonicalization(unittest.TestCase):
    """
    Regression tests for canonicalize_incoming_row_display_values()
    / build_canonical_display_maps().

    Root cause: matching normalization (normalize_text,
    normalize_row_for_matching, normalize_device_value,
    extract_model_connectivity) already correctly resolved an
    incoming row's *meaning* for comparison, but that resolved,
    properly-cased form was never written back to canonical_df --
    only ever a transient lowercase comparison value. So the row an
    Admin actually previewed/imported could still show raw values
    like Device "Pad", Sub-device "ipad", or a Standardized Model
    still carrying an embedded connectivity suffix ("iPad 7 Wi-Fi +
    Cellular") with Connectivity left at "Unknown".
    """

    def _master_df_with_ipad7(self):
        return _df(
            [
                _master_row(
                    provider="CompAsia",
                    device="iPad",
                    sub_device="iPad",
                    model_number="A2197, A2198, A2200",
                    model="iPad 7",
                    retail_price=1849.0,
                    storage=128.0,
                    connectivity="Wi-Fi",
                    trade_in=225.0,
                    model_year=2019.0,
                ),
                _master_row(
                    provider="CompAsia",
                    device="iPad",
                    sub_device="iPad",
                    model_number="A2197, A2198, A2200",
                    model="iPad 7",
                    retail_price=1999.0,
                    storage=32.0,
                    connectivity="Wi-Fi + Cellular",
                    trade_in=280.0,
                    model_year=2019.0,
                ),
            ]
        )

    def test_ipad_bug_report_example_is_canonicalized(self):
        # Exact scenario reported: Device "Pad" (raw Category
        # text), Sub-device "ipad" (lowercase), Standardized Model
        # still carrying its connectivity suffix, and Connectivity
        # left at "Unknown" because nothing had extracted it yet.
        master_df = self._master_df_with_ipad7()
        incoming_df = _df(
            [
                _master_row(
                    provider="3Cat",
                    device="Pad",
                    sub_device="ipad",
                    model_number=None,
                    model="iPad 7 Wi-Fi + Cellular",
                    retail_price=None,
                    storage=32.0,
                    connectivity="Unknown",
                    trade_in=180.0,
                    model_year=2019.0,
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        canonical_row = result["canonical_df"].iloc[0]

        self.assertEqual(canonical_row["Device"], "iPad")
        self.assertEqual(canonical_row["Sub-device"], "iPad")
        self.assertEqual(
            canonical_row["Standardized Model"], "iPad 7"
        )
        self.assertEqual(
            canonical_row["Connectivity"], "Wi-Fi + Cellular"
        )

    def test_incoming_row_own_values_are_preserved_not_master_values(
        self,
    ):
        # The incoming row's own Provider, Storage, and Trade-In
        # value must survive untouched, even though they differ
        # from every candidate Master row's values. Canonicalization
        # must never copy a matched Master row wholesale.
        master_df = self._master_df_with_ipad7()
        incoming_df = _df(
            [
                _master_row(
                    provider="3Cat",
                    device="Pad",
                    sub_device="ipad",
                    model_number=None,
                    model="iPad 7 Wi-Fi + Cellular",
                    retail_price=None,
                    storage=32.0,
                    connectivity="Unknown",
                    trade_in=180.0,
                    model_year=2019.0,
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        canonical_row = result["canonical_df"].iloc[0]

        # Neither Master candidate has Provider "3Cat", Storage 32,
        # or Trade-In 180 -- if these came back unchanged, the
        # canonicalization step did not overwrite them from a
        # matched Master row.
        self.assertEqual(canonical_row["Provider"], "3Cat")
        self.assertEqual(
            float(canonical_row["Storage (GB)"]), 32.0
        )
        self.assertEqual(
            canonical_row["Max. Trade-In Value (RM)"], 180.0
        )

    def test_canonicalization_is_generic_not_hardcoded_to_ipad_7(self):
        # A different Device/Sub-device/generation (Mac mini) run
        # through the exact same mechanism, to confirm this isn't a
        # special case for "iPad 7".
        master_df = _df(
            [
                _master_row(
                    provider="CompAsia",
                    device="Mac",
                    sub_device="Mac mini",
                    model_number="A2686",
                    model="Mac mini",
                    retail_price=2199.0,
                    storage=256.0,
                    connectivity="Unknown",
                    trade_in=650.0,
                    model_year=2023.0,
                )
            ]
        )
        incoming_df = _df(
            [
                _master_row(
                    provider="Switch",
                    device="mac",
                    sub_device="mac mini",
                    model_number=None,
                    model="Mac Mini",
                    retail_price=None,
                    storage=256.0,
                    connectivity="Unknown",
                    trade_in=700.0,
                    model_year=2023.0,
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        canonical_row = result["canonical_df"].iloc[0]

        self.assertEqual(canonical_row["Device"], "Mac")
        self.assertEqual(canonical_row["Sub-device"], "Mac mini")
        # Its own Provider/Trade-In must still survive untouched.
        self.assertEqual(canonical_row["Provider"], "Switch")
        self.assertEqual(
            canonical_row["Max. Trade-In Value (RM)"], 700.0
        )

    def test_unrecognized_model_left_untouched(self):
        # A model/connectivity combination Master has no
        # counterpart for at all must not be forcibly rewritten --
        # canonicalization only acts when Master can confirm a
        # spelling for this Device.
        master_df = self._master_df_with_ipad7()
        incoming_df = _df(
            [
                _master_row(
                    provider="3Cat",
                    device="iPad",
                    sub_device="iPad",
                    model_number=None,
                    model="iPad Quantum Wi-Fi + Cellular",
                    retail_price=None,
                    storage=32.0,
                    connectivity="Unknown",
                    trade_in=180.0,
                    model_year=2019.0,
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        canonical_row = result["canonical_df"].iloc[0]

        self.assertEqual(
            canonical_row["Standardized Model"],
            "iPad Quantum Wi-Fi + Cellular",
        )


class TestNewClassification(unittest.TestCase):
    def test_no_master_counterpart_is_new(self):
        master_df = _df([])
        incoming_df = _df([_master_row()])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "new")
        self.assertIsNone(row["conflict_type"])
        self.assertIsNone(row["match_index"])


class TestInvalidData(unittest.TestCase):
    def test_malformed_numeric_field_is_invalid_data(self):
        master_df = _df([_master_row()])
        incoming_df = _df([_master_row(retail_price="RM 3599")])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "invalid_data")


class TestMacModelNumberNormalization(unittest.TestCase):
    """
    Regression coverage for the Mac Model Number normalization bug:
    a genuine Mac hardware identifier like "MacBookAir8,2" has a
    comma that is part of the identifier itself, not a separator
    between multiple Model Numbers, and must survive
    normalize_model_numbers() as a single token when the row's
    Device is Mac.
    """

    def test_macbookair8_2_is_one_token(self):
        tokens = bi.normalize_model_numbers(
            "MacBookAir8,2", device="Mac"
        )
        self.assertEqual(tokens, frozenset({"MACBOOKAIR8,2"}))

    def test_macbookair9_1_is_one_token(self):
        tokens = bi.normalize_model_numbers(
            "MacBookAir9,1", device="Mac"
        )
        self.assertEqual(tokens, frozenset({"MACBOOKAIR9,1"}))

    def test_iphone_comma_separated_list_is_unchanged(self):
        # Existing iPhone behavior: commas are genuine separators
        # between multiple Apple regulatory Model Numbers.
        tokens = bi.normalize_model_numbers(
            "A2111, A2221, A2223", device="iPhone"
        )
        self.assertEqual(tokens, frozenset({"A2111", "A2221", "A2223"}))

    def test_no_device_argument_preserves_prior_behavior(self):
        # Callers that don't pass device at all (or pass a
        # non-Mac/unknown device) must see exactly the old
        # comma-splits-everything behavior.
        tokens = bi.normalize_model_numbers("A2111, A2221, A2223")
        self.assertEqual(tokens, frozenset({"A2111", "A2221", "A2223"}))

    def test_multiple_mac_identifiers_separated_by_slash(self):
        tokens = bi.normalize_model_numbers(
            "MacBookAir8,2/MacBookAir9,1", device="Mac"
        )
        self.assertEqual(
            tokens, frozenset({"MACBOOKAIR8,2", "MACBOOKAIR9,1"})
        )

    def test_master_mac_row_inherits_without_corruption(self):
        # A Master row's own Mac Model Number must not be mangled
        # into "MACBOOKAIR8" + "2" when normalized for inheritance/
        # comparison.
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MacBookAir8,2",
                    model="MacBook Air i5 1.6GHz 13-inch (Mid 2019)",
                    storage_type="SSD",
                    connectivity="Unknown",
                    model_year=2019.0,
                )
            ]
        )
        tokens = bi.normalize_model_numbers(
            master_df.at[0, "Model Number"],
            device=master_df.at[0, "Device"],
        )
        self.assertEqual(tokens, frozenset({"MACBOOKAIR8,2"}))
        self.assertNotIn("MACBOOKAIR8", tokens)
        self.assertNotIn("2", tokens)


class TestMacModelNumberAppend(unittest.TestCase):
    """
    End-to-end coverage that the Mac-aware normalization flows
    correctly through classification and into the applied Master
    row, without disturbing the existing iPhone-style overlap/
    append behavior.
    """

    def test_new_mac_model_number_is_flagged_for_append(self):
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MacBookAir8,2",
                    model="MacBook Air i5 1.6GHz 13-inch (Mid 2019)",
                    storage_type="SSD",
                    connectivity="Unknown",
                    model_year=2019.0,
                )
            ]
        )
        incoming_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MacBookAir9,1",
                    model="MacBook Air i5 1.6GHz 13-inch (Mid 2019)",
                    storage_type="SSD",
                    connectivity="Unknown",
                    model_year=2019.0,
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "update")
        self.assertIsNone(row["conflict_type"])
        self.assertEqual(row["model_number_flag"], "append")
        self.assertTrue(row["model_number_update_required"])
        self.assertEqual(
            row["model_numbers_to_append"], ["MACBOOKAIR9,1"]
        )

    def test_same_mac_model_number_is_a_plain_match(self):
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MacBookAir8,2",
                    model="MacBook Air i5 1.6GHz 13-inch (Mid 2019)",
                    storage_type="SSD",
                    connectivity="Unknown",
                    model_year=2019.0,
                )
            ]
        )
        incoming_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MacBookAir8,2",
                    model="MacBook Air i5 1.6GHz 13-inch (Mid 2019)",
                    storage_type="SSD",
                    connectivity="Unknown",
                    model_year=2019.0,
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        # An exact Mac Model Number match must be recognized as
        # such -- not corrupted into partial-overlap noise by a
        # bad comma split.
        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "duplicate")
        self.assertEqual(row["model_number_flag"], "match")

    def test_applying_mac_append_preserves_identifier_in_master(self):
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MacBookAir8,2",
                    model="MacBook Air i5 1.6GHz 13-inch (Mid 2019)",
                    storage_type="SSD",
                    connectivity="Unknown",
                    model_year=2019.0,
                )
            ]
        )
        incoming_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MacBookAir9,1",
                    model="MacBook Air i5 1.6GHz 13-inch (Mid 2019)",
                    storage_type="SSD",
                    connectivity="Unknown",
                    model_year=2019.0,
                )
            ]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )

        applied = bi.apply_classified_rows(
            canonical_df=result["canonical_df"],
            row_results=result["rows"],
            master_df=master_df,
        )

        updated_value = applied["master_df"].at[0, "Model Number"]

        # Both identifiers must survive intact as two comma-joined
        # whole tokens, never split into four garbage fragments.
        parts = [p.strip() for p in updated_value.split(", ")]
        self.assertIn("MacBookAir8,2", parts)
        self.assertIn("MACBOOKAIR9,1", parts)
        self.assertEqual(len(parts), 2)


class TestNormalizeMasterModelNumberColumn(unittest.TestCase):
    """
    Regression coverage for the Master-persistence-facing
    normalization: normalize_master_model_number_column() is what
    must run on the Master dataframe immediately before it is
    written to Google Sheets ("Cleaned Master") and to the local
    CSV, so a Mac hardware identifier's internal comma survives
    intact while separator style is still canonicalized.

    Unlike normalize_model_numbers(), this does NOT upper-case or
    otherwise rewrite each identifier's own text -- only the
    separator between multiple identifiers is canonicalized to
    ", ".
    """

    def test_mac_model_number_persisted_format(self):
        # Test 1: a single Mac identifier must be preserved exactly,
        # not split and not re-cased.
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MacBookAir8,2",
                    storage_type="SSD",
                    connectivity="Unknown",
                )
            ]
        )

        result_df = bi.normalize_master_model_number_column(master_df)

        self.assertEqual(
            result_df.at[0, "Model Number"], "MacBookAir8,2"
        )
        self.assertNotEqual(
            result_df.at[0, "Model Number"], "MACBOOKAIR8, 2"
        )
        # Exactly one identifier, not two.
        self.assertEqual(
            len(result_df.at[0, "Model Number"].split(", ")), 1
        )

    def test_multiple_mac_identifiers_persisted_format(self):
        # Test 2: multiple Mac identifiers, mixed separator style
        # in, canonical comma-space style out -- each identifier's
        # own text (including its internal comma) untouched.
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MacBookAir8,2 / MacBookAir9,1",
                    storage_type="SSD",
                    connectivity="Unknown",
                )
            ]
        )

        result_df = bi.normalize_master_model_number_column(master_df)

        self.assertEqual(
            result_df.at[0, "Model Number"],
            "MacBookAir8,2, MacBookAir9,1",
        )

    def test_iphone_model_numbers_persisted_format(self):
        # Test 3: iPhone regression -- multiple regulatory Model
        # Numbers still canonicalize to a comma-space separated
        # list, regardless of how they arrived.
        master_df = _df(
            [_master_row(model_number="A2111 / A2221 / A2223")]
        )

        result_df = bi.normalize_master_model_number_column(master_df)

        self.assertEqual(
            result_df.at[0, "Model Number"], "A2111, A2221, A2223"
        )

    def test_unrelated_master_fields_are_unchanged(self):
        # Test 4: only the Model Number column may change; every
        # other Master field/value is passed through untouched.
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MacBookAir8,2 / MacBookAir9,1",
                    model="MacBook Air i5 1.6GHz 13-inch (Mid 2019)",
                    retail_price=5999.0,
                    storage=256.0,
                    storage_type="SSD",
                    connectivity="Unknown",
                    trade_in=350.0,
                    model_year=2019.0,
                )
            ]
        )

        result_df = bi.normalize_master_model_number_column(master_df)

        for column in bi.CANONICAL_FIELDS:
            if column == "Model Number":
                continue
            with self.subTest(column=column):
                original = master_df.at[0, column]
                updated = result_df.at[0, column]
                if pd.isna(original) and pd.isna(updated):
                    continue
                self.assertEqual(updated, original)

        # And the Model Number column itself DID change (was
        # actually normalized, not silently skipped).
        self.assertNotEqual(
            result_df.at[0, "Model Number"],
            master_df.at[0, "Model Number"],
        )

    def test_already_canonical_value_is_left_untouched(self):
        # A well-formed Mac Model Number should come back byte-
        # identical -- normalization should never rewrite text
        # that wasn't broken in the first place.
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MacBookAir8,2",
                    storage_type="SSD",
                    connectivity="Unknown",
                )
            ]
        )

        result_df = bi.normalize_master_model_number_column(master_df)

        self.assertEqual(
            result_df.at[0, "Model Number"],
            master_df.at[0, "Model Number"],
        )

    def test_blank_model_number_is_left_untouched(self):
        master_df = _df([_master_row(model_number=None)])

        result_df = bi.normalize_master_model_number_column(master_df)

        self.assertTrue(pd.isna(result_df.at[0, "Model Number"]))

    def test_does_not_mutate_the_input_dataframe(self):
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MacBookAir8,2 / MacBookAir9,1",
                    storage_type="SSD",
                    connectivity="Unknown",
                )
            ]
        )
        original_value = master_df.at[0, "Model Number"]

        bi.normalize_master_model_number_column(master_df)

        self.assertEqual(master_df.at[0, "Model Number"], original_value)

    def test_all_caps_mac_identifier_is_recased_to_apple_style(self):
        # This is the exact symptom reported: a Mac identifier that
        # ended up stored in all caps (e.g. from an appended
        # comparison token) must come back in Apple's own casing.
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Pro",
                    model_number="MACBOOKPRO17,1",
                    storage_type="SSD",
                    connectivity="Unknown",
                )
            ]
        )

        result_df = bi.normalize_master_model_number_column(master_df)

        self.assertEqual(
            result_df.at[0, "Model Number"], "MacBookPro17,1"
        )

    def test_recasing_is_idempotent_on_already_correct_casing(self):
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Pro",
                    model_number="MacBookPro17,1",
                    storage_type="SSD",
                    connectivity="Unknown",
                )
            ]
        )

        result_df = bi.normalize_master_model_number_column(master_df)

        self.assertEqual(
            result_df.at[0, "Model Number"], "MacBookPro17,1"
        )

    def test_mixed_case_list_is_recased_and_joined(self):
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="MACBOOKAIR8,2 / MacBookAir9,1",
                    storage_type="SSD",
                    connectivity="Unknown",
                )
            ]
        )

        result_df = bi.normalize_master_model_number_column(master_df)

        self.assertEqual(
            result_df.at[0, "Model Number"],
            "MacBookAir8,2, MacBookAir9,1",
        )

    def test_generic_mac_family_identifier_is_recased(self):
        # Apple Silicon-era generic identifiers like "Mac16,3" (no
        # MacBook/iMac qualifier) must also recase correctly.
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="Mac mini",
                    model_number="MAC16,3",
                    storage_type="SSD",
                    connectivity="Unknown",
                )
            ]
        )

        result_df = bi.normalize_master_model_number_column(master_df)

        self.assertEqual(result_df.at[0, "Model Number"], "Mac16,3")

    def test_unknown_mac_family_prefix_is_left_untouched(self):
        # A product-line prefix we don't recognize should never be
        # guess-recased -- leave it exactly as it arrived.
        master_df = _df(
            [
                _master_row(
                    device="Mac",
                    sub_device="MacBook Air",
                    model_number="FUTUREMACTHING5,1",
                    storage_type="SSD",
                    connectivity="Unknown",
                )
            ]
        )

        result_df = bi.normalize_master_model_number_column(master_df)

        self.assertEqual(
            result_df.at[0, "Model Number"], "FUTUREMACTHING5,1"
        )

    def test_iphone_model_numbers_are_never_recased(self):
        # Recasing is Mac-only; iPhone tokens are already the
        # canonical "A####" form and must never be touched by this
        # logic path.
        master_df = _df([_master_row(model_number="a2111, a2221")])

        result_df = bi.normalize_master_model_number_column(master_df)

        self.assertEqual(
            result_df.at[0, "Model Number"], "a2111, a2221"
        )


class TestProblematicRowEditorReanalysis(unittest.TestCase):
    """
    Admin Bulk Import Preview — Problematic Row editor.

    The editor's "Re-analyze" action (see the new
    /admin/bulk-import/reanalyze-row endpoint in app.py, and the
    identical pattern already used by the Admin Review Queue's
    _recheck_review_queue_record()) is nothing more than:

        effective_record_to_row(effective_record)
            -> pd.DataFrame([...])
            -> analyze_upload(raw_df=..., master_df=..., clean_fn=...)

    run again, one row at a time, against the CURRENT Master
    dataset. These tests exercise exactly that round trip so the
    editor's behavior is verified against the real, unmodified
    bulk-import classification logic -- no separate/duplicated
    classification logic is introduced anywhere for this feature.
    """

    def _load_problematic_row(self, master_df, incoming_row):
        """
        Simulate what happens when the Admin opens "Edit" on a
        Preview row: analyze the original incoming row and assert
        it is in fact problematic (Needs Review or Invalid Data)
        before editing it, exactly like the Preview would have
        classified it in the first place.
        """
        incoming_df = _df([incoming_row])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertIn(
            row["conflict_type"], {"needs_review", "invalid_data"}
        )

        return row

    def _reanalyze(self, master_df, effective_record):
        """
        The exact operation the new /admin/bulk-import/reanalyze-row
        endpoint performs on the Admin's edited effective_record.
        """
        proposed_row = bi.effective_record_to_row(effective_record)
        proposed_df = pd.DataFrame([proposed_row], columns=bi.CANONICAL_FIELDS)

        result = bi.analyze_upload(
            raw_df=proposed_df, master_df=master_df, clean_fn=None
        )

        return result["rows"][0]

    # ---- Loading a problematic row into the editor --------------

    def test_loading_needs_review_row_exposes_effective_record_and_reasons(self):
        master_df = _df([_master_row(retail_price=3599.0)])
        incoming_row = _master_row(retail_price=3799.0)  # Retail Price changed

        row = self._load_problematic_row(master_df, incoming_row)

        self.assertEqual(row["conflict_type"], "needs_review")
        self.assertTrue(row["reasons"])
        self.assertIn("values", row["effective_record"])
        self.assertIn("provenance", row["effective_record"])
        self.assertEqual(
            row["effective_record"]["values"]["Retail Price"], 3799.0
        )

    def test_loading_invalid_data_row_exposes_effective_record(self):
        master_df = _df([_master_row()])
        incoming_row = _master_row(retail_price="RM 3599")

        row = self._load_problematic_row(master_df, incoming_row)

        self.assertEqual(row["conflict_type"], "invalid_data")
        self.assertIn("values", row["effective_record"])

    # ---- Editing + re-analysis, per resolution outcome -----------

    def test_edit_resolving_to_new_stays_new_on_reanalysis(self):
        # Needs Review because Provider's price changed unexpectedly
        # relative to Master; Admin corrects it to a genuinely new
        # Provider, which should now classify as NEW.
        master_df = _df([_master_row(provider="CompAsia")])
        incoming_row = _master_row(provider="CompAsia", retail_price=9999.0)

        row = self._load_problematic_row(master_df, incoming_row)

        edited = row["effective_record"]
        edited["values"]["Provider"] = "Celcom"

        reanalyzed = self._reanalyze(master_df, edited)

        self.assertEqual(reanalyzed["classification"], "new")
        self.assertIsNone(reanalyzed["conflict_type"])

    def test_edit_resolving_to_update_stays_update_on_reanalysis(self):
        master_df = _df([_master_row(retail_price=3599.0)])
        incoming_row = _master_row(retail_price="RM 3599")  # invalid_data

        row = self._load_problematic_row(master_df, incoming_row)

        edited = row["effective_record"]
        edited["values"]["Retail Price"] = 3599.0
        edited["values"]["Max. Trade-In Value (RM)"] = 450.0  # only mutable field changes

        reanalyzed = self._reanalyze(master_df, edited)

        self.assertEqual(reanalyzed["classification"], "update")
        self.assertIsNone(reanalyzed["conflict_type"])
        self.assertEqual(
            reanalyzed["different_fields"], ["Max. Trade-In Value (RM)"]
        )

    def test_edit_resolving_to_duplicate_on_reanalysis(self):
        master_df = _df([_master_row(retail_price=3599.0, trade_in=390.0)])
        incoming_row = _master_row(retail_price="RM 3599")  # invalid_data

        row = self._load_problematic_row(master_df, incoming_row)

        edited = row["effective_record"]
        edited["values"]["Retail Price"] = 3599.0  # now matches Master exactly

        reanalyzed = self._reanalyze(master_df, edited)

        self.assertEqual(reanalyzed["classification"], "conflict")
        self.assertEqual(reanalyzed["conflict_type"], "duplicate")

    def test_edit_still_needs_review_remains_unresolved(self):
        master_df = _df([_master_row(retail_price=3599.0)])
        incoming_row = _master_row(retail_price=3799.0)

        row = self._load_problematic_row(master_df, incoming_row)

        edited = row["effective_record"]
        # A different, still-unexplained Retail Price -- the edit
        # did not actually resolve the conflict.
        edited["values"]["Retail Price"] = 3899.0

        reanalyzed = self._reanalyze(master_df, edited)

        self.assertEqual(reanalyzed["classification"], "conflict")
        self.assertEqual(reanalyzed["conflict_type"], "needs_review")

    def test_edit_still_invalid_data_remains_unresolved(self):
        master_df = _df([_master_row()])
        incoming_row = _master_row(retail_price="RM 3599")

        row = self._load_problematic_row(master_df, incoming_row)

        edited = row["effective_record"]
        # Still malformed -- e.g. Admin mistyped the correction.
        edited["values"]["Retail Price"] = "still not a number"

        reanalyzed = self._reanalyze(master_df, edited)

        self.assertEqual(reanalyzed["classification"], "conflict")
        self.assertEqual(reanalyzed["conflict_type"], "invalid_data")

    # ---- Re-analysis reflects the CURRENT Master dataset ---------

    def test_reanalysis_uses_current_master_dataset_not_original(self):
        # Simulate Master changing between when the Preview was
        # first generated and when the Admin re-analyzes an edited
        # row: the ORIGINAL master had a lower Trade-In Value, the
        # CURRENT master (passed in at re-analysis time) has been
        # updated -- re-analysis must reflect the CURRENT one.
        original_master_df = _df([_master_row(trade_in=390.0)])
        current_master_df = _df([_master_row(trade_in=450.0)])

        incoming_row = _master_row(retail_price="RM 3599")  # invalid_data
        row = self._load_problematic_row(original_master_df, incoming_row)

        edited = row["effective_record"]
        edited["values"]["Retail Price"] = 3599.0
        edited["values"]["Max. Trade-In Value (RM)"] = 450.0

        reanalyzed_against_current = self._reanalyze(
            current_master_df, edited
        )

        # Now an exact duplicate of the CURRENT master, not the
        # stale one the row was originally loaded against.
        self.assertEqual(reanalyzed_against_current["classification"], "conflict")
        self.assertEqual(
            reanalyzed_against_current["conflict_type"], "duplicate"
        )


class TestCollectionDate(unittest.TestCase):
    """
    Collection Date -- Master metadata (the date associated with a
    provider's current pricing data), added alongside the existing
    schema.

    Per spec, it must:
      - never factor into Apple configuration identity/matching/
        duplicate detection (tested here directly; the matching/
        scoring functions themselves are untouched by this field);
      - be stored on genuinely NEW rows;
      - replace the Master value on UPDATE, exactly like
        Max. Trade-In Value (RM);
      - never be overwritten by a DUPLICATE re-import;
      - never be invented for existing rows with no historical date;
      - survive the Review Queue / Problematic Row re-analysis round
        trip the same way every other effective_record field does.
    """

    # ---- UPDATE replaces both Trade-In Value and Collection Date --

    def test_update_replaces_trade_in_value_and_collection_date(self):
        master_df = _df(
            [_master_row(trade_in=1000.0, collection_date="2026-08-01")]
        )
        incoming_df = _df(
            [_master_row(trade_in=1100.0, collection_date="2026-09-01")]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "update")
        self.assertIsNone(row["conflict_type"])

        applied = bi.apply_classified_rows(
            canonical_df=result["canonical_df"],
            row_results=result["rows"],
            master_df=master_df,
        )

        self.assertEqual(
            applied["master_df"].at[0, "Max. Trade-In Value (RM)"], 1100.0
        )
        self.assertEqual(
            applied["master_df"].at[0, "Collection Date"], "2026-09-01"
        )

    # ---- NEW rows receive their Collection Date --------------------

    def test_new_row_receives_its_collection_date(self):
        master_df = _df([])
        incoming_df = _df([_master_row(collection_date="2026-09-01")])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "new")

        applied = bi.apply_classified_rows(
            canonical_df=result["canonical_df"],
            row_results=result["rows"],
            master_df=master_df,
        )

        new_row = applied["master_df"].iloc[-1]
        self.assertEqual(new_row["Collection Date"], "2026-09-01")

    # ---- Collection Date is never identity --------------------------

    def test_collection_date_alone_does_not_create_a_new_configuration(self):
        # Identical Apple configuration and pricing, only Collection
        # Date differs -- must remain an exact duplicate, never "new"
        # or "needs_review" purely because of the date.
        master_df = _df([_master_row(collection_date="2026-08-01")])
        incoming_df = _df([_master_row(collection_date="2026-09-01")])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["classification"], "conflict")
        self.assertEqual(row["conflict_type"], "duplicate")
        self.assertNotIn("Collection Date", row["different_fields"])
        self.assertNotIn("Collection Date", row["unknown_fields"])

    # ---- DUPLICATE never overwrites an existing Collection Date ----

    def test_duplicate_does_not_overwrite_existing_collection_date(self):
        master_df = _df([_master_row(collection_date="2026-08-01")])
        incoming_df = _df([_master_row(collection_date="2026-09-01")])

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )

        applied = bi.apply_classified_rows(
            canonical_df=result["canonical_df"],
            row_results=result["rows"],
            master_df=master_df,
        )

        # Duplicate rows are skipped entirely -- Master untouched.
        self.assertEqual(
            applied["master_df"].at[0, "Collection Date"], "2026-08-01"
        )

    # ---- Existing blank dates are never auto-populated ---------------

    def test_existing_blank_collection_date_is_not_auto_populated(self):
        master_df = _df([_master_row(collection_date=None)])
        self.assertTrue(pd.isna(master_df.at[0, "Collection Date"]))

        incoming_df = _df(
            [_master_row(trade_in=999.0, collection_date=None)]
        )

        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]
        self.assertEqual(row["classification"], "update")

        applied = bi.apply_classified_rows(
            canonical_df=result["canonical_df"],
            row_results=result["rows"],
            master_df=master_df,
        )

        # Trade-In Value updates as normal; there is no incoming
        # Collection Date to replace it with, so it stays blank --
        # never auto-filled with today's date or anything else.
        self.assertEqual(
            applied["master_df"].at[0, "Max. Trade-In Value (RM)"], 999.0
        )
        self.assertTrue(
            pd.isna(applied["master_df"].at[0, "Collection Date"])
        )

    # ---- Survives Review Queue / Problematic Row re-analysis --------

    def test_collection_date_survives_review_queue_reanalysis(self):
        master_df = _df([_master_row(retail_price=3599.0)])
        incoming_row = _master_row(
            retail_price=3799.0, collection_date="2026-09-01"
        )

        incoming_df = _df([incoming_row])
        result = bi.analyze_upload(
            raw_df=incoming_df, master_df=master_df, clean_fn=None
        )
        row = result["rows"][0]

        self.assertEqual(row["conflict_type"], "needs_review")
        self.assertEqual(
            row["effective_record"]["values"]["Collection Date"],
            "2026-09-01",
        )

        # Re-analyze exactly as the Review Queue / Problematic Row
        # editor's "Re-analyze" action does: effective_record -> row
        # -> analyze_upload again.
        proposed_row = bi.effective_record_to_row(row["effective_record"])
        proposed_df = pd.DataFrame(
            [proposed_row], columns=bi.CANONICAL_FIELDS
        )

        reanalyzed = bi.analyze_upload(
            raw_df=proposed_df, master_df=master_df, clean_fn=None
        )["rows"][0]

        self.assertEqual(
            reanalyzed["effective_record"]["values"]["Collection Date"],
            "2026-09-01",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)