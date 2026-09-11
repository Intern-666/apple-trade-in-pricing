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
    chipset="A13 Bionic",
    case_size=None,
    charging_method=None,
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
        "Chipset": chipset,
        "Case Size": case_size,
        "Charging Method": charging_method,
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
            "different_fields": ["Max. Trade-In Value (RM)", "Chipset"],
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
                    chipset="A10",
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
                    chipset="A10",
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
                    chipset="A10",
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
                    chipset="A10",
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
                    chipset="M2",
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
                    chipset="M2",
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
                    chipset="A10",
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


if __name__ == "__main__":
    unittest.main(verbosity=2)