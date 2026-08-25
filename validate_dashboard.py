# =============================================================================
# APPLE TRADE-IN DASHBOARD — AUTOMATED VALIDATION SUITE
# =============================================================================
#
# Purpose:
#   Validate the finalized Apple dataset and model.pkl before using them
#   in the Streamlit dashboard.
#
# Expected CSV columns:
#   Provider
#   Device
#   Sub-device
#   Standardized Model
#   Specification
#   Storage (GB)
#   Storage Type
#   Connectivity
#   Material
#   Max. Trade-In Value (RM)
#
# Expected ML features:
#   storage_type
#   storage
#   standardized_model
#   screen_size
#   ram
#   provider
#   pricing_category
#   model_year
#   model
#   generation
#   device_type
#   device_age
#   connectivity
#   clock_speed_ghz
#   chipset
#
# =============================================================================


from pathlib import Path
import pandas as pd
import numpy as np
import re
import joblib
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path.cwd()

CSV_PATH = BASE_DIR / "master_apple_final.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

CURRENT_YEAR = 2026

def parse_storage(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    match = re.match(
        r"^([\d.]+)\s*(GB|TB)$",
        value
    )

    if not match:
        return np.nan

    number = float(match.group(1))
    unit = match.group(2)

    if unit == "TB":
        number *= 1024

    return number

# =============================================================================
# TEST ENGINE
# =============================================================================

results = []


def test(name, condition, detail=""):
    """
    Register one validation test.
    """

    passed = bool(condition)

    results.append({
        "Test": name,
        "Status": "PASS" if passed else "FAIL",
        "Details": detail
    })

    symbol = "✓" if passed else "✗"

    print(
        f"{symbol} {name}"
        + (f" — {detail}" if detail else "")
    )

    return passed


def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)



# =============================================================================
# LOAD DATA
# =============================================================================

section("LOADING DATA")


try:

    df = pd.read_csv(CSV_PATH)

    print(f"CSV loaded: {CSV_PATH}")
    print(f"Rows: {len(df):,}")

    csv_loaded = True

except Exception as e:

    print(f"FAILED TO LOAD CSV: {e}")

    csv_loaded = False
    df = pd.DataFrame()


test(
    "CSV can be loaded",
    csv_loaded
)


try:

    model = joblib.load(MODEL_PATH)

    print(f"Model loaded: {MODEL_PATH}")
    print(f"Model type: {type(model).__name__}")

    model_loaded = True

except Exception as e:

    print(f"FAILED TO LOAD MODEL: {e}")

    model_loaded = False
    model = None


test(
    "model.pkl can be loaded",
    model_loaded
)


if not csv_loaded:

    print("\nCSV could not be loaded. Validation stopped.")

else:

    # =========================================================================
    # NORMALIZE COLUMN NAMES
    # =========================================================================

    df.columns = (
        df.columns
        .str.strip()
    )


    # =========================================================================
    # EXPECTED CSV STRUCTURE
    # =========================================================================

    section("1. DATASET STRUCTURE")


    expected_columns = [
        "Provider",
        "Device",
        "Sub-device",
        "Standardized Model",
        "Specification",
        "Storage (GB)",
        "Storage Type",
        "Connectivity",
        "Material",
        "Max. Trade-In Value (RM)"
    ]


    missing_columns = [
        col
        for col in expected_columns
        if col not in df.columns
    ]


    test(
        "All required CSV columns exist",
        len(missing_columns) == 0,
        (
            "All expected columns present"
            if not missing_columns
            else f"Missing: {missing_columns}"
        )
    )


    test(
        "Dataset contains records",
        len(df) > 0,
        f"{len(df):,} rows"
    )


    # =========================================================================
    # DEVICE CATEGORIES
    # =========================================================================

    section("2. DEVICE CATEGORIES")


    expected_devices = {
        "iPhone",
        "iPad",
        "Mac",
        "Apple Watch",
        "AirPods"
    }


    actual_devices = set(
        df["Device"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )


    unexpected_devices = actual_devices - expected_devices
    missing_devices = expected_devices - actual_devices


    test(
        "Expected device categories exist",
        len(missing_devices) == 0,
        (
            "All 5 device categories present"
            if not missing_devices
            else f"Missing: {sorted(missing_devices)}"
        )
    )


    test(
        "No unexpected device categories exist",
        len(unexpected_devices) == 0,
        (
            "No unexpected categories"
            if not unexpected_devices
            else f"Unexpected: {sorted(unexpected_devices)}"
        )
    )


    # =========================================================================
    # BASIC DATA INTEGRITY
    # =========================================================================

    section("3. BASIC DATA INTEGRITY")


    missing_model_count = (
        df["Standardized Model"]
        .isna()
        .sum()
    )


    empty_model_count = (
        df["Standardized Model"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )


    test(
        "Every record has a Standardized Model",
        missing_model_count == 0 and empty_model_count == 0,
        f"Invalid model records: {missing_model_count + empty_model_count}"
    )


    duplicate_columns = [
        "Provider",
        "Device",
        "Sub-device",
        "Standardized Model",
        "Specification",
        "Storage (GB)",
        "Storage Type",
        "Connectivity",
        "Material",
        "Max. Trade-In Value (RM)"
    ]


    duplicates = df.duplicated(
        subset=[
            col
            for col in duplicate_columns
            if col in df.columns
        ]
    ).sum()


    test(
        "No duplicate configurations",
        duplicates == 0,
        f"Duplicate rows: {duplicates}"
    )


    if "Storage (GB)" in df.columns:

        storage_numeric = pd.to_numeric(
            df["Storage (GB)"],
            errors="coerce"
        )

        invalid_storage = (
            storage_numeric.notna()
            & (storage_numeric <= 0)
        ).sum()

    else:

        invalid_storage = 0


    test(
        "No invalid storage values",
        invalid_storage == 0,
        f"Invalid storage records: {invalid_storage}"
    )


    if "Max. Trade-In Value (RM)" in df.columns:

        trade_values = pd.to_numeric(
            df["Max. Trade-In Value (RM)"],
            errors="coerce"
        )

        negative_values = (
            trade_values.notna()
            & (trade_values < 0)
        ).sum()

    else:

        negative_values = 0


    test(
        "No negative trade-in values",
        negative_values == 0,
        f"Negative values: {negative_values}"
    )


    # =========================================================================
    # DEVICE-SPECIFIC VALIDATION
    # =========================================================================

    # -------------------------------------------------------------------------
    # iPHONE
    # -------------------------------------------------------------------------

    section("4. IPHONE VALIDATION")


    iphone = df[
        df["Device"] == "iPhone"
    ].copy()


    iphone_models = (
        iphone["Standardized Model"]
        .nunique()
    )


    test(
        "iPhone records exist",
        len(iphone) > 0,
        f"{len(iphone):,} records"
    )


    test(
        "iPhone models exist",
        iphone_models > 0,
        f"{iphone_models} models"
    )


    iphone_subdevice_values = (
        iphone["Sub-device"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    iphone_nonempty_subdevice = (
        iphone_subdevice_values != ""
    ).sum()


    test(
        "iPhone does not require Sub-device",
        iphone_nonempty_subdevice == 0,
        f"Non-empty Sub-device records: {iphone_nonempty_subdevice}"
    )


    iphone_storage = pd.to_numeric(
        iphone["Storage (GB)"],
        errors="coerce"
    )


    test(
        "iPhone has storage data",
        iphone_storage.notna().any(),
        f"Records with storage: {iphone_storage.notna().sum():,}"
    )


    # -------------------------------------------------------------------------
    # IPAD
    # -------------------------------------------------------------------------

    section("5. IPAD VALIDATION")


    ipad = df[
        df["Device"] == "iPad"
    ].copy()


    ipad_models = ipad[
        "Standardized Model"
    ].nunique()


    ipad_subdevices = (
        ipad["Sub-device"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    ipad_connectivity = (
        ipad["Connectivity"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    test(
        "iPad records exist",
        len(ipad) > 0,
        f"{len(ipad):,} records"
    )


    test(
        "iPad models exist",
        ipad_models > 0,
        f"{ipad_models} models"
    )


    test(
        "iPad has Sub-device values",
        (ipad_subdevices != "").any(),
        f"Records with Sub-device: {(ipad_subdevices != '').sum():,}"
    )


    test(
        "iPad has storage data",
        pd.to_numeric(
            ipad["Storage (GB)"],
            errors="coerce"
        ).notna().any(),
        "Storage data available"
    )


    test(
        "iPad has connectivity information",
        (ipad_connectivity != "").any(),
        f"Records with connectivity: {(ipad_connectivity != '').sum():,}"
    )


    # -------------------------------------------------------------------------
    # MAC
    # -------------------------------------------------------------------------

    section("6. MAC VALIDATION")


    mac = df[
        df["Device"] == "Mac"
    ].copy()


    mac_models = mac[
        "Standardized Model"
    ].nunique()


    mac_subdevices = (
        mac["Sub-device"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    mac_storage = pd.to_numeric(
        mac["Storage (GB)"],
        errors="coerce"
    )


    mac_storage_type = (
        mac["Storage Type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    test(
        "Mac records exist",
        len(mac) > 0,
        f"{len(mac):,} records"
    )


    test(
        "Mac models exist",
        mac_models > 0,
        f"{mac_models} models"
    )


    test(
        "Mac has Sub-device values",
        (mac_subdevices != "").any(),
        f"Records with Sub-device: {(mac_subdevices != '').sum():,}"
    )


    test(
        "Mac has storage data",
        mac_storage.notna().any(),
        f"Records with storage: {mac_storage.notna().sum():,}"
    )


    test(
        "Mac has Storage Type information",
        (mac_storage_type != "").any(),
        f"Records with Storage Type: {(mac_storage_type != '').sum():,}"
    )


    # -------------------------------------------------------------------------
    # APPLE WATCH
    # -------------------------------------------------------------------------

    section("7. APPLE WATCH VALIDATION")


    watch = df[
        df["Device"] == "Apple Watch"
    ].copy()


    watch_models = watch[
        "Standardized Model"
    ].nunique()


    watch_subdevices = (
        watch["Sub-device"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    watch_storage = pd.to_numeric(
        watch["Storage (GB)"],
        errors="coerce"
    )


    watch_connectivity = (
        watch["Connectivity"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    test(
        "Apple Watch records exist",
        len(watch) > 0,
        f"{len(watch):,} records"
    )


    test(
        "Apple Watch models exist",
        watch_models > 0,
        f"{watch_models} models"
    )


    test(
        "Apple Watch has Sub-device values",
        (watch_subdevices != "").any(),
        f"Records with Sub-device: {(watch_subdevices != '').sum():,}"
    )


    test(
        "Apple Watch has storage data",
        watch_storage.notna().any(),
        f"Records with storage: {watch_storage.notna().sum():,}"
    )


    test(
        "Apple Watch has connectivity information",
        (watch_connectivity != "").any(),
        f"Records with connectivity: {(watch_connectivity != '').sum():,}"
    )


    # -------------------------------------------------------------------------
    # AIRPODS
    # -------------------------------------------------------------------------

    section("8. AIRPODS VALIDATION")


    airpods = df[
        df["Device"] == "AirPods"
    ].copy()


    airpods_models = airpods[
        "Standardized Model"
    ].nunique()


    airpods_subdevices = (
        airpods["Sub-device"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    airpods_specification = (
        airpods["Specification"]
        .fillna("")
        .astype(str)
        .str.strip()
    )


    test(
        "AirPods records exist",
        len(airpods) > 0,
        f"{len(airpods):,} records"
    )


    test(
        "AirPods models exist",
        airpods_models > 0,
        f"{airpods_models} models"
    )


    test(
        "AirPods has Sub-device values",
        (airpods_subdevices != "").any(),
        f"Records with Sub-device: {(airpods_subdevices != '').sum():,}"
    )


    test(
        "AirPods has charging information",
        (airpods_specification != "").any(),
        f"Records with Specification: {(airpods_specification != '').sum():,}"
    )


    # =========================================================================
    # ML MODEL VALIDATION
    # =========================================================================

    section("9. ML MODEL VALIDATION")


    expected_features = [
        "storage_type",
        "storage",
        "standardized_model",
        "screen_size",
        "ram",
        "provider",
        "pricing_category",
        "model_year",
        "model",
        "generation",
        "device_type",
        "device_age",
        "connectivity",
        "clock_speed_ghz",
        "chipset"
    ]


    if model_loaded:

        try:

            actual_features = list(
                model.feature_names_in_
            )

        except Exception:

            actual_features = []


        missing_features = [
            feature
            for feature in expected_features
            if feature not in actual_features
        ]


        test(
            "Model contains expected input features",
            len(missing_features) == 0,
            (
                "All expected features present"
                if not missing_features
                else f"Missing: {missing_features}"
            )
        )


        test(
            "Model is a Pipeline",
            type(model).__name__ == "Pipeline",
            f"Model type: {type(model).__name__}"
        )


        if hasattr(model, "named_steps"):

            step_names = list(
                model.named_steps.keys()
            )

            test(
                "Model has preprocessing step",
                "preprocessor" in step_names,
                f"Steps: {step_names}"
            )

            final_step = (
                list(model.named_steps.values())[-1]
                if step_names
                else None
            )

            test(
                "Final estimator is ExtraTreesRegressor",
                type(final_step).__name__
                == "ExtraTreesRegressor",
                f"Final estimator: {type(final_step).__name__}"
            )


    # =========================================================================
    # MODEL PREDICTION TESTS
    # =========================================================================

    section("10. ML PREDICTION TESTS")


    def build_test_input(row):

        """
        Build a model input from one real dataset row.

        Dataset columns are mapped into the feature names
        expected by model.pkl.
        """

        model_year = np.nan

        if "Model_Year" in row.index:

            model_year = pd.to_numeric(
                row["Model_Year"],
                errors="coerce"
            )

        device_age = (
            max(0, CURRENT_YEAR - float(model_year))
            if pd.notna(model_year)
            else np.nan
        )


        return pd.DataFrame([{

            "storage_type":
                row.get("Storage Type", ""),

            "storage":
                pd.to_numeric(
                    row.get("Storage (GB)", np.nan),
                    errors="coerce"
                ),

            "standardized_model":
                row.get("Standardized Model", ""),

            "screen_size":
                np.nan,

            "ram":
                np.nan,

            "provider":
                row.get("Provider", ""),

            "pricing_category":
                "",

            "model_year":
                model_year,

            "model":
                row.get("Standardized Model", ""),

            "generation":
                np.nan,

            "device_type":
                row.get("Device", ""),

            "device_age":
                device_age,

            "connectivity":
                row.get("Connectivity", ""),

            "clock_speed_ghz":
                np.nan,

            "chipset":
                "Unknown"

        }])


    if model_loaded and len(df) > 0:

        # -------------------------------------------------------------
        # Select representative records automatically.
        # -------------------------------------------------------------

        test_rows = []


        for device in [
            "iPhone",
            "iPad",
            "Mac",
            "Apple Watch",
            "AirPods"
        ]:

            device_rows = df[
                df["Device"] == device
            ]

            if not device_rows.empty:

                # First representative row
                test_rows.append(
                    device_rows.iloc[0]
                )


        prediction_successes = 0
        prediction_failures = []


        for row in test_rows:

            device = row["Device"]
            model_name = row["Standardized Model"]

            try:

                X_test = build_test_input(row)

                prediction = model.predict(
                    X_test
                )[0]

                prediction = float(prediction)


                if (
                    np.isfinite(prediction)
                    and prediction >= 0
                ):

                    prediction_successes += 1

                else:

                    prediction_failures.append(
                        f"{device} / {model_name}: "
                        f"invalid prediction {prediction}"
                    )

            except Exception as e:

                prediction_failures.append(
                    f"{device} / {model_name}: {e}"
                )


        test(
            "Representative predictions succeed",
            len(prediction_failures) == 0,
            (
                f"{prediction_successes}/{len(test_rows)} predictions passed"
                if not prediction_failures
                else "; ".join(prediction_failures)
            )
        )


    # =========================================================================
    # DUPLICATE CONFIGURATION ANALYSIS
    # =========================================================================

    section("11. CONFIGURATION DUPLICATE ANALYSIS")


    configuration_columns = [
        "Device",
        "Sub-device",
        "Standardized Model",
        "Specification",
        "Storage (GB)",
        "Storage Type",
        "Connectivity",
        "Material"
    ]


    configuration_columns = [
        col
        for col in configuration_columns
        if col in df.columns
    ]


    duplicate_configs = (
        df.duplicated(
            subset=configuration_columns,
            keep=False
        )
    )


    duplicate_config_count = (
        duplicate_configs.sum()
    )


    test(
        "No duplicate device configurations",
        duplicate_config_count == 0,
        f"Duplicate configuration records: {duplicate_config_count}"
    )

    # ============================================================
    # MODEL / MASTER DATA COMPATIBILITY CHECK
    # ============================================================

    import os
    import joblib
    import pandas as pd
    import numpy as np

    print("=" * 100)
    print("MODEL / MASTER DATA COMPATIBILITY CHECK")
    print("=" * 100)


    # ============================================================
    # PATHS
    # ============================================================

    MODEL_PATH = "model.pkl"
    CSV_PATH = "structured_apple_devices_full.csv"


    # ============================================================
    # 1. LOAD MODEL
    # ============================================================

    print("\n[1] MODEL LOAD TEST")

    try:

        model = joblib.load(MODEL_PATH)

        print("PASS — model.pkl loaded successfully")
        print("Model type:", type(model))

    except Exception as e:

        print("FAIL — model.pkl could not be loaded")
        print("Error type:", type(e).__name__)
        print("Error:", e)

        model = None


    # ============================================================
    # 2. LOAD MASTER DATASET
    # ============================================================

    print("\n[2] MASTER DATASET LOAD TEST")

    try:

        master = pd.read_csv(CSV_PATH)

        print("PASS — master CSV loaded")
        print("Rows:", len(master))
        print("Columns:", list(master.columns))

    except Exception as e:

        print("FAIL — master CSV could not be loaded")
        print("Error:", e)

        master = None


    # ============================================================
    # 3. INSPECT MODEL FEATURES
    # ============================================================

    if model is not None:

        print("\n[3] MODEL FEATURE REQUIREMENTS")

        try:

            # Pipeline feature names
            model_features = list(
                model.feature_names_in_
            )

            print("PASS — model exposes feature_names_in_")

        except Exception:

            model_features = [
                "storage_type",
                "storage",
                "standardized_model",
                "screen_size",
                "ram",
                "provider",
                "pricing_category",
                "model_year",
                "model",
                "generation",
                "device_type",
                "device_age",
                "connectivity",
                "clock_speed_ghz",
                "chipset"
            ]

            print(
                "INFO — using known training feature list"
            )

        print("\nModel expects:")

        for i, feature in enumerate(model_features, 1):

            print(f"{i:02d}. {feature}")


    # ============================================================
    # 4. MASTER DATASET FEATURES
    # ============================================================

    if master is not None:

        print("\n[4] MASTER DATASET FEATURES")

        print()

        for col in master.columns:

            print(f"- {col}")


    # ============================================================
    # 5. FEATURE MAPPING
    # ============================================================

    if model is not None and master is not None:

        print("\n[5] MODEL FEATURE → MASTER DATASET MAPPING")

        feature_mapping = {

            "device_type":
                "Device",

            "model":
                "Standardized Model",

            "standardized_model":
                "Standardized Model",

            "provider":
                "Provider",

            "storage":
                "Storage (GB)",

            "storage_type":
                "Storage Type",

            "connectivity":
                "Connectivity",

            # These may not exist in the finalized master.
            "pricing_category":
                None,

            "model_year":
                None,

            "generation":
                None,

            "screen_size":
                None,

            "ram":
                None,

            "clock_speed_ghz":
                None,

            "chipset":
                None,

            "device_age":
                None
        }


        mapping_results = []


        for feature in model_features:

            master_column = feature_mapping.get(feature)

            if master_column is None:

                status = "NOT AVAILABLE"

            elif master_column in master.columns:

                status = "AVAILABLE"

            else:

                status = "MISSING"


            mapping_results.append({

                "Model Feature": feature,

                "Master Column": (
                    master_column
                    if master_column is not None
                    else "—"
                ),

                "Status": status

            })


        mapping_df = pd.DataFrame(
            mapping_results
        )

        print()

        print(
            mapping_df.to_string(
                index=False
            )
        )


    # ============================================================
    # 6. SUMMARY
    # ============================================================

    if model is not None and master is not None:

        available = (
            mapping_df["Status"] == "AVAILABLE"
        ).sum()

        unavailable = (
            mapping_df["Status"] != "AVAILABLE"
        ).sum()

        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)

        print(
            f"Model features: {len(model_features)}"
        )

        print(
            f"Features directly available in master: {available}"
        )

        print(
            f"Features NOT available in master: {unavailable}"
        )


        print("\nFeatures requiring another source / derivation:")

        for feature in mapping_df.loc[
            mapping_df["Status"] != "AVAILABLE",
            "Model Feature"
        ]:

            print(f"- {feature}")


    # ============================================================
    # 7. IMPORTANT INTERPRETATION
    # ============================================================

    print("\n" + "=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    print("""
    The finalized master dataset is the dashboard's market/reference dataset.

    The ML model was trained using a richer feature dataset.

    Therefore:

        MASTER CSV
            ↓
        UI / market configuration / observed prices

        model.pkl
            ↓
        ML valuation using its required training features

    The master CSV does NOT need to contain every feature used by model.pkl.

    A feature being unavailable in the master CSV is therefore
    NOT automatically a data-quality failure.

    The next step is to determine which model features can be
    derived from the selected device/model and which require
    the original ML training dataset.
    """)


    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    section("FINAL VALIDATION SUMMARY")


    results_df = pd.DataFrame(results)


    total_tests = len(results_df)

    passed_tests = (
        results_df["Status"] == "PASS"
    ).sum()

    failed_tests = (
        results_df["Status"] == "FAIL"
    ).sum()


    print()
    print(f"Total tests : {total_tests}")
    print(f"Passed      : {passed_tests}")
    print(f"Failed      : {failed_tests}")
    print()


    if failed_tests == 0:

        print("=" * 80)
        print("OVERALL RESULT: PASS")
        print("=" * 80)
        print()
        print(
            "The finalized dataset and model passed all "
            "automated validation checks."
        )

    else:

        print("=" * 80)
        print("OVERALL RESULT: FAIL")
        print("=" * 80)
        print()

        print("FAILED TESTS:")
        print()

        print(
            results_df[
                results_df["Status"] == "FAIL"
            ].to_string(index=False)
        )


    # =========================================================================
    # DEVICE SUMMARY
    # =========================================================================

    section("DATASET SUMMARY")


    summary = (
        df.groupby("Device")
        .agg(
            Rows=("Device", "size"),
            Models=("Standardized Model", "nunique")
        )
        .reset_index()
    )


    print(
        summary.to_string(index=False)
    )


    # =========================================================================
    # OPTIONAL: SAVE VALIDATION REPORT
    # =========================================================================

    report_path = (
        BASE_DIR / "dashboard_validation_report.csv"
    )


    results_df.to_csv(
        report_path,
        index=False
    )


    print()
    print(
        f"Validation report saved to:\n{report_path}"
    )
