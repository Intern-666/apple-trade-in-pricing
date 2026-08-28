"""
depreciation_curve_fit.py (v3)

Fits trade-in retention decay curves with a THREE-TIER fallback library:

    Tier 1: Sub-device x Provider  (most specific)
    Tier 2: Device x Provider      (fallback)
    Tier 3: Device only, pooled    (final fallback)

IMPORTANT:
    Tier 2 and Tier 3 curves are fitted and stored independently even when
    no current item happens to require them. This ensures the resulting
    fitted_curves.csv is a reusable curve library for future/unseen
    Sub-device or Provider combinations.

GATE (applied at every tier):
    - n_rows           >= MIN_ROWS  (8)
    - n_distinct_ages  >= MIN_AGES (4)
    - R^2               >= MIN_R2  (0.70)

If a tier fails the gate, that tier is NOT available for fallback.

OUTPUTS:
    fitted_curves.csv
        One row per valid/reusable fitted curve.

    predictions_sample.csv
        Predicted retention at age 0-14 for every stored curve.

    unresolved_groups.csv
        Current (Device, Sub-device, Provider) combinations that fail
        all three tiers.
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

REFERENCE_YEAR = 2026

MIN_ROWS = 8
MIN_AGES = 4
MIN_R2 = 0.70

RETENTION_FLOOR = 0.01
A_CEILING = 1.5


# ------------------------------------------------------------------
# Candidate decay curve forms
# ------------------------------------------------------------------

def exponential(age, a, b):
    return a * np.exp(-b * age)


def power_law(age, a, b):
    return a * np.power(age + 1, -b)


def hyperbolic(age, a, b):
    return a / (1 + b * age)


CANDIDATE_FORMS = {
    "exponential": (exponential, [1.0, 0.2]),
    "power_law": (power_law, [1.0, 0.5]),
    "hyperbolic": (hyperbolic, [1.0, 0.3]),
}


# ------------------------------------------------------------------
# Curve fitting
# ------------------------------------------------------------------

def fit_best_curve(ages: np.ndarray, retentions: np.ndarray):
    """
    Try every candidate curve.

    The winning curve is the one with the lowest RMSE.

    'a_hit_ceiling' is retained as a diagnostic flag. A curve is NOT
    rejected solely because a reached the 1.5 ceiling.
    """
    best = None

    for form_name, (func, p0) in CANDIDATE_FORMS.items():
        try:
            params, _ = curve_fit(
                func,
                ages,
                retentions,
                p0=p0,
                maxfev=5000,
                bounds=([0, 0], [A_CEILING, 10.0]),
            )

            preds = func(ages, *params)

            residuals = retentions - preds

            rmse = np.sqrt(np.mean(residuals ** 2))

            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((retentions - np.mean(retentions)) ** 2)

            r2 = (
                1 - ss_res / ss_tot
                if ss_tot > 0
                else float("nan")
            )

            candidate = {
                "form": form_name,
                "a": params[0],
                "b": params[1],
                "rmse": rmse,
                "r2": r2,
            }

            if best is None or rmse < best["rmse"]:
                best = candidate

        except (RuntimeError, ValueError):
            continue

    if best is not None:
        best["a_hit_ceiling"] = bool(
            np.isclose(best["a"], A_CEILING, atol=1e-4)
        )

    return best


def passes_gate(n_rows: int, n_ages: int, r2) -> bool:
    if n_rows < MIN_ROWS:
        return False

    if n_ages < MIN_AGES:
        return False

    if r2 is None:
        return False

    if pd.isna(r2):
        return False

    return r2 >= MIN_R2


def attempt_fit(group_df: pd.DataFrame):
    """
    Attempt a curve fit for one group.

    Always returns a result dictionary so the caller can inspect why
    the group passed or failed.
    """
    ages = group_df["Age"].values.astype(float)
    retentions = group_df["Retention"].values.astype(float)

    n_rows = len(ages)
    n_ages = len(np.unique(ages))

    if n_rows < MIN_ROWS or n_ages < MIN_AGES:
        return {
            "n_rows": n_rows,
            "n_distinct_ages": n_ages,
            "form": None,
            "a": None,
            "b": None,
            "rmse": None,
            "r2": None,
            "a_hit_ceiling": None,
            "passed_gate": False,
        }

    best = fit_best_curve(ages, retentions)

    if best is None:
        return {
            "n_rows": n_rows,
            "n_distinct_ages": n_ages,
            "form": None,
            "a": None,
            "b": None,
            "rmse": None,
            "r2": None,
            "a_hit_ceiling": None,
            "passed_gate": False,
        }

    r2 = best["r2"]

    result = {
        "n_rows": n_rows,
        "n_distinct_ages": n_ages,
        "form": best["form"],
        "a": round(best["a"], 5),
        "b": round(best["b"], 5),
        "rmse": round(best["rmse"], 5),
        "r2": round(r2, 5) if not np.isnan(r2) else None,
        "a_hit_ceiling": best["a_hit_ceiling"],
    }

    result["passed_gate"] = passes_gate(
        n_rows,
        n_ages,
        result["r2"],
    )

    return result


# ------------------------------------------------------------------
# Helpers for curve-library rows
# ------------------------------------------------------------------

def curve_row(
    device,
    subdevice,
    provider,
    tier,
    fit_group,
    result,
):
    """
    Convert a successful fit result into a fitted_curves row.
    """
    return {
        "Device": device,
        "Sub-device": subdevice,
        "Provider": provider,
        "Tier": tier,
        "FitGroup": fit_group,
        "n_rows": result["n_rows"],
        "n_distinct_ages": result["n_distinct_ages"],
        "form": result["form"],
        "a": result["a"],
        "b": result["b"],
        "rmse": result["rmse"],
        "r2": result["r2"],
        "a_hit_ceiling": result["a_hit_ceiling"],
    }


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def main(csv_path: str):

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    raw_len = len(df)

    required_columns = {
        "Device",
        "Sub-device",
        "Provider",
        "MSRP",
        "Max. Trade-In Value (RM)",
        "Model_Year",
    }

    missing = required_columns - set(df.columns)

    if missing:
        print(f"Input CSV is missing required columns: {sorted(missing)}")
        sys.exit(1)

    # --------------------------------------------------------------
    # Clean input
    # --------------------------------------------------------------

    df = df.dropna(
        subset=[
            "MSRP",
            "Max. Trade-In Value (RM)",
            "Model_Year",
        ]
    ).copy()

    df = df[df["MSRP"] > 0]

    df["Age"] = REFERENCE_YEAR - df["Model_Year"]

    df = df[df["Age"] >= 0]

    df["Retention"] = (
        df["Max. Trade-In Value (RM)"] / df["MSRP"]
    )

    df = df[
        (df["Retention"] > 0)
        & (df["Retention"] <= 1.5)
    ]

    print("=" * 70)
    print("DEPRECIATION CURVE FITTING")
    print("=" * 70)

    print(
        f"Usable rows after cleaning: {len(df)} "
        f"(out of {raw_len} total)"
    )

    print("-" * 70)

    # --------------------------------------------------------------
    # Build reusable Tier 2 library
    #
    # One curve for every valid Device x Provider combination.
    # --------------------------------------------------------------

    tier2_curves = {}

    tier2_results = {}

    device_provider_groups = (
        df[["Device", "Provider"]]
        .drop_duplicates()
        .sort_values(["Device", "Provider"])
    )

    for _, item in device_provider_groups.iterrows():

        device = item["Device"]
        provider = item["Provider"]

        group_df = df[
            (df["Device"] == device)
            & (df["Provider"] == provider)
        ]

        result = attempt_fit(group_df)

        tier2_results[(device, provider)] = result

        if result["passed_gate"]:

            fit_group = (
                f"{device} | (device-level) | {provider}"
            )

            tier2_curves[(device, provider)] = curve_row(
                device=device,
                subdevice="(device-level)",
                provider=provider,
                tier="2_device_provider",
                fit_group=fit_group,
                result=result,
            )

    # --------------------------------------------------------------
    # Build reusable Tier 3 library
    #
    # One curve for every valid Device pooled across providers.
    # --------------------------------------------------------------

    tier3_curves = {}

    tier3_results = {}

    device_groups = (
        df[["Device"]]
        .drop_duplicates()
        .sort_values(["Device"])
    )

    for _, item in device_groups.iterrows():

        device = item["Device"]

        group_df = df[df["Device"] == device]

        result = attempt_fit(group_df)

        tier3_results[device] = result

        if result["passed_gate"]:

            fit_group = (
                f"{device} | (pooled, all providers)"
            )

            tier3_curves[device] = curve_row(
                device=device,
                subdevice="(pooled)",
                provider="(pooled)",
                tier="3_device_only",
                fit_group=fit_group,
                result=result,
            )

    # --------------------------------------------------------------
    # Build Tier 1 library
    #
    # One curve for every valid Sub-device x Provider combination.
    # --------------------------------------------------------------

    tier1_curves = {}

    tier1_results = {}

    subdevice_provider_groups = (
        df[
            [
                "Device",
                "Sub-device",
                "Provider",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Device",
                "Sub-device",
                "Provider",
            ]
        )
    )

    for _, item in subdevice_provider_groups.iterrows():

        device = item["Device"]
        subdevice = item["Sub-device"]
        provider = item["Provider"]

        group_df = df[
            (df["Device"] == device)
            & (df["Sub-device"] == subdevice)
            & (df["Provider"] == provider)
        ]

        result = attempt_fit(group_df)

        key = (
            device,
            subdevice,
            provider,
        )

        tier1_results[key] = result

        if result["passed_gate"]:

            fit_group = (
                f"{device} | {subdevice} | {provider}"
            )

            tier1_curves[key] = curve_row(
                device=device,
                subdevice=subdevice,
                provider=provider,
                tier="1_subdevice_provider",
                fit_group=fit_group,
                result=result,
            )

    # --------------------------------------------------------------
    # Create complete reusable curve library
    #
    # IMPORTANT:
    # Tier 2 and Tier 3 are included even when no current item
    # needed them.
    # --------------------------------------------------------------

    all_curves = []

    all_curves.extend(tier1_curves.values())
    all_curves.extend(tier2_curves.values())
    all_curves.extend(tier3_curves.values())

    results_df = pd.DataFrame(all_curves)

    if not results_df.empty:
        results_df = (
            results_df
            .drop_duplicates(subset=["FitGroup"])
            .sort_values(
                [
                    "Device",
                    "Tier",
                    "Provider",
                    "Sub-device",
                ],
                na_position="last",
            )
            .reset_index(drop=True)
        )

    out_path = os.path.join(
        os.path.dirname(csv_path) or ".",
        "fitted_curves.csv",
    )

    results_df.to_csv(
        out_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Determine current unresolved combinations
    #
    # Strict fallback:
    #
    # Tier 1 → Tier 2 → Tier 3 → unresolved
    #
    # NO cross-Sub-device borrowing.
    # --------------------------------------------------------------

    unresolved = []

    current_items = (
        df[
            [
                "Device",
                "Sub-device",
                "Provider",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Device",
                "Sub-device",
                "Provider",
            ]
        )
    )

    for _, item in current_items.iterrows():

        device = item["Device"]
        subdevice = item["Sub-device"]
        provider = item["Provider"]

        tier1_result = tier1_results[
            (device, subdevice, provider)
        ]

        tier2_result = tier2_results[
            (device, provider)
        ]

        tier3_result = tier3_results[
            device
        ]

        # Tier 1 exists and passed.
        if (device, subdevice, provider) in tier1_curves:
            continue

        # Tier 2 exists and passed.
        if (device, provider) in tier2_curves:
            continue

        # Tier 3 exists and passed.
        if device in tier3_curves:
            continue

        # Nothing passed.
        unresolved.append({
            "Device": device,
            "Sub-device": subdevice,
            "Provider": provider,

            "tier1_rows": tier1_result["n_rows"],
            "tier1_ages": tier1_result["n_distinct_ages"],
            "tier1_r2": tier1_result["r2"],

            "tier2_rows": tier2_result["n_rows"],
            "tier2_ages": tier2_result["n_distinct_ages"],
            "tier2_r2": tier2_result["r2"],

            "tier3_rows": tier3_result["n_rows"],
            "tier3_ages": tier3_result["n_distinct_ages"],
            "tier3_r2": tier3_result["r2"],
        })

    unresolved_df = pd.DataFrame(unresolved)

    if not unresolved_df.empty:
        unresolved_df = unresolved_df.drop_duplicates(
            subset=[
                "Device",
                "Sub-device",
                "Provider",
            ]
        )

    unresolved_path = os.path.join(
        os.path.dirname(csv_path) or ".",
        "unresolved_groups.csv",
    )

    unresolved_df.to_csv(
        unresolved_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Console diagnostics
    # --------------------------------------------------------------

    print(
        f"Saved reusable fitted curve library to: {out_path}"
    )

    print(
        f"Saved unresolved groups to: {unresolved_path}"
    )

    print("-" * 70)

    print("Reusable curves by tier:")

    if results_df.empty:
        print("No curves passed the gate.")
    else:
        tier_counts = results_df["Tier"].value_counts()
        print(tier_counts.to_string())

    print("-" * 70)

    print("Current groups resolved by fallback tier:")

    resolved_tier_counts = {
        "1_subdevice_provider": 0,
        "2_device_provider": 0,
        "3_device_only": 0,
        "unresolved": 0,
    }

    for _, item in current_items.iterrows():

        device = item["Device"]
        subdevice = item["Sub-device"]
        provider = item["Provider"]

        if (device, subdevice, provider) in tier1_curves:
            resolved_tier_counts[
                "1_subdevice_provider"
            ] += 1

        elif (device, provider) in tier2_curves:
            resolved_tier_counts[
                "2_device_provider"
            ] += 1

        elif device in tier3_curves:
            resolved_tier_counts[
                "3_device_only"
            ] += 1

        else:
            resolved_tier_counts["unresolved"] += 1

    print(
        pd.Series(resolved_tier_counts).to_string()
    )

    print("-" * 70)

    print("Sample of reusable fitted curves:")

    if not results_df.empty:
        print(
            results_df.head(20).to_string(
                index=False
            )
        )

    print("-" * 70)

    print(
        "Current unresolved items "
        f"(failed all 3 tiers): {len(unresolved_df)}"
    )

    if len(unresolved_df) > 0:
        print(
            unresolved_df.head(10).to_string(
                index=False
            )
        )

    print("-" * 70)

    # --------------------------------------------------------------
    # Sample predictions age 0-14
    # --------------------------------------------------------------

    pred_rows = []

    for _, row in results_df.iterrows():

        for age in range(0, 15):

            retention = predict_retention(
                row["form"],
                row["a"],
                row["b"],
                age,
            )

            pred_rows.append({
                "FitGroup": row["FitGroup"],
                "Tier": row["Tier"],
                "form": row["form"],
                "age": age,
                "predicted_retention": round(
                    retention,
                    4,
                ),
            })

    pred_df = pd.DataFrame(pred_rows)

    pred_out_path = os.path.join(
        os.path.dirname(csv_path) or ".",
        "predictions_sample.csv",
    )

    pred_df.to_csv(
        pred_out_path,
        index=False,
    )

    print(
        f"Saved sample predictions to: {pred_out_path}"
    )

    print("=" * 70)


def predict_retention(
    form: str,
    a: float,
    b: float,
    age: float,
) -> float:

    func = CANDIDATE_FORMS[form][0]

    val = float(
        func(
            np.array([age]),
            a,
            b,
        )[0]
    )

    return max(
        val,
        RETENTION_FLOOR,
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python depreciation_curve_fit.py "
            "/path/to/your_file.csv"
        )
        sys.exit(1)

    main(sys.argv[1])