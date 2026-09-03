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
    - n_rows           >= MIN_ROWS         (8)
    - n_distinct_ages  >= MIN_AGES         (4)
    - relative R^2     >= MIN_RELATIVE_R2  (0.70)

  NOTE: "relative R^2" here is NOT ordinary (absolute-error) R^2.
  Fitting and form-selection both optimize RELATIVE (percentage)
  error across each device's observed age range -- see
  compute_relative_sigma() and relative_weighted_rmse() -- so the
  quality gate checks the matching relative-error notion of R^2
  (see relative_r2()), not the ordinary one. A curve can have a
  poor or even negative ordinary R^2 while still being an
  excellent, gate-passing RELATIVE fit, because ordinary R^2 is
  dominated by absolute error at high-retention young ages in a
  way that doesn't reflect proportional accuracy across a
  device's whole lifetime. Ordinary R^2 is still computed and
  stored in fitted_curves.csv for reference, but no longer used
  to accept or reject a curve.

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

# Threshold for relative_r2() (see its docstring) -- NOT ordinary
# R^2. Kept at the same numeric value as the original MIN_R2 since
# relative_r2 is constructed on the same 0-1-ish "fraction of
# variation explained" scale, so the same threshold remains a
# reasonable bar for "this curve explains most of the proportional
# variation in retention across the device's observed ages".
MIN_RELATIVE_R2 = 0.70

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

def compute_relative_sigma(
    ages: np.ndarray,
    retentions: np.ndarray,
) -> np.ndarray:
    """
    Builds the per-row sigma (uncertainty) array passed to
    curve_fit(), engineered so the optimizer minimizes RELATIVE
    (percentage) error rather than absolute error.

    WHY THIS EXISTS:

    Retention values span roughly two orders of magnitude across a
    device's lifetime (e.g. ~0.38 at age 2 down to ~0.05 at age 8).
    Plain (unweighted) curve_fit minimizes ABSOLUTE squared error,
    so a large absolute error at a high-retention young age (e.g.
    0.09 off at age 2, where retention is ~0.38) always outweighs
    a much SMALLER absolute error at an old age (e.g. 0.04 off at
    age 8, where retention is only ~0.05) -- even though that small
    absolute error represents the curve being wrong by a factor of
    several times over in relative terms. This is exactly why the
    original fitter chose curves that decayed unrealistically fast:
    it was optimizing for the wrong notion of "close".

    curve_fit's `sigma` parameter is the per-point measurement
    uncertainty; a point's residual is divided by its sigma before
    being squared and summed, so points with LARGER sigma count
    for LESS in the fit. Setting sigma proportional to each point's
    retention value converts absolute-error minimization into
    (approximately) relative-error minimization: a point with twice
    the retention value gets twice the sigma, so the same relative
    error contributes the same amount regardless of the point's
    absolute scale.

    Rows are also weighted by 1 / (count of rows sharing that age),
    so an age with many rows (e.g. 14 rows at age 2) doesn't get 14x
    the influence of an age with only 1 row (e.g. age 8) purely by
    row count -- each observed AGE gets comparable influence, then
    within that, relative rather than absolute error is minimized.
    """

    unique_ages, counts = np.unique(ages, return_counts=True)

    age_to_count = dict(zip(unique_ages, counts))

    per_row_age_weight = np.array([
        1.0 / age_to_count[age]
        for age in ages
    ])

    # sigma proportional to the retention value itself (relative
    # weighting), scaled down further for ages with many rows so
    # every age contributes comparably regardless of row count.
    # A small floor avoids division-by-zero for near-zero
    # retention values.
    retention_floor = 1e-4

    sigma = (
        np.maximum(retentions, retention_floor)
        / np.sqrt(per_row_age_weight)
    )

    return sigma


def relative_weighted_rmse(
    ages: np.ndarray,
    retentions: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """
    Selection metric used to choose between candidate curve forms.

    Computes, for each distinct observed AGE, the relative
    (percentage) error between that age's mean actual retention and
    the curve's predicted retention there -- then RMS-combines those
    per-age relative errors, with every age weighted equally
    regardless of how many raw rows happen to sit at that age.

    This is the relative-error counterpart to plain RMSE: it's what
    actually decides which of exponential / power_law / hyperbolic
    "wins" for a given group, so the winner is the form that stays
    proportionally accurate across the device's whole observed
    lifetime -- not just the form that nails the best-populated
    young ages in absolute terms.
    """

    unique_ages = np.unique(ages)

    per_age_relative_sq_errors = []

    for age in unique_ages:

        age_mask = ages == age

        actual = retentions[age_mask].mean()

        predicted = predictions[age_mask].mean()

        if actual <= 0:
            continue

        relative_error = (predicted - actual) / actual

        per_age_relative_sq_errors.append(
            relative_error ** 2
        )

    if not per_age_relative_sq_errors:
        return float("nan")

    return np.sqrt(
        np.mean(per_age_relative_sq_errors)
    )


def relative_r2(
    ages: np.ndarray,
    retentions: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """
    Quality-gate metric, consistent with the relative-error fitting
    and selection objective used above -- this is NOT the same
    quantity as ordinary (absolute-error) R^2, and is named
    differently on purpose so it is never mistaken for one.

    Ordinary R^2 asks: "how much better is this curve than just
    predicting the mean, in ABSOLUTE squared-error terms?" That
    question is the wrong one to gate on once the fitting objective
    itself is relative error -- a curve can be an excellent
    RELATIVE fit (proportionally accurate at every observed age)
    while scoring a poor or even negative ordinary R^2, precisely
    because ordinary R^2 is dominated by the same high-retention
    young-age rows that caused the original bug.

    relative_r2 asks the analogous question in relative-error
    terms: "how much better is this curve's per-age RELATIVE error
    than the relative error of just predicting the mean retention
    for every age?" Defined as:

        1 - (mean squared relative error of the curve)
          / (mean squared relative error of the per-age mean baseline)

    Like ordinary R^2, a value of 1.0 is a perfect relative fit, 0.0
    means "no better than predicting the mean at every age", and
    negative values mean the curve is relatively WORSE than that
    baseline. This keeps the same intuitive 0-1-ish scale as R^2
    (so the existing MIN_R2 threshold and its meaning of "explains
    most of the variation" still applies), while actually measuring
    the thing the pipeline now optimizes for.
    """

    unique_ages = np.unique(ages)

    per_age_actuals = []
    per_age_predicted_rel_sq_errors = []

    for age in unique_ages:

        age_mask = ages == age

        actual = retentions[age_mask].mean()

        predicted = predictions[age_mask].mean()

        if actual <= 0:
            continue

        per_age_actuals.append(actual)

        rel_error = (predicted - actual) / actual

        per_age_predicted_rel_sq_errors.append(
            rel_error ** 2
        )

    if len(per_age_actuals) < 2:
        return float("nan")

    per_age_actuals = np.array(per_age_actuals)

    overall_mean_actual = per_age_actuals.mean()

    # Baseline: relative error of predicting the SAME mean
    # retention for every age (the relative-error equivalent of
    # ordinary R^2's "predict the mean" baseline).
    baseline_rel_sq_errors = (
        (overall_mean_actual - per_age_actuals)
        / per_age_actuals
    ) ** 2

    ss_res_relative = np.sum(
        per_age_predicted_rel_sq_errors
    )

    ss_tot_relative = np.sum(
        baseline_rel_sq_errors
    )

    if ss_tot_relative <= 0:
        return float("nan")

    return 1 - (
        ss_res_relative / ss_tot_relative
    )


def fit_best_curve(ages: np.ndarray, retentions: np.ndarray):
    """
    Try every candidate curve.

    Each candidate is fitted with RELATIVE-error weighting (see
    compute_relative_sigma()), so curve_fit's own optimization
    prioritizes proportional accuracy across the device's observed
    age range instead of being dominated by absolute-value errors
    at high-retention young ages.

    The winning candidate is the one with the lowest RELATIVE
    weighted RMSE (see relative_weighted_rmse()) -- not plain
    absolute RMSE. This is what actually fixes curves that
    previously decayed unrealistically fast: a curve is no longer
    rewarded for nailing young ages in absolute terms while being
    proportionally very wrong at old ages.

    Plain (unweighted, absolute) RMSE is still computed and stored
    for reference/diagnostics, but no longer used to pick the
    winner.

    'a_hit_ceiling' is retained as a diagnostic flag. A curve is NOT
    rejected solely because a reached the 1.5 ceiling.
    """
    best = None

    sigma = compute_relative_sigma(ages, retentions)

    for form_name, (func, p0) in CANDIDATE_FORMS.items():
        try:
            params, _ = curve_fit(
                func,
                ages,
                retentions,
                p0=p0,
                sigma=sigma,
                absolute_sigma=False,
                maxfev=5000,
                bounds=([0, 0], [A_CEILING, 10.0]),
            )

            preds = func(ages, *params)

            residuals = retentions - preds

            rmse = np.sqrt(np.mean(residuals ** 2))

            rel_weighted_rmse = relative_weighted_rmse(
                ages, retentions, preds
            )

            rel_r2 = relative_r2(
                ages, retentions, preds
            )

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
                "relative_weighted_rmse": rel_weighted_rmse,
                "r2": r2,
                "relative_r2": rel_r2,
            }

            if (
                best is None
                or (
                    not np.isnan(rel_weighted_rmse)
                    and (
                        np.isnan(best["relative_weighted_rmse"])
                        or rel_weighted_rmse
                        < best["relative_weighted_rmse"]
                    )
                )
            ):
                best = candidate

        except (RuntimeError, ValueError):
            continue

    if best is not None:
        best["a_hit_ceiling"] = bool(
            np.isclose(best["a"], A_CEILING, atol=1e-4)
        )

    return best


def passes_gate(n_rows: int, n_ages: int, rel_r2) -> bool:
    if n_rows < MIN_ROWS:
        return False

    if n_ages < MIN_AGES:
        return False

    if rel_r2 is None:
        return False

    if pd.isna(rel_r2):
        return False

    return rel_r2 >= MIN_RELATIVE_R2


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

    # The single highest retention actually observed anywhere in
    # this group's real training data (across all its ages). Stored
    # alongside the curve so predict-time code can clamp a forecast
    # to never exceed what was genuinely observed for this exact
    # group -- e.g. a fitted curve should never predict a device
    # retains MORE of its value than the best real recorded case,
    # regardless of what shape the curve otherwise takes.
    max_observed_retention = (
        float(retentions.max())
        if n_rows > 0
        else None
    )

    if n_rows < MIN_ROWS or n_ages < MIN_AGES:
        return {
            "n_rows": n_rows,
            "n_distinct_ages": n_ages,
            "form": None,
            "a": None,
            "b": None,
            "rmse": None,
            "r2": None,
            "relative_r2": None,
            "a_hit_ceiling": None,
            "max_observed_retention": max_observed_retention,
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
            "relative_r2": None,
            "a_hit_ceiling": None,
            "max_observed_retention": max_observed_retention,
            "passed_gate": False,
        }

    r2 = best["r2"]
    rel_r2 = best["relative_r2"]

    result = {
        "n_rows": n_rows,
        "n_distinct_ages": n_ages,
        "form": best["form"],
        "a": round(best["a"], 5),
        "b": round(best["b"], 5),
        "rmse": round(best["rmse"], 5),
        "r2": round(r2, 5) if not np.isnan(r2) else None,
        "relative_r2": round(rel_r2, 5) if not np.isnan(rel_r2) else None,
        "a_hit_ceiling": best["a_hit_ceiling"],
        "max_observed_retention": round(max_observed_retention, 5),
    }

    result["passed_gate"] = passes_gate(
        n_rows,
        n_ages,
        result["relative_r2"],
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
        "relative_r2": result["relative_r2"],
        "max_observed_retention": result["max_observed_retention"],
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

    # Age 0 is excluded from the FITTING input (not from the raw
    # dataset overall) because app.py's forecast loop never calls
    # the fitted curve at age 0 -- it uses the device's raw MSRP
    # directly for a brand-new device instead. Age-0 rows can only
    # pull the curve's shape around at ages that are never actually
    # used, and age-0 retention in the real data is often noisy or
    # non-monotonic relative to age 1 (a newly-released device's
    # trade-in price can lag or lead its near-term resale value),
    # which otherwise forces every candidate curve to compromise
    # its shape at ages 0-1 for a training point that never affects
    # a real forecast.
    df = df[df["Age"] >= 1]

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