"""
tradein_fallback.py

Reusable fallback trade-in value predictor.

STRICT FALLBACK HIERARCHY:

    Tier 1: exact Sub-device + Provider
        ↓
    Tier 2: exact Device + Provider
        ↓
    Tier 4: new-model analog-lineage forecast
        ↓
    Tier 3: exact Device, pooled across providers
        ↓
    unresolved


IMPORTANT:
    The hierarchy intentionally prioritizes Sub-device analog lineage
    over the broad Device-only pooled curve.

Example:

    iPhone + Pro Max + SomeNewProvider

If there is no:
    iPhone + Pro Max + SomeNewProvider
and no:
    iPhone + SomeNewProvider

the predictor attempts:

    iPhone + Pro Max analog lineage
    (prior Pro Max generations, pooled across providers)

ONLY if that fails does it use:

    iPhone pooled across all providers

This prevents a new Pro Max from immediately receiving a broad
iPhone-level curve when useful Pro Max historical lineage exists.


TIER 1
------------------------------------------------------------------
Exact Sub-device + Provider.

Example:
    iPhone + Pro Max + CompAsia


TIER 2
------------------------------------------------------------------
Exact Device + Provider.

Example:
    iPhone + Mini + CompAsia

If no reliable Mini + CompAsia curve exists, a genuine
Device + Provider curve may be used:

    iPhone + (device-level) + CompAsia

The predictor NEVER substitutes another Sub-device merely because
it shares the same Device and Provider.


TIER 4 — NEW-MODEL ANALOG-LINEAGE FORECASTING
------------------------------------------------------------------
Used when Tiers 1 and 2 cannot provide a curve.

The predictor identifies prior generations of the SAME:

    Device + Sub-device

and pools their raw trade-in observations across providers.

Example:

    Target:
        iPhone 18 Pro Max

    Eligible analog lineage:
        iPhone 17 Pro Max
        iPhone 16 Pro Max
        iPhone 15 Pro Max
        ...

Only generations with Model_Year strictly less than the target
Model_Year are eligible.

Provider is intentionally ignored for Tier 4 because the purpose
is to learn depreciation SHAPE rather than provider-specific LEVEL.

At least MIN_ANALOG_GENERATIONS distinct prior years are required.

At most MAX_ANALOG_GENERATIONS recent prior generations are used.

The pooled data must still pass the same quality gates:

    MIN_ROWS
    MIN_AGES
    MIN_R2

Tier 4 never invents a number.


TIER 3
------------------------------------------------------------------
Broad Device-only pooled curve.

This is deliberately attempted AFTER Tier 4.

Example:

    iPhone + Pro Max + SomeNewProvider

If:
    Tier 1 ❌
    Tier 2 ❌
    Tier 4 ❌

then:
    Tier 3 → iPhone pooled curve


UNRESOLVED
------------------------------------------------------------------
If no suitable Tier 1, Tier 2, Tier 4, or Tier 3 result exists,
the predictor returns an unresolved result.

A caller may then layer an external fallback such as an ML model
on top of this module.
"""

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# ------------------------------------------------------------------
# Global constants
# ------------------------------------------------------------------

REFERENCE_YEAR = 2026

RETENTION_FLOOR = 0.01
RETENTION_CEILING = 1.0

# Historical curve-fitting quality gates.
MIN_ROWS = 8
MIN_AGES = 4
MIN_R2 = 0.70

# Maximum allowed starting retention parameter.
A_CEILING = 1.5

# Tier 4 analog-lineage requirements.
MIN_ANALOG_GENERATIONS = 2
MAX_ANALOG_GENERATIONS = 4

# Retention observations above this value are excluded from fitting.
RETENTION_MAX_FOR_FITTING = 1.5


# ------------------------------------------------------------------
# Curve forms
# ------------------------------------------------------------------

def exponential(age, a, b):
    return a * np.exp(-b * age)


def power_law(age, a, b):
    return a * np.power(age + 1, -b)


def hyperbolic(age, a, b):
    return a / (1 + b * age)


CURVE_FUNCS = {
    "exponential": exponential,
    "power_law": power_law,
    "hyperbolic": hyperbolic,
}


# ------------------------------------------------------------------
# Result object
# ------------------------------------------------------------------

@dataclass
class FallbackResult:

    predicted_value: Optional[float]
    predicted_retention: Optional[float]

    matched_group: Optional[str]
    matched_tier: Optional[str]

    form: Optional[str]

    confidence_flag: Optional[str]

    # Tier 4 only.
    # Contains the Model_Years used for the analog-lineage fit.
    analog_models_used: Optional[list] = None

    def as_dict(self):

        return {
            "predicted_value": self.predicted_value,
            "predicted_retention": self.predicted_retention,
            "matched_group": self.matched_group,
            "matched_tier": self.matched_tier,
            "form": self.form,
            "confidence_flag": self.confidence_flag,
            "analog_models_used": self.analog_models_used,
        }


# ------------------------------------------------------------------
# Predictor
# ------------------------------------------------------------------

class TradeInFallback:
    """
    Loads fitted_curves.csv once and performs hierarchical fallback:

        1. exact Sub-device + Provider
        2. exact Device + Provider
        3. new-model analog lineage
        4. Device-only pooled curve
        5. unresolved

    Tier 4 requires raw historical trade-in data.

    If raw_data_path is omitted, Tier 4 is unavailable and the
    predictor falls back from Tier 2 directly to Tier 3.
    """

    def __init__(
        self,
        fitted_curves_path: str,
        raw_data_path: Optional[str] = None,
    ):

        # ----------------------------------------------------------
        # Load fitted curves
        # ----------------------------------------------------------

        self.curves = pd.read_csv(
            fitted_curves_path
        )

        required_cols = {
            "Device",
            "Sub-device",
            "Provider",
            "Tier",
            "FitGroup",
            "form",
            "a",
            "b",
        }

        missing = (
            required_cols
            - set(self.curves.columns)
        )

        if missing:

            raise ValueError(
                "fitted_curves.csv is missing "
                f"expected columns: {missing}"
            )

        # ----------------------------------------------------------
        # Validate curve tiers
        # ----------------------------------------------------------

        valid_tiers = {
            "1_subdevice_provider",
            "2_device_provider",
            "3_device_only",
        }

        invalid_tiers = set(
            self.curves["Tier"].dropna().unique()
        ) - valid_tiers

        if invalid_tiers:

            raise ValueError(
                "fitted_curves.csv contains "
                f"unknown tiers: {invalid_tiers}"
            )

        # ----------------------------------------------------------
        # Validate curve forms
        # ----------------------------------------------------------

        valid_forms = set(
            CURVE_FUNCS.keys()
        )

        invalid_forms = set(
            self.curves["form"].dropna().unique()
        ) - valid_forms

        if invalid_forms:

            raise ValueError(
                "fitted_curves.csv contains "
                f"unknown curve forms: {invalid_forms}"
            )

        # ----------------------------------------------------------
        # Load raw historical data for Tier 4
        # ----------------------------------------------------------

        self.raw_data = None

        if raw_data_path is not None:

            raw = pd.read_csv(
                raw_data_path
            )

            required_raw_cols = {
                "Device",
                "Sub-device",
                "Model_Year",
                "MSRP",
                "Max. Trade-In Value (RM)",
            }

            missing_raw = (
                required_raw_cols
                - set(raw.columns)
            )

            if missing_raw:

                raise ValueError(
                    "Raw trade-in dataset is missing "
                    f"expected columns: {missing_raw}"
                )

            self.raw_data = raw

    # ==============================================================
    # CURVE LOOKUP — TIERS 1 & 2 ONLY
    # ==============================================================

    def _lookup_specific_curve(
        self,
        device: str,
        sub_device: Optional[str],
        provider: str,
    ):
        """
        Search ONLY Tier 1 and Tier 2.

        Tier 1:
            exact Device + Sub-device + Provider

        Tier 2:
            exact Device + Provider

        IMPORTANT:
            Tier 2 only accepts genuine Tier-2 rows.

        It does NOT search arbitrary rows sharing Device +
        Provider because those could be Tier-1 curves belonging
        to another Sub-device.

        Tier 3 is intentionally NOT searched here.

        Returns:
            pandas Series for the selected curve,
            or None.
        """

        df = self.curves

        # ----------------------------------------------------------
        # Tier 1
        # ----------------------------------------------------------

        if sub_device is not None:

            match = df[
                (df["Device"] == device)
                & (
                    df["Sub-device"]
                    == sub_device
                )
                & (
                    df["Provider"]
                    == provider
                )
                & (
                    df["Tier"]
                    == "1_subdevice_provider"
                )
            ]

            if len(match) > 0:

                return match.iloc[0]

        # ----------------------------------------------------------
        # Tier 2
        # ----------------------------------------------------------

        match = df[
            (df["Device"] == device)
            & (
                df["Provider"]
                == provider
            )
            & (
                df["Tier"]
                == "2_device_provider"
            )
        ]

        if len(match) > 0:

            return match.iloc[0]

        # ----------------------------------------------------------
        # No Tier 1 or Tier 2 curve
        # ----------------------------------------------------------

        return None

    # ==============================================================
    # CURVE LOOKUP — TIER 3 ONLY
    # ==============================================================

    def _lookup_device_curve(
        self,
        device: str,
    ):
        """
        Search ONLY the genuine Device-only Tier 3 curve.

        Provider is intentionally ignored.

        Returns:
            pandas Series or None.
        """

        df = self.curves

        match = df[
            (df["Device"] == device)
            & (
                df["Tier"]
                == "3_device_only"
            )
        ]

        if len(match) > 0:

            return match.iloc[0]

        return None

    # ==============================================================
    # TIER 4 — SELECT ANALOG GENERATIONS
    # ==============================================================

    def _select_analog_generations(
        self,
        device: str,
        sub_device: str,
        model_year: int,
    ):
        """
        Identify eligible prior generations for Tier 4.

        Eligibility:
            - same Device
            - same Sub-device
            - Model_Year < target model_year
            - valid positive trade-in value
            - valid positive MSRP

        Selection:
            - group by distinct Model_Year
            - sort newest first
            - take at most MAX_ANALOG_GENERATIONS
            - require at least MIN_ANALOG_GENERATIONS

        Returns:
            (rows, generation_years)

        where:
            rows = pooled raw observations
            generation_years = sorted selected Model_Years

        Returns:
            (None, None) if insufficient lineage exists.
        """

        if self.raw_data is None:

            return None, None

        raw = self.raw_data

        # ----------------------------------------------------------
        # Find eligible prior-generation observations
        # ----------------------------------------------------------

        candidates = raw[
            (raw["Device"] == device)
            & (
                raw["Sub-device"]
                == sub_device
            )
            & (
                raw["Model_Year"]
                < model_year
            )
            & (
                raw["Max. Trade-In Value (RM)"]
                .notna()
            )
            & (
                raw["Max. Trade-In Value (RM)"]
                > 0
            )
            & (
                raw["MSRP"]
                .notna()
            )
            & (
                raw["MSRP"]
                > 0
            )
        ]

        if candidates.empty:

            return None, None

        # ----------------------------------------------------------
        # Determine distinct available generations
        # ----------------------------------------------------------

        available_years = sorted(
            candidates["Model_Year"]
            .dropna()
            .unique(),
            reverse=True,
        )

        if (
            len(available_years)
            < MIN_ANALOG_GENERATIONS
        ):

            return None, None

        # ----------------------------------------------------------
        # Keep the most recent generations
        # ----------------------------------------------------------

        selected_years = (
            available_years[
                :MAX_ANALOG_GENERATIONS
            ]
        )

        selected_rows = candidates[
            candidates["Model_Year"]
            .isin(selected_years)
        ].copy()

        return (
            selected_rows,
            sorted(selected_years),
        )

    # ==============================================================
    # TIER 4 — FIT POOLED ANALOG CURVE
    # ==============================================================

    def _fit_pooled_curve(
        self,
        rows: pd.DataFrame,
        reference_year: int,
    ):
        """
        Fit a fresh depreciation curve to pooled analog data.

        Each raw observation gets its own historical age:

            age = reference_year - Model_Year

        Retention:

            trade_in / MSRP

        Candidate forms:
            exponential
            power_law
            hyperbolic

        The lowest-RMSE candidate is selected, provided it clears:

            MIN_ROWS
            MIN_AGES
            MIN_R2

        Returns:

            (
                form,
                a,
                b,
                r2,
                n_rows,
                n_ages,
                a_hit_ceiling
            )

        or None if no valid curve clears the quality gates.
        """

        # ----------------------------------------------------------
        # Compute historical ages
        # ----------------------------------------------------------

        ages = (
            reference_year
            - rows["Model_Year"].to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------------
        # Compute retention
        # ----------------------------------------------------------

        retention = (
            rows[
                "Max. Trade-In Value (RM)"
            ].to_numpy(
                dtype=float
            )
            /
            rows[
                "MSRP"
            ].to_numpy(
                dtype=float
            )
        )

        # ----------------------------------------------------------
        # Basic validity filter
        # ----------------------------------------------------------

        valid = (
            (retention > 0)
            & (
                retention
                <= RETENTION_MAX_FOR_FITTING
            )
            & (ages >= 0)
        )

        ages = ages[valid]
        retention = retention[valid]

        n_rows = len(ages)

        n_ages = len(
            np.unique(ages)
        )

        # ----------------------------------------------------------
        # Minimum data gates
        # ----------------------------------------------------------

        if (
            n_rows < MIN_ROWS
            or n_ages < MIN_AGES
        ):

            return None

        # ----------------------------------------------------------
        # Fit candidate forms
        # ----------------------------------------------------------

        best = None

        for (
            form_name,
            func,
        ) in CURVE_FUNCS.items():

            try:

                params, _ = curve_fit(
                    func,
                    ages,
                    retention,
                    p0=[1.0, 0.3],
                    bounds=(
                        [0, 0],
                        [
                            A_CEILING,
                            np.inf,
                        ],
                    ),
                    maxfev=10000,
                )

                a, b = params

                predicted = func(
                    ages,
                    a,
                    b,
                )

                r2 = r2_score(
                    retention,
                    predicted,
                )

                rmse = float(
                    np.sqrt(
                        np.mean(
                            (
                                retention
                                - predicted
                            ) ** 2
                        )
                    )
                )

            except (
                RuntimeError,
                ValueError,
            ):

                continue

            # ------------------------------------------------------
            # Keep lowest-RMSE candidate
            # ------------------------------------------------------

            if (
                best is None
                or rmse < best["rmse"]
            ):

                best = {
                    "form": form_name,
                    "a": float(a),
                    "b": float(b),
                    "r2": float(r2),
                    "rmse": rmse,
                    "a_hit_ceiling": bool(
                        np.isclose(
                            a,
                            A_CEILING,
                        )
                    ),
                }

        if best is None:

            return None

        # ----------------------------------------------------------
        # R² quality gate
        # ----------------------------------------------------------

        if best["r2"] < MIN_R2:

            return None

        return (
            best["form"],
            best["a"],
            best["b"],
            best["r2"],
            n_rows,
            n_ages,
            best["a_hit_ceiling"],
        )

    # ==============================================================
    # TIER 4 — PREDICT FROM ANALOG LINEAGE
    # ==============================================================

    def _predict_tier4(
        self,
        device: str,
        sub_device: Optional[str],
        msrp: float,
        model_year: int,
        age: int,
        reference_year: int,
    ) -> Optional[FallbackResult]:
        """
        Attempt Tier 4 analog-lineage forecasting.

        Returns:
            None if Tier 4 cannot be attempted.

        Otherwise returns a FallbackResult describing either:

            successful prediction

        or:

            why Tier 4 failed.
        """

        # ----------------------------------------------------------
        # Tier 4 requires raw data and Sub-device
        # ----------------------------------------------------------

        if (
            self.raw_data is None
            or sub_device is None
        ):

            return None

        # ----------------------------------------------------------
        # Select prior generations
        # ----------------------------------------------------------

        (
            rows,
            generation_years,
        ) = self._select_analog_generations(
            device=device,
            sub_device=sub_device,
            model_year=model_year,
        )

        if rows is None:

            return FallbackResult(
                predicted_value=None,
                predicted_retention=None,
                matched_group=None,
                matched_tier=None,
                form=None,
                confidence_flag=(
                    "Tier 4 not available: fewer than "
                    f"{MIN_ANALOG_GENERATIONS} prior generations "
                    f"of '{device} {sub_device}' with valid "
                    "trade-in history were found."
                ),
                analog_models_used=None,
            )

        # ----------------------------------------------------------
        # Fit pooled analog curve
        # ----------------------------------------------------------

        fit = self._fit_pooled_curve(
            rows=rows,
            reference_year=REFERENCE_YEAR,
        )

        if fit is None:

            return FallbackResult(
                predicted_value=None,
                predicted_retention=None,
                matched_group=None,
                matched_tier=None,
                form=None,
                confidence_flag=(
                    "Tier 4 attempted using analog generations "
                    f"{generation_years} of "
                    f"'{device} {sub_device}', "
                    "but the pooled data did not clear curve "
                    f"quality gates "
                    f"(MIN_ROWS={MIN_ROWS}, "
                    f"MIN_AGES={MIN_AGES}, "
                    f"MIN_R2={MIN_R2})."
                ),
                analog_models_used=generation_years,
            )

        (
            form,
            a,
            b,
            r2,
            n_rows,
            n_ages,
            a_hit_ceiling,
        ) = fit

        # ----------------------------------------------------------
        # Predict target retention
        # ----------------------------------------------------------

        func = CURVE_FUNCS[form]

        raw_retention = float(
            func(
                np.array([age]),
                a,
                b,
            )[0]
        )

        # ----------------------------------------------------------
        # Business constraints
        # ----------------------------------------------------------

        retention = min(
            max(
                raw_retention,
                RETENTION_FLOOR,
            ),
            RETENTION_CEILING,
        )

        predicted_value = round(
            retention
            * float(msrp),
            2,
        )

        # ----------------------------------------------------------
        # Diagnostic flags
        # ----------------------------------------------------------

        flags = [
            (
                f"No historical curve exists for "
                f"'{device} {sub_device}'; "
                "used a new-model analog-lineage forecast "
                "pooled from "
                f"{len(generation_years)} prior generation(s) "
                f"({generation_years}), "
                f"fitted={form}, "
                f"R2={r2:.3f}."
            )
        ]

        if a_hit_ceiling:

            flags.append(
                "Analog curve's 'a' parameter hit the "
                f"{A_CEILING} ceiling; review this forecast."
            )

        if (
            raw_retention
            > RETENTION_CEILING
        ):

            flags.append(
                f"Raw curve predicted "
                f"{raw_retention:.3f} retention; "
                "capped at 1.000 because trade-in value "
                "cannot exceed MSRP."
            )

        return FallbackResult(
            predicted_value=predicted_value,
            predicted_retention=round(
                retention,
                5,
            ),
            matched_group=(
                f"{device} | "
                f"{sub_device} | "
                "analog_lineage"
            ),
            matched_tier="4_analog_lineage",
            form=form,
            confidence_flag=" ".join(
                flags
            ),
            analog_models_used=(
                generation_years
            ),
        )

    # ==============================================================
    # CURVE EVALUATION
    # ==============================================================

    def _predict_from_curve(
        self,
        curve_row,
        msrp: float,
        age: int,
        sub_device: Optional[str],
    ) -> FallbackResult:
        """
        Evaluate an existing fitted curve and produce a prediction.

        Used by Tiers 1-3.
        """

        # ----------------------------------------------------------
        # Read curve parameters
        # ----------------------------------------------------------

        form = curve_row["form"]

        a = float(
            curve_row["a"]
        )

        b = float(
            curve_row["b"]
        )

        func = CURVE_FUNCS[form]

        # ----------------------------------------------------------
        # Calculate raw retention
        # ----------------------------------------------------------

        raw_retention = float(
            func(
                np.array([age]),
                a,
                b,
            )[0]
        )

        # ----------------------------------------------------------
        # Apply retention floor / ceiling
        # ----------------------------------------------------------

        retention = min(
            max(
                raw_retention,
                RETENTION_FLOOR,
            ),
            RETENTION_CEILING,
        )

        predicted_value = round(
            retention
            * float(msrp),
            2,
        )

        # ----------------------------------------------------------
        # Diagnostic flags
        # ----------------------------------------------------------

        flags = []

        matched_tier = curve_row["Tier"]

        # ----------------------------------------------------------
        # Tier 2 diagnostic
        # ----------------------------------------------------------

        if (
            sub_device is not None
            and matched_tier
            == "2_device_provider"
        ):

            flags.append(
                f"No reliable Sub-device-level curve "
                f"for '{sub_device}'; "
                "fell back to Device + Provider."
            )

        # ----------------------------------------------------------
        # Tier 3 diagnostic
        # ----------------------------------------------------------

        if (
            matched_tier
            == "3_device_only"
        ):

            if sub_device is not None:

                flags.append(
                    f"No reliable Sub-device + Provider "
                    f"curve for '{sub_device}', "
                    "no reliable Device + Provider curve, "
                    "and no usable analog-lineage forecast; "
                    "fell back to pooled Device curve."
                )

            else:

                flags.append(
                    "Used pooled Device-only curve."
                )

        # ----------------------------------------------------------
        # a-ceiling diagnostic
        # ----------------------------------------------------------

        if (
            "a_hit_ceiling"
            in curve_row.index
            and bool(
                curve_row[
                    "a_hit_ceiling"
                ]
            )
        ):

            flags.append(
                "Source curve's 'a' parameter hit the "
                f"{A_CEILING} ceiling; "
                "review this curve if needed."
            )

        # ----------------------------------------------------------
        # MSRP cap diagnostic
        # ----------------------------------------------------------

        if (
            raw_retention
            > RETENTION_CEILING
        ):

            flags.append(
                f"Raw curve predicted "
                f"{raw_retention:.3f} retention; "
                "capped at 1.000 because trade-in value "
                "cannot exceed MSRP."
            )

        confidence_flag = (
            " ".join(flags)
            if flags
            else None
        )

        return FallbackResult(
            predicted_value=predicted_value,
            predicted_retention=round(
                retention,
                5,
            ),
            matched_group=curve_row[
                "FitGroup"
            ],
            matched_tier=matched_tier,
            form=form,
            confidence_flag=confidence_flag,
            analog_models_used=None,
        )

    # ==============================================================
    # PREDICTION
    # ==============================================================

    def predict(
        self,
        device: str,
        provider: str,
        msrp: float,
        model_year: int,
        sub_device: Optional[str] = None,
        reference_year: int = REFERENCE_YEAR,
    ) -> FallbackResult:

        # ==========================================================
        # VALIDATE MODEL YEAR
        # ==========================================================

        age = (
            reference_year
            - model_year
        )

        if age < 0:

            return FallbackResult(
                predicted_value=None,
                predicted_retention=None,
                matched_group=None,
                matched_tier=None,
                form=None,
                confidence_flag=(
                    f"model_year {model_year} "
                    "is in the future relative to "
                    f"reference_year {reference_year}"
                ),
            )

        # ==========================================================
        # VALIDATE MSRP
        # ==========================================================

        if (
            msrp is None
            or msrp <= 0
        ):

            return FallbackResult(
                predicted_value=None,
                predicted_retention=None,
                matched_group=None,
                matched_tier=None,
                form=None,
                confidence_flag=(
                    f"Invalid MSRP: {msrp}. "
                    "MSRP must be greater than zero."
                ),
            )

        # ==========================================================
        # TIER 1 + TIER 2
        # ==========================================================

        curve_row = (
            self._lookup_specific_curve(
                device=device,
                sub_device=sub_device,
                provider=provider,
            )
        )

        if curve_row is not None:

            return self._predict_from_curve(
                curve_row=curve_row,
                msrp=msrp,
                age=age,
                sub_device=sub_device,
            )

        # ==========================================================
        # TIER 4
        #
        # This MUST happen before Tier 3.
        # ==========================================================

        tier4_result = (
            self._predict_tier4(
                device=device,
                sub_device=sub_device,
                msrp=msrp,
                model_year=model_year,
                age=age,
                reference_year=reference_year,
            )
        )

        if (
            tier4_result is not None
            and tier4_result.predicted_value
            is not None
        ):

            return tier4_result

        # ==========================================================
        # TIER 3
        #
        # Only reached after Tier 4 fails or is unavailable.
        # ==========================================================

        device_curve = (
            self._lookup_device_curve(
                device=device,
            )
        )

        if device_curve is not None:

            result = self._predict_from_curve(
                curve_row=device_curve,
                msrp=msrp,
                age=age,
                sub_device=sub_device,
            )

            # If Tier 4 was attempted and failed, make the
            # Tier 3 diagnostic explicit.

            if (
                tier4_result is not None
                and tier4_result.confidence_flag
                is not None
            ):

                tier4_message = (
                    tier4_result.confidence_flag
                )

                if result.confidence_flag:

                    result.confidence_flag = (
                        tier4_message
                        + " "
                        + result.confidence_flag
                    )

                else:

                    result.confidence_flag = (
                        tier4_message
                    )

            return result

        # ==========================================================
        # UNRESOLVED
        # ==========================================================

        # If Tier 4 was attempted, preserve its diagnostic.
        if (
            tier4_result is not None
            and tier4_result.confidence_flag
            is not None
        ):

            return FallbackResult(
                predicted_value=None,
                predicted_retention=None,
                matched_group=None,
                matched_tier=None,
                form=None,
                confidence_flag=(
                    tier4_result.confidence_flag
                    + " No Tier 3 Device-only curve "
                    "was available."
                ),
                analog_models_used=(
                    tier4_result.analog_models_used
                ),
            )

        # Tier 4 was unavailable entirely.
        return FallbackResult(
            predicted_value=None,
            predicted_retention=None,
            matched_group=None,
            matched_tier=None,
            form=None,
            confidence_flag=(
                "No fitted curve found for "
                f"device='{device}', "
                f"sub_device='{sub_device}', "
                f"provider='{provider}' "
                "at Tier 1 or Tier 2; "
                "Tier 4 analog-lineage forecasting "
                "was unavailable; "
                "and no Tier 3 Device-only curve exists."
            ),
        )


# ------------------------------------------------------------------
# Command-line quick check
# ------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 7:

        print(
            "Usage: python tradein_fallback.py "
            "<fitted_curves.csv> "
            "<device> "
            "<sub_device_or_None> "
            "<provider> "
            "<msrp> "
            "<model_year> "
            "[raw_data.csv]"
        )

        print()

        print(
            "Example:"
        )

        print(
            "python tradein_fallback.py "
            "data/fitted_curves.csv "
            "iPhone Pro Max SomeNewProvider "
            "6999 2026 "
            "data/master_msrp.csv"
        )

        sys.exit(1)

    fitted_curves_path = sys.argv[1]

    device = sys.argv[2]

    sub_device = (
        None
        if sys.argv[3].lower()
        == "none"
        else sys.argv[3]
    )

    provider = sys.argv[4]

    msrp = float(
        sys.argv[5]
    )

    model_year = int(
        sys.argv[6]
    )

    raw_data_path = (
        sys.argv[7]
        if len(sys.argv) > 7
        else None
    )

    fallback = TradeInFallback(
        fitted_curves_path,
        raw_data_path=raw_data_path,
    )

    result = fallback.predict(
        device=device,
        sub_device=sub_device,
        provider=provider,
        msrp=msrp,
        model_year=model_year,
    )

    for (
        key,
        value,
    ) in result.as_dict().items():

        print(
            f"{key}: {value}"
        )
