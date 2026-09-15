import sys; from dataclasses import dataclass; from typing import Optional; import numpy as np; import pandas as pd; from scipy.optimize import curve_fit; from sklearn.metrics import r2_score; REFERENCE_YEAR = 2026; RETENTION_FLOOR = 0.01; RETENTION_CEILING = 1.0; MIN_ROWS = 8; MIN_AGES = 4; MIN_R2 = 0.7; A_CEILING = 1.5; MIN_ANALOG_GENERATIONS = 2; MAX_ANALOG_GENERATIONS = 4; RETENTION_MAX_FOR_FITTING = 1.5
def exponential(age, a, b):
    return a * np.exp(-b * age)
def power_law(age, a, b):
    return a * np.power(age + 1, -b)
def hyperbolic(age, a, b):
    return a / (1 + b * age)
CURVE_FUNCS = {'exponential': exponential, 'power_law': power_law, 'hyperbolic': hyperbolic}
@dataclass
class FallbackResult:
    predicted_value: Optional[float]; predicted_retention: Optional[float]; matched_group: Optional[str]; matched_tier: Optional[str]; form: Optional[str]; confidence_flag: Optional[str]; analog_models_used: Optional[list] = None
    def as_dict(self):
        return {'predicted_value': self.predicted_value, 'predicted_retention': self.predicted_retention, 'matched_group': self.matched_group, 'matched_tier': self.matched_tier, 'form': self.form, 'confidence_flag': self.confidence_flag, 'analog_models_used': self.analog_models_used}
class TradeInFallback:
    def __init__(self, fitted_curves_path: str, raw_data_path: Optional[str]=None):
        self.curves = pd.read_csv(fitted_curves_path); required_cols = {'Device', 'Sub-device', 'Provider', 'Tier', 'FitGroup', 'form', 'a', 'b'}; missing = required_cols - set(self.curves.columns)
        if missing:
            raise ValueError(f'fitted_curves.csv is missing expected columns: {missing}')
        valid_tiers = {'1_subdevice_provider', '2_device_provider', '3_device_only'}; invalid_tiers = set(self.curves['Tier'].dropna().unique()) - valid_tiers
        if invalid_tiers:
            raise ValueError(f'fitted_curves.csv contains unknown tiers: {invalid_tiers}')
        valid_forms = set(CURVE_FUNCS.keys()); invalid_forms = set(self.curves['form'].dropna().unique()) - valid_forms
        if invalid_forms:
            raise ValueError(f'fitted_curves.csv contains unknown curve forms: {invalid_forms}')
        self.raw_data = None
        if raw_data_path is not None:
            raw = pd.read_csv(raw_data_path); required_raw_cols = {'Device', 'Sub-device', 'Model_Year', 'Retail Price', 'Max. Trade-In Value (RM)'}; missing_raw = required_raw_cols - set(raw.columns)
            if missing_raw:
                raise ValueError(f'Raw trade-in dataset is missing expected columns: {missing_raw}')
            self.raw_data = raw
    def _lookup_specific_curve(self, device: str, sub_device: Optional[str], provider: str):
        df = self.curves
        if sub_device is not None:
            match = df[(df['Device'] == device) & (df['Sub-device'] == sub_device) & (df['Provider'] == provider) & (df['Tier'] == '1_subdevice_provider')]
            if len(match) > 0:
                return match.iloc[0]
        match = df[(df['Device'] == device) & (df['Provider'] == provider) & (df['Tier'] == '2_device_provider')]
        if len(match) > 0:
            return match.iloc[0]
        return None
    def _lookup_device_curve(self, device: str):
        df = self.curves; match = df[(df['Device'] == device) & (df['Tier'] == '3_device_only')]
        if len(match) > 0:
            return match.iloc[0]
        return None
    def _select_analog_generations(self, device: str, sub_device: str, model_year: int):
        if self.raw_data is None:
            return (None, None)
        raw = self.raw_data; candidates = raw[(raw['Device'] == device) & (raw['Sub-device'] == sub_device) & (raw['Model_Year'] < model_year) & raw['Max. Trade-In Value (RM)'].notna() & (raw['Max. Trade-In Value (RM)'] > 0) & raw['Retail Price'].notna() & (raw['Retail Price'] > 0)]
        if candidates.empty:
            return (None, None)
        available_years = sorted(candidates['Model_Year'].dropna().unique(), reverse=True)
        if len(available_years) < MIN_ANALOG_GENERATIONS:
            return (None, None)
        selected_years = available_years[:MAX_ANALOG_GENERATIONS]; selected_rows = candidates[candidates['Model_Year'].isin(selected_years)].copy(); return (selected_rows, sorted(selected_years))
    def _fit_pooled_curve(self, rows: pd.DataFrame, reference_year: int):
        ages = reference_year - rows['Model_Year'].to_numpy(dtype=float); retention = rows['Max. Trade-In Value (RM)'].to_numpy(dtype=float) / rows['Retail Price'].to_numpy(dtype=float); valid = (retention > 0) & (retention <= RETENTION_MAX_FOR_FITTING) & (ages >= 0); ages = ages[valid]; retention = retention[valid]; n_rows = len(ages); n_ages = len(np.unique(ages))
        if n_rows < MIN_ROWS or n_ages < MIN_AGES:
            return None
        best = None
        for form_name, func in CURVE_FUNCS.items():
            try:
                params, _ = curve_fit(func, ages, retention, p0=[1.0, 0.3], bounds=([0, 0], [A_CEILING, np.inf]), maxfev=10000); a, b = params; predicted = func(ages, a, b); r2 = r2_score(retention, predicted); rmse = float(np.sqrt(np.mean((retention - predicted) ** 2)))
            except (RuntimeError, ValueError):
                continue
            if best is None or rmse < best['rmse']:
                best = {'form': form_name, 'a': float(a), 'b': float(b), 'r2': float(r2), 'rmse': rmse, 'a_hit_ceiling': bool(np.isclose(a, A_CEILING))}
        if best is None:
            return None
        if best['r2'] < MIN_R2:
            return None
        return (best['form'], best['a'], best['b'], best['r2'], n_rows, n_ages, best['a_hit_ceiling'])
    def _predict_tier4(self, device: str, sub_device: Optional[str], msrp: float, model_year: int, age: int, reference_year: int) -> Optional[FallbackResult]:
        if self.raw_data is None or sub_device is None:
            return None
        rows, generation_years = self._select_analog_generations(device=device, sub_device=sub_device, model_year=model_year)
        if rows is None:
            return FallbackResult(predicted_value=None, predicted_retention=None, matched_group=None, matched_tier=None, form=None, confidence_flag=f"Tier 4 not available: fewer than {MIN_ANALOG_GENERATIONS} prior generations of '{device} {sub_device}' with valid trade-in history were found.", analog_models_used=None)
        fit = self._fit_pooled_curve(rows=rows, reference_year=REFERENCE_YEAR)
        if fit is None:
            return FallbackResult(predicted_value=None, predicted_retention=None, matched_group=None, matched_tier=None, form=None, confidence_flag=f"Tier 4 attempted using analog generations {generation_years} of '{device} {sub_device}', but the pooled data did not clear curve quality gates (MIN_ROWS={MIN_ROWS}, MIN_AGES={MIN_AGES}, MIN_R2={MIN_R2}).", analog_models_used=generation_years)
        form, a, b, r2, n_rows, n_ages, a_hit_ceiling = fit; func = CURVE_FUNCS[form]; raw_retention = float(func(np.array([age]), a, b)[0]); retention = min(max(raw_retention, RETENTION_FLOOR), RETENTION_CEILING); predicted_value = round(retention * float(msrp), 2); flags = [f"No historical curve exists for '{device} {sub_device}'; used a new-model analog-lineage forecast pooled from {len(generation_years)} prior generation(s) ({generation_years}), fitted={form}, R2={r2:.3f}."]
        if a_hit_ceiling:
            flags.append(f"Analog curve's 'a' parameter hit the {A_CEILING} ceiling; review this forecast.")
        if raw_retention > RETENTION_CEILING:
            flags.append(f'Raw curve predicted {raw_retention:.3f} retention; capped at 1.000 because trade-in value cannot exceed Retail Price.')
        return FallbackResult(predicted_value=predicted_value, predicted_retention=round(retention, 5), matched_group=f'{device} | {sub_device} | analog_lineage', matched_tier='4_analog_lineage', form=form, confidence_flag=' '.join(flags), analog_models_used=generation_years)
    def _predict_from_curve(self, curve_row, msrp: float, age: int, sub_device: Optional[str]) -> FallbackResult:
        form = curve_row['form']; a = float(curve_row['a']); b = float(curve_row['b']); func = CURVE_FUNCS[form]; raw_retention = float(func(np.array([age]), a, b)[0]); group_max_retention = curve_row.get('max_observed_retention'); effective_ceiling = RETENTION_CEILING
        if group_max_retention is not None and (not pd.isna(group_max_retention)):
            effective_ceiling = min(RETENTION_CEILING, float(group_max_retention))
        retention = min(max(raw_retention, RETENTION_FLOOR), effective_ceiling); predicted_value = round(retention * float(msrp), 2); flags = []; matched_tier = curve_row['Tier']
        if sub_device is not None and matched_tier == '2_device_provider':
            flags.append(f"No reliable Sub-device-level curve for '{sub_device}'; fell back to Device + Provider.")
        if matched_tier == '3_device_only':
            if sub_device is not None:
                flags.append(f"No reliable Sub-device + Provider curve for '{sub_device}', no reliable Device + Provider curve, and no usable analog-lineage forecast; fell back to pooled Device curve.")
            else:
                flags.append('Used pooled Device-only curve.')
        if 'a_hit_ceiling' in curve_row.index and bool(curve_row['a_hit_ceiling']):
            flags.append(f"Source curve's 'a' parameter hit the {A_CEILING} ceiling; review this curve if needed.")
        if raw_retention > RETENTION_CEILING:
            flags.append(f'Raw curve predicted {raw_retention:.3f} retention; capped at 1.000 because trade-in value cannot exceed Retail Price.')
        confidence_flag = ' '.join(flags) if flags else None; return FallbackResult(predicted_value=predicted_value, predicted_retention=round(retention, 5), matched_group=curve_row['FitGroup'], matched_tier=matched_tier, form=form, confidence_flag=confidence_flag, analog_models_used=None)
    def predict(self, device: str, provider: str, msrp: float, model_year: int, sub_device: Optional[str]=None, reference_year: int=REFERENCE_YEAR) -> FallbackResult:
        age = reference_year - model_year
        if age < 0:
            return FallbackResult(predicted_value=None, predicted_retention=None, matched_group=None, matched_tier=None, form=None, confidence_flag=f'model_year {model_year} is in the future relative to reference_year {reference_year}')
        if msrp is None or msrp <= 0:
            return FallbackResult(predicted_value=None, predicted_retention=None, matched_group=None, matched_tier=None, form=None, confidence_flag=f'Invalid Retail Price: {msrp}. Retail Price must be greater than zero.')
        curve_row = self._lookup_specific_curve(device=device, sub_device=sub_device, provider=provider)
        if curve_row is not None:
            return self._predict_from_curve(curve_row=curve_row, msrp=msrp, age=age, sub_device=sub_device)
        tier4_result = self._predict_tier4(device=device, sub_device=sub_device, msrp=msrp, model_year=model_year, age=age, reference_year=reference_year)
        if tier4_result is not None and tier4_result.predicted_value is not None:
            return tier4_result
        device_curve = self._lookup_device_curve(device=device)
        if device_curve is not None:
            result = self._predict_from_curve(curve_row=device_curve, msrp=msrp, age=age, sub_device=sub_device)
            if tier4_result is not None and tier4_result.confidence_flag is not None:
                tier4_message = tier4_result.confidence_flag
                if result.confidence_flag:
                    result.confidence_flag = tier4_message + ' ' + result.confidence_flag
                else:
                    result.confidence_flag = tier4_message
            return result
        if tier4_result is not None and tier4_result.confidence_flag is not None:
            return FallbackResult(predicted_value=None, predicted_retention=None, matched_group=None, matched_tier=None, form=None, confidence_flag=tier4_result.confidence_flag + ' No Tier 3 Device-only curve was available.', analog_models_used=tier4_result.analog_models_used)
        return FallbackResult(predicted_value=None, predicted_retention=None, matched_group=None, matched_tier=None, form=None, confidence_flag=f"No fitted curve found for device='{device}', sub_device='{sub_device}', provider='{provider}' at Tier 1 or Tier 2; Tier 4 analog-lineage forecasting was unavailable; and no Tier 3 Device-only curve exists.")
if __name__ == '__main__':
    if len(sys.argv) < 7:
        print('Usage: python tradein_fallback.py <fitted_curves.csv> <device> <sub_device_or_None> <provider> <msrp> <model_year> [raw_data.csv]'); print(); print('Example:'); print('python tradein_fallback.py data/fitted_curves.csv iPhone Pro Max SomeNewProvider 6999 2026 data/master_msrp.csv'); sys.exit(1)
    fitted_curves_path = sys.argv[1]; device = sys.argv[2]; sub_device = None if sys.argv[3].lower() == 'none' else sys.argv[3]; provider = sys.argv[4]; msrp = float(sys.argv[5]); model_year = int(sys.argv[6]); raw_data_path = sys.argv[7] if len(sys.argv) > 7 else None; fallback = TradeInFallback(fitted_curves_path, raw_data_path=raw_data_path); result = fallback.predict(device=device, sub_device=sub_device, provider=provider, msrp=msrp, model_year=model_year)
    for key, value in result.as_dict().items():
        print(f'{key}: {value}')