import re
import difflib

import numpy as np
import pandas as pd


# ============================================================
# CANONICAL SCHEMA
# ============================================================

CANONICAL_FIELDS = [
    "Provider",
    "Device",
    "Sub-device",
    "Model Number",
    "Standardized Model",
    "Retail Price",
    "Storage (GB)",
    "Storage Type",
    "Connectivity",
    "Material",
    "Max. Trade-In Value (RM)",
    "Model_Year",
    "Case Size",
    "Charging Method",
    "Collection Date",
]

CONFIGURATION_FIELDS = [
    "Device",
    "Sub-device",
    "Standardized Model",
    "Model Number",
    "Storage (GB)",
    "Storage Type",
    "Connectivity",
    "Material",
    "Model_Year",
    "Case Size",
    "Charging Method",
]

# Collection Date is Master metadata (the date associated with the
# provider's current pricing data), never Apple configuration
# identity -- it must never affect duplicate detection, matching, or
# classification. DUPLICATE_FIELDS therefore stays exactly what it
# was before Collection Date existed.
DUPLICATE_FIELDS = [f for f in CANONICAL_FIELDS if f != "Collection Date"]

NUMERIC_FIELDS = {
    "Retail Price",
    "Storage (GB)",
    "Max. Trade-In Value (RM)",
    "Model_Year",
    "Case Size",
}

OPTIONAL_FIELDS = {
    "Provider",
    "Model Number",
    "Storage Type",
    "Connectivity",
    "Material",
    "Max. Trade-In Value (RM)",
    "Case Size",
    "Charging Method",
}

DEVICE_APPLICABLE_FIELDS = {
    "iPhone": {
        "Storage (GB)",
        "Model_Year",
    },
    "iPad": {
        "Storage (GB)",
        "Connectivity",
        "Model_Year",
    },
    "Mac": {
        "Storage (GB)",
        "Storage Type",
        "Model_Year",
    },
    "Apple Watch": {
        "Storage (GB)",
        "Connectivity",
        "Material",
        "Case Size",
        "Model_Year",
    },
    "AirPods": {
        "Charging Method",
        "Model_Year",
    },
}


def is_field_applicable(device, field):
    if field in {
        "Provider",
        "Device",
        "Sub-device",
        "Model Number",
        "Standardized Model",
        "Retail Price",
        "Max. Trade-In Value (RM)",
    }:
        return True

    applicable_fields = DEVICE_APPLICABLE_FIELDS.get(
        device,
        set(),
    )

    return field in applicable_fields


COLUMN_SYNONYMS = {
    "Provider": ["provider", "vendor", "source", "company", "seller"],
    "Device": ["category", "device", "device type", "product category", "device family"],
    "Sub-device": ["sub-device", "subdevice", "sub device", "series", "line", "tier"],
    "Model Number": ["model number", "model no", "model no.", "model num", "part number", "part no", "a-number", "a number", "sku"],
    "Standardized Model": ["model name", "model", "device model", "product name", "variant", "device name", "product description"],
    "Retail Price": ["retail price", "msrp", "retail", "sell price", "sale price", "list price"],
    "Storage (GB)": ["storage", "capacity", "storage (gb)", "storage gb"],
    "Storage Type": ["storage type", "memory type"],
    "Connectivity": ["connectivity", "network", "cellular"],
    "Material": ["material", "case material", "band material", "finish"],
    "Max. Trade-In Value (RM)": ["trade-in value", "trade in value", "trade-in", "buyback", "buyback price", "trade value", "max. trade-in value (rm)", "price"],
    "Model_Year": ["year", "model year", "release year"],
    "Case Size": ["case size", "size", "case (mm)"],
    "Charging Method": ["charging method", "charging", "charging case", "specification", "specifications"],
}

DEVICE_ALIASES = {
    "iphone": "iPhone", "phone": "iPhone", "i phone": "iPhone",
    "ipad": "iPad", "pad": "iPad", "tablet": "iPad",
    "mac": "Mac", "macbook": "Mac", "laptop": "Mac", "notebook": "Mac", "imac": "Mac", "mac mini": "Mac", "mac studio": "Mac", "mac pro": "Mac",
    "apple watch": "Apple Watch", "watch": "Apple Watch",
    "airpods": "AirPods", "air pods": "AirPods", "earbuds": "AirPods", "earphones": "AirPods",
}

BRAND_PREFIX_WORDS = {"apple"}
UNKNOWN = "Unknown"


def _is_blank(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "unknown":
            return True
    return False


def _is_unknown(value):
    return _is_blank(value)


def normalize_text(value):
    if _is_blank(value):
        return None
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def strip_brand_words(text):
    if not text:
        return ""
    tokens = [t for t in str(text).split() if t.lower() not in BRAND_PREFIX_WORDS]
    return " ".join(tokens)


def normalize_device_value(value):
    if _is_blank(value):
        return None
    text = normalize_text(value)
    return DEVICE_ALIASES.get(text) if text else None


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, set):
        return [json_safe(v) for v in value]
    if isinstance(value, frozenset):
        return sorted(json_safe(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def build_device_vocabulary(master_df):
    devices, sub_devices, pairs = set(), set(), set()
    if master_df is None or master_df.empty:
        return {"devices": devices, "sub_devices": sub_devices, "pairs": pairs}

    for _, row in master_df.iterrows():
        device = normalize_device_value(row.get("Device"))
        sub_device = normalize_text(row.get("Sub-device"))
        if device:
            devices.add(device)
        if sub_device:
            sub_devices.add(sub_device)
        if device and sub_device:
            pairs.add((device, sub_device))
    return {"devices": devices, "sub_devices": sub_devices, "pairs": pairs}


def build_row_search_text(raw_row):
    parts = []
    for _, value in raw_row.items():
        if not _is_blank(value):
            parts.append(str(value))
    return normalize_text(" ".join(parts)) or ""


def infer_model_year_from_text(value):
    if _is_blank(value):
        return None
    match = re.search(r"\b(20\d{2})\b", str(value))
    return int(match.group(1)) if match else None


def infer_device_subdevice(row_text, vocabulary):
    if not row_text:
        return None, None
    normalized = normalize_text(row_text) or ""
    device_matches = []
    for device in vocabulary.get("devices", set()):
        d_text = normalize_text(device)
        if d_text and re.search(rf"\b{re.escape(d_text)}\b", normalized):
            device_matches.append(device)

    for alias, canonical in DEVICE_ALIASES.items():
        if canonical in vocabulary.get("devices", set()) and re.search(rf"\b{re.escape(alias)}\b", normalized):
            if canonical not in device_matches:
                device_matches.append(canonical)

    device_matches = sorted(set(device_matches))
    if len(device_matches) != 1:
        return None, None

    device = device_matches[0]
    sub_matches = []
    for sub_device in vocabulary.get("sub_devices", set()):
        s_text = normalize_text(sub_device)
        if s_text and re.search(rf"\b{re.escape(s_text)}\b", normalized):
            sub_matches.append(sub_device)

    sub_matches = sorted(set(sub_matches))
    if len(sub_matches) == 1:
        return device, sub_matches[0]

    if len(sub_matches) > 1:
        word_sets = {c: set(normalize_text(c).split()) for c in sub_matches}
        most_specific = [c for c, w in word_sets.items() if not any(o != c and w < word_sets[o] for o in sub_matches)]
        if len(most_specific) == 1:
            return device, most_specific[0]
        paired = [c for c in most_specific if (device, c) in vocabulary.get("pairs", set())]
        if len(paired) == 1:
            return device, paired[0]

    return device, None


def _parse_numeric_value(value, field_name=None):
    if _is_blank(value):
        return None, False
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None, True
        return (None, True) if not np.isfinite(num) else (num, False)

    text = str(value).strip()
    if field_name == "Storage (GB)":
        match = re.fullmatch(r"(?i)\s*(\d+(?:\.\d+)?)\s*(gb|tb)?\s*", text)
        if not match:
            return None, True
        num = float(match.group(1))
        if match.group(2) and match.group(2).lower() == "tb":
            num *= 1024
        return num, False

    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return None, True
    try:
        num = float(text)
    except (TypeError, ValueError):
        return None, True
    return (None, True) if not np.isfinite(num) else (num, False)


def normalize_numeric_for_comparison(value, field_name):
    number, invalid = _parse_numeric_value(value, field_name)
    if invalid:
        return "__INVALID__"
    if number is None:
        return UNKNOWN
    return int(round(number)) if field_name in {"Storage (GB)", "Model_Year"} else round(number, 2)


def _pretokenize_storage_text(value):
    if _is_blank(value):
        return value
    return re.sub(r"(?i)(\d)(gb|tb)\b", r"\1 \2", str(value))


_MAC_MODEL_IDENTIFIER_RE = re.compile(r"[A-Za-z]+\d+,\d+")


def _split_model_number_text(text, is_mac):
    if not is_mac:
        return re.split(r"[,/;|]+", text)
    protected = []
    def _protect(match):
        p = f"\x00{len(protected)}\x00"
        protected.append(match.group(0))
        return p
    protected_text = _MAC_MODEL_IDENTIFIER_RE.sub(_protect, text)
    raw_parts = re.split(r"[,/;|]+", protected_text)
    restored = []
    for part in raw_parts:
        r = part
        for i, orig in enumerate(protected):
            r = r.replace(f"\x00{i}\x00", orig)
        restored.append(r)
    return restored


def normalize_model_numbers(value, device=None):
    if _is_blank(value):
        return frozenset()
    is_mac = normalize_device_value(device) == "Mac"
    parts = _split_model_number_text(str(value), is_mac)
    normalized = set()
    for part in parts:
        token = re.sub(r"[^A-Za-z0-9,]" if (is_mac and _MAC_MODEL_IDENTIFIER_RE.search(part)) else r"[^A-Za-z0-9]", "", part).upper()
        if token:
            normalized.add(token)
    return frozenset(normalized)


_MAC_IDENTIFIER_FAMILY_CASING = {
    "MAC": "Mac", "MACBOOK": "MacBook", "MACBOOKAIR": "MacBookAir", "MACBOOKPRO": "MacBookPro",
    "MACMINI": "Macmini", "MACPRO": "MacPro", "MACSTUDIO": "MacStudio", "IMAC": "iMac", "IMACPRO": "iMacPro"
}
_MAC_IDENTIFIER_LETTERS_RE = re.compile(r"^([A-Za-z]+)(\d+,\d+)$")


def _apply_mac_identifier_casing(text):
    match = _MAC_IDENTIFIER_LETTERS_RE.match(text)
    if not match:
        return text
    letters, suffix = match.groups()
    canonical = _MAC_IDENTIFIER_FAMILY_CASING.get(letters.upper())
    return (canonical + suffix) if canonical else text


def normalize_master_model_number_column(df):
    if df is None or df.empty or "Model Number" not in df.columns:
        return df
    result = df.copy()
    has_device = "Device" in result.columns
    for idx, value in result["Model Number"].items():
        if _is_blank(value):
            continue
        is_mac = normalize_device_value(result.at[idx, "Device"] if has_device else None) == "Mac"
        parts = _split_model_number_text(str(value), is_mac)
        seen, display = set(), []
        for part in parts:
            cleaned = part.strip()
            if not cleaned:
                continue
            if is_mac and _MAC_MODEL_IDENTIFIER_RE.search(cleaned):
                cleaned = _apply_mac_identifier_casing(cleaned)
                key = re.sub(r"[^A-Za-z0-9,]", "", cleaned).upper()
            else:
                key = re.sub(r"[^A-Za-z0-9]", "", cleaned).upper()
            if key and key not in seen:
                seen.add(key)
                display.append(cleaned)
        if display:
            result.at[idx, "Model Number"] = ", ".join(display)
    return result


_TRUSTED_APPLE_MODEL_NUMBER_RE = re.compile(r"^A\d{4}$")


def _is_trusted_apple_model_number(token):
    return bool(_TRUSTED_APPLE_MODEL_NUMBER_RE.match(token))


def build_model_number_subdevice_map(master_df, device):
    mapping = {}
    if master_df is None or "Device" not in master_df.columns or "Model Number" not in master_df.columns or "Sub-device" not in master_df.columns:
        return mapping
    target_device = normalize_device_value(device)
    if target_device != "iPhone":
        return mapping
    for _, row in master_df.iterrows():
        if normalize_device_value(row.get("Device")) != target_device:
            continue
        sub = row.get("Sub-device")
        if _is_blank(sub):
            continue
        for token in normalize_model_numbers(row.get("Model Number")):
            if _is_trusted_apple_model_number(token):
                if token not in mapping:
                    mapping[token] = sub
                elif mapping[token] != sub:
                    mapping[token] = None
    return mapping


def infer_subdevice_from_model_numbers(tokens, mapping):
    if not tokens or not mapping:
        return None
    matched = {mapping.get(t) for t in tokens if mapping.get(t) is not None}
    return next(iter(matched)) if len(matched) == 1 else None


def normalize_connectivity_value(value):
    if _is_blank(value):
        return None
    text = normalize_text(value) or ""
    has_cel = any(p in text for p in ("cellular", "5g", "4g", "lte", "sim"))
    has_wifi = any(p in text for p in ("wi fi", "wifi", "wi-fi"))
    if has_wifi and has_cel:
        return "wi fi cellular"
    if has_cel:
        return "cellular"
    if has_wifi:
        return "wi fi"
    return text or None


def extract_model_connectivity(model_text):
    if not model_text:
        return "", None
    text = normalize_text(model_text) or ""
    has_cel = any(p in text for p in ("cellular", "5g", "4g", "lte", "sim"))
    has_wifi = any(p in text for p in ("wi fi", "wifi", "wi-fi"))
    conn = "wi fi cellular" if (has_wifi and has_cel) else ("cellular" if has_cel else ("wi fi" if has_wifi else None))
    cleaned = re.sub(r"\bwi[\s-]?fi\b|\bwi-fi\b|\bcellular\b|\b5g\b|\b4g\b|\blte\b|\bsim\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(), conn


def build_canonical_display_maps(master_df):
    sub_map, model_map, conn_map = {}, {}, {}
    if master_df is None or master_df.empty:
        return {"sub_device": sub_map, "model": model_map, "connectivity": conn_map}

    for _, row in master_df.iterrows():
        device = normalize_device_value(row.get("Device"))
        if not device:
            continue
        sub = row.get("Sub-device")
        if not _is_blank(sub):
            n_sub = normalize_text(sub)
            if n_sub:
                sub_map.setdefault((device, n_sub), sub)
        model = row.get("Standardized Model")
        if not _is_blank(model):
            stripped, _ = extract_model_connectivity(strip_brand_words(normalize_text(model)))
            if stripped:
                model_map.setdefault((device, stripped), model)
        conn = row.get("Connectivity")
        if not _is_blank(conn):
            n_conn = normalize_connectivity_value(conn)
            if n_conn:
                conn_map.setdefault((device, n_conn), conn)
    return {"sub_device": sub_map, "model": model_map, "connectivity": conn_map}


def canonicalize_incoming_row_display_values(row, display_maps):
    updates = {}
    raw_dev = row.get("Device")
    c_dev = normalize_device_value(raw_dev)
    if c_dev and c_dev != raw_dev:
        updates["Device"] = c_dev
    dev_key = c_dev or normalize_device_value(raw_dev)

    raw_sub = row.get("Sub-device")
    if not _is_blank(raw_sub):
        n_sub = normalize_text(raw_sub)
        if n_sub:
            c_sub = display_maps["sub_device"].get((dev_key, n_sub))
            if c_sub and c_sub != raw_sub:
                updates["Sub-device"] = c_sub

    raw_model = row.get("Standardized Model")
    if not _is_blank(raw_model):
        stripped, embedded_conn = extract_model_connectivity(strip_brand_words(normalize_text(raw_model)))
        c_model = display_maps["model"].get((dev_key, stripped))
        if c_model and c_model != raw_model:
            updates["Standardized Model"] = c_model

        conn_token = normalize_connectivity_value(row.get("Connectivity")) or embedded_conn
        if conn_token:
            c_conn = display_maps["connectivity"].get((dev_key, conn_token))
            if c_conn and c_conn != row.get("Connectivity"):
                updates["Connectivity"] = c_conn

    raw_mn = row.get("Model Number")
    if not _is_blank(raw_mn):
        tokens = normalize_model_numbers(raw_mn, device=dev_key)
        if tokens:
            c_mn = ", ".join(sorted(tokens))
            if c_mn != str(raw_mn).strip():
                updates["Model Number"] = c_mn

    return updates


_ORDINAL_GENERATION_RE = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)
_ALPHANUMERIC_GENERATION_RE = re.compile(r"\b(\d{1,2})([a-z])\b", re.IGNORECASE)
_CHIPSET_CODE_RE = re.compile(r"\b[am]\d+[a-z]?\b", re.IGNORECASE)
_VARIANT_TOKENS = {"air", "mini", "pro", "max", "plus", "se", "ultra", "studio"}


def extract_apple_model_identity(model_text):
    if not model_text:
        return {"product_line": None, "variant": None, "generations": set(), "model_designations": set(), "screen_sizes": set(), "identity_tokens": set()}
    text = normalize_text(model_text) or ""
    tokens = text.split()
    product_line = next((c for c in ["iphone", "ipad", "macbook", "imac", "mac mini", "mac studio", "mac pro", "apple watch", "airpods"] if c in text), None)

    variants, generations, designations, screen_sizes = set(), set(), set(), set()
    for token in tokens:
        if token in _VARIANT_TOKENS:
            variants.add(token)
        if token.isdigit():
            generations.add(token)
        if not _CHIPSET_CODE_RE.fullmatch(token):
            m_ord = _ORDINAL_GENERATION_RE.fullmatch(token)
            if m_ord:
                generations.add(m_ord.group(1))
            m_alph = _ALPHANUMERIC_GENERATION_RE.fullmatch(token)
            if m_alph:
                designations.add(f"{m_alph.group(1)}{m_alph.group(2)}")

    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*[- ]?inch\b", text):
        screen_sizes.add(m.group(1))

    return {"product_line": product_line, "variant": " ".join(sorted(variants)) if variants else None, "generations": generations, "model_designations": designations, "screen_sizes": screen_sizes, "identity_tokens": set(tokens)}


def _designation_numeric_bases(designations):
    return {re.match(r"(\d+)", d).group(1) for d in designations if re.match(r"(\d+)", d)}


def model_identity_conflict(incoming_model, master_model):
    inc, mst = extract_apple_model_identity(incoming_model), extract_apple_model_identity(master_model)
    if inc["product_line"] and mst["product_line"] and inc["product_line"] != mst["product_line"]:
        return True
    if inc["variant"] and mst["variant"] and inc["variant"] != mst["variant"]:
        return True
    if inc["generations"] and mst["generations"] and inc["generations"].isdisjoint(mst["generations"]):
        return True
    if inc["model_designations"] and mst["model_designations"] and inc["model_designations"].isdisjoint(mst["model_designations"]):
        return True
    if inc["screen_sizes"] and mst["screen_sizes"] and inc["screen_sizes"].isdisjoint(mst["screen_sizes"]):
        return True
    
    inc_bases = inc["generations"] | _designation_numeric_bases(inc["model_designations"])
    mst_bases = mst["generations"] | _designation_numeric_bases(mst["model_designations"])
    if inc_bases and mst_bases and inc_bases.isdisjoint(mst_bases):
        return True
    return False




def propose_column_mapping(raw_columns):
    norm_raw = {c: normalize_text(c) or "" for c in raw_columns}
    norm_synonyms = {c: {normalize_text(a) for a in aliases} for c, aliases in COLUMN_SYNONYMS.items()}
    mapping, used = {}, set()

    for canonical in CANONICAL_FIELDS:
        cands = norm_synonyms.get(canonical, set())
        cands.add(normalize_text(canonical))
        for raw, norm_r in norm_raw.items():
            if raw not in used and norm_r in cands:
                mapping[raw] = canonical
                used.add(raw)
                break

    for canonical in CANONICAL_FIELDS:
        if canonical in mapping.values():
            continue
        cands = norm_synonyms.get(canonical, set())
        cands.add(normalize_text(canonical))
        best_raw, best_score = None, 0
        for raw, norm_r in norm_raw.items():
            if raw in used or not norm_r:
                continue
            score = max((100 if a == norm_r else (70 if a in norm_r else (60 if norm_r in a else 0)) for a in cands if a), default=0)
            if score > best_score:
                best_score, best_raw = score, raw
        if best_raw and best_score >= 60:
            mapping[best_raw] = canonical
            used.add(best_raw)

    return {"mapping": mapping, "unmapped_columns": [c for c in raw_columns if c not in used], "missing_required": [f for f in ["Device", "Standardized Model"] if f not in mapping.values()]}


def apply_column_mapping(raw_df, mapping):
    canonical_df = pd.DataFrame(index=raw_df.index)
    rev = {v: k for k, v in mapping.items()}
    for canonical in CANONICAL_FIELDS:
        canonical_df[canonical] = raw_df[rev[canonical]] if rev.get(canonical) in raw_df.columns else np.nan
    return canonical_df


def _get_mapped_value(row, field, mapped_fields):
    return row.get(field) if field in mapped_fields else None


def normalize_row_for_matching(row, mapped_fields=None):
    mapped_fields = set(mapped_fields or CANONICAL_FIELDS)
    raw_dev = _get_mapped_value(row, "Device", mapped_fields)
    raw_model = _get_mapped_value(row, "Standardized Model", mapped_fields)
    device = normalize_device_value(raw_dev)
    model_text, embedded_conn = extract_model_connectivity(strip_brand_words(normalize_text(raw_model)))
    explicit_conn = normalize_connectivity_value(_get_mapped_value(row, "Connectivity", mapped_fields))

    return {
        "device": device or UNKNOWN,
        "sub_device": normalize_text(_get_mapped_value(row, "Sub-device", mapped_fields)) or UNKNOWN,
        "model_text": model_text or UNKNOWN,
        "storage": normalize_numeric_for_comparison(_get_mapped_value(row, "Storage (GB)", mapped_fields), "Storage (GB)"),
        "storage_type": normalize_text(_get_mapped_value(row, "Storage Type", mapped_fields)) or UNKNOWN,
        "connectivity": explicit_conn if explicit_conn is not None else (embedded_conn or UNKNOWN),
        "material": normalize_text(_get_mapped_value(row, "Material", mapped_fields)) or UNKNOWN,
        "model_year": normalize_numeric_for_comparison(_get_mapped_value(row, "Model_Year", mapped_fields), "Model_Year"),
        "provider": normalize_text(_get_mapped_value(row, "Provider", mapped_fields)) or UNKNOWN,
        "case_size": normalize_numeric_for_comparison(_get_mapped_value(row, "Case Size", mapped_fields), "Case Size"),
        "charging_method": normalize_text(_get_mapped_value(row, "Charging Method", mapped_fields)) or UNKNOWN,
        "retail_price": normalize_numeric_for_comparison(_get_mapped_value(row, "Retail Price", mapped_fields), "Retail Price"),
        "trade_in_value": normalize_numeric_for_comparison(_get_mapped_value(row, "Max. Trade-In Value (RM)", mapped_fields), "Max. Trade-In Value (RM)"),
        "model_numbers": normalize_model_numbers(_get_mapped_value(row, "Model Number", mapped_fields), device=raw_dev),
    }


def validate_row_data(row):
    errors = []
    for field in NUMERIC_FIELDS:
        if not _is_blank(row.get(field)) and _parse_numeric_value(row.get(field), field)[1]:
            errors.append(f"{field} contains an invalid value: {row.get(field)!r}")
    dev = row.get("Device")
    if not _is_blank(dev) and normalize_device_value(dev) is None:
        errors.append(f"Device contains an unrecognized value: {dev!r}")
    return errors


def validate_new_row_fields(device, sub_device, model_name, msrp, trade_in_value, storage, storage_type, connectivity, material, case_size, charging_method, model_year):
    errors = []
    if _is_blank(device): errors.append("Device is required.")
    if _is_blank(sub_device): errors.append("Sub-device is required.")
    if _is_blank(model_name): errors.append("Standardized Model is required.")
    if _is_blank(msrp): errors.append("Retail Price is required.")
    elif _parse_numeric_value(msrp, "Retail Price")[1]: errors.append("Retail Price must be numeric.")
    elif _parse_numeric_value(msrp, "Retail Price")[0] < 0: errors.append("Retail Price cannot be negative.")
    return errors


def _field_value(norm, field):
    mapping = {
        "Device": "device", "Sub-device": "sub_device", "Standardized Model": "model_text",
        "Storage (GB)": "storage", "Storage Type": "storage_type", "Connectivity": "connectivity",
        "Material": "material", "Model_Year": "model_year", "Case Size": "case_size",
        "Charging Method": "charging_method", "Provider": "provider", "Retail Price": "retail_price",
        "Max. Trade-In Value (RM)": "trade_in_value",
    }
    return norm.get(mapping[field])


def extract_mac_configuration(model_text):
    if _is_blank(model_text):
        return {"chip": None, "cpu_tier": None, "cpu_frequency": None, "cpu_cores": None, "gpu_cores": None}
    text = normalize_text(model_text) or ""
    return {
        "chip": (re.search(r"\bm[1-4](?:\s+(?:pro|max|ultra))?\b", text) or [None])[0] and re.search(r"\bm[1-4](?:\s+(?:pro|max|ultra))?\b", text).group(0),
        "cpu_tier": (re.search(r"\bi[3579]\b", text) or [None])[0] and re.search(r"\bi[3579]\b", text).group(0),
        "cpu_frequency": float((re.search(r"\b(\d+(?:\.\d+)?)\s*ghz\b", text)).group(1)) if re.search(r"\b(\d+(?:\.\d+)?)\s*ghz\b", text) else None,
        "cpu_cores": int((re.search(r"\b(\d+)\s*core\s+cpu\b", text)).group(1)) if re.search(r"\b(\d+)\s*core\s+cpu\b", text) else None,
        "gpu_cores": int((re.search(r"\b(\d+)\s*core\s+gpu\b", text)).group(1)) if re.search(r"\b(\d+)\s*core\s+gpu\b", text) else None,
    }


def _mac_configuration_match_score(incoming_model, master_model):
    inc, mst = extract_mac_configuration(incoming_model), extract_mac_configuration(master_model)
    score, weight = 0.0, 0.0
    for key in ["chip", "cpu_tier", "cpu_frequency", "cpu_cores", "gpu_cores"]:
        if inc.get(key) is not None and mst.get(key) is not None:
            weight += 1.0
            if inc.get(key) == mst.get(key):
                score += 1.0
    return (score / weight) if weight > 0 else None


def _model_similarity(incoming_model, master_model):
    if _is_unknown(incoming_model) or _is_unknown(master_model):
        return None
    inc, mst = strip_brand_words(normalize_text(incoming_model) or ""), strip_brand_words(normalize_text(master_model) or "")
    if not inc or not mst:
        return None
    if inc == mst:
        return 1.0
    if model_identity_conflict(inc, mst):
        return 0.0
    seq_score = difflib.SequenceMatcher(None, inc, mst).ratio()
    inc_t, mst_t = set(inc.split()), set(mst.split())
    tok_score = (len(inc_t & mst_t) / len(inc_t | mst_t)) if (inc_t and mst_t) else 0.0
    gen_score = max(seq_score, tok_score)
    cfg_score = _mac_configuration_match_score(inc, mst)
    return ((gen_score * 0.60) + (cfg_score * 0.40)) if cfg_score is not None else gen_score


def _structured_field_match_score(incoming_norm, master_norm):
    device = incoming_norm.get("device")
    comparisons = [("Storage (GB)", "storage"), ("Storage Type", "storage_type"), ("Connectivity", "connectivity"), ("Material", "material"), ("Model_Year", "model_year"), ("Case Size", "case_size"), ("Charging Method", "charging_method")]
    score, weight = 0.0, 0.0
    for field_name, key in comparisons:
        if is_field_applicable(device, field_name):
            inc, mst = incoming_norm.get(key), master_norm.get(key)
            if not _is_unknown(inc) and not _is_unknown(mst):
                weight += 1.0
                if inc == mst:
                    score += 1.0
    return (score / weight) if weight > 0 else None


def score_candidate(incoming_norm, master_norm):
    inc_m, mst_m = incoming_norm.get("model_text"), master_norm.get("model_text")
    m_score = _model_similarity(inc_m, mst_m) or 0.0
    s_score = _structured_field_match_score(incoming_norm, master_norm) or 0.0
    conflicts = ["model_identity_conflict"] if model_identity_conflict(inc_m, mst_m) else []
    
    for field_name, key in [("Storage (GB)", "storage"), ("Storage Type", "storage_type"), ("Connectivity", "connectivity"), ("Material", "material"), ("Model_Year", "model_year"), ("Case Size", "case_size"), ("Charging Method", "charging_method")]:
        if is_field_applicable(incoming_norm.get("device"), field_name):
            inc, mst = incoming_norm.get(key), master_norm.get(key)
            if not _is_unknown(inc) and not _is_unknown(mst) and inc != mst:
                conflicts.append(field_name)

    return {
        "confidence": round((m_score * 0.70) + (s_score * 0.30), 4),
        "model_score": round(m_score, 4),
        "structured_score": round(s_score, 4),
        "model_number_match": compare_model_numbers(incoming_norm, master_norm),
        "model_identity_conflict": "model_identity_conflict" in conflicts,
        "conflicts": conflicts,
    }


def find_master_candidates(incoming_norm, master_norm_by_index, master_device_by_index, master_provider_by_index=None):
    inc_dev = incoming_norm.get("device")
    if _is_unknown(inc_dev):
        return []
    candidates = []
    for idx, m_norm in master_norm_by_index.items():
        if master_device_by_index.get(idx) != inc_dev:
            continue
        res = score_candidate(incoming_norm, m_norm)
        if not res["model_identity_conflict"] and res["model_score"] >= 0.65:
            candidates.append({"index": idx, "norm": m_norm, **res})
    return candidates


def compare_model_numbers(incoming_norm, master_norm):
    inc = incoming_norm.get("model_numbers", frozenset())
    mst = master_norm.get("model_numbers", frozenset())
    if not inc and not mst: return "both_absent"
    if not inc and mst: return "master_only"
    if inc and not mst: return "incoming_only"
    if inc & mst: return "match"
    return "append"


def _values_equal(field, incoming_norm, master_norm):
    if field == "Model Number":
        inc = incoming_norm.get("model_numbers", frozenset())
        mst = master_norm.get("model_numbers", frozenset())
        if not inc or not mst: return None
        return bool(inc & mst)
    inc, mst = _field_value(incoming_norm, field), _field_value(master_norm, field)
    if _is_unknown(inc) or _is_unknown(mst): return None
    return inc == mst


def compare_master_relevant_fields(incoming_norm, master_norm):
    unknown_fields, different_fields = [], []
    device = incoming_norm.get("device")
    for field in DUPLICATE_FIELDS:
        if not is_field_applicable(device, field):
            continue
        if field == "Retail Price":
            inc, mst = incoming_norm.get("retail_price"), master_norm.get("retail_price")
            if _is_unknown(inc) or _is_unknown(mst):
                if _is_unknown(inc) and _is_unknown(mst): unknown_fields.append(field)
                continue
            if inc != mst: different_fields.append(field)
            continue
        eq = _values_equal(field, incoming_norm, master_norm)
        if eq is None: unknown_fields.append(field)
        elif not eq: different_fields.append(field)
    return {"exact": not unknown_fields and not different_fields, "unknown_fields": unknown_fields, "different_fields": different_fields}


def configuration_matches(incoming_norm, master_norm):
    unknown_fields, different_fields = [], []
    if _is_unknown(incoming_norm.get("device")) or _is_unknown(master_norm.get("device")):
        unknown_fields.append("Device")
    elif incoming_norm.get("device") != master_norm.get("device"):
        different_fields.append("Device")

    if _is_unknown(incoming_norm.get("sub_device")) or _is_unknown(master_norm.get("sub_device")):
        unknown_fields.append("Sub-device")
    elif incoming_norm.get("sub_device") != master_norm.get("sub_device"):
        different_fields.append("Sub-device")

    inc_m, mst_m = incoming_norm.get("model_text"), master_norm.get("model_text")
    if _is_unknown(inc_m) or _is_unknown(mst_m):
        unknown_fields.append("Standardized Model")
    else:
        if model_identity_conflict(inc_m, mst_m) or _model_similarity(inc_m, mst_m) < 0.80:
            different_fields.append("Standardized Model")
        for f_name, key in [("Storage (GB)", "storage"), ("Storage Type", "storage_type"), ("Connectivity", "connectivity"), ("Material", "material"), ("Model_Year", "model_year"), ("Case Size", "case_size"), ("Charging Method", "charging_method")]:
            if is_field_applicable(incoming_norm.get("device"), f_name):
                inc, mst = incoming_norm.get(key), master_norm.get(key)
                if _is_unknown(inc) or _is_unknown(mst): unknown_fields.append(f_name)
                elif inc != mst: different_fields.append(f_name)

    return {"matches": not different_fields, "unknown_fields": sorted(set(unknown_fields)), "different_fields": sorted(set(different_fields))}


def choose_reference_candidate(candidates, incoming_provider=None):
    if not candidates: return None
    if not _is_unknown(incoming_provider):
        same_prov = [c for c in candidates if not _is_unknown(c["norm"].get("provider")) and c["norm"].get("provider") == incoming_provider]
        if same_prov:
            return max(same_prov, key=lambda c: c["confidence"])
    return max(candidates, key=lambda c: c["confidence"])


def classify_incoming_row(incoming_row_canonical, incoming_norm, master_df, master_norm_by_index, master_device_by_index, master_provider_by_index, mapped_canonical_fields=None):
    reasons, warnings, inherited_fields = [], [], []
    mapped_canonical_fields = mapped_canonical_fields or set()

    validation_errors = validate_row_data(incoming_row_canonical)
    if validation_errors:
        return {"classification": "conflict", "conflict_type": "invalid_data", "reasons": validation_errors, "warnings": [], "match": None, "match_index": None, "multi_match": False, "unknown_fields": [], "different_fields": [], "model_numbers_to_append": [], "model_number_update_required": False, "model_number_flag": None}

    if _is_unknown(incoming_norm.get("device")):
        return {"classification": "conflict", "conflict_type": "needs_review", "reasons": ["Device could not be identified."], "warnings": [], "match": None, "match_index": None, "multi_match": False, "unknown_fields": ["Device"], "different_fields": [], "model_numbers_to_append": [], "model_number_update_required": False, "model_number_flag": None}

    candidates = find_master_candidates(incoming_norm, master_norm_by_index, master_device_by_index, master_provider_by_index)
    if not candidates:
        return {"classification": "new", "conflict_type": None, "reasons": ["No Master counterpart exists for this Apple model/configuration."], "warnings": [], "match": None, "match_index": None, "multi_match": False, "unknown_fields": [], "different_fields": [], "model_numbers_to_append": [], "model_number_update_required": False, "model_number_flag": None}

    best_candidate = choose_reference_candidate(candidates, incoming_norm.get("provider"))
    multi_match = len(candidates) > 1

    # Track if Retail Price was inherited rather than supplied
    retail_price_was_inherited = False
    inc_rp, mst_rp = incoming_norm.get("retail_price"), best_candidate["norm"].get("retail_price")
    if _is_unknown(inc_rp) and not _is_unknown(mst_rp):
        incoming_norm["retail_price"] = mst_rp
        incoming_row_canonical["Retail Price"] = mst_rp
        inherited_fields.append("Retail Price")
        retail_price_was_inherited = True

    inc_st, mst_st = incoming_norm.get("storage_type"), best_candidate["norm"].get("storage_type")
    if "Storage Type" not in mapped_canonical_fields and _is_unknown(inc_st) and not _is_unknown(mst_st):
        incoming_norm["storage_type"] = mst_st
        incoming_row_canonical["Storage Type"] = mst_st
        inherited_fields.append("Storage Type")

    inc_mns, mst_mns = incoming_norm.get("model_numbers", frozenset()), best_candidate["norm"].get("model_numbers", frozenset())
    if "Model Number" not in mapped_canonical_fields and not inc_mns and mst_mns:
        incoming_norm["model_numbers"] = mst_mns
        incoming_row_canonical["Model Number"] = ", ".join(sorted(mst_mns))
        inherited_fields.append("Model Number")

    incoming_norm["_inherited_fields"] = list(inherited_fields)
    comp = configuration_matches(incoming_norm, best_candidate["norm"])

    if comp["unknown_fields"]:
        reasons.append("A Master counterpart exists, but one or more relevant fields are Unknown.")
        return {"classification": "conflict", "conflict_type": "needs_review", "reasons": reasons, "warnings": warnings, "match": best_candidate, "match_index": best_candidate["index"], "multi_match": multi_match, "unknown_fields": comp["unknown_fields"], "different_fields": comp["different_fields"], "model_numbers_to_append": [], "model_number_update_required": False, "model_number_flag": compare_model_numbers(incoming_norm, best_candidate["norm"])}

    if comp["different_fields"]:
        reasons.append("The incoming row represents a valid row that is not an exact match for the selected Master configuration.")
        return {"classification": "new", "conflict_type": None, "reasons": reasons, "warnings": warnings, "match": best_candidate, "match_index": best_candidate["index"], "multi_match": multi_match, "unknown_fields": [], "different_fields": comp["different_fields"], "model_numbers_to_append": [], "model_number_update_required": False, "model_number_flag": compare_model_numbers(incoming_norm, best_candidate["norm"])}

    full_comp = compare_master_relevant_fields(incoming_norm, best_candidate["norm"])
    mn_flag = compare_model_numbers(incoming_norm, best_candidate["norm"])

    if full_comp["unknown_fields"]:
        reasons.append("The Master counterpart exists, but one or more Master-relevant fields are Unknown.")
        return {"classification": "conflict", "conflict_type": "needs_review", "reasons": reasons, "warnings": warnings, "match": best_candidate, "match_index": best_candidate["index"], "multi_match": multi_match, "unknown_fields": full_comp["unknown_fields"], "different_fields": full_comp["different_fields"], "model_numbers_to_append": [], "model_number_update_required": False, "model_number_flag": mn_flag}

    if mn_flag == "append" and not full_comp["unknown_fields"] and not [f for f in full_comp["different_fields"] if f != "Model Number"]:
        existing_mns = set(best_candidate["norm"].get("model_numbers", frozenset()))
        inc_mns = set(incoming_norm.get("model_numbers", frozenset()))
        return {
            "classification": "update", "conflict_type": None, "reasons": ["The Apple configuration already exists in Master, but the incoming Model Number is new and should be appended."],
            "warnings": warnings, "match": best_candidate, "match_index": best_candidate["index"], "multi_match": multi_match, "unknown_fields": [], "different_fields": [], "model_number_flag": "append", "model_numbers_to_append": sorted(inc_mns - existing_mns), "model_number_update_required": True
        }

    if full_comp["exact"]:
        reasons.append("The complete Master-relevant row already exists.")
        return {"classification": "conflict", "conflict_type": "duplicate", "reasons": reasons, "warnings": warnings, "match": best_candidate, "match_index": best_candidate["index"], "multi_match": multi_match, "unknown_fields": [], "different_fields": [], "model_numbers_to_append": [], "model_number_update_required": False, "model_number_flag": "match"}

    diff_fields = full_comp["different_fields"]
    if "Provider" in diff_fields or "Model Number" in diff_fields:
        reasons.append("The Apple configuration exists, but the Provider or Model Number differs.")
        return {"classification": "new", "conflict_type": None, "reasons": reasons, "warnings": warnings, "match": best_candidate, "match_index": best_candidate["index"], "multi_match": multi_match, "unknown_fields": [], "different_fields": diff_fields, "model_numbers_to_append": [], "model_number_update_required": False, "model_number_flag": mn_flag}

    if "Retail Price" in diff_fields:
        # If the retail price was inherited because the upload source lacked it, treat as non-conflicting mutable update
        if retail_price_was_inherited:
            reasons.append("Retail Price was inherited from Master; only mutable value(s) differ, so update.")
            return {"classification": "update", "conflict_type": None, "reasons": reasons, "warnings": warnings, "match": best_candidate, "match_index": best_candidate["index"], "multi_match": multi_match, "unknown_fields": [], "different_fields": diff_fields, "model_numbers_to_append": [], "model_number_update_required": False, "model_number_flag": mn_flag}

        reasons.append("Retail Price differs from the existing Master row; requires Admin review.")
        return {"classification": "conflict", "conflict_type": "needs_review", "reasons": reasons, "warnings": warnings, "match": best_candidate, "match_index": best_candidate["index"], "multi_match": multi_match, "unknown_fields": [], "different_fields": diff_fields, "model_numbers_to_append": [], "model_number_update_required": False, "model_number_flag": mn_flag}

    reasons.append("The Apple configuration, Provider, and Retail Price match; only mutable value(s) differ, update row.")
    return {"classification": "update", "conflict_type": None, "reasons": reasons, "warnings": warnings, "match": best_candidate, "match_index": best_candidate["index"], "multi_match": multi_match, "unknown_fields": [], "different_fields": diff_fields, "model_numbers_to_append": [], "model_number_update_required": False, "model_number_flag": mn_flag}


def _build_strict_identity(row_norm):
    identity = []
    device = row_norm.get("device")
    for field in DUPLICATE_FIELDS:
        if not is_field_applicable(device, field):
            continue
        if field == "Model Number":
            identity.append(tuple(sorted(row_norm.get("model_numbers", frozenset()))))
            continue
        val = _field_value(row_norm, field)
        identity.append(UNKNOWN if _is_unknown(val) else (round(val, 2) if isinstance(val, float) else val))
    return tuple(identity)


def detect_duplicates(canonical_df, mapped_fields=None):
    mapped_fields = set(mapped_fields or CANONICAL_FIELDS)
    groups, norm_by_idx = {}, {}
    for idx, row in canonical_df.iterrows():
        norm = normalize_row_for_matching(row, mapped_fields)
        norm_by_idx[idx] = norm
        groups.setdefault(_build_strict_identity(norm), []).append(idx)
    
    return {idx: {"duplicate_group": groups[_build_strict_identity(norm_by_idx[idx])], "is_first_occurrence": idx == groups[_build_strict_identity(norm_by_idx[idx])][0], "within_file_duplicate": len(groups[_build_strict_identity(norm_by_idx[idx])]) > 1} for idx in canonical_df.index}


def build_master_indexes(master_df):
    m_norm, m_dev, m_prov = {}, {}, {}
    if master_df is None or master_df.empty:
        return m_norm, m_dev, m_prov
    for idx, row in master_df.iterrows():
        norm = normalize_row_for_matching(row, set(CANONICAL_FIELDS))
        m_norm[idx], m_dev[idx], m_prov[idx] = norm, norm.get("device"), norm.get("provider")
    return m_norm, m_dev, m_prov


def build_extra_column_info(raw_df, unmapped_columns):
    res = []
    for col in unmapped_columns:
        s = raw_df[col]
        samples = [str(v) for v in s if not _is_blank(v)][:5]
        res.append({"column": col, "sample_values": samples, "non_empty_count": int(s.notna().sum())})
    return res


def build_summary(classifications):
    summary = {"new": 0, "update": 0, "conflict": 0, "duplicate": 0, "needs_review": 0, "invalid_data": 0}
    for cls, c_type in classifications:
        if cls in ("new", "update"):
            summary[cls] += 1
        elif cls == "conflict":
            summary["conflict"] += 1
            if c_type in ("duplicate", "needs_review", "invalid_data"):
                summary[c_type] += 1
    return summary


def _serialize_preview_row(row):
    if row is None: return None
    if hasattr(row, "to_dict"): row = row.to_dict()
    return {k: (None if pd.isna(v) else (v.item() if isinstance(v, np.generic) else v)) for k, v in row.items()}


def build_effective_record(pre_inference_row, normalized_row, effective_row, inherited_fields, matched_master_row=None):
    inherited_fields = set(inherited_fields or [])
    def _get(r, f): return r.get(f) if (r is not None and hasattr(r, "get")) else None
    
    values, provenance = {}, {}
    for field in CANONICAL_FIELDS:
        eff_val = _get(effective_row, field)
        if field in inherited_fields:
            prov = "master_inherited"
            master_disp = _get(matched_master_row, field)
            if not _is_blank(master_disp): eff_val = master_disp
        elif not _is_blank(eff_val):
            prov = "inferred" if (not _is_blank(_get(normalized_row, field)) and _is_blank(_get(pre_inference_row, field))) else "incoming"
        else:
            prov = "missing"
        values[field], provenance[field] = json_safe(eff_val), prov
    return {"values": values, "provenance": provenance}


def effective_record_to_row(effective_record):
    if not effective_record or not isinstance(effective_record.get("values"), dict): return {}
    return {k: v for k, v in effective_record["values"].items()}


def describe_classification(cls, c_type, reasons):
    defaults = {
        ("new", None): "This is a new Apple configuration and will be added to Master.",
        ("update", None): "This matches an existing Master row; mutable values will be updated.",
        ("conflict", "duplicate"): "This row already exists in Master exactly as supplied.",
        ("conflict", "needs_review"): "A Master counterpart exists, but requires Admin confirmation.",
        ("conflict", "invalid_data"): "This row contains malformed data and must be fixed.",
    }
    def_text = defaults.get((cls, c_type))
    r_text = " ".join(reasons) if reasons else None
    return f"{def_text} {r_text}" if (def_text and r_text) else (def_text or r_text or "")


def analyze_upload(raw_df, master_df, clean_fn, column_mapping_override=None):
    if raw_df is None: raise ValueError("raw_df cannot be None.")
    if not isinstance(raw_df, pd.DataFrame): raise TypeError("raw_df must be a pandas DataFrame.")
    if master_df is None: master_df = pd.DataFrame(columns=CANONICAL_FIELDS)

    prop = propose_column_mapping(list(raw_df.columns))
    if column_mapping_override is not None:
        mapping = dict(column_mapping_override)
        unmapped = [c for c in raw_df.columns if c not in mapping]
        missing_req = [f for f in ["Device", "Standardized Model"] if f not in mapping.values()]
    else:
        mapping, unmapped, missing_req = prop["mapping"], prop["unmapped_columns"], prop["missing_required"]

    canonical_df = apply_column_mapping(raw_df, mapping)
    if "Storage (GB)" in canonical_df.columns:
        canonical_df["Storage (GB)"] = canonical_df["Storage (GB)"].apply(_pretokenize_storage_text)

    if clean_fn is not None:
        cleaned = clean_fn(canonical_df.copy())
        if cleaned is not None: canonical_df = cleaned

    for field in CANONICAL_FIELDS:
        if field not in canonical_df.columns: canonical_df[field] = np.nan
    canonical_df = canonical_df[CANONICAL_FIELDS]
    pre_inf_df = canonical_df.copy()

    vocabulary = build_device_vocabulary(master_df)
    inferred_fields = set()

    canonical_df["Device"] = canonical_df["Device"].astype(object)
    canonical_df["Sub-device"] = canonical_df["Sub-device"].astype(object)

    for idx in canonical_df.index:
        row = canonical_df.loc[idx]
        if _is_blank(row.get("Device")) or _is_blank(row.get("Sub-device")):
            row_text = build_row_search_text(row)
            inf_dev, inf_sub = infer_device_subdevice(row_text, vocabulary)
            if _is_blank(row.get("Device")) and inf_dev:
                canonical_df.at[idx, "Device"] = inf_dev
                inferred_fields.add("Device")
            if _is_blank(row.get("Sub-device")) and inf_sub:
                canonical_df.at[idx, "Sub-device"] = inf_sub
                inferred_fields.add("Sub-device")

    mn_maps = {}
    for idx in canonical_df.index:
        if _is_blank(canonical_df.at[idx, "Sub-device"]):
            dev_val = canonical_df.at[idx, "Device"]
            if not _is_blank(dev_val):
                if dev_val not in mn_maps:
                    mn_maps[dev_val] = build_model_number_subdevice_map(master_df, dev_val)
                inf_sub = infer_subdevice_from_model_numbers(normalize_model_numbers(canonical_df.at[idx, "Model Number"], device=dev_val), mn_maps[dev_val])
                if inf_sub:
                    canonical_df.at[idx, "Sub-device"] = inf_sub
                    inferred_fields.add("Sub-device")

    canonical_df["Model_Year"] = canonical_df["Model_Year"].astype(object)
    for idx in canonical_df.index:
        if _is_blank(canonical_df.at[idx, "Model_Year"]):
            inf_yr = infer_model_year_from_text(canonical_df.at[idx, "Standardized Model"])
            if inf_yr is not None:
                canonical_df.at[idx, "Model_Year"] = inf_yr
                inferred_fields.add("Model_Year")

    display_maps = build_canonical_display_maps(master_df)
    canonical_df["Standardized Model"] = canonical_df["Standardized Model"].astype(object)
    canonical_df["Connectivity"] = canonical_df["Connectivity"].astype(object)

    for idx in canonical_df.index:
        for f, val in canonicalize_incoming_row_display_values(canonical_df.loc[idx], display_maps).items():
            canonical_df.at[idx, f] = val
            inferred_fields.add(f)

    m_norm, m_dev, m_prov = build_master_indexes(master_df)
    mapped_fields = set(mapping.values()) | inferred_fields
    within_file_dups = detect_duplicates(canonical_df, mapped_fields)
    extra_cols = build_extra_column_info(raw_df, unmapped)

    row_results, classifications = [], []
    for idx, row in canonical_df.iterrows():
        norm = normalize_row_for_matching(row, mapped_fields)
        wf_info = within_file_dups.get(idx, {})
        is_wf_dup = wf_info.get("within_file_duplicate", False) and not wf_info.get("is_first_occurrence", False)
        norm_row = row.copy()

        if is_wf_dup:
            cls_res = {"classification": "conflict", "conflict_type": "duplicate", "reasons": ["Identical within-file duplicate."], "warnings": [], "match": None, "match_index": None, "multi_match": False, "unknown_fields": [], "different_fields": [], "model_number_flag": None, "model_numbers_to_append": [], "model_number_update_required": False}
            row_inh = []
        else:
            cls_res = classify_incoming_row(incoming_row_canonical=row, incoming_norm=norm, master_df=master_df, master_norm_by_index=m_norm, master_device_by_index=m_dev, master_provider_by_index=m_prov, mapped_canonical_fields=mapped_fields)
            row_inh = norm.get("_inherited_fields", [])
            canonical_df.loc[idx, "Retail Price"] = row.get("Retail Price")

        selected_m = cls_res.get("match")
        m_master_row = master_df.loc[selected_m.get("index")] if (selected_m and selected_m.get("index") in master_df.index) else None

        eff_rec = build_effective_record(pre_inference_row=pre_inf_df.loc[idx], normalized_row=norm_row, effective_row=row, inherited_fields=row_inh, matched_master_row=m_master_row)
        cls, c_type = cls_res["classification"], cls_res.get("conflict_type")
        classifications.append((cls, c_type))

        row_results.append({
            "row_index": idx, "classification": cls, "conflict_type": c_type, "reasons": cls_res["reasons"], "warnings": cls_res["warnings"],
            "match_index": cls_res["match_index"], "multi_match": cls_res["multi_match"], "unknown_fields": cls_res["unknown_fields"],
            "different_fields": cls_res["different_fields"], "model_number_flag": cls_res["model_number_flag"], "model_numbers_to_append": cls_res.get("model_numbers_to_append", []),
            "model_number_update_required": cls_res.get("model_number_update_required", False), "within_file_duplicate": wf_info.get("within_file_duplicate", False),
            "within_file_duplicate_group": wf_info.get("duplicate_group", []), "incoming_row": _serialize_preview_row(norm_row), "effective_record": eff_rec,
            "what_this_means": describe_classification(cls, c_type, cls_res["reasons"]), "matched_master": _serialize_preview_row(m_master_row) if m_master_row is not None else None,
            "match_confidence": selected_m.get("confidence") if selected_m else None, "matched_master_row": selected_m.get("index") if selected_m else None
        })

    return {
        "mapping": mapping, "unmapped_columns": unmapped, "extra_columns": extra_cols, "missing_required": missing_req,
        "canonical_df": canonical_df, "rows": row_results, "summary": build_summary(classifications),
        "master_norm_by_index": m_norm, "master_device_by_index": m_dev, "master_provider_by_index": m_prov, "within_file_duplicates": within_file_dups
    }


APPLY_NEW, APPLY_UPDATE, APPLY_SKIPPED_DUPLICATE, APPLY_QUEUED, APPLY_ERROR = "applied_new", "applied_update", "skipped_duplicate", "queued_for_review", "error"


def _append_model_numbers(master_df, match_index, model_numbers_to_append):
    if not model_numbers_to_append: return
    existing = master_df.at[match_index, "Model Number"]
    is_mac = normalize_device_value(master_df.at[match_index, "Device"] if "Device" in master_df.columns else None) == "Mac"
    tokens = [t.strip() for t in _split_model_number_text(str(existing), is_mac) if t.strip()] if not _is_blank(existing) else []
    norm_existing = normalize_model_numbers(existing, device=master_df.at[match_index, "Device"] if "Device" in master_df.columns else None)

    for t in model_numbers_to_append:
        if t not in norm_existing:
            tokens.append(t)
            norm_existing = norm_existing | {t}
    master_df.at[match_index, "Model Number"] = ", ".join(tokens)


def apply_classified_rows(canonical_df, row_results, master_df):
    working_df = master_df.copy()
    new_rows, outcomes = [], []

    for rr in row_results:
        idx, cls, c_type, m_idx = rr["row_index"], rr["classification"], rr.get("conflict_type"), rr.get("match_index")
        if cls == "new":
            new_rows.append(canonical_df.loc[idx])
            outcomes.append({"row_index": idx, "action": APPLY_NEW, "match_index": None, "row_result": rr})
            continue
        if cls == "update":
            if m_idx is None or m_idx not in working_df.index:
                outcomes.append({"row_index": idx, "action": APPLY_ERROR, "match_index": m_idx, "row_result": rr, "error": "Matched Master row missing."})
                continue
            if rr.get("model_number_update_required"):
                _append_model_numbers(working_df, m_idx, rr.get("model_numbers_to_append", []))
            else:
                working_df.at[m_idx, "Max. Trade-In Value (RM)"] = canonical_df.at[idx, "Max. Trade-In Value (RM)"]
            # Collection Date is metadata describing the freshness of
            # the provider's pricing data -- on any UPDATE it is
            # replaced by the incoming row's Collection Date, same as
            # Max. Trade-In Value (RM) above. A blank/missing incoming
            # value never overwrites an existing Master date.
            incoming_collection_date = canonical_df.at[idx, "Collection Date"] if "Collection Date" in canonical_df.columns else None
            if not _is_blank(incoming_collection_date):
                # Guard against a float64 Collection Date column (e.g.
                # still entirely blank) rejecting a string value.
                if "Collection Date" in working_df.columns and working_df["Collection Date"].dtype != object:
                    working_df["Collection Date"] = working_df["Collection Date"].astype(object)
                working_df.at[m_idx, "Collection Date"] = incoming_collection_date
            outcomes.append({"row_index": idx, "action": APPLY_UPDATE, "match_index": m_idx, "row_result": rr})
            continue
        if cls == "conflict" and c_type == "duplicate":
            outcomes.append({"row_index": idx, "action": APPLY_SKIPPED_DUPLICATE, "match_index": m_idx, "row_result": rr})
            continue
        outcomes.append({"row_index": idx, "action": APPLY_QUEUED, "match_index": m_idx, "row_result": rr})

    if new_rows:
        working_df = pd.concat([working_df, pd.DataFrame(new_rows, columns=CANONICAL_FIELDS)], ignore_index=True)
    else:
        working_df = working_df.reset_index(drop=True)

    return {"master_df": working_df, "outcomes": outcomes}