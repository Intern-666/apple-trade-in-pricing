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
    "Chipset",
    "Case Size",
    "Charging Method",
]

# These are the fields that must be available for a row to be
# meaningfully identified as an Apple configuration.
#
# Provider is deliberately NOT here.
#
# Provider is considered only after the Apple configuration has
# been identified:
#
#   same configuration + same Provider      -> duplicate if exact
#   same configuration + different Provider -> new
#   same configuration + Unknown Provider   -> needs_review
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
    "Chipset",
    "Case Size",
    "Charging Method",
]

# Duplicate means the entire Master-relevant row is identical.
# Metadata is not part of CANONICAL_FIELDS and therefore is
# automatically excluded.
#
# Retail Price and Max. Trade-In Value are intentionally included:
# they are part of the actual Master data row.
DUPLICATE_FIELDS = list(CANONICAL_FIELDS)

# Fields where a supplied value must be numeric / numeric-like.
NUMERIC_FIELDS = {
    "Retail Price",
    "Storage (GB)",
    "Max. Trade-In Value (RM)",
    "Model_Year",
    "Case Size",
}

# Fields whose values are allowed to be absent.
#
# Missing values are NOT treated as valid empty values.
# They are normalized to "Unknown" and can therefore cause
# needs_review when a Master counterpart exists.
#
# This set exists mainly to keep validation logic explicit.
OPTIONAL_FIELDS = {
    "Provider",
    "Model Number",
    "Storage Type",
    "Connectivity",
    "Material",
    "Max. Trade-In Value (RM)",
    "Case Size",
    "Charging Method",
    "Chipset",
}

# ============================================================
# DEVICE-SPECIFIC FIELD APPLICABILITY
# ============================================================

# These rules describe which configuration fields actually apply
# to each Apple device family in the current Master dataset.
#
# A field that is not applicable to a device is NOT treated as
# Unknown and does NOT participate in configuration comparison
# or duplicate detection.
#
# Chipset is intentionally absent because it is currently treated
# as a ghost field and is excluded from classification logic.

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
    """
    Return whether a Master-relevant field applies to a device.

    Device is expected to already be normalized to the canonical
    device name.
    """
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

    # Chipset is currently a ghost field.
    if field == "Chipset":
        return False

    applicable_fields = DEVICE_APPLICABLE_FIELDS.get(
        device,
        set(),
    )

    return field in applicable_fields


# ============================================================
# COLUMN MAPPING
# ============================================================

COLUMN_SYNONYMS = {
    "Provider": [
        "provider",
        "vendor",
        "source",
        "company",
        "seller",
    ],
    "Device": [
        "category",
        "device",
        "device type",
        "product category",
        "device family",
    ],
    "Sub-device": [
        "sub-device",
        "subdevice",
        "sub device",
        "series",
        "line",
        "tier",
    ],
    "Model Number": [
        "model number",
        "model no",
        "model no.",
        "model num",
        "part number",
        "part no",
        "a-number",
        "a number",
        "sku",
    ],
    "Standardized Model": [
        "model name",
        "model",
        "device model",
        "product name",
        "variant",
        "device name",
        "product description",
    ],
    "Retail Price": [
        "retail price",
        "msrp",
        "retail",
        "sell price",
        "sale price",
        "list price",
    ],
    "Storage (GB)": [
        "storage",
        "capacity",
        "storage (gb)",
        "storage gb",
    ],
    "Storage Type": [
        "storage type",
        "memory type",
    ],
    "Connectivity": [
        "connectivity",
        "network",
        "cellular",
    ],
    "Material": [
        "material",
        "case material",
        "band material",
        "finish",
    ],
    "Max. Trade-In Value (RM)": [
        "trade-in value",
        "trade in value",
        "trade-in",
        "buyback",
        "buyback price",
        "trade value",
        "max. trade-in value (rm)",
        "price",
    ],
    "Model_Year": [
        "year",
        "model year",
        "release year",
    ],
    "Chipset": [
        "chipset",
        "chip",
        "processor",
        "soc",
    ],
    "Case Size": [
        "case size",
        "size",
        "case (mm)",
    ],
    "Charging Method": [
        "charging method",
        "charging",
        "charging case",
        "specification",
        "specifications",
    ],
}


# ============================================================
# DEVICE NORMALIZATION
# ============================================================

DEVICE_ALIASES = {
    "iphone": "iPhone",
    "phone": "iPhone",
    "i phone": "iPhone",

    "ipad": "iPad",
    "pad": "iPad",
    "tablet": "iPad",

    "mac": "Mac",
    "macbook": "Mac",
    "laptop": "Mac",
    "notebook": "Mac",
    "imac": "Mac",
    "mac mini": "Mac",
    "mac studio": "Mac",
    "mac pro": "Mac",

    "apple watch": "Apple Watch",
    "watch": "Apple Watch",

    "airpods": "AirPods",
    "air pods": "AirPods",
    "earbuds": "AirPods",
    "earphones": "AirPods",
}

BRAND_PREFIX_WORDS = {"apple"}

UNKNOWN = "Unknown"


# ============================================================
# GENERAL VALUE HELPERS
# ============================================================

def _is_blank(value):
    """Return True when a value is genuinely missing."""
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        if stripped.lower() == "unknown":
            return True

    return False


def _is_unknown(value):
    """Return True for missing/Unknown values after normalization."""
    return _is_blank(value)


def normalize_text(value):
    """
    General-purpose text normalization.

    Keeps alphanumeric content while making comparisons
    case-insensitive and punctuation-insensitive.
    """
    if _is_blank(value):
        return None

    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text or None


def strip_brand_words(text):
    if not text:
        return ""

    tokens = [
        token
        for token in str(text).split()
        if token.lower() not in BRAND_PREFIX_WORDS
    ]

    return " ".join(tokens)


def normalize_device_value(value):
    if _is_blank(value):
        return None

    text = normalize_text(value)
    if not text:
        return None

    return DEVICE_ALIASES.get(text)


def json_safe(value):
    """Recursively convert common pandas/numpy values into JSON-safe values."""
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


# ============================================================
# DEVICE / SUB-DEVICE INFERENCE
# ============================================================

def build_device_vocabulary(master_df):
    """
    Build Master-derived vocabulary for deterministic inference.

    Returns:
        {
            "devices": set(...),
            "sub_devices": set(...),
            "pairs": [(device, sub_device), ...]
        }
    """
    devices = set()
    sub_devices = set()
    pairs = set()

    if master_df is None or master_df.empty:
        return {
            "devices": devices,
            "sub_devices": sub_devices,
            "pairs": pairs,
        }

    for _, row in master_df.iterrows():
        device = normalize_device_value(row.get("Device"))
        sub_device = normalize_text(row.get("Sub-device"))

        if device:
            devices.add(device)

        if sub_device:
            sub_devices.add(sub_device)

        if device and sub_device:
            pairs.add((device, sub_device))

    return {
        "devices": devices,
        "sub_devices": sub_devices,
        "pairs": pairs,
    }


def build_row_search_text(raw_row):
    """
    Concatenate all meaningful incoming cells so Device/Sub-device
    can be inferred from the row when explicit columns are absent.

    `raw_row` is a pandas Series at its one real call site
    (`canonical_df.loc[index]`). `.values()` is a dict method, not a
    Series one -- Series exposes `.values` as a plain ndarray
    attribute, so calling it raised TypeError. This previously went
    unnoticed because the caller only ever reached this line when
    Device was blank, which essentially never happened for real
    uploads that already supplied Device. `.items()` works
    identically for a Series and for a plain dict, so it is used
    here instead of `.values()`.
    """
    parts = []

    for _, value in raw_row.items():
        if _is_blank(value):
            continue

        parts.append(str(value))

    return normalize_text(" ".join(parts)) or ""

def infer_model_year_from_text(value):
    """
    Extract a 4-digit model year from model text.

    Examples:
      "MacBook Air i5 1.6GHz 13 inch (Mid 2019)" -> 2019
      "MacBook Air i3 1.1GHz 13 inch (Early 2020)" -> 2020

    Returns None when no plausible year is present.
    """
    if _is_blank(value):
        return None

    text = str(value)

    match = re.search(r"\b(20\d{2})\b", text)
    if not match:
        return None

    return int(match.group(1))

def infer_device_subdevice(row_text, vocabulary):
    """
    Infer Device and Sub-device from Master vocabulary.

    This is deliberately deterministic. It does not use fuzzy
    matching to invent a device category.
    """
    if not row_text:
        return None, None

    normalized = normalize_text(row_text) or ""

    device_matches = []
    for device in vocabulary.get("devices", set()):
        device_text = normalize_text(device)
        if not device_text:
            continue

        pattern = rf"\b{re.escape(device_text)}\b"
        if re.search(pattern, normalized):
            device_matches.append(device)

    # Also recognize common raw aliases.
    for alias, canonical in DEVICE_ALIASES.items():
        if canonical not in vocabulary.get("devices", set()):
            continue

        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, normalized):
            if canonical not in device_matches:
                device_matches.append(canonical)

    device_matches = sorted(set(device_matches))

    if len(device_matches) != 1:
        return None, None

    device = device_matches[0]

    sub_matches = []
    for sub_device in vocabulary.get("sub_devices", set()):
        sub_text = normalize_text(sub_device)
        if not sub_text:
            continue

        if re.search(rf"\b{re.escape(sub_text)}\b", normalized):
            sub_matches.append(sub_device)

    sub_matches = sorted(set(sub_matches))

    if len(sub_matches) == 1:
        return device, sub_matches[0]

    if len(sub_matches) > 1:
        # A shorter match (e.g. "ipad", "pro") is often just a
        # component word of a longer match that also matched (e.g.
        # "ipad pro") -- both are legitimate vocabulary phrases, so
        # both end up in sub_matches, even though only the longer
        # one is actually specific. Drop any candidate whose words
        # are a strict subset of another candidate's words; if that
        # leaves exactly one, it is the unambiguous, most specific
        # match. Still no fuzzy matching: this only ever discards a
        # candidate that is a word-for-word subset of another actual
        # match, it never invents or scores partial matches.
        word_sets = {
            candidate: set(normalize_text(candidate).split())
            for candidate in sub_matches
        }

        most_specific = [
            candidate
            for candidate, words in word_sets.items()
            if not any(
                other != candidate
                and words < word_sets[other]
                for other in sub_matches
            )
        ]

        if len(most_specific) == 1:
            return device, most_specific[0]

        # Still tied after specificity filtering (e.g. a bare
        # vocabulary word like "pro" leaking in from a field such
        # as Chipset or the model string itself, alongside the
        # genuinely correct compound like "ipad mini"). Narrow
        # using Device+Sub-device co-occurrence actually observed
        # in Master: if only one of the tied candidates has ever
        # been paired with this already-determined Device, that is
        # the unambiguous answer. This uses only data Master already
        # contains (vocabulary["pairs"]) -- no fuzzy matching, no
        # invented tie-break, and it never fires unless specificity
        # filtering alone could not decide.
        paired_with_device = [
            candidate
            for candidate in most_specific
            if (device, candidate) in vocabulary.get("pairs", set())
        ]

        if len(paired_with_device) == 1:
            return device, paired_with_device[0]

    return device, None


# ============================================================
# NUMERIC NORMALIZATION
# ============================================================

def _parse_numeric_value(value, field_name=None):
    """
    Parse a numeric value without silently accepting malformed
    currency/text values.

    Examples:
        1599       -> 1599.0
        "1599"     -> 1599.0
        "1599.00"  -> 1599.0
        "256 GB"   -> 256.0 for Storage (GB)

    But:
        "RM 1599"  -> invalid
        "$1599"    -> invalid
        "abc"      -> invalid
    """
    if _is_blank(value):
        return None, False

    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None, True

        if not np.isfinite(number):
            return None, True

        return number, False

    text = str(value).strip()

    # Storage may legitimately contain units.
    if field_name == "Storage (GB)":
        storage_match = re.fullmatch(
            r"(?i)\s*(\d+(?:\.\d+)?)\s*(gb|tb)?\s*",
            text,
        )

        if not storage_match:
            return None, True

        number = float(storage_match.group(1))
        unit = storage_match.group(2)

        if unit and unit.lower() == "tb":
            number *= 1024

        return number, False

    # Other numeric fields must be plain numeric values.
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return None, True

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None, True

    if not np.isfinite(number):
        return None, True

    return number, False


def normalize_numeric_for_comparison(value, field_name):
    """
    Return:
        UNKNOWN when missing
        numeric value when valid
        INVALID when malformed
    """
    number, invalid = _parse_numeric_value(value, field_name)

    if invalid:
        return "__INVALID__"

    if number is None:
        return UNKNOWN

    if field_name in {"Storage (GB)", "Model_Year"}:
        return int(round(number))

    return round(number, 2)


def _pretokenize_storage_text(value):
    """
    Preserve compatibility with the previous cleaner by inserting
    a separator before GB/TB when necessary.
    """
    if _is_blank(value):
        return value

    text = str(value)
    text = re.sub(r"(?i)(\d)(gb|tb)\b", r"\1 \2", text)
    return text


# ============================================================
# MODEL NUMBER NORMALIZATION
# ============================================================

def normalize_model_numbers(value):
    if _is_blank(value):
        return frozenset()

    parts = re.split(r"[,/;|]+", str(value))

    normalized = set()

    for part in parts:
        token = re.sub(r"[^A-Za-z0-9]", "", part).upper()

        if token:
            normalized.add(token)

    return frozenset(normalized)


# ============================================================
# MODEL NUMBER -> SUB-DEVICE INFERENCE
#
# Supplements the generic keyword-based Sub-device inference
# (infer_device_subdevice) for rows where the incoming text has
# no recognizable Sub-device keyword at all (e.g. a plain
# "iPhone 17" row with no "Pro"/"Plus"/etc token to match), by
# using the authoritative Model Number -> Sub-device relationship
# already present in Master, instead of assuming "no qualifier
# means Standard".
# ============================================================

_TRUSTED_APPLE_MODEL_NUMBER_RE = re.compile(r"^A\d{4}$")


def _is_trusted_apple_model_number(token):
    """
    Return True for a token that looks like a genuine Apple
    regulatory model number: the letter "A" followed by exactly
    four digits (e.g. "A2111", "A3258").

    Master contains a handful of fictional/test rows with Model
    Number values such as "2", "3", "Test123", or blank. None of
    those match this shape, so they are never trusted as an
    inference source -- without needing to special-case each
    fictional value by name.
    """
    return bool(_TRUSTED_APPLE_MODEL_NUMBER_RE.match(token))


def build_model_number_subdevice_map(master_df, device):
    """
    Build a trusted Model Number -> Sub-device mapping from Master
    rows, scoped to a single Device.

    Only Master rows whose Model Number tokens look like genuine
    Apple regulatory model numbers (see
    _is_trusted_apple_model_number) contribute to the mapping.
    Master's fictional/test rows are therefore silently excluded
    as inference sources, rather than trusted.

    If trusted Master rows disagree about which Sub-device a given
    Model Number token belongs to, that token is mapped to None so
    downstream inference treats it as ambiguous rather than
    guessing.

    Scoped to iPhone only: this reuses the same "A" + four digits
    regulatory model number shape that other Apple devices (iPad,
    Mac, etc.) also use, so without this restriction it would
    start influencing Sub-device inference -- and therefore
    candidate matching -- for those other devices too, which is
    outside what this inference was introduced to fix.
    """
    mapping = {}

    if (
        master_df is None
        or "Device" not in master_df.columns
        or "Model Number" not in master_df.columns
        or "Sub-device" not in master_df.columns
    ):
        return mapping

    target_device = normalize_device_value(device)
    if target_device != "iPhone":
        return mapping

    for _, master_row in master_df.iterrows():
        if (
            normalize_device_value(master_row.get("Device"))
            != target_device
        ):
            continue

        sub_device = master_row.get("Sub-device")
        if _is_blank(sub_device):
            continue

        tokens = normalize_model_numbers(
            master_row.get("Model Number")
        )

        for token in tokens:
            if not _is_trusted_apple_model_number(token):
                continue

            if token not in mapping:
                mapping[token] = sub_device
            elif mapping[token] != sub_device:
                mapping[token] = None

    return mapping


def infer_subdevice_from_model_numbers(
    incoming_model_numbers,
    model_number_map,
):
    """
    Given the incoming row's normalized Model Number tokens and a
    trusted Model Number -> Sub-device mapping (see
    build_model_number_subdevice_map), return the single Sub-device
    every matching, trusted token agrees on.

    Returns None (leave Unknown) when no incoming token has a
    trusted mapping, or when matching tokens disagree.
    """
    if not incoming_model_numbers or not model_number_map:
        return None

    matched_sub_devices = set()

    for token in incoming_model_numbers:
        sub_device = model_number_map.get(token)
        if sub_device is None:
            continue
        matched_sub_devices.add(sub_device)

    if len(matched_sub_devices) == 1:
        return next(iter(matched_sub_devices))

    return None


# ============================================================
# CONNECTIVITY NORMALIZATION
# ============================================================

def normalize_connectivity_value(value):
    if _is_blank(value):
        return None

    text = normalize_text(value) or ""

    has_cellular = any(
        phrase in text
        for phrase in (
            "cellular",
            "5g",
            "4g",
            "lte",
            "sim",
        )
    )

    has_wifi = any(
        phrase in text
        for phrase in (
            "wi fi",
            "wifi",
            "wi-fi",
        )
    )

    if has_wifi and has_cellular:
        return "wi fi cellular"

    if has_cellular:
        return "cellular"

    if has_wifi:
        return "wi fi"

    return text or None


def extract_model_connectivity(model_text):
    """
    Detect connectivity embedded inside Standardized Model and remove
    it from the model comparison text.

    Example:
        "iPad Air 11-inch Wi-Fi + Cellular"
            -> ("ipad air 11 inch", "wi fi cellular")
    """
    if not model_text:
        return "", None

    text = normalize_text(model_text) or ""

    has_cellular = any(
        phrase in text
        for phrase in (
            "cellular",
            "5g",
            "4g",
            "lte",
            "sim",
        )
    )

    has_wifi = any(
        phrase in text
        for phrase in (
            "wi fi",
            "wifi",
            "wi-fi",
        )
    )

    connectivity = None

    if has_wifi and has_cellular:
        connectivity = "wi fi cellular"
    elif has_cellular:
        connectivity = "cellular"
    elif has_wifi:
        connectivity = "wi fi"

    # Remove connectivity words from model text.
    cleaned = re.sub(
        r"\bwi[\s-]?fi\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\bwi-fi\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\bcellular\b|\b5g\b|\b4g\b|\blte\b|\bsim\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned, connectivity


# ============================================================
# INCOMING ROW DISPLAY CANONICALIZATION
#
# Matching normalization (normalize_text, normalize_row_for_matching,
# normalize_device_value, extract_model_connectivity, etc.) already
# correctly resolves an incoming row's *meaning* for comparison
# purposes -- but it only ever produces lowercase, comparison-safe
# text, and that text is never written back to canonical_df. As a
# result the incoming row an Admin actually sees/imports can retain
# raw, inconsistently-cased, or provider-formatted values (e.g.
# Device "Pad" instead of "iPad", Sub-device "ipad" instead of
# "iPad", or a Standardized Model still carrying an embedded
# connectivity suffix such as "iPad 7 Wi-Fi + Cellular" instead of
# "iPad 7" with Connectivity populated separately).
#
# This section builds small Master-derived lookups from the same
# normalized representation back to Master's own properly-cased
# spelling, and canonicalize_incoming_row() below uses them to
# rewrite ONLY the incoming row's own Device / Sub-device /
# Standardized Model / Connectivity fields to that canonical
# spelling. It never substitutes any other field (Provider,
# Storage, Retail Price, Trade-In value, Model Number, ...) and it
# never copies values from any *specific* matched Master row --
# only the generic naming/casing convention Master uses across all
# of its rows for a given Device.
#
# Lookups are scoped per-Device to avoid cross-device collisions
# (for example a bare "cellular" token means something different
# for an Apple Watch GPS + Cellular model than it would for an
# iPad, so the two must not share one global lookup).
# ============================================================

def build_canonical_display_maps(master_df):
    """
    Build Master-derived (Device, normalized_value) -> canonical
    display value lookups for Sub-device, Standardized Model (with
    any embedded connectivity stripped, matching
    extract_model_connectivity's comparison text), and Connectivity.
    """
    sub_device_map = {}
    model_map = {}
    connectivity_map = {}

    if master_df is None or master_df.empty:
        return {
            "sub_device": sub_device_map,
            "model": model_map,
            "connectivity": connectivity_map,
        }

    for _, row in master_df.iterrows():
        device = normalize_device_value(row.get("Device"))
        if not device:
            continue

        raw_sub = row.get("Sub-device")
        if not _is_blank(raw_sub):
            normalized_sub = normalize_text(raw_sub)
            if normalized_sub:
                sub_device_map.setdefault((device, normalized_sub), raw_sub)

        raw_model = row.get("Standardized Model")
        if not _is_blank(raw_model):
            model_text = normalize_text(raw_model)
            model_text = strip_brand_words(model_text)
            stripped_model, _ = extract_model_connectivity(model_text)
            if stripped_model:
                model_map.setdefault((device, stripped_model), raw_model)

        raw_connectivity = row.get("Connectivity")
        if not _is_blank(raw_connectivity):
            normalized_connectivity = normalize_connectivity_value(
                raw_connectivity
            )
            if normalized_connectivity:
                connectivity_map.setdefault(
                    (device, normalized_connectivity), raw_connectivity
                )

    return {
        "sub_device": sub_device_map,
        "model": model_map,
        "connectivity": connectivity_map,
    }


def canonicalize_incoming_row_display_values(row, display_maps):
    """
    Given a canonical_df row (after Device/Sub-device inference has
    already run), return the display-canonical values this row's
    OWN Device, Sub-device, Standardized Model, and Connectivity
    should have -- using Master's naming/casing conventions as the
    source of truth, without altering the row's own factual
    content (Provider, Storage, Retail Price, Trade-In value, etc.
    are never touched here).

    Returns a dict of only the fields that should be updated.
    """
    updates = {}

    # --- Device: fix aliases/casing, e.g. "Pad" -> "iPad" ---
    raw_device = row.get("Device")
    canonical_device = normalize_device_value(raw_device)
    if canonical_device and canonical_device != raw_device:
        updates["Device"] = canonical_device

    device_key = canonical_device or normalize_device_value(raw_device)

    # --- Sub-device: fix casing using Master's own spelling ---
    raw_sub = row.get("Sub-device")
    if not _is_blank(raw_sub):
        normalized_sub = normalize_text(raw_sub)
        if normalized_sub:
            canonical_sub = display_maps["sub_device"].get(
                (device_key, normalized_sub)
            )
            if canonical_sub and canonical_sub != raw_sub:
                updates["Sub-device"] = canonical_sub

    # --- Standardized Model + Connectivity ---
    raw_model = row.get("Standardized Model")
    if not _is_blank(raw_model):
        model_text = normalize_text(raw_model)
        model_text = strip_brand_words(model_text)
        stripped_model, embedded_connectivity = extract_model_connectivity(
            model_text
        )

        canonical_model = display_maps["model"].get(
            (device_key, stripped_model)
        )
        if canonical_model and canonical_model != raw_model:
            updates["Standardized Model"] = canonical_model

        raw_connectivity = row.get("Connectivity")
        explicit_connectivity = normalize_connectivity_value(
            raw_connectivity
        )

        connectivity_token = explicit_connectivity or embedded_connectivity

        if connectivity_token:
            canonical_connectivity = display_maps["connectivity"].get(
                (device_key, connectivity_token)
            )
            if (
                canonical_connectivity
                and canonical_connectivity != raw_connectivity
            ):
                updates["Connectivity"] = canonical_connectivity

    return updates


# ============================================================
# APPLE MODEL IDENTITY
# ============================================================

_ORDINAL_GENERATION_RE = re.compile(
    r"\b(\d{1,2})(st|nd|rd|th)\b",
    re.IGNORECASE,
)

_ALPHANUMERIC_GENERATION_RE = re.compile(
    r"\b(\d{1,2})([a-z])\b",
    re.IGNORECASE,
)

_CHIPSET_CODE_RE = re.compile(
    r"\b[am]\d+[a-z]?\b",
    re.IGNORECASE,
)

_VARIANT_TOKENS = {
    "air",
    "mini",
    "pro",
    "max",
    "plus",
    "se",
    "ultra",
    "studio",
}


def extract_apple_model_identity(model_text):
    """
    Extract coarse Apple model identity components used to prevent
    obviously different models from being treated as the same
    configuration.

    This is NOT the business classifier.
    """
    if not model_text:
        return {
            "product_line": None,
            "variant": None,
            "generations": set(),
            "model_designations": set(),
            "screen_sizes": set(),
            "identity_tokens": set(),
        }

    text = normalize_text(model_text) or ""
    tokens = text.split()

    product_line = None

    product_line_candidates = [
        "iphone",
        "ipad",
        "macbook",
        "imac",
        "mac mini",
        "mac studio",
        "mac pro",
        "apple watch",
        "airpods",
    ]

    for candidate in product_line_candidates:
        if candidate in text:
            product_line = candidate
            break

    variants = set()
    generations = set()
    model_designations = set()
    screen_sizes = set()

    for token in tokens:
        if token in _VARIANT_TOKENS:
            variants.add(token)

        if token.isdigit():
            generations.add(token)

        if _CHIPSET_CODE_RE.fullmatch(token):
            # Chipset codes are not model generations.
            continue

        if re.fullmatch(r"\d{1,2}(st|nd|rd|th)", token):
            match = _ORDINAL_GENERATION_RE.fullmatch(token)
            if match:
                generations.add(match.group(1))
            continue

        if re.fullmatch(r"\d{1,2}[a-z]", token):
            match = _ALPHANUMERIC_GENERATION_RE.fullmatch(token)
            if match:
                model_designations.add(
                    f"{match.group(1)}{match.group(2)}"
                )

    # Physical screen/display size is part of model identity.
    # Examples:
    #   "11-inch" -> "11"
    #   "13-inch" -> "13"
    # This prevents otherwise-similar configurations such as
    # iPad Air 11-inch and iPad Air 13-inch from matching.
    for match in re.finditer(
        r"\b(\d+(?:\.\d+)?)\s*[- ]?inch\b",
        text,
    ):
        screen_sizes.add(match.group(1))

    identity_tokens = set(tokens)

    return {
        "product_line": product_line,
        "variant": " ".join(sorted(variants)) if variants else None,
        "generations": generations,
        "model_designations": model_designations,
        "screen_sizes": screen_sizes,
        "identity_tokens": identity_tokens,
    }


def _designation_numeric_bases(designations):
    """
    Extract the leading numeric portion of alphanumeric model
    designations (e.g. "16e" -> "16").

    Used only to cross-check purely numeric generation tokens
    (e.g. "99") against alphanumeric designation tokens on the
    other side (e.g. "16e"), so that clearly unrelated generations
    are not silently treated as candidates just because one side
    happens to use a bare number and the other an alphanumeric
    designation.
    """
    bases = set()

    for designation in designations:
        match = re.match(r"(\d+)", designation)
        if match:
            bases.add(match.group(1))

    return bases


def model_identity_conflict(incoming_model, master_model):
    """
    Return True when two model names contain an explicit identity
    difference that makes them unsafe to treat as the same model.
    """
    incoming = extract_apple_model_identity(incoming_model)
    master = extract_apple_model_identity(master_model)

    if (
        incoming["product_line"]
        and master["product_line"]
        and incoming["product_line"] != master["product_line"]
    ):
        return True

    if (
        incoming["variant"]
        and master["variant"]
        and incoming["variant"] != master["variant"]
    ):
        return True

    if (
        incoming["generations"]
        and master["generations"]
        and incoming["generations"].isdisjoint(master["generations"])
    ):
        return True

    if (
        incoming["model_designations"]
        and master["model_designations"]
        and incoming["model_designations"].isdisjoint(
            master["model_designations"]
        )
    ):
        return True

    # Physical screen/display size is a hard model-identity distinction
    # when both sides explicitly provide a size.
    #
    # Examples:
    #   13-inch vs 11-inch -> conflict
    #   11-inch vs 11-inch -> compatible
    #
    # This prevents models such as iPad Air 13-inch and iPad Air
    # 11-inch from being treated as the same configuration.
    if (
        incoming["screen_sizes"]
        and master["screen_sizes"]
        and incoming["screen_sizes"].isdisjoint(
            master["screen_sizes"]
        )
    ):
        return True

    # Cross-check: a purely numeric generation on one side (e.g.
    # "99") must not be treated as a usable candidate for an
    # alphanumeric designation on the other side (e.g. "16e") when
    # the underlying generation numbers don't even overlap. This
    # only fires when one side has NO designation of its own to
    # compare directly (the direct designation-vs-designation check
    # above already handles that case), so a bare generation like
    # "16" is deliberately still allowed to be a candidate against
    # a designation like "16e" that shares the same numeric base -
    # that distinction is left to downstream exact-model matching
    # rather than being forced into a hard conflict here.
    incoming_bases = incoming["generations"] | _designation_numeric_bases(
        incoming["model_designations"]
    )
    master_bases = master["generations"] | _designation_numeric_bases(
        master["model_designations"]
    )

    if (
        incoming_bases
        and master_bases
        and incoming_bases.isdisjoint(master_bases)
    ):
        return True

    return False


# ============================================================
# CHIPSET COMPARISON
# ============================================================

def chipset_similarity(incoming, master):
    if _is_unknown(incoming) or _is_unknown(master):
        return None

    incoming_text = normalize_text(incoming)
    master_text = normalize_text(master)

    if not incoming_text or not master_text:
        return None

    if incoming_text == master_text:
        return 1.0

    # A-series / M-series family comparison.
    incoming_code = _CHIPSET_CODE_RE.search(incoming_text)
    master_code = _CHIPSET_CODE_RE.search(master_text)

    if incoming_code and master_code:
        incoming_family = re.match(
            r"([am]\d+)",
            incoming_code.group(0),
            re.IGNORECASE,
        )
        master_family = re.match(
            r"([am]\d+)",
            master_code.group(0),
            re.IGNORECASE,
        )

        if (
            incoming_family
            and master_family
            and incoming_family.group(1).lower()
            == master_family.group(1).lower()
        ):
            return 1.0

    return difflib.SequenceMatcher(
        None,
        incoming_text,
        master_text,
    ).ratio()


# ============================================================
# COLUMN MAPPING
# ============================================================

def _normalize_column_name(name):
    text = normalize_text(name)
    return text or ""


def propose_column_mapping(raw_columns):
    """
    Propose a raw-column -> canonical-column mapping.

    Returns:
        {
            "mapping": {...},
            "unmapped_columns": [...],
            "missing_required": [...]
        }
    """
    normalized_raw = {
        column: _normalize_column_name(column)
        for column in raw_columns
    }

    normalized_synonyms = {
        canonical: {
            _normalize_column_name(alias)
            for alias in aliases
        }
        for canonical, aliases in COLUMN_SYNONYMS.items()
    }

    mapping = {}
    used_raw = set()

    # --------------------------------------------------------
    # Pass 1: exact normalized matches
    # --------------------------------------------------------
    for canonical in CANONICAL_FIELDS:
        candidates = normalized_synonyms.get(canonical, set())
        candidates.add(_normalize_column_name(canonical))

        for raw_column, normalized_raw_name in normalized_raw.items():
            if raw_column in used_raw:
                continue

            if normalized_raw_name in candidates:
                mapping[raw_column] = canonical
                used_raw.add(raw_column)
                break

    # --------------------------------------------------------
    # Pass 2: conservative substring matches
    # --------------------------------------------------------
    for canonical in CANONICAL_FIELDS:
        if canonical in mapping.values():
            continue

        aliases = normalized_synonyms.get(canonical, set())
        aliases.add(_normalize_column_name(canonical))

        best_raw = None
        best_score = 0

        for raw_column, normalized_raw_name in normalized_raw.items():
            if raw_column in used_raw:
                continue

            if not normalized_raw_name:
                continue

            score = 0

            for alias in aliases:
                if not alias:
                    continue

                if alias == normalized_raw_name:
                    score = max(score, 100)
                elif alias in normalized_raw_name:
                    score = max(score, 70)
                elif normalized_raw_name in alias:
                    score = max(score, 60)

            if score > best_score:
                best_score = score
                best_raw = raw_column

        if best_raw is not None and best_score >= 60:
            mapping[best_raw] = canonical
            used_raw.add(best_raw)

    unmapped_columns = [
        column
        for column in raw_columns
        if column not in used_raw
    ]

    # These are the fields needed to establish a usable row identity.
    required = [
        "Device",
        "Standardized Model",
    ]

    missing_required = [
        field
        for field in required
        if field not in mapping.values()
    ]

    return {
        "mapping": mapping,
        "unmapped_columns": unmapped_columns,
        "missing_required": missing_required,
    }


def apply_column_mapping(raw_df, mapping):
    """
    Create a canonical DataFrame with exactly CANONICAL_FIELDS.

    Missing canonical columns remain NaN and will later normalize
    to Unknown.
    """
    canonical_df = pd.DataFrame(index=raw_df.index)

    reverse_mapping = {}

    for raw_column, canonical_column in mapping.items():
        reverse_mapping[canonical_column] = raw_column

    for canonical_column in CANONICAL_FIELDS:
        raw_column = reverse_mapping.get(canonical_column)

        if raw_column is None:
            canonical_df[canonical_column] = np.nan
        else:
            canonical_df[canonical_column] = raw_df[raw_column]

    return canonical_df


# ============================================================
# ROW NORMALIZATION
# ============================================================

def _get_mapped_value(row, field, mapped_fields):
    """
    Read a canonical field only if its source column actually existed.
    """
    if field not in mapped_fields:
        return None

    return row.get(field)


def normalize_row_for_matching(row, mapped_fields=None):
    """
    Convert a canonical row into the normalized representation used
    by matching and exact identity checks.

    Missing values become UNKNOWN at this layer.
    Invalid values are represented by "__INVALID__".
    """
    mapped_fields = set(mapped_fields or CANONICAL_FIELDS)

    raw_device = _get_mapped_value(
        row,
        "Device",
        mapped_fields,
    )

    raw_model = _get_mapped_value(
        row,
        "Standardized Model",
        mapped_fields,
    )

    device = normalize_device_value(raw_device)

    model_normalized = normalize_text(raw_model)
    model_normalized = strip_brand_words(model_normalized)

    model_text, embedded_connectivity = extract_model_connectivity(
        model_normalized
    )

    raw_connectivity = _get_mapped_value(
        row,
        "Connectivity",
        mapped_fields,
    )

    explicit_connectivity = normalize_connectivity_value(
        raw_connectivity
    )

    connectivity = (
        explicit_connectivity
        if explicit_connectivity is not None
        else embedded_connectivity
    )

    storage_raw = _get_mapped_value(
        row,
        "Storage (GB)",
        mapped_fields,
    )

    storage = normalize_numeric_for_comparison(
        storage_raw,
        "Storage (GB)",
    )

    model_year = normalize_numeric_for_comparison(
        _get_mapped_value(row, "Model_Year", mapped_fields),
        "Model_Year",
    )

    case_size = normalize_numeric_for_comparison(
        _get_mapped_value(row, "Case Size", mapped_fields),
        "Case Size",
    )

    retail_price = normalize_numeric_for_comparison(
        _get_mapped_value(row, "Retail Price", mapped_fields),
        "Retail Price",
    )

    trade_in_value = normalize_numeric_for_comparison(
        _get_mapped_value(
            row,
            "Max. Trade-In Value (RM)",
            mapped_fields,
        ),
        "Max. Trade-In Value (RM)",
    )

    def normalized_text_or_unknown(field):
        value = _get_mapped_value(row, field, mapped_fields)

        if _is_blank(value):
            return UNKNOWN

        normalized = normalize_text(value)
        return normalized if normalized else UNKNOWN

    return {
        "device": device or UNKNOWN,
        "sub_device": normalized_text_or_unknown("Sub-device"),
        "model_text": model_text or UNKNOWN,
        "storage": storage,
        "storage_type": normalized_text_or_unknown("Storage Type"),
        "connectivity": connectivity or UNKNOWN,
        "material": normalized_text_or_unknown("Material"),
        "model_year": model_year,
        "provider": normalized_text_or_unknown("Provider"),
        "chipset": normalized_text_or_unknown("Chipset"),
        "case_size": case_size,
        "charging_method": normalized_text_or_unknown(
            "Charging Method"
        ),
        "retail_price": retail_price,
        "trade_in_value": trade_in_value,
        "model_numbers": normalize_model_numbers(
            _get_mapped_value(
                row,
                "Model Number",
                mapped_fields,
            )
        ),
    }


# ============================================================
# INVALID DATA VALIDATION
# ============================================================

def _validate_numeric_field(value, field):
    """
    Validate a numeric field.

    Returns:
        None when valid/missing
        reason string when invalid
    """
    if _is_blank(value):
        return None

    _, invalid = _parse_numeric_value(value, field)

    if invalid:
        return f"{field} contains an invalid value: {value!r}"

    return None


def validate_row_data(row):
    """
    Validate supplied values before attempting Master matching.

    Important distinction:

        missing / blank -> Unknown -> may become Needs Review

        malformed supplied value -> Invalid Data

    This function does NOT classify missing data as invalid.
    """
    errors = []

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------
    for field in NUMERIC_FIELDS:
        error = _validate_numeric_field(
            row.get(field),
            field,
        )

        if error:
            errors.append(error)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device_value = row.get("Device")

    if not _is_blank(device_value):
        normalized_device = normalize_device_value(device_value)

        if normalized_device is None:
            errors.append(
                f"Device contains an unrecognized value: "
                f"{device_value!r}"
            )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model_value = row.get("Standardized Model")

    if not _is_blank(model_value):
        if not str(model_value).strip():
            errors.append(
                "Standardized Model contains an invalid value."
            )

    return errors


# ============================================================
# REQUIRED FIELD VALIDATION FOR ACTUAL NEW MASTER ROWS
# ============================================================

def validate_new_row_fields(
    device,
    sub_device,
    model_name,
    msrp,
    trade_in_value,
    storage,
    storage_type,
    chipset,
    connectivity,
    material,
    case_size,
    charging_method,
    model_year,
):
    """
    Validate fields required before a genuinely NEW row can be
    inserted into Master.

    This is intentionally separate from row classification.

    Classification answers:
        New / Duplicate / Needs Review / Invalid Data

    This function answers:
        Is the proposed NEW Master row structurally complete?
    """
    errors = []

    if _is_blank(device):
        errors.append("Device is required.")

    if _is_blank(sub_device):
        errors.append("Sub-device is required.")

    if _is_blank(model_name):
        errors.append("Standardized Model is required.")

    if _is_blank(msrp):
        errors.append("Retail Price is required.")
    else:
        parsed, invalid = _parse_numeric_value(
            msrp,
            "Retail Price",
        )

        if invalid:
            errors.append("Retail Price must be numeric.")
        elif parsed < 0:
            errors.append("Retail Price cannot be negative.")

    if not _is_blank(trade_in_value):
        parsed, invalid = _parse_numeric_value(
            trade_in_value,
            "Max. Trade-In Value (RM)",
        )

        if invalid:
            errors.append(
                "Max. Trade-In Value (RM) must be numeric."
            )
        elif parsed < 0:
            errors.append(
                "Max. Trade-In Value (RM) cannot be negative."
            )

    normalized_device = normalize_device_value(device)

    if normalized_device != "AirPods":
        if _is_blank(storage):
            errors.append(
                "Storage (GB) is required for non-AirPods."
            )

    if normalized_device == "Mac":
        if _is_blank(storage_type):
            errors.append(
                "Storage Type is required for Mac."
            )

    if normalized_device in {"iPad", "Apple Watch"}:
        if _is_blank(connectivity):
            errors.append(
                "Connectivity is required for this device."
            )

    if normalized_device == "Apple Watch":
        if _is_blank(material):
            errors.append(
                "Material is required for Apple Watch."
            )

        if _is_blank(case_size):
            errors.append(
                "Case Size is required for Apple Watch."
            )

    if normalized_device == "AirPods":
        if _is_blank(charging_method):
            errors.append(
                "Charging Method is required for AirPods."
            )

    if _is_blank(model_year):
        errors.append("Model Year is required.")
    else:
        parsed, invalid = _parse_numeric_value(
            model_year,
            "Model_Year",
        )

        if invalid:
            errors.append("Model Year must be numeric.")
        elif parsed < 1976:
            errors.append("Model Year must be 1976 or later.")

    return errors


# ============================================================
# MATCHING REPRESENTATION
# ============================================================

def _configuration_values(norm):
    device = norm.get("device")

    values = {
        "Device": norm["device"],
        "Sub-device": norm["sub_device"],
        "Standardized Model": norm["model_text"],
        "Model Number": norm["model_numbers"],
    }

    field_mapping = {
        "Storage (GB)": "storage",
        "Storage Type": "storage_type",
        "Connectivity": "connectivity",
        "Material": "material",
        "Model_Year": "model_year",
        "Case Size": "case_size",
        "Charging Method": "charging_method",
    }

    for field, key in field_mapping.items():
        if is_field_applicable(device, field):
            values[field] = norm[key]

    return values


def _field_value(norm, field):
    mapping = {
        "Device": "device",
        "Sub-device": "sub_device",
        "Standardized Model": "model_text",
        "Storage (GB)": "storage",
        "Storage Type": "storage_type",
        "Connectivity": "connectivity",
        "Material": "material",
        "Model_Year": "model_year",
        "Chipset": "chipset",
        "Case Size": "case_size",
        "Charging Method": "charging_method",
        "Provider": "provider",
        "Retail Price": "retail_price",
        "Max. Trade-In Value (RM)": "trade_in_value",
    }

    return norm.get(mapping[field])


def _has_invalid_normalized_value(norm):
    invalid_marker = "__INVALID__"

    for key, value in norm.items():
        if value == invalid_marker:
            return True

    return False

def extract_mac_configuration(model_text):
    """
    Extract configuration-level details from Mac model text.

    This is used only for candidate matching. It does not decide
    the final business classification.

    Examples of configuration differences this can detect:
      - i3 vs i5 vs i7 vs i9
      - 1.1GHz vs 1.2GHz
      - M1 vs M2
      - M1 Pro vs M1 Max
      - 8-Core CPU vs 10-Core CPU
      - 7-Core GPU vs 8-Core GPU
      - 24-Core GPU vs 32-Core GPU
    """
    if _is_blank(model_text):
        return {
            "chip": None,
            "cpu_tier": None,
            "cpu_frequency": None,
            "cpu_cores": None,
            "gpu_cores": None,
        }

    text = normalize_text(model_text) or ""

    chip = None
    cpu_tier = None
    cpu_frequency = None
    cpu_cores = None
    gpu_cores = None

    # Apple Silicon chip/family.
    #
    # Examples:
    #   M1
    #   M1 Pro
    #   M1 Max
    #   M2
    #   M2 Pro
    #   M2 Max
    chip_match = re.search(
        r"\bm[1-4](?:\s+(?:pro|max|ultra))?\b",
        text,
    )

    if chip_match:
        chip = chip_match.group(0)

    # Intel CPU tier.
    #
    # Examples:
    #   i3
    #   i5
    #   i7
    #   i9
    cpu_tier_match = re.search(
        r"\bi[3579]\b",
        text,
    )

    if cpu_tier_match:
        cpu_tier = cpu_tier_match.group(0)

    # CPU frequency.
    #
    # Example:
    #   1.1GHz -> 1.1
    #   2.6GHz -> 2.6
    frequency_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*ghz\b",
        text,
    )

    if frequency_match:
        cpu_frequency = float(
            frequency_match.group(1)
        )

    # CPU core count.
    #
    # Example:
    #   8-Core CPU -> 8
    #   10-Core CPU -> 10
    cpu_core_match = re.search(
        r"\b(\d+)\s*core\s+cpu\b",
        text,
    )

    if cpu_core_match:
        cpu_cores = int(
            cpu_core_match.group(1)
        )

    # GPU core count.
    #
    # Example:
    #   7-Core GPU -> 7
    #   8-Core GPU -> 8
    #   10-Core GPU -> 10
    #   32-Core GPU -> 32
    gpu_core_match = re.search(
        r"\b(\d+)\s*core\s+gpu\b",
        text,
    )

    if gpu_core_match:
        gpu_cores = int(
            gpu_core_match.group(1)
        )

    return {
        "chip": chip,
        "cpu_tier": cpu_tier,
        "cpu_frequency": cpu_frequency,
        "cpu_cores": cpu_cores,
        "gpu_cores": gpu_cores,
    }

def _mac_configuration_match_score(
    incoming_model,
    master_model,
):
    """
    Compare configuration-level Mac details extracted from
    Standardized Model text.

    Known matching attributes contribute positively.
    A known disagreement contributes negatively by receiving
    a zero for that attribute.

    Missing information on either side is neutral.

    Returns:
        None when there is no comparable configuration data.
        Otherwise a score from 0.0 to 1.0.
    """
    incoming = extract_mac_configuration(
        incoming_model
    )

    master = extract_mac_configuration(
        master_model
    )

    comparisons = [
        "chip",
        "cpu_tier",
        "cpu_frequency",
        "cpu_cores",
        "gpu_cores",
    ]

    score = 0.0
    weight = 0.0

    for key in comparisons:
        incoming_value = incoming.get(key)
        master_value = master.get(key)

        if incoming_value is None or master_value is None:
            continue

        weight += 1.0

        if incoming_value == master_value:
            score += 1.0

    if weight == 0:
        return None

    return score / weight

# ============================================================
# MODEL MATCH SCORING
# ============================================================

def _model_similarity(incoming_model, master_model):
    if (
        _is_unknown(incoming_model)
        or _is_unknown(master_model)
    ):
        return None

    incoming = strip_brand_words(
        normalize_text(incoming_model) or ""
    )

    master = strip_brand_words(
        normalize_text(master_model) or ""
    )

    if not incoming or not master:
        return None

    if incoming == master:
        return 1.0

    if model_identity_conflict(incoming, master):
        return 0.0

    sequence_score = difflib.SequenceMatcher(
        None,
        incoming,
        master,
    ).ratio()

    incoming_tokens = set(incoming.split())
    master_tokens = set(master.split())

    if incoming_tokens and master_tokens:
        intersection = len(
            incoming_tokens & master_tokens
        )

        union = len(
            incoming_tokens | master_tokens
        )

        token_score = (
            intersection / union
            if union
            else 0.0
        )
    else:
        token_score = 0.0

    generic_score = max(
        sequence_score,
        token_score,
    )

    # --------------------------------------------------------
    # Mac configuration-aware scoring.
    #
    # Generic text similarity is not sufficient for Mac models
    # because materially different configurations can have nearly
    # identical names.
    #
    # Example:
    #   M2 / 8-core CPU / 8-core GPU
    #   M2 / 8-core CPU / 10-core GPU
    #
    # The configuration score makes those differences meaningful
    # during candidate selection.
    # --------------------------------------------------------
    configuration_score = _mac_configuration_match_score(
        incoming,
        master,
    )

    if configuration_score is not None:
        return (
            (generic_score * 0.60)
            + (configuration_score * 0.40)
        )

    return generic_score


def _structured_field_match_score(
    incoming_norm,
    master_norm,
):
    """
    Score applicable structured configuration fields.

    Fields that do not apply to the device are ignored.

    Unknown is neutral during candidate discovery.
    A known disagreement lowers the score.
    """
    device = incoming_norm.get("device")

    comparisons = [
        ("Storage (GB)", "storage"),
        ("Storage Type", "storage_type"),
        ("Connectivity", "connectivity"),
        ("Material", "material"),
        ("Model_Year", "model_year"),
        ("Case Size", "case_size"),
        ("Charging Method", "charging_method"),
    ]

    score = 0.0
    weight = 0.0

    for field_name, key in comparisons:
        if not is_field_applicable(device, field_name):
            continue

        incoming = incoming_norm.get(key)
        master = master_norm.get(key)

        if _is_unknown(incoming) or _is_unknown(master):
            continue

        weight += 1.0

        if incoming == master:
            score += 1.0

    if weight == 0:
        return None

    return score / weight


# ============================================================
# MASTER CANDIDATE DISCOVERY
# ============================================================

def score_candidate(incoming_norm, master_norm):
    """
    Score one Master row as a possible Apple configuration
    counterpart.

    IMPORTANT:
    This function only answers:
        "How well does this Master row represent the same
         Apple configuration?"

    It does NOT decide:
        New / Duplicate / Needs Review / Invalid Data.
    """
    incoming_model = incoming_norm.get("model_text")
    master_model = master_norm.get("model_text")

    model_score = _model_similarity(
        incoming_model,
        master_model,
    )

    if model_score is None:
        model_score = 0.0

    structured_score = _structured_field_match_score(
        incoming_norm,
        master_norm,
    )

    if structured_score is None:
        structured_score = 0.0

    # Model identity is the strongest configuration signal.
    confidence = (
        (model_score * 0.70)
        + (structured_score * 0.30)
    )

    conflicts = []

    if model_identity_conflict(
        incoming_model,
        master_model,
    ):
        conflicts.append("model_identity_conflict")

    # Known structured disagreements.
    structured_fields = [
        ("Storage (GB)", "storage"),
        ("Storage Type", "storage_type"),
        ("Connectivity", "connectivity"),
        ("Material", "material"),
        ("Model_Year", "model_year"),
        ("Case Size", "case_size"),
        ("Charging Method", "charging_method"),
    ]

    for field_name, key in structured_fields:
        if not is_field_applicable(
            incoming_norm.get("device"),
            field_name,
        ):
            continue

        incoming = incoming_norm.get(key)
        master = master_norm.get(key)

        if _is_unknown(incoming) or _is_unknown(master):
            continue

        if incoming != master:
            conflicts.append(field_name)

    model_number_status = compare_model_numbers(
        incoming_norm,
        master_norm,
    )

    return {
        "confidence": round(confidence, 4),
        "model_score": round(model_score, 4),
        "structured_score": round(structured_score, 4),
        "model_number_match": model_number_status,
        "model_identity_conflict": bool(
            "model_identity_conflict" in conflicts
        ),
        "conflicts": conflicts,
    }


def find_master_candidates(
    incoming_norm,
    master_norm_by_index,
    master_device_by_index,
    master_provider_by_index=None,
):
    """
    Find Master rows corresponding to the incoming Apple
    configuration.

    Provider is deliberately NOT a hard filter.

    A different Provider does not mean a different Apple
    configuration.
    """
    incoming_device = incoming_norm.get("device")

    if _is_unknown(incoming_device):
        return []

    candidates = []

    for index, master_norm in master_norm_by_index.items():
        master_device = master_device_by_index.get(index)

        if master_device != incoming_device:
            continue

        result = score_candidate(
            incoming_norm,
            master_norm,
        )

        # Explicit model identity conflict means this row is not
        # a usable configuration counterpart.
        if result["model_identity_conflict"]:
            continue

        # A candidate needs at least a meaningful model match.
        if result["model_score"] < 0.65:
            continue

        candidates.append({
            "index": index,
            "norm": master_norm,
            **result,
        })

    # Preserve deterministic Master row order.
    # Option B: when multiple suitable Master rows exist,
    # use the first suitable row.
    return candidates


# ============================================================
# EXACT VALUE COMPARISON
# ============================================================

def compare_model_numbers(incoming_norm, master_norm):
    """
    Determine the relationship between incoming and Master model
    numbers.

    Returns:
        "both_absent"
        "master_only"
        "incoming_only"
        "match"          -> incoming model number already exists
        "append"         -> incoming model number is new, but the
                            rest of the row matches
        "mismatch"       -> no model number overlap
    """
    incoming = incoming_norm.get(
        "model_numbers",
        frozenset(),
    )

    master = master_norm.get(
        "model_numbers",
        frozenset(),
    )

    incoming_present = bool(incoming)
    master_present = bool(master)

    if not incoming_present and not master_present:
        return "both_absent"

    if not incoming_present and master_present:
        return "master_only"

    if incoming_present and not master_present:
        return "incoming_only"

    # Any overlap means the incoming model number is already
    # represented by this Master row.
    if incoming & master:
        return "match"

    # No overlap means the incoming model number is new.
    # Whether it should be appended depends on the rest of the
    # Master-relevant row matching.
    return "append"


def _values_equal(field, incoming_norm, master_norm):
    """
    Compare one canonical field for exact duplicate detection.

    Returns:
        True
        False
        None -> one/both values are Unknown
    """
    if field == "Model Number":
        incoming = incoming_norm.get(
            "model_numbers",
            frozenset(),
        )
        master = master_norm.get(
            "model_numbers",
            frozenset(),
        )

        if not incoming or not master:
            return None

        # Per the established Model Number rule: any overlap means
        # the incoming Model Number is already represented by the
        # Master row (e.g. incoming "A" against Master "A, B, C" is
        # an existing match, not a difference). Only a genuinely
        # disjoint set is a real difference; that case is handled
        # separately by the append logic in classify_incoming_row.
        return bool(incoming & master)

    incoming = _field_value(
        incoming_norm,
        field,
    )

    master = _field_value(
        master_norm,
        field,
    )

    if _is_unknown(incoming) or _is_unknown(master):
        return None

    return incoming == master


def compare_master_relevant_fields(
    incoming_norm,
    master_norm,
):
    """
    Compare all Master-relevant fields that apply to the device.

    Non-applicable device fields are ignored.

    Retail Price special rule:
        - If incoming Retail Price is supplied, compare it normally.
        - If incoming Retail Price is missing, use the Master's
          Retail Price as the effective incoming value.
        - Missing incoming Retail Price therefore does NOT create
          an Unknown or a difference.

    Returns:
        {
            "exact": bool,
            "unknown_fields": [...],
            "different_fields": [...]
        }
    """
    unknown_fields = []
    different_fields = []

    device = incoming_norm.get("device")

    for field in DUPLICATE_FIELDS:
        if not is_field_applicable(device, field):
            continue

        # ----------------------------------------------------
        # Retail Price is optional in incoming uploads.
        #
        # If the incoming source did not provide a Retail Price,
        # inherit the Master's Retail Price for comparison.
        # ----------------------------------------------------
        if field == "Retail Price":
            incoming = incoming_norm.get("retail_price")
            master = master_norm.get("retail_price")

            if _is_unknown(incoming):
                if _is_unknown(master):
                    unknown_fields.append(field)
                continue

            if _is_unknown(master):
                unknown_fields.append(field)
                continue

            if incoming != master:
                different_fields.append(field)

            continue

        equal = _values_equal(
            field,
            incoming_norm,
            master_norm,
        )

        if equal is None:
            unknown_fields.append(field)
        elif not equal:
            different_fields.append(field)

    return {
        "exact": (
            not unknown_fields
            and not different_fields
        ),
        "unknown_fields": unknown_fields,
        "different_fields": different_fields,
    }


# ============================================================
# CONFIGURATION COUNTERPART CHECK
# ============================================================

def configuration_matches(
    incoming_norm,
    master_norm,
):
    """
    Determine whether the incoming row and Master row represent
    the same Apple configuration.

    Provider and price are NOT used here.

    Returns:
        {
            "matches": bool,
            "unknown_fields": [...],
            "different_fields": [...]
        }
    """
    unknown_fields = []
    different_fields = []

    # Device
    incoming_device = incoming_norm.get("device")
    master_device = master_norm.get("device")

    if _is_unknown(incoming_device):
        unknown_fields.append("Device")
    elif _is_unknown(master_device):
        unknown_fields.append("Device")
    elif incoming_device != master_device:
        different_fields.append("Device")

    # Sub-device
    incoming_sub = incoming_norm.get("sub_device")
    master_sub = master_norm.get("sub_device")

    if _is_unknown(incoming_sub) or _is_unknown(master_sub):
        unknown_fields.append("Sub-device")
    elif incoming_sub != master_sub:
        different_fields.append("Sub-device")

    # Model
    incoming_model = incoming_norm.get("model_text")
    master_model = master_norm.get("model_text")

    if _is_unknown(incoming_model) or _is_unknown(master_model):
        unknown_fields.append("Standardized Model")
    else:
        if model_identity_conflict(
            incoming_model,
            master_model,
        ):
            different_fields.append(
                "Standardized Model"
            )
        elif _model_similarity(
            incoming_model,
            master_model,
        ) < 0.80:
            different_fields.append(
                "Standardized Model"
            )

        # Structured fields
        for field_name, key in [
            ("Storage (GB)", "storage"),
            ("Storage Type", "storage_type"),
            ("Connectivity", "connectivity"),
            ("Material", "material"),
            ("Model_Year", "model_year"),
            ("Case Size", "case_size"),
            ("Charging Method", "charging_method"),
        ]:
            if not is_field_applicable(
                incoming_norm.get("device"),
                field_name,
            ):
                continue

            incoming = incoming_norm.get(key)
            master = master_norm.get(key)

            if _is_unknown(incoming) or _is_unknown(master):
                unknown_fields.append(field_name)
                continue

            if incoming != master:
                different_fields.append(field_name)

    return {
        "matches": not different_fields,
        "unknown_fields": sorted(set(unknown_fields)),
        "different_fields": sorted(set(different_fields)),
    }


# ============================================================
# INTERNAL MULTI-MATCH HANDLING
# ============================================================

def choose_reference_candidate(candidates, incoming_provider=None):
    """
    Select the Master row to use as the reference configuration.

    Candidate discovery does not use Provider as a hard filter.

    Selection priority:
        1. Prefer candidates from the same Provider.
        2. Within the same Provider group, choose the highest-confidence
           candidate.
        3. If no same-Provider candidate exists, choose the
           highest-confidence candidate overall.

    Multiple matches remain internal only (multi_match).
    """
    if not candidates:
        return None

    if not _is_unknown(incoming_provider):
        same_provider_candidates = [
            candidate
            for candidate in candidates
            if (
                not _is_unknown(
                    candidate["norm"].get("provider")
                )
                and candidate["norm"].get("provider")
                == incoming_provider
            )
        ]

        if same_provider_candidates:
            return max(
                same_provider_candidates,
                key=lambda candidate: candidate["confidence"],
            )

    return max(
        candidates,
        key=lambda candidate: candidate["confidence"],
    )


# ============================================================
# BUSINESS CLASSIFICATION
# ============================================================

def classify_incoming_row(
    incoming_row_canonical,
    incoming_norm,
    master_df,
    master_norm_by_index,
    master_device_by_index,
    master_provider_by_index,
    mapped_canonical_fields=None,
):
    """
    Classify one incoming row as exactly one of:

        new
        update
        conflict
            (conflict_type: duplicate | needs_review | invalid_data)

    Admin-facing hierarchy:

        NEW

        UPDATE

        CONFLICT
        ├── Duplicate
        ├── Needs Review
        └── Invalid Data

    `classification` is always one of {"new", "update", "conflict"}.
    `conflict_type` is set only when classification == "conflict"
    (one of {"duplicate", "needs_review", "invalid_data"}); it is
    None otherwise.

    Internal multi_match handling never becomes a classification.

    Business rules:

    1. Invalid supplied data -> conflict / invalid_data.
    2. No Master configuration counterpart -> new.
    3. Configuration counterpart exists but required comparison
       information contains Unknown -> conflict / needs_review.
    4. Exact Master-relevant row match with no Unknown -> conflict /
       duplicate.
    5. Same Apple configuration but different valid Provider -> new,
       regardless of any pricing difference.
    6. Same Provider + same Apple configuration + Retail Price
       changed (with or without a Trade-In change too) -> conflict /
       needs_review. Retail Price always takes precedence over
       Trade-In.
    7. Same Provider + same Apple configuration + only other mutable
       value(s) changed (Trade-In Value and/or similar), Retail
       Price unchanged -> update.
    8. Any other valid, non-duplicate row is new.
    """
    reasons = []
    warnings = []

    mapped_canonical_fields = (
        mapped_canonical_fields or set()
    )

    # --------------------------------------------------------
    # 1. Validate supplied values
    # --------------------------------------------------------
    validation_errors = validate_row_data(
        incoming_row_canonical
    )

    if validation_errors:
        return {
            "classification": "conflict",
            "conflict_type": "invalid_data",
            "reasons": validation_errors,
            "warnings": [],
            "match": None,
            "match_index": None,
            "multi_match": False,
            "unknown_fields": [],
            "different_fields": [],
            "model_numbers_to_append": [],
            "model_number_update_required": False,
            "model_number_flag": None,
        }

    # --------------------------------------------------------
    # 2. Device must be identifiable
    # --------------------------------------------------------
    incoming_device = incoming_norm.get("device")

    if _is_unknown(incoming_device):
        return {
            "classification": "conflict",
            "conflict_type": "needs_review",
            "reasons": [
                "Device could not be identified."
            ],
            "warnings": [],
            "match": None,
            "match_index": None,
            "multi_match": False,
            "unknown_fields": ["Device"],
            "different_fields": [],
            "model_numbers_to_append": [],
            "model_number_update_required": False,
            "model_number_flag": None,
        }

    # --------------------------------------------------------
    # 3. Find Apple configuration counterparts.
    #
    # Provider is intentionally ignored as a candidate filter.
    # --------------------------------------------------------
    candidates = find_master_candidates(
        incoming_norm,
        master_norm_by_index,
        master_device_by_index,
        master_provider_by_index,
    )

    if not candidates:
        return {
            "classification": "new",
            "conflict_type": None,
            "reasons": [
                "No Master counterpart exists for this "
                "Apple model/configuration."
            ],
            "warnings": [],
            "match": None,
            "match_index": None,
            "multi_match": False,
            "unknown_fields": [],
            "different_fields": [],
            "model_numbers_to_append": [],
            "model_number_update_required": False,
            "model_number_flag": None,
        }

    # --------------------------------------------------------
    # 4. Choose one reference candidate.
    #
    # Multiple equally usable candidates are internal only.
    # --------------------------------------------------------
    best_candidate = choose_reference_candidate(
        candidates,
        incoming_norm.get("provider"),
    )

    multi_match = len(candidates) > 1

    # --------------------------------------------------------
    # 4a. Retail Price inheritance.
    #
    # Incoming datasets may omit Retail Price entirely because
    # their generic "Price" field represents Trade-In Value.
    #
    # If the incoming Retail Price is missing, inherit the
    # selected Master's Retail Price. This prevents a missing
    # source value from being treated as a change.
    #
    # An explicitly supplied Retail Price is never overwritten.
    # --------------------------------------------------------
    incoming_retail_price = incoming_norm.get("retail_price")
    master_retail_price = best_candidate["norm"].get(
        "retail_price"
    )

    if (
        _is_unknown(incoming_retail_price)
        and not _is_unknown(master_retail_price)
    ):
        incoming_norm["retail_price"] = master_retail_price

        # Keep the canonical incoming row consistent with the
        # effective value used for classification.
        incoming_row_canonical["Retail Price"] = (
            master_retail_price
        )

    incoming_storage_type = incoming_norm.get(
        "storage_type"
    )
    master_storage_type = best_candidate["norm"].get(
        "storage_type"
    )

    # Only inherit Storage Type when the incoming source file
    # did not provide a Storage Type field at all.
    #
    # If the source contains the field but this particular row
    # has a blank/Unknown value, preserve Unknown so that the
    # row correctly becomes Needs Review.
    if (
        "Storage Type" not in mapped_canonical_fields
        and _is_unknown(incoming_storage_type)
        and not _is_unknown(master_storage_type)
    ):
        incoming_norm["storage_type"] = master_storage_type

        # Keep the canonical incoming row consistent with the
        # effective value used for classification.
        incoming_row_canonical["Storage Type"] = (
            master_storage_type
        )

    incoming_master_comparison = configuration_matches(
        incoming_norm,
        best_candidate["norm"],
    )

    # --------------------------------------------------------
    # 5. If configuration itself is ambiguous due to Unknown,
    #    Admin needs to review it.
    # --------------------------------------------------------
    if incoming_master_comparison["unknown_fields"]:
        reasons.append(
            "A Master counterpart exists, but one or more "
            "relevant fields are Unknown."
        )

        return {
            "classification": "conflict",
            "conflict_type": "needs_review",
            "reasons": reasons,
            "warnings": warnings,
            "match": best_candidate,
            "match_index": best_candidate["index"],
            "multi_match": multi_match,
            "unknown_fields": (
                incoming_master_comparison[
                    "unknown_fields"
                ]
            ),
            "different_fields": (
                incoming_master_comparison[
                    "different_fields"
                ]
            ),
            "model_numbers_to_append": [],
            "model_number_update_required": False,
            "model_number_flag": compare_model_numbers(
                incoming_norm,
                best_candidate["norm"],
            ),
        }

    # --------------------------------------------------------
    # 6. If there is a known configuration disagreement, this
    #    incoming row is a valid new Master row.
    #
    #    There is no generic "conflict" classification anymore.
    # --------------------------------------------------------
    if incoming_master_comparison["different_fields"]:
        reasons.append(
            "The incoming row represents a valid row that is "
            "not an exact match for the selected Master "
            "configuration."
        )

        return {
            "classification": "new",
            "conflict_type": None,
            "reasons": reasons,
            "warnings": warnings,
            "match": best_candidate,
            "match_index": best_candidate["index"],
            "multi_match": multi_match,
            "unknown_fields": [],
            "different_fields": (
                incoming_master_comparison[
                    "different_fields"
                ]
            ),
            "model_numbers_to_append": [],
            "model_number_update_required": False,
            "model_number_flag": compare_model_numbers(
                incoming_norm,
                best_candidate["norm"],
            ),
        }

    # --------------------------------------------------------
    # 7. Apple configuration matches.
    #
    # Now compare the COMPLETE Master-relevant row.
    # --------------------------------------------------------
    full_comparison = compare_master_relevant_fields(
        incoming_norm,
        best_candidate["norm"],
    )

    model_number_flag = compare_model_numbers(
        incoming_norm,
        best_candidate["norm"],
    )

    # --------------------------------------------------------
    # 8. Unknown in the full row -> Needs Review.
    #
    # Duplicate requires zero Unknown values.
    # --------------------------------------------------------
    if full_comparison["unknown_fields"]:
        reasons.append(
            "The Master counterpart exists, but one or more "
            "Master-relevant fields are Unknown."
        )

        return {
            "classification": "conflict",
            "conflict_type": "needs_review",
            "reasons": reasons,
            "warnings": warnings,
            "match": best_candidate,
            "match_index": best_candidate["index"],
            "multi_match": multi_match,
            "unknown_fields": full_comparison[
                "unknown_fields"
            ],
            "different_fields": full_comparison[
                "different_fields"
            ],
            "model_numbers_to_append": [],
            "model_number_update_required": False,
            "model_number_flag": compare_model_numbers(
                incoming_norm,
                best_candidate["norm"],
            ),
        }

    # --------------------------------------------------------
    # 9. Model Number handling.
    #
    # If the incoming Model Number already exists in Master,
    # it is already represented by that Master row.
    #
    # If the incoming Model Number is new but every other
    # relevant value matches, append it to the existing Master
    # Model Number field instead of creating a new row.
    # --------------------------------------------------------
    if model_number_flag == "append":
        non_model_number_differences = [
            field
            for field in full_comparison["different_fields"]
            if field != "Model Number"
        ]

        if (
            not full_comparison["unknown_fields"]
            and not non_model_number_differences
        ):
            existing_model_numbers = set(
                best_candidate["norm"].get(
                    "model_numbers",
                    frozenset(),
                )
            )

            incoming_model_numbers = set(
                incoming_norm.get(
                    "model_numbers",
                    frozenset(),
                )
            )

            model_numbers_to_append = sorted(
                incoming_model_numbers - existing_model_numbers
            )

            reasons.append(
                "The Apple configuration already exists in Master, "
                "but the incoming Model Number is new and should "
                "be appended to the existing Master row."
            )

            return {
                "classification": "update",
                "conflict_type": None,
                "reasons": reasons,
                "warnings": warnings,
                "match": best_candidate,
                "match_index": best_candidate["index"],
                "multi_match": multi_match,
                "unknown_fields": [],
                "different_fields": [],
                "model_number_flag": "append",
                "model_numbers_to_append": (
                    model_numbers_to_append
                ),
                "model_number_update_required": True,
            }

    # --------------------------------------------------------
    # 9. Exact full row -> Conflict / Duplicate.
    # --------------------------------------------------------
    if full_comparison["exact"]:
        reasons.append(
            "The complete Master-relevant row already exists."
        )

        return {
            "classification": "conflict",
            "conflict_type": "duplicate",
            "reasons": reasons,
            "warnings": warnings,
            "match": best_candidate,
            "match_index": best_candidate["index"],
            "multi_match": multi_match,
            "unknown_fields": [],
            "different_fields": [],
            "model_numbers_to_append": [],
            "model_number_update_required": False,
            "model_number_flag": "match",
        }

    # --------------------------------------------------------
    # 10. Configuration and Provider both match the reference
    #     candidate, but at least one other Master-relevant
    #     value differs.
    #
    #     Rule: a different valid Provider is always NEW,
    #     regardless of any pricing difference (checked first,
    #     independent of price).
    #
    #     Model Number differences are preserved exactly as
    #     before this change: a Model Number that could not be
    #     cleanly appended (i.e. it differs alongside something
    #     else) is NEW, not part of the Retail Price / Trade-In
    #     handling below.
    #
    #     For the SAME Provider with the SAME Model Number, the
    #     remaining Master-relevant fields are mutable pricing
    #     data:
    #
    #       Retail Price changed (with or without a Trade-In
    #       change too) -> CONFLICT / needs_review. Retail Price
    #       always takes precedence over Trade-In.
    #
    #       Only other mutable value(s) changed (Trade-In Value
    #       and/or similar), Retail Price unchanged -> UPDATE.
    # --------------------------------------------------------
    different_fields = full_comparison[
        "different_fields"
    ]

    if "Provider" in different_fields:
        reasons.append(
            "The Apple configuration exists, but the "
            "Provider is different."
        )

        return {
            "classification": "new",
            "conflict_type": None,
            "reasons": reasons,
            "warnings": warnings,
            "match": best_candidate,
            "match_index": best_candidate["index"],
            "multi_match": multi_match,
            "unknown_fields": [],
            "different_fields": different_fields,
            "model_numbers_to_append": [],
            "model_number_update_required": False,
            "model_number_flag": compare_model_numbers(
                incoming_norm,
                best_candidate["norm"],
            ),
        }

    if "Model Number" in different_fields:
        reasons.append(
            "The Apple configuration exists, but the "
            "incoming Master-relevant row is different."
        )

        return {
            "classification": "new",
            "conflict_type": None,
            "reasons": reasons,
            "warnings": warnings,
            "match": best_candidate,
            "match_index": best_candidate["index"],
            "multi_match": multi_match,
            "unknown_fields": [],
            "different_fields": different_fields,
            "model_numbers_to_append": [],
            "model_number_update_required": False,
            "model_number_flag": compare_model_numbers(
                incoming_norm,
                best_candidate["norm"],
            ),
        }

    if "Retail Price" in different_fields:
        # Retail Price always requires Admin review, whether or
        # not Trade-In Value (or anything else) also changed.
        reasons.append(
            "Retail Price differs from the existing Master row "
            "for this Provider and configuration; this requires "
            "Admin review before Master is updated."
        )

        return {
            "classification": "conflict",
            "conflict_type": "needs_review",
            "reasons": reasons,
            "warnings": warnings,
            "match": best_candidate,
            "match_index": best_candidate["index"],
            "multi_match": multi_match,
            "unknown_fields": [],
            "different_fields": different_fields,
            "model_numbers_to_append": [],
            "model_number_update_required": False,
            "model_number_flag": compare_model_numbers(
                incoming_norm,
                best_candidate["norm"],
            ),
        }

    # Only mutable value(s) other than Retail Price changed (e.g.
    # Trade-In Value) -> the existing Master row should be
    # updated rather than treated as a new row or an Admin
    # conflict.
    reasons.append(
        "The Apple configuration, Provider, and Retail Price "
        "match the existing Master row; only mutable value(s) "
        "such as Trade-In Value differ, so the Master row "
        "should be updated."
    )

    return {
        "classification": "update",
        "conflict_type": None,
        "reasons": reasons,
        "warnings": warnings,
        "match": best_candidate,
        "match_index": best_candidate["index"],
        "multi_match": multi_match,
        "unknown_fields": [],
        "different_fields": different_fields,
        "model_numbers_to_append": [],
        "model_number_update_required": False,
        "model_number_flag": compare_model_numbers(
            incoming_norm,
            best_candidate["norm"],
        ),
    }


# ============================================================
# WITHIN-FILE DUPLICATE DETECTION
# ============================================================

def _build_strict_identity(row_norm):
    """
    Build an identity key for duplicate rows within the uploaded
    file.

    Only fields applicable to the row's device participate.

    Chipset is intentionally excluded.
    """
    identity = []

    device = row_norm.get("device")

    for field in DUPLICATE_FIELDS:
        if not is_field_applicable(device, field):
            continue

        if field == "Model Number":
            value = row_norm.get(
                "model_numbers",
                frozenset(),
            )

            identity.append(
                tuple(sorted(value))
            )
            continue

        value = _field_value(
            row_norm,
            field,
        )

        if _is_unknown(value):
            identity.append(UNKNOWN)
        elif isinstance(value, float):
            identity.append(round(value, 2))
        else:
            identity.append(value)

    return tuple(identity)


def detect_duplicates(canonical_df, mapped_fields=None):
    """
    Detect exact duplicates inside the incoming file.

    The first occurrence is processed normally.

    Any subsequent identical occurrence is classified as
    "duplicate" and must not be added to Master.
    """

    mapped_fields = set(
        mapped_fields or CANONICAL_FIELDS
    )

    groups = {}

    normalized_by_index = {}

    for index, row in canonical_df.iterrows():
        normalized = normalize_row_for_matching(
            row,
            mapped_fields,
        )

        normalized_by_index[index] = normalized

        key = _build_strict_identity(
            normalized
        )

        groups.setdefault(key, []).append(index)

    result = {}

    for index in canonical_df.index:
        normalized = normalized_by_index[index]
        key = _build_strict_identity(normalized)

        group = groups.get(key, [])

        result[index] = {
            "duplicate_group": group,
            "is_first_occurrence": (
                bool(group)
                and index == group[0]
            ),
            "within_file_duplicate": (
                len(group) > 1
            ),
        }

    return result


# ============================================================
# MASTER INDEX
# ============================================================

def build_master_indexes(master_df):
    """
    Precompute normalized Master rows for matching.

    Provider is indexed for reporting/debugging but is NOT used
    as a hard candidate filter.
    """
    master_norm_by_index = {}
    master_device_by_index = {}
    master_provider_by_index = {}

    if master_df is None or master_df.empty:
        return (
            master_norm_by_index,
            master_device_by_index,
            master_provider_by_index,
        )

    mapped_fields = set(CANONICAL_FIELDS)

    for index, row in master_df.iterrows():
        normalized = normalize_row_for_matching(
            row,
            mapped_fields,
        )

        master_norm_by_index[index] = normalized
        master_device_by_index[index] = normalized.get(
            "device"
        )
        master_provider_by_index[index] = normalized.get(
            "provider"
        )

    return (
        master_norm_by_index,
        master_device_by_index,
        master_provider_by_index,
    )


# ============================================================
# EXTRA / UNRECOGNIZED COLUMN INFORMATION
# ============================================================

def build_extra_column_info(raw_df, unmapped_columns):
    """
    Prepare information for the separate Admin modal.

    The modal can let Admin inspect the actual contents and decide
    whether to map or remove the column.

    This function does NOT automatically modify or delete anything.
    """
    result = []

    for column in unmapped_columns:
        series = raw_df[column]

        sample_values = []

        for value in series:
            if _is_blank(value):
                continue

            value_text = str(value)

            if value_text not in sample_values:
                sample_values.append(value_text)

            if len(sample_values) >= 5:
                break

        result.append({
            "column": column,
            "sample_values": sample_values,
            "non_empty_count": int(
                series.notna().sum()
            ),
        })

    return result


# ============================================================
# SUMMARY
# ============================================================

def build_summary(classifications):
    """
    Build Admin-facing counts.

    `classifications` is a list of (classification, conflict_type)
    tuples. classification is one of {"new", "update", "conflict"}.
    conflict_type is one of {"duplicate", "needs_review",
    "invalid_data"} when classification == "conflict", else None.

    Top-level counts:
        new
        update
        conflict   <- umbrella total

    conflict is broken down further into its sub-type counts
    (duplicate / needs_review / invalid_data) for convenience, but
    those sub-counts are not separate Admin-facing top-level
    categories on their own.

    multi_match is intentionally absent -- it is internal metadata
    only and must never be Admin-facing.
    """
    summary = {
        "new": 0,
        "update": 0,
        "conflict": 0,
        "duplicate": 0,
        "needs_review": 0,
        "invalid_data": 0,
    }

    for classification, conflict_type in classifications:
        if classification in ("new", "update"):
            summary[classification] += 1
        elif classification == "conflict":
            summary["conflict"] += 1
            if conflict_type in (
                "duplicate",
                "needs_review",
                "invalid_data",
            ):
                summary[conflict_type] += 1

    return summary

def _serialize_preview_row(row):
    """
    Convert a pandas row/dict-like object into a JSON-safe
    dictionary for the Admin bulk-import preview.
    """
    if row is None:
        return None

    if hasattr(row, "to_dict"):
        row = row.to_dict()

    result = {}

    for key, value in row.items():
        if pd.isna(value):
            result[key] = None
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value

    return result


# ============================================================
# MAIN ORCHESTRATION
# ============================================================

def analyze_upload(
    raw_df,
    master_df,
    clean_fn,
    column_mapping_override=None,
):
    """
    Main bulk-import analysis entry point.

    Flow:

        Raw upload
            |
            v
        Column mapping
            |
            v
        Canonical schema
            |
            v
        Existing cleaner
            |
            v
        Device/Sub-device inference
            |
            v
        Validation
            |
            v
        Apple configuration matching
            |
            v
        Business classification
            |
            +--> new
            +--> update
            +--> conflict
                    +--> duplicate
                    +--> needs_review
                    +--> invalid_data

    Extra/unrecognized columns are returned separately for the
    Admin modal.

    `multi_match` is internal metadata only and is never exposed
    as an Admin classification.
    """
    if raw_df is None:
        raise ValueError("raw_df cannot be None.")

    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError(
            "raw_df must be a pandas DataFrame."
        )

    if master_df is None:
        master_df = pd.DataFrame(
            columns=CANONICAL_FIELDS
        )

    # --------------------------------------------------------
    # 1. Column mapping
    # --------------------------------------------------------
    proposed_mapping = propose_column_mapping(
        list(raw_df.columns)
    )

    if column_mapping_override is not None:
        mapping = dict(column_mapping_override)

        mapped_raw_columns = set(mapping.keys())

        unmapped_columns = [
            column
            for column in raw_df.columns
            if column not in mapped_raw_columns
        ]

        missing_required = [
            field
            for field in (
                "Device",
                "Standardized Model",
            )
            if field not in mapping.values()
        ]
    else:
        mapping = proposed_mapping["mapping"]
        unmapped_columns = proposed_mapping[
            "unmapped_columns"
        ]
        missing_required = proposed_mapping[
            "missing_required"
        ]

    # --------------------------------------------------------
    # 2. Canonicalize
    # --------------------------------------------------------
    canonical_df = apply_column_mapping(
        raw_df,
        mapping,
    )

    # --------------------------------------------------------
    # 3. Normalize storage text before the project's cleaner.
    # --------------------------------------------------------
    if "Storage (GB)" in canonical_df.columns:
        canonical_df["Storage (GB)"] = (
            canonical_df["Storage (GB)"]
            .apply(_pretokenize_storage_text)
        )

    # --------------------------------------------------------
    # 4. Run existing project cleaner.
    #
    # clean_fn is intentionally retained as an injected
    # dependency so this rewrite does not need to duplicate
    # unrelated cleaning logic from app.py.
    # --------------------------------------------------------
    if clean_fn is not None:
        cleaned_result = clean_fn(
            canonical_df.copy()
        )

        if cleaned_result is not None:
            canonical_df = cleaned_result

    # Guarantee canonical columns still exist.
    for field in CANONICAL_FIELDS:
        if field not in canonical_df.columns:
            canonical_df[field] = np.nan

    canonical_df = canonical_df[
        CANONICAL_FIELDS
    ]

    # --------------------------------------------------------
    # 5. Infer missing Device/Sub-device from the whole row.
    #
    # Only infer when the explicit field is missing.
    # Never silently overwrite an explicit value.
    #
    # Device and Sub-device are inferred independently of each
    # other: a row that already has an explicit Device but no
    # Sub-device must still get a chance at Sub-device inference.
    # (Root cause of the reported bug: this used to be gated on
    # `Device` being blank only, so any row that already supplied
    # Device -- the overwhelming majority of real uploads -- never
    # even attempted Sub-device inference, regardless of whether
    # Sub-device itself was blank.)
    # --------------------------------------------------------
    vocabulary = build_device_vocabulary(
        master_df
    )

    # Fields that started out absent from the raw upload (no
    # source column mapped to them) but were subsequently
    # populated programmatically, row by row, below. These must
    # be treated as available for normalization/matching/
    # duplicate-detection from this point on -- otherwise a
    # correctly-inferred value sits in canonical_df but every
    # downstream consumer still treats the field as unmapped and
    # discards it back to Unknown. This is intentionally general
    # (not hardcoded to "Sub-device"): any canonical field this
    # loop assigns gets recorded here.
    inferred_fields = set()

    canonical_df["Device"] = canonical_df["Device"].astype(object)
    canonical_df["Sub-device"] = canonical_df["Sub-device"].astype(object)

    for index in canonical_df.index:
        row = canonical_df.loc[index]

        device_is_blank = _is_blank(row.get("Device"))
        sub_device_is_blank = _is_blank(
            row.get("Sub-device")
        )

        if device_is_blank or sub_device_is_blank:
            row_text = build_row_search_text(row)

            inferred_device, inferred_sub_device = (
                infer_device_subdevice(
                    row_text,
                    vocabulary,
                )
            )

            if device_is_blank and inferred_device:
                canonical_df.at[
                    index,
                    "Device",
                ] = inferred_device
                inferred_fields.add("Device")

            if (
                sub_device_is_blank
                and inferred_sub_device
            ):
                canonical_df.at[
                    index,
                    "Sub-device",
                ] = inferred_sub_device
                inferred_fields.add("Sub-device")

    # --------------------------------------------------------
    # 5a. Infer any still-missing Sub-device from the authoritative
    # Model Number -> Sub-device relationship in Master.
    #
    # This only ever runs for rows the generic keyword-based
    # inference above left blank (e.g. "iPhone 17" with no
    # qualifier word to match), and it never overwrites an
    # explicit Sub-device. See build_model_number_subdevice_map /
    # infer_subdevice_from_model_numbers for the trust rules that
    # keep Master's fictional/test rows from being used as a
    # source.
    # --------------------------------------------------------
    model_number_maps_by_device = {}

    for index in canonical_df.index:
        if not _is_blank(canonical_df.at[index, "Sub-device"]):
            continue

        device_value = canonical_df.at[index, "Device"]
        if _is_blank(device_value):
            continue

        if device_value not in model_number_maps_by_device:
            model_number_maps_by_device[device_value] = (
                build_model_number_subdevice_map(
                    master_df,
                    device_value,
                )
            )

        model_number_map = model_number_maps_by_device[
            device_value
        ]
        if not model_number_map:
            continue

        incoming_tokens = normalize_model_numbers(
            canonical_df.at[index, "Model Number"]
        )

        inferred_sub_device = infer_subdevice_from_model_numbers(
            incoming_tokens,
            model_number_map,
        )

        if inferred_sub_device:
            canonical_df.at[
                index, "Sub-device"
            ] = inferred_sub_device
            inferred_fields.add("Sub-device")

    # --------------------------------------------------------
    # 5b. Infer Model_Year from the model text when missing.
    #
    # Some provider files do not have a dedicated Year column,
    # but the year is explicitly included in the model name.
    # Example:
    # "MacBook Air i5 1.6GHz 13 inch (Mid 2019)"
    # -> Model_Year = 2019
    #
    # Never overwrite an explicitly supplied Model_Year.
    # --------------------------------------------------------
    canonical_df["Model_Year"] = canonical_df["Model_Year"].astype(object)

    for index in canonical_df.index:
        if _is_blank(canonical_df.at[index, "Model_Year"]):
            inferred_year = infer_model_year_from_text(
                canonical_df.at[index, "Standardized Model"]
            )

            if inferred_year is not None:
                canonical_df.at[index, "Model_Year"] = inferred_year
                inferred_fields.add("Model_Year")

    # --------------------------------------------------------
    # 5b. Canonicalize the incoming row's OWN display values
    # (Device, Sub-device, Standardized Model, Connectivity)
    # using Master's naming/casing conventions as the source of
    # truth.
    #
    # This runs after inference (so an inferred Sub-device such
    # as the lowercase "ipad mini" produced by infer_device_
    # subdevice's comparison-safe vocabulary also gets re-cased),
    # but it only ever rewrites the four fields above, and only
    # to the extent Master's own data can confirm a canonical
    # spelling for this Device. It never touches Provider,
    # Storage, Retail Price, Trade-In value, or Model Number, and
    # it never copies values from a specific matched Master row --
    # matching itself still happens later, against whichever
    # Master row this canonicalized incoming row turns out to
    # correspond to.
    #
    # Same "trusted field" principle as step 5 above applies here:
    # a field this step newly populates (most notably Connectivity,
    # which the raw upload frequently never supplies a column for
    # at all -- it only ever lived embedded inside the model text)
    # must be recorded in inferred_fields, or every downstream
    # consumer that reads through _get_mapped_value() /
    # mapped_canonical_fields would keep treating that field as
    # unmapped and discard the now-correct value back to Unknown.
    # Concretely: once this step strips "Wi-Fi + Cellular" out of
    # Standardized Model and moves it into Connectivity, the old
    # fallback of re-extracting connectivity from the model text
    # at classification time no longer has anything left to find --
    # so Connectivity must itself be trusted from this point on.
    # --------------------------------------------------------
    display_maps = build_canonical_display_maps(master_df)

    canonical_df["Standardized Model"] = canonical_df["Standardized Model"].astype(object)
    canonical_df["Connectivity"] = canonical_df["Connectivity"].astype(object)

    for index in canonical_df.index:
        row = canonical_df.loc[index]

        updates = canonicalize_incoming_row_display_values(
            row, display_maps
        )

        for field, value in updates.items():
            canonical_df.at[index, field] = value
            inferred_fields.add(field)

    # --------------------------------------------------------
    # 6. Build Master indexes once.
    # --------------------------------------------------------
    (
        master_norm_by_index,
        master_device_by_index,
        master_provider_by_index,
    ) = build_master_indexes(
        master_df
    )

    # --------------------------------------------------------
    # 7. Detect within-file duplicates.
    #
    # Subsequent identical rows are classified as duplicates
    # and prevented from being added to Master.
    #
    # mapped_canonical_fields drives _get_mapped_value() inside
    # normalize_row_for_matching(): it decides whether a
    # canonical field is trusted at all, independent of whether
    # any particular row's value in it happens to be blank. A
    # field the raw upload never supplied but that step 5 above
    # populated via inference is just as legitimate a source as
    # an explicitly mapped raw column, so it must be included
    # here too -- otherwise normalization, duplicate detection,
    # and classification would all silently discard the inferred
    # value and treat the field as still unmapped.
    # --------------------------------------------------------
    mapped_canonical_fields = (
        set(mapping.values()) | inferred_fields
    )

    within_file_duplicates = detect_duplicates(
        canonical_df,
        mapped_canonical_fields,
    )

    # --------------------------------------------------------
    # 8. Extra/unrecognized column information.
    # --------------------------------------------------------
    extra_columns = build_extra_column_info(
        raw_df,
        unmapped_columns,
    )

    # --------------------------------------------------------
    # 9. Classify every row.
    # --------------------------------------------------------
    row_results = []
    classifications = []

    for index, row in canonical_df.iterrows():
        incoming_norm = normalize_row_for_matching(
            row,
            mapped_canonical_fields,
        )

        within_file_info = within_file_duplicates.get(
            index,
            {},
        )

        is_within_file_duplicate = (
            within_file_info.get(
                "within_file_duplicate",
                False,
            )
            and not within_file_info.get(
                "is_first_occurrence",
                False,
            )
        )

        if is_within_file_duplicate:
            classification_result = {
                "classification": "conflict",
                "conflict_type": "duplicate",
                "reasons": [
                    "This row is an identical duplicate of an "
                    "earlier row in the uploaded file and will "
                    "not be added to Master."
                ],
                "warnings": [],
                "match": None,
                "match_index": None,
                "multi_match": False,
                "unknown_fields": [],
                "different_fields": [],
                "model_number_flag": None,
                "model_numbers_to_append": [],
                "model_number_update_required": False,
            }
        else:
            classification_result = classify_incoming_row(
                incoming_row_canonical=row,
                incoming_norm=incoming_norm,
                master_df=master_df,
                master_norm_by_index=master_norm_by_index,
                master_device_by_index=master_device_by_index,
                master_provider_by_index=master_provider_by_index,
                mapped_canonical_fields=mapped_canonical_fields,
            )

            # The classifier may enrich the canonical incoming row
            # with inherited values, such as Retail Price from the
            # selected Master counterpart. Persist those mutations
            # back into canonical_df so previews and downstream
            # consumers see the same effective incoming data used
            # during classification.
            canonical_df.loc[
                index,
                "Retail Price",
            ] = row.get("Retail Price")

        classification = classification_result[
            "classification"
        ]

        conflict_type = classification_result.get(
            "conflict_type"
        )

        classifications.append(
            (classification, conflict_type)
        )

        row_result = {
            "row_index": index,
            "classification": classification,
            "conflict_type": conflict_type,
            "reasons": classification_result["reasons"],
            "warnings": classification_result["warnings"],
            "match_index": classification_result["match_index"],
            "multi_match": classification_result["multi_match"],
            "unknown_fields": classification_result["unknown_fields"],
            "different_fields": classification_result["different_fields"],
            "model_number_flag": classification_result[
                "model_number_flag"
            ],
            "model_numbers_to_append": classification_result.get(
                "model_numbers_to_append", []
            ),
            "model_number_update_required": classification_result.get(
                "model_number_update_required", False
            ),
            "within_file_duplicate": (
                within_file_info.get(
                    "within_file_duplicate",
                    False,
                )
            ),
            "within_file_duplicate_group": (
                within_file_info.get(
                    "duplicate_group",
                    [],
                )
            ),

            # Human-readable data for the Admin preview.
            # These are added to the preview response only;
            # they do not affect classification.
            "incoming_row": _serialize_preview_row(row),

            # Filled below when a Master counterpart exists.
            "matched_master": None,
        }

        selected_match = classification_result.get(
            "match"
        )

        if selected_match is not None:
            matched_index = selected_match.get("index")

            row_result["match_confidence"] = (
                selected_match.get("confidence")
            )

            row_result["matched_master_row"] = (
                matched_index
            )

            # Include the actual Master record for the Admin
            # detail modal. This is display data only.
            if matched_index is not None and matched_index in master_df.index:
                master_row = master_df.loc[matched_index]

                row_result["matched_master"] = (
                    _serialize_preview_row(master_row)
                )
        else:
            row_result["match_confidence"] = None
            row_result["matched_master_row"] = None


        row_results.append(row_result)

    # --------------------------------------------------------
    # 10. Summary
    # --------------------------------------------------------
    summary = build_summary(
        classifications
    )

    # --------------------------------------------------------
    # 11. Return a stable result structure.
    # --------------------------------------------------------
    return {
        "mapping": mapping,
        "unmapped_columns": unmapped_columns,
        "extra_columns": extra_columns,
        "missing_required": missing_required,

        "canonical_df": canonical_df,

        "rows": row_results,

        "summary": summary,

        # Internal indexes are returned because existing callers
        # may need them for preview/detail operations.
        "master_norm_by_index": master_norm_by_index,
        "master_device_by_index": master_device_by_index,
        "master_provider_by_index": master_provider_by_index,

        # Internal-only upload duplicate information.
        "within_file_duplicates": within_file_duplicates,
    }