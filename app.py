from pathlib import Path
import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

CURRENT_YEAR = 2026

BASE_DIR = Path(__file__).parent

ICON_PATH = BASE_DIR / "imycom.png"
MODEL_PATH = BASE_DIR / "model_v2.pkl"
CSV_PATH = BASE_DIR / "master_apple_final_ml.csv"


st.set_page_config(
    page_title="Apple Trade-In Competitive Pricing",
    page_icon=str(ICON_PATH),
    layout="centered"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --red: #E30613;
        --dark-red: #C4000B;
        --white: #FFFFFF;
        --dark: #1F1F1F;
        --grey: #666666;
        --border: #E2E2E2;
    }

    html,
    body,
    [class*="css"],
    .stApp {
        font-family: "Inter", Arial, sans-serif !important;
    }

    .stApp {
        background-color: var(--red);
        color: var(--dark);
    }

    .main {
        background-color: var(--red);
    }

    .block-container {
        background-color: var(--white);
        border-radius: 14px;
        max-width: 1100px;
        padding: 1.5rem 2.2rem 3rem 2.2rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.16);
    }

    .imalaysian-title {
        color: var(--dark) !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
        line-height: 1.2;
        margin: 1rem 0 0.5rem 0 !important;
    }

    .imalaysian-subtitle {
        color: var(--dark) !important;
        font-size: 1.1rem !important;
        font-weight: 400 !important;
        margin-top: 0.3rem !important;
        opacity: 0.92;
    }

    h2,
    h3 {
        color: var(--dark) !important;
        font-weight: 700 !important;
    }

    h2::after,
    h3::after {
        content: "";
        display: block;
        width: 45px;
        height: 3px;
        background-color: var(--red);
        margin-top: 7px;
        border-radius: 1px;
    }

    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.4rem 0 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 6px !important;
        min-height: 42px;
    }

    div[data-baseweb="select"] span {
        color: var(--text-color) !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: var(--red) !important;
    }

    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--red) !important;
        box-shadow: 0 0 0 1px var(--red) !important;
    }

    .stButton > button {
        background-color: var(--red) !important;
        color: var(--white) !important;
        border: 1px solid var(--red) !important;
        border-radius: 6px !important;
        min-height: 46px;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        background-color: var(--dark-red) !important;
        border-color: var(--dark-red) !important;
    }

    .recommendation-card {
        background-color: var(--red);
        border-radius: 10px;
        padding: 1.6rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 5px 14px rgba(0,0,0,0.15);
    }

    .recommendation-title {
        color: var(--white);
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .recommendation-value {
        color: var(--white);
        font-size: 2.8rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 7px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(CSV_PATH)

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # TEXT COLUMNS
    # --------------------------------------------------------

    text_columns = [
        "Pricing Category",
        "Provider",
        "Device",
        "Model",
        "Specification",
        "Standardized Model",
        "Chipset",
        "Storage Type",
        "Connectivity",
        "Material",
        "Model Number",
        "Clock Speed",
        "Sub-device",
        "Mac Generation"
    ]

    for col in text_columns:

        if col in data.columns:

            data[col] = (
                data[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    # --------------------------------------------------------
    # NUMERIC COLUMNS
    # --------------------------------------------------------

    numeric_columns = [
        "Model_Year",
        "Generation",
        "Year",
        "Screen Size (inch)",
        "RAM (GB)",
        "RAM Min (GB)",
        "Max. Trade-In Value (RM)"
    ]

    for col in numeric_columns:

        if col in data.columns:

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    if "Storage (GB)" in data.columns:

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

            if match.group(2) == "TB":
                number *= 1024

            return number

        data["Storage (GB)"] = (
            data["Storage (GB)"]
            .apply(parse_storage)
        )

    return data


model = load_model()
df = load_data()


# ============================================================
# HELPERS
# ============================================================

def clean_options(series):

    if series is None:
        return []

    values = (
        series
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    return sorted(
        values,
        key=lambda x: str(x)
    )


def numeric_options(series):

    values = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
        .unique()
        .tolist()
    )

    return sorted(values)


def format_storage(value):

    if pd.isna(value):
        return "Not available"

    value = float(value)

    if value >= 1024:

        tb = value / 1024

        if tb.is_integer():
            return f"{int(tb)} TB"

        return f"{tb:g} TB"

    return f"{int(value)} GB"


def extract_clock_speed(value):

    if pd.isna(value):
        return np.nan

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*GHz",
        str(value),
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return np.nan


def select_numeric(
    label,
    column,
    source_df,
    key,
    formatter=None
):

    options = numeric_options(
        source_df[column]
        if column in source_df.columns
        else pd.Series(dtype=float)
    )

    if not options:
        return np.nan

    display_options = [
        formatter(x)
        if formatter
        else str(x)
        for x in options
    ]

    selected = st.selectbox(
        label,
        display_options,
        key=key
    )

    return options[
        display_options.index(selected)
    ]


def select_text(
    label,
    column,
    source_df,
    key,
    include_not_specified=False
):

    options = clean_options(
        source_df[column]
        if column in source_df.columns
        else pd.Series(dtype=str)
    )

    if not options:
        return ""

    if include_not_specified:
        options = ["Not specified"] + options

    selected = st.selectbox(
        label,
        options,
        key=key
    )

    if selected == "Not specified":
        return ""

    return selected


# ============================================================
# HEADER
# ============================================================

col1, col2 = st.columns([1, 3])

with col1:

    if ICON_PATH.exists():
        st.image(
            str(ICON_PATH),
            width=220
        )

with col2:

    st.markdown(
        """
        <div class="imalaysian-title">
            Apple Trade-In Competitive Pricing
        </div>

        <div class="imalaysian-subtitle">
            Market-based trade-in valuation and competitive pricing
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DEVICE SELECTION
# ============================================================

st.subheader("Device Selection")


device_types = clean_options(
    df["Device"]
)

if not device_types:

    st.error(
        "No device types are available."
    )

    st.stop()


device_type = st.selectbox(
    "Device Type",
    device_types,
    key="device_type_selection"
)


device_df = df[
    df["Device"].astype(str).str.strip()
    == device_type
].copy()


if device_df.empty:

    st.warning(
        "No records are available for this device type."
    )

    st.stop()


# ============================================================
# SUB-DEVICE
# ============================================================

sub_device = ""


if (
    device_type != "iPhone"
    and "Sub-device" in device_df.columns
):

    sub_device_options = clean_options(
        device_df["Sub-device"]
    )

    if sub_device_options:

        sub_device = st.selectbox(
            "Sub-device",
            sub_device_options,
            key="sub_device_selection"
        )

        device_df = device_df[
            device_df["Sub-device"].astype(str).str.strip()
            == sub_device
        ].copy()


# ============================================================
# MODEL
# ============================================================

if "Standardized Model" not in device_df.columns:

    st.error(
        "Standardized Model column is missing."
    )

    st.stop()


models = clean_options(
    device_df["Standardized Model"]
)


if not models:

    st.warning(
        "No models are available."
    )

    st.stop()


model_name = st.selectbox(
    "Model",
    models,
    key="model_selection"
)


model_df = device_df[
    device_df["Standardized Model"].astype(str).str.strip()
    == str(model_name).strip()
].copy()


if model_df.empty:

    st.error(
        "No records found for this model."
    )

    st.stop()


# ============================================================
# INITIAL VALUES
# ============================================================

storage = np.nan
storage_type = ""
connectivity = ""
charging_method = ""
material = ""
case_size = ""


# ============================================================
# IPHONE
# ============================================================

if device_type == "iPhone":

    storage = select_numeric(
        "Storage",
        "Storage (GB)",
        model_df,
        "iphone_storage",
        format_storage
    )

    if pd.notna(storage):

        selected_df = model_df[
            model_df["Storage (GB)"] == storage
        ].copy()

    else:

        selected_df = model_df.copy()


# ============================================================
# IPAD
# ============================================================

elif device_type == "iPad":

    storage = select_numeric(
        "Storage",
        "Storage (GB)",
        model_df,
        "ipad_storage",
        format_storage
    )

    if pd.notna(storage):

        selected_df = model_df[
            model_df["Storage (GB)"] == storage
        ].copy()

    else:

        selected_df = model_df.copy()


    connectivity = select_text(
        "Connectivity",
        "Connectivity",
        selected_df,
        "ipad_connectivity",
        include_not_specified=True
    )

    if connectivity:

        selected_df = selected_df[
            selected_df["Connectivity"].astype(str).str.strip()
            == connectivity
        ].copy()


# ============================================================
# MAC
# ============================================================

elif device_type == "Mac":

    storage = select_numeric(
        "Storage",
        "Storage (GB)",
        model_df,
        "mac_storage",
        format_storage
    )

    if pd.notna(storage):

        selected_df = model_df[
            model_df["Storage (GB)"] == storage
        ].copy()

    else:

        selected_df = model_df.copy()


    storage_type = select_text(
        "Storage Type",
        "Storage Type",
        selected_df,
        "mac_storage_type",
        include_not_specified=True
    )

    if storage_type:

        selected_df = selected_df[
            selected_df["Storage Type"].astype(str).str.strip()
            == storage_type
        ].copy()


# ============================================================
# APPLE WATCH
# ============================================================

elif device_type == "Apple Watch":

    selected_df = model_df.copy()


    case_size = select_text(
        "Case Size",
        "Specification",
        selected_df,
        "watch_case_size"
    )

    if case_size:

        selected_df = selected_df[
            selected_df["Specification"].astype(str).str.strip()
            == case_size
        ].copy()


    material = select_text(
        "Material",
        "Material",
        selected_df,
        "watch_material",
        include_not_specified=True
    )

    if material:

        selected_df = selected_df[
            selected_df["Material"].astype(str).str.strip()
            == material
        ].copy()


    storage = select_numeric(
        "Storage",
        "Storage (GB)",
        selected_df,
        "watch_storage",
        format_storage
    )

    if pd.notna(storage):

        selected_df = selected_df[
            selected_df["Storage (GB)"] == storage
        ].copy()


    connectivity = select_text(
        "Connectivity",
        "Connectivity",
        selected_df,
        "watch_connectivity",
        include_not_specified=True
    )

    if connectivity:

        selected_df = selected_df[
            selected_df["Connectivity"].astype(str).str.strip()
            == connectivity
        ].copy()


# ============================================================
# AIRPODS
# ============================================================

elif device_type == "AirPods":

    selected_df = model_df.copy()

    charging_method = select_text(
        "Charging Method",
        "Specification",
        selected_df,
        "airpods_charging_method",
        include_not_specified=True
    )

    if charging_method:

        selected_df = selected_df[
            selected_df["Specification"].astype(str).str.strip()
            == charging_method
        ].copy()


# ============================================================
# FALLBACK
# ============================================================

else:

    selected_df = model_df.copy()


# ============================================================
# FALLBACK IF CONFIGURATION FILTER IS EMPTY
# ============================================================

if selected_df.empty:

    st.warning(
        "No exact market configuration was found. "
        "Using the model-level record for the Smart Estimate."
    )

    selected_df = model_df.copy()


if selected_df.empty:

    st.error(
        "No usable record was found."
    )

    st.stop()


# ============================================================
# SELECT RECORD FOR ML FEATURES
# ============================================================

selected = selected_df.iloc[0]


# ============================================================
# EXTRACT FEATURES
# ============================================================

standardized_model = str(
    selected.get(
        "Standardized Model",
        model_name
    )
).strip()


provider = str(
    selected.get(
        "Provider",
        ""
    )
).strip()


pricing_category = str(
    selected.get(
        "Pricing Category",
        ""
    )
).strip()


model_year = pd.to_numeric(
    selected.get(
        "Model_Year",
        np.nan
    ),
    errors="coerce"
)


generation = pd.to_numeric(
    selected.get(
        "Generation",
        np.nan
    ),
    errors="coerce"
)


screen_size = pd.to_numeric(
    selected.get(
        "Screen Size (inch)",
        np.nan
    ),
    errors="coerce"
)


ram = pd.to_numeric(
    selected.get(
        "RAM (GB)",
        np.nan
    ),
    errors="coerce"
)


chipset = selected.get(
    "Chipset",
    ""
)

if pd.isna(chipset):
    chipset = ""

chipset = str(chipset).strip()


storage_type_selected = (
    storage_type
    if storage_type
    else str(
        selected.get(
            "Storage Type",
            ""
        )
    ).strip()
)


connectivity_selected = (
    connectivity
    if connectivity
    else str(
        selected.get(
            "Connectivity",
            ""
        )
    ).strip()
)


clock_speed_ghz = extract_clock_speed(
    selected.get(
        "Clock Speed",
        ""
    )
)


# ============================================================
# DEVICE AGE
# ============================================================

if pd.notna(model_year):

    device_age = max(
        0,
        CURRENT_YEAR - float(model_year)
    )

else:

    device_age = np.nan


# ============================================================
# BUILD COMPLETE ML INPUT
# ============================================================

raw_features = {

    "device_type":
        str(device_type).strip(),

    "model":
        str(model_name).strip(),

    "standardized_model":
        standardized_model,

    "provider":
        provider,

    "pricing_category":
        pricing_category,

    "model_year":
        model_year,

    "device_age":
        device_age,

    "generation":
        generation,

    "screen_size":
        screen_size,

    "clock_speed_ghz":
        clock_speed_ghz,

    "chipset":
        chipset,

    "ram":
        ram,

    "storage":
        storage,

    "storage_type":
        storage_type_selected,

    "connectivity":
        connectivity_selected
}


# ============================================================
# DETERMINE EXACT MODEL SCHEMA
# ============================================================

try:

    expected_features = list(
        model.named_steps[
            "preprocessor"
        ].feature_names_in_
    )

except Exception as e:

    st.error(
        "Could not determine the feature schema "
        "inside model_v2.pkl."
    )

    st.exception(e)

    st.stop()


# ============================================================
# BUILD MODEL INPUT USING ACTUAL PKL SCHEMA
# ============================================================

missing_from_backend = [
    feature
    for feature in expected_features
    if feature not in raw_features
]


if missing_from_backend:

    st.error(
        "The PKL expects features that the app does not provide:"
    )

    st.write(
        missing_from_backend
    )

    st.stop()


input_data = pd.DataFrame([
    {
        feature: raw_features[feature]
        for feature in expected_features
    }
])


# ============================================================
# ML PREDICTION
# ============================================================

try:

    ml_prediction = model.predict(
        input_data
    )[0]

    ml_prediction = max(
        0,
        float(ml_prediction)
    )

except Exception as e:

    st.error(
        "Unable to generate Smart Estimate."
    )

    st.exception(e)

    st.stop()


# ============================================================
# MARKET EVIDENCE
# ============================================================

market_records = selected_df[
    pd.to_numeric(
        selected_df["Max. Trade-In Value (RM)"],
        errors="coerce"
    ).notna()
].copy()


market_prices = pd.to_numeric(
    market_records[
        "Max. Trade-In Value (RM)"
    ],
    errors="coerce"
).dropna()


if not market_prices.empty:

    market_median = float(
        market_prices.median()
    )

else:

    market_median = np.nan


# ============================================================
# DEBUG
# ============================================================

with st.expander("Technical Details"):

    st.write(
        "Model file:",
        MODEL_PATH.name
    )

    st.write(
        "Dataset:",
        CSV_PATH.name
    )

    st.write(
        "Model expected features:",
        expected_features
    )

    st.write(
        "ML input:"
    )

    st.dataframe(
        input_data,
        use_container_width=True
    )

    st.write(
        "Smart Estimate:",
        f"RM {ml_prediction:,.2f}"
    )

    st.write(
        "Market Median:",
        (
            f"RM {market_median:,.2f}"
            if pd.notna(market_median)
            else "No market data"
        )
    )


# ============================================================
# ESTIMATE METHOD
# ============================================================

st.divider()

st.subheader("Estimate Method")


estimate_method = st.radio(
    "Choose how the estimate is calculated",
    [
        "Market Estimate",
        "Smart Estimate"
    ],
    horizontal=True,
    key="estimate_method"
)


# ============================================================
# SELECT FINAL VALUE
# ============================================================

if (
    estimate_method == "Market Estimate"
    and not market_prices.empty
):

    recommended_offer = market_median

    provider_count = (
        market_records["Provider"]
        .replace("", np.nan)
        .dropna()
        .nunique()
    )

    if provider_count >= 3:

        confidence_text = (
            "Based on observed market prices "
            "from multiple providers."
        )

    elif provider_count == 2:

        confidence_text = (
            "Based on observed market prices "
            "from two providers."
        )

    else:

        confidence_text = (
            "Based on observed market pricing."
        )

    estimate_method_display = "Market Estimate"

else:

    recommended_offer = ml_prediction

    confidence_text = (
        "Estimated using historical trade-in data "
        "and device specifications."
    )

    estimate_method_display = "Smart Estimate"


# ============================================================
# FINAL DISPLAY
# ============================================================

st.subheader(
    "Estimated Trade-In Value"
)


st.html(
    f"""
    <div class="recommendation-card">

        <div class="recommendation-title">
            {estimate_method_display}
        </div>

        <div class="recommendation-value">
            RM {recommended_offer:,.0f}
        </div>

    </div>
    """
)


st.caption(
    confidence_text
)


if estimate_method_display == "Smart Estimate":

    st.warning(
        "Estimate only. Actual trade-in values may differ "
        "based on market conditions and device configuration."
    )
