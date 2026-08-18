import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import re


# ============================================================
# CONFIGURATION
# ============================================================

icon_path = os.path.join(os.path.dirname(__file__), "imycom.png")

st.set_page_config(
    page_title="Apple Trade-In Competitive Pricing",
    page_icon=icon_path,
    layout="centered"
)

# ============================================================
# iMALAYSIAN RED THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --red: #E30613;
        --dark-red: #C4000B;
        --deep-red: #A80009;
        --white: #FFFFFF;
        --off-white: #F8F8F8;
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
        padding: 2rem 2.2rem 3rem 2.2rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.16);
    }

    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
        HEADER AREA
        ======================================================== */

        .imalaysian-header {
            background-color: var(--red);
            padding: 3rem 0 1.5rem 0;
            text-align: center;
        }

        .imalaysian-logo-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 1.8rem;
        }

        .imalaysian-title {
            color: var(--dark) !important;
            font-size: 2.5rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
            line-height: 4;
            margin: 0 !important;
        }

        .imalaysian-subtitle {
            color: var(--dark) !important;
            font-size: 1.25rem !important;
            font-weight: 400 !important;
            line-height: 0;
            margin-top: -2rem !important;
            opacity: 0.92;
        }


    /* ========================================================
       WHITE CONTENT AREA
       ======================================================== */

    .content-panel {
        background-color: var(--white);
        border-radius: 12px;
        padding: 1.8rem;
        margin-top: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    h1 {
        color: var(--white) !important;
        font-weight: 800 !important;
    }

    h2 {
        color: var(--dark) !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px;
    }

    h3 {
        color: var(--dark) !important;
        font-weight: 700 !important;
    }


    /* Red accent below section headings */

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


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.4rem 0 !important;
    }


    /* ========================================================
       SELECT BOXES
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: var(--white) !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 6px !important;
        min-height: 42px;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: var(--red) !important;
    }

    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--red) !important;
        box-shadow: 0 0 0 1px var(--red) !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        background-color: var(--red) !important;
        color: var(--white) !important;
        border: 1px solid var(--red) !important;
        border-radius: 6px !important;
        min-height: 46px;
        font-family: "Inter", Arial, sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    .stButton > button:hover {
        background-color: var(--dark-red) !important;
        border-color: var(--dark-red) !important;
        color: var(--white) !important;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background-color: var(--white);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 7px rgba(0,0,0,0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--grey) !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--dark) !important;
        font-weight: 800 !important;
    }


    /* ========================================================
       RECOMMENDATION
       ======================================================== */

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
        font-family: "Inter", Arial, sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-top: 0.3rem;
    }


    /* ========================================================
       INFO / WARNING / SUCCESS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 7px !important;
    }


    /* ========================================================
       TABLE
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 7px;
        overflow: hidden;
    }


    /* ========================================================
       TEXT
       ======================================================== */

    p,
    label,
    span {
        color: var(--dark);
    }

    .stCaption {
        color: var(--grey) !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .imalaysian-footer {
        text-align: center;
        color: rgba(255,255,255,0.85);
        font-size: 0.8rem;
        padding: 1rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)

CURRENT_YEAR = 2026

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
CSV_PATH = os.path.join(
    BASE_DIR,
    "structured_apple_devices_full.csv"
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

    data.columns = data.columns.str.strip()

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
        "Model Number",
        "Clock Speed"
    ]

    for col in text_columns:

        if col in data.columns:

            data[col] = (
                data[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    numeric_columns = [
        "Model_Year",
        "Generation",
        "Year",
        "Screen Size (inch)",
        "RAM (GB)",
        "Storage (GB)",
        "Max. Trade-In Value (RM)"
    ]

    for col in numeric_columns:

        if col in data.columns:

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

    return data


model = load_model()
df = load_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_options(series):
    """
    Return unique non-empty values from a dataframe column.
    """

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
    """
    Return unique numeric values from a dataframe column.
    """

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


def format_ram(value):

    if pd.isna(value):
        return "Not available"

    return f"{float(value):g} GB"


def format_screen(value):

    if pd.isna(value):
        return "Not available"

    return f"{float(value):g} inch"


def format_clock_speed(value):

    if pd.isna(value):
        return "Not available"

    return f"{float(value):g} GHz"


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


def show_selectable_text(
    label,
    column,
    source_df,
    current_value,
    key
):
    """
    Creates a dropdown using values that actually exist
    in the relevant dataset records.
    """

    options = clean_options(
        source_df[column]
        if column in source_df.columns
        else pd.Series(dtype=str)
    )

    current_value = (
        str(current_value).strip()
        if not pd.isna(current_value)
        else ""
    )

    if current_value and current_value not in options:
        options.insert(0, current_value)

    if not options:

        return st.text_input(
            label,
            value=current_value,
            key=key
        )

    return st.selectbox(
        label,
        options,
        index=(
            options.index(current_value)
            if current_value in options
            else 0
        ),
        key=key
    )


def show_selectable_numeric(
    label,
    column,
    source_df,
    current_value,
    key,
    formatter=None
):
    """
    Creates a dropdown for numeric specifications.
    """

    options = numeric_options(
        source_df[column]
        if column in source_df.columns
        else pd.Series(dtype=float)
    )

    if pd.notna(current_value):

        current_value = float(current_value)

        if current_value not in options:
            options.insert(0, current_value)

    if not options:

        return current_value

    formatted_options = [
        formatter(value)
        if formatter
        else str(value)
        for value in options
    ]

    current_index = (
        options.index(current_value)
        if pd.notna(current_value)
        and current_value in options
        else 0
    )

    selected_display = st.selectbox(
        label,
        formatted_options,
        index=current_index,
        key=key
    )

    return options[
        formatted_options.index(selected_display)
    ]

# ============================================================
# HEADER
# ============================================================

logo_path = os.path.join(BASE_DIR, "imycom.png")

col1, col2 = st.columns([1, 3])

with col1:
    st.write("")
    st.write("")

    if os.path.exists(logo_path):
        st.image(logo_path, width=220)

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


# ------------------------------------------------------------
# DEVICE TYPE
# ------------------------------------------------------------

device_types = clean_options(
    df["Device"]
)

device_type = st.selectbox(
    "Device Type",
    device_types
)


device_df = df[
    df["Device"] == device_type
].copy()


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

models = clean_options(
    device_df["Model"]
)

if not models:

    st.warning(
        "No models are available for this device type."
    )

    st.stop()


model_name = st.selectbox(
    "Model",
    models
)


model_df = device_df[
    device_df["Model"] == model_name
].copy()


# ------------------------------------------------------------
# SPECIFICATION
# ------------------------------------------------------------

specifications = clean_options(
    model_df["Specification"]
)

if specifications:

    specification = st.selectbox(
        "Specification",
        specifications
    )

    selected_df = model_df[
        model_df["Specification"] == specification
    ].copy()

else:

    specification = ""

    selected_df = model_df.copy()


if selected_df.empty:

    st.error(
        "No matching configuration was found."
    )

    st.stop()


selected = selected_df.iloc[0]


# ============================================================
# ORIGINAL DETECTED VALUES
# ============================================================

original_model_year = (
    selected["Model_Year"]
    if "Model_Year" in selected
    else np.nan
)

original_generation = (
    selected["Generation"]
    if "Generation" in selected
    else np.nan
)

original_screen_size = (
    selected["Screen Size (inch)"]
    if "Screen Size (inch)" in selected
    else np.nan
)

original_chipset = (
    selected["Chipset"]
    if "Chipset" in selected
    else ""
)

original_ram = (
    selected["RAM (GB)"]
    if "RAM (GB)" in selected
    else np.nan
)

original_storage = (
    selected["Storage (GB)"]
    if "Storage (GB)" in selected
    else np.nan
)

original_storage_type = (
    selected["Storage Type"]
    if "Storage Type" in selected
    else ""
)

original_connectivity = (
    selected["Connectivity"]
    if "Connectivity" in selected
    else ""
)

original_clock_speed_raw = (
    selected["Clock Speed"]
    if "Clock Speed" in selected
    else ""
)


# ============================================================
# DETECTED / ADJUSTABLE DEVICE DETAILS
# ============================================================

st.divider()

st.subheader("Device Details")

st.caption(
    "These values are detected from the selected dataset "
    "record. You can adjust individual specifications if "
    "the detected configuration is not accurate."
)


# ------------------------------------------------------------
# DEVICE-SPECIFIC DETAILS
# ------------------------------------------------------------

model_year = original_model_year
generation = original_generation
screen_size = original_screen_size
chipset = original_chipset
ram = original_ram
storage = original_storage
storage_type = original_storage_type
connectivity = original_connectivity
clock_speed_raw = original_clock_speed_raw


# ============================================================
# IPHONE
# ============================================================

if device_type == "iPhone":

    col1, col2 = st.columns(2)

    with col1:

        model_year = show_selectable_numeric(
            "Model Year",
            "Model_Year",
            model_df,
            original_model_year,
            "iphone_model_year"
        )

        chipset = show_selectable_text(
            "Chipset",
            "Chipset",
            model_df,
            original_chipset,
            "iphone_chipset"
        )

        storage = show_selectable_numeric(
            "Storage",
            "Storage (GB)",
            model_df,
            original_storage,
            "iphone_storage",
            format_storage
        )

    with col2:

        screen_size = show_selectable_numeric(
            "Screen Size",
            "Screen Size (inch)",
            model_df,
            original_screen_size,
            "iphone_screen",
            format_screen
        )

        connectivity = show_selectable_text(
            "Connectivity",
            "Connectivity",
            model_df,
            original_connectivity,
            "iphone_connectivity"
        )


# ============================================================
# IPAD
# ============================================================

elif device_type == "iPad":

    col1, col2 = st.columns(2)

    with col1:

        model_year = show_selectable_numeric(
            "Model Year",
            "Model_Year",
            model_df,
            original_model_year,
            "ipad_model_year"
        )

        chipset = show_selectable_text(
            "Chipset",
            "Chipset",
            model_df,
            original_chipset,
            "ipad_chipset"
        )

        storage = show_selectable_numeric(
            "Storage",
            "Storage (GB)",
            model_df,
            original_storage,
            "ipad_storage",
            format_storage
        )

        ram = show_selectable_numeric(
            "RAM",
            "RAM (GB)",
            model_df,
            original_ram,
            "ipad_ram",
            format_ram
        )

    with col2:

        screen_size = show_selectable_numeric(
            "Screen Size",
            "Screen Size (inch)",
            model_df,
            original_screen_size,
            "ipad_screen",
            format_screen
        )

        connectivity = show_selectable_text(
            "Connectivity",
            "Connectivity",
            model_df,
            original_connectivity,
            "ipad_connectivity"
        )


# ============================================================
# MAC
# ============================================================

elif device_type == "Mac":

    col1, col2 = st.columns(2)

    with col1:

        model_year = show_selectable_numeric(
            "Model Year",
            "Model_Year",
            model_df,
            original_model_year,
            "mac_model_year"
        )

        chipset = show_selectable_text(
            "Chipset",
            "Chipset",
            model_df,
            original_chipset,
            "mac_chipset"
        )

        storage = show_selectable_numeric(
            "Storage",
            "Storage (GB)",
            model_df,
            original_storage,
            "mac_storage",
            format_storage
        )

        ram = show_selectable_numeric(
            "RAM",
            "RAM (GB)",
            model_df,
            original_ram,
            "mac_ram",
            format_ram
        )

    with col2:

        screen_size = show_selectable_numeric(
            "Screen Size",
            "Screen Size (inch)",
            model_df,
            original_screen_size,
            "mac_screen",
            format_screen
        )

        storage_type = show_selectable_text(
            "Storage Type",
            "Storage Type",
            model_df,
            original_storage_type,
            "mac_storage_type"
        )

        clock_speed_raw = show_selectable_text(
            "Clock Speed",
            "Clock Speed",
            model_df,
            original_clock_speed_raw,
            "mac_clock_speed"
        )


# ============================================================
# APPLE WATCH
# ============================================================

elif device_type == "Apple Watch":

    col1, col2 = st.columns(2)

    with col1:

        model_year = show_selectable_numeric(
            "Model Year",
            "Model_Year",
            model_df,
            original_model_year,
            "watch_model_year"
        )

        generation = show_selectable_numeric(
            "Generation",
            "Generation",
            model_df,
            original_generation,
            "watch_generation"
        )

        storage = show_selectable_numeric(
            "Storage",
            "Storage (GB)",
            model_df,
            original_storage,
            "watch_storage",
            format_storage
        )

    with col2:

        screen_size = show_selectable_numeric(
            "Screen Size",
            "Screen Size (inch)",
            model_df,
            original_screen_size,
            "watch_screen",
            format_screen
        )

        connectivity = show_selectable_text(
            "Connectivity",
            "Connectivity",
            model_df,
            original_connectivity,
            "watch_connectivity"
        )


# ============================================================
# AIRPODS
# ============================================================

elif device_type == "AirPods":

    col1, col2 = st.columns(2)

    with col1:

        model_year = show_selectable_numeric(
            "Model Year",
            "Model_Year",
            model_df,
            original_model_year,
            "airpods_model_year"
        )

    with col2:

        generation = show_selectable_numeric(
            "Generation",
            "Generation",
            model_df,
            original_generation,
            "airpods_generation"
        )


# ============================================================
# FALLBACK
# ============================================================

else:

    col1, col2 = st.columns(2)

    with col1:

        model_year = show_selectable_numeric(
            "Model Year",
            "Model_Year",
            model_df,
            original_model_year,
            "fallback_model_year"
        )

    with col2:

        generation = show_selectable_numeric(
            "Generation",
            "Generation",
            model_df,
            original_generation,
            "fallback_generation"
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
# CLOCK SPEED
# ============================================================

clock_speed_ghz = extract_clock_speed(
    clock_speed_raw
)


# ============================================================
# SELECTED DEVICE SUMMARY
# ============================================================

st.divider()

st.subheader("Selected Device")

st.info(
    f"**{model_name}**"
    + (
        f" — {specification}"
        if specification
        else ""
    )
)


# ============================================================
# MARKET BENCHMARK
# ============================================================

st.divider()

st.subheader("Market Benchmark")


# IMPORTANT:
# Market benchmark remains based on the ORIGINAL exact
# Model + Specification selected from the dataset.
#
# Changing a detected specification is for ML estimation.
# It does not invent a market listing that doesn't exist.

market_records = selected_df[
    selected_df[
        "Max. Trade-In Value (RM)"
    ].notna()
].copy()


if market_records.empty:

    st.warning(
        "No observed market price is available for "
        "this configuration."
    )

    market_prices = []

else:

    market_prices = (
        market_records[
            "Max. Trade-In Value (RM)"
        ]
        .astype(float)
        .tolist()
    )


# ============================================================
# PROVIDER COMPARISON
# ============================================================

if not market_records.empty:

    st.write("### Provider Comparison")

    provider_table = (
        market_records
        .groupby("Provider")[
            "Max. Trade-In Value (RM)"
        ]
        .agg(
            ["min", "median", "max", "count"]
        )
        .reset_index()
    )

    provider_table.columns = [
        "Provider",
        "Lowest",
        "Median",
        "Highest",
        "Listings"
    ]

    provider_table["Median"] = (
        provider_table["Median"]
        .round(0)
    )

    display_table = provider_table[
        ["Provider", "Median", "Listings"]
    ].copy()

    display_table["Median"] = (
        display_table["Median"]
        .apply(
            lambda x: f"RM {x:,.0f}"
        )
    )

    display_table.columns = [
        "Provider",
        "Trade-In Value",
        "Listings"
    ]

    st.dataframe(
        display_table,
        width="stretch"
        hide_index=True
    )


# ============================================================
# MARKET RANGE
# ============================================================

market_low = 0.0
market_median = 0.0
market_high = 0.0

if market_prices:

    market_low = min(market_prices)

    market_median = float(
        np.median(market_prices)
    )

    market_high = max(market_prices)

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Lowest",
            f"RM {market_low:,.0f}"
        )

    with c2:

        st.metric(
            "Market Median",
            f"RM {market_median:,.0f}"
        )

    with c3:

        st.metric(
            "Highest",
            f"RM {market_high:,.0f}"
        )

    st.caption(
        f"Based on {len(market_prices)} "
        f"observed market listing(s) for this exact configuration."
    )


# ============================================================
# ML PREDICTION
# ============================================================

st.divider()

st.subheader("Estimated Market Value")


# ------------------------------------------------------------
# STANDARDIZED MODEL
# ------------------------------------------------------------

standardized_model = selected[
    "Standardized Model"
] if "Standardized Model" in selected else ""


# ------------------------------------------------------------
# PROVIDER / CATEGORY
# ------------------------------------------------------------

selected_provider = (
    selected["Provider"]
    if "Provider" in selected
    else "Unknown"
)

selected_category = (
    selected["Pricing Category"]
    if "Pricing Category" in selected
    else "Unknown"
)


# ------------------------------------------------------------
# MODEL INPUT
# ------------------------------------------------------------

input_data = pd.DataFrame([{

    "device_type":
        device_type,

    "model":
        model_name,

    "standardized_model":
        standardized_model,

    "provider":
        selected_provider,

    "pricing_category":
        selected_category,

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
        chipset
        if chipset
        else "Unknown",

    "ram":
        ram,

    "storage":
        storage,

    "storage_type":
        storage_type
        if storage_type
        else "Unknown",

    "connectivity":
        connectivity
        if connectivity
        else "Unknown"
}])


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

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
        "Unable to generate ML prediction."
    )

    st.exception(e)

    st.stop()


st.metric(
    "Estimated Market Value",
    f"RM {ml_prediction:,.0f}"
)


# ============================================================
# COMPETITIVE PRICING RECOMMENDATION
# ============================================================

st.divider()

st.subheader(
    "Competitive Pricing Recommendation"
)


# ============================================================
# DETERMINE MARKET EVIDENCE
# ============================================================

provider_count = 0
market_available = False

if not market_records.empty:

    providers = (
        market_records["Provider"]
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    provider_count = len(providers)

    if market_prices:

        market_available = True


# ============================================================
# RECOMMENDATION LOGIC
# ============================================================

if provider_count >= 3:

    recommended_offer = market_median

    confidence_level = (
        "Strong Market Evidence"
    )

    confidence_description = (
        f"{provider_count} independent providers were "
        "observed for this exact configuration. "
        "The market median is used as the competitive "
        "trade-in recommendation."
    )


elif provider_count == 2:

    recommended_offer = market_median

    confidence_level = (
        "Good Market Evidence"
    )

    confidence_description = (
        "Two independent providers were observed for "
        "this exact configuration. The market median is "
        "used as the competitive trade-in recommendation."
    )


elif provider_count == 1:

    recommended_offer = market_median

    confidence_level = (
        "Limited Market Evidence"
    )

    confidence_description = (
        "Only one independent provider was observed for "
        "this exact configuration. The observed market "
        "price is used as the recommendation, but should "
        "be treated with caution."
    )


else:

    recommended_offer = ml_prediction

    confidence_level = (
        "Model-Based Estimate"
    )

    confidence_description = (
        "No observed market price is available for this "
        "exact configuration. The ML estimate is used "
        "as the recommendation."
    )


# ============================================================
# DISPLAY RECOMMENDATION
# ============================================================

st.markdown(
    f"""
    <div class="recommendation-card">
        <div class="recommendation-title">
            Recommended Trade-In Offer
        </div>
        <div class="recommendation-value">
            RM {recommended_offer:,.0f}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    f"**{confidence_level}**  \n"
    f"{confidence_description}"
)


# ============================================================
# MARKET EVIDENCE
# ============================================================

if market_available:

    st.write("### Market Evidence")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Lowest",
            f"RM {market_low:,.0f}"
        )

    with col2:

        st.metric(
            "Market Median",
            f"RM {market_median:,.0f}"
        )

    with col3:

        st.metric(
            "Highest",
            f"RM {market_high:,.0f}"
        )

    st.caption(
        f"Based on {len(market_prices)} observed "
        f"listing(s) from {provider_count} "
        f"independent provider(s)."
    )


# ============================================================
# MODEL VS MARKET
# ============================================================

st.write("### Model vs Market")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "ML Estimate",
        f"RM {ml_prediction:,.0f}"
    )


with col2:

    if market_available:

        difference = (
            ml_prediction -
            market_median
        )

        percentage_difference = (
            difference /
            market_median *
            100
            if market_median > 0
            else 0
        )

        st.metric(
            "Market Median",
            f"RM {market_median:,.0f}",
            delta=f"{percentage_difference:+.1f}%"
        )

    else:

        st.metric(
            "Market Median",
            "N/A"
        )


# ============================================================
# MARKET POSITION
# ============================================================

if market_available:

    if ml_prediction > market_high:

        st.warning(
            "The ML estimate is above the observed "
            "market range."
        )

    elif ml_prediction < market_low:

        st.info(
            "The ML estimate is below the observed "
            "market range."
        )

    else:

        st.success(
            "The ML estimate falls within the observed "
            "market price range."
        )

else:

    st.info(
        "No direct market benchmark is available. "
        "The ML estimate is being used as the primary "
        "valuation."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "This dashboard combines machine-learning valuation "
    "with observed market trade-in prices. Actual offers "
    "may vary based on device condition, demand, and "
    "current market conditions."
)
