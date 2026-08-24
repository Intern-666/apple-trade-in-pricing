from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import re


# ============================================================
# CONFIGURATION
# ============================================================

ICON_PATH = Path(__file__).parent / "imycom.png"

st.set_page_config(
    page_title="Apple Trade-In Competitive Pricing",
    page_icon=str(ICON_PATH),
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
    label {
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
    "master_apple_final.csv"
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


# ============================================================
# DEVICE TYPE
# ============================================================

device_types = clean_options(
    df["Device"]
    if "Device" in df.columns
    else pd.Series(dtype=str)
)

if not device_types:
    st.error("No device types are available in the master dataset.")
    st.stop()

device_type = st.selectbox(
    "Device Type",
    device_types,
    key="device_type_selection"
)


# ============================================================
# DEVICE DATA
# ============================================================

device_df = df[
    df["Device"].astype(str).str.strip() == device_type
].copy()


if device_df.empty:
    st.warning(
        "No records are available for the selected device type."
    )
    st.stop()


# ============================================================
# SUB-DEVICE
# ============================================================
#
# iPhone:
#     No Sub-device selection.
#
# Other devices:
#     Sub-device improves UX and narrows the available models.
#
# ============================================================

sub_device = ""


if device_type != "iPhone" and "Sub-device" in device_df.columns:

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
#
# IMPORTANT:
# The finalized master dataset uses:
#
#     Standardized Model
#
# rather than the old "Model" column.
#
# ============================================================

if "Standardized Model" not in device_df.columns:
    st.error(
        "The master dataset does not contain "
        "'Standardized Model'."
    )
    st.stop()


models = clean_options(
    device_df["Standardized Model"]
)


if not models:

    st.warning(
        "No models are available for the selected configuration."
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
        "No records were found for the selected model."
    )
    st.stop()


# ============================================================
# INITIAL VALUES
# ============================================================

connectivity = ""
charging_method = ""
storage = np.nan
storage_type = ""


# ============================================================
# STORAGE OPTIONS
# ============================================================

storage_options = numeric_options(
    model_df["Storage (GB)"]
    if "Storage (GB)" in model_df.columns
    else pd.Series(dtype=float)
)


# ============================================================
# IPHONE
# ============================================================

if device_type == "iPhone":

    if storage_options:

        storage = show_selectable_numeric(
            "Storage",
            "Storage (GB)",
            model_df,
            storage_options[0],
            "iphone_storage",
            format_storage
        )

        matching_df = model_df[
            model_df["Storage (GB)"] == storage
        ].copy()

    else:

        matching_df = model_df.copy()


# ============================================================
# IPAD
# ============================================================

elif device_type == "iPad":

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    if storage_options:

        storage = show_selectable_numeric(
            "Storage",
            "Storage (GB)",
            model_df,
            storage_options[0],
            "ipad_storage",
            format_storage
        )

        matching_df = model_df[
            model_df["Storage (GB)"] == storage
        ].copy()

    else:

        matching_df = model_df.copy()


    # --------------------------------------------------------
    # CONNECTIVITY
    # --------------------------------------------------------

    connectivity_options = clean_options(
        matching_df["Connectivity"]
        if "Connectivity" in matching_df.columns
        else pd.Series(dtype=str)
    )

    if connectivity_options:

        connectivity = st.selectbox(
            "Connectivity",
            ["Not specified"] + connectivity_options,
            key="ipad_connectivity"
        )

        if connectivity == "Not specified":
            connectivity = ""

        if connectivity:

            matching_df = matching_df[
                matching_df["Connectivity"].astype(str).str.strip()
                == connectivity
            ].copy()


# ============================================================
# MAC
# ============================================================

elif device_type == "Mac":

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    if storage_options:

        storage = show_selectable_numeric(
            "Storage",
            "Storage (GB)",
            model_df,
            storage_options[0],
            "mac_storage",
            format_storage
        )

        matching_df = model_df[
            model_df["Storage (GB)"] == storage
        ].copy()

    else:

        matching_df = model_df.copy()


    # --------------------------------------------------------
    # STORAGE TYPE
    # --------------------------------------------------------

    storage_type_options = clean_options(
        matching_df["Storage Type"]
        if "Storage Type" in matching_df.columns
        else pd.Series(dtype=str)
    )

    if storage_type_options:

        storage_type = st.selectbox(
            "Storage Type",
            ["Not specified"] + storage_type_options,
            key="mac_storage_type"
        )

        if storage_type == "Not specified":
            storage_type = ""

        if storage_type:

            matching_df = matching_df[
                matching_df["Storage Type"].astype(str).str.strip()
                == storage_type
            ].copy()


# ============================================================
# APPLE WATCH
# ============================================================

elif device_type == "Apple Watch":

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    if storage_options:

        storage = show_selectable_numeric(
            "Storage",
            "Storage (GB)",
            model_df,
            storage_options[0],
            "watch_storage",
            format_storage
        )

        matching_df = model_df[
            model_df["Storage (GB)"] == storage
        ].copy()

    else:

        # Storage is not available for some Watch records.
        storage = np.nan
        matching_df = model_df.copy()


    # --------------------------------------------------------
    # CONNECTIVITY
    # --------------------------------------------------------

    connectivity_options = clean_options(
        matching_df["Connectivity"]
        if "Connectivity" in matching_df.columns
        else pd.Series(dtype=str)
    )

    if connectivity_options:

        connectivity = st.selectbox(
            "Connectivity",
            ["Not specified"] + connectivity_options,
            key="watch_connectivity"
        )

        if connectivity == "Not specified":
            connectivity = ""

        if connectivity:

            matching_df = matching_df[
                matching_df["Connectivity"].astype(str).str.strip()
                == connectivity
            ].copy()


# ============================================================
# AIRPODS
# ============================================================

elif device_type == "AirPods":

    matching_df = model_df.copy()


    # --------------------------------------------------------
    # CHARGING METHOD
    # --------------------------------------------------------
    #
    # AirPods use Specification for charging information.
    #
    # --------------------------------------------------------

    charging_options = clean_options(
        matching_df["Specification"]
        if "Specification" in matching_df.columns
        else pd.Series(dtype=str)
    )

    if charging_options:

        charging_method = st.selectbox(
            "Charging Method",
            ["Not specified"] + charging_options,
            key="airpods_charging_method"
        )

        if charging_method == "Not specified":
            charging_method = ""

        if charging_method:

            matching_df = matching_df[
                matching_df["Specification"].astype(str).str.strip()
                == charging_method
            ].copy()


    # AirPods do not use storage as a customer-facing selection.
    storage = np.nan


# ============================================================
# FALLBACK
# ============================================================

else:

    matching_df = model_df.copy()


# ============================================================
# VALIDATE MARKET MATCH
# ============================================================

if matching_df.empty:

    st.warning(
        "No exact market configuration was found for "
        "the selected specifications. The ML model will "
        "be used for estimation."
    )

    selected_df = model_df.copy()

else:

    selected_df = matching_df.copy()


# ============================================================
# SELECTED RECORD
# ============================================================

selected = selected_df.iloc[0]


# ============================================================
# ORIGINAL DETECTED VALUES
# ============================================================

original_model_year = (
    selected["Model_Year"]
    if "Model_Year" in selected.index
    else np.nan
)

original_generation = (
    selected["Generation"]
    if "Generation" in selected.index
    else np.nan
)

original_screen_size = (
    selected["Screen Size (inch)"]
    if "Screen Size (inch)" in selected.index
    else np.nan
)

original_chipset = (
    selected["Chipset"]
    if "Chipset" in selected.index
    else ""
)

original_ram = (
    selected["RAM (GB)"]
    if "RAM (GB)" in selected.index
    else np.nan
)

original_storage = (
    selected["Storage (GB)"]
    if "Storage (GB)" in selected.index
    else np.nan
)

original_storage_type = (
    selected["Storage Type"]
    if "Storage Type" in selected.index
    else ""
)

original_connectivity = (
    selected["Connectivity"]
    if "Connectivity" in selected.index
    else ""
)

original_clock_speed_raw = (
    selected["Clock Speed"]
    if "Clock Speed" in selected.index
    else ""
)

original_specification = (
    selected["Specification"]
    if "Specification" in selected.index
    else ""
)

# ============================================================
# BACKEND VALUES
# ============================================================

model_year = original_model_year
generation = original_generation
screen_size = original_screen_size
chipset = original_chipset
ram = original_ram
clock_speed_raw = original_clock_speed_raw


# Use the user's selected configuration where applicable.
if storage is not np.nan:
    if pd.notna(storage):
        original_storage = storage

storage = original_storage

if storage_type:
    original_storage_type = storage_type

storage_type = original_storage_type


# ============================================================
# DEVICE DETAILS
# ============================================================

st.divider()

st.subheader("Device Details")

st.caption(
    "These values are detected from the selected dataset "
    "record and are used by the valuation model."
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

selected_description: str = str(model_name)


if sub_device:
    selected_description += f" — {str(sub_device)}"


if pd.notna(storage):
    selected_description += f" — {format_storage(storage)}"


if storage_type:
    selected_description += f" — {str(storage_type)}"


if connectivity:
    selected_description += f" — {str(connectivity)}"


if charging_method:
    selected_description += f" — {str(charging_method)}"


st.info(
    f"**{selected_description}**"
)


# ============================================================
# MARKET EVIDENCE
# ============================================================

market_records = selected_df[
    selected_df["Max. Trade-In Value (RM)"].notna()
].copy()


market_prices = (
    market_records[
        "Max. Trade-In Value (RM)"
    ]
    .astype(float)
    .tolist()
)


# ============================================================
# MARKET MEDIAN
# ============================================================

if market_prices:

    market_median = float(
        np.median(market_prices)
    )

else:

    market_median = np.nan


# ============================================================
# ML INPUT
# ============================================================

input_data = pd.DataFrame([{

    "device_type":
        device_type,

    "model":
        model_name,

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
        chipset if chipset else "Unknown",

    "ram":
        ram,

    "storage":
        storage,

    "storage_type":
        storage_type if storage_type else "Unknown",

    "connectivity":
        connectivity if connectivity else np.nan

}])


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
        "Unable to generate ML prediction."
    )

    st.exception(e)

    st.stop()


# ============================================================
# FINAL VALUATION
# ============================================================

if market_prices:

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

else:

    recommended_offer = ml_prediction

    confidence_text = (
        "Estimated using the valuation model "
        "because no direct market price was available."
    )


# ============================================================
# CUSTOMER-FACING VALUATION
# ============================================================

st.divider()

st.subheader("Estimated Trade-In Value")

st.metric(
    "Recommended Trade-In Offer",
    f"RM {recommended_offer:,.0f}"
)

st.caption(
    confidence_text
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