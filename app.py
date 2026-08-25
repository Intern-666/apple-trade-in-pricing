from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

CURRENT_YEAR = 2026

BASE_DIR = Path(__file__).resolve().parent

ICON_PATH = BASE_DIR / "imycom.png"
CSV_PATH = BASE_DIR / "master_apple_final.csv"


st.set_page_config(
    page_title="Apple Trade-In Competitive Pricing",
    page_icon=str(ICON_PATH),
    layout="wide"
)


# ============================================================
# iMALAYSIAN RED THEME + UI
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


    /* ========================================================
       PAGE BACKGROUND
       ======================================================== */

    .stApp {
        background:
            linear-gradient(
                135deg,
                #A80009 0%,
                #E30613 42%,
                #F13A44 72%,
                #8E0008 100%
            );
        background-attachment: fixed;
        color: var(--dark);
    }

    .main {
        background: transparent !important;
    }


    /* ========================================================
       MAIN CONTAINER
       ======================================================== */

    .block-container {
        max-width: 1250px;
        padding: 1.5rem 2rem 3rem 2rem;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .header-card {
        background: rgba(255, 255, 255, 0.97);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18);
    }

    .imalaysian-title {
        color: var(--dark) !important;
        font-size: 2.25rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.6px;
        line-height: 1.15;
        margin-top: 0.5rem !important;
        margin-bottom: 0.4rem !important;
    }

    .imalaysian-subtitle {
        color: var(--grey) !important;
        font-size: 1rem !important;
        font-weight: 400 !important;
        line-height: 1.4;
        margin: 0 !important;
    }


    /* ========================================================
       PANEL SYSTEM
       ======================================================== */

    .panel {
        background: rgba(255, 255, 255, 0.97);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18);
    }

    .panel-title {
        color: var(--dark);
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .panel-subtitle {
        color: var(--grey);
        font-size: 0.9rem;
        margin-bottom: 1.2rem;
    }


    /* ========================================================
       CONDITION PANEL
       ======================================================== */

    .condition-panel {
        animation: conditionFadeIn 0.65s ease-out;
    }

    @keyframes conditionFadeIn {

        0% {
            opacity: 0;
            transform: translateY(12px);
        }

        100% {
            opacity: 1;
            transform: translateY(0);
        }

    }


    /* ========================================================
       CONDITION SCROLL AREA
       ======================================================== */

    .condition-scroll-wrapper {
        max-height: 650px;
        overflow-y: auto;
        padding-right: 8px;
    }

    .condition-scroll-wrapper::-webkit-scrollbar {
        width: 7px;
    }

    .condition-scroll-wrapper::-webkit-scrollbar-track {
        background: #F1F1F1;
        border-radius: 10px;
    }

    .condition-scroll-wrapper::-webkit-scrollbar-thumb {
        background: #D0D0D0;
        border-radius: 10px;
    }

    .condition-scroll-wrapper::-webkit-scrollbar-thumb:hover {
        background: #AAAAAA;
    }


    /* ========================================================
       CONDITION QUESTION CARD
       ======================================================== */

    .condition-question {
        background: #FAFAFA;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.9rem;
    }

    .condition-question-number {
        color: var(--red);
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.25rem;
    }

    .condition-question-title {
        color: var(--dark);
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .condition-question-description {
        color: var(--grey);
        font-size: 0.82rem;
        line-height: 1.4;
        margin-bottom: 0.7rem;
    }


    /* ========================================================
       PROGRESS
       ======================================================== */

    .progress-container {
        background: #F1F1F1;
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
        margin: 0.7rem 0 1.1rem 0;
    }

    .progress-bar {
        background: linear-gradient(
            90deg,
            var(--deep-red),
            var(--red),
            #F13A44
        );
        height: 100%;
        border-radius: 999px;
        transition: width 0.3s ease;
    }


    /* ========================================================
       ESTIMATE CARD
       ======================================================== */

    .recommendation-card {
        background:
            linear-gradient(
                135deg,
                #A80009,
                #E30613,
                #F13A44
            );
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.2rem;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
        animation: estimateFadeIn 0.5s ease-out;
    }

    @keyframes estimateFadeIn {

        0% {
            opacity: 0;
            transform: translateY(10px);
        }

        100% {
            opacity: 1;
            transform: translateY(0);
        }

    }

    .recommendation-title {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .recommendation-value {
        color: var(--white);
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-top: 0.25rem;
    }

    .recommendation-note {
        color: rgba(255, 255, 255, 0.88);
        font-size: 0.78rem;
        margin-top: 0.3rem;
    }


    /* ========================================================
       SELECT BOXES
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: var(--white) !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 7px !important;
        min-height: 42px;
    }

    div[data-baseweb="select"] span {
        color: var(--dark) !important;
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
        border-radius: 7px !important;
        min-height: 44px;
        font-family: "Inter", Arial, sans-serif !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        background-color: var(--dark-red) !important;
        border-color: var(--dark-red) !important;
    }


    /* ========================================================
       RADIO / CHECKBOX
       ======================================================== */

    div[data-testid="stRadio"] label,
    div[data-testid="stCheckbox"] label {
        color: var(--dark) !important;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 1.2rem 0 !important;
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
       MOBILE
       ======================================================== */

    @media (max-width: 900px) {

        .block-container {
            padding: 1rem;
        }

        .imalaysian-title {
            font-size: 1.7rem !important;
        }

        .condition-scroll-wrapper {
            max-height: none;
            overflow-y: visible;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(CSV_PATH)

    data.columns = data.columns.str.strip()

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
    # NORMALIZE STORAGE
    # --------------------------------------------------------

    if "Storage (GB)" in data.columns:

        def parse_storage(value):

            if pd.isna(value):
                return np.nan

            value = str(value).strip().upper()

            try:
                return float(value)

            except ValueError:
                pass

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


# ============================================================
# INITIALIZE DATASET
# ============================================================

try:

    df = load_data()

except Exception as e:

    st.error(
        "Unable to load master_apple_final.csv."
    )

    st.exception(e)
    st.stop()


# ============================================================
# DATASET VALIDATION
# ============================================================

required_columns = [
    "Device",
    "Standardized Model",
    "Max. Trade-In Value (RM)"
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "The master dataset is missing required columns: "
        f"{missing_columns}"
    )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
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

    if series is None:
        return []

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


def show_selectable_numeric(
    label,
    column,
    source_df,
    current_value,
    key,
    formatter=None
):

    if column not in source_df.columns:
        return np.nan

    options = numeric_options(
        source_df[column]
    )

    if pd.notna(current_value):

        current_value = float(current_value)

        if current_value not in options:
            options.insert(0, current_value)

    if not options:
        return np.nan

    formatted_options = [
        formatter(value)
        if formatter
        else str(value)
        for value in options
    ]

    current_index = (
        options.index(current_value)
        if (
            pd.notna(current_value)
            and current_value in options
        )
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

st.markdown(
    """
    <div class="header-card">
    """,
    unsafe_allow_html=True
)

header_col1, header_col2 = st.columns([1, 4])

with header_col1:

    if ICON_PATH.exists():

        st.image(
            str(ICON_PATH),
            width=180
        )

with header_col2:

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

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# MAIN APPLICATION LAYOUT
# ============================================================
#
# LEFT:
#   Condition Evaluation
#
# RIGHT:
#   Device Selection
#
# ============================================================

condition_col, selection_col = st.columns(
    [1.45, 1],
    gap="large"
)


# ============================================================
# DEVICE SELECTION — RIGHT PANEL
# ============================================================

with selection_col:

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">
                Device Selection
            </div>

            <div class="panel-subtitle">
                Select the device and configuration.
            </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DEVICE TYPE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DEVICE DATA
    # --------------------------------------------------------

    device_df = df[
        df["Device"].astype(str).str.strip()
        == device_type
    ].copy()

    if device_df.empty:

        st.warning(
            "No records are available for this device type."
        )

        st.stop()

    # --------------------------------------------------------
    # SUB-DEVICE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    models = clean_options(
        device_df["Standardized Model"]
    )

    if not models:

        st.warning(
            "No models are available for this configuration."
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

    # --------------------------------------------------------
    # INITIAL VALUES
    # --------------------------------------------------------

    connectivity = ""
    charging_method = ""
    storage = np.nan
    storage_type = ""

    # --------------------------------------------------------
    # STORAGE OPTIONS
    # --------------------------------------------------------

    storage_options = numeric_options(
        model_df["Storage (GB)"]
        if "Storage (GB)" in model_df.columns
        else pd.Series(dtype=float)
    )

    # ========================================================
    # IPHONE
    # ========================================================

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

    # ========================================================
    # IPAD
    # ========================================================

    elif device_type == "iPad":

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

    # ========================================================
    # MAC
    # ========================================================

    elif device_type == "Mac":

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

    # ========================================================
    # APPLE WATCH
    # ========================================================

    elif device_type == "Apple Watch":

        matching_df = model_df.copy()

        # ----------------------------------------------------
        # CASE SIZE
        # ----------------------------------------------------

        case_size_options = clean_options(
            matching_df["Specification"]
            if "Specification" in matching_df.columns
            else pd.Series(dtype=str)
        )

        if case_size_options:

            case_size = st.selectbox(
                "Case Size",
                case_size_options,
                key="watch_case_size"
            )

            matching_df = matching_df[
                matching_df["Specification"].astype(str).str.strip()
                == case_size
            ].copy()

        # ----------------------------------------------------
        # MATERIAL
        # ----------------------------------------------------

        material_options = clean_options(
            matching_df["Material"]
            if "Material" in matching_df.columns
            else pd.Series(dtype=str)
        )

        if material_options:

            material = st.selectbox(
                "Material",
                ["Not specified"] + material_options,
                key="watch_material"
            )

            if material == "Not specified":
                material = ""

            if material:

                matching_df = matching_df[
                    matching_df["Material"].astype(str).str.strip()
                    == material
                ].copy()

        # ----------------------------------------------------
        # STORAGE
        # ----------------------------------------------------

        watch_storage_options = numeric_options(
            matching_df["Storage (GB)"]
            if "Storage (GB)" in matching_df.columns
            else pd.Series(dtype=float)
        )

        if watch_storage_options:

            storage = show_selectable_numeric(
                "Storage",
                "Storage (GB)",
                matching_df,
                watch_storage_options[0],
                "watch_storage",
                format_storage
            )

            matching_df = matching_df[
                matching_df["Storage (GB)"] == storage
            ].copy()

        # ----------------------------------------------------
        # CONNECTIVITY
        # ----------------------------------------------------

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

    # ========================================================
    # AIRPODS
    # ========================================================

    elif device_type == "AirPods":

        matching_df = model_df.copy()

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

        storage = np.nan

    else:

        matching_df = model_df.copy()

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# MARKET PRICE CALCULATION
# ============================================================

# Exact selected configuration first.

market_records = matching_df.copy()

market_prices = pd.to_numeric(
    market_records["Max. Trade-In Value (RM)"],
    errors="coerce"
).dropna()


# ------------------------------------------------------------
# FALLBACK TO MODEL
# ------------------------------------------------------------

used_model_fallback = False

if market_prices.empty:

    market_records = model_df.copy()

    market_prices = pd.to_numeric(
        market_records["Max. Trade-In Value (RM)"],
        errors="coerce"
    ).dropna()

    used_model_fallback = True


# ------------------------------------------------------------
# FINAL VALIDATION
# ------------------------------------------------------------

if market_prices.empty:

    st.error(
        "No market trade-in price is available for this device "
        "configuration."
    )

    st.stop()


# ------------------------------------------------------------
# MEDIAN
# ------------------------------------------------------------

recommended_offer = float(
    market_prices.median()
)


# ------------------------------------------------------------
# PROVIDER COUNT
# ------------------------------------------------------------

provider_count = 0

if "Provider" in market_records.columns:

    provider_count = (
        market_records["Provider"]
        .replace("", np.nan)
        .dropna()
        .nunique()
    )


# ------------------------------------------------------------
# CONFIDENCE TEXT
# ------------------------------------------------------------

if provider_count >= 3:

    confidence_text = (
        "Based on observed market prices from multiple providers."
    )

elif provider_count == 2:

    confidence_text = (
        "Based on observed market prices from two providers."
    )

elif provider_count == 1:

    confidence_text = (
        "Based on observed market pricing from one provider."
    )

else:

    confidence_text = (
        "Based on available market trade-in pricing."
    )


# ============================================================
# CONDITION EVALUATION — LEFT PANEL
# ============================================================

with condition_col:

    st.markdown(
        """
        <div class="panel condition-panel">

            <div class="panel-title">
                Condition Evaluation
            </div>

            <div class="panel-subtitle">
                Tell us about the physical condition of your device.
            </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SELECTED DEVICE SUMMARY
    # --------------------------------------------------------

    storage_display = (
        format_storage(storage)
        if pd.notna(storage)
        else "N/A"
    )

    st.info(
        f"Selected device: **{device_type} — {model_name}**"
        + (
            f" · **{storage_display}**"
            if storage_display != "N/A"
            else ""
        )
    )

    # --------------------------------------------------------
    # PLACEHOLDER CONDITION QUESTIONS
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # These are intentionally placeholders.
    #
    # We will later replace these with DEVICE-SPECIFIC
    # questions based on Sub-device.
    #
    # --------------------------------------------------------

    condition_questions = [
        {
            "number": "01",
            "title": "Overall physical condition",
            "description": (
                "Placeholder question. We will later determine "
                "the appropriate condition criteria for this device."
            ),
            "options": [
                "Excellent",
                "Good",
                "Fair",
                "Poor"
            ],
            "key": "condition_overall"
        },
        {
            "number": "02",
            "title": "Screen / display condition",
            "description": (
                "Placeholder question. This will only appear for "
                "devices that actually have a display."
            ),
            "options": [
                "No visible issues",
                "Minor marks",
                "Visible damage",
                "Severe damage"
            ],
            "key": "condition_display"
        },
        {
            "number": "03",
            "title": "Body / exterior condition",
            "description": (
                "Placeholder question. The final version will "
                "use device-specific exterior components."
            ),
            "options": [
                "Excellent",
                "Minor wear",
                "Noticeable wear",
                "Major damage"
            ],
            "key": "condition_body"
        },
        {
            "number": "04",
            "title": "Functional condition",
            "description": (
                "Placeholder question. The final version will "
                "contain the appropriate functional checks."
            ),
            "options": [
                "Everything works",
                "Minor issue",
                "Some functions affected",
                "Major functional issue"
            ],
            "key": "condition_function"
        }
    ]

    total_questions = len(condition_questions)

    answered_questions = 0

    # --------------------------------------------------------
    # SCROLLABLE QUESTION AREA
    # --------------------------------------------------------

    st.markdown(
        '<div class="condition-scroll-wrapper">',
        unsafe_allow_html=True
    )

    for question in condition_questions:

        st.markdown(
            f"""
            <div class="condition-question">

                <div class="condition-question-number">
                    Question {question["number"]}
                </div>

                <div class="condition-question-title">
                    {question["title"]}
                </div>

                <div class="condition-question-description">
                    {question["description"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        answer = st.radio(
            question["title"],
            question["options"],
            index=None,
            key=question["key"],
            label_visibility="collapsed"
        )

        if answer is not None:
            answered_questions += 1

        st.markdown(
            "<div style='height: 4px;'></div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    progress = (
        answered_questions / total_questions
        if total_questions > 0
        else 0
    )

    progress_percent = int(
        progress * 100
    )

    st.markdown(
        f"""
        <div style="
            margin-top: 1rem;
            font-size: 0.8rem;
            color: #666666;
            font-weight: 600;
        ">
            Condition evaluation progress:
            {answered_questions}/{total_questions}
        </div>

        <div class="progress-container">
            <div
                class="progress-bar"
                style="width: {progress_percent}%"
            ></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # ESTIMATE AT BOTTOM OF CONDITION PANEL
    # ========================================================

    if answered_questions == total_questions:

        st.markdown(
            """
            <div style="
                margin-top: 1rem;
                color: #666666;
                font-size: 0.82rem;
                text-align: center;
            ">
                Condition evaluation complete.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="recommendation-card">

                <div class="recommendation-title">
                    Market Estimate
                </div>

                <div class="recommendation-value">
                    RM {recommended_offer:,.0f}
                </div>

                <div class="recommendation-note">
                    Median of available market trade-in values
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            confidence_text
        )

        if used_model_fallback:

            st.caption(
                "No priced record was available for the exact "
                "configuration, so the estimate uses available "
                "priced records for the selected model."
            )

        else:

            st.caption(
                "The estimate is based on the selected device "
                "configuration. Condition valuation will be "
                "applied in the next stage."
            )

    else:

        st.markdown(
            """
            <div style="
                background: #F8F8F8;
                border: 1px solid #E2E2E2;
                border-radius: 10px;
                padding: 1rem;
                margin-top: 1rem;
                text-align: center;
            ">
                <div style="
                    color: #666666;
                    font-size: 0.85rem;
                    font-weight: 600;
                ">
                    Complete all condition questions to view
                    the market estimate.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )