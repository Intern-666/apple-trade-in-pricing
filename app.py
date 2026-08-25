from pathlib import Path
import re
import textwrap

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
    layout="centered"
)


# ============================================================
# iMALAYSIAN RED THEME
# ============================================================

st.markdown(
    textwrap.dedent("""
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
        --light-red: #FFF1F2;

        --white: #FFFFFF;
        --off-white: #F8F8F8;

        --dark: #1F1F1F;
        --grey: #666666;
        --light-grey: #888888;

        --border: #E2E2E2;
    }


    html,
    body,
    [class*="css"],
    .stApp {
        font-family: "Inter", Arial, sans-serif !important;
    }


    /* ========================================================
       APP BACKGROUND
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(255,255,255,0.12),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #E30613 0%,
                #C4000B 45%,
                #A80009 100%
            );

        color: var(--dark);
    }


    .main {
        background: transparent;
    }


    /* ========================================================
       MAIN CONTAINER
       ======================================================== */

    .block-container {
        background-color: var(--white);

        border-radius: 16px;

        padding:
            1.5rem
            2.2rem
            3rem
            2.2rem;

        margin-top: 1rem;
        margin-bottom: 2rem;

        max-width: 1100px;

        box-shadow:
            0 8px 30px rgba(0, 0, 0, 0.18);
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .imalaysian-title {
        color: var(--dark) !important;

        font-size: 2.5rem !important;
        font-weight: 800 !important;

        letter-spacing: -0.5px;
        line-height: 1.2;

        margin-top: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }


    .imalaysian-subtitle {
        color: var(--dark) !important;

        font-size: 1.15rem !important;
        font-weight: 400 !important;

        line-height: 1.4;

        margin-top: 0 !important;

        opacity: 0.92;
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    h1 {
        color: var(--white) !important;
        font-weight: 800 !important;
    }


    h2,
    h3 {
        color: var(--dark) !important;

        font-weight: 700 !important;

        letter-spacing: -0.3px;
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


    /* ========================================================
       PANELS
       ======================================================== */

    .selection-panel,
    .condition-panel {

        background-color: var(--white);

        border: 1px solid var(--border);

        border-radius: 12px;

        padding: 1.4rem;

        box-shadow:
            0 3px 12px rgba(0, 0, 0, 0.06);
    }


    .condition-panel {

        animation:
            conditionFadeIn
            0.55s
            ease-out
            forwards;
    }


    @keyframes conditionFadeIn {

        from {
            opacity: 0;
            transform: translateY(18px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }

    }


    .panel-title {

        color: var(--dark);

        font-size: 1.35rem;

        font-weight: 800;

        letter-spacing: -0.3px;

        margin-bottom: 0.25rem;
    }


    .panel-subtitle {

        color: var(--grey);

        font-size: 0.92rem;

        line-height: 1.5;

        margin-bottom: 1.25rem;
    }


    /* ========================================================
       PANEL HEADER RED ACCENT
       ======================================================== */

    .panel-title::after {

        content: "";

        display: block;

        width: 38px;
        height: 3px;

        background-color: var(--red);

        margin-top: 7px;

        border-radius: 2px;
    }


    /* ========================================================
       SELECT BOXES
       ======================================================== */

    div[data-baseweb="select"] > div {

        background-color:
            var(--secondary-background-color)
            !important;

        border: 1px solid #CCCCCC !important;

        border-radius: 6px !important;

        min-height: 42px;
    }


    div[data-baseweb="select"] span {

        color:
            var(--text-color)
            !important;
    }


    div[data-baseweb="select"] > div:hover {

        border-color:
            var(--red)
            !important;
    }


    div[data-baseweb="select"] > div:focus-within {

        border-color:
            var(--red)
            !important;

        box-shadow:
            0 0 0 1px var(--red)
            !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {

        background-color:
            var(--red)
            !important;

        color:
            var(--white)
            !important;

        border:
            1px solid var(--red)
            !important;

        border-radius: 6px !important;

        min-height: 46px;

        font-family:
            "Inter",
            Arial,
            sans-serif
            !important;

        font-weight: 700 !important;

        font-size: 0.95rem !important;
    }


    .stButton > button:hover {

        background-color:
            var(--dark-red)
            !important;

        border-color:
            var(--dark-red)
            !important;

        color:
            var(--white)
            !important;
    }


    /* ========================================================
       CONDITION PLACEHOLDER
       ======================================================== */

    .condition-placeholder {

        background-color: var(--off-white);

        border: 1px dashed #D5D5D5;

        border-radius: 10px;

        padding: 1.25rem;

        margin-top: 0.5rem;

        margin-bottom: 1.5rem;
    }


    .condition-placeholder-title {

        color: var(--dark);

        font-weight: 700;

        font-size: 0.95rem;

        margin-bottom: 0.35rem;
    }


    .condition-placeholder-text {

        color: var(--grey);

        font-size: 0.88rem;

        line-height: 1.5;
    }


    /* ========================================================
       CONDITION QUESTIONS PLACEHOLDER
       ======================================================== */

    .question-placeholder {

        border: 1px solid var(--border);

        border-radius: 9px;

        padding: 1rem;

        margin-bottom: 0.8rem;

        background-color: var(--white);
    }


    .question-placeholder-number {

        color: var(--red);

        font-size: 0.78rem;

        font-weight: 800;

        text-transform: uppercase;

        letter-spacing: 0.7px;

        margin-bottom: 0.25rem;
    }


    .question-placeholder-text {

        color: var(--dark);

        font-size: 0.93rem;

        font-weight: 600;
    }


    /* ========================================================
       RECOMMENDATION
       ======================================================== */

    .recommendation-card {

        background:
            linear-gradient(
                135deg,
                var(--red),
                var(--dark-red)
            );

        border-radius: 10px;

        padding: 1.6rem;

        margin: 1rem 0;

        text-align: center;

        box-shadow:
            0 5px 14px rgba(0, 0, 0, 0.15);
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

        font-family:
            "Inter",
            Arial,
            sans-serif;

        font-size: 2.8rem;

        font-weight: 800;

        letter-spacing: -1px;

        margin-top: 0.3rem;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {

        background-color:
            var(--white);

        border:
            1px solid var(--border);

        border-radius: 8px;

        padding: 1rem;

        box-shadow:
            0 2px 7px rgba(0, 0, 0, 0.05);
    }


    div[data-testid="stMetricLabel"] {

        color:
            var(--grey)
            !important;

        font-weight:
            600
            !important;
    }


    div[data-testid="stMetricValue"] {

        color:
            var(--dark)
            !important;

        font-weight:
            800
            !important;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {

        border:
            none
            !important;

        border-top:
            1px solid var(--border)
            !important;

        margin:
            1.4rem 0
            !important;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {

        border-radius:
            7px
            !important;
    }


    /* ========================================================
       TABLE
       ======================================================== */

    div[data-testid="stDataFrame"] {

        border:
            1px solid var(--border);

        border-radius:
            7px;

        overflow:
            hidden;
    }


    /* ========================================================
       TEXT
       ======================================================== */

    p,
    label {

        color:
            var(--dark);
    }


    .stCaption {

        color:
            var(--grey)
            !important;
    }


    </style>
    """),
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


            number = float(
                match.group(1)
            )


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

        current_value = float(
            current_value
        )


        if current_value not in options:

            options.insert(
                0,
                current_value
            )


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
        formatted_options.index(
            selected_display
        )
    ]


# ============================================================
# HEADER
# ============================================================

col1, col2 = st.columns(
    [1, 3],
    vertical_alignment="center"
)


with col1:

    if ICON_PATH.exists():

        st.image(
            str(ICON_PATH),
            width=220
        )


with col2:

    st.markdown(
        textwrap.dedent("""
        <div class="imalaysian-title">
            Apple Trade-In Competitive Pricing
        </div>

        <div class="imalaysian-subtitle">
            Market-based trade-in valuation and competitive pricing
        </div>
        """),
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# MAIN WORKSPACE
# ============================================================

selection_col, condition_col = st.columns(
    [0.9, 1.1],
    gap="large"
)


# ============================================================
# DEVICE SELECTION PANEL
# ============================================================

with selection_col:

    st.html(
        textwrap.dedent("""
        <div class="selection-panel">

            <div class="panel-title">
                Device Selection
            </div>

            <div class="panel-subtitle">
                Select the device and configuration you want
                to evaluate.
            </div>

        </div>
        """),
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
            "No records are available for the selected "
            "device type."
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
                device_df["Sub-device"]
                .astype(str)
                .str.strip()
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
            "No models are available for the selected "
            "configuration."
        )

        st.stop()


    model_name = st.selectbox(
        "Model",
        models,
        key="model_selection"
    )


    model_df = device_df[
        device_df["Standardized Model"]
        .astype(str)
        .str.strip()
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
                model_df["Storage (GB)"]
                == storage
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
                model_df["Storage (GB)"]
                == storage
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
                ["Not specified"]
                + connectivity_options,
                key="ipad_connectivity"
            )


            if connectivity == "Not specified":

                connectivity = ""


            if connectivity:

                matching_df = matching_df[
                    matching_df["Connectivity"]
                    .astype(str)
                    .str.strip()
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
                model_df["Storage (GB)"]
                == storage
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
                ["Not specified"]
                + storage_type_options,
                key="mac_storage_type"
            )


            if storage_type == "Not specified":

                storage_type = ""


            if storage_type:

                matching_df = matching_df[
                    matching_df["Storage Type"]
                    .astype(str)
                    .str.strip()
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

            if "Specification"
            in matching_df.columns

            else pd.Series(dtype=str)
        )


        if case_size_options:

            case_size = st.selectbox(
                "Case Size",
                case_size_options,
                key="watch_case_size"
            )


            matching_df = matching_df[
                matching_df["Specification"]
                .astype(str)
                .str.strip()
                == case_size
            ].copy()


        # ----------------------------------------------------
        # MATERIAL
        # ----------------------------------------------------

        material_options = clean_options(

            matching_df["Material"]

            if "Material"
            in matching_df.columns

            else pd.Series(dtype=str)
        )


        if material_options:

            material = st.selectbox(
                "Material",
                ["Not specified"]
                + material_options,
                key="watch_material"
            )


            if material == "Not specified":

                material = ""


            if material:

                matching_df = matching_df[
                    matching_df["Material"]
                    .astype(str)
                    .str.strip()
                    == material
                ].copy()


        # ----------------------------------------------------
        # STORAGE
        # ----------------------------------------------------

        watch_storage_options = numeric_options(

            matching_df["Storage (GB)"]

            if "Storage (GB)"
            in matching_df.columns

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
                matching_df["Storage (GB)"]
                == storage
            ].copy()


        # ----------------------------------------------------
        # CONNECTIVITY
        # ----------------------------------------------------

        connectivity_options = clean_options(

            matching_df["Connectivity"]

            if "Connectivity"
            in matching_df.columns

            else pd.Series(dtype=str)
        )


        if connectivity_options:

            connectivity = st.selectbox(
                "Connectivity",
                ["Not specified"]
                + connectivity_options,
                key="watch_connectivity"
            )


            if connectivity == "Not specified":

                connectivity = ""


            if connectivity:

                matching_df = matching_df[
                    matching_df["Connectivity"]
                    .astype(str)
                    .str.strip()
                    == connectivity
                ].copy()


    # ========================================================
    # AIRPODS
    # ========================================================

    elif device_type == "AirPods":

        matching_df = model_df.copy()


        charging_options = clean_options(

            matching_df["Specification"]

            if "Specification"
            in matching_df.columns

            else pd.Series(dtype=str)
        )


        if charging_options:

            charging_method = st.selectbox(
                "Charging Method",
                ["Not specified"]
                + charging_options,
                key="airpods_charging_method"
            )


            if charging_method == "Not specified":

                charging_method = ""


            if charging_method:

                matching_df = matching_df[
                    matching_df["Specification"]
                    .astype(str)
                    .str.strip()
                    == charging_method
                ].copy()


        storage = np.nan


    # ========================================================
    # FALLBACK
    # ========================================================

    else:

        matching_df = model_df.copy()


# ============================================================
# CONDITION EVALUATION PANEL
# ============================================================

with condition_col:

    st.markdown(
        textwrap.dedent("""
        <div class="condition-panel">

            <div class="panel-title">
                Condition Evaluation
            </div>

            <div class="panel-subtitle">
                Tell us about the physical condition of your
                device.
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PLACEHOLDER INTRODUCTION
    # --------------------------------------------------------

    st.markdown(
        textwrap.dedent("""
        <div class="condition-placeholder">

            <div class="condition-placeholder-title">
                Condition assessment
            </div>

            <div class="condition-placeholder-text">
                Condition questions will appear here based on
                the selected device and sub-device.
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PLACEHOLDER QUESTIONS
    # --------------------------------------------------------

    placeholder_questions = [

        "Display / screen condition",

        "Body and exterior condition",

        "Buttons, ports and physical components",

        "Functional condition",

        "Signs of major damage"
    ]


    for index, question in enumerate(
        placeholder_questions,
        start=1
    ):

        st.html(
            textwrap.dedent(f"""
            <div class="question-placeholder">

                <div class="question-placeholder-number">
                    Question {index}
                </div>

                <div class="question-placeholder-text">
                    {question}
                </div>

            </div>
            """),
        )


        st.selectbox(
            "Condition",
            [
                "Placeholder — Excellent",
                "Placeholder — Good",
                "Placeholder — Fair",
                "Placeholder — Poor"
            ],
            key=f"condition_placeholder_{index}",
            label_visibility="collapsed"
        )


    # ========================================================
    # MARKET PRICE CALCULATION
    # ========================================================

    st.divider()

    st.markdown(
        textwrap.dedent("""
        <div class="panel-title">
            Market Estimate
        </div>

        <div class="panel-subtitle">
            Current market value before condition adjustment.
        </div>
        """),
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SELECTED MARKET RECORDS
    # --------------------------------------------------------

    market_records = matching_df.copy()


    market_prices = pd.to_numeric(
        market_records[
            "Max. Trade-In Value (RM)"
        ],
        errors="coerce"
    ).dropna()


    # --------------------------------------------------------
    # FALLBACK TO MODEL
    # --------------------------------------------------------

    if market_prices.empty:

        market_records = model_df.copy()


        market_prices = pd.to_numeric(
            market_records[
                "Max. Trade-In Value (RM)"
            ],
            errors="coerce"
        ).dropna()


    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if market_prices.empty:

        st.error(
            "No market trade-in price is available for this "
            "device configuration."
        )

        st.stop()


    # --------------------------------------------------------
    # MEDIAN
    # --------------------------------------------------------

    recommended_offer = float(
        market_prices.median()
    )


    # ========================================================
    # PROVIDER INFORMATION
    # ========================================================

    provider_count = 0


    if "Provider" in market_records.columns:

        provider_count = (
            market_records["Provider"]
            .replace("", np.nan)
            .dropna()
            .nunique()
        )


    # ========================================================
    # CONFIDENCE TEXT
    # ========================================================

    if provider_count >= 3:

        confidence_text = (
            "Based on observed market prices from "
            "multiple providers."
        )

    elif provider_count == 2:

        confidence_text = (
            "Based on observed market prices from "
            "two providers."
        )

    elif provider_count == 1:

        confidence_text = (
            "Based on observed market pricing from "
            "one provider."
        )

    else:

        confidence_text = (
            "Based on available market trade-in pricing."
        )


    # ========================================================
    # CUSTOMER-FACING VALUATION
    # ========================================================

    st.markdown(
        textwrap.dedent(f"""
        <div class="recommendation-card">

            <div class="recommendation-title">
                Market Estimate
            </div>

            <div class="recommendation-value">
                RM {recommended_offer:,.0f}
            </div>

        </div>
        """),
        unsafe_allow_html=True
    )


    st.caption(
        confidence_text
    )


    st.caption(
        "The displayed market estimate is the median of "
        "available trade-in values for the selected device "
        "configuration."
    )