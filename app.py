"""
Walmart Weekly Sales Predictor
================================
A professional, production-style Streamlit dashboard that serves predictions
from a pre-trained RandomForestRegressor saved at:
    models/walmart_sales_model.joblib

This app does NOT train or modify the model in any way. It only loads it
and uses it for inference.

Run with:
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_PATH = "model/walmart_sales_model.joblib"

# The exact feature order the model was trained on. Used only to validate
# that the loaded model's saved feature list matches what this app expects.
EXPECTED_FEATURES = [
    "Store",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Year",
    "Month",
    "Week",
    "Previous_Week_Sales",
    "Previous_4_Weeks_Sales",
]

# Reported offline model performance (from training/evaluation, not computed here)
MODEL_METRICS = {
    "MAE": "$45,153",
    "RMSE": "$64,937",
    "R2": "0.9852",
}

# Reported feature importance (from training, not computed here)
FEATURE_IMPORTANCE = {
    "Previous_Week_Sales": 83.79,
    "Previous_4_Weeks_Sales": 11.22,
    "Week": 3.32,
    "Temperature": 0.30,
    "CPI": 0.28,
    "Unemployment": 0.28,
    "Holiday_Flag": 0.23,
    "Fuel_Price": 0.21,
    "Store": 0.17,
    "Month": 0.16,
    "Year": 0.03,
}

# Default input values, also used as the "reset to defaults" state
DEFAULT_VALUES = {
    "store": 1,
    "holiday_flag": 0,
    "year": 2012,
    "month": 10,
    "week": 40,
    "temperature": 60.0,
    "fuel_price": 3.0,
    "cpi": 210.0,
    "unemployment": 8.0,
    "previous_week_sales": 1000000.0,
    "previous_4_weeks_sales": 1000000.0,
}


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Walmart Sales Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# MINIMAL, SAFE CSS (styling only — never used to render page content)
# =============================================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /*
      Fixed, branded color palette. Deliberately NOT tied to the
      Light/Dark theme toggle -- the sidebar, main area, and cards keep
      their own distinct professional colors either way, so the app
      never looks "plain black" or "plain white". The Settings menu
      toggle still works (Streamlit's own chrome responds to it), it
      just doesn't repaint these custom-styled regions.
    */

    /* ---------- Animated rising sales-themed background icons ---------- */
    .float-icon {
        position: fixed;
        bottom: -80px;
        font-weight: 800;
        z-index: 0;
        pointer-events: none;
        animation-name: rise;
        animation-timing-function: ease-in;
        animation-iteration-count: infinite;
        user-select: none;
    }
    .float-icon.dollar { color: rgba(99, 102, 241, 0.28); }
    .float-icon.arrow   { color: rgba(20, 184, 166, 0.30); }
    .float-icon.coin    { color: rgba(245, 158, 11, 0.26); }

    .float-icon.i1  { left: 5%;  font-size: 46px; animation-duration: 16s; animation-delay: 0s; }
    .float-icon.i2  { left: 16%; font-size: 30px; animation-duration: 12s; animation-delay: 2s; }
    .float-icon.i3  { left: 28%; font-size: 54px; animation-duration: 20s; animation-delay: 4s; }
    .float-icon.i4  { left: 41%; font-size: 34px; animation-duration: 14s; animation-delay: 1s; }
    .float-icon.i5  { left: 55%; font-size: 26px; animation-duration: 11s; animation-delay: 5s; }
    .float-icon.i6  { left: 66%; font-size: 48px; animation-duration: 18s; animation-delay: 3s; }
    .float-icon.i7  { left: 78%; font-size: 32px; animation-duration: 13s; animation-delay: 6s; }
    .float-icon.i8  { left: 89%; font-size: 40px; animation-duration: 17s; animation-delay: 2.5s; }
    .float-icon.i9  { left: 11%; font-size: 24px; animation-duration: 10s; animation-delay: 7s; }
    .float-icon.i10 { left: 72%; font-size: 28px; animation-duration: 15s; animation-delay: 8s; }

    @keyframes rise {
        0%   { transform: translate(0, 0) rotate(0deg); opacity: 0; }
        10%  { opacity: 0.9; }
        90%  { opacity: 0.5; }
        100% { transform: translate(30px, -110vh) rotate(8deg); opacity: 0; }
    }

    /* ---------- Main content area ---------- */
    .stApp {
        background-color: #eef1f8;
    }
    .block-container {
        position: relative;
        z-index: 1;
        max-width: 1150px;
        padding-top: 0.5rem;
        padding-bottom: 3rem;
    }

    /* Streamlit's default top toolbar has a white background by default --
       make it transparent so it blends with our page background instead
       of showing as a white strip above the content */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    .stApp, .stApp p, .stApp span, .stApp label {
        color: #1e293b;
    }
    h1 { color: #172554 !important; }
    h3, h4 { color: #1e293b !important; }

    /* Divider lines need real breathing room above and below them */
    hr {
        margin: 2rem 0 !important;
    }

    /* ---------- Sidebar (drawer) ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }

    /* Project title box -- makes it obvious this is the main app name */
    section[data-testid="stSidebar"] h1 {
        border: 1px solid rgba(129, 140, 248, 0.5) !important;
        background: rgba(129, 140, 248, 0.12);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 0.6rem;
        letter-spacing: 0.3px;
    }

    /* Section labels (Model / Performance / Technology) -- boxed pill so
       they read clearly as headings, distinct from the plain rows below them */
    section[data-testid="stSidebar"] h3 {
        display: inline-block;
        border: 1px solid rgba(129, 140, 248, 0.55) !important;
        background: rgba(129, 140, 248, 0.18);
        color: #c7d2fe !important;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.92rem;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        margin-top: 1.1rem;
        margin-bottom: 0.5rem;
    }

    /* ---------- Cards: bordered containers (native st.container(border=True)) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-radius: 16px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 3px 14px rgba(15, 23, 42, 0.06);
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.4rem;
    }

    /* ---------- KPI metric cards -- unique color per card ---------- */
    div[data-testid="stMetric"] {
        background: #ffffff;
        padding: 18px 20px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetricValue"] {
        color: #1e293b !important;
    }
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stMetric"] { border-top: 4px solid #4f46e5; }
    div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetric"] { border-top: 4px solid #0d9488; }
    div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetric"] { border-top: 4px solid #b45309; }
    div[data-testid="column"]:nth-of-type(4) div[data-testid="stMetric"] { border-top: 4px solid #be123c; }

    /* ---------- Inputs ---------- */
    div[data-testid="stNumberInputContainer"],
    div[data-baseweb="input"],
    div[data-baseweb="select"] {
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    div[data-testid="stNumberInputContainer"] > div,
    div[data-baseweb="input"] > div {
        background: transparent !important;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"]:focus-within,
    div[data-testid="stNumberInputContainer"]:focus-within {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
    }
    div[data-baseweb="input"] input {
        color: #1e293b !important;
        background: transparent !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    div[data-baseweb="select"] * {
        color: #1e293b !important;
        background: transparent !important;
    }

    /* Input labels (Store Number, Holiday Week, etc.) -- small, muted,
       uppercase caption style so they read as field labels, not body text */
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label p {
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        color: #64748b !important;
        margin-bottom: 4px !important;
    }

    /* The +/- step buttons on number inputs -- flush against the same
       bordered box so the whole control reads as one unified field */
    div[data-testid="stNumberInputContainer"] {
        display: flex;
        align-items: center;
        padding: 6px;
    }
    div[data-testid="stNumberInput"] button {
        border-radius: 6px !important;
        border: none !important;
        background: #f1f5f9 !important;
        margin: 0 2px;
    }
    div[data-testid="stNumberInput"] button:hover {
        background: #e0e7ff !important;
    }
    div[data-testid="stNumberInput"] button svg {
        fill: #4f46e5 !important;
    }

    /* Give each input its own breathing room within the section grid */
    div[data-testid="stNumberInput"], div[data-testid="stSelectbox"] {
        margin-bottom: 0.6rem;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4f46e5, #4338ca);
        color: #ffffff !important;
        border: none;
        padding: 0.6rem 1rem;
        font-size: 1.02rem;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
    }
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div {
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.08);
    }
    .stButton > button[kind="secondary"] {
        background: #ffffff;
        color: #4f46e5 !important;
        border: 1px solid #4f46e5;
    }
    .stButton > button[kind="secondary"] p,
    .stButton > button[kind="secondary"] span,
    .stButton > button[kind="secondary"] div {
        color: #4f46e5 !important;
    }
    </style>

    <div class="float-icon dollar i1">$</div>
    <div class="float-icon arrow i2">&#8593;</div>
    <div class="float-icon dollar i3">$</div>
    <div class="float-icon arrow i4">&#8593;</div>
    <div class="float-icon dollar i5">$</div>
    <div class="float-icon arrow i6">&#8593;</div>
    <div class="float-icon dollar i7">$</div>
    <div class="float-icon arrow i8">&#8593;</div>
    <div class="float-icon dollar i9">$</div>
    <div class="float-icon arrow i10">&#8593;</div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    """
    Load the trained model bundle from disk.

    Returns a tuple: (model, features, error_message)
    Only one of (model, error_message) will be meaningfully set.
    """
    try:
        model_data = joblib.load(path)
    except FileNotFoundError:
        return None, None, (
            "Model file not found. Expected it at "
            f"`{path}`. Make sure the model has been trained and saved "
            "before running this app."
        )
    except Exception as exc:  # covers corrupted / invalid joblib files
        return None, None, f"The model file could not be loaded. Details: {exc}"

    if not isinstance(model_data, dict) or "model" not in model_data or "features" not in model_data:
        return None, None, (
            "The model file was loaded but is missing the expected "
            "`model` and `features` keys. This app expects a dictionary "
            "of the form {'model': ..., 'features': [...]}."
        )

    return model_data["model"], model_data["features"], None


def validate_features(saved_features):
    """
    Confirm the model's saved feature list exactly matches what this
    app is built to send. Returns None if valid, otherwise an error string.
    """
    if list(saved_features) != EXPECTED_FEATURES:
        return (
            "The model's saved feature list does not match the features "
            "this application expects. Prediction has been disabled to "
            "avoid producing incorrect results.\n\n"
            f"Expected: {EXPECTED_FEATURES}\n\n"
            f"Found: {list(saved_features)}"
        )
    return None


# =============================================================================
# SIDEBAR
# =============================================================================

def display_sidebar():
    with st.sidebar:
        st.title("Walmart Sales Predictor")
        st.caption(
            "A machine-learning powered tool that estimates a store's "
            "weekly sales from historical sales, store details, time "
            "features, and economic indicators."
        )

        st.divider()

        st.subheader("Model")
        st.write("Random Forest Regressor")

        st.subheader("Performance")
        st.write(f"MAE: {MODEL_METRICS['MAE']}")
        st.write(f"RMSE: {MODEL_METRICS['RMSE']}")
        st.write(f"R² Score: {MODEL_METRICS['R2']}")

        st.subheader("Technology")
        st.write("Python")
        st.write("Pandas")
        st.write("Scikit-learn")
        st.write("Streamlit")

        st.divider()
        st.caption("Portfolio Project • Retail Sales Forecasting")


# =============================================================================
# HEADER + KPI CARDS
# =============================================================================

def display_header():
    components.html(
        """
        <style>
            html, body { margin: 0; padding: 0; background: transparent; }
        </style>
        <div style="
            font-family: 'Source Sans Pro', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            border: 1px solid #312e81;
            border-left: 6px solid #6366f1;
            border-radius: 16px;
            padding: 1.5rem 1.7rem;
            box-sizing: border-box;
        ">
            <h1 style="margin: 0 0 0.5rem 0; font-size: 2.2rem; font-weight: 800; color: #ffffff;">
                Walmart Weekly Sales Predictor
            </h1>
            <p style="color: #cbd5e1; margin: 0; font-size: 1rem; line-height: 1.5;">
                Machine-learning powered weekly sales prediction using historical
                sales, store information, time features and economic indicators.
            </p>
        </div>
        """,
        height=140,
    )


def display_kpi_cards():
    with st.container(border=True):
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", MODEL_METRICS["MAE"])
        col2.metric("RMSE", MODEL_METRICS["RMSE"])
        col3.metric("R² Score", MODEL_METRICS["R2"])


# =============================================================================
# INPUT COLLECTION
# =============================================================================

def initialize_input_state():
    """Seed session_state with defaults on first run so widgets are controllable."""
    for key, value in DEFAULT_VALUES.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_inputs():
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value


def section_header(text: str, color: str):
    """Render a section heading with a colored left bar and a guaranteed
    gap before the text, using inline styles so it never depends on
    Streamlit's own internal CSS specificity."""
    st.markdown(
        f"""
        <div style="
            border-left: 5px solid {color};
            padding-left: 28px;
            margin: 0.3rem 0 1rem 0;
        ">
            <span style="
                color: #1e293b;
                font-size: 1.5rem;
                font-weight: 700;
                line-height: 1.3;
            ">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def collect_inputs():
    """Render all input widgets, organized into logical sections, each as
    its own visually distinct card."""

    with st.container(border=True):
        section_header("Store Information", "#4f46e5")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.number_input("Store Number", min_value=1, max_value=45, step=1, key="store")
        with col2:
            st.selectbox(
                "Holiday Week",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                key="holiday_flag",
            )
        with col3:
            st.number_input("Year", min_value=2010, max_value=2030, step=1, key="year")

    with st.container(border=True):
        section_header("Time Information", "#0d9488")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Month", min_value=1, max_value=12, step=1, key="month")
        with col2:
            st.number_input("Week Number", min_value=1, max_value=53, step=1, key="week")

    with st.container(border=True):
        section_header("Economic Indicators", "#b45309")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.number_input("Temperature (°F)", step=0.1, key="temperature")
        with col2:
            st.number_input("Fuel Price ($)", min_value=0.0, step=0.01, key="fuel_price")
        with col3:
            st.number_input("CPI", min_value=0.0, step=0.1, key="cpi")
        with col4:
            st.number_input("Unemployment (%)", min_value=0.0, step=0.1, key="unemployment")

    with st.container(border=True):
        section_header("Historical Sales", "#be123c")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                "Previous Week Sales ($)", min_value=0.0, step=10000.0, key="previous_week_sales"
            )
        with col2:
            st.number_input(
                "Sales 4 Weeks Ago ($)", min_value=0.0, step=10000.0, key="previous_4_weeks_sales"
            )

    st.button("Reset to Defaults", on_click=reset_inputs)


def get_current_inputs():
    """Read the current widget values back out of session_state, in a dict."""
    return {key: st.session_state[key] for key in DEFAULT_VALUES}


# =============================================================================
# PREDICTION
# =============================================================================

def create_input_dataframe(values: dict, features: list) -> pd.DataFrame:
    """
    Build a single-row DataFrame in the exact column order the model
    expects, using the saved `features` list as the source of truth.
    """
    ordered_values = [
        values["store"],
        values["holiday_flag"],
        values["temperature"],
        values["fuel_price"],
        values["cpi"],
        values["unemployment"],
        values["year"],
        values["month"],
        values["week"],
        values["previous_week_sales"],
        values["previous_4_weeks_sales"],
    ]
    return pd.DataFrame([ordered_values], columns=features)


def make_prediction(model, input_df: pd.DataFrame):
    """Run inference, raising a caught exception on failure."""
    return model.predict(input_df)[0]


def display_prediction_result(prediction: float, previous_week_sales: float):
    """Render the predicted sales figure as a polished result card, with a
    trend comparison against the previous week's sales the user entered."""

    st.success("Prediction generated successfully!")

    delta = prediction - previous_week_sales
    pct_change = (delta / previous_week_sales * 100) if previous_week_sales else 0
    trend_word = "increase" if delta >= 0 else "decrease"

    with st.container(border=True):
        st.subheader("Predicted Weekly Sales")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.metric(
                label="Estimated Sales for the Week",
                value=f"${prediction:,.2f}",
                delta=f"{pct_change:+.1f}% vs previous week",
            )
        with col2:
            st.metric(
                label="Previous Week Sales",
                value=f"${previous_week_sales:,.2f}",
            )

        st.caption(
            f"This is a **{trend_word}** of **${abs(delta):,.2f}** compared to the "
            "previous week's sales you entered, based on the store, time, and "
            "economic indicators provided above."
        )


# =============================================================================
# SUPPORTING SECTIONS
# =============================================================================

def display_feature_importance():
    with st.expander("Feature Importance"):
        st.caption(
            "Relative importance of each feature as determined during "
            "model training."
        )
        importance_df = (
            pd.DataFrame(
                {
                    "Feature": list(FEATURE_IMPORTANCE.keys()),
                    "Importance (%)": list(FEATURE_IMPORTANCE.values()),
                }
            )
            .sort_values("Importance (%)", ascending=True)
            .set_index("Feature")
        )
        st.bar_chart(importance_df, horizontal=True)


def display_how_it_works():
    with st.expander("How this prediction works"):
        st.write(
            "This app loads a pre-trained Random Forest Regressor and "
            "feeds it the 11 values you provide above, in the exact order "
            "the model was trained on. The model then outputs a single "
            "estimated weekly sales figure for that store."
        )
        st.write(
            "The two strongest predictors are last week's sales and the "
            "sales from four weeks prior — together they account for "
            "roughly 95% of the model's decision-making, which is typical "
            "for short-term retail forecasting."
        )


def display_model_info(model):
    with st.expander("Model Information"):
        st.write(f"**Algorithm:** {type(model).__name__}")
        n_estimators = getattr(model, "n_estimators", None)
        if n_estimators is not None:
            st.write(f"**Number of trees:** {n_estimators}")
        st.write(f"**Input features:** {len(EXPECTED_FEATURES)}")


def display_footer():
    st.divider()
    st.caption("Walmart Weekly Sales Forecasting • Machine Learning Portfolio Project")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    display_sidebar()
    display_header()
    st.divider()
    display_kpi_cards()
    st.divider()

    model, features, load_error = load_model(MODEL_PATH)

    if load_error:
        st.error(load_error)
        st.stop()

    feature_error = validate_features(features)
    if feature_error:
        st.error(feature_error)
        st.stop()

    initialize_input_state()
    collect_inputs()

    st.divider()

    if st.button("Predict Weekly Sales", type="primary", use_container_width=True):
        values = get_current_inputs()
        input_df = create_input_dataframe(values, features)

        try:
            prediction = make_prediction(model, input_df)
        except Exception as exc:
            st.error(
                "Something went wrong while generating the prediction. "
                f"Details: {exc}"
            )
        else:
            display_prediction_result(prediction, values["previous_week_sales"])

    st.divider()
    display_feature_importance()
    display_how_it_works()
    display_model_info(model)

    display_footer()


if __name__ == "__main__":
    main()
