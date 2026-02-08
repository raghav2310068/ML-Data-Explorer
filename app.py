import streamlit as st
import pandas as pd

from core.data_loader import DataLoader
from core.overview import DataOverview
from core.eda import EDA
from core.preprocessing import Preprocessing
from core.encoding import Encoder
from core.text_processing import TextProcessor
from core.exporter import Exporter

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="ML Data Explorer", layout="wide")
st.title("📊 ML Data Explorer")
st.caption("Understand, preprocess, and analyze your data step by step")

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False

# --------------------------------------------------
# HELPER: SHOW UPDATED DATASET
# --------------------------------------------------
def show_updated_dataset(title="Updated Dataset Preview"):
    with st.expander(title, expanded=True):
        st.caption("This is the current state of your dataset after the last operation.")
        st.dataframe(st.session_state.df, use_container_width=True)

# --------------------------------------------------
# SIDEBAR – DATASET UPLOAD
# --------------------------------------------------
st.sidebar.header("⚙️ Input Panel")

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload CSV Dataset",
    type=["csv"],
    key="file_uploader"
)

# Load dataset once
if uploaded_file and not st.session_state.dataset_loaded:
    loader = DataLoader()
    st.session_state.df = loader.load_csv(uploaded_file)
    st.session_state.dataset_loaded = True
    st.sidebar.success("Dataset loaded successfully")

# Explicit reload
if uploaded_file and st.session_state.dataset_loaded:
    if st.sidebar.button("🔄 Reload Dataset", key="reload_dataset"):
        loader = DataLoader()
        st.session_state.df = loader.load_csv(uploaded_file)
        st.sidebar.success("Dataset reloaded successfully")

# --------------------------------------------------
# MAIN GUARD
# --------------------------------------------------
if st.session_state.df is None:
    st.info("👈 Upload a dataset to begin")
    st.stop()

tabs = st.tabs([
    "📄 Overview",
    "📊 EDA",
    "🩹 Cleaning & Columns",
    "🔄 Encoding",
    "🧠 Text Processing",
    "💾 Export"
])

# --------------------------------------------------
# TAB 1: OVERVIEW
# --------------------------------------------------
with tabs[0]:
    st.subheader("Dataset Overview")

    df_current = st.session_state.df
    overview = DataOverview(df_current)
    info = overview.basic_info()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", info["rows"])
    c2.metric("Columns", info["columns"])
    c3.metric("Missing Values", info["missing"])

    st.markdown("### Column Summary")
    st.dataframe(overview.column_summary())

    st.markdown("### Statistical Summary")
    st.dataframe(overview.stats())

    show_updated_dataset("Current Dataset")

# --------------------------------------------------
# TAB 2: EDA
# --------------------------------------------------
with tabs[1]:
    st.subheader("Exploratory Data Analysis")
    st.info("Analyze univariate, bivariate, and multivariate relationships.")

    df_current = st.session_state.df
    eda = EDA(df_current)

    numeric_cols = df_current.select_dtypes(include="number").columns.tolist()
    categorical_cols = df_current.select_dtypes(include="object").columns.tolist()

    # ---------- UNIVARIATE ----------
    st.markdown("### Univariate Analysis")
    uni_col = st.selectbox("Select column", df_current.columns, key="uni_col")

    if uni_col in numeric_cols:
        st.pyplot(eda.univariate_numeric(uni_col))
    else:
        st.pyplot(eda.univariate_categorical(uni_col))

    st.divider()

    # ---------- BIVARIATE ----------
    st.markdown("### Bivariate Analysis")
    bivar_type = st.selectbox(
        "Relationship type",
        ["Numeric vs Numeric", "Categorical vs Numeric"],
        key="bivar_type"
    )

    if bivar_type == "Numeric vs Numeric" and len(numeric_cols) >= 2:
        x = st.selectbox("X-axis", numeric_cols, key="bivar_x")
        y = st.selectbox("Y-axis", numeric_cols, key="bivar_y")

        plot_type = st.selectbox(
            "Plot type",
            ["Scatter", "Line", "Regression"],
            key="bivar_plot_num"
        )

        st.pyplot(eda.numeric_vs_numeric_plot(x, y, plot_type))

    if bivar_type == "Categorical vs Numeric" and categorical_cols and numeric_cols:
        cat = st.selectbox("Categorical variable", categorical_cols, key="bivar_cat")
        num = st.selectbox("Numeric variable", numeric_cols, key="bivar_num")

        plot_type = st.selectbox(
            "Plot type",
            ["Box", "Violin", "Bar (Mean)"],
            key="bivar_plot_cat"
        )

        st.pyplot(eda.categorical_vs_numeric_plot(cat, num, plot_type))

    st.divider()
    st.markdown("### Multivariate Analysis")
    st.pyplot(eda.correlation_heatmap())

# --------------------------------------------------
# TAB 3: CLEANING, ROW & COLUMN REMOVAL
# --------------------------------------------------
with tabs[2]:
    st.subheader("Data Cleaning & Column Management")

    df_current = st.session_state.df
    prep = Preprocessing(df_current)

    # ----- Missing Values -----
    st.markdown("### 🩹 Fill Missing Values")
    na_col = st.selectbox("Select column", df_current.columns, key="na_col")

    if pd.api.types.is_numeric_dtype(df_current[na_col]):
        na_method = st.selectbox("Method", ["mean", "median", "mode"], key="na_method_num")
    else:
        na_method = st.selectbox("Method", ["mode"], key="na_method_cat")

    if st.button("Apply Missing Value Strategy", key="na_apply"):
        st.session_state.df = prep.fill_na(na_col, na_method)
        st.success("Missing values handled")
        show_updated_dataset()

    st.divider()

    # ----- Remove Rows with Null Values -----
    st.markdown("### 🧹 Remove Rows with Missing Values")
    st.info("Permanently remove rows based on missing-value conditions.")

    drop_strategy = st.radio(
        "Choose removal strategy",
        [
            "Drop rows with ANY null values",
            "Drop rows with ALL null values",
            "Drop rows with nulls in selected columns"
        ],
        key="drop_row_strategy"
    )

    subset_cols = None
    if drop_strategy == "Drop rows with nulls in selected columns":
        subset_cols = st.multiselect(
            "Select columns",
            df_current.columns.tolist(),
            key="drop_row_subset"
        )

    confirm_rows = st.checkbox(
        "I understand that rows will be permanently removed",
        key="drop_rows_confirm"
    )

    if st.button("Remove Rows", key="drop_rows_apply"):
        if not confirm_rows:
            st.warning("Please confirm before removing rows.")
        elif drop_strategy == "Drop rows with nulls in selected columns" and not subset_cols:
            st.warning("Please select at least one column.")
        else:
            before = len(df_current)

            if drop_strategy == "Drop rows with ANY null values":
                st.session_state.df = df_current.dropna(how="any")
            elif drop_strategy == "Drop rows with ALL null values":
                st.session_state.df = df_current.dropna(how="all")
            else:
                st.session_state.df = df_current.dropna(subset=subset_cols)

            after = len(st.session_state.df)
            st.success(f"Removed {before - after} rows")
            show_updated_dataset()

    st.divider()

    # ----- Drop Multiple Columns -----
    st.markdown("### 🗑 Drop Columns")
    drop_cols = st.multiselect(
        "Select columns to drop",
        df_current.columns.tolist(),
        key="drop_cols_multi"
    )

    confirm_cols = st.checkbox(
        "I understand these columns will be permanently removed",
        key="drop_cols_confirm"
    )

    if st.button("Drop Selected Columns", key="drop_apply"):
        if not drop_cols:
            st.warning("Select at least one column.")
        elif not confirm_cols:
            st.warning("Please confirm column deletion.")
        else:
            st.session_state.df = df_current.drop(columns=drop_cols)
            st.success("Columns dropped successfully")
            show_updated_dataset()

# --------------------------------------------------
# TAB 4: ENCODING (MULTI-COLUMN)
# --------------------------------------------------
with tabs[3]:
    st.subheader("Categorical Encoding")
    st.info("Encode multiple categorical columns at once.")

    df_current = st.session_state.df
    enc = Encoder(df_current)

    cat_cols = df_current.select_dtypes(include="object").columns.tolist()

    if not cat_cols:
        st.warning("No categorical columns available.")
    else:
        selected_cols = st.multiselect(
            "Select categorical columns",
            cat_cols,
            key="enc_multi_cols"
        )

        method = st.radio(
            "Encoding method",
            ["Label Encoding", "One-Hot Encoding"],
            key="enc_method"
        )

        if st.button("Apply Encoding", key="enc_apply"):
            if not selected_cols:
                st.warning("Select at least one column.")
            else:
                for col in selected_cols:
                    if method == "Label Encoding":
                        st.session_state.df = enc.label_encode(col)
                    else:
                        st.session_state.df = enc.one_hot_encode(col)

                st.success("Encoding applied")
                show_updated_dataset()

# --------------------------------------------------
# TAB 5: TEXT PROCESSING
# --------------------------------------------------
with tabs[4]:
    st.subheader("Advanced Text Processing")

    df_current = st.session_state.df
    tp = TextProcessor(df_current)

    text_cols = df_current.select_dtypes(include="object").columns.tolist()

    if not text_cols:
        st.warning("No text columns available.")
    else:
        col = st.selectbox("Select text column", text_cols, key="text_col")

        if st.button("Clean Text", key="text_clean"):
            st.session_state.df = tp.basic_clean(col)
            st.success("Text cleaned")
            show_updated_dataset()

        if st.button("Add Length Features", key="text_len"):
            st.session_state.df = tp.add_length_features(col)
            st.success("Text features added")
            show_updated_dataset()

        if st.button("Generate TF-IDF Features", key="text_tfidf"):
            st.session_state.df = tp.tfidf_features(col)
            st.success("TF-IDF features generated")
            show_updated_dataset()

# --------------------------------------------------
# TAB 6: EXPORT
# --------------------------------------------------
with tabs[5]:
    st.subheader("Export Dataset")

    exporter = Exporter(st.session_state.df)

    st.download_button(
        "⬇ Download Modified Dataset (CSV)",
        exporter.to_csv(),
        "modified_dataset.csv",
        "text/csv",
        key="download_csv"
    )
