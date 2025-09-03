# app.py
import io
import os
import math
import numpy as np
import pandas as pd
import streamlit as st

# ML imports (logic unchanged)
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# optional plotly (the dashboard uses plotly when available)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ---------------------------------------------
# Page config + Combined CSS (landing + dashboard styles)
# ---------------------------------------------
st.set_page_config(
    page_title="Car Price Prediction", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

COMBINED_CSS = """
<style>
/* Basic container spacing */
.block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }

/* Landing */
.landing-container {
    background: linear-gradient(135deg, #2E86C1 0%, #6C3483 100%);
    color: white;
    padding: 5rem 3rem;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}
.landing-title { 
    font-size: 3rem; 
    font-weight: 800; 
    margin-bottom: 0.8rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}
.landing-subtitle { 
    font-size: 1.3rem; 
    margin-bottom: 2rem; 
    opacity: 0.95;
    font-weight: 300;
}

/* Get Started Button */
.stButton>button {
    background: linear-gradient(135deg, #2E86C1 0%, #6C3483 100%);
    color: white;
    border: none;
    padding: 0.8rem 2rem;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(46, 134, 193, 0.4);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(46, 134, 193, 0.6);
}

/* KPI cards from dashboard */
.kpi {
    background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(16,185,129,0.15));
    border-radius: 16px;
    padding: 18px 16px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}

.kpi:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 25px rgba(0,0,0,0.12);
}

.kpi h3 { 
    margin: 0; 
    font-size: 0.95rem; 
    opacity: 0.85; 
    font-weight: 500;
}
.kpi p { 
    margin: 8px 0 0 0; 
    font-size: 1.5rem; 
    font-weight: 700; 
}

/* Section card look */
.section-card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px 24px;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 15px 35px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}

/* Tab-like top radio styling */
div[role="radiogroup"] {
    background: rgba(255,255,255,0.7);
    border-radius: 12px;
    padding: 8px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

div[role="radiogroup"] > label {
    border: none;
    padding: 0.7rem 1.5rem;
    margin-right: 0.5rem;
    border-radius: 10px;
    background: transparent;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.3s ease;
}
div[role="radiogroup"] > label:hover {
    background: rgba(46, 134, 193, 0.1);
    color: #2E86C1;
}
div[role="radiogroup"] > label[data-checked="true"] {
    background: #2E86C1;
    color: white;
    box-shadow: 0 4px 10px rgba(46, 134, 193, 0.3);
}

/* Sidebar styling - تم تغيير لون الخلفية إلى أزرق فاتح */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2E86C1 0%, #1B4F72 100%);
    color: white;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] .stButton>button {
    background: rgba(255, 255, 255, 0.2);
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.3);
    width: 100%;
    margin-top: 1rem;
}

section[data-testid="stSidebar"] .stButton>button:hover {
    background: rgba(255, 255, 255, 0.3);
}

/* Dataframe styling */
.dataframe {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.7) 0%, rgba(248,249,250,0.7) 100%);
    border-radius: 12px;
    padding: 15px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* Dark mode tweaks */
@media (prefers-color-scheme: dark) {
  .section-card { 
      background: rgba(24,24,27,0.5); 
      border-color: rgba(255,255,255,0.1); 
  }
  .kpi { 
      background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(16,185,129,0.2)); 
  }
  div[role="radiogroup"] {
      background: rgba(24,24,27,0.5);
  }
  div[role="radiogroup"] > label { 
      background: transparent; 
      color: #fff; 
  }
  div[role="radiogroup"] > label:hover { 
      background: rgba(46, 134, 193, 0.2); 
      color: #2E86C1; 
  }
  div[role="radiogroup"] > label[data-checked="true"] { 
      background: #2E86C1; 
      color: white; 
  }
  
  [data-testid="stMetric"] {
      background: rgba(24,24,27,0.5);
      border-color: rgba(255,255,255,0.1);
  }
}

/* Hide default menu/footer for a cleaner look */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Hide navigation tabs initially */
.hidden-tabs {
    display: none;
}
</style>
"""
st.markdown(COMBINED_CSS, unsafe_allow_html=True)

# ---------------------------------------------
# Session state defaults
# ---------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "df_clean" not in st.session_state:
    st.session_state["df_clean"] = None
if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = None
if "target_col" not in st.session_state:
    st.session_state["target_col"] = "Price"
if "selected_features" not in st.session_state:
    st.session_state["selected_features"] = None
if "model_name" not in st.session_state:
    st.session_state["model_name"] = "RandomForestRegressor"
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "🏠 Home"
if "app_started" not in st.session_state:
    st.session_state["app_started"] = False
if "selected_feature_insight" not in st.session_state:
    st.session_state["selected_feature_insight"] = None

DEFAULT_FILE = "cars_with_estimated_prices.csv"

# ---------------------------------------------
# Utilities: data loader (cached)
# ---------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(file: io.BytesIO | str) -> pd.DataFrame:
    if isinstance(file, str):
        return pd.read_csv(file)
    else:
        file.seek(0)
        return pd.read_csv(file)

def human_readable_bytes(n):
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

# ---------------------------------------------
# Cleaning & Feature Engineering (logic preserved)
# ---------------------------------------------
def to_numeric_clean(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "-": np.nan})
        .astype(float)
    )

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    engine_cols = [c for c in df.columns if "engine volume" in c.lower()]
    if engine_cols:
        df["Engine volume_num"] = to_numeric_clean(df[engine_cols[0]])
    mileage_cols = [c for c in df.columns if "mileage" in c.lower()]
    if mileage_cols:
        df["Mileage_num"] = to_numeric_clean(df[mileage_cols[0]])
    doors_cols = [c for c in df.columns if "door" in c.lower()]
    if doors_cols:
        df["Doors_num"] = (
            df[doors_cols[0]].astype(str).str.extract(r"(\d+)", expand=False).replace("", np.nan).astype(float)
        )
    levy_cols = [c for c in df.columns if c.lower() == "levy"]
    if levy_cols:
        df["Levy_num"] = to_numeric_clean(df[levy_cols[0]])
    year_cols = [c for c in df.columns if c.lower() in ["prod. year", "prod year", "prod_year", "production year", "year"]]
    if "Levy_num" in df.columns and year_cols:
        py = year_cols[0]
        if df[py].isna().all():
            df["Levy_num"] = df["Levy_num"].fillna(df["Levy_num"].median())
        else:
            df["Levy_num"] = df.groupby(py)["Levy_num"].transform(lambda x: x.fillna(x.median()))
    if year_cols:
        py = year_cols[0]
        df["Car_Age"] = (pd.Timestamp.today().year - pd.to_numeric(df[py], errors="coerce")).astype(float)
    for col in ["Engine volume", "Doors", "Levy", "Mileage"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    for col in ["Mileage_num", "Engine volume_num", "Doors_num"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    tgt = st.session_state["target_col"]
    if tgt in df.columns:
        df = df[~df[tgt].isna()].copy()
    return df

# ---------------------------------------------
# Pipeline & training (logic preserved)
# ---------------------------------------------
def build_pipeline(df: pd.DataFrame, target_col: str) -> Tuple[Pipeline, List[str], List[str]]:
    all_cols = [c for c in df.columns if c != target_col]
    num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in all_cols if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )
    model_name = st.session_state.get("model_name", "RandomForestRegressor")
    if model_name == "LinearRegression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    return pipe, num_cols, cat_cols

def train_model(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pipe, num_cols, cat_cols = build_pipeline(df, target_col)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(math.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
    }
    return pipe, metrics

# ---------------------------------------------
# Page: Home (Landing)
# ---------------------------------------------
def page_home():
    st.markdown(
        "<div class='landing-container'>"
        "<div class='landing-title'>🚗 Car Price Prediction</div>"
        "<div class='landing-subtitle'>Discover insights, preprocess data, train models and predict car prices with a clean, interactive UI.</div>"
        "</div>",
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 0.8, 1])
    with col2:
        if st.button("Get Started →", key="start_button", use_container_width=True):
            st.session_state["app_started"] = True
            st.session_state["active_tab"] = "📂 Data"
            st.rerun()
    
    st.markdown("---")
    
    # Features section
    st.subheader("✨ Key Features")
    features_col1, features_col2, features_col3 = st.columns(3)
    
    with features_col1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; border-radius: 12px; background: rgba(46, 134, 193, 0.1);'>
            <h3>📊 Data Exploration</h3>
            <p>Interactive visualizations and detailed analysis of your car dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
    with features_col2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; border-radius: 12px; background: rgba(46, 134, 193, 0.1);'>
            <h3>🤖 ML Models</h3>
            <p>Train and compare multiple machine learning models for accurate predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with features_col3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem; border-radius: 12px; background: rgba(46, 134, 193, 0.1);'>
            <h3>🔮 Price Prediction</h3>
            <p>Get instant price estimates based on car specifications and features</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------
# Data tab with internal tabs
# ---------------------------------------------
def data_page_dashboard(df: pd.DataFrame):
    # Header & KPIs
    title_col, kpi_col = st.columns([0.72, 0.28], gap="large")
    with title_col:
        st.title("📊 Data Overview")
        st.caption("Explore your dataset with interactive visualizations and detailed insights")
    rows, cols = df.shape
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    mem = df.memory_usage(deep=True).sum()
    miss_total = int(df.isna().sum().sum())
    miss_pct = (miss_total / (rows * cols) * 100) if rows * cols > 0 else 0
    dups = int(df.duplicated().sum())
    with kpi_col:
        st.markdown('<div class="kpi"><h3>Number of Rows</h3><p>{:,}</p></div>'.format(rows), unsafe_allow_html=True)
        st.markdown('<div class="kpi" style="margin-top:12px;"><h3>Number of Columns</h3><p>{:,}</p></div>'.format(cols), unsafe_allow_html=True)
        st.markdown('<div class="kpi" style="margin-top:12px;"><h3>Memory Size</h3><p>{}</p></div>'.format(human_readable_bytes(mem)), unsafe_allow_html=True)
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Numerical Columns", len(num_cols))
    with c2: st.metric("Text Columns", len(cat_cols))
    with c3: st.metric("Missing Values", f"{miss_total:,}", f"{miss_pct:.1f}%")
    with c4: st.metric("Duplicate Rows", f"{dups:,}")

    # Data summary section
    st.subheader("📋 Data Summary")
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown("**Sample Data (First 5 Rows)**")
        st.dataframe(df.head(), use_container_width=True)
    
    with summary_col2:
        st.markdown("**Data Types**")
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.notnull().sum().values
        })
        st.dataframe(dtypes_df, use_container_width=True, height=200)
    
    # Quick insights section
    st.subheader("🔍 Quick Insights")
    
    # Numeric columns statistics
    if len(num_cols) > 0:
        st.markdown("**Numeric Columns Statistics**")
        numeric_stats = df[num_cols].describe().T
        st.dataframe(numeric_stats.style.format("{:.2f}"), use_container_width=True)
    
    # Categorical columns statistics
    if len(cat_cols) > 0:
        st.markdown("**Categorical Columns Statistics**")
        for col in cat_cols[1:4]:  # Show only first 3 categorical columns
            with st.expander(f"Value counts for {col}"):
                value_counts = df[col].value_counts().head(10)
                st.dataframe(value_counts, use_container_width=True)
    
    # Internal tabs for different views
    internal_tabs = st.tabs(["📊 Overview", "📖 EDA", "📈 Visualization", "💡 Insights"])

    with internal_tabs[0]:
        st.subheader("Quick Visual Insights")
        left, right = st.columns([0.6, 0.4], gap="large")
        with left:
            na_counts = df.isna().sum().sort_values(ascending=False)
            na_counts = na_counts[na_counts > 0].head(15)
            if not na_counts.empty:
                if PLOTLY_OK:
                    fig = px.bar(na_counts.sort_values(), orientation="h", title="Top Columns with Missing Values", labels={"value":"Number of Missing Values", "index":"Column"})
                    fig.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(na_counts)
            else:
                st.info("No missing values 👍")
        with right:
            if len(cat_cols) > 0:
                cat_choice = st.selectbox("Choose a text column to display the most frequent values:", cat_cols, index=0)
                vc = df[cat_choice].astype(str).value_counts().head(15)
                if PLOTLY_OK:
                    fig2 = px.bar(vc.sort_values(), orientation="h", title=f"Most Frequent Values in {cat_choice}", labels={"value":"Frequency", "index":cat_choice})
                    fig2.update_layout(height=420, margin=dict(l=10,r=10,t=60,b=10))
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.bar_chart(vc)
            else:
                st.warning("No text columns in the data.")
        
        st.markdown("---")
        st.subheader("Correlation Heatmap (Numeric Columns)")
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            if PLOTLY_OK:
                # تحسين ألوان الهيت ماب
                fig = px.imshow(corr, 
                               text_auto=True, 
                               color_continuous_scale="RdBu_r",  # تغيير إلى ألوان أفضل
                               aspect="auto",
                               title="Correlation Heatmap")
                fig.update_layout(height=600, margin=dict(l=10,r=10,t=60,b=10))
                st.plotly_chart(fig, use_container_width=True)
                
                # إظهار أعلى العلاقات الإيجابية والسلبية
                corr_matrix = corr.unstack()
                sorted_corr = corr_matrix.sort_values(ascending=False, key=abs)
                # إزالة التكرارات والقيم القطرية (العلاقة مع النفس = 1)
                unique_corr = sorted_corr[sorted_corr.index.map(lambda x: x[0] != x[1])]
                unique_corr = unique_corr[~unique_corr.index.duplicated()]
                
                top_corr = unique_corr.head(10)
                st.write("**Top Correlations:**")
                for idx, val in top_corr.items():
                    st.write(f"{idx[0]} - {idx[1]}: {val:.3f}")
                    
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap="RdBu_r", fmt=".2f", ax=ax, center=0)
                st.pyplot(fig)
        else:
            st.info("Not enough numeric columns to create a correlation heatmap.")


    with internal_tabs[1]: # EDA Tab
        st.subheader("Exploratory Data Analysis (Detailed)")
        st.write("DataFrame Info:")
        buf = io.StringIO()
        df.info(buf=buf)
        st.text(buf.getvalue())
        st.markdown("---")
        st.write("Statistical Summary (Numeric Columns):")
        st.dataframe(df.describe(include=[np.number]).T)
        st.markdown("---")
        st.write("Missing values by column:")
        null_summary = pd.DataFrame({
            "Null Count": df.isnull().sum(),
            "Null %": (df.isnull().sum() / len(df) * 100).round(2),
            "Dtype": df.dtypes
        })
        st.dataframe(null_summary.sort_values(by="Null Count", ascending=False))
        st.markdown("---")
        st.write("Column preview / top values")
        col = st.selectbox("Select a column to preview", options=df.columns.tolist())
        if col:
            st.dataframe(df[col].value_counts(dropna=False).head(50).to_frame("count"))

    with internal_tabs[2]:
        # قسم التصورات العامة
        st.markdown("### 📊 General Visualizations")
        
        col1, col2, col3 = st.columns([0.32, 0.32, 0.36])
        with col1:
            x_col = st.selectbox("Horizontal (X)", df.columns, index=0, key="x_col_selector")
        with col2:
            y_col = st.selectbox("Vertical (Y)", df.columns, index=min(1, len(df.columns)-1), key="y_col_selector")
        with col3:
            chart_type = st.selectbox("Chart Type", ["Scatter", "Box", "Bar", "Histogram", "Violin", "Density Heatmap"], key="chart_type_selector")
        
        try:
            if PLOTLY_OK:
                if chart_type == "Scatter":
                    fig = px.scatter(df, x=x_col, y=y_col, opacity=0.7, trendline="ols")
                elif chart_type == "Box":
                    fig = px.box(df, x=x_col, y=y_col, points="suspectedoutliers")
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_col, y=y_col)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_col, nbins=40)
                elif chart_type == "Violin":
                    fig = px.violin(df, x=x_col, y=y_col, box=True, points="all")
                elif chart_type == "Density Heatmap":
                    fig = px.density_heatmap(df, x=x_col, y=y_col)
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, opacity=0.7)
                fig.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Plotly not available — fallback to Matplotlib/Streamlit charts.")
                fig, ax = plt.subplots(figsize=(8,5))
                if chart_type == "Histogram":
                    ax.hist(df[x_col].dropna().values, bins=40)
                else:
                    ax.scatter(df[x_col].dropna().values[:2000], df[y_col].dropna().values[:2000], s=10)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed to render chart: {e}")

    with internal_tabs[3]: # Insights Tab
        st.subheader("💡 Data Insights")
        
        # رؤى تلقائية عن البيانات
        st.markdown("### Automated Data Insights")
        
        insights_container = st.container()
        
        with insights_container:
            # رؤى حول القيم المفقودة
            if miss_total > 0:
                missing_cols = null_summary[null_summary["Null Count"] > 0]
                top_missing = missing_cols.iloc[0]
                st.warning(f"**Missing Values Alert**: {miss_total} missing values found. Highest in '{top_missing.name}' column ({top_missing['Null %']}%). Consider imputation or removal.")
            
            # رؤى حول التكرارات
            if dups > 0:
                st.warning(f"**Duplicates Found**: {dups} duplicate rows detected. Removing duplicates can improve model performance.")
            
            # رؤى حول الأعمدة الرقمية
            if len(num_cols) > 0:
                st.info(f"**Numeric Analysis**: {len(num_cols)} numeric columns available for analysis. Consider feature scaling for better model performance.")
            
            # رؤى حول الأعمدة النصية
            if len(cat_cols) > 0:
                st.info(f"**Categorical Analysis**: {len(cat_cols)} categorical columns detected. These will need encoding before model training.")
            
            # رؤى حول التوزيعات
            if len(num_cols) > 0:
                # البحث عن توزيعات غير طبيعية
                skewed_cols = []
                for col in num_cols:
                    try:
                        skewness = df[col].skew()
                        if abs(skewness) > 1:  # توزيع غير طبيعي
                            skewed_cols.append((col, skewness))
                    except:
                        continue
                
                if skewed_cols:
                    skewed_cols.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_skewed = skewed_cols[0]
                    st.warning(f"**Skewed Distribution**: '{top_skewed[0]}' has high skewness ({top_skewed[1]:.2f}). Consider transformation (log, sqrt) for better model performance.")
            
            # إذا لم توجد رؤى مهمة
            if miss_total == 0 and dups == 0 and not skewed_cols:
                st.success("**Data Quality**: Your data appears to be clean with no major issues detected.")
        
        # توصيات للتحليل
        st.markdown("### 📋 Analysis Recommendations")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.write("**Data Preparation:**")
            st.write("1. Handle missing values if any exist")
            st.write("2. Remove duplicate rows")
            st.write("3. Encode categorical variables")
            st.write("4. Scale numeric features")
        
        with rec_col2:
            st.write("**Modeling Strategy:**")
            st.write("1. Start with Random Forest for baseline")
            st.write("2. Try Linear Regression for interpretability")
            st.write("3. Evaluate using multiple metrics")
            st.write("4. Consider feature engineering")
        
        # إذا كان هناك عمود سعر، نعرض توصيات إضافية
        if any('price' in col.lower() for col in df.columns):
            st.success("**Price column detected!** You can proceed to train prediction models in the Train tab.")

# ---------------------------------------------
# Page: Preprocess, Train, Predict (unchanged logic)
# ---------------------------------------------
def page_preprocess():
    st.header("🧹 Data Preprocessing")
    st.markdown("Clean and prepare your data for machine learning modeling")
    
    df = st.session_state.get("raw_df")
    if df is None:
        st.info("Please load data first from the Data tab.")
        return
    
    with st.expander("🔍 Preview Raw Data"):
        st.dataframe(df.head())
    
    st.write("Running cleaning & feature engineering steps (engine volume, mileage, doors, Levy_num, etc.).")
    
    if st.button("Run Cleaning / Engineering", use_container_width=True):
        with st.spinner("Running cleaning & feature engineering..."):
            dfc = clean_and_engineer(df)
            st.session_state["df_clean"] = dfc
        st.success(f"Done — cleaned data shape: {st.session_state['df_clean'].shape}")
        
        with st.expander("🔍 Preview Cleaned Data"):
            st.dataframe(st.session_state["df_clean"].head(50))

def page_train():
    st.header("🤖 Model Training")
    st.markdown("Train machine learning models to predict car prices")
    
    dfc = st.session_state.get("df_clean")
    if dfc is None:
        st.info("Please run Preprocess first.")
        return
        
    # choose target
    numeric_cols = [c for c in dfc.columns if pd.api.types.is_numeric_dtype(dfc[c])]
    if not numeric_cols:
        st.error("No numeric columns found (can't select target).")
        return
        
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select target column", options=numeric_cols, index=0)
            st.session_state["target_col"] = target_col
            
        with col2:
            model_name = st.selectbox("Select model", ["RandomForestRegressor", "LinearRegression"], index=0)
            st.session_state["model_name"] = model_name
            
    default_feats = [c for c in dfc.columns if c != target_col]
    selected_feats = st.multiselect("Select features to include:", options=default_feats, default=default_feats)
    st.session_state["selected_features"] = selected_feats
    
    test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    
    if st.button("Train Model", use_container_width=True):
        with st.spinner("Training model..."):
            train_df = dfc[selected_feats + [target_col]]
            pipe, metrics = train_model(train_df, target_col=target_col, test_size=test_size)
        st.session_state["pipeline"] = pipe
        
        st.success("✅ Training complete!")
        
        # Display metrics in a nice way
        st.subheader("📈 Model Performance")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Mean Absolute Error", f"{metrics['MAE']:.2f}")
        with m2:
            st.metric("Root Mean Squared Error", f"{metrics['RMSE']:.2f}")
        with m3:
            st.metric("R² Score", f"{metrics['R2']:.4f}")

def page_predict():
    st.header("🔮 Price Prediction")
    st.markdown("Predict car prices based on the trained model")
    
    pipe = st.session_state.get("pipeline")
    dfc = st.session_state.get("df_clean")
    features = st.session_state.get("selected_features")
    target_col = st.session_state.get("target_col")
    
    if pipe is None or dfc is None or features is None or target_col is None:
        st.info("Please train a model first.")
        return
        
    feature_cols = [c for c in features if c != target_col]
    st.write("Enter car specifications to predict price:")
    
    user_values = {}
    n_cols = 3
    grid = [st.columns(n_cols) for _ in range(math.ceil(len(feature_cols)/n_cols))]
    
    for idx, colname in enumerate(feature_cols):
        col_widget = grid[idx // n_cols][idx % n_cols]
        series = dfc[colname]
        with col_widget:
            if pd.api.types.is_numeric_dtype(series):
                val = float(series.dropna().median()) if series.dropna().size else 0.0
                user_values[colname] = st.number_input(colname, value=val, step=1.0)
            else:
                options = series.dropna().unique().tolist() or ["N/A"]
                user_values[colname] = st.selectbox(colname, options=options, index=0)
                
    if st.button("Predict Price", use_container_width=True):
        Xnew = pd.DataFrame([user_values])
        try:
            pred = float(pipe.predict(Xnew)[0])
            st.success(f"💰 Predicted Price: **{pred:,.2f}**")
            
            # Show confidence indicator based on model type
            if st.session_state["model_name"] == "RandomForestRegressor":
                st.info("ℹ️ Random Forest models typically provide more accurate predictions for complex datasets")
            else:
                st.info("ℹ️ Linear Regression models work best when relationships between features and price are linear")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}. Please ensure all features are selected and valid.")

# ---------------------------------------------
# Main App Logic
# ---------------------------------------------

# إخفاء التبويبات إذا لم يبدأ التطبيق بعد
if not st.session_state.get("app_started", False):
    page_home()
else:
    # Top Navigation (tab-like horizontal radio)
    TOP_TABS = ["🏠 Home", "📂 Data", "🧹 Preprocess", "🤖 Train", "🔮 Predict"]
    selected = st.radio("Navigation", options=TOP_TABS, index=TOP_TABS.index(st.session_state["active_tab"]), horizontal=True, label_visibility="visible")
    if selected != st.session_state["active_tab"]:
        st.session_state["active_tab"] = selected

    # Handle data loading in the sidebar, but show content in the main page
    if st.session_state["active_tab"] == "📂 Data":
        with st.sidebar:
            st.header("📤 Data Loading")
            st.caption("Upload a CSV file or use the provided sample.")
            uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
            
            # جعل العينة غير مفعلة تلقائياً
            use_sample = st.toggle("Use sample cars_with_estimated_prices.csv", value=False)
            
            sample_path = DEFAULT_FILE
        
        df = None
        if uploaded is not None:
            try:
                df = load_data(uploaded)
                st.session_state["raw_df"] = df.copy()
                st.success("✅ File loaded successfully.")
            except Exception as e:
                st.error(f"❌ Error reading the file: {e}")
        elif use_sample and os.path.exists(sample_path):
            try:
                df = load_data(sample_path)
                st.session_state["raw_df"] = df.copy()
                st.info(f"ℹ️ Using sample file: {sample_path}")
            except Exception as e:
                st.error(f"❌ Could not read the sample: {e}")
        else:
            st.session_state["raw_df"] = None

    # Render selected top tab
    if st.session_state["active_tab"] == "🏠 Home":
        page_home()
    elif st.session_state["active_tab"] == "📂 Data":
        if st.session_state["raw_df"] is not None:
            data_page_dashboard(st.session_state["raw_df"])
        else:
            st.info("Please upload a CSV file from the sidebar or enable the sample option.")
    elif st.session_state["active_tab"] == "🧹 Preprocess":
        page_preprocess()
    elif st.session_state["active_tab"] == "🤖 Train":
        page_train()
    elif st.session_state["active_tab"] == "🔮 Predict":
        page_predict()