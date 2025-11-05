# ======================================================================
# 🟩 ADVANCED TIME SERIES ANALYZER – Single-file Streamlit App (app.py)
# ======================================================================
# Paste this file as `app.py`. Requires the usual packages (see README or
# requirements.txt): streamlit, pandas, numpy, matplotlib, seaborn, plotly,
# statsmodels, openpyxl.
# Run locally: `streamlit run app.py`
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ======================================================================
# 🟩 BASIC CONFIGURATION
# ======================================================================
st.set_page_config(layout="wide", page_title="Advanced Time Series Analyzer")
st.title("📊 Advanced Time Series Analysis & Visualization")

# ======================================================================
# 🟩 SECTION 1: DATA INPUT & PREPARATION
# ======================================================================
st.sidebar.header("1. Data Input")
use_sample = st.sidebar.checkbox("Use sample dataset (provided)", value=True)
uploaded_file = None
if not use_sample:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if use_sample:
    # The sample CSV file should be in the same folder as app.py.
    # If you deployed to Streamlit Cloud, upload sample_time_series.csv to the repo.
    try:
        df = pd.read_csv("sample_time_series.csv", parse_dates=["date"])
    except Exception:
        # If sample isn't present, create a minimal fallback sample
        dates = pd.date_range(start="2023-01-01", periods=365)
        df = pd.DataFrame({
            "date": dates,
            "series_a": np.random.randn(len(dates)).cumsum() + 10,
            "series_b": np.random.randn(len(dates)).cumsum() + 20
        })
else:
    if uploaded_file is None:
        st.info("Please upload a CSV/Excel file or enable 'Use sample dataset' in the sidebar.")
        st.stop()
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()

# 🟩 DATETIME COLUMN DETECTION
st.sidebar.header("2. Datetime Settings")
possible_dt = [c for c in df.columns if "date" in c.lower() or np.issubdtype(df[c].dtype, np.datetime64)]
if possible_dt:
    dt_col = st.sidebar.selectbox("Select datetime column", options=possible_dt, index=0)
else:
    dt_col = st.sidebar.selectbox("Select datetime column", options=df.columns)
    try:
        df[dt_col] = pd.to_datetime(df[dt_col])
    except Exception:
        st.error("Could not parse the chosen datetime column. Ensure it contains ISO-like dates.")
        st.stop()

df[dt_col] = pd.to_datetime(df[dt_col])
df = df.sort_values(dt_col).reset_index(drop=True)
df.set_index(dt_col, inplace=True)

# ======================================================================
# 🟩 SECTION 2: UI CUSTOMIZATION & FILTERS
# ======================================================================
st.sidebar.header("3. Visualization & Filters")
plot_backend = st.sidebar.selectbox("Plot engine", ["Plotly (interactive)", "Matplotlib (static)"])
theme = st.sidebar.selectbox("Theme/template", ["default", "plotly_white", "plotly_dark", "seaborn", "classic"])
bg_color = st.sidebar.color_picker("Background color", "#ffffff")
show_grid = st.sidebar.checkbox("Show grid lines", value=True)
line_width = st.sidebar.slider("Line width", min_value=1, max_value=6, value=2)
marker_size = st.sidebar.slider("Marker size (for plotly)", min_value=4, max_value=12, value=6)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns detected. Upload a dataset with numeric time series columns.")
    st.stop()

# 🟩 Numeric filter (optional)
st.sidebar.header("4. Numeric Filter (optional)")
filter_col = st.sidebar.selectbox("Filter rows by numeric column", options=[None] + numeric_cols, index=0)
if filter_col:
    minv, maxv = float(df[filter_col].min()), float(df[filter_col].max())
    lo, hi = st.sidebar.slider("Filter range", min_value=minv, max_value=maxv, value=(minv, maxv))
    df = df[(df[filter_col] >= lo) & (df[filter_col] <= hi)]

# ======================================================================
# 🟩 SECTION 3: DATA PREVIEW & DESCRIPTIVE STATISTICS
# ======================================================================
st.header("🧾 Descriptive Statistics & Data Overview")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Data preview")
    st.dataframe(df.head(50))
with col2:
    st.subheader("Summary statistics")
    st.write(df.describe(include="all"))
    st.subheader("Missing values")
    st.write(df.isna().sum())

# ======================================================================
# 🟩 SECTION 4: DEPENDENT / INDEPENDENT VARIABLE SELECTION
# ======================================================================
st.header("🔎 Variable Selection")
dep_var = st.selectbox("Dependent variable (Y)", options=numeric_cols, index=0)
indep_var = st.selectbox("Independent variable (X) — optional", options=[None] + numeric_cols, index=0)

# ======================================================================
# 🟩 SECTION 5: TIME SERIES PLOT (LINE PLOT / MULTI-SERIES)
# ======================================================================
st.header("📈 Time Series Plot")
if plot_backend.startswith("Plotly"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[dep_var], mode="lines+markers",
                             name=dep_var, line=dict(width=line_width), marker=dict(size=marker_size)))
    if indep_var and indep_var != dep_var:
        fig.add_trace(go.Scatter(x=df.index, y=df[indep_var], mode="lines",
                                 name=indep_var, line=dict(width=line_width)))
    fig.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color, title=f"Time series: {dep_var}")
    if not show_grid:
        fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    if theme == "seaborn":
        sns.set()
    else:
        plt.style.use('classic' if theme == "classic" else 'default')
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df[dep_var], linewidth=line_width, marker='o', markersize=marker_size/2)
    if indep_var and indep_var != dep_var:
        ax.plot(df.index, df[indep_var], linewidth=line_width, alpha=0.8)
    ax.set_facecolor(bg_color)
    ax.grid(show_grid)
    ax.set_title(f"Time series: {dep_var}")
    st.pyplot(fig)

# ======================================================================
# 🟩 SECTION 6: HISTOGRAM / DISTRIBUTION
# ======================================================================
st.header("📊 Histogram / Distribution")
hist_col = st.selectbox("Choose column for histogram", options=numeric_cols, index=0, key="hist")
bins = st.slider("Bins", min_value=5, max_value=100, value=30, key="bins")
if plot_backend.startswith("Plotly"):
    fig_h = px.histogram(df.reset_index(), x=hist_col, nbins=bins, title=f"Histogram: {hist_col}")
    fig_h.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color)
    if not show_grid:
        fig_h.update_xaxes(showgrid=False); fig_h.update_yaxes(showgrid=False)
    st.plotly_chart(fig_h, use_container_width=True)
else:
    fig_h, axh = plt.subplots(figsize=(8, 4))
    axh.hist(df[hist_col].dropna(), bins=bins)
    axh.set_title(f"Histogram: {hist_col}")
    axh.set_facecolor(bg_color)
    axh.grid(show_grid)
    st.pyplot(fig_h)

# ======================================================================
# 🟩 SECTION 7: SCATTER PLOT
# ======================================================================
st.header("🔹 Scatter Plot")
sx = st.selectbox("Scatter X (independent)", options=numeric_cols, index=0, key="sx")
sy = st.selectbox("Scatter Y (dependent)", options=numeric_cols, index=min(1, len(numeric_cols)-1), key="sy")
color_by_options = [None] + [c for c in df.columns if df[c].nunique() < 50]
color_by = st.selectbox("Color by (categorical) — optional", options=color_by_options, index=0)
if plot_backend.startswith("Plotly"):
    scdf = df.reset_index()
    fig_s = px.scatter(scdf, x=sx, y=sy, color=color_by if color_by else None,
                       title=f"{sy} vs {sx}")
    fig_s.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color)
    st.plotly_chart(fig_s, use_container_width=True)
else:
    fig_s, axs = plt.subplots(figsize=(8, 5))
    axs.scatter(df[sx], df[sy], s=20)
    axs.set_facecolor(bg_color)
    axs.grid(show_grid)
    axs.set_title(f"{sy} vs {sx}")
    st.pyplot(fig_s)

# ======================================================================
# 🟩 SECTION 8: CORRELATION HEATMAP
# ======================================================================
st.header("📉 Correlation Heatmap")
corr = df[numeric_cols].corr()
fig_c, axc = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axc)
axc.set_facecolor(bg_color)
st.pyplot(fig_c)

# ======================================================================
# 🟩 SECTION 9: BOXPLOTS
# ======================================================================
st.header("📦 Boxplots")
box_col = st.selectbox("Choose column for boxplot", options=numeric_cols, index=0, key="box")
fig_b, axb = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x=box_col, ax=axb)
axb.set_facecolor(bg_color)
st.pyplot(fig_b)

# ======================================================================
# 🟩 SECTION 10: DECOMPOSITION (TREND / SEASONAL / RESIDUAL)
# ======================================================================
st.header("🔬 Seasonal Decomposition & Residuals")
period = st.number_input("Seasonal period (days)", min_value=2, max_value=2000, value=365)
try:
    decomp = seasonal_decompose(df[dep_var].dropna(), period=int(period), model='additive', extrapolate_trend='freq')
    # Plot decomposition with plotly for interactivity
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=decomp.observed.index, y=decomp.observed, name="Observed"))
    fig_d.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, name="Trend"))
    fig_d.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, name="Seasonal"))
    fig_d.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, name="Residual"))
    fig_d.update_layout(title=f"Decomposition: {dep_var}", plot_bgcolor=bg_color, paper_bgcolor=bg_color)
    st.plotly_chart(fig_d, use_container_width=True)
except Exception as e:
    st.warning(f"Decomposition failed: {e}")

# ======================================================================
# 🟩 SECTION 11: ACF / PACF
# ======================================================================
st.header("📡 ACF & PACF")
try:
    fig_acf, ax_acf = plt.subplots(figsize=(10, 3))
    plot_acf(df[dep_var].dropna(), ax=ax_acf, lags=40)
    ax_acf.set_facecolor(bg_color)
    st.pyplot(fig_acf)

    fig_pacf, ax_pacf = plt.subplots(figsize=(10, 3))
    plot_pacf(df[dep_var].dropna(), ax=ax_pacf, lags=40, method='ywm')
    ax_pacf.set_facecolor(bg_color)
    st.pyplot(fig_pacf)
except Exception as e:
    st.warning(f"ACF/PACF plotting failed: {e}")

# ======================================================================
# 🟩 SECTION 12: EXPORT / DOWNLOAD PROCESSED DATA
# ======================================================================
st.header("💾 Download Processed / Filtered Data")
to_download = df.reset_index()
csv_bytes = to_download.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV of processed data", csv_bytes, file_name="processed_time_series.csv", mime="text/csv")

# ======================================================================
# 🟩 SECTION 13: MACHINE LEARNING FORECASTING (PROPHET MODEL)
# ======================================================================
from prophet import Prophet

st.header("🤖 Machine Learning Forecasting (Prophet Model)")

# Select dependent and independent variables
dep_forecast = st.selectbox("Select dependent variable (target for forecasting)", numeric_cols, index=0, key="forecast_dep")
regressors = st.multiselect("Select independent variables (optional regressors)", [c for c in numeric_cols if c != dep_forecast])

# Select forecast horizon
periods = st.number_input("Forecast horizon (future days to predict)", min_value=7, max_value=1000, value=90)

if st.button("Run Forecast"):
    # Prepare data for Prophet
    df_prophet = df.reset_index()[[df.index.name, dep_forecast] + regressors].rename(columns={df.index.name: "ds", dep_forecast: "y"})

    # Initialize model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    for reg in regressors:
        model.add_regressor(reg)

    # Fit model
    try:
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=int(periods), freq="D")

        # Add regressors to future if available
        if regressors:
            last_vals = df_prophet[regressors].iloc[-1:].to_dict(orient="records")[0]
            for reg in regressors:
                future[reg] = last_vals[reg]

        forecast = model.predict(future)

        # Plot results
        st.subheader("Forecast Plot")
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="lines", name="Observed"))
        fig_f.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
        fig_f.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], fill=None, mode="lines",
                                   line_color="lightgray", name="Upper Bound"))
        fig_f.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], fill='tonexty', mode="lines",
                                   line_color="lightgray", name="Lower Bound", opacity=0.3))
        fig_f.update_layout(title=f"Forecast for {dep_forecast} (Prophet Model)",
                            plot_bgcolor=bg_color, paper_bgcolor=bg_color)
        st.plotly_chart(fig_f, use_container_width=True)

        st.subheader("Forecast Components")
        st.pyplot(model.plot_components(forecast))

        st.subheader("Forecast Table (Last 30 predictions)")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30))

    except Exception as e:
        st.error(f"Forecasting failed: {e}")

# ======================================================================
# 🟩 FOOTER & HELP
# ======================================================================
st.sidebar.header("Help & Run")
st.sidebar.markdown("Requirements: `streamlit pandas numpy matplotlib seaborn plotly statsmodels openpyxl`")
st.sidebar.markdown("Run locally: `streamlit run app.py`")
st.sidebar.markdown("Search for `# 🟩 SECTION` to locate code blocks for editing.")

# End of file

