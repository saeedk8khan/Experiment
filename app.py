
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import io

st.set_page_config(layout="wide", page_title="Advanced Time Series Explorer")

st.title("Advanced Time Series Explorer")

# Sidebar: sample data or upload
st.sidebar.header("Data input & parsing")
use_sample = st.sidebar.checkbox("Use sample dataset (provided)", value=True)
uploaded_file = None
if not use_sample:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if use_sample:
    df = pd.read_csv("sample_time_series.csv", parse_dates=["date"])
else:
    if uploaded_file is None:
        st.info("Upload a CSV or Excel file or enable 'Use sample dataset' in the sidebar.")
        st.stop()
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, parse_dates=True)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

# Auto-detect datetime column or let user choose
st.sidebar.header("Datetime settings")
possible_dt = [c for c in df.columns if "date" in c.lower() or np.issubdtype(df[c].dtype, np.datetime64)]
dt_col = None
if possible_dt:
    dt_col = st.sidebar.selectbox("Select datetime column", options=possible_dt, index=0)
else:
    dt_col = st.sidebar.selectbox("Select datetime column", options=["--choose--"] + list(df.columns))
    if dt_col == "--choose--":
        st.error("No datetime column detected. Upload a file with a date column or convert one column to datetime.")
        st.stop()
    # try convert
    try:
        df[dt_col] = pd.to_datetime(df[dt_col])
    except Exception as e:
        st.error(f"Could not parse the chosen column as datetime: {e}")
        st.stop()

df = df.sort_values(dt_col).reset_index(drop=True)
df[dt_col] = pd.to_datetime(df[dt_col])
df.set_index(dt_col, inplace=True)

st.sidebar.header("Plot customizations")
plot_backend = st.sidebar.selectbox("Plot backend", ["plotly", "matplotlib"])
theme = st.sidebar.selectbox("Plot theme/template", ["default", "plotly_white", "plotly_dark", "seaborn", "classic"])
bg_color = st.sidebar.color_picker("Background color", value="#ffffff")
show_grid = st.sidebar.checkbox("Show grid lines", value=True)
line_width = st.sidebar.slider("Line width", min_value=1, max_value=4, value=2)
marker_size = st.sidebar.slider("Marker size (plotly)", min_value=4, max_value=12, value=6)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns detected for plotting.")
    st.stop()

st.sidebar.header("Modeling / filtering")
# simple numeric filter based on one numeric column
filter_col = st.sidebar.selectbox("Optional numeric filter column", options=[None] + numeric_cols, index=0)
if filter_col:
    minv, maxv = float(df[filter_col].min()), float(df[filter_col].max())
    lo, hi = st.sidebar.slider("Filter range", min_value=minv, max_value=maxv, value=(minv, maxv))
    df = df[(df[filter_col] >= lo) & (df[filter_col] <= hi)]

# main layout
st.header("Data preview & summary")
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("First rows")
    st.dataframe(df.head(50))

with col2:
    st.subheader("Summary statistics and missing values")
    st.write(df.describe())
    missing = df.isna().sum().to_frame("missing_count")
    st.write(missing)

st.header("Time series visualization")
dep_var = st.selectbox("Dependent (target) variable", options=numeric_cols, index=0)
indep_var = st.selectbox("Independent (optional) - for scatter or comparison", options=[None] + numeric_cols, index=0)

# Time series line plot
st.subheader("Time series plot")
if plot_backend == "plotly":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[dep_var], mode="lines+markers", name=dep_var,
                             line=dict(width=line_width), marker=dict(size=marker_size)))
    if indep_var and indep_var != dep_var:
        fig.add_trace(go.Scatter(x=df.index, y=df[indep_var], mode="lines", name=indep_var,
                                 line=dict(width=line_width)))
    fig.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color)
    if not show_grid:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    if theme == "seaborn":
        sns.set()
    else:
        plt.style.use('classic' if theme == "classic" else 'default')
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df[dep_var], linewidth=line_width, marker="o", markersize=marker_size)
    if indep_var and indep_var != dep_var:
        ax.plot(df.index, df[indep_var], linewidth=line_width, alpha=0.8)
    ax.set_facecolor(bg_color)
    ax.grid(show_grid)
    st.pyplot(fig)

# Histogram selector
st.subheader("Histogram / Distribution")
hist_col = st.selectbox("Choose column for histogram", options=numeric_cols, index=0, key="hist")
bins = st.slider("Bins", min_value=5, max_value=100, value=30, key="bins")
if plot_backend == "plotly":
    fig2 = px.histogram(df, x=hist_col, nbins=bins, title=f"Histogram of {hist_col}")
    fig2.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color)
    if not show_grid:
        fig2.update_xaxes(showgrid=False); fig2.update_yaxes(showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)
else:
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.hist(df[hist_col].dropna(), bins=bins)
    ax2.set_facecolor(bg_color)
    ax2.grid(show_grid)
    st.pyplot(fig2)

# Scatter plot
st.subheader("Scatter plot")
scatter_x = st.selectbox("X (independent)", options=numeric_cols, index=0, key="sx")
scatter_y = st.selectbox("Y (dependent)", options=numeric_cols, index=min(1, len(numeric_cols)-1), key="sy")
color_by = st.selectbox("Color by (optional categorical)", options=[None] + [c for c in df.columns if df[c].nunique() < 50], index=0)
if plot_backend == "plotly":
    fig3 = px.scatter(df.reset_index(), x=scatter_x, y=scatter_y, color=color_by, title=f"{scatter_y} vs {scatter_x}")
    fig3.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color)
    st.plotly_chart(fig3, use_container_width=True)
else:
    fig3, ax3 = plt.subplots(figsize=(6,4))
    ax3.scatter(df[scatter_x], df[scatter_y], s=20)
    ax3.set_facecolor(bg_color)
    ax3.grid(show_grid)
    st.pyplot(fig3)

# Correlation heatmap
st.subheader("Correlation heatmap")
corr = df[numeric_cols].corr()
fig4, ax4 = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
ax4.set_facecolor(bg_color)
st.pyplot(fig4)

# Boxplots (various)
st.subheader("Boxplots")
box_col = st.selectbox("Choose numeric for boxplot", options=numeric_cols, index=0, key="box")
fig5, ax5 = plt.subplots(figsize=(6,4))
sns.boxplot(data=df, x=box_col, ax=ax5)
ax5.set_facecolor(bg_color)
st.pyplot(fig5)

# Decomposition and ACF/PACF
st.header("Decomposition & ACF/PACF")
period = st.number_input("Seasonal period (days)", min_value=2, max_value=730, value=365)
try:
    decompose_result = seasonal_decompose(df[dep_var].dropna(), period=int(period), model="additive", extrapolate_trend='freq')
    st.subheader("Seasonal decomposition (observed / trend / seasonal / resid)")
    cols = st.columns(1)
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=decompose_result.observed.index, y=decompose_result.observed, name="observed"))
    fig6.add_trace(go.Scatter(x=decompose_result.trend.index, y=decompose_result.trend, name="trend"))
    fig6.add_trace(go.Scatter(x=decompose_result.seasonal.index, y=decompose_result.seasonal, name="seasonal"))
    fig6.add_trace(go.Scatter(x=decompose_result.resid.index, y=decompose_result.resid, name="resid"))
    fig6.update_layout(plot_bgcolor=bg_color, paper_bgcolor=bg_color)
    st.plotly_chart(fig6, use_container_width=True)
except Exception as e:
    st.warning(f"Decomposition failed: {e}")

# ACF PACF
try:
    fig_acf, ax_acf = plt.subplots(figsize=(8,3))
    plot_acf(df[dep_var].dropna(), ax=ax_acf, lags=40)
    st.pyplot(fig_acf)
    fig_pacf, ax_pacf = plt.subplots(figsize=(8,3))
    plot_pacf(df[dep_var].dropna(), ax=ax_pacf, lags=40, method='ywm')
    st.pyplot(fig_pacf)
except Exception as e:
    st.warning(f"ACF/PACF plots failed: {e}")

# Export filtered/processed dataset
st.header("Download processed / filtered data")
to_download = df.reset_index()
csv_bytes = to_download.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV of processed data", csv_bytes, file_name="processed_time_series.csv", mime="text/csv")

st.sidebar.header("Help & requirements")
st.sidebar.write("""
Requirements: pip install streamlit pandas numpy matplotlib seaborn plotly statsmodels openpyxl
Run: streamlit run app.py
""")
