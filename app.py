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
# 🟩 SECTION X: UNIT ROOT TEST (ADF & PP) — FINAL STABLE VERSION
# ======================================================================

import io
import numpy as np
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron

st.header("🔍 Unit Root Test (ADF & PP)")

# Variable selection
selected_vars = st.multiselect("Select Variable(s) for Unit Root Test", df.columns)
selected_lag = st.selectbox("Select Lag Order", list(range(0, 10)), index=0)

# Transformation choice
diff_option = st.radio(
    "Select Data Level:",
    ["Level (Original Data)", "First Difference"],
    horizontal=True
)

if st.button("Run Unit Root Tests"):
    try:
        results = []

        for col in selected_vars:
            series = df[col].dropna()

            # Apply differencing if needed
            if diff_option == "First Difference":
                series = series.diff().dropna()

            # ----- ADF TEST -----
            adf_result = adfuller(series, maxlag=selected_lag, autolag=None)
            adf_stat = round(adf_result[0], 4)
            adf_p = round(adf_result[1], 4)
            adf_usedlag = adf_result[2]
            adf_nobs = adf_result[3]
            adf_crit = adf_result[4]
            adf_icbest = adf_result[5] if len(adf_result) > 5 else np.nan

            results.append({
                "Variable": col,
                "Test": "ADF",
                "Level": "First Diff" if diff_option == "First Difference" else "Level",
                "Lag Used": adf_usedlag,
                "AIC": round(adf_icbest, 4) if not np.isnan(adf_icbest) else None,
                "Test Statistic": adf_stat,
                "p-Value": adf_p,
                "1% CV": round(adf_crit.get('1%'), 4),
                "5% CV": round(adf_crit.get('5%'), 4),
                "10% CV": round(adf_crit.get('10%'), 4)
            })

            # ----- PHILLIPS–PERRON TEST -----
            pp_res = PhillipsPerron(series, lags=selected_lag)
            results.append({
                "Variable": col,
                "Test": "PP",
                "Level": "First Diff" if diff_option == "First Difference" else "Level",
                "Lag Used": selected_lag,
                "AIC": None,
                "Test Statistic": round(pp_res.stat, 4),
                "p-Value": round(pp_res.pvalue, 4),
                "1% CV": None,
                "5% CV": None,
                "10% CV": None
            })

        # Combine results
        results_df = pd.DataFrame(results)

        # Display
        st.subheader("📊 Unit Root Test Results")
        st.dataframe(results_df, use_container_width=True)

        # Copy results
        st.markdown("#### 📋 Copy Results")
        st.code(results_df.to_markdown(index=False), language="markdown")

        # Download Excel
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            results_df.to_excel(writer, index=False, sheet_name="UnitRootResults")

        st.download_button(
            label="📥 Download Unit Root Test Results (Excel)",
            data=excel_buf.getvalue(),
            file_name="unit_root_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error while running Unit Root Tests: {e}")


# ======================================================================
# 🟩 SECTION X: COINTEGRATION ANALYSIS (ENGLE–GRANGER & JOHANSEN)
# ======================================================================

import io
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

st.header("🔗 Cointegration Analysis")

# Variable selection
st.subheader("Variable Selection")
selected_vars = st.multiselect("Select Variables for Cointegration Analysis", df.columns.tolist())

# Lag selection
selected_lag = st.selectbox("Select Maximum Lag", list(range(0, 10)), index=1)

# Test selection buttons
col_btn1, col_btn2 = st.columns(2)
run_engle = col_btn1.button("Run Engle–Granger Cointegration Test")
run_johansen = col_btn2.button("Run Johansen Cointegration Test")

# ====================== ENGLE–GRANGER TEST ======================
if run_engle:
    try:
        if len(selected_vars) < 2:
            st.warning("Please select at least two variables for Engle–Granger test.")
        else:
            results = []
            y = selected_vars[0]
            x_vars = selected_vars[1:]

            for x in x_vars:
                score, pvalue, _ = coint(df[y].dropna(), df[x].dropna(), maxlag=selected_lag)
                results.append({
                    "Test Type": "Engle–Granger",
                    "Dependent (Y)": y,
                    "Independent (X)": x,
                    "Lag Used": selected_lag,
                    "Test Statistic": round(score, 4),
                    "p-Value": round(pvalue, 4)
                })

            results_df = pd.DataFrame(results)

            st.subheader("📊 Engle–Granger Cointegration Test Results")
            st.dataframe(results_df, use_container_width=True)

            # Copy results
            st.markdown("#### 📋 Copy Results")
            st.code(results_df.to_markdown(index=False), language="markdown")

            # Download results
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                results_df.to_excel(writer, index=False, sheet_name="EngleGranger")
            st.download_button(
                label="📥 Download Engle–Granger Results (Excel)",
                data=excel_buf.getvalue(),
                file_name="engle_granger_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Error while running Engle–Granger test: {e}")

# ====================== JOHANSEN TEST ======================
if run_johansen:
    try:
        if len(selected_vars) < 2:
            st.warning("Please select at least two variables for Johansen test.")
        else:
            data = df[selected_vars].dropna()

            johansen_result = coint_johansen(data, det_order=0, k_ar_diff=selected_lag)
            trace_stats = johansen_result.lr1
            trace_crit = johansen_result.cvt
            maxeig_stats = johansen_result.lr2
            maxeig_crit = johansen_result.cvm

            results_trace = pd.DataFrame({
                "Test": ["Johansen Trace"] * len(trace_stats),
                "Rank": list(range(len(trace_stats))),
                "Test Statistic": trace_stats.round(4),
                "Crit 90%": trace_crit[:, 0].round(4),
                "Crit 95%": trace_crit[:, 1].round(4),
                "Crit 99%": trace_crit[:, 2].round(4)
            })

            results_maxeig = pd.DataFrame({
                "Test": ["Johansen Max-Eigen"] * len(maxeig_stats),
                "Rank": list(range(len(maxeig_stats))),
                "Test Statistic": maxeig_stats.round(4),
                "Crit 90%": maxeig_crit[:, 0].round(4),
                "Crit 95%": maxeig_crit[:, 1].round(4),
                "Crit 99%": maxeig_crit[:, 2].round(4)
            })

            results_df = pd.concat([results_trace, results_maxeig], ignore_index=True)

            st.subheader("📊 Johansen Cointegration Test Results")
            st.dataframe(results_df, use_container_width=True)

            # Copy results
            st.markdown("#### 📋 Copy Results")
            st.code(results_df.to_markdown(index=False), language="markdown")

            # Download results
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                results_df.to_excel(writer, index=False, sheet_name="Johansen")
            st.download_button(
                label="📥 Download Johansen Results (Excel)",
                data=excel_buf.getvalue(),
                file_name="johansen_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Error while running Johansen test: {e}")


# ======================================================================
# 🟩 SECTION 3: DESCRIPTIVE STATISTICS & DATA OVERVIEW (RAW / LOG OPTION)
# ======================================================================

import io
import numpy as np

st.header("🧾 Descriptive Statistics & Data Overview")

# 🟩 User selects whether to use raw or log-transformed data
data_option = st.radio(
    "Select data type for descriptive statistics:",
    ["Raw Data", "Log-Transformed Data"],
    horizontal=True
)

# 🟩 Prepare data according to user selection
if data_option == "Log-Transformed Data":
    # Replace non-positive values with NaN to avoid log errors
    log_df = df.copy()
    for col in log_df.select_dtypes(include=[np.number]).columns:
        log_df[col] = log_df[col].apply(lambda x: np.log(x) if x > 0 else np.nan)
    display_df = log_df
else:
    display_df = df

col1, col2 = st.columns(2)

# 🟩 Data Preview
with col1:
    st.subheader(f"{data_option} Preview")
    st.dataframe(display_df.head(50), use_container_width=True)

# 🟩 Summary Statistics with 3 decimals
with col2:
    st.subheader(f"{data_option} Summary Statistics (Rounded to 3 Decimals)")
    summary_df = display_df.describe(include="all").round(3)
    st.dataframe(summary_df, use_container_width=True)

# 🟩 Download descriptive stats as Excel
excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    summary_df.to_excel(writer, index=True, sheet_name=f"{data_option}_Stats")

st.download_button(
    label=f"📥 Download {data_option} Summary Statistics (Excel)",
    data=excel_buf.getvalue(),
    file_name=f"{data_option.lower().replace(' ', '_')}_summary_statistics.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ======================================================================
# 🟩 SECTION 4: DEPENDENT / INDEPENDENT VARIABLE SELECTION
# ======================================================================
st.header("🔎 Variable Selection")
dep_var = st.selectbox("Dependent variable (Y)", options=numeric_cols, index=0)
indep_var = st.selectbox("Independent variable (X) — optional", options=[None] + numeric_cols, index=0)


# ======================================================================
# 🟩 SECTION 5: TIME SERIES PLOT (Enhanced – Single or All Variables)
# ======================================================================
st.header("📈 Time Series Plot")

plot_mode = st.radio(
    "Select Plot Mode:",
    options=["Single Variable", "All Variables"],
    index=0,
    horizontal=True
)

if plot_backend.startswith("Plotly"):
    fig = go.Figure()

    if plot_mode == "Single Variable":
        # Plot only the selected dependent (and optional independent) variable
        fig.add_trace(go.Scatter(
            x=df.index, y=df[dep_var], mode="lines+markers",
            name=dep_var, line=dict(width=line_width), marker=dict(size=marker_size)
        ))
        if indep_var and indep_var != dep_var:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[indep_var], mode="lines",
                name=indep_var, line=dict(width=line_width)
            ))

    else:
        # Plot all numeric variables together
        for col in numeric_cols:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines",
                name=col, line=dict(width=line_width)
            ))

    fig.update_layout(
        title="Time Series Plot" if plot_mode == "All Variables" else f"Time Series: {dep_var}",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        legend_title="Variables"
    )
    if not show_grid:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

else:
    # Matplotlib backend
    if theme == "seaborn":
        sns.set()
    else:
        plt.style.use('classic' if theme == "classic" else 'default')

    fig, ax = plt.subplots(figsize=(12, 4))

    if plot_mode == "Single Variable":
        ax.plot(df.index, df[dep_var], linewidth=line_width, marker='o', markersize=marker_size/2, label=dep_var)
        if indep_var and indep_var != dep_var:
            ax.plot(df.index, df[indep_var], linewidth=line_width, alpha=0.8, label=indep_var)
    else:
        for col in numeric_cols:
            ax.plot(df.index, df[col], linewidth=line_width, label=col)

    ax.set_facecolor(bg_color)
    ax.grid(show_grid)
    ax.set_title("Time Series Plot" if plot_mode == "All Variables" else f"Time Series: {dep_var}")
    ax.legend(loc="upper right", fontsize="small")
    st.pyplot(fig)

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
# ======================================================================
# 🟩 SECTION 8: CORRELATION HEATMAP (WITH COLOR PALETTE SELECTION)
# ======================================================================
st.header("📉 Correlation Heatmap")

# Sidebar color options for heatmap
st.sidebar.header("Heatmap Settings")
heatmap_palette = st.sidebar.selectbox(
    "Select heatmap color palette",
    ["coolwarm", "viridis", "plasma", "cividis", "magma", "crest", "rocket", "Spectral", "icefire", "vlag"],
    index=0
)

corr = df[numeric_cols].corr()
fig_c, axc = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap=heatmap_palette,
    ax=axc,
    cbar_kws={"shrink": 0.8}
)
axc.set_facecolor(bg_color)
axc.set_title("Correlation Heatmap", fontsize=14, weight="bold")
st.pyplot(fig_c)

# ======================================================================
# 🟩 SECTION 9: DISTRIBUTION COMPARISON (BAR, BOX, VIOLIN, STRIP)
# ======================================================================
import matplotlib.pyplot as plt
import seaborn as sns

st.header("📊 Distribution Comparison")

# Select a numeric column
col_to_plot = st.selectbox("Select numeric variable for distribution analysis", numeric_cols, key="dist_col")

# Sidebar controls for background and style
st.sidebar.header("Distribution Chart Settings")
sns_style = st.sidebar.selectbox("Select Seaborn Style", ["whitegrid", "darkgrid", "white", "ticks", "dark"], index=1)
plt_style = st.sidebar.selectbox("Select Plot Style", ["default", "seaborn-v0_8-colorblind", "seaborn-v0_8-poster", "classic"], index=0)
sns.set_style(sns_style)
plt.style.use(plt_style)

# 🟩 Function: create_distribution_chart
def create_distribution_chart(data_series, y_label, title):
    """
    Creates a figure with 4 horizontal subplots: bar, box, violin, strip.
    Each shows the same data in a different style.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    
    # 🟩 Subplot 1: Bar Plot (Mean & Std Dev)
    mean_val = data_series.mean()
    std_val = data_series.std()
    sns.barplot(x=["Mean"], y=[mean_val], ax=axes[0], color="skyblue", ci=None)
    axes[0].errorbar(x=[0], y=[mean_val], yerr=std_val, fmt='o', color='black', capsize=5)
    axes[0].set_xlabel("Mean & Std Dev")
    axes[0].set_ylabel(y_label)
    axes[0].set_title("Bar Plot")

    # 🟩 Subplot 2: Box Plot
    sns.boxplot(y=data_series, ax=axes[1], color="lightgreen")
    axes[1].set_xlabel("Quartile Box")
    axes[1].set_title("Box Plot")

    # 🟩 Subplot 3: Violin Plot
    sns.violinplot(y=data_series, ax=axes[2], color="lightcoral")
    axes[2].set_xlabel("Density")
    axes[2].set_title("Violin Plot")

    # 🟩 Subplot 4: Strip Plot (with mean line)
    sns.stripplot(y=data_series, ax=axes[3], color="gray", jitter=True, alpha=0.6)
    axes[3].axhline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.2f}")
    axes[3].set_xlabel("Data Points")
    axes[3].set_title("Strip Plot")
    axes[3].legend(loc="upper right", fontsize="small")

    # 🟩 Shared layout
    fig.suptitle(title, fontsize=16, weight="bold")
    fig.tight_layout(pad=2)
    return fig

# 🟩 Generate and Display the Combined Distribution Plot
if col_to_plot:
    fig = create_distribution_chart(df[col_to_plot].dropna(), y_label=col_to_plot, title=f"Distribution Overview: {col_to_plot}")
    st.pyplot(fig)


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
# 🟩 SECTION 12: EXPORT / DOWNLOAD PROCESSED DATA
# ======================================================================
st.header("💾 Download Processed / Filtered Data")
to_download = df.reset_index()
csv_bytes = to_download.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV of processed data", csv_bytes, file_name="processed_time_series.csv", mime="text/csv")


# ======================================================================
# 🟩 SECTION 7: GRANGER CAUSALITY TEST (FINAL CLEAN VERSION)
# ======================================================================

import io
from statsmodels.tsa.stattools import grangercausalitytests

st.header("🔁 Granger Causality Test")

# 🟩 User input selections
dependent_var = st.selectbox("Select Dependent Variable", df.columns)
independent_vars = st.multiselect("Select Independent Variable(s)", [c for c in df.columns if c != dependent_var])
max_lag = st.selectbox("Select Lag Order", list(range(1, 10)), index=0)

if st.button("Run Granger Causality Test"):
    try:
        results_list = []

        # Loop through selected independent variables
        for indep in independent_vars:
            temp_df = df[[dependent_var, indep]].dropna()

            # Run the Granger test for the selected lag only
            gc_res = grangercausalitytests(temp_df, maxlag=max_lag, verbose=False)
            f_test = gc_res[max_lag][0]['ssr_ftest'][0]
            p_value = gc_res[max_lag][0]['ssr_ftest'][1]
            results_list.append({
                "Hypothesis Tested": f"{indep} ➜ {dependent_var}",
                "Lag": max_lag,
                "F-test Statistic": round(f_test, 4),
                "Prob > F": round(p_value, 4)
            })

            # Reverse direction test (dependent ➜ independent)
            temp_df_rev = df[[indep, dependent_var]].dropna()
            gc_rev = grangercausalitytests(temp_df_rev, maxlag=max_lag, verbose=False)
            f_test_rev = gc_rev[max_lag][0]['ssr_ftest'][0]
            p_value_rev = gc_rev[max_lag][0]['ssr_ftest'][1]
            results_list.append({
                "Hypothesis Tested": f"{dependent_var} ➜ {indep}",
                "Lag": max_lag,
                "F-test Statistic": round(f_test_rev, 4),
                "Prob > F": round(p_value_rev, 4)
            })

        # 🟩 Convert results to DataFrame
        gc_df = pd.DataFrame(results_list)

        st.subheader("📊 Granger Causality Results")
        st.dataframe(gc_df, use_container_width=True)

        # 🟩 Copy results
        st.markdown("#### 📋 Copy Results")
        st.code(gc_df.to_markdown(index=False), language="markdown")

        # 🟩 Download Excel
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            gc_df.to_excel(writer, index=False, sheet_name="GrangerResults")

        st.download_button(
            label="📥 Download Results (Excel)",
            data=excel_buf.getvalue(),
            file_name="granger_causality_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Error while running Granger Causality Test: {e}")


# ======================================================================
# 🟩 SECTION: QUANTILE-ON-QUANTILE (QQR) ANALYSIS
# ======================================================================
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.header("🧭 Quantile-on-Quantile (QQR) Analysis — 2D heatmap & 3D surface")

# -----------------------
# UI: inputs
# -----------------------
st.markdown("Select dependent (Y) and independent (X). If your data is panel (long), optionally choose the panel column and a specific group (e.g., country).")

col_y, col_x = st.columns(2)
with col_y:
    q_y = st.selectbox("Dependent variable (Y)", options=numeric_cols, index=0)
with col_x:
    q_x = st.selectbox("Independent variable (X)", options=[c for c in numeric_cols if c != q_y], index=0)

# Panel options
panel_col = st.selectbox("Panel column (optional)", options=[None] + [c for c in df.columns if df[c].nunique() > 1 and df[c].dtype == object], index=0)
selected_group = None
if panel_col:
    groups = ["All"] + sorted(df[panel_col].dropna().unique().tolist())
    selected_group = st.selectbox("Select group (or All)", options=groups, index=0)

# Q-grid settings
col_quant = st.columns(3)
with col_quant[0]:
    n_q = st.number_input("Number of quantiles (grid size)", min_value=5, max_value=101, value=21, step=2)
with col_quant[1]:
    bw = st.slider("Bandwidth (quantile fraction)", min_value=0.01, max_value=0.3, value=0.05, step=0.01,
                   help="For each X-quantile θ we take observations with X between quantile(θ-bw/2) and quantile(θ+bw/2).")
with col_quant[2]:
    min_q = st.number_input("Min quantile", min_value=0.0, max_value=0.49, value=0.01, step=0.01)
    max_q = st.number_input("Max quantile", min_value=0.51, max_value=1.0, value=0.99, step=0.01)

if min_q >= max_q:
    st.error("Min quantile must be smaller than Max quantile.")
    st.stop()

taus = np.linspace(min_q, max_q, n_q)
thetas = np.linspace(min_q, max_q, n_q)

# Plot appearance
st.sidebar.header("QQR Plot Settings")
colorscale = st.sidebar.selectbox("Heatmap colorscale (Plotly)", ["Viridis","Cividis","Plasma","Magma","Turbo","RdBu","IceFire","Viridis"], index=0)
surface_colorscale = st.sidebar.selectbox("Surface colorscale", ["Viridis","Cividis","Plasma","Magma","Turbo","RdBu"], index=0)
interp_method = st.sidebar.selectbox("Heatmap interpolation", ["nearest","bilinear"], index=0)

# Option to compute per-country or overall
per_group = False
if panel_col:
    per_group = st.sidebar.checkbox("Apply per selected group (panel)", value=False)
    # if checked, user must select a single group (selected_group already provides it)

# Run button
if st.button("Run QQR Analysis"):
    try:
        # subset by group if needed
        if panel_col and selected_group and selected_group != "All":
            subdf = df[df[panel_col] == selected_group].copy()
            if subdf.empty:
                st.error("Selected group has no data.")
                st.stop()
        else:
            subdf = df.copy()

        x_series = subdf[q_x].dropna()
        y_series = subdf[q_y].dropna()
        # ensure same index alignment if necessary
        joined = pd.concat([x_series, y_series], axis=1).dropna()
        x = joined[q_x]
        y = joined[q_y]
        if x.empty or y.empty:
            st.error("No overlapping valid observations for chosen variables.")
            st.stop()

        # Function to compute QQR matrix using local quantile-window approach
        def compute_qqr_matrix(x, y, thetas, taus, bandwidth):
            """
            For each theta (quantile of x), select obs where x between quantile(theta - bw/2) and quantile(theta + bw/2).
            Then compute the tau-quantile of y on that subset.
            Returns matrix of shape (len(taus), len(thetas)) where rows=taus, cols=thetas.
            """
            n_t = len(thetas)
            n_tau = len(taus)
            mat = np.full((n_tau, n_t), np.nan)
            # precompute quantiles of x for speed
            for j, th in enumerate(thetas):
                low_q = max(0.0, th - bandwidth/2)
                high_q = min(1.0, th + bandwidth/2)
                x_low = x.quantile(low_q)
                x_high = x.quantile(high_q)
                subset_mask = (x >= x_low) & (x <= x_high)
                subset_y = y.loc[subset_mask]
                if subset_y.size == 0:
                    # leave column as NaN
                    continue
                # compute requested taus for this subset
                for i, tau in enumerate(taus):
                    try:
                        mat[i, j] = float(np.quantile(subset_y, tau))
                    except Exception:
                        mat[i, j] = np.nan
            return mat

        with st.spinner("Computing QQR matrix..."):
            mat = compute_qqr_matrix(x, y, thetas, taus, bw)

        # DataFrame for download & display: index=taus as strings, columns=thetas
        col_names = [f"theta_{round(t,3)}" for t in thetas]
        row_names = [f"tau_{round(t,3)}" for t in taus]
        mat_df = pd.DataFrame(mat, index=row_names, columns=col_names)

        st.subheader("🔢 QQR Matrix (rows = τ for Y; cols = θ for X)")
        st.dataframe(mat_df, use_container_width=True)

        # Copyable table
        st.markdown("#### 📋 Copy QQR Matrix (plain text)")
        st.code(mat_df.to_string(), language="text")

        # Download CSV
        csv_buf = io.StringIO()
        mat_df.to_csv(csv_buf)
        st.download_button("📥 Download QQR Matrix (CSV)", csv_buf.getvalue().encode('utf-8'),
                           file_name=f"qqr_matrix_{q_x}_vs_{q_y}.csv", mime="text/csv")

        # -----------------------
        # 2D Heatmap (Plotly)
        # -----------------------
        st.subheader("🖼️ 2D Heatmap (Quantiles of Y vs Quantiles of X)")
        # px.imshow expects z as 2D array with rows mapping to y-axis (taus) and cols to x-axis (thetas)
        try:
            fig_h = px.imshow(mat,
                              labels=dict(x="θ (quantile of X)", y="τ (quantile of Y)", color="Quantile(Y|X≈θ)"),
                              x=np.round(thetas, 4),
                              y=np.round(taus, 4),
                              aspect="auto",
                              color_continuous_scale=colorscale,
                              origin='lower',
                              zmin=np.nanmin(mat) if np.isfinite(np.nanmin(mat)) else None,
                              zmax=np.nanmax(mat) if np.isfinite(np.nanmax(mat)) else None)
            fig_h.update_layout(height=600)
            fig_h.update_xaxes(tickmode='array')
            fig_h.update_yaxes(tickmode='array')
            st.plotly_chart(fig_h, use_container_width=True)
        except Exception as e:
            st.error(f"Heatmap plotting failed: {e}")

        # Safe download of heatmap as PNG (Plotly)
        st.subheader("📥 Download Heatmap")
        try:
            img_bytes = fig_h.to_image(format="png")
            st.download_button("Download heatmap PNG", img_bytes, file_name=f"qqr_heatmap_{q_x}_vs_{q_y}.png", mime="image/png")
        except Exception:
            st.warning("PNG export for heatmap not available in this environment. Try locally for full export.")

        # -----------------------
        # 3D Surface (Plotly)
        # -----------------------
        st.subheader("🌐 3D Surface (Quantile Surface)")
        try:
            surface = go.Figure(data=[go.Surface(z=mat, x=np.round(thetas,4), y=np.round(taus,4), colorscale=surface_colorscale)])
            surface.update_layout(scene=dict(
                xaxis_title='θ (quantile of X)',
                yaxis_title='τ (quantile of Y)',
                zaxis_title='Quantile(Y|X≈θ)'
            ), autosize=True, height=700)
            st.plotly_chart(surface, use_container_width=True)
        except Exception as e:
            st.error(f"3D surface plotting failed: {e}")

        # Safe download of surface as PNG
        st.subheader("📥 Download 3D Surface")
        try:
            img2 = surface.to_image(format="png")
            st.download_button("Download 3D surface PNG", img2, file_name=f"qqr_surface_{q_x}_vs_{q_y}.png", mime="image/png")
        except Exception:
            st.warning("PNG export for 3D surface not available in this environment. Run locally for full export.")

    except Exception as e:
        st.error(f"Error during QQR computation: {e}")


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
