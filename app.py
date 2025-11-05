# ======================================================================
# 🟩 SECTION 6: DESCRIPTIVE STATISTICS (ROUNDED + DOWNLOAD & COPY)
# ======================================================================
import io
import pyperclip  # optional for local copy, not required for web
import base64

st.header("📈 Descriptive Statistics")

# Compute descriptive statistics and round to 3 decimals
desc_stats = df.describe().round(3)

# Display dataframe with Streamlit built-in tool
st.dataframe(desc_stats)

# 🟩 Convert DataFrame to CSV for download
csv_buffer = io.StringIO()
desc_stats.to_csv(csv_buffer)
csv_data = csv_buffer.getvalue()
b64 = base64.b64encode(csv_data.encode()).decode()

# 🟩 Download button
st.download_button(
    label="📥 Download Descriptive Statistics as CSV",
    data=csv_data,
    file_name="descriptive_statistics.csv",
    mime="text/csv"
)

# 🟩 Copy-to-Clipboard (browser compatible using markdown)
st.markdown("""
**📋 Copy Table Data**
Click below, then press `Ctrl + C` / `Cmd + C` after selecting all text.
""")
st.text(desc_stats.to_string())

st.success("Descriptive statistics rounded to 3 decimals. You can copy or download results above.")
