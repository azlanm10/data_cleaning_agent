import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent
from data_cleaning_agent.utils import quality_metrics

load_dotenv()

st.set_page_config(page_title="Data Cleaning Agent", layout="wide")
st.title("🧹 Data Cleaning Agent")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    # Sidebar controls
    st.sidebar.header("Controls")
    user_instructions = st.sidebar.text_area(
        "Custom cleaning instructions (optional)",
        placeholder="Example: Don't remove outliers for salary"
    )

    show_code = st.sidebar.checkbox("Show generated cleaning code", value=True)
    show_debug = st.sidebar.checkbox("Show debug / error details", value=True)

    # Layout
    left, right = st.columns([1.15, 1.05])

    with left:
        st.subheader("Raw Data Preview")
        st.write(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
        st.dataframe(df_raw.head(50), width=1000)

    with right:
        st.subheader("Data Quality (Raw Data)")
        raw_metrics = quality_metrics(df_raw)

        a, b, c = st.columns(3)
        a.metric("Rows", raw_metrics["shape"][0])
        b.metric("Columns", raw_metrics["shape"][1])
        c.metric("Duplicate rows", raw_metrics["dup_rows"])

        st.metric("Rows >50% missing", raw_metrics["row_missing_over_50"])

        st.caption("Missingness by column (top 15)")
        miss_top = raw_metrics["missing_pct"].head(15).reset_index()
        miss_top.columns = ["column", "missing_percent"]
        st.bar_chart(miss_top.set_index("column"))

        st.caption("Dtype distribution")
        dtype_df = raw_metrics["dtype_counts"].rename_axis("dtype").reset_index(name="count")
        st.dataframe(dtype_df, width=1000)

        if not raw_metrics["outlier_counts"].empty:
            st.caption("Outliers per numeric column (IQR 1.5 baseline, top 15)")
            out_df = raw_metrics["outlier_counts"].head(15).rename_axis("column").reset_index(name="outlier_count")
            st.bar_chart(out_df.set_index("column"))

    st.divider()

    if st.button("Clean Data", type="primary"):
        with st.spinner("Cleaning..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = LightweightDataCleaningAgent(model=llm, log=True)

            try:
                print(user_instructions)
                agent.invoke_agent(
                    data_raw=df_raw,
                    user_instructions=user_instructions if user_instructions else None,
                )

                df_cleaned = agent.get_data_cleaned()
                code = agent.get_data_cleaner_function()
                # If you haven't added get_error(), read it directly:
                err = agent.response.get("data_cleaner_error") if agent.response else None

                if err:
                    st.error("Cleaning failed.")
                    if show_debug:
                        st.code(err)
                        if show_code and code:
                            st.subheader("Generated Code (debug)")
                            st.code(code, language="python")
                else:
                    st.success("Done!")

                    c1, c2 = st.columns([1.15, 1.0])
                    with c1:
                        st.subheader("Cleaned Data Preview")
                        st.write(f"Shape: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
                        st.dataframe(df_cleaned.head(50), width=1000)

                        csv = df_cleaned.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download Cleaned Data",
                            data=csv,
                            file_name="cleaned_data.csv",
                            mime="text/csv"
                        )

                    with c2:
                        st.subheader("Data Quality (Cleaned)")
                        cleaned_metrics = quality_metrics(df_cleaned)

                        x, y, z = st.columns(3)
                        x.metric("Rows", cleaned_metrics["shape"][0], delta=cleaned_metrics["shape"][0] - raw_metrics["shape"][0])
                        y.metric("Columns", cleaned_metrics["shape"][1], delta=cleaned_metrics["shape"][1] - raw_metrics["shape"][1])
                        z.metric("Duplicate rows", cleaned_metrics["dup_rows"], delta=cleaned_metrics["dup_rows"] - raw_metrics["dup_rows"])

                        st.metric("Rows >50% missing", cleaned_metrics["row_missing_over_50"], delta=cleaned_metrics["row_missing_over_50"] - raw_metrics["row_missing_over_50"])

                        st.caption("Missingness by column (top 15)")
                        miss2 = cleaned_metrics["missing_pct"].head(15).reset_index()
                        miss2.columns = ["column", "missing_percent"]
                        st.bar_chart(miss2.set_index("column"))

                    if show_code:
                        with st.expander("Show generated cleaning code"):
                            st.code(code or "# No code returned", language="python")

                    if show_code and code:
                        st.download_button(
                            "⬇️ Download cleaning code (.py)",
                            data=code.encode("utf-8"),
                            file_name="data_cleaner_generated.py",
                            mime="text/x-python"
                        )

            except Exception as e:
                st.error("Unexpected error in app execution.")
                if show_debug:
                    st.exception(e)



st.markdown("""
This agent allows you to:

- 📁 **Upload the raw data**
- 🔍 **View the data quality metrics**
- 📝 **Provide custom cleaning instructions**
- 🧹 **Clean the data**
- 🔍 **View the data quality metrics of the cleaned data**
- 📥 **Download the cleaned data**
- 📥 **Download the cleaning code**
- 🔍 **View the debug / error details**
""")


# Optional: add branding or image
# st.image("logo.png", width=200)