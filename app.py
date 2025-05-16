import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

GROQ_API_KEY = "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"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def tabular_summary(df):
    summary_data = []
    for col in df.columns:
        col_data = df[col]
        missing = col_data.isnull().sum()
        dtype = str(col_data.dtype)

        if pd.api.types.is_numeric_dtype(col_data):
            min_val = col_data.min()
            max_val = col_data.max()
            mean_val = col_data.mean()
            top_vals = "N/A"
        else:
            min_val = max_val = mean_val = "N/A"
            top_vals_series = col_data.value_counts(dropna=True).head(3)
            top_vals = ", ".join([f"{idx} ({cnt})" for idx, cnt in zip(top_vals_series.index, top_vals_series.values)])

        summary_data.append({
            "Column": col,
            "Data Type": dtype,
            "Missing Values": missing,
            "Min": min_val,
            "Max": max_val,
            "Mean": mean_val,
            "Top Values": top_vals
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df

def generate_summary_text(df):
    n_rows, n_cols = df.shape
    summary = [f"This dataset has {n_rows} rows and {n_cols} columns."]
    for col in df.columns:
        col_data = df[col]
        missing = col_data.isnull().sum()
        col_type = "numbers" if pd.api.types.is_numeric_dtype(col_data) else "words or labels"
        if pd.api.types.is_numeric_dtype(col_data):
            min_val = col_data.min()
            max_val = col_data.max()
            avg_val = col_data.mean()
            summary.append(
                f"'{col}' mostly has {col_type}. Smallest value is about {min_val:.2f}, biggest is around {max_val:.2f}, average near {avg_val:.2f}. Missing spots: {missing}."
            )
        else:
            top_vals = col_data.value_counts(dropna=True).head(3)
            top_vals_str = ", ".join([f"'{v}' appears {c} times" for v, c in zip(top_vals.index, top_vals.values)])
            summary.append(
                f"'{col}' contains {col_type}. Most common are: {top_vals_str}. Missing spots: {missing}."
            )
    return "\n".join(summary)

st.title("⚡ Groq AI Dataset Summarizer & Query")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

df = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ File loaded successfully!")
        st.subheader("Preview of your dataset:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = None

if df is not None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Summary Table"):
            summary_df = tabular_summary(df)
            st.subheader("📊 Dataset Summary Table")
            st.dataframe(summary_df)

    with col2:
        query = st.text_area("Ask anything about your data:")
        if st.button("Ask AI") and query.strip():
            st.subheader("🧠 AI Answer:")
            # Generate summary text on demand when the AI is asked
            summary_text = generate_summary_text(df)
            
            prompt = f"""
You are a helpful data analyst AI.

Here is the summary of the dataset:

{summary_text}

Based on this summary, please answer the following question:

{query}

Provide a clear, concise, and insightful answer suitable for a non-technical person.
"""
            payload = {
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 512,
            }

            try:
                with st.spinner("Getting answer from Groq..."):
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    output = response.json()["choices"][0]["message"]["content"]
                    st.markdown(output)
            except Exception as e:
                st.error(f"API error: {e}")

else:
    st.info("Upload a CSV or Excel file to get started.")
