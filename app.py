

import os
import glob
import base64
import pandas as pd
import streamlit as st
import openai
from openai import OpenAI

# --- CONFIGURE OPENAI CLIENT ---
try:
    api_key = "#"
except:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Set OPENAI_API_KEY in secrets.toml or env var.")
        st.stop()
client = OpenAI(api_key=api_key)

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="Ozone Metrics & SHAP", layout="wide")
st.title("🌍 Ozone Evaluation Metrics & SHAP Analysis (GPT-5 Vision)")

# --- LOAD METRICS ---
@st.cache_data
def load_metrics():
    try:
        return pd.read_excel("evaluation_metrics.xlsx")
    except:
        return pd.DataFrame()

metrics = load_metrics()
if metrics.empty:
    st.error("Metrics file not found or empty.")
    st.stop()

# --- LOAD ADDITIONAL EVALUATION METRICS FILE (evaluation_metric.xlsx) ---
@st.cache_data
def load_additional_metrics():
    try:
        return pd.read_excel("evaluation_metrics.xlsx")
    except:
        return pd.DataFrame()

additional_metrics = load_additional_metrics()
if additional_metrics.empty:
    st.warning("Additional evaluation_metric.xlsx file not found or empty.")
    additional_metrics_text = "No additional evaluation metrics data available."
else:
    additional_metrics_text = additional_metrics.to_markdown(index=False)

# --- SIDEBAR: COUNTRY & MODEL ---
countries = sorted(metrics["Country"].unique())
country = st.sidebar.selectbox("Country", countries)
models = sorted(metrics[metrics.Country == country]["Model"].unique())
model = st.sidebar.selectbox("Model", models)

# --- DISPLAY METRICS ---
st.subheader(f"Metrics for {country} / {model}")
df = metrics[(metrics.Country == country) & (metrics.Model == model)]
st.dataframe(df, use_container_width=True)

# --- FIND SHAP SUMMARY IMAGE ---
model_clean = model.replace(" ", "_")
img_path = os.path.join("shap_values", country, model_clean, f"{country}_{model_clean}_summary.png")
if not os.path.exists(img_path):
    st.warning(f"{country}_{model_clean}_summary.png not found.")
    st.stop()

# --- SHOW IMAGE & ANALYZE BUTTON ---
st.subheader("SHAP Summary Plot")
st.image(img_path, use_container_width=True)
if st.button(f"Analyze {os.path.basename(img_path)}"):
    img_bytes = open(img_path, "rb").read()
    data_url = f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"
    prompt_text = (
        f"Imagine yourself as a data scientist and a machine learning engineer. "
        f"Here is the SHAP plot {os.path.basename(img_path)} where the X-axis shows features "
        f"and the Y-axis shows corresponding SHAP values. Rate each feature from 0 to 1 "
        f"and show the percentage contributions of each feature in prediction in a properly structured table. "
        f"Also give a small interpretation of the graph.\n\n"
        f"Additionally, here is the evaluation metrics data from 'evaluation_metric.xlsx':\n Give insights based on this data as well.\n\n"
        f"{additional_metrics_text}"
    )
    resp = client.responses.create(
        model="gpt-5",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": data_url}
            ]
        }],
    )
    analysis = resp.output_text or resp.choices[0].message.content
    st.markdown("**GPT-5 Analysis:**")
    st.markdown(analysis.strip())
    st.markdown("---")
