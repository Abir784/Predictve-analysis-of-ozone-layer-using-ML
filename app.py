import streamlit as st
import pandas as pd
import os
import textwrap
import google.generativeai as genai

# --- Configure Gemini API ---
genai.configure(api_key="AIzaSyDVwL7MldTQwY7J9oLinOxKsWjvoanJlYk")  # Replace with your real key

model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- Load Functions ---
@st.cache_data
def load_metrics():
    return pd.read_excel("evaluation_metrics.xlsx")

@st.cache_data
def load_predictions(country):
    file_path = os.path.join("country_predictions", f"{country}_actual_vs_predicted.xlsx")
    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    return pd.DataFrame()

@st.cache_data
def load_raw_data(country_code):
    file_path = os.path.join("combined_dataset", f"totalozone_{country_code}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

# --- Page Setup ---
st.set_page_config(page_title="🌍 Ozone ML + Gemini LLM Dashboard", layout="wide")

st.title("🌍 Ozone Analysis using ML & Gemini LLM")

# --- Sidebar: Country & Model Selection ---
st.sidebar.header("⚙️ Configuration")
country_codes = ['ARG', 'ASM', 'ATA', 'AUS', 'BRA', 'CAN', 'CHN', 'GRL', 'IND', 'MEX', 'NZL', 'USA']
country = st.sidebar.selectbox("🌐 Select Country", country_codes)
mode = st.sidebar.radio("📊 Analysis Mode", ["Single Model", "Compare All Models"])

metrics_df = load_metrics()
pred_df = load_predictions(country)
raw_data_df = load_raw_data(country)

# --- Handle No Data ---
if pred_df.empty or metrics_df.empty:
    st.error("❌ Data not available for the selected country.")
    st.stop()

# --- Filter Data ---
filtered_metrics = metrics_df[metrics_df['Country'] == country]
filtered_preds = pred_df[pred_df['Country'] == country]

# --- Optional Model Selector ---
if mode == "Single Model":
    available_models = filtered_metrics['Model'].unique().tolist()
    selected_model = st.sidebar.selectbox("🧠 Select Model", available_models)
else:
    selected_model = None

# --- Tabs ---
tabs = st.tabs(["📊 Dashboard", "💬 Ask Custom Questions"])

# --------------------
# 📊 Tab 1: Dashboard
# --------------------
with tabs[0]:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Selected Country", country)
    with col2:
        st.metric("Mode", mode)
    with col3:
        if mode == "Single Model":
            st.metric("Model", selected_model)

    st.subheader("📈 Evaluation Metrics")
    st.dataframe(filtered_metrics, use_container_width=True)

    st.subheader("📊 Sample Predictions (first 100 rows)")
    if mode == "Single Model" and selected_model:
        display_preds = filtered_preds[filtered_preds["Model"] == selected_model]
    else:
        display_preds = filtered_preds
    st.dataframe(display_preds.head(100), use_container_width=True)

    st.subheader("📄 Raw Ozone Data (first 100 rows)")
    st.dataframe(raw_data_df.head(100), use_container_width=True)

    # --- Prompt Generation ---
    def generate_prompt(extra_web_context=""):
        metrics_text = filtered_metrics.to_string(index=False)
        raw_sample = raw_data_df.head(100).to_string(index=False)

        if mode == "Single Model" and selected_model:
            pred_text = filtered_preds[filtered_preds['Model'] == selected_model][['Date', 'Actual', 'Predicted', 'Predicted_Tuned']].head(100).to_string(index=False)
            prompt = textwrap.dedent(f"""
                Analyze ozone concentration data and ML predictions for {country} using model: {selected_model}.

                🔹 Raw Data Sample (first 100 rows):
                {raw_sample}

                🔹 Predictions from {selected_model}:
                {pred_text}

                🔹 Evaluation Metrics:
                {metrics_text}

                🔹 Extra Context from Web:
                {extra_web_context}

                Tasks:
                1. Compare pre- and post-1989 ozone trends.
                2. Analyze prediction accuracy including tuned predictions.
                3. Explain how raw data has been transformed.
                4. Discuss Montreal Protocol's effect.
            """)
        else:
            pred_texts = []
            for m in filtered_preds['Model'].unique():
                df_model = filtered_preds[filtered_preds['Model'] == m]
                pred_sample = df_model[['Date', 'Actual', 'Predicted', 'Predicted_Tuned']].head(100).to_string(index=False)
                pred_texts.append(f"🔸 Model: {m}\n{pred_sample}")
            all_preds_text = "\n\n".join(pred_texts)

            prompt = textwrap.dedent(f"""
                Perform a comparative analysis of ozone data for {country} using multiple machine learning models.

                🔹 Raw Ozone Data (first 100 rows):
                {raw_sample}

                🔹 Prediction Samples:
                {all_preds_text}

                🔹 Evaluation Metrics:
                {metrics_text}

                🔹 Extra Context from Web:
                {extra_web_context}

                Tasks:
                1. Evaluate how ozone levels have changed pre- and post-1989 (Montreal Protocol).
                2. Compare model performances with and without tuning.
                3. Explain data preprocessing steps for each model.
                4. Identify the best performing model and justify.
            """)
        return prompt

    # --- Button to Run Gemini Analysis ---
    if st.button("🔍 Interpret with Gemini"):
        with st.spinner("Generating insights..."):
            prompt_text = generate_prompt()
            try:
                response = model.generate_content(prompt_text)
                st.subheader("🤖 Gemini LLM Interpretation")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"LLM generation failed: {e}")

# -----------------------------
# 💬 Tab 2: Ask Custom Questions
# -----------------------------
with tabs[1]:
    st.subheader("💬 Ask a Custom Question")
    chat_input = st.text_area("Type your question (e.g., how did ozone change after 1989?)")

    if st.button("🧠 Get Answer"):
        with st.spinner("Thinking..."):
            combined_info = ""

            if not pred_df.empty and not metrics_df.empty:
                combined_info = f"""
                Evaluation Metrics:\n{metrics_df.head(50).to_string(index=False)}\n\n
                Sample Predictions:\n{pred_df.head(50).to_string(index=False)}\n\n
                Raw Ozone Data:\n{raw_data_df.head(50).to_string(index=False)}
                """
            try:
                chat_prompt = f"""Answer the following user question based on ozone data and predictions:\n\nQuestion: {chat_input}\n\nAvailable Data:\n{combined_info}"""
                reply = model.generate_content(chat_prompt)
                st.subheader("🤖 Gemini's Response")
                st.markdown(reply.text)
            except Exception as e:
                st.error(f"Failed to generate response: {e}")




# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import requests

# HUGGINGFACE_API_TOKEN = "your_hf_api_key"  # 🔐 Replace with your actual key
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # Or any other instruct model

# API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
# HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

# # --- Streamlit Title ---
# st.title("🌍 Ozone Forecast Analysis & LLM Interpretation Dashboard (via Hugging Face API)")

# # --- Load Data ---
# @st.cache_data
# def load_data():
#     metrics_df = pd.read_excel("evaluation_metrics.xlsx", engine='openpyxl')
#     actual_pred_df = pd.read_excel("actual_vs_predicted.xlsx", engine='openpyxl')
#     return metrics_df, actual_pred_df

# metrics_df, actual_pred_df = load_data()

# # --- Sidebar ---
# country = st.sidebar.selectbox("Select Country", metrics_df['Country'].unique())
# model = st.sidebar.selectbox("Select Model", metrics_df[metrics_df['Country'] == country]['Model'].unique())

# # --- Filtered Data ---
# filtered_metrics = metrics_df[(metrics_df['Country'] == country) & (metrics_df['Model'] == model)]
# filtered_preds = actual_pred_df[(actual_pred_df['Country'] == country) & (actual_pred_df['Model'] == model)]

# # --- Evaluation Display ---
# st.subheader(f"📊 Model Evaluation - {model} ({country})")
# st.write(filtered_metrics)

# # --- Plotting ---
# st.subheader("📈 Actual vs Predicted Ozone Values")
# fig, ax = plt.subplots(figsize=(10, 4))
# ax.plot(filtered_preds['Index'], filtered_preds['Actual'], label='Actual', color='blue')
# ax.plot(filtered_preds['Index'], filtered_preds['Predicted'], label='Predicted', color='orange')
# ax.set_xlabel("Time Index")
# ax.set_ylabel("Ozone Level")
# ax.legend()
# st.pyplot(fig)

# # --- Prompt Builder ---
# def build_context_prompt(user_question):
#     metrics_text = filtered_metrics.to_string(index=False)
#     pred_text = filtered_preds.head(10).to_string(index=False)
#     return f"""
# You are an expert assistant analyzing ozone predictions using machine learning.

# Country: {country}
# Model: {model}
# Montreal Protocol (1989) is a key year.

# Evaluation metrics:
# {metrics_text}

# Sample actual vs predicted ozone values:
# {pred_text}

# User question:
# {user_question}

# Answer:"""

# # --- API Call Function ---
# def query_huggingface_api(prompt):
#     payload = {
#         "inputs": prompt,
#         "parameters": {"max_new_tokens": 300, "temperature": 0.7}
#     }
#     response = requests.post(API_URL, headers=HEADERS, json=payload)
#     return response.json()

# # --- Chat Section ---
# st.subheader("💬 Ask Questions About the Predictions")
# user_question = st.text_input("Ask the AI about the prediction data:")

# if st.button("📡 Ask AI via Hugging Face API"):
#     if user_question.strip():
#         with st.spinner("Querying Hugging Face Model..."):
#             prompt = build_context_prompt(user_question)
#             result = query_huggingface_api(prompt)
#             try:
#                 answer = result[0]["generated_text"].replace(prompt, "").strip()
#                 st.markdown("### 🤖 AI Answer")
#                 st.markdown(answer)
#             except Exception as e:
#                 st.error(f"Error from Hugging Face API: {result}")
#     else:
#         st.warning("Please enter a question.")



















# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from openai import OpenAI
# from dotenv import load_dotenv

# # --- Load environment variables ---
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# # --- OpenAI client ---
# client = OpenAI(api_key=api_key)

# # --- Streamlit App Title ---
# st.title("🌍 Ozone Forecast Analysis & LLM Interpretation Dashboard")

# # --- Load Data ---
# @st.cache_data
# def load_data():
#     metrics_df = pd.read_excel("evaluation_metrics.xlsx", engine='openpyxl')
#     actual_pred_df = pd.read_excel("actual_vs_predicted.xlsx", engine='openpyxl')
#     return metrics_df, actual_pred_df

# metrics_df, actual_pred_df = load_data()

# # --- Sidebar Filters ---
# country = st.sidebar.selectbox("Select Country", metrics_df['Country'].unique())
# model = st.sidebar.selectbox("Select Model", metrics_df[metrics_df['Country'] == country]['Model'].unique())

# # --- Filtered Data ---
# filtered_metrics = metrics_df[(metrics_df['Country'] == country) & (metrics_df['Model'] == model)]
# filtered_preds = actual_pred_df[(actual_pred_df['Country'] == country) & (actual_pred_df['Model'] == model)]

# # --- Show Evaluation Metrics ---
# st.subheader(f"📊 Model Evaluation - {model} ({country})")
# st.write(filtered_metrics)

# # --- Plot Actual vs Predicted ---
# st.subheader("📈 Actual vs Predicted Ozone Values")
# fig, ax = plt.subplots(figsize=(10, 4))
# ax.plot(filtered_preds['Index'], filtered_preds['Actual'], label='Actual', color='blue')
# ax.plot(filtered_preds['Index'], filtered_preds['Predicted'], label='Predicted', color='orange')
# ax.set_xlabel("Time Index")
# ax.set_ylabel("Ozone Level")
# ax.legend()
# st.pyplot(fig)

# # --- Prompt Generator ---
# def generate_prompt(country, model, metrics_df, pred_df):
#     metrics_text = metrics_df.to_string(index=False)
#     pred_text = pred_df.head(10).to_string(index=False)
#     return f"""
#     Analyze ozone concentration prediction performance for {country}, modeled using {model}.
#     The data spans from before to after the Montreal Protocol (1989).

#     Model evaluation metrics:
#     {metrics_text}

#     Sample actual vs predicted values:
#     {pred_text}

#     Tasks:
#     1. Comment on how ozone levels have changed before vs after 1989.
#     2. Evaluate model performance and comment on accuracy.
#     3. Identify potential recovery patterns or anomalies.
#     4. Assess Montreal Protocol's impact based on observed trends.
#     """

# # --- LLM Interpretation Button ---
# if st.button("🧠 Generate LLM Interpretation"):
#     with st.spinner("Analyzing with GPT-4..."):
#         try:
#             prompt = generate_prompt(country, model, filtered_metrics, filtered_preds)

#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[{"role": "user", "content": prompt}]
#             )

#             interpretation = response.choices[0].message.content
#             st.markdown("### 🤖 LLM Interpretation")
#             st.markdown(interpretation)

#         except Exception as e:
#             st.error(f"OpenAI API Error: {e}")
