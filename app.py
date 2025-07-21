import streamlit as st
import pandas as pd
import os
import textwrap
import openai

# --- CONFIGURE PERPLEXITY API ---
client = openai.OpenAI(
    api_key="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",
    base_url="https://api.perplexity.ai"
)

def query_perplexity(prompt, model="sonar-deep-research", system_message="Be helpful, factual, and cite sources if applicable.", max_tokens=4000):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].message.content

# --- DATA LOADERS ---
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

# --- PAGE SETUP ---
st.set_page_config(page_title="🌍 Ozone ML + Perplexity LLM Dashboard", layout="wide")
st.title("🌍 Ozone Analysis using ML & Perplexity LLM")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
country_codes = ['ARG', 'ASM', 'ATA', 'AUS', 'BRA', 'CAN', 'CHN', 'GRL', 'IND', 'MEX', 'NZL', 'USA']
country = st.sidebar.selectbox("🌐 Select Country", country_codes)
mode = st.sidebar.radio("📊 Analysis Mode", ["Single Model", "Compare All Models"])

metrics_df = load_metrics()
pred_df = load_predictions(country)
raw_data_df = load_raw_data(country)

# --- ERROR IF MISSING ---
if pred_df.empty or metrics_df.empty or raw_data_df.empty:
    st.error("❌ Required data not available.")
    st.stop()

# --- FILTER DATA ---
filtered_metrics = metrics_df[metrics_df['Country'] == country]
filtered_preds = pred_df[pred_df['Country'] == country]

if mode == "Single Model":
    available_models = filtered_metrics['Model'].unique().tolist()
    selected_model = st.sidebar.selectbox("🧠 Select Model", available_models)
else:
    selected_model = None

tabs = st.tabs(["📊 Dashboard", "💬 Ask Custom Questions"])

# -------------------------
# 📊 TAB 1: DASHBOARD
# -------------------------
with tabs[0]:
    col1, col2, col3 = st.columns(3)
    col1.metric("Country", country)
    col2.metric("Mode", mode)
    col3.metric("Model", selected_model if selected_model else "All")

    st.subheader("📈 Evaluation Metrics")
    st.dataframe(filtered_metrics, use_container_width=True)

    st.subheader("📊 All Predictions")
    display_preds = (filtered_preds[filtered_preds["Model"] == selected_model] if selected_model else filtered_preds)
    st.dataframe(display_preds, use_container_width=True)

    st.subheader("📄 Full Raw Ozone Data")
    st.dataframe(raw_data_df, use_container_width=True)

    # --- PROMPT GENERATION (LIMITED ROWS TO AVOID 413 ERROR) ---
    def generate_prompt(extra_web_context=""):
        metrics_text = filtered_metrics.to_string(index=False)
        raw_data_text = raw_data_df[['daily_date', 'daily_columno3']].head(15).to_string(index=False)

        if mode == "Single Model" and selected_model:
            pred_text = filtered_preds[filtered_preds['Model'] == selected_model][['Date', 'Actual', 'Predicted', 'Predicted_Tuned']].head(15).to_string(index=False)
            prompt = textwrap.dedent(f"""
                Analyze ozone data and ML predictions for {country} using model: {selected_model}.

                🔹 Sample Raw Ozone Data:
                {raw_data_text}

                🔹 Predictions (from {selected_model}):
                {pred_text}

                🔹 Evaluation Metrics:
                {metrics_text}

                🔹 Extra Web Context (optional):
                {extra_web_context}

                Tasks:
                1. Compare Regional Ozone Levels 
                2. Evaluate the accuracy of predictions (raw and tuned) [Models are trained using normalized data] .
                3. Explain any data transformations or patterns you notice.
                4. Comment on model performance based on metrics.
                5. Give marks in range of 0-1 for explainibility, data transformation, accuracy and predictions in tabular format [of the selected model].
            """)
        else:
            pred_texts = []
            for model_name in filtered_preds['Model'].unique():
                df_model = filtered_preds[filtered_preds['Model'] == model_name][['Date', 'Actual', 'Predicted', 'Predicted_Tuned']].head(10)
                model_text = df_model.to_string(index=False)
                pred_texts.append(f"🔸 Model: {model_name}\n{model_text}")
            all_preds_text = "\n\n".join(pred_texts)

            prompt = textwrap.dedent(f"""
                Perform a comparative analysis of ozone levels in {country} using all available ML models.

                🔹 Sample Raw Ozone Data:
                {raw_data_text}

                🔹 Model Predictions (Sampled):
                {all_preds_text}

                🔹 Evaluation Metrics:
                {metrics_text}

                🔹 Extra Web Context (optional):
                {extra_web_context}

                Tasks:
                1. Analyze ozone concentrations before vs after 1989.
                2. Compare model accuracy - which performed best?
                3. Note data transformations or outliers.
                4. Evaluate how models responded to ozone trends.
                5. Give marks in range of 0-1 for each model based on its explainibility,data transformation,accuracy and predictions in tabular format.[Read the whole data of all models to understand the predicted vs actual values]

            """)
        return prompt

    # --- INTERPRET WITH PERPLEXITY ---
    if st.button("🔍 Interpret with Perplexity LLM"):
        with st.spinner("Generating insights with Perplexity..."):
            try:
                prompt_text = generate_prompt()
                response_text = query_perplexity(prompt_text)
                st.subheader("🤖 Perplexity Response")
                st.markdown(response_text)
            except Exception as e:
                st.error(f"Perplexity request failed: {e}")

# ------------------------------
# 💬 TAB 2: CUSTOM QUESTIONS
# ------------------------------
with tabs[1]:
    st.subheader("💬 Ask Perplexity a Question")
    chat_input = st.text_area("Type your question here (based on ozone data, model accuracy, Montreal Protocol effects, etc.)")

    if st.button("🧠 Get Perplexity Answer"):
        with st.spinner("Thinking..."):
            # Limit data snippet size
            data_snippet = f"""
            📊 Evaluation Metrics:
            {filtered_metrics.to_string(index=False)}

            📈 Prediction Sample:
            {display_preds[['Date', 'Actual', 'Predicted', 'Predicted_Tuned']].head(10).to_string(index=False)}

            📄 Raw Ozone Data Sample:
            {raw_data_df[['daily_date', 'daily_columno3']].head(10).to_string(index=False)}
            """
            try:
                user_question_prompt = f"""
                Using the following ozone dataset and predictions, answer this question:

                Question:
                {chat_input}

                Data:
                {data_snippet}
                """
                reply = query_perplexity(user_question_prompt)
                st.subheader("🤖 Perplexity's Response")
                st.markdown(reply)
            except Exception as e:
                st.error(f"Failed to get answer: {e}")



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
