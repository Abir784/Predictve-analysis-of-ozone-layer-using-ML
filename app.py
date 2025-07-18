import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
from duckduckgo_search import DDGS

# --- Set API Keys ---
GEMINI_API_KEY = "AIzaSyDVwL7MldTQwY7J9oLinOxKsWjvoanJlYk"
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def get_gemini_model():
    return genai.GenerativeModel("gemini-1.5-flash-latest")

model = get_gemini_model()

# --- Load Multi-Country Data ---
@st.cache_data
def load_data():
    metrics_df = pd.read_excel("evaluation_metrics.xlsx", engine='openpyxl')
    actual_pred_df = pd.read_excel("actual_vs_predicted.xlsx", engine='openpyxl')
    return metrics_df, actual_pred_df

metrics_df, actual_pred_df = load_data()

# --- Streamlit App UI ---
st.title("🌍 Ozone Forecast Analysis & Explainable AI Chatbot")

# --- Sidebar Filters for 12 countries ---
countries = sorted(metrics_df['Country'].unique())
country = st.sidebar.selectbox("Select Country", countries)
model_name = st.sidebar.selectbox("Select ML Model", metrics_df[metrics_df['Country'] == country]['Model'].unique())

# --- Filter data based on selection ---
filtered_metrics = metrics_df[(metrics_df['Country'] == country) & (metrics_df['Model'] == model_name)]
filtered_preds = actual_pred_df[(actual_pred_df['Country'] == country) & (actual_pred_df['Model'] == model_name)]

# --- Show evaluation metrics ---
st.subheader(f"📊 Evaluation - {model_name} ({country})")
st.write(filtered_metrics)

# --- Actual vs Predicted Plot ---
st.subheader("📈 Actual vs Predicted Ozone Levels")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(filtered_preds['Index'], filtered_preds['Actual'], label='Actual', color='blue')
ax.plot(filtered_preds['Index'], filtered_preds['Predicted'], label='Predicted', color='orange')
ax.set_xlabel("Time Index")
ax.set_ylabel("Ozone Level")
ax.legend()
st.pyplot(fig)

# --- Create Prompt Function ---
def generate_prompt(extra_web_context=""):
    metrics_text = filtered_metrics.to_string(index=False)
    pred_text = filtered_preds.to_string(index=False)  

    return f"""
Analyze ozone concentration predictions for {country}, using model: {model_name}.
The analysis should consider changes before and after the Montreal Protocol (1989).

Model Evaluation:
{metrics_text}

Sample Prediction Data:
{pred_text}

Additional Context from Web:
{extra_web_context}

Tasks:
1. Compare pre- and post-1989 ozone trends.
2. Assess prediction accuracy.
3. Highlight patterns or anomalies.
4. Discuss Montreal Protocol impact.
"""

# --- Web Search Integration ---
def search_web_context(query, max_results=3):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region='wt-wt', safesearch='Moderate', max_results=max_results):
            results.append(f"{r['title']}\n{r['body']}\n{r['href']}")
    return "\n\n".join(results)

# --- Generate Interpretation ---
if st.button("🧠 Generate Gemini Explanation"):
    with st.spinner("Searching web for more context..."):
        web_context = search_web_context(f"Ozone recovery trends in {country} after Montreal Protocol")
    prompt = generate_prompt(extra_web_context=web_context)
    with st.spinner("Generating explanation with Gemini..."):
        response = model.generate_content(prompt)
        st.markdown("### 🤖 Gemini's Interpretation")
        st.markdown(response.text)

# --- Ask Custom Question ---
st.markdown("## 💬 Ask a Custom Question")
user_query = st.text_input("Ask about ozone, predictions, policy impact, etc.")

if user_query:
    with st.spinner("Fetching context & generating answer..."):
        web_context = search_web_context(user_query + f" in {country}")
        full_prompt = generate_prompt(extra_web_context=web_context) + f"\nUser Question:\n{user_query}"
        response = model.generate_content(full_prompt)
        st.markdown("### 💡 Gemini's Answer")
        st.markdown(response.text)
