"""
Simple Data Analysis Crew - A minimal CrewAI + Streamlit app.
Two agents (Analyst + Advisor) analyze uploaded CSV data.
"""

import os

os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Simple Data Analysis Crew", layout="wide")
st.title("Simple Data Analysis Crew")

# ── Sidebar: LLM selection ───────────────────────────────────
with st.sidebar:
    st.header("Settings")

    provider = st.selectbox("LLM Provider", ["Gemini", "Groq", "Ollama"])

    model_options = {
        "Gemini": ["gemini/gemini-2.5-flash"],
        "Groq": ["groq/llama-3.3-70b-versatile", "groq/llama-3.1-8b-instant"],
        "Ollama": ["ollama/llama3.2", "ollama/mistral"],
    }
    model = st.selectbox("Model", model_options[provider])

    if provider == "Ollama":
        ollama_url = st.text_input("Ollama Base URL", "http://localhost:11434")

# ── File upload ──────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    st.caption(f"{df.shape[0]} rows x {df.shape[1]} columns")

# ── Run button ───────────────────────────────────────────────
if uploaded_file and st.button("Analyze Data", type="primary"):

    # Build data summary for agent context
    summary_parts = [
        f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns",
        f"Columns: {', '.join(df.columns.tolist())}",
        f"\nData types:\n{df.dtypes.to_string()}",
        f"\nMissing values:\n{df.isnull().sum().to_string()}",
    ]
    if df.select_dtypes("number").shape[1] > 0:
        summary_parts.append(
            f"\nNumeric stats:\n{df.describe().to_string()}"
        )
    if df.select_dtypes("object").shape[1] > 0:
        for col in df.select_dtypes("object").columns[:5]:
            summary_parts.append(
                f"\nTop values for '{col}':\n{df[col].value_counts().head(5).to_string()}"
            )

    data_context = "\n".join(summary_parts)

    # Create LLM
    llm_kwargs = {"model": model, "temperature": 0.3}
    if provider == "Ollama":
        llm_kwargs["base_url"] = ollama_url
    llm = LLM(**llm_kwargs)

    # ── Agents ───────────────────────────────────────────────
    analyst = Agent(
        role="Data Analyst",
        goal="Analyze the dataset and find key patterns, trends, and anomalies.",
        backstory="You are an experienced data analyst who excels at extracting insights from raw data.",
        llm=llm,
        verbose=True,
    )

    advisor = Agent(
        role="Business Advisor",
        goal="Turn data analysis into clear, actionable business recommendations.",
        backstory="You are a senior business consultant who translates data findings into strategy.",
        llm=llm,
        verbose=True,
    )

    # ── Tasks ────────────────────────────────────────────────
    analysis_task = Task(
        description=(
            f"Analyze this dataset and provide:\n"
            f"1. Key statistics and distributions\n"
            f"2. Notable patterns or correlations\n"
            f"3. Data quality issues\n"
            f"4. Top 5 interesting findings\n\n"
            f"DATA SUMMARY:\n{data_context}"
        ),
        expected_output="A structured data analysis report with statistics, patterns, and findings.",
        agent=analyst,
    )

    insights_task = Task(
        description=(
            "Based on the data analysis above, provide:\n"
            "1. Top 3 actionable business recommendations\n"
            "2. Key risks or concerns found in the data\n"
            "3. Suggested next steps\n"
        ),
        expected_output="A concise business insights report with recommendations and next steps.",
        agent=advisor,
        context=[analysis_task],
    )

    # ── Crew ─────────────────────────────────────────────────
    crew = Crew(
        agents=[analyst, advisor],
        tasks=[analysis_task, insights_task],
        process=Process.sequential,
        verbose=True,
    )

    with st.spinner("Crew is analyzing your data..."):
        result = crew.kickoff()

    # ── Display results ──────────────────────────────────────
    st.divider()
    tab_analysis, tab_insights = st.tabs(["Data Analysis", "Business Insights"])

    with tab_analysis:
        st.markdown(analysis_task.output.raw if analysis_task.output else "No output.")

    with tab_insights:
        st.markdown(insights_task.output.raw if insights_task.output else "No output.")

    st.divider()
    st.subheader("Full Report")
    full_report = result.raw if hasattr(result, "raw") else str(result)
    st.markdown(full_report)
    st.download_button("Download Report", full_report, file_name="analysis_report.md")
