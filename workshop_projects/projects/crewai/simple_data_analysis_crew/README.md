# Simple Data Analysis Crew

A Streamlit app that uses CrewAI multi-agent system to analyze CSV data and provide business insights. Two AI agents — a **Data Analyst** and a **Business Advisor** — collaborate sequentially to find patterns and deliver actionable recommendations.

## Prerequisites

- Python 3.10+
- API key for at least one supported LLM provider:
  - **Gemini** — [Get key](https://aistudio.google.com/apikey)
  - **Groq** — [Get key](https://console.groq.com/keys)
  - **Ollama** (local, no key needed) — [Install Ollama](https://ollama.com/)

## Setup

1. **Create and activate a virtual environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Copy the example env file and add your API key(s):

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set the key for your chosen provider:

   ```
   GEMINI_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   ```

## Run

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Usage

1. Select an **LLM provider** and **model** in the sidebar.
2. Upload a CSV file (a sample `Stocks.csv` is included).
3. Click **Analyze Data**.
4. View results in the **Data Analysis** and **Business Insights** tabs.
5. Download the full report as a markdown file using the download button.

## Supported LLM Providers & Models

| Provider | Models |
|----------|--------|
| Gemini   | `gemini-2.5-flash` |
| Groq     | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` |
| Ollama   | `llama3.2`, `mistral` |

> For Ollama, make sure the Ollama server is running and the model is pulled before use.
