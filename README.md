# Web Summarizer Agent (LangChain + FastAPI)

A compact project that demonstrates how to build a polite web‑summarizing agent with LangChain, add short‑term conversation memory, and expose everything behind a simple API.

## What this project delivers

- **Part 1 – Agent + Web integration:** Accepts a URL, loads web content, summarizes it, and answers polite follow‑up questions strictly from that summary.
- **Part 2 – Conversation memory:** Remembers the **last three** messages for follow‑ups (windowed memory).
- **Part 3 – API endpoint:** A FastAPI POST endpoint that returns JSON with both a **summary** and a concise **main topic**, with basic error handling.

---

## Quickstart

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Configure environment

Create a `.env` file (see variables below). Example:
```env
USER_AGENT="myagent"
OPENAI_API_KEY="your_openai_api_key"
```

> **Notes**
> - `USER_AGENT` is used by the web page loader.
> - `OPENAI_API_KEY` is only needed if you switch the LLM to OpenAI in `searchtool.py` (the default uses an Ollama local model).

### 3) Run the API
```bash
uvicorn app:app --reload
```
The server defaults to `http://127.0.0.1:8000`.

---

## API

### `POST /summarize`

**Request body:**
```json
{
  "input": "https://example.com/article"
}
```

**Behavior:**
- If `input` is a **URL**, the service will load the page and return a fresh **summary** and **main_topic**.
- If `input` is **not** a URL and a summary exists from a previous call, it will treat the input as a **follow‑up question** and answer **only** using the cached summary + short memory.
- If no summary exists yet and `input` is not a URL, it returns a 400 asking for a URL first.

**Successful response (URL input):**
```json
{
  "summary": "<polite concise summary>",
  "main_topic": "<short title-cased topic>"
}
```

**Successful response (follow‑up question):**
```json
{
  "question": "Who is the author?",
  "response": "I don't know based on the summary."
}
```

**Error responses:**
- `400` — No context yet (send a URL first).
- `500` — Summarization or follow‑up handler raised an exception.

---

## How the parts map to the code

### Part 1 — LangChain Agent + Web Integration

- **Web loading:** `WebBaseLoader(url)` pulls the page into LangChain documents.
- **Summarization:** `load_summarize_chain(..., chain_type="stuff")` performs a single‑call summary (see “Summarization Chain” below).
- **Agent tool (optional):** A `Tool` wrapping the summarizer is defined and can be used by a LangChain agent (initialized with `initialize_agent`). The API path uses the summarizer directly for clarity.

### Part 2 — Conversation Memory

- Uses `ConversationBufferWindowMemory(k=3)` in two places:
  - A general agent memory (if you use the agent path).
  - A dedicated follow‑up chain memory that tracks the **last three** QA turns and restricts answers to the **cached summary** only.

### Part 3 — FastAPI Endpoint

- `POST /summarize`:
  - If the input **looks like a URL**, it:
    1) loads and summarizes the page,
    2) caches the summary for follow‑ups,
    3) generates a short **topic** from the summary (<= 6 words, title case).
  - Otherwise, it treats the input as a **follow‑up question** and answers from the cached summary + memory window.
  - Includes URL validation and basic exception handling.

---

## Summarization Chain (Methodology)

This project intentionally uses LangChain’s simplest summarization approach to stay transparent and testable.

> **Stuffing is the simplest approach, where all relevant data is packed into the prompt to provide context for the language model. In LangChain, this method is implemented as the StuffDocumentsChain.**
>
> **Pros:**
> - **Single LLM Call:** Requires only one call to the LLM.
> - **Comprehensive Context:** The LLM processes all the data at once, which can be beneficial for generating text.

**Why “stuff” here?**
- Deterministic, low‑latency behavior for small/medium pages.
- Easy to reason about and debug.
- A clean baseline for later upgrades (e.g., map‑reduce chains, chunking, RAG).

---

## Key Modules (Walkthrough)

### `searchtool.py`
- **LLM setup:** Defaults to `ChatOllama(model="phi4-mini:latest", temperature=0.1)`. A commented `ChatOpenAI` alternative is provided.
- **Summarize flow:** `summarize_url(url)` → loads docs via `WebBaseLoader`, runs a `stuff` summarize chain, caches to `latest_summary`, and **clears follow‑up memory** so questions align with the newest context.
- **Topic generation:** `topic_from_summary(summary)` uses a small prompt and post‑processing to return a concise, title‑cased topic (<= 6 words).
- **Follow‑ups:** `answer_followup(question)` runs a strictly scoped prompt:
  - If the answer isn’t in the summary, it **must** reply: `I don't know based on the summary.` and suggest summarizing another URL.
- **Agent hook:** A `Tool` called **Web Summarizer** is defined and `get_agent()` returns a ZERO_SHOT_REACT_DESCRIPTION agent using that tool and a 3‑message memory window.

### `app.py`
- **Endpoint:** `POST /summarize` with body `{"input": "<string>"}`.
- **URL detection:** A robust regex identifies URLs.
- **Branches:**
  1. **URL** → summarize + generate topic.
  2. **Non‑URL with context** → follow‑up answer from cached summary.
  3. **No context yet** → 400 asking for a URL.
- **Logging:** Writes to `log.txt` and console.

### `requirements.txt`
Includes the minimal set to run LangChain with either Ollama or OpenAI backends, FastAPI, Uvicorn, and BeautifulSoup (used by the web loader under the hood).

### Environment & Git Hygiene
- `.env` holds secrets and configuration (e.g., `USER_AGENT`, optional `OPENAI_API_KEY`).
- `.gitignore` ignores `.env` and Python cache directories to keep secrets out of version control.

---

## Usage Examples

### Summarize a page
```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"input": "https://example.com"}'
```

### Ask a follow‑up (after summarizing)
```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"input": "What is the main argument?"}'
```



## Troubleshooting

- **“No context available. Please provide a URL first.”** — You asked a question before summarizing any page; POST a URL first.
- **Ollama not installed / model missing** — Install and start Ollama and pull the required model (or switch to OpenAI).
- **Empty or blocked pages** — Some sites block non‑browser agents; try another URL or set an appropriate `USER_AGENT` in `.env`.



## Environment Setup with a Virtual Environment (venv)

Create an isolated Python environment and install dependencies:

```bash
# Create and activate a virtual environment
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) Copy example env
# cp .env.example .env

# Run the API locally
uvicorn app:app --reload
```

---

## Run with Docker

Make sure you have a `.env` file alongside your project (you can start from `.env.example`). Then build and run the container:

```bash
docker build -t web-summarizer .
docker run --rm -p 8000:8000 --env-file .env web-summarizer
```

This maps the container’s port `8000` to your host’s `8000` and passes environment variables from `.env` into the container.

> **Note:** Ensure your project includes a `Dockerfile` configured to launch Uvicorn. If you need a minimal one, you can use:
> ```dockerfile
> FROM python:3.11-slim
> WORKDIR /app
> COPY requirements.txt .
> RUN pip install --no-cache-dir -r requirements.txt
> COPY . .
> ENV PYTHONUNBUFFERED=1
> EXPOSE 8000
> CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
> ```

---

## Set up Ollama & Phi‑4 Mini (Local LLM)

This project defaults to an Ollama model for fast, private local summaries.

### 1) Install Ollama
- **macOS / Linux (one‑liner):**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- **Windows:** Install from the Ollama website (GUI installer).

After installing, verify:
```bash
ollama --version
```

### 2) Start the Ollama service
On most systems the background service starts automatically. If needed, you can run:
```bash
ollama serve
```

### 3) Pull the Phi‑4 Mini model
Pull the lightweight Phi‑4 Mini model:
```bash
# Primary tag used by this project
ollama pull phi4-mini

# (If your registry uses the alternate tag)
# ollama pull phi4:mini
```

Test it quickly:
```bash
ollama run phi4-mini "Say hello from Phi‑4 Mini"
```

> **Model name in code:** By default the code expects an Ollama model named `phi4-mini` (the `:latest` tag is implied).  
> If your local Ollama uses the `phi4:mini` tag instead, update the model string in `searchtool.py` accordingly.

### 4) Environment variables
Make sure your `.env` contains:
```env
USER_AGENT="myagent"
# Only needed if switching to OpenAI in searchtool.py
# OPENAI_API_KEY="sk-..."
```

Now you can run locally or with Docker as shown above.
