# 🔬 Prompt Coroner v2 — LangGraph Multi-Agent Edition

> Paste a broken prompt → 6 specialized AI agents diagnose it in a stateful graph → get a forensic autopsy + 3 strategic rewrites. Powered by **Groq (free)** + **LangGraph** + **LangChain**.

---

## ⚡ Quick Start (2 minutes)

### Step 1 — Get your FREE Groq API key
1. Go to **https://console.groq.com**
2. Sign up (free, no credit card needed)
3. Click **API Keys → Create API Key** → Copy it

### Step 2 — Run

**Mac / Linux:**
```bash
bash run.sh
```

**Windows:**
```
Double-click run.bat
```

> First run downloads `sentence-transformers` (~90MB). Takes ~2 min once. Instant after that.

### Step 3 — Open browser
```
http://localhost:5000
```

---

## 🧠 Why LangGraph? (The real answer)

Most projects slap LangChain on top of 3 API calls and call it "multi-agent." That's fake.

Here's why **every** LangGraph primitive in this project is genuinely necessary:

### 1. `StateGraph` — Shared typed state across all agents
```python
class CorpseState(TypedDict):
    raw_prompt: str
    severity: Literal["high", "medium", "low"]
    failure_tags: list[str]
    autopsy_rows: list[dict]
    similar_cases: list[str]
    deep_dive_notes: str
    reformulations: list[dict]
```
Every agent reads from and writes back to this shared state object.
The synthesizer at the end sees ALL findings from ALL agents — that's only
possible because LangGraph maintains state across the entire graph run.

### 2. `Send` API — True parallel fan-out
```python
def route_after_triage(state):
    return [
        Send("classifier_agent",  state),
        Send("autopsy_agent",     state),
        Send("similarity_agent",  state),
    ]
```
These 3 agents fire simultaneously, not sequentially.
LangChain's `RunnableParallel` can't do this with conditional routing after.
LangGraph's `Send` API is the only clean way to express this.

### 3. Conditional edges — The deep-dive agent only fires when needed
```python
def route_after_parallel(state):
    if state.get("severity") == "high":
        return "deep_dive_agent"   # extra senior investigation
    return "synthesizer_agent"     # skip straight to rewrites
```
This is a real conditional branch in the execution graph.
A "high severity" prompt triggers a 4th parallel-merged investigation.
A "low severity" prompt skips it entirely. Can't do this with plain API calls.

### 4. LangChain `ChatPromptTemplate` — Structured, reusable prompts
```python
AUTOPSY_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content="..."),
    HumanMessage(content="Prompt: {clean_prompt}\nOutput: {raw_output}")
])
chain = AUTOPSY_PROMPT | llm
```
Each agent has a typed, testable prompt template.
The `|` pipe operator chains prompt → LLM → parser cleanly.

### 5. LangChain Vector Store — Real retrieval memory
```python
_vector_store = SKLearnVectorStore.from_documents(_past_docs, embeddings)
results = vs.similarity_search(state["clean_prompt"], k=3)
```
Every prompt you analyze gets embedded and stored.
Future prompts retrieve semantically similar past cases.
The system gets **smarter the more you use it** — this is real RAG, not fake memory.

---

## 📁 Project Structure

```
prompt-coroner-v2/
├── graph.py          ← LangGraph StateGraph with 6 agent nodes
├── app.py            ← Flask server (thin wrapper around graph.py)
├── static/
│   └── index.html    ← Full UI with agent pipeline tracker
├── requirements.txt
├── run.sh            ← Mac/Linux launcher
├── run.bat           ← Windows launcher
└── README.md
```

---

## 🔄 Agent Pipeline

```
START
  └─► intake_agent        Cleans & normalises the prompt
        └─► triage_agent  Scores severity (high/med/low), routes the graph
              ├─► classifier_agent  [PARALLEL] Tags failure modes
              ├─► autopsy_agent     [PARALLEL] Line-by-line root cause
              └─► similarity_agent  [PARALLEL] Finds similar past prompts (RAG)
                    └─► deep_dive_agent  [CONDITIONAL — HIGH severity only]
                          └─► synthesizer_agent  Merges all findings → 3 rewrites
                                └─► END → Flask UI
```

---

## 💸 Cost

| Component | Cost |
|---|---|
| Groq API (llama-3.3-70b) | **Free** — 14,400 req/day |
| LangGraph / LangChain | **Free** — open source |
| sentence-transformers (embeddings) | **Free** — runs locally |
| Flask | **Free** |
| **Total** | **₹0** |

---

## 📝 Resume bullet point

> "Built a multi-agent LLM pipeline using LangGraph StateGraph with conditional routing,
> parallel agent fan-out via the Send API, and LangChain RAG retrieval —
> enabling forensic prompt failure analysis with severity-conditional deep-dive reasoning."
