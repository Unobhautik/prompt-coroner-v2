"""
Prompt Coroner — LangGraph Multi-Agent Pipeline
================================================
6 specialized agents in a stateful, conditional graph.
Every LangChain/LangGraph primitive here earns its place.
"""

import os, json, re
from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict, Annotated, Literal
from operator import add

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import SKLearnVectorStore
from sklearn.feature_extraction.text import HashingVectorizer

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# ── LLM (Groq, free) ─────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.environ["GROQ_API_KEY"]
)

# ── Embeddings for similarity search (sklearn hashing — no PyTorch) ─
class SklearnHashEmbeddings(Embeddings):
    """Fixed-size sparse hashing vectors; good enough for in-session similarity."""

    def __init__(self, n_features: int = 256):
        self._hv = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._hv.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._hv.transform([text]).toarray()[0].tolist()


embeddings = SklearnHashEmbeddings()

# ── In-memory vector store (grows as users submit prompts) ─────────
_past_docs: list[Document] = []
_vector_store = None

def _get_vector_store():
    global _vector_store
    if _vector_store is None or not _past_docs:
        return None
    return SKLearnVectorStore.from_documents(_past_docs, embeddings)

def _add_to_history(prompt: str, tags: list[str]):
    """Store this autopsy result for future similarity lookups."""
    doc = Document(
        page_content=prompt,
        metadata={"tags": ", ".join(tags)}
    )
    _past_docs.append(doc)
    global _vector_store
    _vector_store = SKLearnVectorStore.from_documents(_past_docs, embeddings)


# ══════════════════════════════════════════════════════════════════
# SHARED STATE — every agent reads/writes this TypedDict
# ══════════════════════════════════════════════════════════════════
class CorpseState(TypedDict):
    # inputs
    raw_prompt: str
    raw_output: str          # optional bad AI output

    # set by intake_agent
    clean_prompt: str

    # set by triage_agent
    severity: Literal["high", "medium", "low"]
    health_score: int

    # set by classifier_agent (parallel node)
    failure_tags: list[str]
    death_cause: str

    # set by autopsy_agent (parallel node)
    autopsy_rows: list[dict]

    # set by similarity_agent (parallel node)
    similar_cases: list[str]

    # set by deep_dive_agent (conditional — only if severity==high)
    deep_dive_notes: str

    # final output set by synthesizer_agent
    reformulations: list[dict]


# ══════════════════════════════════════════════════════════════════
# HELPER — clean JSON from LLM response
# ══════════════════════════════════════════════════════════════════
def _parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ══════════════════════════════════════════════════════════════════
# NODE 1 — intake_agent
# Cleans and normalises the raw prompt before anything else runs.
# Strips markdown artefacts, trims whitespace, flags empty prompts.
# ══════════════════════════════════════════════════════════════════
INTAKE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the intake clerk at a prompt hospital.
Clean the user's raw prompt:
- Strip markdown code fences and backticks
- Trim extra whitespace
- Preserve the intent entirely — do NOT rewrite it
- Do NOT fix typos, grammar, or spelling — we need the original to analyze
Return ONLY the cleaned prompt text, nothing else."""),
    ("human", "Raw prompt:\n{raw_prompt}")
])

def intake_agent(state: CorpseState) -> dict:
    chain = INTAKE_PROMPT | llm
    result = chain.invoke({"raw_prompt": state["raw_prompt"]})
    return {"clean_prompt": result.content.strip()}


# ══════════════════════════════════════════════════════════════════
# NODE 2 — triage_agent
# Decides severity BEFORE spinning up the expensive parallel nodes.
# This is the conditional router — its output determines the graph path.
# ══════════════════════════════════════════════════════════════════
TRIAGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the triage nurse at a prompt emergency room.
Quickly assess the severity of the USER PROMPT shown below and return ONLY JSON (no markdown):
{{
  "severity": "high" | "medium" | "low",
  "health_score": 0-100,
  "quick_reason": "One sentence max"
}}
high = severely broken, will always produce garbage (score 0-30)
medium = has clear issues but might sometimes work (score 31-70)
low = minor issues, mostly fine (score 71-100)
health_score must move INVERSELY to severity."""),
    ("human", "Prompt to triage:\n{clean_prompt}\n\nBad output received (if any):\n{raw_output}")
])

def triage_agent(state: CorpseState) -> dict:
    chain = TRIAGE_PROMPT | llm
    result = chain.invoke({
        "clean_prompt": state["clean_prompt"],
        "raw_output": state.get("raw_output", "(none provided)")
    })
    data = _parse_json(result.content)
    return {
        "severity": data["severity"],
        "health_score": data.get("health_score", 50),
    }


# ══════════════════════════════════════════════════════════════════
# CONDITIONAL EDGE — decides which parallel nodes to Send() to
# This is the core LangGraph routing — impossible with plain API calls.
# All 3 parallel agents always run, but deep_dive only fires if high.
# ══════════════════════════════════════════════════════════════════
def route_after_triage(state: CorpseState):
    """Fan out to 3 parallel agents simultaneously using Send API."""
    return [
        Send("classifier_agent", state),
        Send("autopsy_agent",    state),
        Send("similarity_agent", state),
    ]


# ══════════════════════════════════════════════════════════════════
# NODE 3 — classifier_agent  [PARALLEL]
# Tags the exact failure modes. Runs concurrently with autopsy + similarity.
# ══════════════════════════════════════════════════════════════════
CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a prompt failure classifier.
Identify all failure modes present in the USER PROMPT shown below.
Return ONLY JSON (no markdown):
{{
  "failure_tags": ["tag1", "tag2"],
  "death_cause": "One punchy sentence on the main reason this prompt fails"
}}
Valid tags: Ambiguous goal, No role specified, Missing context, Too vague,
No output format, Conflicting instructions, No examples given, Too broad,
No constraints, Poor structure, Hallucination bait, No tone specified,
Wrong model assumption, Prompt injection risk, Typos/garbled English, Spelling errors,
Unclear audience, Missing length spec, No call to action"""),
    ("human", "Prompt:\n{clean_prompt}\n\nBad output:\n{raw_output}")
])

def classifier_agent(state: CorpseState) -> dict:
    chain = CLASSIFIER_PROMPT | llm
    result = chain.invoke({
        "clean_prompt": state["clean_prompt"],
        "raw_output": state.get("raw_output", "(none)")
    })
    data = _parse_json(result.content)
    return {
        "failure_tags": data.get("failure_tags", []),
        "death_cause": data.get("death_cause", "Unknown cause")
    }


# ══════════════════════════════════════════════════════════════════
# NODE 4 — autopsy_agent  [PARALLEL]
# Deep forensic analysis — line by line. Runs concurrently.
# ══════════════════════════════════════════════════════════════════
AUTOPSY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a forensic prompt pathologist performing a post-mortem.
Dissect the USER PROMPT shown below and identify up to 5 specific failure points.
For "location", quote the actual broken words or phrases from the user's prompt — NEVER quote template placeholders.
Return ONLY a JSON array (no markdown):
[
  {{"issue": "Short label", "detail": "One precise sentence on why this kills the prompt", "severity": "high"|"medium"|"low", "location": "Phrase from the original prompt that is the problem"}}
]
Be surgical. Quote the actual broken parts of the user's prompt."""),
    ("human", "Prompt under autopsy:\n{clean_prompt}\n\nBad output received:\n{raw_output}")
])

def autopsy_agent(state: CorpseState) -> dict:
    chain = AUTOPSY_PROMPT | llm
    result = chain.invoke({
        "clean_prompt": state["clean_prompt"],
        "raw_output": state.get("raw_output", "(none)")
    })
    return {"autopsy_rows": _parse_json(result.content)}


# ══════════════════════════════════════════════════════════════════
# NODE 5 — similarity_agent  [PARALLEL]
# Searches the vector store for past prompts with similar failure modes.
# This is REAL LangChain retrieval — not a fake "memory" layer.
# Gets smarter the more prompts you run through it.
# ══════════════════════════════════════════════════════════════════
def similarity_agent(state: CorpseState) -> dict:
    vs = _get_vector_store()
    if vs is None:
        return {"similar_cases": ["No past cases yet — this is the first autopsy."]}

    # sklearn's NearestNeighbors errors if k > number of stored docs, so cap it.
    k = max(1, min(3, len(_past_docs)))
    try:
        results = vs.similarity_search(state["clean_prompt"], k=k)
    except Exception as e:
        return {"similar_cases": [f"Similarity search skipped ({type(e).__name__})."]}

    cases = []
    for doc in results:
        tags = doc.metadata.get("tags", "unknown tags")
        snippet = doc.page_content[:120] + ("..." if len(doc.page_content) > 120 else "")
        cases.append(f'"{snippet}" → failed due to: {tags}')
    return {"similar_cases": cases if cases else ["No similar past cases found."]}


# ══════════════════════════════════════════════════════════════════
# MERGE POINT — join parallel results, then decide deep_dive
# ══════════════════════════════════════════════════════════════════
def route_after_parallel(state: CorpseState):
    """Only run deep_dive_agent if triage marked it as HIGH severity."""
    if state.get("severity") == "high":
        return "deep_dive_agent"
    return "synthesizer_agent"


# ══════════════════════════════════════════════════════════════════
# NODE 6 — deep_dive_agent  [CONDITIONAL — only fires on HIGH severity]
# Extra investigation that is expensive/slow — only worth running when broken.
# ══════════════════════════════════════════════════════════════════
DEEP_DIVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior prompt engineer called in for critical cases only.
The triage team flagged this as HIGH severity.
Given the failure tags and autopsy, perform a deeper investigation:
1. What assumptions did the author make that were wrong?
2. What would a naive AI do with this prompt (and why)?
3. What is the single most important fix?
Return a plain text paragraph — no JSON, no lists. Max 4 sentences. Be sharp."""),
    ("human", """Prompt: {clean_prompt}
Failure tags: {failure_tags}
Autopsy summary: {autopsy_rows}""")
])

def deep_dive_agent(state: CorpseState) -> dict:
    chain = DEEP_DIVE_PROMPT | llm
    result = chain.invoke({
        "clean_prompt": state["clean_prompt"],
        "failure_tags": ", ".join(state.get("failure_tags", [])),
        "autopsy_rows": json.dumps(state.get("autopsy_rows", []))
    })
    return {"deep_dive_notes": result.content.strip()}


# ══════════════════════════════════════════════════════════════════
# NODE 7 — synthesizer_agent
# Final node: merges ALL upstream findings into 3 strategic rewrites.
# This is the only agent that sees everything — it synthesizes, not generates.
# ══════════════════════════════════════════════════════════════════
SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the chief prompt surgeon. You receive a full diagnostic report
and produce 3 strategically distinct rewrites of the user's broken prompt below.

CRITICAL RULES:
- Rewrite the USER'S ORIGINAL PROMPT to actually accomplish what THEY were trying to do.
- Each rewrite must be a concrete, ready-to-use prompt — NO placeholders, NO meta-language about "failure_tags" or "autopsy_rows".
- Do NOT include words like "failure_tags", "autopsy_rows", or "similar_cases" in any rewrite — those are internal diagnostics, not part of the user's goal.
- Address the specific failure modes found — not generic improvements.

Return ONLY a JSON array (no markdown):
[
  {{
    "label": "Strategy name (3-5 words)",
    "strategy": "One sentence: what specific failures this addresses and how",
    "prompt": "The full rewritten prompt — concrete, ready to paste into an LLM"
  }},
  {{
    "label": "...",
    "strategy": "...",
    "prompt": "..."
  }},
  {{
    "label": "...",
    "strategy": "...",
    "prompt": "..."
  }}
]

The 3 strategies must be meaningfully different approaches:
- v1: Direct fix (add role + format + constraints)
- v2: Context-first (add background, examples, scope)
- v3: Chain-of-thought (step-by-step reasoning instruction)"""),
    ("human", """Original broken prompt:
{clean_prompt}

Failure tags: {failure_tags}
Autopsy findings: {autopsy_rows}
Similar past cases: {similar_cases}
Deep-dive notes (if any): {deep_dive_notes}""")
])

def synthesizer_agent(state: CorpseState) -> dict:
    chain = SYNTHESIZER_PROMPT | llm
    result = chain.invoke({
        "clean_prompt": state["clean_prompt"],
        "failure_tags": ", ".join(state.get("failure_tags", [])),
        "autopsy_rows": json.dumps(state.get("autopsy_rows", [])),
        "similar_cases": "\n".join(state.get("similar_cases", [])),
        "deep_dive_notes": state.get("deep_dive_notes", "N/A — not a high severity case")
    })
    reformulations = _parse_json(result.content)

    # Store this autopsy in vector memory for future similarity searches
    _add_to_history(state["clean_prompt"], state.get("failure_tags", []))

    return {"reformulations": reformulations}


# ══════════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ══════════════════════════════════════════════════════════════════
def build_graph():
    g = StateGraph(CorpseState)

    # Register all nodes
    g.add_node("intake_agent",      intake_agent)
    g.add_node("triage_agent",      triage_agent)
    g.add_node("classifier_agent",  classifier_agent)
    g.add_node("autopsy_agent",     autopsy_agent)
    g.add_node("similarity_agent",  similarity_agent)
    g.add_node("deep_dive_agent",   deep_dive_agent)
    g.add_node("synthesizer_agent", synthesizer_agent)

    # Linear start
    g.add_edge(START, "intake_agent")
    g.add_edge("intake_agent", "triage_agent")

    # Conditional parallel fan-out using Send API
    g.add_conditional_edges("triage_agent", route_after_triage, ["classifier_agent", "autopsy_agent", "similarity_agent"])

    # Fan-in: all 3 parallel nodes merge into route_after_parallel
    g.add_conditional_edges("classifier_agent", route_after_parallel, ["deep_dive_agent", "synthesizer_agent"])
    g.add_conditional_edges("autopsy_agent",    route_after_parallel, ["deep_dive_agent", "synthesizer_agent"])
    g.add_conditional_edges("similarity_agent", route_after_parallel, ["deep_dive_agent", "synthesizer_agent"])

    # Conditional deep dive only on HIGH severity
    g.add_edge("deep_dive_agent", "synthesizer_agent")

    # End
    g.add_edge("synthesizer_agent", END)

    return g.compile()


# Compiled graph — imported by app.py
coroner_graph = build_graph()


def run_autopsy(prompt: str, bad_output: str = "") -> dict:
    """Entry point called by Flask. Returns the full state."""
    initial_state: CorpseState = {
        "raw_prompt": prompt,
        "raw_output": bad_output,
        "clean_prompt": "",
        "severity": "medium",
        "health_score": 50,
        "failure_tags": [],
        "death_cause": "",
        "autopsy_rows": [],
        "similar_cases": [],
        "deep_dive_notes": "",
        "reformulations": [],
    }
    result = coroner_graph.invoke(initial_state)
    return result
