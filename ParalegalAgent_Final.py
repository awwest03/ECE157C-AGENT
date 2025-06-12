import os
import re
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ————— Load environment & keys —————
load_dotenv()
openai_key        = os.getenv("OPENAI_API_KEY")
courtlistener_key = os.getenv("COURTLISTENER_API_KEY")
assert openai_key,        "Missing OPENAI_API_KEY in .env"
assert courtlistener_key, "Missing COURTLISTENER_API_KEY in .env"
os.environ["OPENAI_API_KEY"] = openai_key

# ————— LLM Initialization —————
llm = ChatOpenAI(
    openai_api_key=openai_key,
    temperature=0,
    model="gpt-4o-mini",
    streaming=False,
)

# ————— State Definition —————
class ParalegalState(BaseModel):
    scenario: str
    search_results: list[dict]        = []
    tried_ids: set[str]              = set()
    last_op: dict | None             = None
    last_summary: dict | None        = None
    current_reasoning: str | None    = None
    reason_judgment: str | None      = None
    precedents: list[dict]           = []
    precedents_reasonings: list[str] = []
    n_target: int                    = 1

# ————— Node: search (compress + API call) —————
def search_node(state: ParalegalState):
    # 1) Compress scenario into 3–7 words
    prompt = f"""
You are a legal search assistant. Given this scenario, output only a short search query
(3–7 words, letters and numbers only, no punctuation) suitable for input to a case-law API.

Scenario:
{state.scenario}

Output example:
rear end collision negligence
"""
    raw_q = llm.invoke(prompt).content.strip()
    # keep only word characters and take first 7 words
    words = re.findall(r"\w+", raw_q)
    q = " ".join(words[:7])
    print("🔍 Searching for query:", q)

    # 2) Call CourtListener API with minimal parameters
    url     = "https://www.courtlistener.com/api/rest/v4/opinions/"
    headers = {"Authorization": f"Token {courtlistener_key}"}
    params  = {
        "search":    q,
        "page_size": state.n_target * 5,
        "order":     "search_score",     # ← sort by relevance score
    }
    resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    state.search_results = resp.json().get("results", [])
    print(f"🔍 Retrieved {len(state.search_results)} hits")
    return state

# ————— Node: fetch —————
def fetch_node(state: ParalegalState):
    base = "https://www.courtlistener.com"
    for hit in state.search_results:
        cid = hit["id"]
        if cid in state.tried_ids:
            continue
        state.tried_ids.add(cid)

        # fetch opinion JSON
        opinion_url = f"{base}/api/rest/v4/opinions/{cid}/"
        resp = requests.get(
            opinion_url,
            params={"plain_text": 1},
            headers={"Authorization": f"Token {courtlistener_key}"},
            timeout=20
        )
        resp.raise_for_status()
        data = resp.json()

        # fetch cluster metadata
        clu = data["cluster"]
        if clu.startswith("/"):
            clu = base + clu
        cluster = requests.get(
            clu,
            headers={"Authorization": f"Token {courtlistener_key}"},
            timeout=10
        ).json()

        # extract metadata
        name = (
            cluster.get("case_name")
            or cluster.get("case_name_full")
            or cluster.get("case_name_short")
            or ""
        )
        citations = cluster.get("citations", [])
        cite = citations[0] if isinstance(citations, list) and citations else ""
        date = cluster.get("date_filed", "")

        state.last_op = {
            "id":   cid,
            "name": name,
            "cite": cite,
            "date": date,
            "text": data.get("plain_text", "")[:40000],
        }
        print(f"Fetched: {name} ({cite}, {date})")
        break
    return state

# ————— Node: summarize —————
def summarize_case(state: ParalegalState):
    prompt = f"""
You are a paralegal. Read the opinion text and output exactly two headers,
each followed by one bullet point. No other text.

Issue:
- 

CaseReasoning:
- 

Opinion Text:
\"\"\"{state.last_op['text']}\"\"\"
"""
    raw = llm.invoke(prompt).content.strip()
    parts = re.split(r'^(Issue|CaseReasoning):', raw, flags=re.MULTILINE)[1:]
    parsed = {}
    for header, body in zip(parts[0::2], parts[1::2]):
        line = next((l.lstrip('-').strip() for l in body.splitlines() if l.strip()), "")
        parsed[header] = line
    state.last_summary = parsed
    return state

# ————— Node: actor —————
def actor_node(state: ParalegalState):
    cs = state.last_summary
    prompt = f"""
Scenario: {state.scenario}

Issue: {cs['Issue']}
CaseReasoning: {cs['CaseReasoning']}

Question: Is this case relevant? Answer "relevant" or "irrelevant" and give one sentence of reasoning.
"""
    state.current_reasoning = llm.invoke(prompt).content.strip()
    return state

# ————— Node: evaluator —————
def evaluator_node(state: ParalegalState):
    if state.current_reasoning.lower().startswith("irrelevant"):
        state.reason_judgment = "reject: not relevant"
        return state

    prev = "\n".join(f"- {r}" for r in state.precedents_reasonings) or "(none)"
    cs   = state.last_summary
    prompt = f"""
Scenario: {state.scenario}

Issue: {cs['Issue']}
CaseReasoning: {cs['CaseReasoning']}

Current reasoning: {state.current_reasoning}

Previous reasonings:
{prev}

Question: Is the current reasoning redundant or incorrect? Answer "keep" or "reject" and give one sentence explanation.
"""
    state.reason_judgment = llm.invoke(prompt).content.strip()
    return state

# ————— Node: collect —————
def collect_case(state: ParalegalState):
    if state.reason_judgment.lower().startswith("keep"):
        entry = {
            "name":          state.last_op["name"],
            "cite":          state.last_op["cite"],
            "date":          state.last_op["date"],
            "Issue":         state.last_summary["Issue"],
            "CaseReasoning": state.last_summary["CaseReasoning"],
            "Relevance":     state.current_reasoning,
        }
        state.precedents.append(entry)
        state.precedents_reasonings.append(state.current_reasoning)
    return state

# ————— Build & Run the Graph —————
graph = StateGraph(ParalegalState)
graph.set_entry_point("search")

graph.add_node("search",    search_node)
graph.add_node("fetch",     fetch_node)
graph.add_node("summarize", summarize_case)
graph.add_node("actor",     actor_node)
graph.add_node("evaluator", evaluator_node)
graph.add_node("collect",   collect_case)

graph.add_edge("search",    "fetch")
graph.add_edge("fetch",     "summarize")
graph.add_edge("summarize", "actor")
graph.add_edge("actor",     "evaluator")

graph.add_conditional_edges(
    source="evaluator",
    path=lambda s: s.reason_judgment.lower().startswith("keep"),
    path_map={True: "collect", False: "fetch"},
)
graph.add_conditional_edges(
    source="collect",
    path=lambda s: len(s.precedents) >= s.n_target,
    path_map={True: END, False: "fetch"},
)

compiled = graph.compile()
result   = compiled.invoke({
    "scenario": "Man sues another man for destruction of shared property. Other man promises to pay, but doesn't",
    "n_target": 1
})

# ————— Print the selected precedent —————
prec = result["precedents"][0]
print(f"\nCase: {prec['name']} ({prec['cite']}, {prec['date']})")
print(f"Issue: {prec['Issue']}")
print(f"Court's Reasoning: {prec['CaseReasoning']}")
print(f"Relevance Explanation: {prec['Relevance']}")


