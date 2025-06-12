import os
import json
import re
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# —————————————— Load environment & keys ——————————————
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
courtlistener_key = os.getenv("COURTLISTENER_API_KEY")
assert openai_key, "Missing OPENAI_API_KEY in .env"
assert courtlistener_key, "Missing COURTLISTENER_API_KEY in .env"

# Make sure the OpenAI client sees the key
os.environ["OPENAI_API_KEY"] = openai_key

# —————————————— LLM Initialization ——————————————
llm = ChatOpenAI(
    openai_api_key=openai_key,
    temperature=0,
    model="gpt-4o-mini",
    streaming=False
)

# —————————————— State Definition ——————————————
class ParalegalState(BaseModel):
    scenario: str
    search_results: list[dict] = []
    tried_ids: set[str] = set()
    last_op: dict | None = None
    last_summary: dict | None = None
    current_reasoning: str | None = None
    reason_judgment: str | None = None
    precedents: list[dict] = []
    precedents_reasonings: list[str] = []
    n_target: int = 1

# —————————————— Node Implementations ——————————————

def search_node(state: ParalegalState):
    """Search CourtListener for cases matching the scenario."""
    url = "https://www.courtlistener.com/api/rest/v4/opinions/"
    headers = {"Authorization": f"Token {courtlistener_key}"}
    params = {
        "search": state.scenario,
        "page_size": state.n_target * 2,  # fetch a buffer
    }
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    state.search_results = resp.json().get("results", [])
    return state

def fetch_node(state: ParalegalState):
    """Fetch the next-untried case text from search_results."""
    for hit in state.search_results:
        if hit["id"] not in state.tried_ids:
            state.tried_ids.add(hit["id"])
            api_url = f"https://www.courtlistener.com/api/rest/v4/opinions/{hit['id']}/"
            headers = {"Authorization": f"Token {courtlistener_key}"}
            resp = requests.get(api_url, params={"plain_text": 1}, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            state.last_op = {
                "id":    data["id"],
                "cite":  data.get("citation", ""),
                "name":  data.get("caseName", ""),
                "date":  data.get("date_filed", ""),
                # Clip to ~40k chars to stay within token limits
                "text":  data.get("plain_text", "")[:40000],
            }
            break
    return state

def summarize_case(state: ParalegalState):
    """Produce a 4‐section bullet summary from the raw opinion text."""
    prompt = f"""
You are a paralegal. Read the following opinion text and output a plain-text summary
with exactly these headers (and only these headers), each followed by bullet points:

Facts:
- (up to 3 bullets)

Issue:
- (1 bullet)

Holding:
- (1 bullet)

Reasoning:
- (up to 2 bullets)

Opinion Text:
\"\"\"{state.last_op['text']}\"\"\"
"""
    raw = llm.invoke(prompt).content.strip()
    # Split on the headers
    parts = re.split(r'^(Facts|Issue|Holding|Reasoning):', raw, flags=re.MULTILINE)[1:]
    parsed = {}
    for header, body in zip(parts[0::2], parts[1::2]):
        lines = [line.strip() for line in body.strip().splitlines() if line.strip()]
        # Remove any leading '-'
        items = [l.lstrip('-').strip() for l in lines]
        parsed[header] = items
    state.last_summary = parsed
    return state

def actor_node(state: ParalegalState):
    """Decide if the current case is relevant to the scenario."""
    cs = state.last_summary
    prompt = f"""
Scenario: {state.scenario}

Case Summary:
Facts: {'; '.join(cs.get('Facts', []))}
Issue: {'; '.join(cs.get('Issue', []))}
Holding: {'; '.join(cs.get('Holding', []))}
Reasoning: {'; '.join(cs.get('Reasoning', []))}

Question: Is this case relevant to the scenario? Answer "relevant" or "irrelevant"
and give one sentence of reasoning.
"""
    state.current_reasoning = llm.invoke(prompt).content.strip()
    return state

def evaluator_node(state: ParalegalState):
    """Check if the new reasoning is redundant or incorrect."""
    prev = "\n".join(f"- {r}" for r in state.precedents_reasonings) or "(none)"
    cs = state.last_summary
    prompt = f"""
Scenario: {state.scenario}

Case Summary:
Facts: {'; '.join(cs.get('Facts', []))}
Issue: {'; '.join(cs.get('Issue', []))}
Holding: {'; '.join(cs.get('Holding', []))}
Reasoning: {'; '.join(cs.get('Reasoning', []))}

Current reasoning: {state.current_reasoning}

Previous reasonings:
{prev}

Question: Is the current reasoning redundant or incorrect compared to prior reasonings?
Answer "keep" or "reject" and give one sentence of explanation.
"""
    state.reason_judgment = llm.invoke(prompt).content.strip()
    return state

def collect_case(state: ParalegalState):
    """If the evaluator says to keep, store the summary + reasoning."""
    if state.reason_judgment.lower().startswith("keep"):
        state.precedents.append(state.last_summary.copy())
        state.precedents_reasonings.append(state.current_reasoning)
    return state

# —————————————— Build & Run the Graph ——————————————
graph = StateGraph(ParalegalState)
graph.set_entry_point("search")

# Add the nodes
graph.add_node("search",    search_node)
graph.add_node("fetch",     fetch_node)
graph.add_node("summarize", summarize_case)
graph.add_node("actor",     actor_node)
graph.add_node("evaluator", evaluator_node)
graph.add_node("collect",   collect_case)

# Straightthrough flow
graph.add_edge("search",    "fetch")
graph.add_edge("fetch",     "summarize")
graph.add_edge("summarize", "actor")
graph.add_edge("actor",     "evaluator")

# If evaluator says "keep" → collect, else → search
graph.add_conditional_edges(
    source="evaluator",
    path=lambda s: s.reason_judgment.lower().startswith("keep"),
    path_map={True: "collect", False: "search"},
)

# After collecting, stop when we've hit n_target, else loop back
graph.add_conditional_edges(
    source="collect",
    path=lambda s: len(s.precedents) >= s.n_target,
    path_map={True: END, False: "search"},
)

compiled = graph.compile()

# Invoke with your scenario and desired number of precedents
result = compiled.invoke({
    "scenario": "Slip-and-fall in a grocery store parking lot at night",
    "n_target": 3
})

# Print out the found precedents + reasoning
# Inspect what keys we actually got back
print("Result keys:", list(result.keys()))

# Then extract by key
precedents = result["precedents"]
reasonings = result["precedents_reasonings"]

print("\nFound precedents:")
for i, (case, reasoning) in enumerate(zip(precedents, reasonings), start=1):
    print(f"\nCase {i}: {case['name']} ({case['cite']}, {case['date']})")
    print("Reasoning:", reasoning)

