import os, getpass # API CALL

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "default"
os.environ["USER_AGENT"] = "myagent"

# SEMANTIC SCHOLAR TOOL
from langchain.tools import Tool 
import requests

def search_semantic_scholar(query: str, limit: int = 5):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,authors,year,venue,url"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()["data"]
    else:
        return f"Error: {response.status_code}, {response.text}"

semantic_scholar_tool = Tool(
    name="SemanticScholarSearch",
    func=lambda q: search_semantic_scholar(q, limit=5),
    description="Searches Semantic Scholar for recent papers related to a topic"
)
# AGENT INITIALIZATION
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)

agent = initialize_agent(
    tools=[semantic_scholar_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
# RUN THE AGENT
research_topic = "microLED optoelectronics recent advances"
agent.run(f"Find recent papers about {research_topic}. Summarize each and tell me who the authors are, what the key findings are, how they are moving the field foward, and comment on the novelty of the work.")


# STATE INITIALIZATION
state = {
    "topic": "...",               # USER INPUT
    "papers": [],                 # RAW SS RESULTS
    "summaries": [],              # SUMMARIES PER PAPER
    "agreements": [],             # EXTRACTED COMMON FINDINGS
    "scores": [],                 # EVALUATION OF EACH PAPER
    "global_summary": "..."       # CUMULATIVE SUMMARY
}


# FETCH PAPER'S NODE
def fetch_papers_node(state):
    papers = search_semantic_scholar(state["topic"], limit=5)
    state["papers"] = papers
    return state

# SUMMARIZE PAPER'S NODE
def summarize_papers_node(state):
    from langchain.prompts import PromptTemplate
    summaries = []
    for paper in state["papers"]:
        prompt = f"Summarize this paper:\nTitle: {paper['title']}\nAbstract: {paper['abstract']}"
        result = agent.run(prompt)
        summaries.append({"title": paper["title"], "summary": result})
    state["summaries"] = summaries
    return state

# COMPARE PAPER'S NODE
def compare_papers_node(state):
    summaries_text = "\n\n".join([s["summary"] for s in state["summaries"]])
    compare_prompt = f"Compare the following summaries. What are the common points and disagreements?\n\n{summaries_text}"
    result = agent.run(compare_prompt)

    # crude split, you can improve it
    if "Disagreements:" in result:
        agreements, disagreements = result.split("Disagreements:")
    else:
        agreements, disagreements = result, ""
    
    state["agreements"].append(agreements.strip())
    state["disagreements"].append(disagreements.strip())
    return state

# EVALUATE PAPER'S NODE
def evaluate_node(state):
    scores = []
    for paper in state["papers"]:
        eval_prompt = f"Evaluate this paper for relevance, novelty, and quality:\nTitle: {paper['title']}\nAbstract: {paper['abstract']}"
        eval_result = agent.run(eval_prompt)
        scores.append({"title": paper["title"], "evaluation": eval_result})
    state["scores"] = scores
    return state

# UPDATE SUMMARY NODE
def update_summary_node(state):
    prompt = "Here is the cumulative summary so far:\n" + state.get("global_summary", "")
    prompt += "\n\nHere are the new summaries:\n"
    for s in state["summaries"]:
        prompt += f"\n- {s['summary']}"

    result = agent.run(f"Update the global summary:\n{prompt}")
    state["global_summary"] = result
    return state

# GRAPH INITIALIZATION
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

class AgentState(BaseModel):
    topic: str
    papers: list[dict]
    summaries: list[dict]

workflow = StateGraph(AgentState)

workflow.add_node("fetch_papers", fetch_papers_node)
workflow.add_node("summarize_papers", summarize_papers_node)
workflow.add_node("compare_summaries", compare_papers_node)
workflow.add_node("evaluate_papers", evaluate_node)
workflow.add_node("update_summary", update_summary_node)

# EDGES
workflow.set_entry_point("fetch_papers")
workflow.add_edge("fetch_papers", "summarize_papers")
workflow.add_edge("summarize_papers", "compare_summaries")
workflow.add_edge("compare_summaries", "evaluate_papers")
workflow.add_edge("evaluate_papers", "update_summary")
workflow.add_edge("update_summary", "fetch_papers") 

# BUILD GRAPH
graph = workflow.compile()

# RUN THE GRAPH
initial_state = {"topic": "microLED optoelectronics"}
graph.invoke(initial_state)

# DISPLAY GRAPH
from IPython.display import Image, display
display(Image(graph.get_graph(xray=True).draw_mermaid_png(max_retries=5, retry_delay=2)))





