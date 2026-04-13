#!/usr/bin/env python3
"""
diagnostic_agent.py

Orchestrating diagnostic agent for the PaaS benchmark.

Given an incident scenario (a natural-language question describing symptoms),
this agent reasons over the problem and iteratively calls two sub-agents:

  sql_agent  : queries benchmark_db.sqlite for log evidence
  doc_agent  : retrieves relevant documentation from doc_chroma_db

The agent uses an iterative reasoning loop:
  1. Analyse the question to form an initial hypothesis.
  2. Query logs (SQL) to find supporting or refuting evidence.
  3. Query docs to understand what the evidence means.
  4. Repeat up to MAX_TURNS if more evidence is needed.
  5. Synthesise a final diagnosis.

The full reasoning trace (all tool calls and their results) is recorded
so that evaluate.py can score both retrieval quality and reasoning quality
independently of the final answer.

Edit the CONFIG section, then run:
    python diagnostic_agent.py

Dependencies:
    pip install langchain-community langchain-openai langgraph sqlalchemy
    pip install chromadb sentence-transformers ollama
"""

import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from query_doc_agent import query as doc_query
from query_sql_agent_qwen import build_agent as build_sql_agent, _extract_answer


# ============================================================================
# CONFIG
# ============================================================================

# --- LLM for the orchestrator ---
# The diagnostic agent does the high-level reasoning.
# Using the same Qwen endpoint as the SQL agent keeps everything local.
LLM_MODEL    = "qwen2.5-coder:7b"
LLM_TEMP     = 0.0
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_API_KEY  = "ollama"

# --- Sub-agent settings ---
SQL_DB_URI       = "sqlite:///./data/benchmark_db.sqlite"
DOC_DB_PATH      = "./doc_chroma_db"
DOC_COLLECTION   = "docs"
DOC_N_RESULTS    = 5     # docs to retrieve per doc_agent call
SQL_MAX_ITER     = 12    # max steps for the SQL sub-agent
DOC_LLM_MODEL    = "llama3.2"  # doc agent can use a lighter model

# --- Orchestrator behaviour ---
MAX_TURNS    = 6     # max tool calls by the diagnostic agent
VERBOSE      = True
OUTPUT_FILE  = "diagnostic_results.json"


# ============================================================================
# TOOL CALL TRACE
#
# We instrument each tool so that every call and its result are captured
# in a trace list. evaluate.py reads this trace to score:
#   - which sub-agent was called and when
#   - which docs were retrieved (doc_agent calls)
#   - whether the agent called the right agents for the incident type
# ============================================================================

_trace = []

def _record(tool_name, inputs, result):
    _trace.append({
        "tool":   tool_name,
        "inputs": inputs,
        "result": result,
    })


# ============================================================================
# SUB-AGENT TOOL WRAPPERS
#
# Each sub-agent is exposed to the diagnostic agent as a single LangChain tool.
# The tool docstring is what the LLM reads to decide when to use it —
# keep these precise and distinct.
# ============================================================================

# Build the SQL sub-agent once at module load time (expensive, reused across calls).
_sql_agent, _sql_system_prompt = build_sql_agent(
    db_uri=SQL_DB_URI,
    verbose=False,  # suppress SQL agent's own trace; diagnostic agent traces it
)


@tool
def query_logs(question: str) -> str:
    """
    Query the platform log database (SQLite) to find evidence about a specific
    incident. Use this tool when you need to:
      - Find error messages, warning patterns, or sequences of events in logs
      - Count occurrences of a specific event or component
      - Identify timestamps, node IDs, or instance IDs involved in an incident
      - Determine the order of events (what happened first vs. what followed)

    Input: a natural language question about the logs.
    Output: a text answer derived from SQL queries over the log database.

    Example inputs:
      "What ERROR-level events appear in the ROUTER component?"
      "How many instances of app-q0k5oz had health check failures?"
      "What is the sequence of events for the NATS message bus around 2008-11-10T10:30?"
    """
    result = _sql_agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": SQL_MAX_ITER},
    )
    answer = _extract_answer(result)
    _record("query_logs", {"question": question}, answer)
    return answer


@tool
def query_docs(question: str) -> str:
    """
    Search the platform documentation corpus for operational knowledge.
    Use this tool when you need to:
      - Understand what a specific error message or log pattern means
      - Find the investigation steps for a failure pattern
      - Look up configuration thresholds, normal baselines, or alert values
      - Understand how two platform components interact or depend on each other
      - Find the root cause category for a set of log symptoms

    Input: a natural language question about platform behaviour or operations.
    Output: an answer synthesised from retrieved runbooks, error references,
            config notes, and architecture documentation.

    Example inputs:
      "What does POOL EXHAUSTED mean and what causes it?"
      "What is the runbook for investigating simultaneous instance crashes on a cell?"
      "What configuration value must exceed instance startup time to prevent oscillation?"
    """
    result = doc_query(
        question=question,
        n_results=DOC_N_RESULTS,
        llm_model=DOC_LLM_MODEL,
        db_path=DOC_DB_PATH,
        collection_name=DOC_COLLECTION,
        verbose=False,  # suppress doc agent's own output
    )
    # Record the full retrieved_docs list so evaluate.py can score retrieval
    _record("query_docs", {"question": question}, {
        "answer":         result["answer"],
        "retrieved_docs": result["retrieved_docs"],
    })
    return result["answer"]


# ============================================================================
# DIAGNOSTIC AGENT
# ============================================================================

SYSTEM_PROMPT = """You are an expert platform reliability engineer diagnosing
incidents on a Cloud Foundry-compatible PaaS platform.

You have two tools:
  query_logs  — search the live log database for evidence of what happened
  query_docs  — search platform documentation for operational knowledge

## Diagnostic approach

1. Read the incident description carefully. Identify the symptoms and the
   affected component(s).
2. Use query_logs to find the specific log evidence for this incident.
   Start with the most distinctive symptom (error messages, specific component).
3. Use query_docs to understand what the log evidence means and what the
   root cause category is.
4. If the first round of evidence is ambiguous, do a second round:
   query_logs for more specific evidence, then query_docs for confirmation.
5. Synthesise a final diagnosis that states:
   - The root cause (specific and technical, not vague)
   - The key log evidence that supports the diagnosis
   - The failure pattern (e.g. connection pool exhaustion, cell OOM, cert expiry)
   - The recommended fix

## Rules
- Never guess the root cause without log evidence.
- Always use query_logs before concluding — your diagnosis must be grounded in
  the actual log data, not just documentation knowledge.
- Be specific: name the component, the error message, the threshold value.
- If the evidence points to a red herring (a symptom that looks like a cause),
  say so and identify what the actual upstream cause is.
- Final answer must be concise: root cause in one sentence, evidence in 2-3 bullet
  points, recommended fix in one sentence."""


def build_diagnostic_agent():
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMP,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )

    agent = create_react_agent(
        llm,
        tools=[query_logs, query_docs],
        prompt=SYSTEM_PROMPT,
    )
    agent._max_iterations = MAX_TURNS
    return agent


# ============================================================================
# QUERY PIPELINE
# ============================================================================

def diagnose(
    incident_id,
    question,
    agent=None,
    verbose=VERBOSE,
):
    """
    Run a single incident scenario through the diagnostic agent.

    Args:
        incident_id : e.g. "INC-008" — used to label the trace for eval
        question    : operator-style symptom description
        agent       : pre-built agent (build once, reuse across scenarios)
        verbose     : print trace to stdout

    Returns:
        dict with:
          incident_id      : label
          question         : input
          diagnosis        : final answer from the agent
          status           : "ok" or "error"
          tool_call_trace  : list of {tool, inputs, result} dicts
    """
    global _trace
    _trace = []  # reset trace for this scenario

    if agent is None:
        agent = build_diagnostic_agent()

    max_iter = getattr(agent, "_max_iterations", MAX_TURNS)

    if verbose:
        print("\n" + "=" * 70)
        print(f"INCIDENT : {incident_id}")
        print(f"QUESTION : {question[:100]}...")
        print("=" * 70)

    diagnosis = None
    status    = "ok"

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": max_iter * 3},  # langgraph counts node visits
        )
        diagnosis = _extract_answer(result)

    except Exception as e:
        diagnosis = f"Agent error: {e}"
        status    = "error"

    if verbose:
        print(f"\n{'─' * 70}")
        print(f"TOOL CALLS: {len(_trace)}")
        for i, call in enumerate(_trace, 1):
            q = call["inputs"].get("question", "")[:60]
            print(f"  {i}. {call['tool']}({q!r})")
        print(f"\nDIAGNOSIS:")
        print("-" * 70)
        print(diagnosis)
        print("-" * 70)

    return {
        "incident_id":     incident_id,
        "question":        question,
        "diagnosis":       diagnosis,
        "status":          status,
        "tool_call_trace": list(_trace),  # copy; _trace reset on next call
    }


def save_results(results, output_file=OUTPUT_FILE):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {len(results)} result(s) to {output_file}")


# ============================================================================
# MAIN — run a sample of scenarios
# ============================================================================

if __name__ == "__main__":

    # Build the diagnostic agent once — reused across all scenarios
    agent = build_diagnostic_agent()

    # Sample scenarios — replace with the full BENCHMARK_CASES list from
    # benchmark_incidents.py to run all 25 incidents.
    scenarios = [
        {
            "incident_id": "INC-008",
            "question": (
                "payments-api is returning sustained 503s. Metrics show the "
                "database connection pool climbing. An autoscaler scale-out fired "
                "but the 503s continued even after new instances started. "
                "What is the root cause?"
            ),
        },
        {
            "incident_id": "INC-016",
            "question": (
                "Multiple platform components appear to be failing simultaneously: "
                "the routing table is stale, cell heartbeats are missing, and Diego "
                "is evacuating cells. But the cells themselves show no errors in their "
                "own logs. What is causing this?"
            ),
        },
        {
            "incident_id": "INC-025",
            "question": (
                "At a specific time, all inter-service communication on the platform "
                "failed simultaneously with TLS errors. Six hours earlier there were "
                "CERT WARNING messages in the logs. What happened?"
            ),
        },
    ]

    all_results = []
    for scenario in scenarios:
        result = diagnose(
            incident_id=scenario["incident_id"],
            question=scenario["question"],
            agent=agent,
            verbose=VERBOSE,
        )
        all_results.append(result)

    save_results(all_results)
