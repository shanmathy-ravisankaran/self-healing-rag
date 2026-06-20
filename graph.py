"""LangGraph workflow wiring for the Self-Healing RAG pipeline.

The graph flow is:

START -> scope_check -> conditional route

In-scope route:
retriever -> generator -> critic -> conditional route

After the critic runs, a conditional edge reads ``state["is_grounded"]``:
- grounded answers go to END
- ungrounded answers rewrite the search query and retry up to two times
- ungrounded answers after the retry limit go to a fallback node, then END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from nodes import (
    GraphState,
    critic_node,
    generator_node,
    out_of_scope_node,
    retriever_node,
    scope_check_node,
    rewrite_query_node,
)


MAX_RETRIES = 2
FALLBACK_ANSWER = "I don't have enough information to answer that confidently."


def route_after_scope_check(state: GraphState) -> str:
    """Choose whether to continue into RAG or end with a scope refusal.

    Conditional edges in LangGraph inspect the current state and return a route
    label. The label is mapped to the next node below: in-scope questions go to
    retrieval, while off-topic questions go straight to ``out_of_scope`` and
    then END.
    """
    if state["in_scope"]:
        return "in_scope"

    return "out_of_scope"


def fallback_node(state: GraphState) -> dict:
    """Return a conservative final answer after the retry limit is reached.

    This node updates ``answer`` so the user receives a clear response instead
    of an unsupported answer. It also logs why the graph stopped retrying.
    """
    log_entry = "Retry limit reached. Returning fallback answer."

    return {
        "answer": FALLBACK_ANSWER,
        "reasoning_log": state["reasoning_log"] + [log_entry],
        "trace": state["trace"]
        + [
            {
                "node": "fallback",
                "summary": log_entry,
                "retry_count": state["retry_count"],
            }
        ],
    }


def route_after_critic(state: GraphState) -> str:
    """Choose the next edge after the critic node.

    This is the core LangGraph idea for self-healing control flow: the graph can
    inspect the current state and choose a different next step at runtime. Here,
    the critic's ``is_grounded`` flag determines whether to finish, retry, or
    fall back. The retry counter is incremented inside ``rewrite_query_node``,
    which means this check represents the number of completed rewrite/retry
    attempts so far.
    """
    if state["is_grounded"]:
        return "end"

    if state["retry_count"] < MAX_RETRIES:
        return "retry"

    return "fallback"


def build_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Register each callable as a node in the graph.
    workflow.add_node("scope_check", scope_check_node)
    workflow.add_node("out_of_scope", out_of_scope_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("fallback", fallback_node)

    # The scope guard runs before any retrieval or generation work.
    workflow.add_edge(START, "scope_check")
    workflow.add_conditional_edges(
        "scope_check",
        route_after_scope_check,
        {
            "in_scope": "retriever",
            "out_of_scope": "out_of_scope",
        },
    )
    workflow.add_edge("out_of_scope", END)

    # In-scope questions continue through the original RAG path.
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "critic")

    # Conditional edges map route labels from route_after_critic() to graph
    # destinations. Returning "retry" sends execution through rewrite_query,
    # which changes state["search_query"] before looping back to retriever.
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "end": END,
            "retry": "rewrite_query",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("rewrite_query", "retriever")
    workflow.add_edge("fallback", END)

    return workflow.compile()


graph = build_graph()


def run_query(question: str) -> GraphState:
    """Initialize graph state, run the workflow, and return the final state."""
    initial_state: GraphState = {
        "question": question,
        "search_query": question,
        "retrieved_chunks": [],
        "answer": "",
        "in_scope": True,
        "is_grounded": False,
        "retry_count": 0,
        "reasoning_log": [],
        "trace": [],
    }

    return graph.invoke(initial_state)
