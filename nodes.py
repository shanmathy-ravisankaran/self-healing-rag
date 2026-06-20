"""LangGraph node functions for the Self-Healing RAG workflow.

Each node receives the current graph state and returns a dictionary of fields to
update. LangGraph merges those returned values back into the state before
passing it to the next node.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


class GraphState(TypedDict):
    """Shared state that moves through the LangGraph workflow."""

    question: str
    search_query: str
    retrieved_chunks: list[str]
    answer: str
    in_scope: bool
    is_grounded: bool
    retry_count: int
    reasoning_log: list[str]
    trace: list[dict]


def _format_context(chunks: list[str]) -> str:
    """Format retrieved chunks so the LLM can see clear context boundaries."""
    return "\n\n".join(
        f"Context chunk {index}:\n{chunk}" for index, chunk in enumerate(chunks, start=1)
    )


def _parse_critic_json(content: str) -> dict:
    """Parse the critic response, tolerating simple Markdown JSON fences."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    return json.loads(cleaned)


def scope_check_node(state: GraphState) -> dict:
    """Decide whether the question belongs in this 10-K RAG workflow.

    This node runs before retrieval. If the question is off-topic, the graph can
    skip the expensive retrieval/generation/critic loop and route directly to
    ``out_of_scope_node``.
    """
    load_dotenv()

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    messages = [
        SystemMessage(
            content=(
                "You are a scope filter for a financial document Q&A "
                "assistant. The assistant answers questions about National "
                "Presto Industries based on their 10-K SEC filing -- this "
                "includes topics like revenue, net income, expenses, business "
                "segments, products, risks, regulations, dividends, stock "
                "information, balance sheet items, and corporate governance.\n\n"
                "Mark a question as IN SCOPE (true) if it is about National "
                "Presto Industries, its business, financials, or anything that "
                "could reasonably appear in a 10-K filing -- even if you're "
                "not sure this specific filing contains the answer. Only mark "
                "a question as OUT OF SCOPE (false) if it is clearly unrelated "
                "to the company or to financial/business/corporate topics "
                "entirely, such as questions about cooking, personal advice, "
                "other unrelated topics, or general knowledge questions with "
                "no connection to National Presto Industries or "
                "business/finance.\n\n"
                'Respond with only JSON: {"in_scope": true/false}'
            )
        ),
        HumanMessage(content=state["question"]),
    ]

    response = llm.invoke(messages)
    try:
        verdict = _parse_critic_json(str(response.content))
        in_scope = bool(verdict["in_scope"])
    except (json.JSONDecodeError, KeyError, TypeError):
        in_scope = False

    log_entry = (
        "Scope check: question is in scope."
        if in_scope
        else "Scope check: question is outside the 10-K assistant scope."
    )

    return {
        "in_scope": in_scope,
        "reasoning_log": state["reasoning_log"] + [log_entry],
        "trace": state["trace"]
        + [
            {
                "node": "scope_check",
                "summary": log_entry,
                "retry_count": state["retry_count"],
            }
        ],
    }


def out_of_scope_node(state: GraphState) -> dict:
    """Return an immediate refusal for questions outside the 10-K scope."""
    answer = (
        "That's outside what I can help with -- I can only answer questions "
        "about National Presto Industries' 10-K filing (financials, business "
        "segments, risks, corporate governance, etc.)."
    )
    log_entry = "Out of scope: returned the scope guard response."

    return {
        "answer": answer,
        "reasoning_log": state["reasoning_log"] + [log_entry],
        "trace": state["trace"]
        + [
            {
                "node": "out_of_scope",
                "summary": log_entry,
                "retry_count": state["retry_count"],
            }
        ],
    }


def retriever_node(state: GraphState) -> dict:
    """Retrieve relevant chunks and return state updates for LangGraph.

    LangGraph nodes usually return only the fields they changed. This node
    returns new ``retrieved_chunks`` and an updated ``reasoning_log`` so later
    nodes can use the retrieved evidence and display the retrieval trace.
    """
    load_dotenv()

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # FAISS indexes saved by LangChain include pickle metadata, so LangChain
    # requires this opt-in when loading an index you created locally.
    vector_store = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # Ask FAISS for the 4 chunks most semantically similar to the current
    # search query. On the first attempt this matches the user's question; on
    # retries it may be a rewritten query designed to find better evidence.
    docs = vector_store.similarity_search(state["search_query"], k=4)
    retrieved_chunks = [doc.page_content for doc in docs]

    log_entry = (
        f"Retriever found {len(retrieved_chunks)} chunks for search query: "
        f"{state['search_query']!r}"
    )

    return {
        "retrieved_chunks": retrieved_chunks,
        "reasoning_log": state["reasoning_log"] + [log_entry],
        "trace": state["trace"]
        + [
            {
                "node": "retriever",
                "summary": log_entry,
                "retry_count": state["retry_count"],
            }
        ],
    }


def generator_node(state: GraphState) -> dict:
    """Generate an answer and return the updated answer field.

    The answer is stored in ``state['answer']`` for the critic node to inspect.
    Returning a partial dictionary keeps the node focused on the field it owns.
    """
    load_dotenv()

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    context = _format_context(state["retrieved_chunks"])

    messages = [
        SystemMessage(
            content=(
                "Answer ONLY using the provided context, and say you don't have "
                "enough information if the context doesn't answer the question. "
                "If the question refers to a relative time period like 'last "
                "year' or 'most recent', use the most recent fiscal year "
                "explicitly stated in the document or context, and state which "
                "year you used in your answer. "
                "This document is National Presto's 10-K for fiscal year 2025. "
                "Financial tables typically show 3 years of data with the MOST "
                "RECENT year listed FIRST (leftmost column), not last. If the "
                "question asks about 'last year' or 'most recent year', that "
                "means fiscal year 2025 specifically -- the first/leftmost "
                "column in any multi-year table. Always double check which "
                "column header matches the year you're citing before answering, "
                "and explicitly state the year and confirm it matches the "
                "leftmost/most recent column."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{state['question']}\n\n"
                f"Provided context:\n{context}"
            )
        ),
    ]

    response = llm.invoke(messages)
    answer = str(response.content)
    log_entry = "Generator drafted an answer."

    return {
        "answer": answer,
        "reasoning_log": state["reasoning_log"] + [log_entry],
        "trace": state["trace"]
        + [
            {
                "node": "generator",
                "summary": log_entry,
                "retry_count": state["retry_count"],
            }
        ],
    }


def rewrite_query_node(state: GraphState) -> dict:
    """Rewrite the search query before a retry and return updated state.

    The user's original ``question`` stays unchanged so the generator keeps
    answering what was actually asked. Only ``search_query`` changes, which
    gives the retriever a better chance of finding supporting evidence on the
    next pass through the graph.
    """
    load_dotenv()

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    latest_critic_reason = next(
        (
            entry.removeprefix("Critic: ").strip()
            for entry in reversed(state["reasoning_log"])
            if entry.startswith("Critic: ")
        ),
        "The previous answer was not grounded in the retrieved context.",
    )

    messages = [
        SystemMessage(
            content=(
                "You rewrite retrieval search queries for a RAG system. "
                "Return only the rewritten search query, with no quotes, labels, "
                "or explanation. Make it broader or differently worded if that "
                "would help find stronger evidence."
            )
        ),
        HumanMessage(
            content=(
                f"Original user question:\n{state['question']}\n\n"
                f"Previous search query:\n{state['search_query']}\n\n"
                f"Critic rejection reason:\n{latest_critic_reason}\n\n"
                "Rewrite the search query for the next retrieval attempt."
            )
        ),
    ]

    response = llm.invoke(messages)
    rewritten_query = str(response.content).strip()
    if not rewritten_query:
        rewritten_query = state["question"]

    new_retry_count = state["retry_count"] + 1
    log_entry = f"Rewrote search query for retry {new_retry_count}: {rewritten_query!r}"

    return {
        "search_query": rewritten_query,
        "retry_count": new_retry_count,
        "reasoning_log": state["reasoning_log"] + [log_entry],
        "trace": state["trace"]
        + [
            {
                "node": "rewrite_query",
                "summary": log_entry,
                "retry_count": new_retry_count,
            }
        ],
    }


def critic_node(state: GraphState) -> dict:
    """Check whether the answer is grounded and return routing metadata.

    ``is_grounded`` will later drive conditional LangGraph edges: grounded
    answers can finish, while ungrounded answers can loop back for another try.
    The critic reason is appended to ``reasoning_log`` so the CLI can show why
    the graph accepted or rejected the answer.
    """
    load_dotenv()

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    context = _format_context(state["retrieved_chunks"])

    messages = [
        SystemMessage(
            content=(
                "You are a strict grounding critic. Respond only with valid JSON "
                'in this exact shape: {"grounded": true/false, "reason": "..."}'
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{state['question']}\n\n"
                f"Retrieved chunks:\n{context}\n\n"
                f"Answer:\n{state['answer']}\n\n"
                "Is the answer fully supported by the retrieved chunks?"
            )
        ),
    ]

    response = llm.invoke(messages)
    try:
        verdict = _parse_critic_json(str(response.content))
        is_grounded = bool(verdict["grounded"])
        reason = str(verdict["reason"])
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        is_grounded = False
        reason = f"Critic response could not be parsed as expected JSON: {error}"

    log_entry = f"Critic: {reason}"

    return {
        "is_grounded": is_grounded,
        "reasoning_log": state["reasoning_log"] + [log_entry],
        "trace": state["trace"]
        + [
            {
                "node": "critic",
                "summary": log_entry,
                "retry_count": state["retry_count"],
            }
        ],
    }
