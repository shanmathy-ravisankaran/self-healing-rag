"""Throwaway script for manually testing the LangGraph workflow.

Usage:

    python test_graph.py "What does the document say about refunds?"
"""

from __future__ import annotations

import sys
from pprint import pprint

from graph import run_query


def main() -> None:
    """Run one test question through the graph and print the final state."""
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Test question: ").strip()

    final_state = run_query(question)

    print("\nAnswer:")
    print(final_state["answer"])

    print("\nReasoning log:")
    for index, entry in enumerate(final_state["reasoning_log"], start=1):
        print(f"{index}. {entry}")

    print("\nFull final state:")
    pprint(final_state)


if __name__ == "__main__":
    main()
