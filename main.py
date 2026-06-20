"""Simple command-line interface for the Self-Healing RAG pipeline."""

from __future__ import annotations

from graph import run_query


def print_result(final_state: dict) -> None:
    """Print the graph's final answer and step-by-step reasoning trace."""
    print("\nAnswer:")
    print(final_state["answer"])

    print("\nReasoning trace:")
    for index, entry in enumerate(final_state["reasoning_log"], start=1):
        print(f"{index}. {entry}")
    print()


def main() -> None:
    """Keep asking questions until the user types 'exit'."""
    print("Self-Healing RAG CLI")
    print("Type a question, or type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            print("Goodbye.")
            break
        if not question:
            continue

        try:
            final_state = run_query(question)
        except Exception as error:
            print(f"\nError: {error}\n")
            continue

        print_result(final_state)


if __name__ == "__main__":
    main()
