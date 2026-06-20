"""Evaluate the Self-Healing RAG graph on a fixed question set.

Run:

    python eval.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from graph import run_query

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing dependency: matplotlib. Install project dependencies with:\n\n"
        "    python -m pip install -r requirements.txt\n\n"
        "Then rerun: python eval.py"
    ) from error


TEST_QUESTIONS = [
    {
        "question": "What were National Presto's total net sales for fiscal year 2025?",
        "expected_type": "direct_hit",
    },
    {
        "question": "What is National Presto's net income for fiscal year 2025?",
        "expected_type": "direct_hit",
    },
    {
        "question": "What are Presto's main business segments?",
        "expected_type": "direct_hit",
    },
    {
        "question": "What dividend did the company announce?",
        "expected_type": "direct_hit",
    },
    {
        "question": "What ticker symbol does Presto trade under?",
        "expected_type": "direct_hit",
    },
    {
        "question": "Is Presto facing any major risks?",
        "expected_type": "should_retry",
    },
    {
        "question": "What environmental compliance issues does Presto face?",
        "expected_type": "should_retry",
    },
    {
        "question": "What intellectual property risks are mentioned?",
        "expected_type": "should_retry",
    },
    {
        "question": "What is National Presto's planned marketing budget for next year?",
        "expected_type": "should_refuse",
    },
    {
        "question": "What was Presto's stock price at year end?",
        "expected_type": "should_refuse",
    },
    {
        "question": "What new products is Presto planning to launch next year?",
        "expected_type": "should_refuse",
    },
    {
        "question": "What was Presto's total cost of sales for fiscal 2025?",
        "expected_type": "direct_hit",
    },
    {
        "question": "What were total current assets on the balance sheet?",
        "expected_type": "direct_hit",
    },
    {
        "question": "How many holders of record did Presto's stock have?",
        "expected_type": "direct_hit",
    },
    {
        "question": "What risks does Presto face from pandemics or health crises?",
        "expected_type": "should_retry",
    },
]

RESULTS_PATH = Path("eval_results.json")
CHART_PATH = Path("eval_chart.png")


def percent(numerator: int, denominator: int) -> float:
    """Return a percentage, avoiding division by zero."""
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


def truncate(text: str, limit: int = 50) -> str:
    """Shorten question text for the console table."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def is_correct_refusal(row: dict) -> bool:
    """Return whether a should_refuse answer declined appropriately.

    The graph can refuse either by reaching the fallback node or by generating
    a grounded answer that says the document does not contain the requested
    information. This intentionally avoids requiring an exact string match.
    """
    answer = row["answer"].lower()
    if "don't have enough information" in answer:
        return True

    absence_phrases = [
        "does not contain",
        "doesn't contain",
        "does not provide",
        "doesn't provide",
        "does not disclose",
        "doesn't disclose",
        "does not mention",
        "doesn't mention",
        "not provided",
        "not disclosed",
        "not mentioned",
        "not specified",
        "no information",
    ]
    document_terms = ["document", "context", "10-k", "annual report", "filing"]

    indicates_missing_info = any(phrase in answer for phrase in absence_phrases)
    references_document = any(term in answer for term in document_terms)

    return bool(row["is_grounded"]) and indicates_missing_info and references_document


def run_evaluation() -> list[dict]:
    """Run every test question through the graph and collect result rows."""
    results = []

    for index, item in enumerate(TEST_QUESTIONS, start=1):
        question = item["question"]
        expected_type = item["expected_type"]
        print(f"[{index}/{len(TEST_QUESTIONS)}] {question}")

        started_at = time.time()
        final_state = run_query(question)
        elapsed_seconds = time.time() - started_at

        answer = final_state["answer"]
        result = {
            "question": question,
            "answer": answer,
            "is_grounded": bool(final_state["is_grounded"]),
            "retry_count": int(final_state["retry_count"]),
            "time_seconds": elapsed_seconds,
            "expected_type": expected_type,
            "reasoning_log": final_state["reasoning_log"],
            "trace": final_state["trace"],
        }
        results.append(result)

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate evaluation metrics from result rows."""
    total = len(results)
    grounded_count = sum(1 for row in results if row["is_grounded"])
    retry_count = sum(1 for row in results if row["retry_count"] > 0)
    should_refuse = [
        row for row in results if row["expected_type"] == "should_refuse"
    ]
    correct_refusals = sum(1 for row in should_refuse if is_correct_refusal(row))
    total_time = sum(row["time_seconds"] for row in results)

    return {
        "total": total,
        "grounded_percent": percent(grounded_count, total),
        "required_retry_percent": percent(retry_count, total),
        "correct_refusal_percent": percent(correct_refusals, len(should_refuse)),
        "average_time_seconds": total_time / total if total else 0.0,
    }


def print_report(results: list[dict], metrics: dict) -> None:
    """Print aggregate metrics and a compact per-question table."""
    print("\nEvaluation Summary")
    print("==================")
    print(f"Total questions tested: {metrics['total']}")
    print(f"Grounded: {metrics['grounded_percent']:.1f}%")
    print(f"Required at least 1 retry: {metrics['required_retry_percent']:.1f}%")
    print(
        "Correct refusal for should_refuse questions: "
        f"{metrics['correct_refusal_percent']:.1f}%"
    )
    print(f"Average time per question: {metrics['average_time_seconds']:.2f}s")

    print("\nPer-question results")
    print("--------------------")
    print(f"{'Question':50}  {'Grounded':8}  {'Retries':7}  {'Time':>8}")
    print(f"{'-' * 50}  {'-' * 8}  {'-' * 7}  {'-' * 8}")

    for row in results:
        grounded = "yes" if row["is_grounded"] else "no"
        print(
            f"{truncate(row['question']):50}  "
            f"{grounded:8}  "
            f"{row['retry_count']:<7}  "
            f"{row['time_seconds']:>7.2f}s"
        )


def save_results(results: list[dict], metrics: dict) -> None:
    """Save detailed results and aggregate metrics to JSON."""
    payload = {
        "metrics": metrics,
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to {RESULTS_PATH.resolve()}")


def save_chart(metrics: dict) -> None:
    """Create and save a simple percentage bar chart."""
    labels = ["Grounded %", "Required retry %", "Correct refusal %"]
    values = [
        metrics["grounded_percent"],
        metrics["required_retry_percent"],
        metrics["correct_refusal_percent"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#2563eb", "#f59e0b", "#10b981"])

    ax.set_title("Self-Healing RAG Evaluation Metrics", fontsize=14, pad=14)
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#e5e7eb", linewidth=1)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160)
    plt.close(fig)
    print(f"Saved chart to {CHART_PATH.resolve()}")


def main() -> None:
    """Run the full evaluation and write reports."""
    results = run_evaluation()
    metrics = compute_metrics(results)
    print_report(results, metrics)
    save_results(results, metrics)
    save_chart(metrics)


if __name__ == "__main__":
    main()
