"""Run the demo question once and export the final graph state as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from graph import run_query


DEMO_QUESTION = "Is Presto facing any major risks?"
OUTPUT_PATH = Path("demo_run.json")


def main() -> None:
    """Run the demo query and save the complete final state."""
    final_state = run_query(DEMO_QUESTION)

    OUTPUT_PATH.write_text(
        json.dumps(final_state, indent=2),
        encoding="utf-8",
    )

    print(f"Saved demo run to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
