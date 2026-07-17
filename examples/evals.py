"""
Evals for the example agents — run them end-to-end against a REAL model and grade
the runs with explicit checks.

    python -m examples.evals            # 1 pass per scenario
    python -m examples.evals --repeat 3 # expose flakiness

Grading philosophy: check OUTCOMES the agent cannot fake.
  - log_detective must name the actual failing endpoint and root cause that are in
    the log, and must have used tools to get there.
  - copy_polisher must have iterated (scored at least twice), measurably improved
    the score, and purged the buzzwords the judge punishes.
"""
from __future__ import annotations

import argparse

from agentloop import Scenario, markdown_report, run

from .copy_polisher import polish
from .log_detective import investigate


# ── graders ─────────────────────────────────────────────────────────────────────
def grade_log_detective(res: dict) -> list[str]:
    fails: list[str] = []
    d = res.get("diagnosis") or {}
    text = " ".join([str(v) for v in d.values()] + [res.get("answer") or ""]).lower()
    if not d:
        fails.append("never filed a report")
    if "/api/checkout" not in text and "checkout" not in text:
        fails.append("did not identify /api/checkout as the failing endpoint")
    if "pool" not in text and "db_pool" not in text:
        fails.append("did not identify the db pool exhaustion root cause")
    if res.get("tool_calls", 0) < 2:
        fails.append(f"only {res.get('tool_calls')} tool calls (guessed instead of investigating?)")
    if res.get("stopped_reason") == "error":
        fails.append("agent errored out")
    return fails


def grade_copy_polisher(res: dict) -> list[str]:
    fails: list[str] = []
    hist = res.get("score_history") or []
    final_copy = ((res.get("package") or {}).get("copy") or "").lower()
    if len(hist) < 2:
        fails.append(f"did not iterate (scored {len(hist)}x, need >=2)")
    if hist and res.get("final_score", 0) < 70:
        fails.append(f"final score {res.get('final_score')} < 70")
    if len(hist) >= 2 and hist[-1] <= hist[0]:
        fails.append(f"score did not improve: {hist}")
    for word in ("synergistic", "leverage", "revolutionize", "cutting-edge", "next-generation"):
        if word in final_copy:
            fails.append(f"buzzword survived: {word}")
    if res.get("stopped_reason") == "error":
        fails.append("agent errored out")
    return fails


SCENARIOS = [
    Scenario("log_detective/incident", lambda: investigate(), grade_log_detective),
    Scenario("copy_polisher/buzzword_draft", lambda: polish(), grade_copy_polisher),
    Scenario(
        "copy_polisher/vague_draft",
        lambda: polish(
            brief=(
                "A one-paragraph app-store description for a habit tracker. Audience: people "
                "who abandoned habit apps before. Tone: honest, encouraging, zero pressure."
            ),
            draft="This app helps you do things better every day and reach your goals.",
        ),
        grade_copy_polisher,
    ),
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--markdown", action="store_true", help="also print a Markdown table")
    args = ap.parse_args()
    report = run(SCENARIOS, repeat=args.repeat, title="AGENTLOOP EXAMPLE EVALS")
    if args.markdown:
        print("\n" + markdown_report(report, title="Example evals"))