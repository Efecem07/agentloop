"""
Eval harness — "how do you know the agent works?"

Eyeballing one run is not an answer. An eval runs the agent over a set of SCENARIOS
and GRADES each run against explicit checks, then reports a success rate — the same
idea as a test suite, but for non-deterministic agent behaviour.

Usage:
    scenarios = [
        Scenario(
            name="backend_engineer",
            run=lambda: my_agent_entrypoint(input_a, input_b),
            grade=my_grader,          # (result) -> list of failure reasons, [] = pass
        ),
        ...
    ]
    report = await run_evals(scenarios)
    print_report(report)

Design choices:
  - A grader returns REASONS, not a bool: when a scenario fails you immediately see
    why ("final score 48 < 60", "did not iterate"), which is what you act on.
  - Checks are explicit and cheap (assertions over the result dict / trace), not
    LLM-judged by default — deterministic grading keeps eval noise low. Nothing
    stops a grader from calling an LLM judge if a scenario needs one.
  - repeat=N reruns each scenario to expose flakiness (agents are stochastic; a
    scenario that passes 1/3 of the time is a bug, not a pass).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class Scenario:
    name: str
    run: Callable[[], Awaitable[Any]]           # runs the agent, returns its result
    grade: Callable[[Any], list[str]]           # returns failure reasons; [] = pass
    meta: dict = field(default_factory=dict)    # free-form, echoed into the report


@dataclass
class ScenarioResult:
    name: str
    ok: bool
    fails: list[str]
    seconds: float
    result: Any = None
    meta: dict = field(default_factory=dict)


@dataclass
class Report:
    passed: int
    total: int
    rows: list[ScenarioResult]

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


async def run_evals(scenarios: list[Scenario], repeat: int = 1) -> Report:
    """Run every scenario `repeat` times, grade each run. A scenario instance passes
    only if ALL its repeats pass (strict by design: flaky = failing)."""
    rows: list[ScenarioResult] = []
    passed = 0
    for sc in scenarios:
        fails_all: list[str] = []
        last_result: Any = None
        t0 = time.monotonic()
        for i in range(repeat):
            tag = f"run {i+1}/{repeat}: " if repeat > 1 else ""
            try:
                last_result = await sc.run()
            except Exception as e:  # noqa: BLE001 - an eval must survive a crashing agent
                fails_all.append(f"{tag}exception: {e}")
                continue
            for reason in sc.grade(last_result):
                fails_all.append(tag + reason)
        seconds = time.monotonic() - t0
        ok = not fails_all
        passed += int(ok)
        rows.append(ScenarioResult(sc.name, ok, fails_all, round(seconds, 1), last_result, sc.meta))
    return Report(passed, len(scenarios), rows)


def print_report(report: Report, title: str = "EVALS") -> None:
    print(f"\n=== {title}: {report.passed}/{report.total} passed "
          f"({report.success_rate:.0%}) ===")
    for r in report.rows:
        flag = "PASS" if r.ok else "FAIL"
        extra = " ".join(f"{k}={v}" for k, v in _row_stats(r).items())
        print(f"[{flag}] {r.name}  ({r.seconds}s)  {extra}")
        for reason in r.fails:
            print(f"       why: {reason}")


def markdown_report(report: Report, title: str = "Evals") -> str:
    """Render the report as a Markdown table (handy for READMEs and CI summaries)."""
    lines = [
        f"**{title}: {report.passed}/{report.total} passed ({report.success_rate:.0%})**",
        "",
        "| scenario | result | details | time |",
        "|---|---|---|---|",
    ]
    for r in report.rows:
        flag = "✅ pass" if r.ok else "❌ fail"
        stats = ", ".join(f"{k}={v}" for k, v in _row_stats(r).items()) or "-"
        if r.fails:
            stats = "; ".join(r.fails)
        lines.append(f"| {_cell(r.name)} | {flag} | {_cell(stats)} | {r.seconds}s |")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    """Make a value safe inside a Markdown table cell. Failure reasons are written by
    graders and routinely contain the very characters that break a table: a pipe from
    a regex or a range, a newline from a multi-line assertion."""
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _row_stats(r: ScenarioResult) -> dict:
    """Pull common, interesting numbers out of a result dict if they exist."""
    out: dict = {}
    if isinstance(r.result, dict):
        for key in ("score_history", "final_score", "tool_calls", "stopped_reason"):
            if r.result.get(key) is not None:
                out[key] = r.result[key]
    return out


def run(scenarios: list[Scenario], repeat: int = 1, title: str = "EVALS") -> Report:
    """Sync convenience wrapper: run + print."""
    report = asyncio.run(run_evals(scenarios, repeat=repeat))
    print_report(report, title=title)
    return report