"""Eval harness tests — no network, no agent needed: scenarios are just async callables."""
from __future__ import annotations

from agentloop import Scenario, markdown_report, run_evals


async def good_run():
    return {"final_score": 90, "score_history": [55, 90], "tool_calls": 4, "stopped_reason": "final"}


async def bad_run():
    return {"final_score": 40, "score_history": [40], "tool_calls": 1, "stopped_reason": "final"}


async def crashing_run():
    raise RuntimeError("agent exploded")


def grade(res: dict) -> list[str]:
    fails = []
    if res.get("final_score", 0) < 70:
        fails.append(f"final score {res.get('final_score')} < 70")
    if len(res.get("score_history") or []) < 2:
        fails.append("did not iterate")
    return fails


async def test_pass_fail_and_crash_are_reported():
    report = await run_evals([
        Scenario("good", good_run, grade),
        Scenario("bad", bad_run, grade),
        Scenario("crash", crashing_run, grade),
    ])
    assert report.total == 3
    assert report.passed == 1
    by_name = {r.name: r for r in report.rows}
    assert by_name["good"].ok
    assert "final score 40 < 70" in by_name["bad"].fails
    assert any("exception" in f for f in by_name["crash"].fails)


async def test_repeat_makes_flaky_scenarios_fail():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        # passes on odd calls, fails on even calls
        return {"final_score": 90 if calls["n"] % 2 else 10, "score_history": [1, 2]}

    report = await run_evals([Scenario("flaky", flaky, grade)], repeat=3)
    assert report.passed == 0  # strict: one bad repeat fails the scenario
    assert any("run 2/3" in f for f in report.rows[0].fails)


async def test_markdown_report_renders():
    report = await run_evals([Scenario("good", good_run, grade)])
    md = markdown_report(report)
    assert "| good | ✅ pass |" in md
    assert "1/1 passed" in md


async def test_markdown_report_escapes_table_breaking_characters():
    """Graders write failure reasons in prose; a pipe from a regex or a newline from a
    multi-line assertion used to split the row and mangle the whole table."""
    async def run_it():
        return {"final_score": 10}

    def pipey_grade(_res: dict) -> list[str]:
        return ["pattern /a|b/ did not match", "line one\nline two"]

    report = await run_evals([Scenario("pipes", run_it, pipey_grade)])
    md = markdown_report(report)
    body = [ln for ln in md.splitlines() if ln.startswith("| pipes")]
    assert len(body) == 1                              # one row, not split across lines
    assert "/a\\|b/" in body[0]                        # the reason's pipe is escaped
    assert len(body[0].split(" | ")) == 4              # still exactly four columns