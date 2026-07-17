"""
Example 1: log detective — an agent with DETERMINISTIC tools.

Given a production log, the agent must find what broke, when, and why. The tools are
plain Python (grep / count / slice) — only the DECISIONS are the model's. This is the
cleanest way to see the loop think: which tool it reaches for, how it narrows down,
when it decides it has enough evidence.

Run:  python -m examples.log_detective
"""
from __future__ import annotations

import asyncio
import collections
import re

from agentloop import NO_ARGS, Agent, Tool

# A synthetic incident log: healthy traffic, a deploy marker, then /api/checkout
# starts throwing 500s (db pool exhausted) while everything else stays healthy.
SAMPLE_LOG = "\n".join(
    [
        "2026-07-14T09:58:01 INFO  GET /api/products 200 34ms",
        "2026-07-14T09:58:12 INFO  POST /api/checkout 200 88ms",
        "2026-07-14T09:58:40 INFO  GET /api/products 200 30ms",
        "2026-07-14T09:59:03 INFO  POST /api/checkout 200 91ms",
        "2026-07-14T10:00:00 INFO  DEPLOY version=2.41.0 service=checkout-svc",
        "2026-07-14T10:00:41 INFO  GET /api/products 200 31ms",
        "2026-07-14T10:01:02 ERROR POST /api/checkout 500 2004ms db_pool_exhausted connections=50/50",
        "2026-07-14T10:01:18 ERROR POST /api/checkout 500 2010ms db_pool_exhausted connections=50/50",
        "2026-07-14T10:01:30 INFO  GET /api/products 200 29ms",
        "2026-07-14T10:01:44 ERROR POST /api/checkout 500 2008ms db_pool_exhausted connections=50/50",
        "2026-07-14T10:02:05 ERROR POST /api/checkout 500 2013ms db_pool_exhausted connections=50/50",
        "2026-07-14T10:02:22 INFO  GET /api/health 200 2ms",
        "2026-07-14T10:02:40 ERROR POST /api/checkout 500 2006ms db_pool_exhausted connections=50/50",
        "2026-07-14T10:03:01 INFO  GET /api/products 200 33ms",
        "2026-07-14T10:03:19 ERROR POST /api/checkout 500 2011ms db_pool_exhausted connections=50/50",
    ]
)


# ── tools (pure Python, no LLM inside) ──────────────────────────────────────────
async def _search_log(state: dict, pattern: str = "", **_) -> dict:
    lines = [l for l in state["log"].splitlines() if re.search(pattern, l)] if pattern else []
    return {"matches": lines[:20], "match_count": len(lines)}


async def _count_by_status(state: dict, **_) -> dict:
    counts: collections.Counter = collections.Counter()
    for line in state["log"].splitlines():
        m = re.search(r" (\d{3}) \d+ms", line)
        if m:
            counts[m.group(1)] += 1
    return dict(counts)


async def _tail_log(state: dict, n: int = 10, **_) -> dict:
    lines = state["log"].splitlines()
    return {"lines": lines[-int(n):]}


async def _report(state: dict, failing_endpoint: str = "", root_cause: str = "",
                  started_at: str = "", evidence: str = "", **_) -> dict:
    diagnosis = {
        "failing_endpoint": failing_endpoint,
        "root_cause": root_cause,
        "started_at": started_at,
        "evidence": evidence,
    }
    state["diagnosis"] = diagnosis
    return diagnosis


SYSTEM = """You are an incident-response agent investigating a production log.
Work with tools, do not guess: count statuses to see IF something is failing, search
to find WHICH endpoint and WHY, and check what happened right before the errors began
(deploys are prime suspects). When you know the failing endpoint, the root cause and
the start time, call report, then answer with one short diagnosis sentence."""


def build_agent(llm=None) -> Agent:
    tools = [
        Tool("count_by_status", "Count log lines per HTTP status code across the whole log.", NO_ARGS, _count_by_status),
        Tool(
            "search_log",
            "Return log lines matching a regex pattern (max 20 shown, with total count).",
            {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
            _search_log,
        ),
        Tool(
            "tail_log",
            "Return the last n log lines.",
            {"type": "object", "properties": {"n": {"type": "integer"}}, "required": []},
            _tail_log,
        ),
        Tool(
            "report",
            "File the final diagnosis once you know it.",
            {
                "type": "object",
                "properties": {
                    "failing_endpoint": {"type": "string"},
                    "root_cause": {"type": "string"},
                    "started_at": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["failing_endpoint", "root_cause"],
            },
            _report,
        ),
    ]
    return Agent("log_detective", SYSTEM, tools, max_steps=8, temperature=0.2, llm=llm)


async def investigate(log: str = SAMPLE_LOG) -> dict:
    agent = build_agent()
    state: dict = {"log": log}
    result = await agent.run("Something seems wrong in production. Investigate this log and diagnose it.", state=state)
    return {
        "diagnosis": state.get("diagnosis"),
        "answer": result.output,
        "tool_calls": result.tool_calls,
        "stopped_reason": result.stopped_reason,
        "trace": [s.short() for s in result.steps],
    }


if __name__ == "__main__":
    res = asyncio.run(investigate())
    print("DIAGNOSIS:", res["diagnosis"])
    print("ANSWER:", res["answer"])
    print("TRACE:")
    for line in res["trace"]:
        print("  " + line)