"""
Example 2: copy polisher — an agent that ITERATES UNTIL A SCORE IMPROVES.

This is the loop-with-feedback pattern: the agent has a measurable goal (a strict
LLM judge scores the text 0-100), tools to act (rewrite), and a signal to decide
when to stop (score >= target, or no improvement between rounds). The score history
lands in shared state, so evals can verify the agent actually iterated instead of
one-shotting.

The same pattern powers real product features (e.g. tailoring a CV until its
job-match score stops improving) — this example keeps the shape, generically.

Run:  python -m examples.copy_polisher
"""
from __future__ import annotations

import asyncio

from agentloop import NO_ARGS, Agent, Tool, chat_json

DEFAULT_BRIEF = (
    "A landing-page hero paragraph for a bookkeeping app aimed at freelancers. "
    "Audience: non-accountants who fear tax season. Tone: plain, confident, no hype."
)

DEFAULT_DRAFT = (
    "Our synergistic next-generation platform leverages cutting-edge technology to "
    "revolutionize your workflow and empower your business to unlock its full potential."
)


# ── tools ───────────────────────────────────────────────────────────────────────
async def _score_copy(state: dict, **_) -> dict:
    """Strict LLM judge. Returns score + concrete problems. Drives the loop."""
    data = await chat_json(
        "You are a ruthless direct-response copy editor. Score the copy 0-100 for the "
        "given brief. Be strict, do not inflate. Buzzwords (synergy, leverage, empower, "
        "revolutionize, cutting-edge, next-generation) cap the score at 40. Return ONLY "
        'JSON: {"score": int, "problems": [str]} — problems are concrete and short.',
        f"BRIEF: {state['brief']}\n\nCOPY:\n{state['copy']}",
        max_tokens=400,
    )
    score = int(data.get("score") or 0)
    scores = state.setdefault("scores", [])
    prev = scores[-1] if scores else None
    scores.append(score)
    out = {"score": score, "problems": data.get("problems", []), "rounds_so_far": len(scores)}
    # Agentic feedback: tools should guide the agent, including telling it when to STOP.
    if score >= 85:
        out["advice"] = "Score is strong. Call finalize now."
    elif prev is not None and score <= prev:
        out["advice"] = "No improvement versus last round. Stop rewriting and call finalize."
    return out


async def _rewrite_copy(state: dict, focus: str = "", **_) -> dict:
    data = await chat_json(
        "You rewrite marketing copy to fit a brief. Plain language, specific benefit, "
        "no buzzwords, at most 45 words. Return ONLY JSON: {\"copy\": str}.",
        f"BRIEF: {state['brief']}\n\nCURRENT COPY:\n{state['copy']}\n\n"
        f"Fix these problems first: {focus or 'see brief'}",
        max_tokens=300,
    )
    new_copy = (data.get("copy") or "").strip()
    if new_copy:
        state["copy"] = new_copy
    return {"copy": new_copy}


async def _finalize(state: dict, **_) -> dict:
    package = {
        "copy": state["copy"],
        "final_score": (state.get("scores") or [0])[-1],
        "score_history": state.get("scores", []),
    }
    state["final_package"] = package
    state["done"] = True
    return package


SYSTEM = """You are a copy-polishing agent. Goal: maximise the judge's score for the copy.
Loop: score_copy first to get a baseline -> rewrite_copy targeting the reported problems ->
score_copy again to confirm improvement. Repeat until score >= 85 OR a round does not improve.
Then call finalize and reply with one short sentence on what you changed.
Do not narrate every step; act through tool calls."""


def build_agent(llm=None) -> Agent:
    tools = [
        Tool("score_copy", "Strictly score the current copy against the brief; returns score and concrete problems.", NO_ARGS, _score_copy),
        Tool(
            "rewrite_copy",
            "Rewrite the current copy to fix the reported problems.",
            {"type": "object", "properties": {"focus": {"type": "string", "description": "problems to fix, comma-separated"}}, "required": []},
            _rewrite_copy,
        ),
        Tool("finalize", "Assemble the final result. Call once the score is good enough.", NO_ARGS, _finalize),
    ]
    return Agent("copy_polisher", SYSTEM, tools, max_steps=10, temperature=0.3, llm=llm)


async def polish(brief: str = DEFAULT_BRIEF, draft: str = DEFAULT_DRAFT) -> dict:
    agent = build_agent()
    state: dict = {"brief": brief, "copy": draft}
    result = await agent.run(f"Polish this copy until the judge is satisfied. Brief: {brief}", state=state)
    package = state.get("final_package") or await _finalize(state)
    return {
        "package": package,
        "final_score": package.get("final_score", 0),
        "score_history": state.get("scores", []),
        "tool_calls": result.tool_calls,
        "stopped_reason": result.stopped_reason,
        "agent_note": result.output,
        "trace": [s.short() for s in result.steps],
    }


if __name__ == "__main__":
    res = asyncio.run(polish())
    print("SCORES:", res["score_history"], "->", res["final_score"])
    print("FINAL COPY:", res["package"]["copy"])
    print("TRACE:")
    for line in res["trace"]:
        print("  " + line)