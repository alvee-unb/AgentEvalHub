"""
planner.py — Planner node: analyses user prompt and produces a structured plan.
"""

from __future__ import annotations
import logging
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import AgentState
from src.prompts.templates import PLANNER_SYSTEM, PLANNER_USER

logger = logging.getLogger(__name__)


def parse_planner_output(text: str) -> tuple[list[str], bool, str]:
    """Extract risk_flags, requires_safety_check, plan from planner response."""
    risk_flags: list[str] = []
    requires_check = False
    plan = ""

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("RISK_FLAGS:"):
            raw = line.replace("RISK_FLAGS:", "").strip()
            if raw.lower() != "none":
                risk_flags = [f.strip() for f in raw.split(",") if f.strip()]
        elif line.startswith("REQUIRES_SAFETY_CHECK:"):
            val = line.replace("REQUIRES_SAFETY_CHECK:", "").strip().upper()
            requires_check = val == "YES"
        elif line.startswith("PLAN:"):
            plan = line.replace("PLAN:", "").strip()

    return risk_flags, requires_check, plan


def planner_node(state: AgentState, llm: ChatOllama) -> dict:
    """
    Planner node: determines intent, identifies risk flags, produces a plan.
    Returns a partial state update dict.
    """
    logger.info("[planner] Processing prompt (len=%d)", len(state.user_prompt))
    t0 = time.perf_counter()

    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=PLANNER_USER.format(prompt=state.user_prompt)),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content
        logger.debug("[planner] Raw output:\n%s", raw)
    except Exception as exc:
        logger.error("[planner] LLM call failed: %s", exc)
        return {
            "error": f"planner LLM error: {exc}",
            "node_trace": state.node_trace + ["planner"],
            "latency_ms": {**state.latency_ms, "planner": (time.perf_counter() - t0) * 1000},
        }

    risk_flags, requires_check, plan = parse_planner_output(raw)

    latency = (time.perf_counter() - t0) * 1000
    logger.info(
        "[planner] Done in %.0fms | risk_flags=%s | safety_check=%s",
        latency, risk_flags, requires_check,
    )

    return {
        "plan": plan or raw,
        "risk_flags": risk_flags,
        "requires_safety_check": requires_check or bool(risk_flags),
        "node_trace": state.node_trace + ["planner"],
        "latency_ms": {**state.latency_ms, "planner": latency},
    }
