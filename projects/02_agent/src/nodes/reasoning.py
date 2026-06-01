"""
reasoning.py — Reasoning node: chain-of-thought safety assessment.
"""

from __future__ import annotations
import logging
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import AgentState
from src.prompts.templates import REASONING_SYSTEM, REASONING_USER

logger = logging.getLogger(__name__)


def parse_reasoning_output(text: str) -> tuple[str, str, str]:
    """Extract reasoning, safety_assessment, unsafe_reason."""
    reasoning = ""
    assessment = "SAFE"
    unsafe_reason = "N/A"

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("SAFETY_ASSESSMENT:"):
            val = line.replace("SAFETY_ASSESSMENT:", "").strip().upper()
            if val in ("SAFE", "BORDERLINE", "UNSAFE"):
                assessment = val
        elif line.startswith("UNSAFE_REASON:"):
            unsafe_reason = line.replace("UNSAFE_REASON:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    if not reasoning:
        reasoning = text  # fallback: use full output as reasoning

    return reasoning, assessment, unsafe_reason


def reasoning_node(state: AgentState, llm: ChatOllama) -> dict:
    """
    Reasoning node: performs chain-of-thought safety analysis.
    Returns a partial state update dict.
    """
    logger.info("[reasoning] Assessing safety | flags=%s", state.risk_flags)
    t0 = time.perf_counter()

    messages = [
        SystemMessage(content=REASONING_SYSTEM),
        HumanMessage(content=REASONING_USER.format(
            prompt=state.user_prompt,
            plan=state.plan or "No plan available.",
            risk_flags=", ".join(state.risk_flags) if state.risk_flags else "none",
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content
        logger.debug("[reasoning] Raw output:\n%s", raw)
    except Exception as exc:
        logger.error("[reasoning] LLM call failed: %s", exc)
        return {
            "error": f"reasoning LLM error: {exc}",
            "node_trace": state.node_trace + ["reasoning"],
            "latency_ms": {**state.latency_ms, "reasoning": (time.perf_counter() - t0) * 1000},
        }

    reasoning, assessment, unsafe_reason = parse_reasoning_output(raw)

    latency = (time.perf_counter() - t0) * 1000
    logger.info("[reasoning] Done in %.0fms | assessment=%s", latency, assessment)

    return {
        "reasoning": reasoning,
        "safety_assessment": assessment,
        "node_trace": state.node_trace + ["reasoning"],
        "latency_ms": {**state.latency_ms, "reasoning": latency},
    }
