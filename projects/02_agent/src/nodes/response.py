"""
response.py — Response node: produces the final user-facing output.
"""

from __future__ import annotations
import logging
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import AgentState
from src.prompts.templates import RESPONSE_SYSTEM, RESPONSE_USER

logger = logging.getLogger(__name__)

_REFUSAL_KEYWORDS = ("i cannot", "i can't", "i'm unable", "i will not", "i won't",
                     "i am not able", "not able to", "refuse", "cannot provide",
                     "against my guidelines")


def _detect_refusal(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _REFUSAL_KEYWORDS)


def response_node(state: AgentState, llm: ChatOllama) -> dict:
    """
    Response node: generates the final response to the user.
    Returns a partial state update dict.
    """
    assessment = state.safety_assessment or "SAFE"
    logger.info("[response] Generating response | assessment=%s", assessment)
    t0 = time.perf_counter()

    # Determine unsafe_reason from reasoning text if available
    unsafe_reason = "N/A"
    if assessment == "UNSAFE" and state.reasoning:
        for line in (state.reasoning or "").splitlines():
            if "unsafe_reason:" in line.lower():
                unsafe_reason = line.split(":", 1)[-1].strip()
                break

    messages = [
        SystemMessage(content=RESPONSE_SYSTEM),
        HumanMessage(content=RESPONSE_USER.format(
            prompt=state.user_prompt,
            safety_assessment=assessment,
            unsafe_reason=unsafe_reason,
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content
        logger.debug("[response] Raw output:\n%s", raw)
    except Exception as exc:
        logger.error("[response] LLM call failed: %s", exc)
        return {
            "error": f"response LLM error: {exc}",
            "node_trace": state.node_trace + ["response"],
            "latency_ms": {**state.latency_ms, "response": (time.perf_counter() - t0) * 1000},
        }

    refused = _detect_refusal(raw) or assessment == "UNSAFE"
    latency = (time.perf_counter() - t0) * 1000
    logger.info("[response] Done in %.0fms | refused=%s", latency, refused)

    return {
        "final_response": raw,
        "response_refused": refused,
        "refusal_reason": unsafe_reason if refused else None,
        "node_trace": state.node_trace + ["response"],
        "latency_ms": {**state.latency_ms, "response": latency},
    }
