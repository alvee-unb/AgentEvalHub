"""
agent.py — Builds and runs the LangGraph safety evaluation agent.

Architecture:
    User Input
        ↓
    Planner Node   — identifies intent, risk flags, safety check need
        ↓
    Reasoning Node — chain-of-thought safety assessment
        ↓
    Response Node  — generates final user-facing answer
"""

from __future__ import annotations

import logging
import time
from typing import Any

import src.ssl_fix as _ssl; _ssl.apply()  # must be before ollama import

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from src.state import AgentState
from src.nodes.planner import planner_node
from src.nodes.reasoning import reasoning_node
from src.nodes.response import response_node

logger = logging.getLogger(__name__)


def _make_llm(model_cfg: dict) -> ChatOllama:
    return ChatOllama(
        model=model_cfg["name"],
        base_url=model_cfg.get("base_url", "http://localhost:11434"),
        temperature=model_cfg.get("temperature", 0.1),
        num_predict=model_cfg.get("num_predict", 512),
    )


def build_graph(config: dict) -> Any:
    """
    Construct and compile the LangGraph agent graph.
    Returns a compiled graph ready for .invoke().
    """
    primary_llm = _make_llm(config["models"]["primary"])
    secondary_llm = _make_llm(config["models"]["secondary"])

    # Use primary LLM for planner/response, secondary for reasoning
    def _planner(state: AgentState) -> dict:
        return planner_node(state, primary_llm)

    def _reasoning(state: AgentState) -> dict:
        return reasoning_node(state, secondary_llm)

    def _response(state: AgentState) -> dict:
        return response_node(state, primary_llm)

    def _should_reason(state: AgentState) -> str:
        """Conditional edge: skip reasoning if no safety check needed and no error."""
        if state.error:
            return "response"
        return "reasoning" if state.requires_safety_check else "response"

    # Build graph
    builder = StateGraph(AgentState)
    builder.add_node("planner",   _planner)
    builder.add_node("reasoning", _reasoning)
    builder.add_node("response",  _response)

    builder.set_entry_point("planner")
    builder.add_conditional_edges("planner", _should_reason, {
        "reasoning": "reasoning",
        "response":  "response",
    })
    builder.add_edge("reasoning", "response")
    builder.add_edge("response",  END)

    return builder.compile()


def run_agent(
    prompt: str,
    config: dict,
    model_override: str | None = None,
    system_prompt: str | None = None,
) -> AgentState:
    """
    Run the agent on a single prompt.

    Args:
        prompt:          User input string.
        config:          Loaded YAML config dict.
        model_override:  Override the primary model name.
        system_prompt:   Optional system prompt context.

    Returns:
        Final AgentState after all nodes have executed.
    """
    graph = build_graph(config)

    model_name = model_override or config["models"]["primary"]["name"]
    initial = AgentState(
        user_prompt=prompt,
        system_prompt=system_prompt,
        model_name=model_name,
    )

    logger.info("Running agent | model=%s | prompt_len=%d", model_name, len(prompt))
    t0 = time.perf_counter()

    result = graph.invoke(initial)

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Agent complete | total=%.0fms | nodes=%s | refused=%s",
        total_ms,
        result.get("node_trace", []),
        result.get("response_refused", False),
    )

    # Return as AgentState
    return AgentState(**result)
