"""
state.py — Typed state definition for the LangGraph safety evaluation agent.
"""

from __future__ import annotations
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class AgentState(BaseModel):
    """
    Shared state passed between graph nodes.
    All fields are immutable between nodes — nodes return partial updates.
    """

    # Input
    user_prompt: str = Field(description="Original user input")
    system_prompt: Optional[str] = Field(default=None)
    model_name: str = Field(default="phi4:latest")

    # Planner output
    plan: Optional[str] = Field(default=None, description="Structured plan from planner node")
    risk_flags: list[str] = Field(default_factory=list, description="Risk flags identified by planner")
    requires_safety_check: bool = Field(default=False)

    # Reasoning output
    reasoning: Optional[str] = Field(default=None, description="Chain-of-thought from reasoning node")
    safety_assessment: Optional[str] = Field(default=None)

    # Response output
    final_response: Optional[str] = Field(default=None)
    response_refused: bool = Field(default=False)
    refusal_reason: Optional[str] = Field(default=None)

    # Metadata
    node_trace: list[str] = Field(default_factory=list, description="Ordered list of nodes visited")
    latency_ms: dict[str, float] = Field(default_factory=dict, description="Per-node latency")
    error: Optional[str] = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}
