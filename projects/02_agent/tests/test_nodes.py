"""test_nodes.py — Unit tests for planner, reasoning, response nodes using mocked LLMs."""

import pytest
from unittest.mock import MagicMock

from src.state import AgentState
from src.nodes.planner import planner_node, parse_planner_output
from src.nodes.reasoning import reasoning_node, parse_reasoning_output
from src.nodes.response import response_node, _detect_refusal


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


def base_state(**kwargs) -> AgentState:
    return AgentState(user_prompt="test prompt for evaluation", **kwargs)


# ── parse_planner_output ──────────────────────────────────────────────────────

class TestParsePlannerOutput:
    def test_parses_risk_flags(self):
        text = "RISK_FLAGS: injection, jailbreak\nREQUIRES_SAFETY_CHECK: YES\nPLAN: Check carefully."
        flags, check, plan = parse_planner_output(text)
        assert "injection" in flags
        assert "jailbreak" in flags
        assert check is True

    def test_none_risk_flags(self):
        text = "RISK_FLAGS: none\nREQUIRES_SAFETY_CHECK: NO\nPLAN: Answer directly."
        flags, check, plan = parse_planner_output(text)
        assert flags == []
        assert check is False

    def test_plan_extracted(self):
        text = "RISK_FLAGS: none\nREQUIRES_SAFETY_CHECK: NO\nPLAN: Provide safe helpful answer."
        _, _, plan = parse_planner_output(text)
        assert "safe" in plan.lower()

    def test_malformed_output_graceful(self):
        flags, check, plan = parse_planner_output("random text with no structure")
        assert isinstance(flags, list)
        assert isinstance(check, bool)


# ── parse_reasoning_output ────────────────────────────────────────────────────

class TestParseReasoningOutput:
    def test_parses_unsafe(self):
        text = "REASONING: This is harmful.\nSAFETY_ASSESSMENT: UNSAFE\nUNSAFE_REASON: Contains malware request."
        reasoning, assessment, reason = parse_reasoning_output(text)
        assert assessment == "UNSAFE"
        assert "malware" in reason

    def test_parses_safe(self):
        text = "REASONING: No issues.\nSAFETY_ASSESSMENT: SAFE\nUNSAFE_REASON: N/A"
        _, assessment, _ = parse_reasoning_output(text)
        assert assessment == "SAFE"

    def test_parses_borderline(self):
        text = "REASONING: Ambiguous.\nSAFETY_ASSESSMENT: BORDERLINE\nUNSAFE_REASON: N/A"
        _, assessment, _ = parse_reasoning_output(text)
        assert assessment == "BORDERLINE"

    def test_fallback_full_text_as_reasoning(self):
        text = "Some free-form response with no structure at all"
        reasoning, _, _ = parse_reasoning_output(text)
        assert reasoning == text


# ── planner_node ──────────────────────────────────────────────────────────────

class TestPlannerNode:
    def test_returns_plan(self):
        llm = make_llm("INTENT: test\nRISK_FLAGS: none\nREQUIRES_SAFETY_CHECK: NO\nPLAN: Answer safely.")
        state = base_state()
        result = planner_node(state, llm)
        assert "plan" in result
        assert "node_trace" in result
        assert "planner" in result["node_trace"]

    def test_adds_latency(self):
        llm = make_llm("INTENT: x\nRISK_FLAGS: none\nREQUIRES_SAFETY_CHECK: NO\nPLAN: x")
        result = planner_node(base_state(), llm)
        assert "planner" in result["latency_ms"]
        assert result["latency_ms"]["planner"] >= 0

    def test_detects_risk_flags(self):
        llm = make_llm("RISK_FLAGS: injection, jailbreak\nREQUIRES_SAFETY_CHECK: YES\nPLAN: Refuse.")
        result = planner_node(base_state(), llm)
        assert result["risk_flags"] != []
        assert result["requires_safety_check"] is True

    def test_llm_error_returns_error_key(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("connection refused")
        result = planner_node(base_state(), llm)
        assert "error" in result
        assert "planner" in result["error"]


# ── reasoning_node ────────────────────────────────────────────────────────────

class TestReasoningNode:
    def test_returns_safety_assessment(self):
        llm = make_llm("REASONING: Looks fine.\nSAFETY_ASSESSMENT: SAFE\nUNSAFE_REASON: N/A")
        state = base_state(plan="Answer normally.", risk_flags=[])
        result = reasoning_node(state, llm)
        assert result["safety_assessment"] == "SAFE"
        assert "reasoning" in result["node_trace"]

    def test_unsafe_assessment(self):
        llm = make_llm("REASONING: Harmful.\nSAFETY_ASSESSMENT: UNSAFE\nUNSAFE_REASON: Malware.")
        state = base_state(plan="Refuse.", risk_flags=["malware"])
        result = reasoning_node(state, llm)
        assert result["safety_assessment"] == "UNSAFE"

    def test_llm_error_returns_error_key(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("timeout")
        result = reasoning_node(base_state(), llm)
        assert "error" in result


# ── response_node ─────────────────────────────────────────────────────────────

class TestResponseNode:
    def test_safe_response_not_refused(self):
        llm = make_llm("Water boils at 100 degrees Celsius at sea level.")
        state = base_state(safety_assessment="SAFE", reasoning="No issues.")
        result = response_node(state, llm)
        assert result["final_response"] is not None
        assert result["response_refused"] is False

    def test_refusal_detected(self):
        llm = make_llm("I cannot provide instructions for that.")
        state = base_state(safety_assessment="UNSAFE", reasoning="Harmful content.")
        result = response_node(state, llm)
        assert result["response_refused"] is True

    def test_response_in_node_trace(self):
        llm = make_llm("Here is my answer.")
        state = base_state(safety_assessment="SAFE")
        result = response_node(state, llm)
        assert "response" in result["node_trace"]


# ── _detect_refusal ───────────────────────────────────────────────────────────

class TestDetectRefusal:
    def test_detects_cannot(self):
        assert _detect_refusal("I cannot help with that.")
    def test_detects_refuse(self):
        assert _detect_refusal("I refuse to answer this question.")
    def test_detects_unable(self):
        assert _detect_refusal("I'm unable to assist with this request.")
    def test_safe_text_not_refused(self):
        assert not _detect_refusal("Water boils at 100 degrees Celsius.")
    def test_case_insensitive(self):
        assert _detect_refusal("I CANNOT provide that information.")
