---
language:
- en
license: apache-2.0
task_categories:
- text-classification
tags:
- ai-safety
- red-teaming
- evaluation
- benchmark
pretty_name: AI Safety Benchmark Dataset
size_categories:
- n<1K
---

# AI Safety Benchmark Dataset

A curated benchmark dataset for evaluating AI model safety across six risk categories.
Designed for use with Inspect AI, MLflow, and LangGraph evaluation pipelines.

## Dataset Summary

| Field | Value |
|---|---|
| Records | 50 |
| Categories | 6 |
| Harmful Intent | 29 |
| Benign | 21 |
| Mean Risk Score | 7.48 / 10 |
| License | Apache-2.0 |
| Language | English |

## Category Distribution

| Category | Count |
|---|---|
| Prompt Injection | 9 |
| Jailbreak | 9 |
| Hallucination | 8 |
| Tool Abuse | 8 |
| Data Leakage | 8 |
| Unsafe Advice | 8 |

## Severity Distribution

| Severity | Count |
|---|---|
| High | 26 |
| Medium | 13 |
| Critical | 10 |
| Low | 1 |

## Schema

```python
{
  "id":                        str,   # UUIDv4
  "version":                   str,
  "category":                  str,   # See categories below
  "severity":                  str,   # low | medium | high | critical
  "tags":                      str,   # pipe-separated evaluation tags
  "prompt":                    str,   # Input prompt for model evaluation
  "system_prompt":             str | None,
  "expected_behaviour":        str,   # refuse | answer_safely | detect_inject | stay_grounded
  "harmful_intent":            bool,
  "ground_truth_safe_response":str | None,
  "risk_score":                float, # 0.0 – 10.0
  "attack_vector":             str | None,
  "source":                    str,
  "created_at":                str,   # ISO-8601 UTC
  "notes":                     str | None
}
```

## Usage

```python
import json

with open("data/processed/ai_safety_benchmark.json") as f:
    records = json.load(f)

# Filter by category
injections = [r for r in records if r["category"] == "prompt_injection"]
```

## Categories

| ID | Description |
|---|---|
| `prompt_injection` | Attempts to override model instructions via crafted inputs |
| `jailbreak` | Techniques designed to bypass safety training |
| `hallucination` | Queries designed to elicit fabricated or unverifiable information |
| `tool_abuse` | Misuse of tool-calling capabilities for harmful ends |
| `data_leakage` | Attempts to extract system prompts, PII, or configuration |
| `unsafe_advice` | Requests for advice that could cause real-world harm |

## Citation

```bibtex
@dataset{ai_safety_benchmark_2025,
  title  = {AI Safety Benchmark Dataset},
  author = {AgentEvalHub},
  year   = {2025},
  note   = {Synthetic dataset for AI evaluation research}
}
```
