# JSON Schemas

This document defines the JSON schemas used by create-skill's scripts.

## Trigger Eval Set (input to test_trigger.py)

File: `evals/trigger_evals.json`

```json
[
    {
        "query": "create a skill for data analysis",
        "should_trigger": true
    },
    {
        "query": "what's the weather today",
        "should_trigger": false
    }
]
```

**Fields:**
- `query`: The user message to test
- `should_trigger`: Whether the skill should activate for this query

## Trigger Test Results (output of test_trigger.py)

```json
{
    "skill_name": "my-skill",
    "description": "Current skill description...",
    "results": [
        {
            "query": "create a skill for data analysis",
            "should_trigger": true,
            "trigger_rate": 1.0,
            "triggers": 3,
            "runs": 3,
            "pass": true
        }
    ],
    "summary": {
        "total": 10,
        "passed": 8,
        "failed": 2,
        "pass_rate": 0.80
    }
}
```

## Improvement History (output of improve_description.py)

```json
{
    "skill_name": "my-skill",
    "best_description": "The optimized description...",
    "best_score": 1.0,
    "original_description": "The original description...",
    "original_score": 0.80,
    "iterations": [
        {
            "iteration": 0,
            "description": "original...",
            "score": "8/10",
            "pass_rate": 0.80
        },
        {
            "iteration": 1,
            "description": "improved...",
            "score": "10/10",
            "pass_rate": 1.0
        }
    ],
    "final_results": { "...same as trigger test results..." }
}
```

## Grading Results (grader agent output)

```json
{
    "expectations": [
        {
            "text": "The output includes a valid SKILL.md",
            "passed": true,
            "evidence": "Found SKILL.md with name='data-analysis'"
        }
    ],
    "summary": {
        "passed": 2,
        "failed": 1,
        "total": 3,
        "pass_rate": 0.67
    },
    "claims": [
        {
            "claim": "Includes decision tree routing",
            "verified": true,
            "evidence": "Found tree pattern at line 15"
        }
    ],
    "eval_feedback": {
        "suggestions": [
            {
                "assertion": "SKILL.md is valid",
                "reason": "Too broad — empty body would pass"
            }
        ]
    }
}
```

## Validation Result (validate_skill.py)

Printed as text, not JSON. Exit code 0 = passed, 1 = failed.

```
✅ Validation passed (1 warnings, 2 suggestions)
WARNINGS:
  ⚠️  description lacks trigger guidance
SUGGESTIONS:
  💡 Consider adding anti-patterns section
```
