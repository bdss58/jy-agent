#!/usr/bin/env python3
"""Test whether a skill's description triggers correctly for various queries.

Usage:
    python scripts/test_trigger.py <skill-directory> [--eval-set evals.json]
    python scripts/test_trigger.py --help

Tests the skill router (LLM-based) to see if the skill triggers (activates)
for queries it should handle, and doesn't trigger for queries it shouldn't.

If no eval set is provided, generates default test queries from the description.

Eval set format (evals/trigger_evals.json):
    [
        {"query": "create a skill for data analysis", "should_trigger": true},
        {"query": "what's the weather today", "should_trigger": false},
        ...
    ]
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# Add project root to path so we can import jyagent
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # skills/create-skill/scripts/ → project root
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_skill_meta(skill_dir: str) -> dict:
    """Parse skill name and description from SKILL.md frontmatter."""
    skill_md = Path(skill_dir) / "SKILL.md"
    content = skill_md.read_text(encoding='utf-8')

    # Split frontmatter
    content = content.lstrip()
    if not content.startswith('---'):
        raise ValueError("No YAML frontmatter")
    end = re.search(r'\n---\s*\n', content[3:])
    if not end:
        raise ValueError("Unclosed frontmatter")
    fm = content[3:3 + end.start()]

    # Extract name and description
    name = desc = ""
    lines = fm.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('name:'):
            name = line.split(':', 1)[1].strip().strip('"').strip("'")
        elif line.startswith('description:'):
            val = line.split(':', 1)[1].strip()
            if val in ('>-', '>', '|', '|-'):
                # Collect block scalar
                desc_lines = []
                i += 1
                while i < len(lines) and (lines[i].startswith('  ') or not lines[i].strip()):
                    if lines[i].strip():
                        desc_lines.append(lines[i].strip())
                    i += 1
                desc = ' '.join(desc_lines)
                continue
            else:
                desc = val.strip('"').strip("'")
        i += 1

    return {"name": name, "description": desc}


def _generate_default_evals(skill_name: str, description: str) -> list[dict]:
    """Generate reasonable default test queries from the skill's description."""
    # These always-negative queries should never trigger any specific skill
    negative_queries = [
        "what's the weather today",
        "tell me a joke",
        "how do I make pasta carbonara",
        "what time is it in Tokyo",
        "explain quantum computing to a 5 year old",
        "translate 'hello' to French",
    ]

    # Try to extract positive triggers from description
    positive_queries = []

    # Look for "TRIGGER on:" patterns
    trigger_match = re.search(r'TRIGGER\s+on:?\s*(.+?)(?:\.|DO NOT)', description, re.IGNORECASE)
    if trigger_match:
        triggers_text = trigger_match.group(1)
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', triggers_text)
        positive_queries.extend(quoted)

    # Look for "Use this skill when/whenever" patterns
    use_match = re.search(r'(?:use (?:this skill )?when(?:ever)?|activate when)\s+(.+?)(?:\.|$)', description, re.IGNORECASE)
    if use_match:
        context = use_match.group(1)
        # Use context as a query
        if len(context) > 10 and len(context) < 200:
            positive_queries.append(context[:100])

    # Fallback: construct queries from skill name
    if not positive_queries:
        name_words = skill_name.replace('-', ' ')
        positive_queries = [
            f"help me with {name_words}",
            f"I need to do some {name_words}",
            f"can you {name_words}",
        ]

    evals = []
    for q in positive_queries:
        evals.append({"query": q, "should_trigger": True})
    for q in negative_queries[:3]:  # Use 3 negatives
        evals.append({"query": q, "should_trigger": False})

    return evals


def test_trigger(skill_dir: str, eval_set: list[dict], runs_per_query: int = 1,
                 verbose: bool = False) -> dict:
    """
    Test skill triggering using jyagent's SkillManager LLM router.

    For each query, calls auto_activate_for_query() and checks if the target
    skill gets activated.

    Returns results dict with per-query outcomes and summary.
    """
    from jyagent.skills import SkillManager

    meta = _parse_skill_meta(skill_dir)
    target_name = meta["name"]

    # Create a fresh SkillManager with ALL skills (realistic routing scenario)
    mgr = SkillManager()
    mgr.discover()

    if target_name not in mgr._skills:
        raise ValueError(f"Skill '{target_name}' not found. Available: {mgr.list_skills()}")

    results = []

    for item in eval_set:
        query = item["query"]
        should_trigger = item["should_trigger"]
        trigger_count = 0

        for run_idx in range(runs_per_query):
            # Reset: deactivate all skills before each test
            mgr.deactivate_all()

            start = time.time()
            activated = mgr.auto_activate_for_query(query)
            elapsed = time.time() - start

            triggered = target_name in activated

            if triggered:
                trigger_count += 1

            if verbose:
                status = "✅" if (triggered == should_trigger) else "❌"
                print(f"  {status} [{elapsed:.1f}s] query=\"{query[:60]}\" "
                      f"triggered={triggered} (activated: {activated})",
                      file=sys.stderr)

        trigger_rate = trigger_count / runs_per_query
        did_pass = (trigger_rate >= 0.5) if should_trigger else (trigger_rate < 0.5)

        results.append({
            "query": query,
            "should_trigger": should_trigger,
            "trigger_rate": trigger_rate,
            "triggers": trigger_count,
            "runs": runs_per_query,
            "pass": did_pass,
        })

    passed = sum(1 for r in results if r["pass"])
    total = len(results)

    return {
        "skill_name": target_name,
        "description": meta["description"][:200],
        "results": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 2) if total > 0 else 0,
        }
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("skill_dir", help="Path to the skill directory")
    parser.add_argument("--eval-set", help="Path to trigger eval set JSON file")
    parser.add_argument("--runs", type=int, default=1, help="Runs per query (default: 1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-query results")
    parser.add_argument("--output", "-o", help="Write results JSON to file")
    args = parser.parse_args()

    # Load or generate eval set
    if args.eval_set:
        eval_set = json.loads(Path(args.eval_set).read_text())
    else:
        meta = _parse_skill_meta(args.skill_dir)
        eval_set = _generate_default_evals(meta["name"], meta["description"])
        if args.verbose:
            print(f"Generated {len(eval_set)} default test queries", file=sys.stderr)

    results = test_trigger(args.skill_dir, eval_set, runs_per_query=args.runs,
                           verbose=args.verbose)

    # Print summary
    s = results["summary"]
    print(f"\n{'=' * 50}")
    print(f"Trigger Test Results: {results['skill_name']}")
    print(f"{'=' * 50}")
    print(f"Passed: {s['passed']}/{s['total']} ({s['pass_rate']:.0%})")

    if s['failed'] > 0:
        print(f"\nFailed cases:")
        for r in results["results"]:
            if not r["pass"]:
                expected = "trigger" if r["should_trigger"] else "NOT trigger"
                actual = f"triggered {r['triggers']}/{r['runs']}"
                print(f"  ❌ \"{r['query'][:60]}\" — expected {expected}, {actual}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
