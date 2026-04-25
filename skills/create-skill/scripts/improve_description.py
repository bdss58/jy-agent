#!/usr/bin/env python3
"""Improve a skill description based on trigger eval results.

Usage:
    python scripts/improve_description.py <skill-dir> [--iterations N] [--eval-set FILE]
    python scripts/improve_description.py --help

Takes trigger test results and uses an LLM to generate an improved description.
Can run multiple iterations, keeping the best-scoring version.

Flow:
  1. Run trigger tests with current description
  2. If not perfect, call LLM to improve description based on failures
  3. Re-test with new description
  4. Repeat until max iterations or perfect score
  5. Save best description back to SKILL.md
"""

import json
import os
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from test_trigger import test_trigger, _parse_skill_meta, _generate_default_evals


def _call_llm(prompt: str, model: str = None) -> str:
    """Call Anthropic API to improve description."""
    import anthropic

    model = model or os.environ.get("SKILL_ROUTER_MODEL", "claude-sonnet-4-20250514")
    client = anthropic.Anthropic()

    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def improve_description(
    skill_name: str,
    current_description: str,
    eval_results: dict,
    history: list[dict],
    skill_body_summary: str = "",
) -> str:
    """Call LLM to generate an improved description based on eval failures."""

    failed_triggers = [r for r in eval_results["results"]
                       if r["should_trigger"] and not r["pass"]]
    false_triggers = [r for r in eval_results["results"]
                      if not r["should_trigger"] and not r["pass"]]

    score = f"{eval_results['summary']['passed']}/{eval_results['summary']['total']}"

    prompt = f"""You are optimizing a skill description for an agent skill called "{skill_name}".

The description appears in the agent's "available_skills" list. A routing LLM decides
whether to activate the skill based solely on this description. Your goal: trigger for
relevant queries, don't trigger for irrelevant ones.

Current description:
"{current_description}"

Current score: {score}
"""

    if failed_triggers:
        prompt += "\nFAILED TO TRIGGER (should have triggered but didn't):\n"
        for r in failed_triggers:
            prompt += f'  - "{r["query"]}" (triggered {r["triggers"]}/{r["runs"]} times)\n'

    if false_triggers:
        prompt += "\nFALSE TRIGGERS (triggered but shouldn't have):\n"
        for r in false_triggers:
            prompt += f'  - "{r["query"]}" (triggered {r["triggers"]}/{r["runs"]} times)\n'

    if history:
        prompt += "\nPREVIOUS ATTEMPTS (don't repeat — try something structurally different):\n"
        for h in history:
            prompt += f'  Score {h["score"]}: "{h["description"][:150]}..."\n'

    if skill_body_summary:
        prompt += f"\nSkill purpose (for context): {skill_body_summary}\n"

    prompt += """
Guidelines:
- Keep description under 200 words and 1024 characters
- Use imperative voice: "Use this skill when..." not "This skill does..."
- Focus on user intent, not implementation details
- Be "pushy" — describe specific trigger contexts
- Include TRIGGER and DO NOT TRIGGER guidance
- Generalize from failures — don't overfit to specific test queries
- Be distinctive — the description competes with other skills

Respond with ONLY the new description text inside <description> tags."""

    response = _call_llm(prompt)
    match = re.search(r'<description>(.*?)</description>', response, re.DOTALL)
    if match:
        desc = match.group(1).strip().strip('"')
    else:
        desc = response.strip().strip('"')

    # Enforce 1024 char limit
    if len(desc) > 1024:
        desc = desc[:1020] + "..."

    return desc


def _update_skill_description(skill_dir: str, new_description: str):
    """Write the new description back to SKILL.md, preserving everything else."""
    skill_md = Path(skill_dir) / "SKILL.md"
    content = skill_md.read_text(encoding='utf-8')

    # Find the frontmatter block
    if not content.lstrip().startswith('---'):
        raise ValueError("No frontmatter in SKILL.md")

    # Strategy: replace the description field in frontmatter
    # This is a simplified approach — find description and replace up to next top-level key
    lines = content.split('\n')
    new_lines = []
    i = 0
    in_frontmatter = False
    in_description = False
    description_written = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == '---':
            if not in_frontmatter:
                in_frontmatter = True
                new_lines.append(line)
                i += 1
                continue
            else:
                # End of frontmatter
                if in_description and not description_written:
                    # Write new description before closing ---
                    new_lines.append(f"description: >-")
                    # Wrap at ~80 chars with 2-space indent
                    words = new_description.split()
                    current_line = " "
                    for word in words:
                        if len(current_line) + len(word) + 1 > 78:
                            new_lines.append(current_line)
                            current_line = f"  {word}"
                        else:
                            current_line += f" {word}"
                    if current_line.strip():
                        new_lines.append(current_line)
                    description_written = True
                in_frontmatter = False
                new_lines.append(line)
                i += 1
                continue

        if in_frontmatter:
            if stripped.startswith('description:'):
                in_description = True
                # Write new description
                new_lines.append(f"description: >-")
                words = new_description.split()
                current_line = " "
                for word in words:
                    if len(current_line) + len(word) + 1 > 78:
                        new_lines.append(current_line)
                        current_line = f"  {word}"
                    else:
                        current_line += f" {word}"
                if current_line.strip():
                    new_lines.append(current_line)
                description_written = True
                # Skip old description lines
                i += 1
                while i < len(lines):
                    next_stripped = lines[i].strip()
                    if next_stripped == '---' or (next_stripped and not lines[i].startswith(' ') and ':' in next_stripped):
                        break
                    i += 1
                continue
            elif in_description and (line.startswith('  ') or not stripped):
                i += 1
                continue
            else:
                in_description = False
                new_lines.append(line)
        else:
            new_lines.append(line)

        i += 1

    skill_md.write_text('\n'.join(new_lines), encoding='utf-8')


def run_improvement_loop(skill_dir: str, eval_set: list[dict] = None,
                         max_iterations: int = 3, verbose: bool = False,
                         dry_run: bool = False) -> dict:
    """
    Run the description improvement loop.

    Returns dict with best description, score history, and final results.
    """
    meta = _parse_skill_meta(skill_dir)
    skill_name = meta["name"]
    original_desc = meta["description"]

    if eval_set is None:
        eval_set = _generate_default_evals(skill_name, original_desc)

    # Initial test
    print(f"🧪 Testing current description for '{skill_name}'...")
    results = test_trigger(skill_dir, eval_set, verbose=verbose)
    s = results["summary"]
    print(f"   Score: {s['passed']}/{s['total']} ({s['pass_rate']:.0%})")

    history = []
    best_desc = original_desc
    best_score = s["pass_rate"]
    best_results = results

    history.append({
        "iteration": 0,
        "description": original_desc,
        "score": f"{s['passed']}/{s['total']}",
        "pass_rate": s["pass_rate"],
    })

    if s["pass_rate"] == 1.0:
        print("✅ Perfect score — no improvement needed!")
        return {
            "skill_name": skill_name,
            "best_description": best_desc,
            "best_score": best_score,
            "iterations": history,
            "final_results": best_results,
        }

    # Read first 5 lines of body for context
    skill_md = Path(skill_dir) / "SKILL.md"
    content = skill_md.read_text()
    body_start = content.split('---', 2)[-1].strip()[:500]

    for iteration in range(1, max_iterations + 1):
        print(f"\n🔄 Iteration {iteration}/{max_iterations} — improving description...")

        new_desc = improve_description(
            skill_name=skill_name,
            current_description=best_desc,
            eval_results=results,
            history=history,
            skill_body_summary=body_start,
        )

        print(f"   New description ({len(new_desc)} chars):")
        print(f"   \"{new_desc[:120]}...\"")

        if not dry_run:
            # Temporarily update SKILL.md
            _update_skill_description(skill_dir, new_desc)

            # Re-discover skills (description changed)
            from jyagent.skills import get_skill_manager
            mgr = get_skill_manager()
            mgr.discover()

        # Test new description
        results = test_trigger(skill_dir, eval_set, verbose=verbose)
        s = results["summary"]
        print(f"   Score: {s['passed']}/{s['total']} ({s['pass_rate']:.0%})")

        history.append({
            "iteration": iteration,
            "description": new_desc,
            "score": f"{s['passed']}/{s['total']}",
            "pass_rate": s["pass_rate"],
        })

        if s["pass_rate"] > best_score:
            best_desc = new_desc
            best_score = s["pass_rate"]
            best_results = results
            print(f"   📈 New best! ({best_score:.0%})")

        if s["pass_rate"] == 1.0:
            print(f"\n✅ Perfect score reached at iteration {iteration}!")
            break

    # Apply best description
    if not dry_run and best_desc != original_desc:
        _update_skill_description(skill_dir, best_desc)
        from jyagent.skills import get_skill_manager
        get_skill_manager().discover()
        print(f"\n✅ Best description applied (score: {best_score:.0%})")
    elif best_desc == original_desc:
        # Restore original if no improvement
        _update_skill_description(skill_dir, original_desc)
        from jyagent.skills import get_skill_manager
        get_skill_manager().discover()
        print(f"\n📌 Original description kept (no improvement found)")

    return {
        "skill_name": skill_name,
        "best_description": best_desc,
        "best_score": best_score,
        "original_description": original_desc,
        "original_score": history[0]["pass_rate"],
        "iterations": history,
        "final_results": best_results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("skill_dir", help="Path to the skill directory")
    parser.add_argument("--eval-set", help="Path to trigger eval set JSON file")
    parser.add_argument("--iterations", "-n", type=int, default=3, help="Max improvement iterations")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes to SKILL.md")
    parser.add_argument("--output", "-o", help="Write full results JSON to file")
    args = parser.parse_args()

    eval_set = None
    if args.eval_set:
        eval_set = json.loads(Path(args.eval_set).read_text())

    results = run_improvement_loop(
        args.skill_dir,
        eval_set=eval_set,
        max_iterations=args.iterations,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
