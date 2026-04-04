# Orchestration Patterns

How to coordinate Claude Code and Codex beyond simple "ask both, pick one."

## Pattern A: Consult → Synthesize → Single Agent Executes

The default pattern. Get two plans, synthesize, hand to one agent.

```
1. Frame problem
2. claude -p "PLAN ONLY: ..."     → Plan A
3. codex exec "PLAN ONLY: ..."    → Plan B  (parallel with step 2)
4. Compare and synthesize best plan
5. Hand synthesized plan to best-fit agent for execution
6. Verify results
```

**When to use**: Most dual-planning scenarios. Low overhead, clear ownership.

## Pattern B: Split Execution

Divide the work between agents based on their strengths.

```
1. Frame problem, collect dual plans
2. Synthesize and identify subtasks
3. Claude Code → subtask A (e.g., core business logic, complex refactor)
4. Codex → subtask B (e.g., tests, CI config, docs)
5. Integrate both outputs
6. Cross-review (optional)
```

**When to use**: Task has clearly separable parts. Common splits:
- Backend (Claude Code) / Frontend (Codex)
- Implementation (one) / Tests (other)
- Code changes (one) / Infra/CI changes (other)

**Critical rule**: Never let both agents modify the same files simultaneously.
Use separate branches or clearly partition files.

## Pattern C: Implement → Cross-Review → Fix

One builds, the other audits. Quality over speed.

```
1. Frame problem, collect dual plans
2. Pick primary agent → implement
3. Other agent reviews the implementation (read-only)
4. Primary agent fixes review findings
5. Final verification
```

**When to use**: High-stakes changes — auth, payments, data migrations.
The cross-review catches model-specific blind spots.

**Tip**: Give the reviewer the original plan, not just "review this code."
They can check for plan deviations, not just code quality.

## Pattern D: Competitive Implementation

Both agents implement independently. Compare results. Pick the winner.

```
1. Frame problem
2. git checkout -b feat/approach-a && git checkout -b feat/approach-b
3. Claude Code implements on branch approach-a
4. Codex implements on branch approach-b
5. Compare: code quality, test coverage, approach elegance
6. Pick the better one (or merge best parts from both)
```

**When to use**: Rarely — it's 2x cost. But valuable when:
- Truly uncertain which approach is better
- The task is a self-contained module (not touching shared files)
- Learning how each agent handles a problem type

## Pattern E: Cascade (Escalation)

Start with the fast/cheap agent. Escalate to the stronger one only if needed.

```
1. Codex (fast profile) attempts the task
2. If result is good → done
3. If result is poor or task is harder than expected →
   Claude Code (opus) tackles it with Codex's partial work as context
```

**When to use**: Uncertain task difficulty. Start cheap, escalate if needed.
Saves cost on easy tasks, brings the big gun for hard ones.

## Choosing a Pattern

| Scenario | Pattern | Cost | Quality |
|----------|---------|------|---------|
| Standard planning | A (Consult) | 2x plan, 1x exec | Good |
| Separable subtasks | B (Split) | 1x plan, 1x exec each | Good |
| High-stakes changes | C (Cross-Review) | 2x plan, 1.5x exec | Best |
| Truly uncertain approach | D (Competitive) | 2x plan, 2x exec | Best but expensive |
| Unknown difficulty | E (Cascade) | 1-2x exec | Good, cost-efficient |
