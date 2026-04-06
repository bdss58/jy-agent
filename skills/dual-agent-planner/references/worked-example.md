# Worked Example: Dual Planning in Action

A real-world-style walkthrough of using dual planning for a database migration.

## The Task

> "We need to migrate our user service from PostgreSQL to DynamoDB.
> The service has 15 API endpoints, 8 SQLAlchemy models, and serves
> 50k requests/day. We can't have downtime."

## Phase 1: Frame the Problem

```
CONTEXT: Python/FastAPI user service, PostgreSQL via SQLAlchemy,
  15 endpoints in src/api/, 8 models in src/models/, ~50k req/day.
  Tests in tests/ with 85% coverage.

GOAL: Migrate data layer from PostgreSQL to DynamoDB while maintaining
  all API contracts and zero downtime.

CONSTRAINTS:
  - Zero downtime (dual-write migration pattern likely needed)
  - API response format must not change
  - Must maintain test coverage ≥ 85%
  - Timeline: 2 weeks

SUCCESS CRITERIA:
  - All existing API tests pass against DynamoDB backend
  - Load test shows ≤ 10% latency regression at p99
  - Rollback procedure documented and tested
```

## Phase 2: Collect Plans

Both agents are given the same problem statement with "PLAN ONLY."

### Claude Code's Plan (summary)

> **Approach**: Repository pattern with adapter abstraction.
> 1. Create `Repository` protocol classes as abstraction layer
> 2. Implement `PostgresRepository` (wrap existing SQLAlchemy code)
> 3. Implement `DynamoRepository` (new DynamoDB code)
> 4. Add `DualWriteRepository` that writes to both, reads from Postgres
> 5. Deploy dual-write → verify → switch reads to Dynamo → remove Postgres
>
> Files: 12 new, 8 modified. Estimated 4-5 days coding + 3 days testing.
> Risk: DynamoDB access patterns may not map cleanly to relational queries.

### Codex's Plan (summary)

> **Approach**: Feature-flag-driven migration with shadow reads.
> 1. Add `boto3` DynamoDB client alongside SQLAlchemy
> 2. Feature flag per endpoint to switch between backends
> 3. Shadow read phase: query both, compare results, log discrepancies
> 4. Data migration script with CDC (Change Data Capture) from Postgres
> 5. Progressive rollout: flip endpoints one by one
>
> Files: 8 new, 15 modified. Estimated 5-6 days coding + 4 days testing.
> Risk: Feature flags add complexity; shadow reading doubles query load.

## Phase 3: Compare & Synthesize

### Where They Agree (high confidence)
- Need a dual-write/dual-read phase (no big-bang cutover)
- DynamoDB access patterns must be designed upfront
- Rollback plan is critical
- Existing tests must keep passing throughout

### Where They Differ

| Aspect | Claude Code | Codex | My Recommendation |
|--------|------------|-------|-------------------|
| Abstraction | Repository protocol | Feature flags per endpoint | **Repository** — cleaner, reusable, testable |
| Migration granularity | All-at-once switch | Per-endpoint rollout | **Per-endpoint** — safer, easier to debug |
| Data sync | Dual-write from app | CDC from database | **Dual-write** — simpler, no CDC infra needed |
| Shadow validation | Not mentioned | Shadow reads + compare | **Yes, add shadow reads** — critical for trust |

### Synthesized Plan

Combine repository pattern (Claude Code) with per-endpoint rollout (Codex):

1. **Week 1 Days 1-2**: Create `Repository` protocol + `PostgresRepository` wrapper
2. **Week 1 Days 3-4**: Implement `DynamoRepository` + design access patterns
3. **Week 1 Day 5**: `DualWriteRepository` — writes to both, reads from Postgres
4. **Week 2 Days 1-2**: Shadow read validation — compare results, log mismatches
5. **Week 2 Days 3-4**: Per-endpoint flag to switch reads to DynamoDB
6. **Week 2 Day 5**: Load test, rollback drill, documentation

## Phase 4: Execute

- **Claude Code** gets the Repository abstraction layer (Phase 1-2) — this is
  deep multi-file refactoring, its strength.
- **Codex** gets the DualWrite + shadow read infrastructure (Phase 3-4) — this
  is well-scoped infrastructure with clear test criteria.
- **Integration** done together with cross-review.

## Phase 5: Cross-Review

After Claude Code finishes the Repository layer:
```bash
codex exec --sandbox read-only \
  "Review the Repository abstraction in src/repositories/. Check that DynamoDB
   access patterns are correct for our query needs: <list queries>"
```

After Codex finishes the DualWrite layer:
```bash
claude -p --model sonnet --bare --allowedTools "Read" \
  "Review the dual-write and shadow-read implementation. Verify data consistency
   guarantees and check for race conditions."
```

## Outcome

The synthesized plan was better than either individual plan:
- **From Claude Code**: Clean repository abstraction (maintainable, testable)
- **From Codex**: Per-endpoint rollout + shadow reads (safer, observable)
- **Neither alone proposed**: The combination of both approaches
