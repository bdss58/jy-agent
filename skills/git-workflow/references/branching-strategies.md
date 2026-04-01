# Branching Strategies: Detailed Comparison

## Trunk-Based Development

**Best for:** Web apps, SaaS, small teams (1-5), continuous deployment.

```
main ──●──●──●──●──●──●──●──●──●──●──●──●──────
       |     |     |     |     |     |
       └─●──●┘    └──●──●──┘    └──●────┘
       (hours)     (1 day)      (hours)
```

### Rules
- Feature branches live **< 2 days** (ideally < 24 hours)
- PRs are **< 400 lines** changed
- `main` is **always deployable**
- CI must pass before merge
- Feature flags for incomplete features

### Workflow
```bash
# 1. Start work
git checkout main && git pull
git checkout -b yourname/add-search-filter

# 2. Small, focused commits
git commit -m "feat(search): add search input component"
git commit -m "feat(search): add filter logic to product grid"

# 3. Push and create PR (same day)
git push -u origin yourname/add-search-filter
# Create PR, get review, merge same day

# 4. On merge, CI/CD deploys to production automatically
```

### Prerequisites
- Strong CI/CD pipeline (tests run < 10 min)
- Feature flags (LaunchDarkly, Unleash, or DIY)
- Code review culture (PRs reviewed within 4 hours)
- Good test coverage (> 80%)

---

## GitHub Flow

**Best for:** Teams of 5-20, regular releases, GitHub/GitLab hosted projects.

```
main ──────●───────────●───────────●──────────
           ↑           ↑           ↑
       PR merge    PR merge    PR merge
           |           |           |
feature/A ─┘  feature/B ─┘  feature/C ─┘
```

### Rules
- One long-lived branch: `main` (always deployable)
- Feature branches for all changes (named: `feature/`, `fix/`, `docs/`)
- All changes via Pull Request
- CI + review required before merge
- Deploy from `main` after merge

### Workflow
```bash
# 1. Create feature branch
git checkout main && git pull
git checkout -b feature/user-dashboard

# 2. Work on feature (1-5 days)
git commit -m "feat(dashboard): add layout skeleton"
git commit -m "feat(dashboard): integrate chart component"

# 3. Keep branch updated
git fetch origin
git rebase origin/main  # or merge

# 4. Push and open PR
git push -u origin feature/user-dashboard
# Open PR with description, link issues

# 5. After review + CI, merge to main
# Squash merge for clean history, or merge commit to preserve details
```

---

## GitFlow

**Best for:** Versioned releases, mobile apps, libraries, on-premise software, LTS support.

```
main     ──v1.0──────────────v2.0──────────v2.0.1──
              \              / ↑              / ↑
develop  ──────────────────────┼────────────────┼──
            \    /  \    /     |    \      /    |
feature/A ───┘    feature/B ──┘     \    /     |
                                  release/2.0   |
                                              hotfix/2.0.1
```

### Branches
| Branch | Purpose | Lifetime |
|--------|---------|----------|
| `main` | Production releases, tagged | Permanent |
| `develop` | Integration branch | Permanent |
| `feature/*` | New features | Days-weeks |
| `release/*` | Release stabilization | Days |
| `hotfix/*` | Production bug fixes | Hours-days |

### Workflow
```bash
# Feature development
git checkout develop
git checkout -b feature/user-dashboard
# ... work ...
git checkout develop
git merge --no-ff feature/user-dashboard
git branch -d feature/user-dashboard

# Create a release
git checkout develop
git checkout -b release/2.1.0
# Bug fixes only on release branch
git commit -m "fix: dashboard loading state"
# Merge to main AND develop
git checkout main
git merge --no-ff release/2.1.0
git tag v2.1.0
git checkout develop
git merge --no-ff release/2.1.0

# Hotfix
git checkout main
git checkout -b hotfix/2.1.1
git commit -m "fix(auth): patch XSS vulnerability"
git checkout main
git merge --no-ff hotfix/2.1.1
git tag v2.1.1
git checkout develop
git merge --no-ff hotfix/2.1.1
```

### When GitFlow is overkill
- If you deploy continuously (use trunk-based instead)
- If you have < 3 developers (too much ceremony)
- If you don't maintain multiple versions

---

## Comparison Table

| Factor | Trunk-Based | GitHub Flow | GitFlow |
|--------|-------------|-------------|---------|
| Branch lifetime | Hours | 1-5 days | Days-weeks |
| Merge frequency | Multiple/day | Daily-weekly | Weekly-monthly |
| Merge conflicts | Rare, small | Occasional | Common, painful |
| Deploy frequency | Multiple/day | Daily-weekly | Weekly-monthly |
| Feature flags needed | Essential | Helpful | Optional |
| CI/CD requirements | Strong | Moderate | Basic |
| Code review speed | < 4 hours | < 1 day | Can be slow |
| Team size sweet spot | 1-10 | 5-20 | 10+ |
| Risk per deploy | Low (small changes) | Medium | Higher (big batches) |
| Best for | Web apps, SaaS | General | Versioned releases |
| DORA metrics correlation | High performance | Medium | Lower |

---

## Migration: GitFlow → Trunk-Based

If your team wants to move from GitFlow to trunk-based:

1. **Start with GitHub Flow** as intermediate step (drop develop branch)
2. **Shorten branch lifetimes** gradually: 2 weeks → 1 week → 2 days → 1 day
3. **Invest in CI/CD** — tests must be fast and reliable
4. **Introduce feature flags** before dropping long-lived branches
5. **Set PR size limits** — reject PRs > 400 lines
6. **Measure** — track deployment frequency, lead time, change failure rate (DORA metrics)
