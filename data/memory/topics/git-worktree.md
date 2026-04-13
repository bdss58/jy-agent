# Git Worktree Workflow for Self-Upgrades

## Rule (set by Jianyong on 2026-04-01)
**Always use `git worktree` when modifying jy-agent's own code** (self-upgrades).
This prevents the running agent from being disrupted by in-flight changes to its own source.

## Workflow

### 1. Create worktree branch
```bash
cd /Users/jyxc-dz-0100398/jy-agent
git worktree add ../jy-agent-upgrade upgrade/description -b upgrade/description
cd ../jy-agent-upgrade
```

### 2. Make changes in worktree
- Edit files in the worktree directory (../jy-agent-upgrade/)
- Test changes there
- Commit with conventional commit messages

### 3. Merge back to main
```bash
cd /Users/jyxc-dz-0100398/jy-agent  # back to main worktree
git merge upgrade/description
# Or: git merge --squash upgrade/description  (for single commit)
```

### 4. Clean up
```bash
git worktree remove ../jy-agent-upgrade
git branch -d upgrade/description
```

## Why worktree (not regular branch)?
- Regular branch checkout in the main directory would modify files the running agent is using
- Worktree creates a separate directory with its own working tree
- The agent keeps running on main while changes are made in parallel
- No risk of partial file states breaking the running agent

## Past usage
- 2026-04-01: Used worktree to fix planner.py issues, merged back to main successfully