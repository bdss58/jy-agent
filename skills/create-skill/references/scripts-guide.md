# Using Scripts in Skills

Source: https://agentskills.io/skill-creation/using-scripts.md

Skills can run shell commands and bundle reusable scripts in `scripts/`.
This guide covers when to use one-off commands vs bundled scripts, how to
make scripts self-contained, and how to design their interfaces for agents.

## One-off commands (no `scripts/` needed)

When an existing package already does the job, reference it directly in
`SKILL.md`. Modern runtimes auto-resolve dependencies at runtime:

| Tool       | Ecosystem  | Notes                                                |
|------------|------------|------------------------------------------------------|
| `uvx`      | Python     | Ships with `uv`. Aggressive caching, very fast.      |
| `pipx run` | Python     | Mature alternative; broader OS package availability.|
| `npx`      | Node.js    | Bundled with npm.                                    |
| `bunx`     | Bun        | Drop-in replacement for `npx` in Bun environments.   |
| `deno run` | Deno       | Requires `--allow-*` flags for fs/network access.    |
| `go run`   | Go         | `go run pkg/path@version`                            |

```bash
uvx ruff@0.8.0 check .
npx eslint@9 --fix .
go run golang.org/x/tools/cmd/goimports@v0.28.0 .
```

**Tips:**
- **Pin versions** (`@0.8.0`, not `@latest`) for reproducibility.
- **State prerequisites** in `SKILL.md`. For runtime-level requirements
  (Python 3.14+, Node 18+, Bun), use the `compatibility` frontmatter field.
- **Promote complex commands to scripts.** When a command grows enough
  flags that it's hard to get right on the first try, a tested script in
  `scripts/` is more reliable.

## Self-contained scripts (PEP 723 / inline deps)

Bundle a script with its own dependencies declared inline. The agent runs
the script with a single command — no separate manifest or install step.

### Python (PEP 723 + `uv run`)

```python
# scripts/extract.py
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "beautifulsoup4>=4.12,<5",
# ]
# ///
from bs4 import BeautifulSoup
import sys

html = sys.stdin.read()
print(BeautifulSoup(html, "html.parser").select_one("p.info").get_text())
```

```bash
uv run scripts/extract.py < page.html
```

`uv run` creates an isolated env, installs declared deps, and runs the
script. `pipx run scripts/extract.py` also supports PEP 723.

For full reproducibility, generate a lockfile with `uv lock --script`.

### Deno (TypeScript with `npm:` / `jsr:` specifiers)

```typescript
// scripts/extract.ts
#!/usr/bin/env -S deno run --allow-read
import * as cheerio from "npm:cheerio@1.0.0";
// …
```

```bash
deno run --allow-read scripts/extract.ts
```

### Bun (auto-installs missing packages)

Bun installs npm packages on-the-fly when no `node_modules` exists, making
single-file scripts trivial — but it's only appropriate for Bun-native
environments.

## Designing scripts for agentic use

### Self-contained, minimal startup

The agent calls the script via shell; assume nothing about the working
directory beyond "the skill root." Use relative paths from the skill
directory, not absolute paths.

### Helpful error messages

The error message directly shapes the agent's next attempt. An opaque
"Error: invalid input" wastes a turn.

```
Error: --format must be one of: json, csv, table.
Received: "xml"
```

### Structured output

Prefer JSON / CSV / TSV over free-form text. The agent (and `jq`, `cut`,
`awk`) can compose with structured output. Send **data** to stdout,
**diagnostics** to stderr.

### `--help` first

Every bundled script should have a useful `--help`. In `SKILL.md`,
instruct the agent: "Run `python scripts/foo.py --help` before using it.
Do not read the source — treat it as a black box."

## Referencing scripts from `SKILL.md`

Use **relative paths from the skill directory root**:

```markdown
## Available scripts
- `scripts/validate.sh` — Validates configuration files
- `scripts/process.py` — Processes input data

## Workflow
1. Validate input:
   ```bash
   bash scripts/validate.sh "$INPUT_FILE"
   ```
2. Process:
   ```bash
   uv run scripts/process.py --input results.json
   ```
```

## When *not* to write a script

If a one-off shell command does the job, skip `scripts/`. Adding a script
for a single `curl | jq` pipeline is overhead. Bundle a script when:

- The logic is non-trivial enough that you want to test it once.
- You see the agent **reinventing the same logic across runs** in eval
  transcripts (charting, format parsing, validation). That's a strong
  signal to bake it into `scripts/`.
- Deterministic behavior matters (the script should produce identical
  output for identical input).
