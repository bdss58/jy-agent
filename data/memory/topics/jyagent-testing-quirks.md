---
created: 2026-04-25T11:47:42+08:00
updated: 2026-04-25T11:47:42+08:00
---
# jyagent Testing Quirks

Collected gotchas hit while writing tests against the `jyagent` package.
Read this when writing or debugging tests under `tests/`.

## Function-name shadowing in `jyagent/tools/__init__.py`

The package re-exports tool functions at the top level for ergonomic
imports (e.g. `from jyagent.tools import web_fetch`). This **shadows the
submodule of the same name** — `jyagent.tools.web_fetch` resolves to the
*function*, not the module, even after `import jyagent.tools.web_fetch`.

### Symptom

```python
import jyagent.tools.web_fetch as web_fetch_mod
monkeypatch.setattr(web_fetch_mod, "_fetch_cffi", lambda url, **kw: (200, html))
# AttributeError: <function web_fetch at 0x...> has no attribute '_fetch_cffi'
```

### Fix

Resolve the actual module object via `sys.modules` after triggering the
import (which still populates `sys.modules` correctly):

```python
import sys
import jyagent.tools.web_fetch  # noqa: F401  — load into sys.modules
web_fetch_mod = sys.modules["jyagent.tools.web_fetch"]
monkeypatch.setattr(web_fetch_mod, "_fetch_cffi", lambda url, **kw: (200, html))
```

### Affected modules

Any submodule whose function is re-exported in `jyagent/tools/__init__.py`.
Confirmed: `web_fetch`. Likely: any other tool with a top-level re-export.
Check `jyagent/tools/__init__.py` before assuming `import jyagent.tools.X`
gives you the module.

### Don't "fix" the package

The re-export is intentional ergonomics for tool authors / agent code.
The fix lives in the *test*, not in `__init__.py`.

---

(Add further jyagent test gotchas below as they're discovered.)
