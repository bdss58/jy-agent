# tests/test_runtime_llm_independence.py — Pins the no-llm-imports
# runtime/LLM independence invariant.
#
# The runtime package must be importable without eagerly loading any
# module from ``jyagent.llm``.  A regression here would put us back in
# the "engine imports LLMOwner" state the LLMClient Protocol +
# runtime-owned llm_types move were supposed to fix.
#
# Implementation note: we run the assertion in a subprocess.  An
# in-process version would have to reset ``sys.modules`` (drop every
# pre-imported jyagent module so the next import is fresh), but that
# poisons module-level references held by other test files —
# ``mock.patch("jyagent.runtime.loop.verification.VERIFICATION_ENABLED", True)``
# silently fails because the patched module is no longer the one the
# test's ``should_verify`` import points at.  Subprocess isolation
# avoids the cross-test contamination entirely.

from __future__ import annotations

import subprocess
import sys
import textwrap


class TestRuntimeLLMIndependence:
    def test_importing_runtime_does_not_load_llm(self):
        """``import jyagent.runtime`` must not pull in ``jyagent.llm``.

        Runtime defines the contracts (LLMClient Protocol, LLMOptions,
        ModelSpec); the llm package implements them.  A stray import of
        ``jyagent.llm`` (directly, or via a top-level module that pulls
        it in transitively) breaks the dependency direction and
        prevents test fakes / alternative providers from driving the
        runtime without the real SDK installed.
        """
        script = textwrap.dedent(
            """
            import sys
            import jyagent.runtime  # noqa: F401
            leaked = sorted(n for n in sys.modules if n.startswith("jyagent.llm"))
            if leaked:
                print("LEAK:" + ",".join(leaked))
                sys.exit(1)
            sys.exit(0)
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, (
            f"jyagent.runtime pulled in jyagent.llm — dependency direction "
            f"reversed again.\n  stdout: {proc.stdout}\n  stderr: {proc.stderr}"
        )

    def test_llmoptions_modelspec_canonical_home_is_runtime(self):
        """After the 2026-04 move, ``LLMOptions`` and ``ModelSpec`` live
        in ``jyagent.runtime.loop.llm_types``.  ``jyagent.llm.types``
        re-exports them for backward compat.  All four historical
        import paths must resolve to the same object.

        Run in subprocess for the same reason as the import-leak test
        above — keeps test ordering side-effects out of the picture.
        """
        script = textwrap.dedent(
            """
            import sys
            from jyagent.runtime.loop.llm_types import LLMOptions, ModelSpec
            from jyagent.runtime import LLMOptions as L2, ModelSpec as M2
            from jyagent.llm.types import LLMOptions as L3, ModelSpec as M3
            from jyagent.llm import LLMOptions as L4, ModelSpec as M4
            assert LLMOptions is L2 is L3 is L4, "LLMOptions identity broke"
            assert ModelSpec is M2 is M3 is M4, "ModelSpec identity broke"
            assert LLMOptions.__module__ == "jyagent.runtime.loop.llm_types", (
                f"Canonical home is runtime.loop.llm_types, got "
                f"{LLMOptions.__module__}"
            )
            sys.exit(0)
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, (
            f"LLMOptions/ModelSpec identity check failed.\n"
            f"  stdout: {proc.stdout}\n  stderr: {proc.stderr}"
        )
