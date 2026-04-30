# tests/test_stats_concurrency.py — Concurrency invariants for the
# session-stats pricing table.
#
# ``set_model_pricing`` used to mutate the inner
# ``_MODEL_PRICING[provider]`` dict without a lock, while
# ``_lookup_pricing`` iterates over the same dict's items().  Under concurrent
# registration (e.g. an MCP-driven extension that adds models after sub-agents
# are already running), this can raise ``RuntimeError: dictionary changed size
# during iteration``.
#
# This test reproduces the race deterministically — it fails on the
# pre-fix code in well under a second on CPython 3.14, and passes after
# the lock is added in stats.py.

from __future__ import annotations

import threading
import time

import pytest

from jyagent.runtime import stats as st


class TestPricingTableConcurrency:
    def test_concurrent_set_and_lookup_does_not_raise(self):
        """High-contention writer + reader pair must not raise.

        On CPython 3.14 with the GIL this race is theoretical — single
        dict ops are bytecode-atomic and the C-level ``sorted()`` doesn't
        yield mid-iteration.  But on free-threaded CPython (PEP 703) and
        with multiple concurrent writers, the lock matters.  This test's
        primary value is twofold: it pins the no-deadlock invariant, and
        it guards against a future refactor that drops the lock and
        introduces a real race on free-threaded builds.
        """
        stop = threading.Event()
        errors: list[BaseException] = []

        def _writer() -> None:
            i = 0
            while not stop.is_set():
                try:
                    st.set_model_pricing(
                        "race-test", f"model-{i % 64}", (1.0, 2.0)
                    )
                    i += 1
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)
                    return

        def _reader() -> None:
            while not stop.is_set():
                try:
                    st._lookup_pricing("race-test", "model-31-suffix")
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)
                    return

        threads = [
            threading.Thread(target=_writer, name="pricing-writer"),
            threading.Thread(target=_reader, name="pricing-reader"),
            threading.Thread(target=_reader, name="pricing-reader-2"),
        ]
        for t in threads:
            t.start()

        # Run for a short window; the bug fires within ~10ms on the
        # pre-fix code, so 0.5 s is plenty of headroom while keeping the
        # test fast.
        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive(), f"thread {t.name} did not exit"

        assert not errors, (
            f"Concurrent set_model_pricing / _lookup_pricing raised: "
            f"{[type(e).__name__ + ': ' + str(e) for e in errors]}"
        )

    def test_set_model_pricing_is_visible_to_lookup(self):
        """Sanity: the lock doesn't break the basic read-after-write
        contract."""
        st.set_model_pricing("vis-test", "alpha", (3.0, 4.0))
        p = st._lookup_pricing("vis-test", "alpha-large")
        assert p is not None
        assert p.input_per_million == 3.0
        assert p.output_per_million == 4.0
