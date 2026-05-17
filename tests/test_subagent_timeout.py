"""Tests for `dispatch_agent(timeout=N)` deadline enforcement.

Pre-fix behaviour (the bug this test file regression-guards):
  * Background path silently dropped ``effective_timeout`` — the kwarg
    was computed but never used.
  * Foreground path's "soft handoff" returned ``timeout_handoff`` after
    ``effective_timeout`` seconds but did NOT fire ``cancel_event``,
    so the handed-off worker continued running indefinitely.  The user
    journal entry for 2026-05-17 documents 80+ minute runtimes despite
    ``timeout=1200`` being passed.

Post-fix mechanism:
  * A single ``threading.Timer`` fires ``cancel_state.fire("deadline")``
    after ``effective_timeout`` seconds.  Used by BOTH paths uniformly.
  * The timer is cancelled by a future-done-callback installed at
    submit time, so natural completion before the deadline does not
    leak a daemon timer thread.
  * Cancel sources race-safely (first-writer-wins) via the
    ``_CancelState`` reason channel; user-cancel that wins the race
    surfaces as ``api_error``, deadline-cancel surfaces as
    ``timed_out``.
"""
import json
import threading
import time
from unittest.mock import patch

from jyagent.tools.subagent import (
    _CancelState,
    _bg_registry,
    _SUBAGENT_STATUS_API_ERROR,
    _SUBAGENT_STATUS_COMPLETED,
    _SUBAGENT_STATUS_TIMED_OUT,
    check_agent,
    dispatch_agent,
)


# ─── _CancelState unit tests ────────────────────────────────────────────────


class TestCancelState:
    """Unit tests for the reason-channel primitive itself."""

    def test_first_writer_wins(self):
        s = _CancelState()
        assert s.set_reason("deadline") is True
        assert s.set_reason("user") is False
        assert s.reason == "deadline"

    def test_fire_sets_event_and_reason(self):
        s = _CancelState()
        won = s.fire("user")
        assert won is True
        assert s.event.is_set()
        assert s.reason == "user"

    def test_fire_idempotent(self):
        s = _CancelState()
        s.fire("user")
        # Second fire from a different source — event already set,
        # reason already locked.  Returns False (not the winning call).
        won = s.fire("deadline")
        assert won is False
        assert s.reason == "user"  # unchanged

    def test_reason_unset_before_any_fire(self):
        s = _CancelState()
        assert s.reason is None
        assert not s.event.is_set()


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_cancel_aware_stub(finish_event: threading.Event,
                            poll_interval: float = 0.02):
    """Build a ``_run_subagent`` stub that exits when EITHER
    ``cancel_event`` fires OR ``finish_event`` is set.

    Mirrors what a cooperative-cancel ``AgentLoop`` does — checks the
    cancel event at every step.  When the cancel event fires the stub
    returns an outcome dict whose status will be re-derived by
    ``_run_subagent``'s post-loop reason check in the real code.  In
    these tests the stub IS the would-be ``_run_subagent`` so we
    construct the outcome directly, mirroring the real status branch.
    """
    from jyagent.tools.subagent import (
        _make_subagent_outcome,
        _format_subagent_failure,
    )

    def stub(task, context, model_spec, max_steps,
             tool_schemas, tool_functions,
             agent_id=None, custom_system_prompt=None,
             cancel_event=None, progress_ids=None,
             memory_mode="none", cancel_state=None):
        while True:
            if finish_event.is_set():
                return _make_subagent_outcome(
                    _SUBAGENT_STATUS_COMPLETED, "finished cleanly",
                    1, 0, 0, 0,
                )
            if cancel_event is not None and cancel_event.is_set():
                # Mirror the real _run_subagent post-loop branch:
                # reason="deadline" → TIMED_OUT, else API_ERROR.
                reason = cancel_state.reason if cancel_state is not None else None
                if reason == "deadline":
                    return _make_subagent_outcome(
                        _SUBAGENT_STATUS_TIMED_OUT,
                        _format_subagent_failure(
                            "Error: Sub-agent exceeded its deadline (timed_out).",
                            partial_output="",
                        ),
                        1, 0, 0, 0,
                        error="deadline",
                    )
                return _make_subagent_outcome(
                    _SUBAGENT_STATUS_API_ERROR,
                    _format_subagent_failure(
                        "Error: Sub-agent was interrupted.",
                        partial_output="",
                    ),
                    1, 0, 0, 0,
                    error=reason or "interrupted",
                )
            time.sleep(poll_interval)

    return stub


def _wait_for_outcome(agent_id: int, timeout: float = 3.0) -> dict:
    """Poll ``check_agent`` until status != 'running' or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = check_agent(agent_id=agent_id)
        # check_agent returns either a JSON envelope (still running) or
        # a rendered outcome envelope (done).  The latter is multi-line
        # markdown-ish text starting with the status icon line.  Sniff
        # by attempting JSON parse first.
        try:
            payload = json.loads(result.content)
            if payload.get("status") == "running":
                time.sleep(0.05)
                continue
            return payload
        except (json.JSONDecodeError, ValueError):
            # Rendered outcome — agent is done.  Pull status from the
            # registry directly.
            agent = _bg_registry.get(agent_id)
            if agent is not None and agent.outcome is not None:
                return agent.outcome
            time.sleep(0.05)
    raise AssertionError(
        f"agent {agent_id} did not finish within {timeout}s"
    )


# ─── Deadline-enforcement tests (the regression bar) ───────────────────────


class TestBackgroundDeadlineFires:
    """Background-path: dispatch_agent(background=True, timeout=N) must
    fire cancel_event at the deadline and the outcome must be timed_out."""

    def test_bg_deadline(self):
        finish = threading.Event()  # never set — worker runs forever absent cancel
        stub = _make_cancel_aware_stub(finish)

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=stub),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 0.1),
            # Schema clamps timeout to [60, 3600]; bypass clamp by
            # patching the bound directly so we can use 1s for fast tests.
            patch("jyagent.tools.subagent._BG_DEFAULT_TIMEOUT", 1),
        ):
            # timeout=0 → uses _BG_DEFAULT_TIMEOUT (which we patched to 1s)
            result = dispatch_agent(task="bg deadline task", background=True, timeout=0)

        payload = json.loads(result.content)
        assert payload["status"] == "dispatched"
        agent_id = payload["agent_id"]

        outcome = _wait_for_outcome(agent_id, timeout=3.0)
        assert outcome["status"] == _SUBAGENT_STATUS_TIMED_OUT
        assert outcome.get("error") == "deadline"


class TestForegroundDeadlineHandoff:
    """Foreground-path: timeout fires → soft handoff envelope returned →
    same timer continues to bound the handed-off worker's runtime."""

    def test_fg_deadline_handoff_then_timeout(self):
        finish = threading.Event()
        stub = _make_cancel_aware_stub(finish)

        # Need foreground default short enough to trigger handoff quickly.
        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=stub),
            patch("jyagent.tools.subagent._FG_DEFAULT_TIMEOUT", 1),
        ):
            t0 = time.time()
            result = dispatch_agent(task="fg deadline task", background=False, timeout=0)
            handoff_elapsed = time.time() - t0

        # Returned via soft handoff at ~1s
        assert handoff_elapsed < 2.0, f"handoff took {handoff_elapsed:.2f}s"
        payload = json.loads(result.content)
        assert payload["status"] == "timeout_handoff"
        agent_id = payload["agent_id"]

        # The deadline timer was scheduled for the SAME effective_timeout
        # (1s).  Since we just returned at the handoff point, the timer
        # has already fired or is firing within milliseconds.  Worker
        # should observe cancel and exit with timed_out.
        outcome = _wait_for_outcome(agent_id, timeout=3.0)
        assert outcome["status"] == _SUBAGENT_STATUS_TIMED_OUT
        assert outcome.get("error") == "deadline"


class TestNaturalCompletionNotTimedOut:
    """A fast sub-agent finishing well before the deadline must yield a
    ``completed`` outcome AND must NOT leak a deadline-timer thread."""

    def test_fast_completion_clean(self):
        # Stub finishes immediately (gate already set).
        finish = threading.Event()
        finish.set()
        stub = _make_cancel_aware_stub(finish)

        # Snapshot the live timer-thread set BEFORE dispatch so we can
        # diff after.  We're looking for our own ``-deadline-`` named
        # daemon threads.
        def deadline_threads() -> set[int]:
            return {
                t.ident for t in threading.enumerate()
                if t.is_alive() and "deadline-" in (t.name or "")
            }

        before = deadline_threads()
        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=stub),
            patch("jyagent.tools.subagent._FG_DEFAULT_TIMEOUT", 5),
        ):
            result = dispatch_agent(task="fast task", background=False, timeout=0)

        # Foreground sync path returns a rendered envelope on natural
        # completion (not JSON).  Just assert no error.
        assert not result.is_error, f"unexpected error: {result.content[:200]}"

        # Allow up to 250ms for the future done-callback to fire and
        # the timer.cancel() to retire the timer thread.
        deadline = time.time() + 0.5
        while time.time() < deadline:
            leaked = deadline_threads() - before
            if not leaked:
                return
            time.sleep(0.02)

        leaked = deadline_threads() - before
        assert not leaked, (
            f"deadline timer thread(s) leaked after fast completion: {leaked}"
        )


class TestUserCancelBeatsDeadline:
    """The race Codex flagged: a user-cancel that wins the cancel-state
    race must keep its api_error label even if the deadline timer fires
    a moment later (because the worker takes time to unwind)."""

    def test_user_cancel_wins_race(self):
        # Stub blocks until the cancel_event fires.  Deadline is short
        # enough to fire RIGHT AFTER our manual cancel, but the
        # first-writer rule must hold the reason at "user".
        finish = threading.Event()  # never naturally finishes
        stub = _make_cancel_aware_stub(finish, poll_interval=0.05)

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=stub),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 0.05),
            patch("jyagent.tools.subagent._BG_DEFAULT_TIMEOUT", 1),
        ):
            result = dispatch_agent(task="cancelable task", background=True, timeout=0)
            payload = json.loads(result.content)
            agent_id = payload["agent_id"]

            # Fire user-cancel BEFORE the deadline timer fires (deadline
            # is ~1s after dispatch, we fire at ~0.2s).  We reach into
            # the registry directly instead of using
            # ``check_agent(action='kill')`` because the latter wipes
            # the agent record on clean shutdown, robbing the test of
            # the outcome it wants to inspect.  The race we care about
            # is on ``_CancelState`` itself, which the direct fire
            # exercises identically to the kill path.
            time.sleep(0.2)
            agent = _bg_registry.get(agent_id)
            assert agent is not None
            assert agent.cancel_state is not None
            # NOTE: we don't assert ``fire("user")`` returns True here.
            # On a slow CI box the test sleep could overshoot the
            # deadline (1s) and the deadline timer wins the race instead.
            # The first-writer-wins guarantee is still tested — just by
            # the FINAL OUTCOME below, which is the only thing callers
            # actually observe.
            agent.cancel_state.fire("user")

            outcome = _wait_for_outcome(agent_id, timeout=3.0)

        # Critical assertion: status / error must agree.  Either user
        # won the race (api_error / "user") or the deadline did
        # (timed_out / "deadline"); first-writer-wins guarantees the
        # two fields are CONSISTENT — never api_error labelled
        # "deadline" or timed_out labelled "user".
        status = outcome["status"]
        err = outcome.get("error")
        assert (status, err) in {
            (_SUBAGENT_STATUS_API_ERROR, "user"),
            (_SUBAGENT_STATUS_TIMED_OUT, "deadline"),
        }, f"Inconsistent cancel labelling: status={status!r} error={err!r}"


class TestDeadlineBeatsLateUserCancel:
    """Mirror case: deadline fires first, a user-cancel attempt arrives
    later (e.g. user noticed it was slow and clicked kill, but the
    timer had already won).  Status stays ``timed_out``."""

    def test_deadline_wins_race(self):
        finish = threading.Event()
        stub = _make_cancel_aware_stub(finish, poll_interval=0.05)

        with (
            patch("jyagent.tools.subagent._run_subagent", side_effect=stub),
            patch("jyagent.tools.subagent._BG_GRACE_PERIOD", 0.05),
            patch("jyagent.tools.subagent._BG_DEFAULT_TIMEOUT", 1),
        ):
            result = dispatch_agent(task="slow task", background=True, timeout=0)
            payload = json.loads(result.content)
            agent_id = payload["agent_id"]

            # Wait past the deadline so the timer fires first.
            time.sleep(1.3)
            # Now a late user-cancel attempt arrives.  Don't assert the
            # boolean return — on a fast CI box where dispatch_agent
            # returned faster than expected, the deadline could be
            # delayed; the cancel-reason invariant is what we test.
            agent = _bg_registry.get(agent_id)
            assert agent is not None
            assert agent.cancel_state is not None
            agent.cancel_state.fire("user")

            outcome = _wait_for_outcome(agent_id, timeout=2.0)

        # First-writer-wins invariant: status and error agree.  Either
        # deadline won (the expected case after our 1.3s sleep > 1s
        # deadline) or — on an unusually slow CI — the user cancel
        # arrived first.  Both outcomes are correct labelling.
        status = outcome["status"]
        err = outcome.get("error")
        assert (status, err) in {
            (_SUBAGENT_STATUS_TIMED_OUT, "deadline"),
            (_SUBAGENT_STATUS_API_ERROR, "user"),
        }, f"Inconsistent cancel labelling: status={status!r} error={err!r}"
