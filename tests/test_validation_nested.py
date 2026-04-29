"""L-5 (codex review 2026-04-29): nested additionalProperties enforcement.

The review flagged a potential gap in ``_validate_schema_value`` —
unknown-key detection only walks ``properties`` from the OUTER schema
at any given recursion level, so nested objects need their OWN
``additionalProperties=False`` to be enforced.  After tracing the code
this is actually correct (each recursive call passes the child schema
through, so the child's ``additionalProperties=False`` gates ITS keys),
but the behaviour wasn't pinned by a regression test.  This module
provides that pin.
"""

from __future__ import annotations

from jyagent.runtime.tools.validation import validate_tool_input


def _fake_fn(**kwargs):  # noqa: ARG001 — signature only matters for **kwargs
    return "ok"


def _schema(input_schema: dict) -> dict:
    return {"name": "fake_tool", "input_schema": input_schema}


class TestNestedAdditionalProperties:
    """Each level's ``additionalProperties=False`` must reject keys not
    declared at THAT level — inner ``additionalProperties=False`` does
    NOT inherit upward, and outer ``additionalProperties=False`` does
    NOT inherit downward.
    """

    def test_inner_additional_property_rejected(self):
        """A nested object with ``additionalProperties=False`` must reject
        an unknown key at its own level."""
        schema = _schema({
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "outer": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "known_inner": {"type": "string"},
                    },
                },
            },
        })
        # Unknown nested key "evil" must be rejected.
        err = validate_tool_input(
            "fake_tool",
            {"outer": {"known_inner": "ok", "evil": "leaked"}},
            _fake_fn,
            schema,
        )
        assert err is not None, "nested additionalProperties=False not enforced"
        assert "outer.evil" in err or "evil" in err

    def test_inner_no_additional_properties_allows_extras(self):
        """When the INNER schema does NOT set ``additionalProperties=False``,
        unknown inner keys are accepted (matches JSON Schema default
        ``additionalProperties=true``).  Default tools rely on this for
        permissive sub-objects."""
        schema = _schema({
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "outer": {
                    "type": "object",
                    # No additionalProperties — defaults to true.
                    "properties": {
                        "known_inner": {"type": "string"},
                    },
                },
            },
        })
        err = validate_tool_input(
            "fake_tool",
            {"outer": {"known_inner": "ok", "extra": "ignored"}},
            _fake_fn,
            schema,
        )
        assert err is None, (
            f"unexpectedly rejected extras when inner schema is permissive: {err}"
        )

    def test_outer_additional_property_still_rejected(self):
        """The outer ``additionalProperties=False`` keeps working even
        when nested objects exist (sanity that the recursion didn't
        accidentally clobber the outer-level enforcement)."""
        schema = _schema({
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "known_inner": {"type": "string"},
                    },
                },
            },
        })
        err = validate_tool_input(
            "fake_tool",
            {"outer": {"known_inner": "ok"}, "outer_extra": "leaked"},
            _fake_fn,
            schema,
        )
        assert err is not None
        # Validator joins the path with dots; assert the outer-level key
        # is named in the error.
        assert "outer_extra" in err

    def test_doubly_nested_additional_property_rejected(self):
        """Three levels deep: ``additionalProperties=False`` at each
        level enforces against keys at THAT level."""
        schema = _schema({
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "leaf": {"type": "integer"},
                            },
                        },
                    },
                },
            },
        })
        err = validate_tool_input(
            "fake_tool",
            {"a": {"b": {"leaf": 1, "smuggled": "x"}}},
            _fake_fn,
            schema,
        )
        assert err is not None, "deeply-nested additionalProperties=False not enforced"
        assert "smuggled" in err

