# validation.py — JSON Schema subset validation for tool inputs.
import re
import inspect


def _value_type_name(value) -> str:
    """Return a JSON-schema-ish type name for a Python value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _schema_type_matches(expected_type: str, value) -> bool:
    """Return True if *value* matches the JSON-schema primitive type."""
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return True


def _validate_schema_value(value, schema: dict, path: str = "input") -> list[str]:
    """Validate a value against a small JSON Schema subset used by tools."""
    if not isinstance(schema, dict):
        return []

    errors = []
    expected_type = schema.get("type")
    if expected_type is None:
        if "properties" in schema or "required" in schema:
            expected_type = "object"
        elif "items" in schema:
            expected_type = "array"

    if expected_type is not None:
        allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_schema_type_matches(schema_type, value) for schema_type in allowed_types):
            expected = " or ".join(str(t) for t in allowed_types)
            return [f"{path} must be {expected}, got {_value_type_name(value)}"]

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        allowed = ", ".join(repr(v) for v in enum_values[:10])
        if len(enum_values) > 10:
            allowed += ", ..."
        errors.append(f"{path} must be one of [{allowed}], got {value!r}")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if min_length is not None and len(value) < min_length:
            errors.append(f"{path} must be at least {min_length} chars, got {len(value)}")
        max_length = schema.get("maxLength")
        if max_length is not None and len(value) > max_length:
            errors.append(f"{path} must be at most {max_length} chars, got {len(value)}")
        pattern = schema.get("pattern")
        if pattern:
            try:
                if re.search(pattern, value) is None:
                    errors.append(f"{path} must match pattern {pattern!r}")
            except re.error:
                pass

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            errors.append(f"{path} must be >= {minimum}, got {value}")
        maximum = schema.get("maximum")
        if maximum is not None and value > maximum:
            errors.append(f"{path} must be <= {maximum}, got {value}")
        exclusive_minimum = schema.get("exclusiveMinimum")
        if exclusive_minimum is not None and value <= exclusive_minimum:
            errors.append(f"{path} must be > {exclusive_minimum}, got {value}")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if exclusive_maximum is not None and value >= exclusive_maximum:
            errors.append(f"{path} must be < {exclusive_maximum}, got {value}")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            errors.append(f"{path} must have at least {min_items} items, got {len(value)}")
        max_items = schema.get("maxItems")
        if max_items is not None and len(value) > max_items:
            errors.append(f"{path} must have at most {max_items} items, got {len(value)}")

        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                errors.extend(_validate_schema_value(item, item_schema, f"{path}[{idx}]"))
                if len(errors) >= 10:
                    break

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for key in required:
            if key not in value:
                errors.append(f"{path}.{key} is required")

        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}.{key} is not allowed")

        for key, prop_schema in properties.items():
            if key in value:
                errors.extend(_validate_schema_value(value[key], prop_schema, f"{path}.{key}"))
                if len(errors) >= 10:
                    break

    return errors


def validate_tool_input(tool_name: str, tool_input, fn, tool_schema: dict | None) -> str | None:
    """Validate incoming tool arguments before execution.

    Returns an error string if validation fails, None if valid.
    """
    if tool_input is None:
        tool_input = {}

    if not isinstance(tool_input, dict):
        return (
            f"Error: Tool {tool_name} expected object input, "
            f"got {_value_type_name(tool_input)}."
        )

    try:
        sig = inspect.signature(fn)
        accepts_kwargs = False
        required = []
        allowed = set()

        for pname, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_kwargs = True
                continue
            if param.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            allowed.add(pname)
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        missing = [pname for pname in required if pname not in tool_input]
        if missing:
            return (
                f"Error: Tool {tool_name} called with missing required parameters: {missing}. "
                f"Received: {list(tool_input.keys())}. "
                f"Try breaking the operation into smaller steps."
            )

        if not accepts_kwargs:
            unexpected = [key for key in tool_input if key not in allowed]
            if unexpected:
                return (
                    f"Error: Tool {tool_name} called with unexpected parameters: {unexpected}. "
                    f"Allowed: {sorted(allowed)}."
                )
    except (ValueError, TypeError):
        pass

    if tool_schema:
        schema_errors = _validate_schema_value(tool_input, tool_schema.get("input_schema", {}))
        if schema_errors:
            preview = "; ".join(schema_errors[:3])
            if len(schema_errors) > 3:
                preview += f"; and {len(schema_errors) - 3} more"
            return f"Error: Tool {tool_name} called with invalid parameters: {preview}"

    return None
