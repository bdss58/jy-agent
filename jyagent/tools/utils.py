# Utility functions shared by tool modules.

from ..config import BINARY_EXTS


def strip_unsupported_schema_keys(properties: dict) -> dict:
    """Strip keys not supported by Bedrock's JSON Schema validator (e.g., 'default')."""
    unsupported_keys = {"default"}
    cleaned = {}
    for prop_name, prop_def in properties.items():
        if isinstance(prop_def, dict):
            cleaned[prop_name] = {k: v for k, v in prop_def.items() if k not in unsupported_keys}
        else:
            cleaned[prop_name] = prop_def
    return cleaned
