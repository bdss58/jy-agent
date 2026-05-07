"""Shared low-level utilities used across multiple subpackages.

This package is the home for helpers that have no logical owner — anything
that ``memory`` and ``tools`` both need, anything provider-neutral that
``runtime`` and ``llm`` would otherwise duplicate, etc.

Rule of thumb: a helper belongs here only if at least two unrelated
subpackages call it.  Otherwise keep it private to its owning module.

Public API is re-exported here for short import paths.
"""
from .files import atomic_write

__all__ = ["atomic_write"]
