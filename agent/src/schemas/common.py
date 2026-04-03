"""Shared schema validation helpers for PuntLab's normalized data models.

Purpose: centralize the canonical text, datetime, numeric, and slug
normalization routines reused across fixture, odds, stats, and news schemas.
Scope: pure helper functions only; no runtime state or external integrations.
Dependencies: standard-library validation helpers and Pydantic-aware callers.
"""

from __future__ import annotations

import re
from math import isfinite
from typing import Final

_NON_ALNUM_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def require_non_blank_text(value: str, field_name: str) -> str:
    """Trim a required string and reject blank content.

    Args:
        value: Raw field value supplied to a schema.
        field_name: Field name used in the error message.

    Returns:
        The normalized non-empty string.

    Raises:
        ValueError: If the normalized string is empty.
    """

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank.")
    return normalized


def normalize_optional_text(value: str | None) -> str | None:
    """Trim optional text and collapse empty strings to `None`.

    Args:
        value: Optional raw string value.

    Returns:
        A trimmed string or `None` when the input has no content.
    """

    if value is None:
        return None

    normalized = value.strip()
    return normalized or None


def ensure_timezone_aware[T](value: T, field_name: str) -> T:
    """Reject naive datetimes to keep fixture and article timing deterministic.

    Args:
        value: Value to validate, usually a `datetime`.
        field_name: Field name used in the error message.

    Returns:
        The original value when it is timezone-aware or not a datetime-like
        object with timezone support.

    Raises:
        ValueError: If the provided datetime is naive.
    """

    if hasattr(value, "utcoffset") and callable(value.utcoffset):
        offset = value.utcoffset()
        if offset is None:
            raise ValueError(f"{field_name} must include timezone information.")
    return value


def require_finite_number(value: float, field_name: str) -> float:
    """Ensure a float is finite before it enters normalized schemas.

    Args:
        value: Numeric field value.
        field_name: Field name used in the error message.

    Returns:
        The original float when it is finite.

    Raises:
        ValueError: If the number is NaN or infinite.
    """

    if not isfinite(value):
        raise ValueError(f"{field_name} must be a finite number.")
    return value


def slugify_segment(value: str) -> str:
    """Convert display text into a lowercase URL-safe segment.

    Args:
        value: Display text such as a country or competition name.

    Returns:
        A slug safe for use in sportsbook URL path segments.
    """

    normalized = require_non_blank_text(value, "slug_source").lower()
    collapsed = _NON_ALNUM_PATTERN.sub("-", normalized)
    return collapsed.strip("-")
