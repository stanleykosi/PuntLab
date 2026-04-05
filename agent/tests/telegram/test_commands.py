"""Tests for PuntLab Telegram command metadata and argument parsing.

Purpose: verify command registration metadata, help text shape, and `/slip`
argument parsing rules before runtime bot wiring.
Scope: unit tests for pure helpers in `src.telegram.commands`.
Dependencies: pytest plus aiogram command data models.
"""

from __future__ import annotations

import pytest
from src.telegram.commands import (
    build_bot_commands,
    build_help_text,
    list_command_names,
    parse_slip_number_argument,
)


def test_build_bot_commands_includes_all_expected_commands() -> None:
    """Command menu builder should expose every configured command once."""

    commands = build_bot_commands()
    assert [command.command for command in commands] == list(list_command_names())
    assert len(commands) == 8


def test_build_help_text_includes_usage_lines_for_all_commands() -> None:
    """Help text should list each command with its public usage string."""

    help_text = build_help_text()
    for command_name in list_command_names():
        if command_name == "slip":
            assert "/slip <number>" in help_text
            continue
        assert f"/{command_name}" in help_text


@pytest.mark.parametrize(
    ("raw_args", "expected"),
    [
        ("1", 1),
        ("42", 42),
        ("   7   ", 7),
    ],
)
def test_parse_slip_number_argument_accepts_positive_integer_tokens(
    raw_args: str,
    expected: int,
) -> None:
    """Parser should accept one positive integer token."""

    assert parse_slip_number_argument(raw_args) == expected


@pytest.mark.parametrize(
    "raw_args",
    [
        None,
        "",
        " ",
        "0",
        "-1",
        "abc",
        "1 2",
    ],
)
def test_parse_slip_number_argument_rejects_invalid_shapes(raw_args: str | None) -> None:
    """Parser should fail fast for missing, malformed, and non-positive inputs."""

    with pytest.raises(ValueError):
        parse_slip_number_argument(raw_args)
