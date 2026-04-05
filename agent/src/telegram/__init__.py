"""Telegram delivery package for PuntLab.

Purpose: expose canonical bot bootstrapping and command handlers for the
primary Telegram client surface.
Scope: aiogram bot construction, dispatcher wiring, and user-facing command
entry points consumed by runtime startup modules.
Dependencies: imported by the agent runtime and Telegram-focused tests.
"""

from src.telegram.bot import (
    BotRunMode,
    TelegramApplication,
    build_bot,
    build_dispatcher,
    create_telegram_application,
    run_telegram_application,
)
from src.telegram.commands import (
    build_bot_commands,
    build_help_text,
    list_command_names,
    parse_slip_number_argument,
    telegram_commands_router,
)

__all__ = [
    "BotRunMode",
    "TelegramApplication",
    "build_bot",
    "build_bot_commands",
    "build_dispatcher",
    "build_help_text",
    "create_telegram_application",
    "list_command_names",
    "parse_slip_number_argument",
    "run_telegram_application",
    "telegram_commands_router",
]
