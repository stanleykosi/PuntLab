"""Telegram delivery package for PuntLab.

Purpose: expose canonical bot bootstrapping and command handlers for the
primary Telegram client surface.
Scope: aiogram bot construction, dispatcher wiring, and user-facing command
entry points consumed by runtime startup modules.
Dependencies: imported by the agent runtime and Telegram-focused tests.
"""

from src.telegram.admin import (
    build_admin_keyboard,
    is_admin_telegram_id,
    parse_accumulator_action,
    register_manual_pipeline_trigger,
    telegram_admin_router,
)
from src.telegram.bot import (
    BotRunMode,
    TelegramApplication,
    build_bot,
    build_dispatcher,
    create_telegram_application,
    run_telegram_application,
)
from src.telegram.broadcast import broadcast_daily, send_to_user
from src.telegram.commands import (
    build_bot_commands,
    build_help_text,
    list_command_names,
    parse_slip_number_argument,
    telegram_commands_router,
)
from src.telegram.formatters import (
    format_accumulator_message,
    format_history_message,
    format_stats_message,
    format_welcome_message,
)

__all__ = [
    "BotRunMode",
    "TelegramApplication",
    "build_bot",
    "build_admin_keyboard",
    "build_bot_commands",
    "build_dispatcher",
    "build_help_text",
    "broadcast_daily",
    "create_telegram_application",
    "format_accumulator_message",
    "format_history_message",
    "format_stats_message",
    "format_welcome_message",
    "is_admin_telegram_id",
    "list_command_names",
    "parse_accumulator_action",
    "parse_slip_number_argument",
    "register_manual_pipeline_trigger",
    "run_telegram_application",
    "send_to_user",
    "telegram_admin_router",
    "telegram_commands_router",
]
