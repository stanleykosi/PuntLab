"""Telegram command handlers for PuntLab's primary client surface.

Purpose: define and register canonical user-facing bot commands, including
registration on `/start` and informational responses for delivery workflows.
Scope: command metadata, argument validation, user registration lookup, and
message handlers for `/start`, `/today`, `/slip`, `/history`, `/stats`,
`/subscribe`, and `/help`.
Dependencies: aiogram routing primitives plus PuntLab's async database
connection and SQLAlchemy user model for Telegram registration.
"""

from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import BotCommand, Message
from aiogram.types import User as TelegramUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.connection import get_session
from src.db.models import User

telegram_commands_router = Router(name="telegram-commands")

_COMMAND_SPECS: tuple[tuple[str, str], ...] = (
    ("start", "Welcome and register your Telegram profile"),
    ("today", "View today's accumulator recommendations"),
    ("slip", "View details for a specific slip number"),
    ("history", "View recent recommendation history"),
    ("stats", "View PuntLab performance statistics"),
    ("subscribe", "View subscription plans and upgrade options"),
    ("admin", "Open admin controls (admin-only)"),
    ("help", "Show usage instructions"),
)


def build_bot_commands() -> tuple[BotCommand, ...]:
    """Build the canonical Telegram command menu for PuntLab.

    Outputs:
        Ordered bot commands configured during bot startup so users can
        discover supported command handlers from Telegram's command UI.
    """

    return tuple(
        BotCommand(command=command, description=description)
        for command, description in _COMMAND_SPECS
    )


def build_help_text() -> str:
    """Build a concise help message covering all currently wired commands.

    Outputs:
        Multi-line help text suitable for direct Telegram message delivery.
    """

    lines = ["PuntLab Command Guide:"]
    for command, description in _COMMAND_SPECS:
        command_usage = f"/{command} <number>" if command == "slip" else f"/{command}"
        lines.append(f"{command_usage} - {description}")
    lines.append("")
    lines.append("Use /start first to register your Telegram account.")
    return "\n".join(lines)


def parse_slip_number_argument(raw_args: str | None) -> int:
    """Parse and validate `/slip` command arguments.

    Inputs:
        raw_args: Raw argument string from aiogram's `CommandObject.args`.

    Outputs:
        A positive integer slip number.

    Raises:
        ValueError: If the argument is missing, malformed, or non-positive.
    """

    if raw_args is None:
        raise ValueError("Usage: /slip <number>")

    tokens = raw_args.strip().split()
    if len(tokens) != 1:
        raise ValueError("Usage: /slip <number>")

    token = tokens[0]
    if not token.isdigit():
        raise ValueError("Slip number must be a positive integer.")

    slip_number = int(token)
    if slip_number <= 0:
        raise ValueError("Slip number must be greater than zero.")
    return slip_number


@telegram_commands_router.message(CommandStart())
async def handle_start_command(message: Message) -> None:
    """Register the sender and return a welcome message.

    Inputs:
        message: Telegram message event for `/start`.

    Behavior:
        - Upserts the sender into the users table keyed by `telegram_id`
        - Replies with registration status and next-step commands
    """

    telegram_user = message.from_user
    if telegram_user is None:
        await message.answer("Unable to identify your Telegram profile. Please try again.")
        return

    try:
        registered_user = await _register_or_update_user(telegram_user)
    except Exception:
        await message.answer(
            "Registration failed due to a temporary database issue. Please try again shortly."
        )
        return

    await message.answer(
        "Welcome to PuntLab.\n"
        f"Your profile is registered on the `{registered_user.subscription_tier}` tier.\n"
        "Use /today for the latest slips or /help to see all commands."
    )


@telegram_commands_router.message(Command("today"))
async def handle_today_command(message: Message) -> None:
    """Handle `/today` requests for the sender's current subscription tier."""

    registered_user = await _load_registered_user_for_message(message)
    if registered_user is None:
        return

    await message.answer(
        "Today's recommendations are prepared by tier.\n"
        f"Current tier: `{registered_user.subscription_tier}`.\n"
        "Live slip formatting is enabled in the next delivery step."
    )


@telegram_commands_router.message(Command("slip"))
async def handle_slip_command(message: Message, command: CommandObject) -> None:
    """Handle `/slip <number>` lookups with strict argument validation."""

    registered_user = await _load_registered_user_for_message(message)
    if registered_user is None:
        return

    try:
        slip_number = parse_slip_number_argument(command.args)
    except ValueError as exc:
        await message.answer(str(exc))
        return

    await message.answer(
        f"Slip #{slip_number} details are not yet linked in this step.\n"
        f"Request accepted for tier `{registered_user.subscription_tier}`."
    )


@telegram_commands_router.message(Command("history"))
async def handle_history_command(message: Message) -> None:
    """Handle `/history` requests for registered users."""

    registered_user = await _load_registered_user_for_message(message)
    if registered_user is None:
        return

    await message.answer(
        "History retrieval is active in the API and web roadmap.\n"
        f"Your current tier is `{registered_user.subscription_tier}`."
    )


@telegram_commands_router.message(Command("stats"))
async def handle_stats_command(message: Message) -> None:
    """Handle `/stats` requests for registered users."""

    registered_user = await _load_registered_user_for_message(message)
    if registered_user is None:
        return

    await message.answer(
        "Performance metrics are available after settlement is enabled.\n"
        f"You are registered on tier `{registered_user.subscription_tier}`."
    )


@telegram_commands_router.message(Command("subscribe"))
async def handle_subscribe_command(message: Message) -> None:
    """Handle `/subscribe` plan information requests."""

    registered_user = await _load_registered_user_for_message(message)
    if registered_user is None:
        return

    await message.answer(
        "Subscription plans:\n"
        "- free: 1 daily accumulator\n"
        "- plus: up to 10 accumulators\n"
        "- elite: full slate access\n"
        f"Current tier: `{registered_user.subscription_tier}`."
    )


@telegram_commands_router.message(Command("help"))
async def handle_help_command(message: Message) -> None:
    """Return the canonical command help text."""

    await message.answer(build_help_text())


async def _load_registered_user_for_message(message: Message) -> User | None:
    """Load the sender's registered profile or emit registration guidance.

    Inputs:
        message: Incoming Telegram message from a command handler.

    Outputs:
        The persisted `User` row for the sender, or `None` when unavailable.
    """

    telegram_user = message.from_user
    if telegram_user is None:
        await message.answer("Unable to identify your Telegram profile. Please try again.")
        return None

    try:
        registered_user = await _get_user_by_telegram_id(telegram_user.id)
    except Exception:
        await message.answer(
            "Unable to load your profile due to a temporary database issue. Please try again."
        )
        return None

    if registered_user is None:
        await message.answer("Please run /start first so we can register your profile.")
        return None

    return registered_user


async def _register_or_update_user(telegram_user: TelegramUser) -> User:
    """Create or update the Telegram user profile in the database.

    Inputs:
        telegram_user: Telegram sender object provided by aiogram.

    Outputs:
        Persisted `User` row representing the sender.
    """

    async with get_session() as session:
        user = await _get_user_by_telegram_id_in_session(session, telegram_user.id)
        if user is None:
            user = User(
                telegram_id=telegram_user.id,
                telegram_username=_normalize_username(telegram_user.username),
                display_name=_extract_display_name(telegram_user),
                subscription_tier="free",
                subscription_status="active",
                is_admin=False,
            )
            session.add(user)
        else:
            user.telegram_username = _normalize_username(telegram_user.username)
            user.display_name = _extract_display_name(telegram_user)

        try:
            await session.flush()
            await session.commit()
        except Exception:
            await session.rollback()
            raise

        return user


async def _get_user_by_telegram_id(telegram_id: int) -> User | None:
    """Fetch a user by Telegram ID using a managed async session."""

    async with get_session() as session:
        return await _get_user_by_telegram_id_in_session(session, telegram_id)


async def _get_user_by_telegram_id_in_session(
    session: AsyncSession,
    telegram_id: int,
) -> User | None:
    """Fetch a user by Telegram ID in the provided async session."""

    statement = select(User).where(User.telegram_id == telegram_id)
    result = await session.execute(statement)
    return result.scalar_one_or_none()


def _extract_display_name(telegram_user: TelegramUser) -> str:
    """Return the best available display name for a Telegram sender."""

    full_name = telegram_user.full_name.strip()
    if full_name:
        return full_name
    fallback_name = telegram_user.first_name.strip()
    return fallback_name or f"user-{telegram_user.id}"


def _normalize_username(username: str | None) -> str | None:
    """Normalize optional Telegram usernames into compact lowercase form."""

    if username is None:
        return None
    normalized = username.strip().lstrip("@")
    if not normalized:
        return None
    return normalized.lower()


def list_command_names() -> tuple[str, ...]:
    """Return the canonical command names used by handlers and tests."""

    return tuple(command for command, _ in _COMMAND_SPECS)


__all__ = [
    "build_bot_commands",
    "build_help_text",
    "handle_help_command",
    "handle_history_command",
    "handle_slip_command",
    "handle_start_command",
    "handle_stats_command",
    "handle_subscribe_command",
    "handle_today_command",
    "list_command_names",
    "parse_slip_number_argument",
    "telegram_commands_router",
]
