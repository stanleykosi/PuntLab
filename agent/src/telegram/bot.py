"""Telegram bot bootstrap and runtime wiring for PuntLab.

Purpose: construct the aiogram bot + dispatcher pair, register command
routers, and provide polling/webhook startup utilities for the agent runtime.
Scope: bot factory, dispatcher composition, lifecycle hook registration, and
execution helpers for polling or webhook registration modes.
Dependencies: aiogram primitives, PuntLab settings, and command metadata from
`src.telegram.commands`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties

from src.config import get_settings
from src.telegram.admin import build_admin_router
from src.telegram.commands import build_bot_commands, telegram_commands_router


class BotRunMode(StrEnum):
    """Supported Telegram runtime registration modes."""

    POLLING = "polling"
    WEBHOOK = "webhook"


@dataclass(slots=True, frozen=True)
class TelegramApplication:
    """Container for the canonical aiogram bot runtime components.

    Inputs:
        Built aiogram `Bot` and `Dispatcher` instances.

    Outputs:
        Immutable application bundle that runtime callers can pass to
        execution helpers without reconstructing dependencies.
    """

    bot: Bot
    dispatcher: Dispatcher


def build_bot(*, token: str | None = None) -> Bot:
    """Create the canonical aiogram `Bot` instance for PuntLab.

    Inputs:
        token: Optional explicit bot token. When omitted, this factory loads
            the token from runtime settings (`TELEGRAM_BOT_TOKEN`).

    Outputs:
        Configured aiogram bot with HTML parse mode enabled by default.

    Raises:
        ValueError: If no token is available.
    """

    resolved_token = token or get_settings().telegram.bot_token
    if not resolved_token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN is required to initialize the Telegram bot."
        )

    return Bot(
        token=resolved_token,
        default=DefaultBotProperties(parse_mode="HTML"),
    )


def build_dispatcher(
    *,
    command_router: Router | None = None,
    admin_router: Router | None = None,
) -> Dispatcher:
    """Create the canonical dispatcher with command handlers and hooks.

    Inputs:
        command_router: Optional command router override used by tests or
            custom runtime wiring.
        admin_router: Optional admin router override used by tests or
            custom runtime wiring.

    Outputs:
        Dispatcher with command routes included plus startup/shutdown hooks.
    """

    dispatcher = Dispatcher()
    dispatcher.include_router(command_router or telegram_commands_router)
    dispatcher.include_router(admin_router or build_admin_router())
    register_lifecycle_hooks(dispatcher)
    return dispatcher


def register_lifecycle_hooks(dispatcher: Dispatcher) -> None:
    """Register startup and shutdown hooks on the provided dispatcher."""

    dispatcher.startup.register(on_startup)
    dispatcher.shutdown.register(on_shutdown)


async def on_startup(bot: Bot) -> None:
    """Configure command metadata when the Telegram bot starts polling.

    Inputs:
        bot: Active aiogram bot instance passed by dispatcher lifecycle hooks.

    Behavior:
        Publishes the canonical command menu so users see supported commands
        in Telegram's built-in command picker.
    """

    await bot.set_my_commands(list(build_bot_commands()))


async def on_shutdown(bot: Bot) -> None:
    """Run shutdown cleanup actions for the Telegram bot runtime.

    Inputs:
        bot: Active aiogram bot instance passed by dispatcher lifecycle hooks.

    Behavior:
        Removes webhook registration to keep polling-mode restarts clean.
    """

    await bot.delete_webhook(drop_pending_updates=False)


def create_telegram_application(
    *,
    token: str | None = None,
    command_router: Router | None = None,
    admin_router: Router | None = None,
) -> TelegramApplication:
    """Create the immutable Telegram application bundle.

    Inputs:
        token: Optional explicit bot token override.
        command_router: Optional router override for tests or custom wiring.
        admin_router: Optional admin router override for tests or custom wiring.

    Outputs:
        Built `TelegramApplication` containing bot and dispatcher instances.
    """

    return TelegramApplication(
        bot=build_bot(token=token),
        dispatcher=build_dispatcher(
            command_router=command_router,
            admin_router=admin_router,
        ),
    )


async def run_telegram_application(
    application: TelegramApplication,
    *,
    mode: BotRunMode = BotRunMode.POLLING,
    webhook_url: str | None = None,
    webhook_secret_token: str | None = None,
    drop_pending_updates: bool = True,
    polling_timeout: int = 10,
) -> None:
    """Run Telegram registration in polling mode or webhook mode.

    Inputs:
        application: Telegram bot + dispatcher bundle created by
            `create_telegram_application`.
        mode: Runtime mode (`polling` or `webhook`).
        webhook_url: HTTPS webhook endpoint URL used in webhook mode.
        webhook_secret_token: Optional secret token for webhook signature checks.
        drop_pending_updates: Whether to discard pending updates during startup.
        polling_timeout: Long-polling timeout in seconds for polling mode.

    Behavior:
        - `polling`: clears webhook state, then starts aiogram polling loop.
        - `webhook`: registers webhook URL and exits, allowing external HTTP
          infrastructure to push updates to the bot.

    Raises:
        ValueError: If webhook mode is selected without a URL or polling
            timeout is invalid.
    """

    if polling_timeout <= 0:
        raise ValueError("polling_timeout must be greater than zero.")

    if mode is BotRunMode.POLLING:
        # Clearing webhook state avoids conflicting update delivery channels
        # when deployments switch between webhook and polling.
        await application.bot.delete_webhook(
            drop_pending_updates=drop_pending_updates
        )
        await application.dispatcher.start_polling(
            application.bot,
            polling_timeout=polling_timeout,
        )
        return

    if mode is BotRunMode.WEBHOOK:
        normalized_webhook_url = (webhook_url or "").strip()
        if not normalized_webhook_url:
            raise ValueError("webhook_url is required when mode='webhook'.")

        await application.bot.set_webhook(
            url=normalized_webhook_url,
            secret_token=webhook_secret_token,
            drop_pending_updates=drop_pending_updates,
        )
        return

    raise ValueError(f"Unsupported bot run mode: {mode}")


__all__ = [
    "BotRunMode",
    "TelegramApplication",
    "build_bot",
    "build_dispatcher",
    "create_telegram_application",
    "on_shutdown",
    "on_startup",
    "register_lifecycle_hooks",
    "run_telegram_application",
]
