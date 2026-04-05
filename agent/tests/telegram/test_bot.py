"""Tests for PuntLab Telegram bot bootstrap utilities.

Purpose: verify bot/dispatcher factories and runtime mode registration logic.
Scope: unit tests for `src.telegram.bot`.
Dependencies: pytest plus lightweight fake runtime objects for webhook/polling
execution paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from aiogram import Router
from src.telegram.bot import (
    BotRunMode,
    TelegramApplication,
    build_bot,
    build_dispatcher,
    create_telegram_application,
    run_telegram_application,
)

_VALID_TOKEN = "123456:ABCdefGhIjklMNopQrstUVwxyZ0123456789"


@dataclass(slots=True)
class FakeBot:
    """Minimal async bot stub for runtime registration tests."""

    set_webhook_calls: list[dict[str, object]]
    delete_webhook_calls: list[dict[str, object]]

    async def set_webhook(
        self,
        *,
        url: str,
        secret_token: str | None = None,
        drop_pending_updates: bool | None = None,
    ) -> None:
        """Record webhook registration inputs."""

        self.set_webhook_calls.append(
            {
                "url": url,
                "secret_token": secret_token,
                "drop_pending_updates": drop_pending_updates,
            }
        )

    async def delete_webhook(self, *, drop_pending_updates: bool = False) -> None:
        """Record webhook deletion requests."""

        self.delete_webhook_calls.append(
            {"drop_pending_updates": drop_pending_updates}
        )


@dataclass(slots=True)
class FakeDispatcher:
    """Minimal async dispatcher stub for runtime registration tests."""

    polling_calls: list[dict[str, object]]

    async def start_polling(
        self,
        bot: FakeBot,
        *,
        polling_timeout: int,
    ) -> None:
        """Record polling invocations and their timeout configuration."""

        self.polling_calls.append(
            {
                "bot": bot,
                "polling_timeout": polling_timeout,
            }
        )


def test_build_bot_uses_explicit_token() -> None:
    """Bot factory should accept explicit token overrides."""

    bot = build_bot(token=_VALID_TOKEN)
    assert bot.token == _VALID_TOKEN


def test_build_bot_requires_token_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bot factory should fail fast when no token is available."""

    fake_settings = SimpleNamespace(telegram=SimpleNamespace(bot_token=None))
    monkeypatch.setattr("src.telegram.bot.get_settings", lambda: fake_settings)

    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN is required"):
        build_bot()


def test_build_dispatcher_includes_provided_command_router() -> None:
    """Dispatcher factory should include provided command routers."""

    router = Router(name="custom-telegram-router")
    dispatcher = build_dispatcher(command_router=router)

    assert router in dispatcher.sub_routers
    assert len(dispatcher.startup.handlers) >= 1
    assert len(dispatcher.shutdown.handlers) >= 1


def test_create_telegram_application_builds_bot_and_dispatcher() -> None:
    """Application factory should return immutable runtime components."""

    application = create_telegram_application(token=_VALID_TOKEN)

    assert application.bot.token == _VALID_TOKEN
    assert application.dispatcher is not None


@pytest.mark.asyncio
async def test_run_telegram_application_in_polling_mode_starts_polling() -> None:
    """Polling mode should clear webhook state and start dispatcher polling."""

    fake_bot = FakeBot(set_webhook_calls=[], delete_webhook_calls=[])
    fake_dispatcher = FakeDispatcher(polling_calls=[])
    application = TelegramApplication(  # type: ignore[arg-type]
        bot=fake_bot,
        dispatcher=fake_dispatcher,
    )

    await run_telegram_application(
        application,
        mode=BotRunMode.POLLING,
        polling_timeout=12,
        drop_pending_updates=True,
    )

    assert fake_bot.delete_webhook_calls == [{"drop_pending_updates": True}]
    assert len(fake_dispatcher.polling_calls) == 1
    assert fake_dispatcher.polling_calls[0]["polling_timeout"] == 12


@pytest.mark.asyncio
async def test_run_telegram_application_in_webhook_mode_registers_webhook() -> None:
    """Webhook mode should register the provided webhook URL and return."""

    fake_bot = FakeBot(set_webhook_calls=[], delete_webhook_calls=[])
    fake_dispatcher = FakeDispatcher(polling_calls=[])
    application = TelegramApplication(  # type: ignore[arg-type]
        bot=fake_bot,
        dispatcher=fake_dispatcher,
    )

    await run_telegram_application(
        application,
        mode=BotRunMode.WEBHOOK,
        webhook_url="https://example.com/telegram/webhook",
        webhook_secret_token="secret-123",
        drop_pending_updates=False,
    )

    assert fake_dispatcher.polling_calls == []
    assert fake_bot.set_webhook_calls == [
        {
            "url": "https://example.com/telegram/webhook",
            "secret_token": "secret-123",
            "drop_pending_updates": False,
        }
    ]


@pytest.mark.asyncio
async def test_run_telegram_application_webhook_mode_requires_url() -> None:
    """Webhook mode should fail fast when no URL is supplied."""

    fake_bot = FakeBot(set_webhook_calls=[], delete_webhook_calls=[])
    fake_dispatcher = FakeDispatcher(polling_calls=[])
    application = TelegramApplication(  # type: ignore[arg-type]
        bot=fake_bot,
        dispatcher=fake_dispatcher,
    )

    with pytest.raises(ValueError, match="webhook_url is required"):
        await run_telegram_application(application, mode=BotRunMode.WEBHOOK)
