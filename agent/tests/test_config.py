"""Tests for PuntLab's canonical agent configuration module.

Purpose: verify environment parsing, schedule validation, and the canonical
competition catalog before provider and pipeline layers depend on them.
Scope: pure unit tests for `src.config` with no external services required.
Dependencies: pytest and the Pydantic-based settings models.
"""

from __future__ import annotations

import pytest
from src.config import (
    DEFAULT_PIPELINE_START_HOUR,
    DEFAULT_PUBLISH_HOUR,
    SUPPORTED_COMPETITIONS,
    WAT_TIMEZONE_NAME,
    EnvironmentName,
    MarketType,
    Settings,
    SportName,
)


def test_settings_parses_env_aliases_and_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should parse env vars and expose grouped helper models."""

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("PIPELINE_START_HOUR", "7")
    monkeypatch.setenv("PUBLISH_HOUR", "10")
    monkeypatch.setenv("LLM_PRIMARY_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_SECONDARY_PROVIDER", "openai")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_PRIMARY_MODEL", "tencent/hy3-preview:free")
    monkeypatch.setenv("LLM_SECONDARY_MODEL", "gpt-4o")
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("API_FOOTBALL_KEY", "rapidapi-key")
    monkeypatch.setenv("SUPABASE_URL", "https://puntlab.supabase.co")
    monkeypatch.setenv("DATABASE_URL", "postgresql://puntlab:secret@localhost:5432/puntlab")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:ABC")
    monkeypatch.setenv("ADMIN_TELEGRAM_IDS", "123, 456")
    monkeypatch.setenv("PAYSTACK_SECRET_KEY", "sk_test_123")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    settings = Settings(_env_file=None)

    assert settings.environment is EnvironmentName.PRODUCTION
    assert settings.log_level == "DEBUG"
    assert settings.app.pipeline_start_hour == DEFAULT_PIPELINE_START_HOUR
    assert settings.app.publish_hour == DEFAULT_PUBLISH_HOUR
    assert settings.app.timezone_name == WAT_TIMEZONE_NAME
    assert settings.llm.primary_provider == "openrouter"
    assert settings.llm.secondary_provider == "openai"
    assert settings.llm.fallback_provider == "anthropic"
    assert settings.llm.primary_model == "tencent/hy3-preview:free"
    assert settings.llm.secondary_model == "gpt-4o"
    assert settings.llm.fallback_model == "claude-sonnet-4-20250514"
    assert settings.llm.openai_api_key == "sk-openai"
    assert settings.data_providers.api_football_key == "rapidapi-key"
    assert settings.database.supabase_url == "https://puntlab.supabase.co"
    assert settings.redis.url == "redis://localhost:6379/1"
    assert settings.telegram.bot_token == "123456:ABC"
    assert settings.telegram.admin_telegram_ids == (123, 456)
    assert settings.paystack.secret_key == "sk_test_123"
    assert settings.langfuse.host == "https://cloud.langfuse.com"


def test_settings_normalize_blank_optional_values() -> None:
    """Blank optional secrets should collapse to `None` instead of empty strings."""

    settings = Settings(
        _env_file=None,
        openai_api_key="   ",
        telegram_bot_token="",
        paystack_public_key="  ",
    )

    assert settings.llm.openai_api_key is None
    assert settings.telegram.bot_token is None
    assert settings.paystack.public_key is None


def test_settings_reject_publish_hour_not_after_pipeline_start() -> None:
    """Publishing must happen after the daily pipeline begins."""

    with pytest.raises(ValueError, match="PUBLISH_HOUR must be later"):
        Settings(
            _env_file=None,
            pipeline_start_hour=11,
            publish_hour=10,
        )


def test_supported_competitions_cover_v1_sports_and_slate() -> None:
    """The canonical competition catalog should match the technical spec."""

    league_codes = {competition.league_code for competition in SUPPORTED_COMPETITIONS}
    soccer_competitions = [
        competition
        for competition in SUPPORTED_COMPETITIONS
        if competition.sport is SportName.SOCCER
    ]
    basketball_competitions = [
        competition
        for competition in SUPPORTED_COMPETITIONS
        if competition.sport is SportName.BASKETBALL
    ]

    assert len(SUPPORTED_COMPETITIONS) == 24
    assert len(soccer_competitions) == 23
    assert len(basketball_competitions) == 1
    assert {"EPL", "LL", "SA", "UCL", "UEL", "UECL", "NBA"} <= league_codes
    assert SUPPORTED_COMPETITIONS[0].name == "Premier League"
    assert SUPPORTED_COMPETITIONS[-1].league_code == "NBA"


def test_supported_market_taxonomy_matches_specification() -> None:
    """The canonical market taxonomy should expose all V1 market families."""

    expected_market_values = {
        "1x2",
        "over_under_0.5",
        "over_under_1.5",
        "over_under_2.5",
        "over_under_3.5",
        "btts",
        "double_chance",
        "draw_no_bet",
        "asian_handicap",
        "correct_score",
        "half_time_full_time",
        "moneyline",
        "point_spread",
        "total_points",
    }

    assert {market.value for market in MarketType} == expected_market_values
