"""Runtime configuration for the PuntLab agent bootstrap.

Purpose: centralizes environment-backed settings needed to install, validate,
and start the Python agent before the full configuration surface is added.
Scope: startup configuration only for this scaffolding phase.
Dependencies: uses `pydantic-settings` for typed environment parsing.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EnvironmentName = Literal["development", "staging", "production"]


class Settings(BaseSettings):
    """Typed environment-backed configuration for the agent bootstrap.

    Inputs:
        Environment variables from `.env` or the host process.

    Outputs:
        A validated settings object consumed by `src.main`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "puntlab-agent"
    environment: EnvironmentName = "development"
    log_level: str = "INFO"
    pipeline_start_hour: int = Field(default=7, ge=0, le=23)
    publish_hour: int = Field(default=10, ge=0, le=23)
    bootstrap_heartbeat_seconds: int = Field(default=300, ge=5, le=3600)

    database_url: str = Field(
        default="postgresql://puntlab:puntlab@localhost:5432/puntlab",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    admin_telegram_ids: tuple[int, ...] = Field(default=(), alias="ADMIN_TELEGRAM_IDS")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Normalize the configured log level.

        Args:
            value: Raw log level string from environment configuration.

        Returns:
            Uppercased logging level name.
        """

        normalized = value.strip().upper()
        allowed_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if normalized not in allowed_levels:
            raise ValueError(
                f"Unsupported LOG_LEVEL '{value}'. Expected one of: {sorted(allowed_levels)}."
            )
        return normalized

    @field_validator("admin_telegram_ids", mode="before")
    @classmethod
    def parse_admin_ids(cls, value: object) -> tuple[int, ...]:
        """Parse admin Telegram IDs from env values into a typed tuple.

        Args:
            value: String, integer, sequence, or empty value from the environment.

        Returns:
            Tuple of Telegram user IDs.
        """

        if value in (None, "", ()):
            return ()

        if isinstance(value, int):
            return (value,)

        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            try:
                return tuple(int(part) for part in parts)
            except ValueError as exc:
                raise ValueError(
                    "ADMIN_TELEGRAM_IDS must be a comma-separated list of integers."
                ) from exc

        if isinstance(value, (list, tuple, set)):
            try:
                return tuple(int(item) for item in value)
            except (TypeError, ValueError) as exc:
                raise ValueError("ADMIN_TELEGRAM_IDS contains a non-integer value.") from exc

        raise TypeError("ADMIN_TELEGRAM_IDS must be a string or sequence of integers.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache runtime settings for the current process.

    Returns:
        A validated `Settings` instance.
    """

    return Settings()
