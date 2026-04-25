"""Canonical runtime configuration for the PuntLab agent.

Purpose: centralize all environment-backed settings, competition metadata,
market taxonomies, and deployment constants the Python agent depends on.
Scope: application settings, grouped provider credentials, canonical league
configuration, and helper accessors shared across the agent codebase.
Dependencies: uses `pydantic-settings` for environment parsing and Pydantic
models for typed configuration groupings.
"""

from __future__ import annotations

from enum import StrEnum
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class EnvironmentName(StrEnum):
    """Supported runtime environments for the agent."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SportName(StrEnum):
    """Sports supported by the canonical V1 configuration."""

    SOCCER = "soccer"
    BASKETBALL = "basketball"


class CompetitionType(StrEnum):
    """Top-level competition categories used by the ingestion pipeline."""

    DOMESTIC_LEAGUE = "domestic_league"
    CONTINENTAL_COMPETITION = "continental_competition"
    FRANCHISE_LEAGUE = "franchise_league"


class MarketType(StrEnum):
    """Canonical internal market taxonomy used across providers and resolvers."""

    MATCH_RESULT = "1x2"
    OVER_UNDER_05 = "over_under_0.5"
    OVER_UNDER_15 = "over_under_1.5"
    OVER_UNDER_25 = "over_under_2.5"
    OVER_UNDER_35 = "over_under_3.5"
    BTTS = "btts"
    DOUBLE_CHANCE = "double_chance"
    DRAW_NO_BET = "draw_no_bet"
    ASIAN_HANDICAP = "asian_handicap"
    CORRECT_SCORE = "correct_score"
    HT_FT = "half_time_full_time"
    MONEYLINE = "moneyline"
    POINT_SPREAD = "point_spread"
    TOTAL_POINTS = "total_points"


class CompetitionConfig(BaseModel):
    """Canonical metadata for one supported competition.

    Inputs:
        Static competition definitions declared in this module.

    Outputs:
        A validated, reusable configuration object for pipeline filtering,
        provider routing, and user-facing labeling.
    """

    key: str = Field(description="Stable internal identifier for the competition.")
    name: str = Field(description="Display name used across the product.")
    country: str = Field(description="Country or regional owner of the competition.")
    sport: SportName = Field(description="Sport the competition belongs to.")
    league_code: str = Field(description="Short canonical code stored in the database.")
    slug: str = Field(description="URL-safe slug for future provider or route mapping.")
    priority: int = Field(ge=1, description="Stable ordering priority for scheduling.")
    competition_type: CompetitionType = Field(
        description="Competition classification used by pipeline logic."
    )
    include_in_daily_analysis: bool = Field(
        default=True,
        description="Whether the competition is enabled in the canonical V1 slate.",
    )

    @field_validator("key", "name", "country", "league_code", "slug")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Reject blank text values in competition definitions."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("Competition metadata values must not be blank.")
        return normalized


class AppConfig(BaseModel):
    """Grouped agent runtime settings unrelated to third-party services."""

    app_name: str
    environment: EnvironmentName
    log_level: str
    pipeline_start_hour: int
    publish_hour: int
    bootstrap_heartbeat_seconds: int
    timezone_name: str


class LLMProviderConfig(BaseModel):
    """Grouped LLM provider credentials and provider ordering."""

    openai_api_key: str | None
    anthropic_api_key: str | None
    openrouter_api_key: str | None
    primary_provider: str = "openrouter"
    secondary_provider: str = "openai"
    fallback_provider: str = "anthropic"
    primary_model: str = "tencent/hy3-preview:free"
    secondary_model: str = "gpt-4o"
    fallback_model: str = "claude-sonnet-4-20250514"

    @field_validator("primary_provider", "secondary_provider", "fallback_provider")
    @classmethod
    def validate_provider_name(cls, value: str) -> str:
        """Normalize and validate provider slot names."""

        normalized = value.strip().lower()
        if normalized not in {"openai", "anthropic", "openrouter"}:
            raise ValueError(
                "LLM provider slots must be one of: openai, anthropic, openrouter."
            )
        return normalized

    @field_validator("primary_model", "secondary_model", "fallback_model")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """Reject blank model names in provider slot configuration."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("LLM model names must not be blank.")
        return normalized


class DataProviderConfig(BaseModel):
    """Grouped sports data provider credentials used by ingestion."""

    api_football_key: str | None
    balldontlie_api_key: str | None
    tavily_api_key: str | None


class DatabaseConfig(BaseModel):
    """Grouped database and Supabase connection settings."""

    database_url: str
    supabase_url: str | None
    supabase_anon_key: str | None
    supabase_service_key: str | None


class RedisConfig(BaseModel):
    """Grouped Redis connectivity settings."""

    url: str


class TelegramConfig(BaseModel):
    """Grouped Telegram delivery settings for the primary client surface."""

    bot_token: str | None
    admin_telegram_ids: tuple[int, ...]


class PaystackConfig(BaseModel):
    """Grouped Paystack credentials for payment workflows."""

    secret_key: str | None
    public_key: str | None
    webhook_secret: str | None


class LangfuseConfig(BaseModel):
    """Grouped Langfuse observability settings for LLM tracing."""

    public_key: str | None
    secret_key: str | None
    host: str


def _build_supported_competitions() -> tuple[CompetitionConfig, ...]:
    """Construct the canonical competition list for daily analysis."""

    raw_competitions = (
        {
            "key": "england_premier_league",
            "name": "Premier League",
            "country": "England",
            "sport": SportName.SOCCER,
            "league_code": "EPL",
            "slug": "premier-league",
            "priority": 1,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "spain_la_liga",
            "name": "La Liga",
            "country": "Spain",
            "sport": SportName.SOCCER,
            "league_code": "LL",
            "slug": "la-liga",
            "priority": 2,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "italy_serie_a",
            "name": "Serie A",
            "country": "Italy",
            "sport": SportName.SOCCER,
            "league_code": "SA",
            "slug": "serie-a",
            "priority": 3,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "germany_bundesliga",
            "name": "Bundesliga",
            "country": "Germany",
            "sport": SportName.SOCCER,
            "league_code": "BL1",
            "slug": "bundesliga",
            "priority": 4,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "france_ligue_1",
            "name": "Ligue 1",
            "country": "France",
            "sport": SportName.SOCCER,
            "league_code": "FL1",
            "slug": "ligue-1",
            "priority": 5,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "netherlands_eredivisie",
            "name": "Eredivisie",
            "country": "Netherlands",
            "sport": SportName.SOCCER,
            "league_code": "ERD",
            "slug": "eredivisie",
            "priority": 6,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "portugal_primeira_liga",
            "name": "Primeira Liga",
            "country": "Portugal",
            "sport": SportName.SOCCER,
            "league_code": "PPL",
            "slug": "primeira-liga",
            "priority": 7,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "belgium_pro_league",
            "name": "Pro League",
            "country": "Belgium",
            "sport": SportName.SOCCER,
            "league_code": "BPL",
            "slug": "pro-league",
            "priority": 8,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "turkey_super_lig",
            "name": "Super Lig",
            "country": "Turkey",
            "sport": SportName.SOCCER,
            "league_code": "TSL",
            "slug": "super-lig",
            "priority": 9,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "scotland_premiership",
            "name": "Premiership",
            "country": "Scotland",
            "sport": SportName.SOCCER,
            "league_code": "SPL",
            "slug": "premiership",
            "priority": 10,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "austria_bundesliga",
            "name": "Bundesliga",
            "country": "Austria",
            "sport": SportName.SOCCER,
            "league_code": "ABL",
            "slug": "bundesliga",
            "priority": 11,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "czech_republic_first_league",
            "name": "First League",
            "country": "Czech Republic",
            "sport": SportName.SOCCER,
            "league_code": "CFL",
            "slug": "first-league",
            "priority": 12,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "switzerland_super_league",
            "name": "Super League",
            "country": "Switzerland",
            "sport": SportName.SOCCER,
            "league_code": "CSL",
            "slug": "super-league",
            "priority": 13,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "greece_super_league",
            "name": "Super League",
            "country": "Greece",
            "sport": SportName.SOCCER,
            "league_code": "GSL",
            "slug": "super-league",
            "priority": 14,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "denmark_superliga",
            "name": "Superliga",
            "country": "Denmark",
            "sport": SportName.SOCCER,
            "league_code": "DSL",
            "slug": "superliga",
            "priority": 15,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "serbia_superliga",
            "name": "SuperLiga",
            "country": "Serbia",
            "sport": SportName.SOCCER,
            "league_code": "SSL",
            "slug": "superliga",
            "priority": 16,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "norway_eliteserien",
            "name": "Eliteserien",
            "country": "Norway",
            "sport": SportName.SOCCER,
            "league_code": "ESN",
            "slug": "eliteserien",
            "priority": 17,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "croatia_hnl",
            "name": "HNL",
            "country": "Croatia",
            "sport": SportName.SOCCER,
            "league_code": "HNL",
            "slug": "hnl",
            "priority": 18,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "ukraine_premier_league",
            "name": "Premier League",
            "country": "Ukraine",
            "sport": SportName.SOCCER,
            "league_code": "UPL",
            "slug": "premier-league",
            "priority": 19,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "israel_premier_league",
            "name": "Premier League",
            "country": "Israel",
            "sport": SportName.SOCCER,
            "league_code": "IPL",
            "slug": "premier-league",
            "priority": 20,
            "competition_type": CompetitionType.DOMESTIC_LEAGUE,
        },
        {
            "key": "uefa_champions_league",
            "name": "UEFA Champions League",
            "country": "Europe",
            "sport": SportName.SOCCER,
            "league_code": "UCL",
            "slug": "uefa-champions-league",
            "priority": 21,
            "competition_type": CompetitionType.CONTINENTAL_COMPETITION,
        },
        {
            "key": "uefa_europa_league",
            "name": "UEFA Europa League",
            "country": "Europe",
            "sport": SportName.SOCCER,
            "league_code": "UEL",
            "slug": "uefa-europa-league",
            "priority": 22,
            "competition_type": CompetitionType.CONTINENTAL_COMPETITION,
        },
        {
            "key": "uefa_conference_league",
            "name": "UEFA Conference League",
            "country": "Europe",
            "sport": SportName.SOCCER,
            "league_code": "UECL",
            "slug": "uefa-conference-league",
            "priority": 23,
            "competition_type": CompetitionType.CONTINENTAL_COMPETITION,
        },
        {
            "key": "nba",
            "name": "NBA",
            "country": "United States",
            "sport": SportName.BASKETBALL,
            "league_code": "NBA",
            "slug": "nba",
            "priority": 24,
            "competition_type": CompetitionType.FRANCHISE_LEAGUE,
        },
    )

    return tuple(CompetitionConfig.model_validate(item) for item in raw_competitions)


WAT_TIMEZONE_NAME = "Africa/Lagos"
WAT_TIMEZONE = ZoneInfo(WAT_TIMEZONE_NAME)
WAT_UTC_OFFSET_HOURS = 1
DEFAULT_PIPELINE_START_HOUR = 7
DEFAULT_PUBLISH_HOUR = 10
SUPPORTED_MARKET_TYPES = tuple(MarketType)
SUPPORTED_COMPETITIONS = _build_supported_competitions()
AGENT_ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE_PATH = AGENT_ROOT_DIRECTORY / ".env"


class Settings(BaseSettings):
    """Typed environment-backed configuration for the full agent runtime.

    Inputs:
        Environment variables from `.env` or the host process.

    Outputs:
        A validated settings object with both flat env-backed attributes and
        grouped helper accessors for the major service domains.
    """

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_FILE_PATH),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_name: str = "puntlab-agent"

    environment: EnvironmentName = Field(
        default=EnvironmentName.DEVELOPMENT,
        alias="ENVIRONMENT",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    pipeline_start_hour: int = Field(
        default=DEFAULT_PIPELINE_START_HOUR,
        alias="PIPELINE_START_HOUR",
        ge=0,
        le=23,
    )
    publish_hour: int = Field(
        default=DEFAULT_PUBLISH_HOUR,
        alias="PUBLISH_HOUR",
        ge=0,
        le=23,
    )
    bootstrap_heartbeat_seconds: int = Field(default=300, ge=5, le=3600)

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    llm_primary_provider: str = Field(default="openrouter", alias="LLM_PRIMARY_PROVIDER")
    llm_secondary_provider: str = Field(default="openai", alias="LLM_SECONDARY_PROVIDER")
    llm_fallback_provider: str = Field(default="anthropic", alias="LLM_FALLBACK_PROVIDER")
    llm_primary_model: str = Field(
        default="tencent/hy3-preview:free",
        alias="LLM_PRIMARY_MODEL",
    )
    llm_secondary_model: str = Field(default="gpt-4o", alias="LLM_SECONDARY_MODEL")
    llm_fallback_model: str = Field(
        default="claude-sonnet-4-20250514",
        alias="LLM_FALLBACK_MODEL",
    )

    api_football_key: str | None = Field(default=None, alias="API_FOOTBALL_KEY")
    balldontlie_api_key: str | None = Field(default=None, alias="BALLDONTLIE_API_KEY")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")

    supabase_url: str | None = Field(default=None, alias="SUPABASE_URL")
    supabase_anon_key: str | None = Field(default=None, alias="SUPABASE_ANON_KEY")
    supabase_service_key: str | None = Field(default=None, alias="SUPABASE_SERVICE_KEY")
    database_url: str = Field(
        default="postgresql://puntlab:puntlab@localhost:5432/puntlab",
        alias="DATABASE_URL",
    )

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    telegram_bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    admin_telegram_ids: Annotated[tuple[int, ...], NoDecode] = Field(
        default=(),
        alias="ADMIN_TELEGRAM_IDS",
    )

    paystack_secret_key: str | None = Field(default=None, alias="PAYSTACK_SECRET_KEY")
    paystack_public_key: str | None = Field(default=None, alias="PAYSTACK_PUBLIC_KEY")
    paystack_webhook_secret: str | None = Field(
        default=None,
        alias="PAYSTACK_WEBHOOK_SECRET",
    )

    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        alias="LANGFUSE_HOST",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Normalize the configured log level and reject unsupported values."""

        normalized = value.strip().upper()
        allowed_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if normalized not in allowed_levels:
            raise ValueError(
                f"Unsupported LOG_LEVEL '{value}'. Expected one of: {sorted(allowed_levels)}."
            )
        return normalized

    @field_validator(
        "openai_api_key",
        "anthropic_api_key",
        "openrouter_api_key",
        "api_football_key",
        "balldontlie_api_key",
        "tavily_api_key",
        "supabase_url",
        "supabase_anon_key",
        "supabase_service_key",
        "telegram_bot_token",
        "paystack_secret_key",
        "paystack_public_key",
        "paystack_webhook_secret",
        "langfuse_public_key",
        "langfuse_secret_key",
        mode="before",
    )
    @classmethod
    def normalize_optional_string(cls, value: object) -> str | None:
        """Coerce blank optional string settings to `None`."""

        if value is None:
            return None

        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None

        raise TypeError("Optional string settings must be supplied as strings.")

    @field_validator("admin_telegram_ids", mode="before")
    @classmethod
    def parse_admin_ids(cls, value: object) -> tuple[int, ...]:
        """Parse admin Telegram IDs from strings or sequences into integers."""

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

    @model_validator(mode="after")
    def validate_schedule_hours(self) -> Settings:
        """Ensure the pipeline schedule remains forward-moving within a day."""

        if self.publish_hour <= self.pipeline_start_hour:
            raise ValueError(
                "PUBLISH_HOUR must be later than PIPELINE_START_HOUR so the daily run "
                "finishes before publishing."
            )
        return self

    @cached_property
    def app(self) -> AppConfig:
        """Return grouped application runtime settings."""

        return AppConfig(
            app_name=self.app_name,
            environment=self.environment,
            log_level=self.log_level,
            pipeline_start_hour=self.pipeline_start_hour,
            publish_hour=self.publish_hour,
            bootstrap_heartbeat_seconds=self.bootstrap_heartbeat_seconds,
            timezone_name=WAT_TIMEZONE_NAME,
        )

    @cached_property
    def llm(self) -> LLMProviderConfig:
        """Return grouped LLM provider settings."""

        return LLMProviderConfig(
            openai_api_key=self.openai_api_key,
            anthropic_api_key=self.anthropic_api_key,
            openrouter_api_key=self.openrouter_api_key,
            primary_provider=self.llm_primary_provider,
            secondary_provider=self.llm_secondary_provider,
            fallback_provider=self.llm_fallback_provider,
            primary_model=self.llm_primary_model,
            secondary_model=self.llm_secondary_model,
            fallback_model=self.llm_fallback_model,
        )

    @cached_property
    def data_providers(self) -> DataProviderConfig:
        """Return grouped sports data provider settings."""

        return DataProviderConfig(
            api_football_key=self.api_football_key,
            balldontlie_api_key=self.balldontlie_api_key,
            tavily_api_key=self.tavily_api_key,
        )

    @cached_property
    def database(self) -> DatabaseConfig:
        """Return grouped database and Supabase settings."""

        return DatabaseConfig(
            database_url=self.database_url,
            supabase_url=self.supabase_url,
            supabase_anon_key=self.supabase_anon_key,
            supabase_service_key=self.supabase_service_key,
        )

    @cached_property
    def redis(self) -> RedisConfig:
        """Return grouped Redis settings."""

        return RedisConfig(url=self.redis_url)

    @cached_property
    def telegram(self) -> TelegramConfig:
        """Return grouped Telegram delivery settings."""

        return TelegramConfig(
            bot_token=self.telegram_bot_token,
            admin_telegram_ids=self.admin_telegram_ids,
        )

    @cached_property
    def paystack(self) -> PaystackConfig:
        """Return grouped Paystack payment settings."""

        return PaystackConfig(
            secret_key=self.paystack_secret_key,
            public_key=self.paystack_public_key,
            webhook_secret=self.paystack_webhook_secret,
        )

    @cached_property
    def langfuse(self) -> LangfuseConfig:
        """Return grouped Langfuse observability settings."""

        return LangfuseConfig(
            public_key=self.langfuse_public_key,
            secret_key=self.langfuse_secret_key,
            host=self.langfuse_host,
        )

    @property
    def timezone(self) -> ZoneInfo:
        """Return the canonical West Africa Time zone for scheduling."""

        return WAT_TIMEZONE

    @property
    def competitions(self) -> tuple[CompetitionConfig, ...]:
        """Return the canonical V1 competition catalog."""

        return SUPPORTED_COMPETITIONS

    @property
    def supported_market_types(self) -> tuple[MarketType, ...]:
        """Return the canonical internal market taxonomy."""

        return SUPPORTED_MARKET_TYPES

    def get_competitions_by_sport(self, sport: SportName) -> tuple[CompetitionConfig, ...]:
        """Return the configured competitions for a single sport."""

        return tuple(
            competition
            for competition in SUPPORTED_COMPETITIONS
            if competition.sport == sport
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache runtime settings for the current process."""

    return Settings()


__all__ = [
    "AppConfig",
    "CompetitionConfig",
    "CompetitionType",
    "DEFAULT_PIPELINE_START_HOUR",
    "DEFAULT_PUBLISH_HOUR",
    "DataProviderConfig",
    "DatabaseConfig",
    "EnvironmentName",
    "LLMProviderConfig",
    "LangfuseConfig",
    "MarketType",
    "PaystackConfig",
    "RedisConfig",
    "Settings",
    "SportName",
    "SUPPORTED_COMPETITIONS",
    "SUPPORTED_MARKET_TYPES",
    "TelegramConfig",
    "WAT_TIMEZONE",
    "WAT_TIMEZONE_NAME",
    "WAT_UTC_OFFSET_HOURS",
    "get_settings",
]
