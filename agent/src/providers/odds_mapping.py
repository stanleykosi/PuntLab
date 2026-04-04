"""Provider-agnostic odds market catalog and canonical mapping helpers.

Purpose: preserve the full upstream odds market universe without discarding
unsupported markets while exposing a separate canonical projection for the
subset PuntLab can score today.
Scope: group `NormalizedOdds` rows into provider-market catalogs, infer
canonical `MarketType` mappings from provider-native labels and metadata, and
filter scoreable rows without mutating the lossless source records.
Dependencies: shared runtime enums from `src.config`, normalized odds schemas
from `src.schemas.odds`, and the shared text normalization helpers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from src.config import MarketType, SportName
from src.schemas.common import normalize_optional_text, require_non_blank_text
from src.schemas.odds import JSONPrimitive, NormalizedOdds

_NON_ALNUM_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_SOCCER_TOTAL_MARKETS: Final[dict[float, MarketType]] = {
    0.5: MarketType.OVER_UNDER_05,
    1.5: MarketType.OVER_UNDER_15,
    2.5: MarketType.OVER_UNDER_25,
    3.5: MarketType.OVER_UNDER_35,
}
_MATCH_RESULT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "1x2",
        "full_time_result",
        "h2h",
        "head_to_head",
        "match_result",
        "match_winner",
        "moneyline",
        "winner",
    }
)
_BTTS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "both_teams_score",
        "both_teams_to_score",
        "btts",
    }
)
_DOUBLE_CHANCE_KEYS: Final[frozenset[str]] = frozenset({"double_chance"})
_DRAW_NO_BET_KEYS: Final[frozenset[str]] = frozenset(
    {
        "draw_no_bet",
        "dnb",
    }
)
_CORRECT_SCORE_KEYS: Final[frozenset[str]] = frozenset({"correct_score"})
_HT_FT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "half_full",
        "half_time_full_time",
        "halftime_fulltime",
        "ht_ft",
    }
)
_TOTAL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "alternate_totals",
        "goals_over_under",
        "over_under",
        "team_total",
        "team_totals",
        "total",
        "totals",
    }
)
_SPREAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "alternate_spreads",
        "asian_handicap",
        "handicap",
        "point_spread",
        "spread",
        "spreads",
    }
)
_HOME_ALIASES: Final[frozenset[str]] = frozenset({"1", "home", "team1"})
_AWAY_ALIASES: Final[frozenset[str]] = frozenset({"2", "away", "team2"})
_DRAW_ALIASES: Final[frozenset[str]] = frozenset({"draw", "x"})
_YES_ALIASES: Final[frozenset[str]] = frozenset({"yes"})
_NO_ALIASES: Final[frozenset[str]] = frozenset({"no"})
_DOUBLE_CHANCE_SELECTIONS: Final[dict[str, str]] = {
    "1x": "1X",
    "12": "12",
    "x2": "X2",
    "away_or_draw": "X2",
    "draw_or_away": "X2",
    "draw_or_home": "1X",
    "home_or_away": "12",
    "home_or_draw": "1X",
}
_HT_FT_TOKEN_MAP: Final[dict[str, str]] = {
    "1": "1",
    "2": "2",
    "away": "2",
    "draw": "X",
    "home": "1",
    "team1": "1",
    "team2": "2",
    "x": "X",
}
_LINE_BASED_MARKETS: Final[frozenset[MarketType]] = frozenset(
    {
        MarketType.ASIAN_HANDICAP,
        MarketType.POINT_SPREAD,
        MarketType.TOTAL_POINTS,
        MarketType.OVER_UNDER_05,
        MarketType.OVER_UNDER_15,
        MarketType.OVER_UNDER_25,
        MarketType.OVER_UNDER_35,
    }
)
_BASKETBALL_MARKETS: Final[frozenset[MarketType]] = frozenset(
    {
        MarketType.MONEYLINE,
        MarketType.POINT_SPREAD,
        MarketType.TOTAL_POINTS,
    }
)
_SOCCER_MARKETS: Final[frozenset[MarketType]] = frozenset(set(MarketType) - _BASKETBALL_MARKETS)


class CanonicalOddsSelection(BaseModel):
    """Lossless provider selection plus its current canonical mapping result.

    Inputs:
        One `NormalizedOdds` row and the mapping decision produced from its
        provider-native labels and metadata.

    Outputs:
        A selection record that preserves the source row verbatim while also
        exposing the scoreable projection used by scoring and ranking helpers.
    """

    model_config = ConfigDict(extra="forbid")

    source: NormalizedOdds = Field(
        description="Original normalized odds row preserved without mutation."
    )
    canonical_market: MarketType | None = Field(
        default=None,
        description="Current PuntLab market mapping for this selection when supported.",
    )
    canonical_selection: str = Field(
        description="Current PuntLab selection label derived from provider-native inputs."
    )
    provider_selection_id: int | str | None = Field(
        default=None,
        description="Provider-native selection identifier when supplied in metadata.",
    )
    mapping_basis: str = Field(
        description="How the canonical mapping was produced: rule, source_hint, or unmapped."
    )
    mapping_reason: str | None = Field(
        default=None,
        description="Human-readable explanation for unmapped or constrained selections.",
    )

    def to_normalized_odds(self) -> NormalizedOdds:
        """Return the scoreable projection as a canonical `NormalizedOdds` row."""

        return self.source.model_copy(
            update={
                "market": self.canonical_market,
                "selection": self.canonical_selection,
            }
        )

    @property
    def is_scoreable(self) -> bool:
        """Report whether the selection maps into PuntLab's current taxonomy."""

        return self.canonical_market is not None


class ProviderMarketCatalogEntry(BaseModel):
    """Catalog entry representing one provider market and all of its selections.

    Inputs:
        A set of `NormalizedOdds` rows that share the same provider-native
        market identity.

    Outputs:
        A grouped market record with selection-level mappings while retaining
        shared provider metadata, period, and market identifiers.
    """

    model_config = ConfigDict(extra="forbid")

    fixture_ref: str = Field(description="Fixture reference shared across the market.")
    provider: str = Field(description="Bookmaker or provider label that surfaced the market.")
    provider_market_name: str = Field(description="Original upstream provider market name.")
    provider_market_key: str = Field(description="Stable normalized provider market key.")
    provider_market_id: int | str | None = Field(
        default=None,
        description="Provider-native market identifier when present.",
    )
    market_label: str = Field(description="Display-ready provider market label.")
    period: str | None = Field(
        default=None,
        description="Provider-normalized period scope such as `match` or `first_half`.",
    )
    participant_scope: str | None = Field(
        default=None,
        description="Provider-normalized participant scope such as `match`, `team`, or `player`.",
    )
    raw_metadata: dict[str, JSONPrimitive] = Field(
        default_factory=dict,
        description="Shared metadata that is identical across every selection in the market.",
    )
    last_updated: datetime | None = Field(
        default=None,
        description="Most recent source timestamp across the market selections.",
    )
    selections: tuple[CanonicalOddsSelection, ...] = Field(
        default_factory=tuple,
        description="All preserved selections for the provider-native market.",
    )

    def scoreable_rows(self) -> tuple[NormalizedOdds, ...]:
        """Return only the current scoreable canonical rows for this market."""

        return tuple(
            selection.to_normalized_odds()
            for selection in self.selections
            if selection.is_scoreable
        )

    def scoreable_market_types(self) -> tuple[MarketType, ...]:
        """Return the unique canonical market types represented in this market."""

        ordered_markets: list[MarketType] = []
        seen: set[MarketType] = set()
        for selection in self.selections:
            market = selection.canonical_market
            if market is None or market in seen:
                continue
            ordered_markets.append(market)
            seen.add(market)
        return tuple(ordered_markets)


class OddsMarketCatalog(BaseModel):
    """Full odds catalog retaining every provider market plus scoring filters.

    Inputs:
        Grouped `ProviderMarketCatalogEntry` records assembled from one odds
        ingestion batch.

    Outputs:
        Helper methods for retrieving the full provider universe, the scoreable
        canonical subset, and the currently unmapped remainder.
    """

    model_config = ConfigDict(extra="forbid")

    markets: tuple[ProviderMarketCatalogEntry, ...] = Field(
        default_factory=tuple,
        description="All grouped provider markets in source order.",
    )

    def all_rows(self) -> tuple[NormalizedOdds, ...]:
        """Return the full lossless set of source rows across all catalog markets."""

        return tuple(
            selection.source
            for market in self.markets
            for selection in market.selections
        )

    def scoreable_rows(self) -> tuple[NormalizedOdds, ...]:
        """Return the current canonical subset PuntLab can score today."""

        return tuple(
            selection.to_normalized_odds()
            for market in self.markets
            for selection in market.selections
            if selection.is_scoreable
        )

    def unmapped_rows(self) -> tuple[NormalizedOdds, ...]:
        """Return the provider rows that are preserved but not yet scoreable."""

        return tuple(
            selection.source
            for market in self.markets
            for selection in market.selections
            if not selection.is_scoreable
        )


def build_odds_market_catalog(
    odds_rows: Sequence[NormalizedOdds],
    *,
    sport_by_fixture: Mapping[str, SportName] | None = None,
) -> OddsMarketCatalog:
    """Build a lossless provider-market catalog from normalized odds rows.

    Args:
        odds_rows: Flat odds rows emitted by one or more providers.
        sport_by_fixture: Optional explicit sport overrides keyed by
            `fixture_ref`. This is useful when provider metadata does not
            expose the sport directly.

    Returns:
        An `OddsMarketCatalog` that preserves every source row and exposes the
        scoreable subset as a derived projection.
    """

    grouped_rows: dict[_MarketGroupingKey, list[NormalizedOdds]] = {}
    market_order: list[_MarketGroupingKey] = []
    for row in odds_rows:
        group_key = (
            row.fixture_ref,
            row.provider,
            row.provider_market_name,
            row.provider_market_key or row.provider_market_name,
            row.provider_market_id,
            row.period,
            row.participant_scope,
        )
        if group_key not in grouped_rows:
            grouped_rows[group_key] = []
            market_order.append(group_key)
        grouped_rows[group_key].append(row)

    catalog_entries: list[ProviderMarketCatalogEntry] = []
    for group_key in market_order:
        rows = grouped_rows[group_key]
        selections = tuple(
            _build_selection_catalog_entry(
                row,
                explicit_sport=(
                    sport_by_fixture.get(row.fixture_ref)
                    if sport_by_fixture is not None
                    else None
                ),
            )
            for row in rows
        )
        first_row = rows[0]
        catalog_entries.append(
            ProviderMarketCatalogEntry(
                fixture_ref=first_row.fixture_ref,
                provider=first_row.provider,
                provider_market_name=first_row.provider_market_name,
                provider_market_key=(
                    first_row.provider_market_key
                    or _normalize_machine_key(first_row.provider_market_name)
                ),
                provider_market_id=first_row.provider_market_id,
                market_label=first_row.market_label or first_row.provider_market_name,
                period=first_row.period,
                participant_scope=first_row.participant_scope,
                raw_metadata=_shared_market_metadata(rows),
                last_updated=_latest_timestamp(rows),
                selections=selections,
            )
        )

    return OddsMarketCatalog(markets=tuple(catalog_entries))


def canonicalize_odds_rows(
    odds_rows: Sequence[NormalizedOdds],
    *,
    sport_by_fixture: Mapping[str, SportName] | None = None,
) -> tuple[CanonicalOddsSelection, ...]:
    """Project flat odds rows into canonical mapping results without grouping."""

    return tuple(
        _build_selection_catalog_entry(
            row,
            explicit_sport=(
                sport_by_fixture.get(row.fixture_ref)
                if sport_by_fixture is not None
                else None
            ),
        )
        for row in odds_rows
    )


def filter_scoreable_odds(
    odds_rows: Sequence[NormalizedOdds],
    *,
    sport_by_fixture: Mapping[str, SportName] | None = None,
) -> tuple[NormalizedOdds, ...]:
    """Return only the current canonically scoreable odds rows."""

    return tuple(
        selection.to_normalized_odds()
        for selection in canonicalize_odds_rows(
            odds_rows,
            sport_by_fixture=sport_by_fixture,
        )
        if selection.is_scoreable
    )


def filter_unmapped_odds(
    odds_rows: Sequence[NormalizedOdds],
    *,
    sport_by_fixture: Mapping[str, SportName] | None = None,
) -> tuple[NormalizedOdds, ...]:
    """Return only the preserved provider rows that remain unmapped today."""

    return tuple(
        selection.source
        for selection in canonicalize_odds_rows(
            odds_rows,
            sport_by_fixture=sport_by_fixture,
        )
        if not selection.is_scoreable
    )


def group_markets_by_fixture(
    markets: Sequence[ProviderMarketCatalogEntry],
) -> dict[str, tuple[ProviderMarketCatalogEntry, ...]]:
    """Group catalog markets by fixture while preserving catalog order."""

    grouped: dict[str, list[ProviderMarketCatalogEntry]] = {}
    for market in markets:
        grouped.setdefault(market.fixture_ref, []).append(market)
    return {fixture_ref: tuple(entries) for fixture_ref, entries in grouped.items()}


def group_markets_by_canonical_market(
    markets: Sequence[ProviderMarketCatalogEntry],
) -> dict[MarketType | None, tuple[ProviderMarketCatalogEntry, ...]]:
    """Group markets by each canonical market type they currently represent.

    Markets that contain both supported and unsupported line variants can
    appear in more than one bucket so callers can inspect the scoreable subset
    without losing the full provider-native grouping.
    """

    grouped: dict[MarketType | None, list[ProviderMarketCatalogEntry]] = {}
    for market in markets:
        represented_markets = {
            selection.canonical_market
            for selection in market.selections
        } or {None}
        for represented_market in represented_markets:
            grouped.setdefault(represented_market, []).append(market)
    return {market_type: tuple(entries) for market_type, entries in grouped.items()}


def _build_selection_catalog_entry(
    row: NormalizedOdds,
    *,
    explicit_sport: SportName | None,
) -> CanonicalOddsSelection:
    """Build one canonical selection entry from a lossless source row."""

    market, mapping_basis, mapping_reason = _infer_canonical_market(
        row,
        explicit_sport=explicit_sport,
    )
    canonical_selection = _infer_canonical_selection(
        row,
        market=market,
    )
    return CanonicalOddsSelection(
        source=row,
        canonical_market=market,
        canonical_selection=canonical_selection,
        provider_selection_id=_extract_provider_selection_id(row.raw_metadata),
        mapping_basis=mapping_basis,
        mapping_reason=mapping_reason,
    )


def _infer_canonical_market(
    row: NormalizedOdds,
    *,
    explicit_sport: SportName | None,
) -> tuple[MarketType | None, str, str | None]:
    """Infer the current canonical market type from one provider-native row."""

    resolved_sport = _resolve_sport(row, explicit_sport=explicit_sport)
    provider_keys = _collect_market_keys(row)
    candidate_market: MarketType | None = None
    mapping_basis = "unmapped"
    mapping_reason: str | None = None

    if provider_keys & _BTTS_KEYS:
        candidate_market = MarketType.BTTS
        mapping_basis = "rule"
    elif provider_keys & _DOUBLE_CHANCE_KEYS:
        candidate_market = MarketType.DOUBLE_CHANCE
        mapping_basis = "rule"
    elif provider_keys & _DRAW_NO_BET_KEYS:
        candidate_market = MarketType.DRAW_NO_BET
        mapping_basis = "rule"
    elif provider_keys & _CORRECT_SCORE_KEYS:
        candidate_market = MarketType.CORRECT_SCORE
        mapping_basis = "rule"
    elif provider_keys & _HT_FT_KEYS:
        candidate_market = MarketType.HT_FT
        mapping_basis = "rule"
    elif provider_keys & _TOTAL_KEYS:
        candidate_market = _map_total_market(row, sport=resolved_sport)
        mapping_basis = "rule" if candidate_market is not None else "unmapped"
        if candidate_market is None:
            mapping_reason = "provider market is a totals family that PuntLab cannot score yet."
    elif provider_keys & _SPREAD_KEYS:
        candidate_market = _map_spread_market(row, sport=resolved_sport)
        mapping_basis = "rule" if candidate_market is not None else "unmapped"
        if candidate_market is None:
            mapping_reason = "provider market is a spread family that PuntLab cannot score yet."
    elif provider_keys & _MATCH_RESULT_KEYS:
        candidate_market = _map_match_result_market(sport=resolved_sport)
        mapping_basis = "rule" if candidate_market is not None else "unmapped"
        if candidate_market is None:
            mapping_reason = "match-result mapping requires soccer or basketball sport context."
    elif row.market is not None:
        candidate_market = row.market
        mapping_basis = "source_hint"

    if candidate_market is None:
        return None, mapping_basis, mapping_reason

    scope_reason = _validate_supported_scope(row, candidate_market)
    if scope_reason is not None:
        return None, "unmapped", scope_reason

    sport_reason = _validate_sport_alignment(candidate_market, resolved_sport)
    if sport_reason is not None:
        return None, "unmapped", sport_reason

    line_reason = _validate_line_requirements(row, candidate_market)
    if line_reason is not None:
        return None, "unmapped", line_reason

    return candidate_market, mapping_basis, None


def _infer_canonical_selection(
    row: NormalizedOdds,
    *,
    market: MarketType | None,
) -> str:
    """Infer the canonical selection label for one provider-native row."""

    if market is None:
        return row.selection
    home_team, away_team = _extract_teams(row.raw_metadata)
    raw_selection = row.provider_selection_name
    if market == MarketType.MATCH_RESULT:
        return _normalize_match_result_selection(
            raw_selection,
            home_team=home_team,
            away_team=away_team,
            allow_draw=True,
            fallback=row.selection,
        )
    if market in {MarketType.MONEYLINE, MarketType.DRAW_NO_BET}:
        return _normalize_match_result_selection(
            raw_selection,
            home_team=home_team,
            away_team=away_team,
            allow_draw=False,
            fallback=row.selection,
        )
    if market in {MarketType.ASIAN_HANDICAP, MarketType.POINT_SPREAD}:
        return _normalize_side_selection(
            raw_selection,
            home_team=home_team,
            away_team=away_team,
            fallback=row.selection,
        )
    if market in {
        MarketType.OVER_UNDER_05,
        MarketType.OVER_UNDER_15,
        MarketType.OVER_UNDER_25,
        MarketType.OVER_UNDER_35,
        MarketType.TOTAL_POINTS,
    }:
        return _normalize_binary_selection(
            raw_selection,
            positive_aliases={"over"},
            negative_aliases={"under"},
            positive_value="over",
            negative_value="under",
            fallback=row.selection,
        )
    if market == MarketType.BTTS:
        return _normalize_binary_selection(
            raw_selection,
            positive_aliases=_YES_ALIASES,
            negative_aliases=_NO_ALIASES,
            positive_value="yes",
            negative_value="no",
            fallback=row.selection,
        )
    if market == MarketType.DOUBLE_CHANCE:
        return _normalize_double_chance_selection(
            raw_selection,
            home_team=home_team,
            away_team=away_team,
            fallback=row.selection,
        )
    if market == MarketType.HT_FT:
        return _normalize_ht_ft_selection(raw_selection, fallback=row.selection)
    return require_non_blank_text(raw_selection, "provider_selection_name")


def _collect_market_keys(row: NormalizedOdds) -> set[str]:
    """Collect normalized provider-market identifiers for rule matching."""

    candidate_values = (
        row.provider_market_key,
        row.provider_market_name,
        row.market_label,
    )
    normalized_keys: set[str] = set()
    for value in candidate_values:
        if value is None:
            continue
        normalized_keys.add(_normalize_machine_key(value))
    if isinstance(row.provider_market_id, str):
        normalized_keys.add(_normalize_machine_key(row.provider_market_id))
    return normalized_keys


def _map_match_result_market(*, sport: SportName | None) -> MarketType | None:
    """Map generic winner markets into soccer or basketball canonical families."""

    if sport == SportName.BASKETBALL:
        return MarketType.MONEYLINE
    if sport == SportName.SOCCER or sport is None:
        return MarketType.MATCH_RESULT
    return None


def _map_total_market(
    row: NormalizedOdds,
    *,
    sport: SportName | None,
) -> MarketType | None:
    """Map provider totals markets into soccer or basketball canonical families."""

    if sport == SportName.BASKETBALL:
        return MarketType.TOTAL_POINTS if row.line is not None else None
    if sport in {SportName.SOCCER, None}:
        if row.line is None:
            return None
        return _SOCCER_TOTAL_MARKETS.get(row.line)
    return None


def _map_spread_market(
    row: NormalizedOdds,
    *,
    sport: SportName | None,
) -> MarketType | None:
    """Map provider spread markets into soccer or basketball canonical families."""

    if row.line is None:
        return None
    if sport == SportName.BASKETBALL:
        return MarketType.POINT_SPREAD
    if sport in {SportName.SOCCER, None}:
        return MarketType.ASIAN_HANDICAP
    return None


def _validate_supported_scope(
    row: NormalizedOdds,
    market: MarketType,
) -> str | None:
    """Reject currently unsupported team-, player-, or period-specific markets."""

    normalized_period = normalize_optional_text(row.period) or "match"
    normalized_scope = normalize_optional_text(row.participant_scope) or "match"
    if normalized_period != "match":
        return (
            f"period `{normalized_period}` is preserved but not scoreable in the current "
            f"canonical `{market.value}` taxonomy."
        )
    if normalized_scope != "match":
        return (
            f"participant scope `{normalized_scope}` is preserved but not scoreable in the "
            f"current canonical `{market.value}` taxonomy."
        )
    return None


def _validate_sport_alignment(
    market: MarketType,
    sport: SportName | None,
) -> str | None:
    """Reject market mappings that conflict with the resolved sport context."""

    if sport == SportName.BASKETBALL and market in _SOCCER_MARKETS:
        return f"`{market.value}` is a soccer market but the row resolved to basketball."
    if sport == SportName.SOCCER and market in _BASKETBALL_MARKETS:
        return f"`{market.value}` is a basketball market but the row resolved to soccer."
    return None


def _validate_line_requirements(
    row: NormalizedOdds,
    market: MarketType,
) -> str | None:
    """Reject line-based canonical mappings when the provider row lacks a line."""

    if market in _LINE_BASED_MARKETS and row.line is None:
        return f"`{market.value}` requires a numeric line, but the provider row omitted one."
    return None


def _resolve_sport(
    row: NormalizedOdds,
    *,
    explicit_sport: SportName | None,
) -> SportName | None:
    """Resolve the row sport from explicit overrides, metadata, or source hints."""

    if explicit_sport is not None:
        return explicit_sport
    raw_sport = row.raw_metadata.get("sport")
    if isinstance(raw_sport, str):
        normalized_sport = raw_sport.strip().lower()
        if normalized_sport == SportName.SOCCER.value:
            return SportName.SOCCER
        if normalized_sport == SportName.BASKETBALL.value:
            return SportName.BASKETBALL
    raw_sport_key = row.raw_metadata.get("sport_key")
    if isinstance(raw_sport_key, str):
        normalized_key = raw_sport_key.strip().lower()
        if normalized_key.startswith("soccer_"):
            return SportName.SOCCER
        if normalized_key.startswith("basketball_"):
            return SportName.BASKETBALL
    if row.market in _BASKETBALL_MARKETS:
        return SportName.BASKETBALL
    if row.market in _SOCCER_MARKETS:
        return SportName.SOCCER
    return None


def _shared_market_metadata(rows: Sequence[NormalizedOdds]) -> dict[str, JSONPrimitive]:
    """Extract the metadata key-value pairs shared by every row in one market."""

    if not rows:
        return {}
    shared_metadata = dict(rows[0].raw_metadata)
    for row in rows[1:]:
        for key in tuple(shared_metadata.keys()):
            if row.raw_metadata.get(key) != shared_metadata[key]:
                shared_metadata.pop(key, None)
    return shared_metadata


def _latest_timestamp(rows: Sequence[NormalizedOdds]) -> datetime | None:
    """Return the most recent `last_updated` timestamp across grouped rows."""

    timestamps = [row.last_updated for row in rows if row.last_updated is not None]
    if not timestamps:
        return None
    return max(timestamps)


def _extract_provider_selection_id(
    raw_metadata: Mapping[str, JSONPrimitive],
) -> int | str | None:
    """Extract one provider-native selection identifier from preserved metadata."""

    candidate = raw_metadata.get("provider_selection_id")
    if isinstance(candidate, int):
        return candidate
    if isinstance(candidate, str):
        normalized = candidate.strip()
        return normalized or None
    return None


def _extract_teams(
    raw_metadata: Mapping[str, JSONPrimitive],
) -> tuple[str | None, str | None]:
    """Extract home and away team labels from preserved provider metadata."""

    home_team = raw_metadata.get("home_team")
    away_team = raw_metadata.get("away_team")
    return (
        home_team.strip() if isinstance(home_team, str) and home_team.strip() else None,
        away_team.strip() if isinstance(away_team, str) and away_team.strip() else None,
    )


def _normalize_match_result_selection(
    raw_selection: str,
    *,
    home_team: str | None,
    away_team: str | None,
    allow_draw: bool,
    fallback: str,
) -> str:
    """Normalize winner-style provider labels into `home`, `away`, or `draw`."""

    normalized_selection = _normalize_machine_key(raw_selection)
    if normalized_selection in _HOME_ALIASES or _matches_team_alias(raw_selection, home_team):
        return "home"
    if normalized_selection in _AWAY_ALIASES or _matches_team_alias(raw_selection, away_team):
        return "away"
    if allow_draw and normalized_selection in _DRAW_ALIASES:
        return "draw"
    return fallback


def _normalize_side_selection(
    raw_selection: str,
    *,
    home_team: str | None,
    away_team: str | None,
    fallback: str,
) -> str:
    """Normalize side-based selections into `home` or `away` when possible."""

    normalized_selection = _normalize_machine_key(raw_selection)
    if normalized_selection in _HOME_ALIASES or _matches_team_alias(raw_selection, home_team):
        return "home"
    if normalized_selection in _AWAY_ALIASES or _matches_team_alias(raw_selection, away_team):
        return "away"
    return fallback


def _normalize_binary_selection(
    raw_selection: str,
    *,
    positive_aliases: set[str] | frozenset[str],
    negative_aliases: set[str] | frozenset[str],
    positive_value: str,
    negative_value: str,
    fallback: str,
) -> str:
    """Normalize binary provider outcomes such as over/under or yes/no."""

    normalized_selection = _normalize_machine_key(raw_selection)
    if normalized_selection in positive_aliases or normalized_selection.startswith(
        f"{positive_value}_"
    ):
        return positive_value
    if normalized_selection in negative_aliases or normalized_selection.startswith(
        f"{negative_value}_"
    ):
        return negative_value
    return fallback


def _normalize_double_chance_selection(
    raw_selection: str,
    *,
    home_team: str | None,
    away_team: str | None,
    fallback: str,
) -> str:
    """Normalize double-chance labels into the canonical `1X`, `12`, or `X2` tokens."""

    normalized_selection = _normalize_machine_key(raw_selection)
    if home_team is not None:
        normalized_selection = normalized_selection.replace(
            _normalize_machine_key(home_team),
            "home",
        )
    if away_team is not None:
        normalized_selection = normalized_selection.replace(
            _normalize_machine_key(away_team),
            "away",
        )
    normalized_selection = normalized_selection.replace("home", "1").replace("away", "2")
    mapped_selection = _DOUBLE_CHANCE_SELECTIONS.get(normalized_selection)
    if mapped_selection is not None:
        return mapped_selection
    return fallback


def _normalize_ht_ft_selection(raw_selection: str, *, fallback: str) -> str:
    """Normalize half-time/full-time labels into compact `1/X/2` token pairs."""

    parts = [part for part in re.split(r"\s*/\s*", raw_selection.strip()) if part]
    if len(parts) != 2:
        return fallback
    normalized_parts: list[str] = []
    for part in parts:
        token = _HT_FT_TOKEN_MAP.get(_normalize_machine_key(part))
        if token is None:
            return fallback
        normalized_parts.append(token)
    return "/".join(normalized_parts)


def _matches_team_alias(raw_selection: str, team_name: str | None) -> bool:
    """Report whether the provider selection clearly refers to one team."""

    if team_name is None:
        return False
    normalized_selection = _normalize_machine_key(raw_selection)
    normalized_team = _normalize_machine_key(team_name)
    return (
        normalized_selection == normalized_team
        or normalized_selection.startswith(f"{normalized_team}_")
        or normalized_selection.endswith(f"_{normalized_team}")
    )


def _normalize_machine_key(value: str) -> str:
    """Convert provider labels into deterministic machine-friendly keys."""

    normalized = require_non_blank_text(value, "provider_value").lower()
    compacted = _NON_ALNUM_PATTERN.sub("_", normalized).strip("_")
    if not compacted:
        raise ValueError("provider-derived keys must contain at least one alphanumeric.")
    return compacted


type _MarketGroupingKey = tuple[
    str,
    str,
    str,
    str,
    int | str | None,
    str | None,
    str | None,
]


__all__ = [
    "CanonicalOddsSelection",
    "OddsMarketCatalog",
    "ProviderMarketCatalogEntry",
    "build_odds_market_catalog",
    "canonicalize_odds_rows",
    "filter_scoreable_odds",
    "filter_unmapped_odds",
    "group_markets_by_canonical_market",
    "group_markets_by_fixture",
]
