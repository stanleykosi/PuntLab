"""Canonical SportyBet market resolver for PuntLab's recommendation pipeline.

Purpose: choose the best available bookmaker market for one analyzed fixture by
trying the fast SportyBet API interceptor first, then the browser fallback, and
finally a supplied pool of normalized external odds.
Scope: fixture-aware source fallback, canonical market/selection matching,
line-based tie breaking, and `ResolvedMarket` construction for downstream
accumulator building.
Dependencies: shared normalized fixture, odds, and analysis schemas plus the
SportyBet HTTP and browser scrapers that provide bookmaker market snapshots.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from math import isclose

from src.cache.client import RedisClient
from src.config import WAT_TIMEZONE, MarketType
from src.providers.base import ProviderError
from src.providers.odds_mapping import filter_scoreable_odds
from src.schemas.accumulators import ResolutionSource, ResolvedMarket
from src.schemas.analysis import MatchScore, RankedMatch
from src.schemas.fixtures import NormalizedFixture
from src.schemas.odds import NormalizedOdds
from src.scrapers.sportybet_api import SportyBetAPIClient
from src.scrapers.sportybet_browser import SportyBetBrowserScraper

_LINE_BASED_MARKETS = frozenset(
    {
        MarketType.ASIAN_HANDICAP,
        MarketType.OVER_UNDER_05,
        MarketType.OVER_UNDER_15,
        MarketType.OVER_UNDER_25,
        MarketType.OVER_UNDER_35,
        MarketType.POINT_SPREAD,
        MarketType.TOTAL_POINTS,
    }
)
_MATCH_RESULT_HOME = frozenset({"1", "home"})
_MATCH_RESULT_AWAY = frozenset({"2", "away"})
_MATCH_RESULT_DRAW = frozenset({"draw", "x"})
_YES_SELECTIONS = frozenset({"yes"})
_NO_SELECTIONS = frozenset({"no"})
_OVER_SELECTIONS = frozenset({"over"})
_UNDER_SELECTIONS = frozenset({"under"})
_DOUBLE_CHANCE_PATTERN = re.compile(r"^(?:1x|x2|12)$", re.IGNORECASE)
_HT_FT_PATTERN = re.compile(r"^(1|x|2)\s*/\s*(1|x|2)$", re.IGNORECASE)
_NUMERIC_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
_LINE_TOLERANCE = 0.001


class MarketResolver:
    """Resolve one analyzed fixture into the best available bookmaker market.

    Args:
        cache: Optional shared Redis cache used when the resolver constructs
            the SportyBet scraper dependencies itself.
        api_client: Optional injected SportyBet API interceptor.
        browser_scraper: Optional injected SportyBet browser fallback scraper.
        clock: Optional timezone-aware clock for deterministic tests.
    """

    def __init__(
        self,
        cache: RedisClient | None = None,
        *,
        api_client: SportyBetAPIClient | None = None,
        browser_scraper: SportyBetBrowserScraper | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the resolver and lazily create shared scraper dependencies."""

        shared_cache = cache
        self._owns_cache = False
        if shared_cache is None and (api_client is None or browser_scraper is None):
            shared_cache = RedisClient()
            self._owns_cache = True

        self._owns_api_client = api_client is None
        self._owns_browser_scraper = browser_scraper is None
        self._api_client = api_client or SportyBetAPIClient(shared_cache)
        self._browser_scraper = browser_scraper or SportyBetBrowserScraper(shared_cache)
        self._cache = shared_cache
        self._clock = clock or (lambda: datetime.now(WAT_TIMEZONE))

    async def resolve(
        self,
        fixture: NormalizedFixture,
        analysis: MatchScore | RankedMatch,
        *,
        external_odds: Sequence[NormalizedOdds] = (),
        use_cache: bool = True,
    ) -> ResolvedMarket:
        """Resolve the best matching market for one analyzed fixture.

        Inputs:
            fixture: Canonical fixture metadata for the scored match.
            analysis: Match scoring output containing the recommended market and
                recommended selection to resolve.
            external_odds: Optional fallback pool of normalized external odds.
                Rows may already be pre-filtered or may represent a broader
                slate; the resolver will match the relevant fixture rows.
            use_cache: Whether the SportyBet scrapers may reuse cached market
                snapshots before fetching fresh data.

        Outputs:
            One resolved bookmaker market that downstream accumulator building
            can use directly.

        Raises:
            ValueError: If the analysis does not include a recommended market
                and selection, or if required fixture data is malformed.
            ProviderError: If none of the configured sources can supply a
                compatible market for the recommendation.
        """

        recommended_market = analysis.recommended_market
        recommended_selection = analysis.recommended_selection
        if recommended_market is None or recommended_selection is None:
            raise ValueError(
                "analysis must include recommended_market and recommended_selection."
            )

        normalized_target_selection, target_line = self._normalize_target_selection(
            recommended_selection,
            market=recommended_market,
            fixture=fixture,
        )
        sportybet_url = self._build_sportybet_url(fixture)
        diagnostics: list[str] = []

        if fixture.sportradar_id is not None:
            try:
                api_rows = await self._api_client.fetch_markets(
                    fixture.sportradar_id,
                    fixture=fixture,
                    use_cache=use_cache,
                )
            except ProviderError as exc:
                diagnostics.append(f"SportyBet API failed: {exc}")
            else:
                resolved_api_market = self._select_resolved_market(
                    rows=api_rows,
                    fixture=fixture,
                    market=recommended_market,
                    normalized_selection=normalized_target_selection,
                    target_line=target_line,
                    recommended_odds=analysis.recommended_odds,
                    resolution_source=ResolutionSource.SPORTYBET_API,
                    sportybet_url=sportybet_url,
                )
                if resolved_api_market is not None:
                    return resolved_api_market
                diagnostics.append(
                    "SportyBet API returned markets, but none matched the recommended "
                    f"market `{recommended_market.value}` and selection `{recommended_selection}`."
                )

            if sportybet_url is not None:
                try:
                    browser_rows = await self._browser_scraper.scrape_markets(
                        sportybet_url,
                        fixture=fixture,
                        use_cache=use_cache,
                    )
                except ProviderError as exc:
                    diagnostics.append(f"SportyBet browser fallback failed: {exc}")
                else:
                    resolved_browser_market = self._select_resolved_market(
                        rows=browser_rows,
                        fixture=fixture,
                        market=recommended_market,
                        normalized_selection=normalized_target_selection,
                        target_line=target_line,
                        recommended_odds=analysis.recommended_odds,
                        resolution_source=ResolutionSource.SPORTYBET_BROWSER,
                        sportybet_url=sportybet_url,
                    )
                    if resolved_browser_market is not None:
                        return resolved_browser_market
                    diagnostics.append(
                        "SportyBet browser fallback returned markets, but none matched the "
                        f"recommended market `{recommended_market.value}` and selection "
                        f"`{recommended_selection}`."
                    )
            else:
                diagnostics.append(
                    "SportyBet browser fallback was skipped because the public fixture URL "
                    "could not be constructed."
                )
        else:
            diagnostics.append(
                "SportyBet resolution was skipped because fixture.sportradar_id is missing."
            )

        resolved_external_market = self._select_resolved_market(
            rows=external_odds,
            fixture=fixture,
            market=recommended_market,
            normalized_selection=normalized_target_selection,
            target_line=target_line,
            recommended_odds=analysis.recommended_odds,
            resolution_source=ResolutionSource.EXTERNAL_ODDS,
            sportybet_url=sportybet_url,
        )
        if resolved_external_market is not None:
            return resolved_external_market

        diagnostics.append(
            "External odds fallback did not contain a compatible canonical market for the fixture."
        )
        raise ProviderError(
            "market-resolver",
            (
                "Could not resolve a bookmaker market for fixture "
                f"{fixture.get_fixture_ref()} ({fixture.home_team} vs {fixture.away_team}). "
                f"Recommended market: `{recommended_market.value}`. "
                f"Recommended selection: `{recommended_selection}`. "
                f"Diagnostics: {' | '.join(diagnostics)}"
            ),
        )

    async def aclose(self) -> None:
        """Close owned scraper dependencies and any shared cache created here."""

        if self._owns_api_client:
            await self._api_client.aclose()
        if self._owns_browser_scraper:
            await self._browser_scraper.aclose()
        if self._owns_cache and self._cache is not None:
            await self._cache.close()

    def _select_resolved_market(
        self,
        *,
        rows: Sequence[NormalizedOdds],
        fixture: NormalizedFixture,
        market: MarketType,
        normalized_selection: str,
        target_line: float | None,
        recommended_odds: float | None,
        resolution_source: ResolutionSource,
        sportybet_url: str | None,
    ) -> ResolvedMarket | None:
        """Filter, rank, and materialize the best candidate from one source."""

        matching_rows = self._matching_rows(
            rows=rows,
            fixture=fixture,
            market=market,
            normalized_selection=normalized_selection,
            target_line=target_line,
        )
        if not matching_rows:
            return None

        best_row = min(
            matching_rows,
            key=lambda row: self._candidate_sort_key(
                row=row,
                market=market,
                recommended_odds=recommended_odds,
            ),
        )
        return self._build_resolved_market(
            row=best_row,
            fixture=fixture,
            market=market,
            resolution_source=resolution_source,
            sportybet_url=sportybet_url,
        )

    def _matching_rows(
        self,
        *,
        rows: Sequence[NormalizedOdds],
        fixture: NormalizedFixture,
        market: MarketType,
        normalized_selection: str,
        target_line: float | None,
    ) -> tuple[NormalizedOdds, ...]:
        """Return source rows that match the requested fixture, market, and leg."""

        relevant_rows = tuple(row for row in rows if self._row_matches_fixture(row, fixture))
        if not relevant_rows:
            return ()

        sport_by_fixture = {row.fixture_ref: fixture.sport for row in relevant_rows}
        canonical_rows = filter_scoreable_odds(
            relevant_rows,
            sport_by_fixture=sport_by_fixture,
        )

        matching_rows: list[NormalizedOdds] = []
        for row in canonical_rows:
            if row.market != market:
                continue
            candidate_selection, _ = self._normalize_target_selection(
                row.selection,
                market=market,
                fixture=fixture,
            )
            if candidate_selection != normalized_selection:
                continue
            if target_line is not None:
                if row.line is None or not isclose(row.line, target_line, abs_tol=_LINE_TOLERANCE):
                    continue
            matching_rows.append(row)
        return tuple(matching_rows)

    def _build_resolved_market(
        self,
        *,
        row: NormalizedOdds,
        fixture: NormalizedFixture,
        market: MarketType,
        resolution_source: ResolutionSource,
        sportybet_url: str | None,
    ) -> ResolvedMarket:
        """Enrich one chosen odds row with resolution metadata."""

        return ResolvedMarket(
            fixture_ref=fixture.get_fixture_ref(),
            market=market,
            selection=row.selection,
            odds=row.odds,
            provider=row.provider,
            provider_market_name=row.provider_market_name,
            provider_selection_name=row.provider_selection_name,
            provider_market_key=row.provider_market_key,
            provider_selection_key=row.provider_selection_key,
            sportybet_available=row.sportybet_available,
            market_label=row.market_label,
            line=row.line,
            period=row.period,
            participant_scope=row.participant_scope,
            provider_market_id=row.provider_market_id,
            raw_metadata=row.raw_metadata,
            last_updated=row.last_updated,
            resolution_source=resolution_source,
            sport=fixture.sport,
            competition=fixture.competition,
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            sportybet_market_id=self._extract_sportybet_market_id(row),
            sportybet_url=sportybet_url,
            resolved_at=self._clock().astimezone(UTC),
        )

    def _candidate_sort_key(
        self,
        *,
        row: NormalizedOdds,
        market: MarketType,
        recommended_odds: float | None,
    ) -> tuple[float, float, float]:
        """Rank matching rows so the resolver picks the most faithful candidate."""

        odds_delta = (
            abs(row.odds - recommended_odds)
            if recommended_odds is not None
            else 0.0
        )
        if market in _LINE_BASED_MARKETS:
            line_magnitude = abs(row.line) if row.line is not None else float("inf")
            return (odds_delta, line_magnitude, -row.odds)
        return (odds_delta, 0.0, -row.odds)

    def _row_matches_fixture(self, row: NormalizedOdds, fixture: NormalizedFixture) -> bool:
        """Match one normalized odds row to the requested fixture."""

        fixture_ref = fixture.get_fixture_ref()
        if row.fixture_ref == fixture_ref:
            return True
        if fixture.sportradar_id is not None:
            requested_sportradar_id = row.raw_metadata.get("requested_sportradar_id")
            preserved_sportradar_id = row.raw_metadata.get("sportradar_id")
            if requested_sportradar_id == fixture.sportradar_id:
                return True
            if preserved_sportradar_id == fixture.sportradar_id:
                return True
            if row.fixture_ref == fixture.sportradar_id:
                return True

        home_team = row.raw_metadata.get("home_team")
        away_team = row.raw_metadata.get("away_team")
        if not isinstance(home_team, str) or not isinstance(away_team, str):
            return False
        if home_team.strip().casefold() != fixture.home_team.casefold():
            return False
        if away_team.strip().casefold() != fixture.away_team.casefold():
            return False

        raw_commence_time = row.raw_metadata.get("commence_time")
        if isinstance(raw_commence_time, str):
            parsed_commence_time = self._parse_datetime(raw_commence_time)
            if parsed_commence_time is not None:
                fixture_date = fixture.kickoff.astimezone(UTC).date()
                row_date = parsed_commence_time.astimezone(UTC).date()
                return fixture_date == row_date

        return True

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        """Parse one ISO datetime string conservatively for fixture matching."""

        normalized = value.strip()
        if not normalized:
            return None
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.utcoffset() is None:
            return None
        return parsed

    def _build_sportybet_url(self, fixture: NormalizedFixture) -> str | None:
        """Construct the public SportyBet URL when the fixture supports it."""

        if fixture.sportradar_id is None:
            return None
        return self._api_client.build_sportybet_url(fixture)

    @staticmethod
    def _extract_sportybet_market_id(row: NormalizedOdds) -> int | None:
        """Parse the preserved provider market ID into an integer when possible."""

        if isinstance(row.provider_market_id, int):
            return row.provider_market_id
        if isinstance(row.provider_market_id, str) and row.provider_market_id.strip().isdigit():
            return int(row.provider_market_id.strip())
        return None

    def _normalize_target_selection(
        self,
        selection: str,
        *,
        market: MarketType,
        fixture: NormalizedFixture,
    ) -> tuple[str, float | None]:
        """Normalize one target selection into canonical comparison tokens."""

        normalized = selection.strip()
        if not normalized:
            raise ValueError("recommended_selection must not be blank.")
        normalized_key = self._normalize_key(normalized)
        target_line = self._extract_numeric_line(normalized)
        normalized_selection_text = normalized.casefold()

        if market == MarketType.MATCH_RESULT:
            if normalized_key in _MATCH_RESULT_HOME or normalized_key == self._normalize_key(
                fixture.home_team
            ):
                return "home", None
            if normalized_key in _MATCH_RESULT_AWAY or normalized_key == self._normalize_key(
                fixture.away_team
            ):
                return "away", None
            if normalized_key in _MATCH_RESULT_DRAW:
                return "draw", None
            return normalized.lower(), None

        if market in {
            MarketType.MONEYLINE,
            MarketType.DRAW_NO_BET,
            MarketType.ASIAN_HANDICAP,
            MarketType.POINT_SPREAD,
        }:
            if (
                normalized_key in _MATCH_RESULT_HOME
                or fixture.home_team.casefold() in normalized_selection_text
            ):
                return "home", target_line
            if (
                normalized_key in _MATCH_RESULT_AWAY
                or fixture.away_team.casefold() in normalized_selection_text
            ):
                return "away", target_line
            return normalized.lower(), target_line

        if market in {
            MarketType.OVER_UNDER_05,
            MarketType.OVER_UNDER_15,
            MarketType.OVER_UNDER_25,
            MarketType.OVER_UNDER_35,
            MarketType.TOTAL_POINTS,
        }:
            if normalized_key in _OVER_SELECTIONS or normalized_key.startswith("over_"):
                return "over", target_line
            if normalized_key in _UNDER_SELECTIONS or normalized_key.startswith("under_"):
                return "under", target_line
            return normalized.lower(), target_line

        if market == MarketType.BTTS:
            if normalized_key in _YES_SELECTIONS:
                return "yes", None
            if normalized_key in _NO_SELECTIONS:
                return "no", None
            return normalized.lower(), None

        if market == MarketType.DOUBLE_CHANCE and _DOUBLE_CHANCE_PATTERN.fullmatch(normalized_key):
            return normalized_key.upper(), None

        if market == MarketType.HT_FT:
            match = _HT_FT_PATTERN.fullmatch(normalized)
            if match is not None:
                return f"{match.group(1).upper()}/{match.group(2).upper()}", None

        return normalized, target_line

    @staticmethod
    def _normalize_key(value: str) -> str:
        """Convert free-form labels into simple lowercase comparison keys."""

        return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")

    @staticmethod
    def _extract_numeric_line(value: str) -> float | None:
        """Extract the first numeric token from a selection label when present."""

        match = _NUMERIC_PATTERN.search(value)
        if match is None:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None


__all__ = ["MarketResolver"]
