"""Composite scoring engine for PuntLab's match-analysis pipeline.

Purpose: combine deterministic and qualitative factors into one canonical
match score, confidence estimate, and recommended market selection.
Scope: fixture-scoped factor orchestration, data-completeness and market
agreement confidence handling, and recommendation selection from the
canonically mapped scoreable subset of the full odds universe.
Dependencies: scoring-factor modules, odds mapping helpers, shared normalized
schemas, and the canonical scoring weights exported from `src.scoring.weights`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose
from statistics import fmean
from typing import Final

from src.config import MarketType, SportName
from src.providers.odds_mapping import filter_scoreable_odds
from src.schemas.analysis import MatchContext, MatchScore, ScoreFactorBreakdown
from src.schemas.fixtures import NormalizedFixture
from src.schemas.odds import NormalizedOdds
from src.schemas.stats import InjuryData, TeamStats
from src.scoring.factors.context import NEUTRAL_CONTEXT_SCORE
from src.scoring.factors.form import analyze_form
from src.scoring.factors.h2h import analyze_h2h
from src.scoring.factors.injuries import analyze_injuries
from src.scoring.factors.odds_value import analyze_odds_value
from src.scoring.factors.venue import analyze_venue
from src.scoring.weights import ScoringWeights, get_default_scoring_weights

_SOCCER_TOTAL_MARKETS: Final[frozenset[MarketType]] = frozenset(
    {
        MarketType.OVER_UNDER_05,
        MarketType.OVER_UNDER_15,
        MarketType.OVER_UNDER_25,
        MarketType.OVER_UNDER_35,
    }
)
_LOW_COMPLEXITY_MARKETS: Final[frozenset[MarketType]] = frozenset(
    {
        MarketType.MATCH_RESULT,
        MarketType.MONEYLINE,
        MarketType.BTTS,
        MarketType.OVER_UNDER_05,
        MarketType.OVER_UNDER_15,
        MarketType.OVER_UNDER_25,
        MarketType.OVER_UNDER_35,
        MarketType.TOTAL_POINTS,
    }
)
_MEDIUM_COMPLEXITY_MARKETS: Final[frozenset[MarketType]] = frozenset(
    {
        MarketType.DOUBLE_CHANCE,
        MarketType.DRAW_NO_BET,
        MarketType.ASIAN_HANDICAP,
        MarketType.POINT_SPREAD,
    }
)
_OVER_SELECTIONS: Final[frozenset[str]] = frozenset({"over"})
_UNDER_SELECTIONS: Final[frozenset[str]] = frozenset({"under"})
_YES_SELECTIONS: Final[frozenset[str]] = frozenset({"yes"})
_NO_SELECTIONS: Final[frozenset[str]] = frozenset({"no"})
_DOUBLE_CHANCE_SELECTIONS: Final[frozenset[str]] = frozenset({"1X", "12", "X2"})
_SOCCER_BASELINE_TOTAL: Final[float] = 2.4
_BASKETBALL_BASELINE_TOTAL: Final[float] = 224.5

type _CandidateGroupKey = tuple[MarketType, str, float | None, str | None, str | None]


@dataclass(frozen=True, slots=True)
class _ResolvedTeamStats:
    """Fixture-specific view of the available team-stat snapshots.

    Inputs:
        Canonical `TeamStats` rows already matched to the target fixture.

    Outputs:
        The best matching home and away snapshots plus the ordered subset used
        for factor calculation.
    """

    home: TeamStats | None
    away: TeamStats | None
    relevant: tuple[TeamStats, ...]


@dataclass(frozen=True, slots=True)
class _FixtureInsights:
    """Derived fixture signals used for market-selection heuristics.

    Inputs:
        Matched team stats plus optional context and injury data.

    Outputs:
        Directional and totals-oriented signals that help the scoring engine
        choose the most coherent market from the scoreable odds subset.
    """

    side_edge: float
    home_strength: float
    away_strength: float
    expected_total: float
    btts_likelihood: float
    draw_likelihood: float


@dataclass(frozen=True, slots=True)
class _CandidateRecommendation:
    """One grouped market candidate considered for recommendation selection.

    Inputs:
        Scoreable odds rows representing the same canonical market, line, and
        selection across one or more providers.

    Outputs:
        The best available offered odds plus agreement and fit metadata used
        to rank candidate recommendations.
    """

    market: MarketType
    selection: str
    line: float | None
    rows: tuple[NormalizedOdds, ...]
    representative_row: NormalizedOdds
    recommended_odds: float
    fit_score: float
    agreement_score: float
    provider_count: int


class ScoringEngine:
    """Calculate composite match scores and recommendation selections.

    Inputs:
        Validated fixture metadata, team stats, normalized odds, optional
        match context, optional injuries, and optional H2H fixtures.

    Outputs:
        A canonical `MatchScore` record containing factor breakdowns, the
        composite score, confidence, and the best-fit recommended market and
        selection when scoreable odds are available.
    """

    def __init__(self, *, weights: ScoringWeights | None = None) -> None:
        """Initialize the scoring engine with validated factor weights.

        Inputs:
            weights: Optional custom `ScoringWeights` instance. When omitted,
                the engine uses PuntLab's canonical defaults.

        Outputs:
            A scoring engine ready to score one fixture at a time.
        """

        self.weights = (
            weights.model_copy(deep=True)
            if weights is not None
            else get_default_scoring_weights()
        )

    def calculate_match_score(
        self,
        fixture: NormalizedFixture,
        team_stats: Sequence[TeamStats],
        odds: Sequence[NormalizedOdds],
        *,
        context: MatchContext | None = None,
        injuries: Sequence[InjuryData] | None = None,
        h2h_data: Sequence[NormalizedFixture] | None = None,
    ) -> MatchScore:
        """Calculate the canonical composite score for one fixture.

        Inputs:
            fixture: Canonical fixture to score.
            team_stats: Team-stat snapshots for the fixture teams.
            odds: Fixture-scoped odds rows or a broader odds slate.
            context: Optional qualitative match context. Missing context is
                treated as a neutral qualitative signal.
            injuries: Optional fixture-scoped injury rows. `None` means the
                data source is missing; an empty sequence means the availability
                picture is known and clear.
            h2h_data: Optional historical head-to-head fixtures.

        Outputs:
            A fully validated `MatchScore` containing factor breakdowns,
            composite score, confidence, and recommendation metadata.

        Raises:
            TypeError: If the fixture is not canonical or the team stats are not
                canonical `TeamStats` rows.
            ValueError: If the fixture teams cannot be matched to any supplied
                team-stat snapshots.
        """

        if not isinstance(fixture, NormalizedFixture):
            raise TypeError("calculate_match_score expects a NormalizedFixture instance.")

        matched_team_stats = self._resolve_team_stats(fixture, team_stats)
        relevant_odds = self._relevant_odds_for_fixture(fixture, odds)
        scoreable_odds = self._scoreable_odds_for_fixture(relevant_odds, fixture)

        form_score = analyze_form(matched_team_stats.relevant)
        venue_score = analyze_venue(fixture, matched_team_stats.relevant)
        h2h_score = self._calculate_h2h_score(fixture, h2h_data)
        injury_score = self._calculate_injury_score(injuries)
        odds_value_score = self._calculate_odds_value_score(relevant_odds)
        context_score = (
            context.qualitative_score if context is not None else NEUTRAL_CONTEXT_SCORE
        )
        statistical_score = self._calculate_statistical_score(matched_team_stats)

        factors = ScoreFactorBreakdown(
            form=form_score,
            h2h=h2h_score,
            injury_impact=injury_score,
            odds_value=odds_value_score,
            context=context_score,
            venue=venue_score,
            statistical=statistical_score,
        )
        composite_score = self._calculate_composite_score(factors)

        recommendation = self._select_best_candidate(
            fixture=fixture,
            matched_team_stats=matched_team_stats,
            scoreable_odds=scoreable_odds,
            context=context,
            injuries=injuries or (),
            composite_score=composite_score,
        )

        data_completeness = self._calculate_data_completeness(
            matched_team_stats=matched_team_stats,
            relevant_odds=relevant_odds,
            scoreable_odds=scoreable_odds,
            context=context,
            injuries=injuries,
            h2h_data=h2h_data,
        )
        odds_agreement = self._calculate_odds_agreement(recommendation)
        recommendation_fit = recommendation.fit_score if recommendation is not None else 0.0
        confidence = self.calculate_confidence(
            composite_score=composite_score,
            data_completeness=data_completeness,
            odds_agreement=odds_agreement,
            recommendation_fit=recommendation_fit,
        )

        return MatchScore(
            fixture_ref=fixture.get_fixture_ref(),
            sport=fixture.sport,
            competition=fixture.competition,
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            composite_score=composite_score,
            confidence=confidence,
            factors=factors,
            recommended_market=recommendation.market if recommendation is not None else None,
            recommended_selection=(
                recommendation.selection if recommendation is not None else None
            ),
            recommended_odds=(
                recommendation.recommended_odds if recommendation is not None else None
            ),
            qualitative_summary=self._build_qualitative_summary(context),
        )

    def calculate_confidence(
        self,
        *,
        composite_score: float,
        data_completeness: float,
        odds_agreement: float,
        recommendation_fit: float,
    ) -> float:
        """Calculate a bounded confidence score for one scored fixture.

        Inputs:
            composite_score: Weighted factor score from the main engine.
            data_completeness: How complete the supporting dataset is.
            odds_agreement: How closely providers align on the recommendation.
            recommendation_fit: How well the selected market fits the fixture
                profile according to the engine's heuristics.

        Outputs:
            A bounded confidence score in the `0.0-1.0` range.
        """

        confidence = (
            (composite_score * 0.55)
            + (data_completeness * 0.20)
            + (odds_agreement * 0.15)
            + (recommendation_fit * 0.10)
        )
        return _clamp(confidence)

    def select_best_market(
        self,
        fixture: NormalizedFixture,
        team_stats: Sequence[TeamStats],
        odds: Sequence[NormalizedOdds],
        *,
        context: MatchContext | None = None,
        injuries: Sequence[InjuryData] | None = None,
        composite_score: float = 0.5,
    ) -> MarketType | None:
        """Return the best-fit market family from the scoreable odds subset."""

        candidate = self._select_best_candidate(
            fixture=fixture,
            matched_team_stats=self._resolve_team_stats(fixture, team_stats),
            scoreable_odds=self._scoreable_odds_for_fixture(
                self._relevant_odds_for_fixture(fixture, odds),
                fixture,
            ),
            context=context,
            injuries=injuries or (),
            composite_score=composite_score,
        )
        return candidate.market if candidate is not None else None

    def select_best_selection(
        self,
        fixture: NormalizedFixture,
        team_stats: Sequence[TeamStats],
        odds: Sequence[NormalizedOdds],
        *,
        context: MatchContext | None = None,
        injuries: Sequence[InjuryData] | None = None,
        composite_score: float = 0.5,
    ) -> str | None:
        """Return the best-fit selection label from the scoreable odds subset."""

        candidate = self._select_best_candidate(
            fixture=fixture,
            matched_team_stats=self._resolve_team_stats(fixture, team_stats),
            scoreable_odds=self._scoreable_odds_for_fixture(
                self._relevant_odds_for_fixture(fixture, odds),
                fixture,
            ),
            context=context,
            injuries=injuries or (),
            composite_score=composite_score,
        )
        return candidate.selection if candidate is not None else None

    def _calculate_composite_score(self, factors: ScoreFactorBreakdown) -> float:
        """Combine all factor scores into the canonical weighted composite."""

        composite = (
            (factors.form * self.weights.form)
            + (factors.h2h * self.weights.h2h)
            + (factors.injury_impact * self.weights.injury_impact)
            + (factors.odds_value * self.weights.odds_value)
            + (factors.context * self.weights.context)
            + (factors.venue * self.weights.venue)
            + (factors.statistical * self.weights.statistical)
        )
        return _clamp(composite)

    def _resolve_team_stats(
        self,
        fixture: NormalizedFixture,
        team_stats: Sequence[TeamStats],
    ) -> _ResolvedTeamStats:
        """Match the richest home and away team snapshots to the fixture."""

        normalized_team_stats = tuple(team_stats)
        if not normalized_team_stats:
            raise ValueError("calculate_match_score requires at least one TeamStats record.")

        for stats in normalized_team_stats:
            if not isinstance(stats, TeamStats):
                raise TypeError("calculate_match_score expects TeamStats instances only.")
            if stats.sport != fixture.sport:
                raise ValueError(
                    "calculate_match_score requires TeamStats from the same sport as the fixture."
                )

        home_stats = self._select_best_team_snapshot(
            normalized_team_stats,
            team_id=fixture.home_team_id,
            team_name=fixture.home_team,
        )
        away_stats = self._select_best_team_snapshot(
            normalized_team_stats,
            team_id=fixture.away_team_id,
            team_name=fixture.away_team,
        )
        relevant = tuple(
            stats
            for stats in (home_stats, away_stats)
            if stats is not None
        )
        if not relevant:
            raise ValueError(
                "calculate_match_score could not match any TeamStats records to the fixture."
            )

        return _ResolvedTeamStats(home=home_stats, away=away_stats, relevant=relevant)

    def _relevant_odds_for_fixture(
        self,
        fixture: NormalizedFixture,
        odds: Sequence[NormalizedOdds],
    ) -> tuple[NormalizedOdds, ...]:
        """Filter a broader odds slate down to the requested fixture."""

        relevant_rows: list[NormalizedOdds] = []
        for row in odds:
            if not isinstance(row, NormalizedOdds):
                raise TypeError("calculate_match_score expects NormalizedOdds instances only.")
            if self._row_matches_fixture(row, fixture):
                relevant_rows.append(row)
        return tuple(relevant_rows)

    def _scoreable_odds_for_fixture(
        self,
        relevant_odds: Sequence[NormalizedOdds],
        fixture: NormalizedFixture,
    ) -> tuple[NormalizedOdds, ...]:
        """Project relevant odds rows into the current scoreable subset."""

        return filter_scoreable_odds(
            relevant_odds,
            sport_by_fixture={row.fixture_ref: fixture.sport for row in relevant_odds},
        )

    def _calculate_h2h_score(
        self,
        fixture: NormalizedFixture,
        h2h_data: Sequence[NormalizedFixture] | None,
    ) -> float:
        """Return a neutral H2H score when historical meetings are unavailable."""

        if h2h_data is None:
            return 0.5
        if not h2h_data:
            return 0.5
        try:
            return analyze_h2h(fixture, h2h_data)
        except ValueError:
            return 0.5

    def _calculate_injury_score(self, injuries: Sequence[InjuryData] | None) -> float:
        """Return a neutral injury score when availability data is missing."""

        if injuries is None:
            return 0.5
        return analyze_injuries(injuries)

    def _calculate_odds_value_score(self, relevant_odds: Sequence[NormalizedOdds]) -> float:
        """Return the conservative odds-value score when no scoreable odds exist."""

        if not relevant_odds:
            return 0.1
        try:
            return analyze_odds_value(relevant_odds)
        except ValueError:
            return 0.1

    def _calculate_statistical_score(
        self,
        matched_team_stats: _ResolvedTeamStats,
    ) -> float:
        """Estimate a low-weight statistical edge from advanced and core stats."""

        home_score = self._team_profile_score(matched_team_stats.home)
        away_score = self._team_profile_score(matched_team_stats.away)

        if matched_team_stats.home is not None and matched_team_stats.away is not None:
            return _clamp(0.5 + ((home_score - away_score) * 0.5))
        if matched_team_stats.home is not None:
            return _clamp(0.5 + ((home_score - 0.5) * 0.5))
        if matched_team_stats.away is not None:
            return _clamp(0.5 + ((0.5 - away_score) * 0.5))
        return 0.5

    def _calculate_data_completeness(
        self,
        *,
        matched_team_stats: _ResolvedTeamStats,
        relevant_odds: Sequence[NormalizedOdds],
        scoreable_odds: Sequence[NormalizedOdds],
        context: MatchContext | None,
        injuries: Sequence[InjuryData] | None,
        h2h_data: Sequence[NormalizedFixture] | None,
    ) -> float:
        """Estimate how complete the supporting data set is for confidence."""

        components = (
            1.0 if len(matched_team_stats.relevant) == 2 else 0.65,
            1.0 if scoreable_odds else 0.30 if relevant_odds else 0.0,
            1.0 if context is not None else 0.0,
            1.0 if injuries is not None else 0.0,
            1.0 if h2h_data is not None else 0.0,
        )
        return _clamp(sum(components) / len(components))

    def _calculate_odds_agreement(
        self,
        recommendation: _CandidateRecommendation | None,
    ) -> float:
        """Estimate provider agreement for the selected recommendation."""

        if recommendation is None:
            return 0.0
        if recommendation.provider_count <= 1:
            return 0.55
        offered_odds = tuple(row.odds for row in recommendation.rows)
        consensus = fmean(offered_odds)
        if consensus <= 0:
            return 0.0
        odds_range = max(offered_odds) - min(offered_odds)
        tightness = _clamp(1.0 - (odds_range / max(consensus * 0.35, 0.01)))
        coverage = _clamp(recommendation.provider_count / 3.0)
        return _clamp((tightness * 0.70) + (coverage * 0.30))

    def _select_best_candidate(
        self,
        *,
        fixture: NormalizedFixture,
        matched_team_stats: _ResolvedTeamStats,
        scoreable_odds: Sequence[NormalizedOdds],
        context: MatchContext | None,
        injuries: Sequence[InjuryData],
        composite_score: float,
    ) -> _CandidateRecommendation | None:
        """Select the best recommendation candidate from grouped odds rows."""

        if not scoreable_odds:
            return None

        insights = self._build_fixture_insights(
            fixture=fixture,
            matched_team_stats=matched_team_stats,
            context=context,
            injuries=injuries,
        )
        grouped_candidates = self._group_candidate_rows(scoreable_odds)
        recommendation_candidates: list[_CandidateRecommendation] = []

        for rows in grouped_candidates.values():
            representative_row = max(rows, key=lambda row: row.odds)
            if representative_row.market is None:
                continue
            fit_score = self._score_market_fit(
                market=representative_row.market,
                selection=representative_row.selection,
                line=representative_row.line,
                insights=insights,
            )
            agreement_score = self._calculate_candidate_agreement(rows)
            provider_count = len({row.provider.casefold() for row in rows})

            candidate = _CandidateRecommendation(
                market=representative_row.market,
                selection=representative_row.selection,
                line=representative_row.line,
                rows=rows,
                representative_row=representative_row,
                recommended_odds=representative_row.odds,
                fit_score=fit_score,
                agreement_score=agreement_score,
                provider_count=provider_count,
            )
            recommendation_candidates.append(candidate)

        if not recommendation_candidates:
            return None

        return max(
            recommendation_candidates,
            key=lambda candidate: self._candidate_score(
                candidate=candidate,
                composite_score=composite_score,
            ),
        )

    def _group_candidate_rows(
        self,
        scoreable_odds: Sequence[NormalizedOdds],
    ) -> dict[_CandidateGroupKey, tuple[NormalizedOdds, ...]]:
        """Group odds rows into selection-level recommendation candidates."""

        grouped: dict[_CandidateGroupKey, list[NormalizedOdds]] = {}
        for row in scoreable_odds:
            market = row.market
            if market is None:
                continue
            group_key = (
                market,
                row.selection,
                row.line,
                row.period,
                row.participant_scope,
            )
            grouped.setdefault(group_key, []).append(row)
        return {key: tuple(rows) for key, rows in grouped.items()}

    def _build_fixture_insights(
        self,
        *,
        fixture: NormalizedFixture,
        matched_team_stats: _ResolvedTeamStats,
        context: MatchContext | None,
        injuries: Sequence[InjuryData],
    ) -> _FixtureInsights:
        """Derive directional and totals-oriented signals for market selection."""

        home_strength = self._team_profile_score(matched_team_stats.home)
        away_strength = self._team_profile_score(matched_team_stats.away)
        injury_edge = self._injury_side_edge(
            injuries=injuries,
            home_team_id=fixture.home_team_id,
            away_team_id=fixture.away_team_id,
            home_team_name=fixture.home_team,
            away_team_name=fixture.away_team,
        )
        context_edge = self._context_side_edge(context)
        raw_side_edge = home_strength - away_strength
        side_edge = _clamp_signed(
            (raw_side_edge * 0.60)
            + (context_edge * 0.20)
            + (injury_edge * 0.20)
        )

        expected_total = self._estimate_expected_total(
            sport=fixture.sport,
            home_stats=matched_team_stats.home,
            away_stats=matched_team_stats.away,
            context=context,
        )
        btts_likelihood = self._estimate_btts_likelihood(
            home_stats=matched_team_stats.home,
            away_stats=matched_team_stats.away,
            context=context,
        )
        draw_likelihood = self._estimate_draw_likelihood(
            sport=fixture.sport,
            side_edge=side_edge,
            expected_total=expected_total,
        )

        return _FixtureInsights(
            side_edge=side_edge,
            home_strength=home_strength,
            away_strength=away_strength,
            expected_total=expected_total,
            btts_likelihood=btts_likelihood,
            draw_likelihood=draw_likelihood,
        )

    def _candidate_score(
        self,
        *,
        candidate: _CandidateRecommendation,
        composite_score: float,
    ) -> float:
        """Return the final ranking score for one recommendation candidate."""

        consensus_odds = fmean(row.odds for row in candidate.rows)
        best_price_edge = _clamp(
            (candidate.recommended_odds - consensus_odds) / max(consensus_odds, 1.0)
        )
        provider_coverage = _clamp(candidate.provider_count / 3.0)
        complexity = self._market_complexity_score(candidate.market)
        return (
            (candidate.fit_score * 0.50)
            + (composite_score * 0.25)
            + (candidate.agreement_score * 0.10)
            + (best_price_edge * 0.05)
            + (provider_coverage * 0.05)
            + (complexity * 0.05)
        )

    def _score_market_fit(
        self,
        *,
        market: MarketType,
        selection: str,
        line: float | None,
        insights: _FixtureInsights,
    ) -> float:
        """Score how well one canonical market selection fits the fixture."""

        normalized_selection = selection.strip()
        if market == MarketType.MATCH_RESULT:
            return self._score_match_result_selection(
                selection=normalized_selection,
                side_edge=insights.side_edge,
                draw_likelihood=insights.draw_likelihood,
            )
        if market == MarketType.MONEYLINE:
            return self._score_side_selection(
                selection=normalized_selection,
                side_edge=insights.side_edge,
                draw_protection=False,
            )
        if market == MarketType.DRAW_NO_BET:
            return self._score_side_selection(
                selection=normalized_selection,
                side_edge=insights.side_edge,
                draw_protection=True,
            )
        if market == MarketType.DOUBLE_CHANCE:
            return self._score_double_chance_selection(
                selection=normalized_selection,
                side_edge=insights.side_edge,
                draw_likelihood=insights.draw_likelihood,
            )
        if market in _SOCCER_TOTAL_MARKETS or market == MarketType.TOTAL_POINTS:
            return self._score_totals_selection(
                selection=normalized_selection,
                line=line,
                expected_total=insights.expected_total,
                sport=(
                    SportName.BASKETBALL
                    if market == MarketType.TOTAL_POINTS
                    else SportName.SOCCER
                ),
            )
        if market == MarketType.BTTS:
            if normalized_selection in _YES_SELECTIONS:
                return insights.btts_likelihood
            if normalized_selection in _NO_SELECTIONS:
                return 1.0 - insights.btts_likelihood
            return 0.0
        if market in {MarketType.ASIAN_HANDICAP, MarketType.POINT_SPREAD}:
            return self._score_spread_selection(
                selection=normalized_selection,
                line=line,
                side_edge=insights.side_edge,
                sport=(
                    SportName.BASKETBALL
                    if market == MarketType.POINT_SPREAD
                    else SportName.SOCCER
                ),
            )
        if market == MarketType.CORRECT_SCORE:
            return _clamp(
                (0.15 + (abs(insights.side_edge) * 0.25))
                * self._market_complexity_score(market)
            )
        if market == MarketType.HT_FT:
            return _clamp(
                (0.18 + (max(insights.home_strength, insights.away_strength) * 0.20))
                * self._market_complexity_score(market)
            )
        return 0.0

    def _score_match_result_selection(
        self,
        *,
        selection: str,
        side_edge: float,
        draw_likelihood: float,
    ) -> float:
        """Score one soccer 1X2 selection against the fixture profile."""

        if selection == "home":
            return _clamp(0.5 + (side_edge * 0.5))
        if selection == "away":
            return _clamp(0.5 - (side_edge * 0.5))
        if selection == "draw":
            return draw_likelihood
        return 0.0

    def _score_side_selection(
        self,
        *,
        selection: str,
        side_edge: float,
        draw_protection: bool,
    ) -> float:
        """Score a side-based market such as moneyline or draw-no-bet."""

        base_score = 0.0
        if selection == "home":
            base_score = _clamp(0.5 + (side_edge * 0.5))
        elif selection == "away":
            base_score = _clamp(0.5 - (side_edge * 0.5))
        if draw_protection:
            return _clamp(0.20 + (base_score * 0.80))
        return base_score

    def _score_double_chance_selection(
        self,
        *,
        selection: str,
        side_edge: float,
        draw_likelihood: float,
    ) -> float:
        """Score a canonical double-chance selection."""

        if selection not in _DOUBLE_CHANCE_SELECTIONS:
            return 0.0
        if selection == "1X":
            return _clamp(0.35 + (max(side_edge, 0.0) * 0.45) + (draw_likelihood * 0.20))
        if selection == "X2":
            return _clamp(0.35 + (max(-side_edge, 0.0) * 0.45) + (draw_likelihood * 0.20))
        return _clamp(0.25 + ((1.0 - draw_likelihood) * 0.75))

    def _score_totals_selection(
        self,
        *,
        selection: str,
        line: float | None,
        expected_total: float,
        sport: SportName,
    ) -> float:
        """Score totals markets against the fixture's expected scoring output."""

        if line is None:
            return 0.0
        baseline_range = 20.0 if sport == SportName.BASKETBALL else 2.2
        normalized_edge = _clamp_signed((expected_total - line) / baseline_range)
        if selection in _OVER_SELECTIONS:
            return _clamp(0.5 + (normalized_edge * 0.5))
        if selection in _UNDER_SELECTIONS:
            return _clamp(0.5 - (normalized_edge * 0.5))
        return 0.0

    def _score_spread_selection(
        self,
        *,
        selection: str,
        line: float | None,
        side_edge: float,
        sport: SportName,
    ) -> float:
        """Score side-and-line markets such as handicaps and point spreads."""

        if line is None:
            return 0.0
        line_scale = 10.0 if sport == SportName.BASKETBALL else 2.0
        expected_margin = side_edge * line_scale
        target_margin = abs(line)

        if selection == "home":
            margin_edge = expected_margin - target_margin
            return _clamp(0.5 + ((margin_edge / line_scale) * 0.5))
        if selection == "away":
            margin_edge = (-expected_margin) - target_margin
            return _clamp(0.5 + ((margin_edge / line_scale) * 0.5))
        return 0.0

    def _calculate_candidate_agreement(self, rows: Sequence[NormalizedOdds]) -> float:
        """Calculate provider agreement for one candidate group."""

        provider_count = len({row.provider.casefold() for row in rows})
        if provider_count <= 1:
            return 0.55
        offered_odds = tuple(row.odds for row in rows)
        consensus = fmean(offered_odds)
        if consensus <= 0:
            return 0.0
        odds_range = max(offered_odds) - min(offered_odds)
        tightness = _clamp(1.0 - (odds_range / max(consensus * 0.35, 0.01)))
        coverage = _clamp(provider_count / 3.0)
        return _clamp((tightness * 0.65) + (coverage * 0.35))

    def _team_profile_score(self, team_stats: TeamStats | None) -> float:
        """Estimate one team's overall strength profile from canonical stats."""

        if team_stats is None:
            return 0.5

        matches_played = max(team_stats.matches_played, 1)
        win_rate = team_stats.wins / matches_played
        attack = self._attack_rate(team_stats)
        defense = self._conceded_rate(team_stats)
        goal_balance = _clamp((attack - defense + 2.0) / 4.0)
        form_score = self._form_score(team_stats)
        advanced_score = self._advanced_metric_score(team_stats)

        return _clamp(
            (win_rate * 0.35)
            + (goal_balance * 0.30)
            + (form_score * 0.20)
            + (advanced_score * 0.15)
        )

    def _form_score(self, team_stats: TeamStats) -> float:
        """Convert one team's compact form or season record into a bounded score."""

        if team_stats.form:
            weighted_points = 0.0
            weighted_max = 0.0
            for index, result in enumerate(team_stats.form[-10:], start=1):
                points = 3 if result == "W" else 1 if result == "D" else 0
                weighted_points += points * index
                weighted_max += 3 * index
            return _clamp(weighted_points / weighted_max)

        matches_played = max(team_stats.matches_played, 1)
        season_points = (team_stats.wins * 3) + team_stats.draws
        return _clamp(season_points / (matches_played * 3))

    def _attack_rate(self, team_stats: TeamStats) -> float:
        """Return goals or points scored per match for one team snapshot."""

        if team_stats.avg_goals_scored is not None:
            return team_stats.avg_goals_scored
        matches_played = max(team_stats.matches_played, 1)
        return team_stats.goals_for / matches_played

    def _conceded_rate(self, team_stats: TeamStats) -> float:
        """Return goals or points conceded per match for one team snapshot."""

        if team_stats.avg_goals_conceded is not None:
            return team_stats.avg_goals_conceded
        matches_played = max(team_stats.matches_played, 1)
        return team_stats.goals_against / matches_played

    def _advanced_metric_score(self, team_stats: TeamStats) -> float:
        """Estimate a bounded advanced-metrics edge from known provider keys."""

        metrics = team_stats.advanced_metrics
        if not metrics:
            return 0.5

        candidate_scores: list[float] = []
        if "xg_diff" in metrics:
            candidate_scores.append(_clamp((metrics["xg_diff"] + 1.5) / 3.0))
        if "net_rating" in metrics:
            candidate_scores.append(_clamp((metrics["net_rating"] + 15.0) / 30.0))
        if "elo" in metrics:
            candidate_scores.append(_clamp((metrics["elo"] - 1300.0) / 400.0))
        if "point_differential" in metrics:
            candidate_scores.append(_clamp((metrics["point_differential"] + 15.0) / 30.0))
        if "offensive_rating" in metrics and "defensive_rating" in metrics:
            net_rating = metrics["offensive_rating"] - metrics["defensive_rating"]
            candidate_scores.append(_clamp((net_rating + 20.0) / 40.0))
        if not candidate_scores:
            return 0.5
        return _clamp(fmean(candidate_scores))

    def _estimate_expected_total(
        self,
        *,
        sport: SportName,
        home_stats: TeamStats | None,
        away_stats: TeamStats | None,
        context: MatchContext | None,
    ) -> float:
        """Estimate expected combined scoring for totals-market selection."""

        if sport == SportName.BASKETBALL:
            baseline = _BASKETBALL_BASELINE_TOTAL
        else:
            baseline = _SOCCER_BASELINE_TOTAL

        if home_stats is None or away_stats is None:
            return baseline

        home_attack = self._attack_rate(home_stats)
        away_attack = self._attack_rate(away_stats)
        home_defense = self._conceded_rate(home_stats)
        away_defense = self._conceded_rate(away_stats)

        expected_home = (home_attack + away_defense) / 2.0
        expected_away = (away_attack + home_defense) / 2.0
        total = expected_home + expected_away

        if context is not None:
            intensity_adjustment = (context.rivalry_factor * 0.08) - (
                ((context.pressure_home + context.pressure_away) / 2.0) * 0.04
            )
            if sport == SportName.BASKETBALL:
                total += intensity_adjustment * 12.0
            else:
                total += intensity_adjustment

        return total

    def _estimate_btts_likelihood(
        self,
        *,
        home_stats: TeamStats | None,
        away_stats: TeamStats | None,
        context: MatchContext | None,
    ) -> float:
        """Estimate soccer both-teams-to-score likelihood from team profiles."""

        if home_stats is None or away_stats is None:
            return 0.5

        home_attack = self._attack_rate(home_stats)
        away_attack = self._attack_rate(away_stats)
        home_conceded = self._conceded_rate(home_stats)
        away_conceded = self._conceded_rate(away_stats)

        scoring_pressure = (
            _clamp(home_attack / 2.0)
            + _clamp(away_attack / 2.0)
            + _clamp(home_conceded / 2.0)
            + _clamp(away_conceded / 2.0)
        ) / 4.0
        if context is not None:
            scoring_pressure = _clamp(
                scoring_pressure
                + (context.rivalry_factor * 0.05)
                + ((context.qualitative_score - 0.5) * 0.08)
            )
        return scoring_pressure

    def _estimate_draw_likelihood(
        self,
        *,
        sport: SportName,
        side_edge: float,
        expected_total: float,
    ) -> float:
        """Estimate draw likelihood for soccer market selection heuristics."""

        if sport == SportName.BASKETBALL:
            return 0.0
        openness_penalty = max(0.0, expected_total - _SOCCER_BASELINE_TOTAL) / 3.0
        return _clamp(0.58 - (abs(side_edge) * 0.40) - (openness_penalty * 0.20))

    def _context_side_edge(self, context: MatchContext | None) -> float:
        """Translate qualitative context into a signed home-versus-away lean."""

        if context is None:
            return 0.0
        return _clamp_signed(
            ((context.morale_home - context.morale_away) * 0.65)
            + ((context.pressure_away - context.pressure_home) * 0.35)
        )

    def _injury_side_edge(
        self,
        *,
        injuries: Sequence[InjuryData],
        home_team_id: str | None,
        away_team_id: str | None,
        home_team_name: str,
        away_team_name: str,
    ) -> float:
        """Translate injury burdens into a signed home-versus-away lean."""

        if not injuries:
            return 0.0

        home_identifiers = {
            value.casefold()
            for value in (home_team_id, home_team_name)
            if value is not None
        }
        away_identifiers = {
            value.casefold()
            for value in (away_team_id, away_team_name)
            if value is not None
        }
        home_burden = 0.0
        away_burden = 0.0

        for injury in injuries:
            burden = 1.75 if injury.is_key_player else 1.0
            if injury.team_id.casefold() in home_identifiers:
                home_burden += burden
            elif injury.team_id.casefold() in away_identifiers:
                away_burden += burden

        if isclose(home_burden + away_burden, 0.0, abs_tol=0.001):
            return 0.0
        return _clamp_signed(
            (away_burden - home_burden) / max(home_burden + away_burden, 1.0)
        )

    def _market_complexity_score(self, market: MarketType) -> float:
        """Favor simpler markets over highly specific ones by default."""

        if market in _LOW_COMPLEXITY_MARKETS:
            return 1.0
        if market in _MEDIUM_COMPLEXITY_MARKETS:
            return 0.75
        if market == MarketType.HT_FT:
            return 0.35
        if market == MarketType.CORRECT_SCORE:
            return 0.15
        return 0.50

    def _build_qualitative_summary(self, context: MatchContext | None) -> str | None:
        """Return the stored qualitative summary for the scored fixture."""

        if context is None:
            return None
        return context.news_summary or context.key_narrative

    def _select_best_team_snapshot(
        self,
        team_stats: Sequence[TeamStats],
        *,
        team_id: str | None,
        team_name: str,
    ) -> TeamStats | None:
        """Select the richest matching team snapshot for one fixture side."""

        matching_rows = tuple(
            stats
            for stats in team_stats
            if self._matches_fixture_team(stats, team_id=team_id, team_name=team_name)
        )
        if not matching_rows:
            return None
        return max(matching_rows, key=self._team_snapshot_quality)

    def _matches_fixture_team(
        self,
        team_stats: TeamStats,
        *,
        team_id: str | None,
        team_name: str,
    ) -> bool:
        """Report whether one team-stat row belongs to the fixture team."""

        if team_id is not None and team_stats.team_id == team_id:
            return True
        return team_stats.team_name.casefold() == team_name.casefold()

    def _team_snapshot_quality(self, team_stats: TeamStats) -> float:
        """Estimate how useful one team snapshot is for scoring."""

        quality = float(team_stats.matches_played)
        if team_stats.form:
            quality += min(len(team_stats.form), 10) * 0.5
        if team_stats.avg_goals_scored is not None:
            quality += 1.0
        if team_stats.avg_goals_conceded is not None:
            quality += 1.0
        if team_stats.advanced_metrics:
            quality += min(len(team_stats.advanced_metrics), 4) * 0.5
        return quality

    def _row_matches_fixture(self, row: NormalizedOdds, fixture: NormalizedFixture) -> bool:
        """Match one normalized odds row to the requested fixture."""

        fixture_ref = fixture.get_fixture_ref()
        if row.fixture_ref == fixture_ref:
            return True
        if fixture.sportradar_id is not None:
            if row.fixture_ref == fixture.sportradar_id:
                return True
            preserved_sportradar_id = row.raw_metadata.get("sportradar_id")
            if (
                isinstance(preserved_sportradar_id, str)
                and preserved_sportradar_id == fixture.sportradar_id
            ):
                return True

        home_team = row.raw_metadata.get("home_team")
        away_team = row.raw_metadata.get("away_team")
        if not isinstance(home_team, str) or not isinstance(away_team, str):
            return False
        return (
            home_team.strip().casefold() == fixture.home_team.casefold()
            and away_team.strip().casefold() == fixture.away_team.casefold()
        )


def _clamp(value: float) -> float:
    """Clamp arbitrary numeric values into PuntLab's factor score range."""

    return max(0.0, min(1.0, value))


def _clamp_signed(value: float) -> float:
    """Clamp signed directional values into a stable `-1.0..1.0` range."""

    return max(-1.0, min(1.0, value))


__all__ = ["ScoringEngine"]
