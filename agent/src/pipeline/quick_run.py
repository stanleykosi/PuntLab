"""Interactive quick-run CLI for PuntLab's pipeline with live progress reporting.

Purpose: provide a single-command local runner that executes pipeline nodes in
order while printing real-time stage progress, timing, and state-change
summaries suitable for manual debugging.
Scope: sequential node execution, per-stage diagnostics, optional delivery
execution, timeout enforcement, and end-of-run reporting.
Dependencies: canonical pipeline nodes, approval router, and `PipelineState`
for typed state transitions. Quick-run delivery is CLI-only by default and does
not depend on database-backed Telegram broadcasting.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import re
import traceback
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from time import perf_counter
from uuid import uuid4

from src.cache.client import RedisClient
from src.config import get_settings
from src.llm import AllProvidersFailedError, get_llm
from src.pipeline.nodes.accumulator_building import accumulator_building_node
from src.pipeline.nodes.approval import approval_node
from src.pipeline.nodes.explanation import explanation_node
from src.pipeline.nodes.ingestion import ingestion_node
from src.pipeline.nodes.market_resolution import market_resolution_node
from src.pipeline.nodes.ranking import ranking_node
from src.pipeline.nodes.research import research_node
from src.pipeline.nodes.scoring import scoring_node
from src.pipeline.router import approval_router
from src.pipeline.state import ApprovalStatus, PipelineStage, PipelineState
from src.providers.orchestrator import ProviderOrchestrator
from src.schemas.accumulators import AccumulatorSlip, AccumulatorStatus, ExplainedAccumulator
from src.schemas.users import DeliveryChannel, DeliveryResult, DeliveryStatus
from src.scrapers.sportybet_fixture_stats import (
    DEFAULT_WIDGET_KEYS,
    SportyBetFixtureStatsScraper,
)

type NodeUpdate = dict[str, object]
type PipelineNode = Callable[[PipelineState], Awaitable[NodeUpdate]]
type PrintFn = Callable[[str], None]
type LiveProgressState = dict[str, object]

_FIXTURE_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:sr:match:\d+|api-football:\d+|balldontlie:\d+)\b",
    flags=re.IGNORECASE,
)
_SPINNER_FRAMES = ("-", "\\", "|", "/")
_DEFAULT_HEARTBEAT_SECONDS = 8.0
_MIN_HEARTBEAT_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class StageNode:
    """One executable pipeline stage definition for quick-run orchestration.

    Inputs:
        `name`: stable stage label displayed in progress output.
        `node`: async callable that accepts `PipelineState` and returns a
            partial state update dictionary.

    Outputs:
        Immutable stage descriptor consumed by `run_quick_pipeline`.
    """

    name: str
    node: PipelineNode
    llm_tasks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StageExecutionRecord:
    """Execution result details for one quick-run stage.

    Inputs:
        Runtime metadata captured before and after one stage call.

    Outputs:
        Immutable stage report row used by final summary rendering.
    """

    name: str
    status: str
    duration_seconds: float
    before_counts: dict[str, int]
    after_counts: dict[str, int]
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class QuickRunReport:
    """Top-level quick-run outcome bundle.

    Inputs:
        Final state snapshot and ordered stage execution records.

    Outputs:
        One immutable report object suitable for CLI exit decisions and tests.
    """

    state: PipelineState
    stage_records: tuple[StageExecutionRecord, ...]
    total_duration_seconds: float
    success: bool


class _QuickRunInMemoryRedis:
    """Small Redis-compatible cache used by local quick-run executions."""

    def __init__(self) -> None:
        """Initialize in-memory value and expiration stores."""

        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def get(self, name: str) -> str | None:
        """Return a stored string value when present."""

        return self.values.get(name)

    async def set(self, name: str, value: str, ex: int | None = None) -> bool:
        """Persist a string value with an optional TTL."""

        self.values[name] = value
        if ex is not None:
            self.expirations[name] = ex
        return True

    async def incr(self, name: str, amount: int = 1) -> int:
        """Increment a numeric value and return the new count."""

        next_value = int(self.values.get(name, "0")) + amount
        self.values[name] = str(next_value)
        return next_value

    async def expire(self, name: str, time: int) -> bool:
        """Attach a TTL to an existing key."""

        if name not in self.values:
            return False
        self.expirations[name] = time
        return True

    async def ping(self) -> bool:
        """Report the local cache as available."""

        return True

    async def aclose(self) -> None:
        """Match the async Redis protocol."""


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for local pipeline quick-run execution."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the pipeline with live progress and an end-of-run report. "
            "Delivery is skipped by default for faster local debugging."
        )
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Optional run date in ISO format (YYYY-MM-DD). Defaults to today in WAT.",
    )
    parser.add_argument(
        "--with-delivery",
        action="store_true",
        help=(
            "Execute quick-run delivery after approval if routing resolves to delivery. "
            "Quick-run delivery publishes slips to this CLI (no Telegram/DB dependency)."
        ),
    )
    parser.add_argument(
        "--approval-wait-seconds",
        type=int,
        default=0,
        help="Approval wait duration passed to approval_node (default: 0).",
    )
    parser.add_argument(
        "--strict-delivery",
        action="store_true",
        help=(
            "Deprecated alias. Delivery failures are treated as fatal by default "
            "when --with-delivery is enabled."
        ),
    )
    parser.add_argument(
        "--allow-delivery-failure",
        action="store_true",
        help=(
            "Allow quick-run to continue when delivery fails. "
            "Use only for analysis debugging."
        ),
    )
    parser.add_argument(
        "--node-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Per-stage timeout in seconds. Set to 0 to disable stage timeouts "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--max-error-lines",
        type=int,
        default=20,
        help="Maximum number of error messages printed in the final report.",
    )
    parser.add_argument(
        "--show-traceback",
        action="store_true",
        help="Print full traceback details when a stage fails.",
    )
    parser.add_argument(
        "--no-llm-trace",
        action="store_true",
        help="Disable per-node LLM provider/model trace lines.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=_DEFAULT_HEARTBEAT_SECONDS,
        help=(
            "Live progress heartbeat interval in seconds while a node is running "
            f"(default: {_DEFAULT_HEARTBEAT_SECONDS:g}). Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--no-live-dashboard",
        action="store_true",
        help="Disable in-node live dashboard heartbeats.",
    )
    return parser


async def _invoke_approval_node(
    state: PipelineState,
    *,
    wait_seconds: int,
) -> NodeUpdate:
    """Wrap `approval_node` so it conforms to the generic stage callable shape."""

    return await approval_node(state, wait_seconds=wait_seconds)


def _apply_update(state: PipelineState, update: NodeUpdate) -> PipelineState:
    """Apply one node update onto state and return a validated merged snapshot."""

    merged = {**state.model_dump(mode="python"), **update}
    return PipelineState.model_validate(merged)


def _state_counts(state: PipelineState) -> dict[str, int]:
    """Return key pipeline counters used for stage delta reporting."""

    return {
        "fixtures": len(state.fixtures),
        "odds_rows": len(state.odds_data),
        "catalog_rows": len(state.odds_market_catalog.all_rows()),
        "team_stats": len(state.team_stats),
        "player_stats": len(state.player_stats),
        "fixture_details": len(state.fixture_details),
        "injuries": len(state.injuries),
        "news": len(state.news_articles),
        "contexts": len(state.match_contexts),
        "scores": len(state.match_scores),
        "ranked": len(state.ranked_matches),
        "resolved": len(state.resolved_markets),
        "accumulators": len(state.accumulators),
        "explained": len(state.explained_accumulators),
        "delivery": len(state.delivery_results),
        "errors": len(state.errors),
    }


def _render_count_delta(before: dict[str, int], after: dict[str, int]) -> str:
    """Render a compact counter delta string for one stage transition."""

    highlight_keys = (
        "fixtures",
        "catalog_rows",
        "odds_rows",
        "contexts",
        "scores",
        "ranked",
        "resolved",
        "accumulators",
        "explained",
        "delivery",
        "errors",
    )
    fragments: list[str] = []
    for key in highlight_keys:
        delta = after[key] - before[key]
        if delta == 0 and key not in {"errors"}:
            continue
        fragments.append(f"{key}={after[key]} ({delta:+d})")
    return " | ".join(fragments) if fragments else "no counter changes"


def _render_progress_bar(completed: int, total: int, *, width: int = 18) -> str:
    """Render a compact ASCII progress bar for live stage dashboards."""

    if total <= 0:
        return f"[{'-' * width}] --%"
    ratio = max(0.0, min(1.0, completed / total))
    filled = int(ratio * width)
    if filled >= width:
        body = "=" * width
    else:
        body = ("=" * filled) + ">" + ("-" * max(0, width - filled - 1))
    return f"[{body}] {ratio * 100:5.1f}%"


def _render_elapsed_label(seconds: float) -> str:
    """Format elapsed seconds into a short wall-clock label."""

    whole_seconds = max(0, int(seconds))
    minutes, sec = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _record_explanation_progress(
    state: LiveProgressState,
    payload: Mapping[str, object],
) -> None:
    """Update the mutable live-state snapshot from explanation callbacks."""

    state.update(payload)
    event = str(payload.get("event", ""))
    if event == "stage_started":
        state["completed_slips"] = 0
        return
    if event == "slip_started":
        state["slip_total_legs"] = int(payload.get("leg_count", 0))
        return
    if event == "slip_completed":
        completed_slips = int(state.get("completed_slips", 0)) + 1
        state["completed_slips"] = completed_slips


def _render_stage_dashboard(
    *,
    stage_name: str,
    elapsed_seconds: float,
    live_state: LiveProgressState,
) -> str:
    """Render a stage-specific live dashboard string for heartbeat output."""

    elapsed_label = _render_elapsed_label(elapsed_seconds)
    if stage_name != "explanation":
        return f"elapsed={elapsed_label} | activity={_default_stage_activity(stage_name)}"

    total_slips = int(live_state.get("total_slips", 0))
    completed_slips = int(live_state.get("completed_slips", 0))
    current_slip = live_state.get("slip_number") or "?"
    event = str(live_state.get("event", "starting"))

    slip_bar = _render_progress_bar(completed_slips, total_slips)
    return (
        f"elapsed={elapsed_label} | phase={event} | slip={current_slip}/{total_slips or '?'} "
        f"| slips {slip_bar}"
    )


def _default_stage_activity(stage_name: str) -> str:
    """Return short activity hints for non-explanation live heartbeats."""

    hints = {
        "ingestion": "fetching fixtures, odds, stats, injuries, and news",
        "research": "building fixture contexts and news-backed narratives",
        "scoring": "computing composite scores for each fixture",
        "ranking": "sorting fixtures by score and confidence",
        "market_resolution": "mapping ranked picks to sportsbook markets",
        "accumulator_building": "constructing slip combinations",
        "approval": "waiting for approval status updates",
        "delivery": "publishing approved slips to the CLI",
    }
    return hints.get(stage_name, "processing stage workload")


async def _quick_run_cli_delivery_node(state: PipelineState) -> NodeUpdate:
    """Publish approved slips to quick-run output without external dependencies.

    Inputs:
        state: Current quick-run pipeline state.

    Outputs:
        A partial update that marks approved slips as published and records
        one successful CLI delivery result per approved slip.

    Raises:
        ValueError: If approval is not approved or if no approved explained
            slips are available for publication.
    """

    if state.approval_status is not ApprovalStatus.APPROVED:
        raise ValueError(
            "quick-run CLI delivery requires approval_status='approved' before publication."
        )

    approved_explained = tuple(
        slip for slip in state.explained_accumulators if slip.status is AccumulatorStatus.APPROVED
    )
    if not approved_explained:
        raise ValueError(
            "quick-run CLI delivery requires at least one approved explained accumulator."
        )

    published_at = datetime.now(UTC)
    published_explained = _mark_published_explained_quick_run(
        state.explained_accumulators,
        published_at=published_at,
    )
    published_accumulators = _mark_published_accumulators_quick_run(
        state.accumulators,
        published_at=published_at,
    )

    delivery_results = [
        DeliveryResult(
            accumulator_id=slip.slip_id,
            user_id=None,
            channel=DeliveryChannel.API,
            status=DeliveryStatus.SENT,
            subscription_tier=None,
            recipient="cli",
            error_message=None,
            delivered_at=published_at,
        )
        for slip in approved_explained
    ]

    return {
        "current_stage": PipelineStage.DELIVERY,
        "accumulators": published_accumulators,
        "explained_accumulators": published_explained,
        "delivery_results": delivery_results,
    }


def _mark_published_explained_quick_run(
    explained_accumulators: Sequence[ExplainedAccumulator],
    *,
    published_at: datetime,
) -> list[ExplainedAccumulator]:
    """Mark approved explained slips as published for CLI quick-run delivery."""

    return [
        (
            slip.model_copy(
                update={
                    "is_published": True,
                    "published_at": published_at,
                }
            )
            if slip.status is AccumulatorStatus.APPROVED
            else slip
        )
        for slip in explained_accumulators
    ]


def _mark_published_accumulators_quick_run(
    accumulators: Sequence[AccumulatorSlip],
    *,
    published_at: datetime,
) -> list[AccumulatorSlip]:
    """Mark approved base slips as published for CLI quick-run delivery."""

    return [
        (
            slip.model_copy(
                update={
                    "is_published": True,
                    "published_at": published_at,
                }
            )
            if slip.status is AccumulatorStatus.APPROVED
            else slip
        )
        for slip in accumulators
    ]


def _render_slip_header(slip: AccumulatorSlip) -> str:
    """Render one concise slip header line for final CLI summary output."""

    published_label = "yes" if slip.is_published else "no"
    return (
        f"#{slip.slip_number} | odds={slip.total_odds:.2f} | legs={slip.leg_count} "
        f"| confidence={slip.confidence:.2f} | status={slip.status.value} "
        f"| published={published_label}"
    )


def _print_slip_details(
    slips: Sequence[AccumulatorSlip],
    *,
    print_fn: PrintFn,
    max_slips: int = 15,
) -> None:
    """Print deterministic terminal-friendly slip details for quick-run users.

    Inputs:
        slips: Ordered slips to print.
        print_fn: Line-emitter function used by quick-run reporting.
        max_slips: Upper bound for lines to keep logs readable.

    Outputs:
        None. Emits one summary line per slip and one line per leg.
    """

    if not slips:
        return

    print_fn(f"  slips (showing up to {max_slips}):")
    for slip in slips[:max_slips]:
        print_fn(f"    - {_render_slip_header(slip)}")
        if isinstance(slip, ExplainedAccumulator):
            print_fn(f"      rationale: {slip.rationale}")
        for leg in slip.legs:
            market_label = leg.market_label or (
                leg.canonical_market.value
                if leg.canonical_market is not None
                else leg.market
            )
            print_fn(
                "      "
                f"leg {leg.leg_number}: {leg.home_team} vs {leg.away_team} | "
                f"market={market_label} | "
                f"selection={leg.selection} | "
                f"odds={leg.odds:.2f} | provider={leg.provider}"
            )
            if leg.rationale:
                print_fn(f"      edge: {leg.rationale}")


def _render_explanation_event_line(payload: Mapping[str, object]) -> str | None:
    """Render one concise explanation progress event line for the CLI."""

    event = str(payload.get("event", "")).strip().casefold()
    slip_number = payload.get("slip_number")
    duration_seconds = payload.get("duration_seconds")
    had_error = bool(payload.get("had_error", False))

    if event == "stage_started":
        return (
            "explanation stage started "
            f"(total_slips={int(payload.get('total_slips', 0))})"
        )
    if event == "slip_started":
        return (
            f"slip #{slip_number} started "
            f"(legs={int(payload.get('leg_count', 0))})"
        )
    if event == "slip_rationale_started":
        return f"slip #{slip_number} accumulator rationale started"
    if event == "slip_completed":
        duration_label = (
            f"{float(duration_seconds):.2f}s"
            if isinstance(duration_seconds, (int, float))
            else "-"
        )
        status = "fallback" if had_error else "ok"
        return (
            f"slip #{slip_number} completed "
            f"(status={status}, duration={duration_label})"
        )
    if event == "stage_completed":
        return "explanation stage completed"
    return None


def _parse_run_date(raw_value: str | None) -> date:
    """Parse an optional run-date argument or default to today's WAT date."""

    settings = get_settings()
    if raw_value is None:
        return datetime.now(settings.timezone).date()
    try:
        return date.fromisoformat(raw_value.strip())
    except ValueError as exc:
        raise ValueError("--run-date must be a valid ISO date (YYYY-MM-DD).") from exc


def _build_default_pre_approval_nodes() -> tuple[StageNode, ...]:
    """Return the canonical pre-approval stage order."""

    return (
        StageNode(name="ingestion", node=_quick_run_ingestion_node),
        StageNode(name="research", node=research_node, llm_tasks=("research",)),
        StageNode(name="scoring", node=scoring_node, llm_tasks=("market_scoring",)),
        StageNode(name="ranking", node=ranking_node, llm_tasks=("ranking",)),
        StageNode(
            name="market_resolution",
            node=market_resolution_node,
            llm_tasks=("market_resolution",),
        ),
        StageNode(
            name="accumulator_building",
            node=accumulator_building_node,
            llm_tasks=("accumulator_builder",),
        ),
        StageNode(
            name="explanation",
            node=explanation_node,
            llm_tasks=("accumulator_rationale",),
        ),
    )


async def _quick_run_ingestion_node(state: PipelineState) -> NodeUpdate:
    """Run ingestion with local cache infrastructure for CLI quick-run tests."""

    cache = RedisClient(redis_client=_QuickRunInMemoryRedis())
    fixture_stats_scraper = SportyBetFixtureStatsScraper(
        navigation_timeout_ms=30_000,
        post_mount_wait_ms=2_000,
        response_timeout_ms=30_000,
        response_body_timeout_ms=10_000,
        settle_delay_ms=500,
        widget_timeout_ms=8_000,
        widget_keys=DEFAULT_WIDGET_KEYS,
    )
    orchestrator = ProviderOrchestrator(
        cache=cache,
        sportybet_fixture_stats_scraper=fixture_stats_scraper,
        sportybet_fixture_detail_retries=1,
        sportybet_fixture_detail_retry_backoff_seconds=1.0,
        sportybet_fixture_detail_concurrency=2,
        sportybet_fixture_detail_limit=None,
    )
    try:
        return await ingestion_node(state, orchestrator=orchestrator)
    finally:
        await orchestrator.aclose()


def _describe_llm_instance(llm_instance: object) -> tuple[str, str, str]:
    """Return provider/model/endpoint details for one instantiated LLM.

    Inputs:
        llm_instance: Constructed LangChain model object returned by `get_llm`.

    Outputs:
        A tuple of `(provider_label, model_name, endpoint)` suitable for quick
        progress logging.
    """

    model_name = str(
        getattr(llm_instance, "model_name", None) or getattr(llm_instance, "model", "unknown")
    )
    endpoint = str(
        getattr(llm_instance, "openai_api_base", None)
        or getattr(llm_instance, "base_url", "")
        or ""
    )
    class_name = type(llm_instance).__name__
    endpoint_lower = endpoint.lower()
    if "openrouter" in endpoint_lower:
        provider_label = "openrouter"
    elif class_name == "ChatAnthropic":
        provider_label = "anthropic"
    elif class_name == "ChatOpenAI":
        provider_label = "openai"
    else:
        provider_label = class_name
    return provider_label, model_name, endpoint or "-"


async def _emit_llm_trace_for_stage(
    *,
    stage_node: StageNode,
    stage_label: str,
    print_fn: PrintFn,
) -> None:
    """Print the concrete LLM selection for stage tasks that use the LLM.

    Inputs:
        stage_node: Stage metadata including optional LLM task keys.
        stage_label: Formatted stage ordinal label such as `2/8`.
        print_fn: Output function for progress lines.

    Outputs:
        None. This helper emits trace lines and never raises to keep the stage
        runner resilient even when provider selection fails.
    """

    for task_key in stage_node.llm_tasks:
        try:
            llm_instance = await get_llm(task_key)
            provider, model_name, endpoint = _describe_llm_instance(llm_instance)
            print_fn(
                f"[{stage_label}] llm task={task_key} provider={provider} "
                f"model={model_name} endpoint={endpoint}"
            )
        except AllProvidersFailedError as exc:
            print_fn(f"[{stage_label}] llm task={task_key} unavailable: {exc}")
        except Exception as exc:  # pragma: no cover - defensive logging guard
            print_fn(f"[{stage_label}] llm task={task_key} probe failed: {exc}")


async def run_quick_pipeline(
    *,
    run_date: date,
    with_delivery: bool,
    approval_wait_seconds: int,
    ignore_delivery_errors: bool,
    node_timeout_seconds: float,
    show_traceback: bool,
    max_error_lines: int,
    llm_trace_enabled: bool = True,
    heartbeat_seconds: float = _DEFAULT_HEARTBEAT_SECONDS,
    live_dashboard_enabled: bool = True,
    print_fn: PrintFn = print,
    pre_approval_nodes: Sequence[StageNode] | None = None,
    approval_stage: StageNode | None = None,
    delivery_stage: StageNode | None = None,
    approval_route_resolver: Callable[[PipelineState], str] = approval_router,
) -> QuickRunReport:
    """Execute the pipeline sequentially with live progress and summary output.

    Inputs:
        run_date: Date the run should analyze.
        with_delivery: Whether to execute the delivery stage.
        approval_wait_seconds: Wait duration forwarded to approval stage.
        ignore_delivery_errors: If `True`, delivery failures are captured and
            reported without aborting the quick run.
        node_timeout_seconds: Per-stage timeout threshold.
        show_traceback: Whether stage failures should print full tracebacks.
        max_error_lines: Maximum number of errors shown in final summary.
        llm_trace_enabled: Whether quick-run should print per-node LLM
            provider/model details for tasks that invoke `get_llm`.
        heartbeat_seconds: Interval for live heartbeat lines while a stage is
            in flight. Set to `0` to disable heartbeat output.
        live_dashboard_enabled: Whether stage heartbeats should render live
            dashboards while nodes are executing.
        print_fn: Output function used for live progress lines.
        pre_approval_nodes: Optional stage override for tests.
        approval_stage: Optional approval-stage override for tests.
        delivery_stage: Optional delivery-stage override for tests.
        approval_route_resolver: Optional router override for tests.

    Outputs:
        `QuickRunReport` containing final state, per-stage timing, and success.
    """

    if approval_wait_seconds < 0:
        raise ValueError("approval_wait_seconds must be zero or positive.")
    if node_timeout_seconds < 0:
        raise ValueError("node_timeout_seconds must be zero or positive.")
    if max_error_lines <= 0:
        raise ValueError("max_error_lines must be a positive integer.")
    if heartbeat_seconds < 0:
        raise ValueError("heartbeat_seconds must be zero or positive.")
    if 0 < heartbeat_seconds < _MIN_HEARTBEAT_SECONDS:
        raise ValueError(
            f"heartbeat_seconds must be 0 or at least {_MIN_HEARTBEAT_SECONDS:.1f} seconds."
        )

    state = PipelineState(
        run_id=f"quick-{uuid4()}",
        run_date=run_date,
        started_at=datetime.now(UTC),
    )

    started_at = perf_counter()
    stage_records: list[StageExecutionRecord] = []

    heartbeat_label = (
        "off"
        if heartbeat_seconds == 0 or not live_dashboard_enabled
        else f"{heartbeat_seconds:.1f}s"
    )
    print_fn(
        f"[quick-run] run_id={state.run_id} run_date={state.run_date.isoformat()} "
        f"delivery={'on' if with_delivery else 'off'} "
        f"timeout={'off' if node_timeout_seconds == 0 else f'{node_timeout_seconds:.0f}s'} "
        f"heartbeat={heartbeat_label}"
    )

    async def execute_stage(stage_node: StageNode, *, ordinal: int, total: int) -> bool:
        """Execute one stage with timeout, progress output, and failure capture."""

        nonlocal state

        before_counts = _state_counts(state)
        print_fn(f"[{ordinal}/{total}] start {stage_node.name}")
        if llm_trace_enabled and stage_node.llm_tasks:
            await _emit_llm_trace_for_stage(
                stage_node=stage_node,
                stage_label=f"{ordinal}/{total}",
                print_fn=print_fn,
            )
        stage_started = perf_counter()
        live_state: LiveProgressState = {}
        if stage_node.name == "explanation":
            live_state.update(
                {
                    "total_slips": len(state.accumulators),
                    "completed_slips": 0,
                    "event": "queued",
                }
            )

        async def invoke_stage_node() -> NodeUpdate:
            """Invoke the stage node with optional live-progress wiring."""

            if stage_node.node is explanation_node:
                def on_explanation_progress(payload: Mapping[str, object]) -> None:
                    """Capture and print explanation-stage progress updates."""

                    _record_explanation_progress(live_state, payload)
                    event_line = _render_explanation_event_line(payload)
                    if event_line is None:
                        return
                    print_fn(f"[{ordinal}/{total}] explain {event_line}")

                return await explanation_node(
                    state,
                    progress_callback=on_explanation_progress,
                )
            return await stage_node.node(state)

        try:
            stage_task = asyncio.create_task(invoke_stage_node())
            spinner_index = 0

            while not stage_task.done():
                elapsed = perf_counter() - stage_started
                if node_timeout_seconds != 0 and elapsed >= node_timeout_seconds:
                    stage_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await stage_task
                    raise TimeoutError

                if not live_dashboard_enabled or heartbeat_seconds == 0:
                    poll_window = 0.25
                    if node_timeout_seconds != 0:
                        poll_window = min(
                            poll_window,
                            max(0.01, node_timeout_seconds - elapsed),
                        )
                    try:
                        await asyncio.wait_for(asyncio.shield(stage_task), timeout=poll_window)
                    except TimeoutError:
                        continue
                    continue

                heartbeat_window = heartbeat_seconds
                if node_timeout_seconds != 0:
                    heartbeat_window = min(
                        heartbeat_window,
                        max(0.01, node_timeout_seconds - elapsed),
                    )

                try:
                    await asyncio.wait_for(asyncio.shield(stage_task), timeout=heartbeat_window)
                except TimeoutError:
                    if stage_task.done():
                        break
                    spinner_frame = _SPINNER_FRAMES[spinner_index % len(_SPINNER_FRAMES)]
                    spinner_index += 1
                    dashboard = _render_stage_dashboard(
                        stage_name=stage_node.name,
                        elapsed_seconds=perf_counter() - stage_started,
                        live_state=live_state,
                    )
                    print_fn(
                        f"[{ordinal}/{total}] live {stage_node.name} {spinner_frame} {dashboard}"
                    )
                    continue

            update = await stage_task
            state = _apply_update(state, update)
            duration = perf_counter() - stage_started
            after_counts = _state_counts(state)
            stage_records.append(
                StageExecutionRecord(
                    name=stage_node.name,
                    status="ok",
                    duration_seconds=duration,
                    before_counts=before_counts,
                    after_counts=after_counts,
                )
            )
            print_fn(
                f"[{ordinal}/{total}] done {stage_node.name} in {duration:.2f}s | "
                f"{_render_count_delta(before_counts, after_counts)}"
            )
            return True
        except TimeoutError as exc:
            duration = perf_counter() - stage_started
            error_message = (
                f"Stage '{stage_node.name}' timed out after {node_timeout_seconds:.0f}s."
            )
            state = _apply_update(state, {"errors": [*state.errors, error_message]})
            after_counts = _state_counts(state)
            stage_records.append(
                StageExecutionRecord(
                    name=stage_node.name,
                    status="failed",
                    duration_seconds=duration,
                    before_counts=before_counts,
                    after_counts=after_counts,
                    error_message=error_message,
                )
            )
            print_fn(f"[{ordinal}/{total}] failed {stage_node.name}: {error_message}")
            if show_traceback:
                print_fn("".join(traceback.format_exception(exc)))
            return False
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            duration = perf_counter() - stage_started
            error_message = f"Stage '{stage_node.name}' failed: {exc}"
            state = _apply_update(state, {"errors": [*state.errors, error_message]})
            after_counts = _state_counts(state)
            stage_records.append(
                StageExecutionRecord(
                    name=stage_node.name,
                    status="failed",
                    duration_seconds=duration,
                    before_counts=before_counts,
                    after_counts=after_counts,
                    error_message=error_message,
                )
            )
            print_fn(f"[{ordinal}/{total}] failed {stage_node.name}: {exc}")
            if show_traceback:
                print_fn(traceback.format_exc())
            return False

    pre_nodes = (
        tuple(pre_approval_nodes)
        if pre_approval_nodes is not None
        else _build_default_pre_approval_nodes()
    )
    approval_stage_node = approval_stage or StageNode(
        name="approval",
        node=lambda stage_state: _invoke_approval_node(
            stage_state,
            wait_seconds=approval_wait_seconds,
        ),
    )
    delivery_stage_node = delivery_stage or StageNode(
        name="delivery",
        node=_quick_run_cli_delivery_node,
    )

    planned_total = len(pre_nodes) + 1 + (1 if with_delivery else 0)
    stage_index = 0

    for stage in pre_nodes:
        stage_index += 1
        completed = await execute_stage(stage, ordinal=stage_index, total=planned_total)
        if not completed:
            total_duration = perf_counter() - started_at
            _print_final_report(
                report=QuickRunReport(
                    state=state,
                    stage_records=tuple(stage_records),
                    total_duration_seconds=total_duration,
                    success=False,
                ),
                max_error_lines=max_error_lines,
                print_fn=print_fn,
            )
            return QuickRunReport(
                state=state,
                stage_records=tuple(stage_records),
                total_duration_seconds=total_duration,
                success=False,
            )

    stage_index += 1
    approval_completed = await execute_stage(
        approval_stage_node,
        ordinal=stage_index,
        total=planned_total,
    )
    if not approval_completed:
        total_duration = perf_counter() - started_at
        _print_final_report(
            report=QuickRunReport(
                state=state,
                stage_records=tuple(stage_records),
                total_duration_seconds=total_duration,
                success=False,
            ),
            max_error_lines=max_error_lines,
            print_fn=print_fn,
        )
        return QuickRunReport(
            state=state,
            stage_records=tuple(stage_records),
            total_duration_seconds=total_duration,
            success=False,
        )

    if with_delivery:
        route = approval_route_resolver(state)
        if route == "delivery":
            stage_index += 1
            delivery_completed = await execute_stage(
                delivery_stage_node,
                ordinal=stage_index,
                total=planned_total,
            )
            if not delivery_completed and ignore_delivery_errors:
                last = stage_records[-1]
                stage_records[-1] = StageExecutionRecord(
                    name=last.name,
                    status="failed_ignored",
                    duration_seconds=last.duration_seconds,
                    before_counts=last.before_counts,
                    after_counts=last.after_counts,
                    error_message=last.error_message,
                )
                print_fn(
                    "[quick-run] delivery failure was ignored by configuration; "
                    "analysis summary remains available."
                )
            elif not delivery_completed:
                total_duration = perf_counter() - started_at
                report = QuickRunReport(
                    state=state,
                    stage_records=tuple(stage_records),
                    total_duration_seconds=total_duration,
                    success=False,
                )
                _print_final_report(
                    report=report,
                    max_error_lines=max_error_lines,
                    print_fn=print_fn,
                )
                return report
        else:
            print_fn(
                f"[quick-run] delivery skipped because approval route resolved to '{route}'."
            )
    else:
        print_fn("[quick-run] delivery skipped (enable with --with-delivery).")

    total_duration = perf_counter() - started_at
    success = all(record.status in {"ok", "failed_ignored"} for record in stage_records)
    report = QuickRunReport(
        state=state,
        stage_records=tuple(stage_records),
        total_duration_seconds=total_duration,
        success=success,
    )
    _print_final_report(report=report, max_error_lines=max_error_lines, print_fn=print_fn)
    return report


def _print_final_report(
    *,
    report: QuickRunReport,
    max_error_lines: int,
    print_fn: PrintFn,
) -> None:
    """Render a comprehensive end-of-run summary for local debugging."""

    state = report.state
    print_fn("[quick-run] final report")
    print_fn(f"  success: {report.success}")
    print_fn(f"  duration_seconds: {report.total_duration_seconds:.2f}")
    print_fn(f"  stage_count: {len(report.stage_records)}")

    for record in report.stage_records:
        stage_delta = _render_count_delta(record.before_counts, record.after_counts)
        suffix = f" | error={record.error_message}" if record.error_message else ""
        print_fn(
            f"  - {record.name}: {record.status} in {record.duration_seconds:.2f}s"
            f" | {stage_delta}{suffix}"
        )

    counts = _state_counts(state)
    print_fn("  state_counts:")
    print_fn(f"    fixtures={counts['fixtures']}")
    print_fn(f"    catalog_rows={counts['catalog_rows']}")
    print_fn(f"    odds_rows={counts['odds_rows']}")
    print_fn(f"    contexts={counts['contexts']}")
    print_fn(f"    scores={counts['scores']}")
    print_fn(f"    ranked={counts['ranked']}")
    print_fn(f"    resolved={counts['resolved']}")
    print_fn(f"    accumulators={counts['accumulators']}")
    print_fn(f"    explained={counts['explained']}")
    print_fn(f"    delivery={counts['delivery']}")
    print_fn(f"    errors={counts['errors']}")

    if state.explained_accumulators:
        top_slip = state.explained_accumulators[0]
        print_fn(
            "  top_slip: "
            f"#{top_slip.slip_number} | odds={top_slip.total_odds:.2f} "
            f"| legs={top_slip.leg_count}"
        )
        _print_slip_details(state.explained_accumulators, print_fn=print_fn)
    elif state.accumulators:
        _print_slip_details(state.accumulators, print_fn=print_fn)

    if state.errors:
        print_fn(f"  recent_errors (showing up to {max_error_lines}):")
        for error_message in state.errors[:max_error_lines]:
            print_fn(f"    - {error_message}")
        _print_error_breakdown(state.errors, print_fn=print_fn)


def _print_error_breakdown(errors: Sequence[str], *, print_fn: PrintFn) -> None:
    """Print grouped root-cause counts to make noisy runs quickly diagnosable.

    Inputs:
        errors: Ordered pipeline diagnostic messages accumulated during the run.
        print_fn: Output function used by quick-run reporting.

    Outputs:
        None. Emits an operator-facing error summary sorted by frequency.
    """

    if not errors:
        return

    grouped = Counter(_normalize_error_summary_key(message) for message in errors)
    print_fn("  error_summary:")
    for summary, count in grouped.most_common(10):
        print_fn(f"    - count={count} cause={summary}")


def _normalize_error_summary_key(message: str) -> str:
    """Normalize one diagnostic message into a compact summary key.

    Inputs:
        message: Raw error diagnostic from `PipelineState.errors`.

    Outputs:
        A stable root-cause key suitable for frequency counting.
    """

    normalized = " ".join(message.split())
    normalized_lower = normalized.casefold()

    if normalized_lower.startswith("research llm analysis failed for "):
        return "research: llm analysis failed"
    if normalized_lower.startswith("tavily research fetch failed for "):
        return "research: tavily fetch failed"
    if normalized_lower.startswith("scoring failed for "):
        return "scoring: fixture scoring failed"
    if normalized_lower.startswith("market resolution failed for "):
        return "market_resolution: fixture resolution failed"
    if normalized_lower.startswith("market resolution skipped for "):
        return "market_resolution: fixture skipped"
    if normalized_lower.startswith("balldontlie stats fetch failed"):
        return "ingestion: balldontlie stats fetch failed"
    if normalized_lower.startswith("odds coverage is incomplete for fixtures:"):
        return "ingestion: fixtures missing odds coverage"

    return _FIXTURE_IDENTIFIER_PATTERN.sub("<fixture>", normalized)


async def _async_main(args: argparse.Namespace) -> int:
    """Run the quick-run CLI and return a process exit code."""

    # Delivery is strict by default so delivery failures are never silently
    # reported as success in production-like quick runs.
    ignore_delivery_errors = bool(args.allow_delivery_failure)

    report = await run_quick_pipeline(
        run_date=_parse_run_date(args.run_date),
        with_delivery=args.with_delivery,
        approval_wait_seconds=args.approval_wait_seconds,
        ignore_delivery_errors=ignore_delivery_errors,
        node_timeout_seconds=args.node_timeout_seconds,
        show_traceback=args.show_traceback,
        max_error_lines=args.max_error_lines,
        llm_trace_enabled=not args.no_llm_trace,
        heartbeat_seconds=args.heartbeat_seconds,
        live_dashboard_enabled=not args.no_live_dashboard,
    )
    return 0 if report.success else 1


def run() -> None:
    """CLI entry point for quick-run execution with progress reporting."""

    parser = _build_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_async_main(args)))


__all__ = [
    "QuickRunReport",
    "StageExecutionRecord",
    "StageNode",
    "run",
    "run_quick_pipeline",
]
