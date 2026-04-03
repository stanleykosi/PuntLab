"""Bootstrap entry point for the PuntLab agent runtime.

Purpose: provides a real process entry point that validates configuration,
configures logging, and keeps the service alive until the full scheduler and
pipeline orchestration layers are implemented in later steps.
Scope: process startup, startup diagnostics, graceful shutdown handling.
Dependencies: `src.config.Settings` for runtime configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from collections.abc import Sequence

from src.config import Settings, get_settings

LOGGER = logging.getLogger("puntlab.agent")


def configure_logging(log_level: str) -> None:
    """Configure root logging for the agent process.

    Args:
        log_level: The desired logging level from runtime configuration.
    """

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the agent bootstrap process.

    Returns:
        A configured argument parser for startup control flags.
    """

    parser = argparse.ArgumentParser(description="Run the PuntLab agent runtime.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and imports, then exit successfully.",
    )
    return parser


def install_signal_handlers(shutdown_event: asyncio.Event) -> None:
    """Install POSIX signal handlers to support graceful shutdown.

    Args:
        shutdown_event: Event set when the process should stop its idle loop.
    """

    loop = asyncio.get_running_loop()

    for signame in ("SIGINT", "SIGTERM"):
        if not hasattr(signal, signame):
            continue

        loop.add_signal_handler(getattr(signal, signame), shutdown_event.set)


def log_startup_summary(settings: Settings) -> None:
    """Emit a concise, non-secret startup summary for operators.

    Args:
        settings: Loaded application settings instance.
    """

    LOGGER.info(
        "Agent bootstrap ready",
        extra={
            "environment": settings.environment,
            "pipeline_start_hour": settings.pipeline_start_hour,
            "publish_hour": settings.publish_hour,
            "database_configured": bool(settings.database_url),
            "redis_configured": bool(settings.redis_url),
            "admin_count": len(settings.admin_telegram_ids),
        },
    )


async def run_service(settings: Settings) -> None:
    """Keep the agent process alive until interrupted.

    This gives Docker and local development a stable runtime target before the
    scheduler, Telegram bot, and LangGraph pipeline land in later steps.

    Args:
        settings: Loaded application settings instance.
    """

    shutdown_event = asyncio.Event()
    install_signal_handlers(shutdown_event)
    log_startup_summary(settings)

    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=settings.bootstrap_heartbeat_seconds,
            )
        except TimeoutError:
            LOGGER.info("Agent bootstrap heartbeat: runtime is idle and healthy.")

    LOGGER.info("Shutdown signal received; stopping agent bootstrap loop.")


async def async_main(argv: Sequence[str] | None = None) -> int:
    """Run the asynchronous agent bootstrap workflow.

    Args:
        argv: Optional argument vector for easier testing.

    Returns:
        Process exit code: `0` for success.
    """

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    settings = get_settings()
    configure_logging(settings.log_level)

    if args.check:
        log_startup_summary(settings)
        LOGGER.info("Configuration check completed successfully.")
        return 0

    await run_service(settings)
    return 0


def run() -> int:
    """Execute the agent runtime from synchronous contexts.

    Returns:
        Process exit code returned from the async runtime.
    """

    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(run())
