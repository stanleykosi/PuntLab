"""Tests for PuntLab's normalized news article schema.

Purpose: verify article publication validation, ordered team matching, and
JSON-safe serialization for research-stage inputs.
Scope: unit tests for `src.schemas.news.NewsArticle`.
Dependencies: pytest, Pydantic URL parsing, and the shared sport enum.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from src.config import SportName
from src.schemas.news import NewsArticle


def test_news_article_normalizes_team_names_and_serializes_urls() -> None:
    """Article schemas should deduplicate teams and serialize URLs as strings."""

    article = NewsArticle(
        headline="Arsenal prepare for Chelsea clash",
        url="https://www.bbc.com/sport/football/example-match-preview",
        published_at=datetime(2026, 4, 3, 5, 30, tzinfo=UTC),
        source="BBC Sport",
        source_provider="rss",
        sport=SportName.SOCCER,
        teams=(" Arsenal ", "Chelsea", "arsenal"),
        relevance_score=0.8,
    )

    dumped = article.model_dump(mode="json")

    assert article.teams == ("Arsenal", "Chelsea")
    assert dumped["url"] == "https://www.bbc.com/sport/football/example-match-preview"
    assert dumped["sport"] == "soccer"


def test_news_article_rejects_naive_timestamps() -> None:
    """Publication timestamps should be timezone-aware."""

    with pytest.raises(ValueError, match="timezone information"):
        NewsArticle(
            headline="Preview",
            url="https://www.espn.com/example",
            published_at=datetime(2026, 4, 3, 5, 30),
            source="ESPN",
            source_provider="tavily",
        )
