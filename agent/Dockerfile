# Container image definition for the PuntLab Python agent.
# Scope: builds the canonical Python 3.12 runtime with project dependencies and
# the Playwright Chromium browser used by later SportyBet scraping steps.
# Dependencies: relies on `pyproject.toml` and the `src/` package tree.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./pyproject.toml
COPY src ./src

RUN pip install --upgrade pip \
    && pip install . \
    && python -m playwright install --with-deps chromium

CMD ["python", "-m", "src.main"]
