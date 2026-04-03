-- Purpose: Define the canonical initial PostgreSQL schema for PuntLab.
-- Scope: Creates the core relational tables, indexes, and constraints from
-- Dependencies: Requires PostgreSQL with the pgcrypto extension available for
-- gen_random_uuid().

BEGIN;

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Competitions & Leagues
CREATE TABLE competitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    country VARCHAR(100),
    sport VARCHAR(50) NOT NULL CHECK (sport IN ('soccer', 'basketball')),
    league_code VARCHAR(50) UNIQUE,
    api_football_id INTEGER,
    football_data_id INTEGER,
    is_active BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Fixtures / Matches
CREATE TABLE fixtures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    competition_id UUID REFERENCES competitions(id),
    sportradar_id VARCHAR(100) UNIQUE,
    home_team VARCHAR(255) NOT NULL,
    away_team VARCHAR(255) NOT NULL,
    home_team_id VARCHAR(100),
    away_team_id VARCHAR(100),
    match_date DATE NOT NULL,
    kickoff_time TIMESTAMPTZ,
    venue VARCHAR(255),
    status VARCHAR(50) DEFAULT 'scheduled',
    home_score INTEGER,
    away_score INTEGER,
    api_football_id INTEGER,
    sportybet_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_fixtures_date ON fixtures(match_date);
CREATE INDEX idx_fixtures_sportradar ON fixtures(sportradar_id);

-- Odds Data
CREATE TABLE odds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fixture_id UUID REFERENCES fixtures(id),
    provider VARCHAR(100) NOT NULL,
    market_type VARCHAR(100) NOT NULL,
    market_label VARCHAR(255),
    selection VARCHAR(255) NOT NULL,
    odds_value DECIMAL(8,3) NOT NULL,
    sportybet_market_id INTEGER,
    is_available BOOLEAN DEFAULT true,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (fixture_id, provider, market_type, selection)
);

CREATE INDEX idx_odds_fixture ON odds(fixture_id);

-- Team Statistics
CREATE TABLE team_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id VARCHAR(100) NOT NULL,
    team_name VARCHAR(255) NOT NULL,
    competition_id UUID REFERENCES competitions(id),
    season VARCHAR(20),
    matches_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    goals_for INTEGER DEFAULT 0,
    goals_against INTEGER DEFAULT 0,
    clean_sheets INTEGER DEFAULT 0,
    form VARCHAR(50),
    position INTEGER,
    points INTEGER,
    home_wins INTEGER DEFAULT 0,
    away_wins INTEGER DEFAULT 0,
    avg_goals_scored DECIMAL(4,2),
    avg_goals_conceded DECIMAL(4,2),
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);

-- Injuries & Suspensions
CREATE TABLE injuries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fixture_id UUID REFERENCES fixtures(id),
    team_id VARCHAR(100) NOT NULL,
    player_name VARCHAR(255) NOT NULL,
    injury_type VARCHAR(100),
    reason VARCHAR(500),
    is_key_player BOOLEAN DEFAULT false,
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);

-- Match Analysis (per-match scoring output)
CREATE TABLE match_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,
    fixture_id UUID REFERENCES fixtures(id),
    match_date DATE NOT NULL,
    form_score DECIMAL(5,3),
    h2h_score DECIMAL(5,3),
    injury_impact_score DECIMAL(5,3),
    odds_value_score DECIMAL(5,3),
    context_score DECIMAL(5,3),
    venue_score DECIMAL(5,3),
    composite_score DECIMAL(5,3) NOT NULL,
    confidence DECIMAL(5,3) NOT NULL,
    global_rank INTEGER,
    recommended_market VARCHAR(100),
    recommended_selection VARCHAR(255),
    recommended_odds DECIMAL(8,3),
    news_summary TEXT,
    context_notes TEXT,
    llm_assessment TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_analyses_run ON match_analyses(run_id);
CREATE INDEX idx_analyses_date ON match_analyses(match_date);

-- Accumulator Slips
CREATE TABLE accumulators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL,
    slip_date DATE NOT NULL,
    slip_number INTEGER NOT NULL,
    total_odds DECIMAL(10,3) NOT NULL,
    leg_count INTEGER NOT NULL,
    confidence DECIMAL(5,3) NOT NULL,
    rationale TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    outcome VARCHAR(50),
    is_published BOOLEAN DEFAULT false,
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_accumulators_date ON accumulators(slip_date);

-- Accumulator Legs
CREATE TABLE accumulator_legs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    accumulator_id UUID REFERENCES accumulators(id) ON DELETE CASCADE,
    fixture_id UUID REFERENCES fixtures(id),
    analysis_id UUID REFERENCES match_analyses(id),
    leg_number INTEGER NOT NULL,
    market_type VARCHAR(100) NOT NULL,
    selection VARCHAR(255) NOT NULL,
    odds_value DECIMAL(8,3) NOT NULL,
    provider VARCHAR(100) NOT NULL,
    rationale TEXT,
    outcome VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    telegram_id BIGINT UNIQUE,
    telegram_username VARCHAR(255),
    email VARCHAR(255) UNIQUE,
    display_name VARCHAR(255),
    subscription_tier VARCHAR(50) DEFAULT 'free',
    subscription_status VARCHAR(50) DEFAULT 'active',
    subscription_expires_at TIMESTAMPTZ,
    paystack_customer_id VARCHAR(255),
    is_admin BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_telegram ON users(telegram_id);

-- Pipeline Runs (execution log)
CREATE TABLE pipeline_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_date DATE NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'running',
    trigger VARCHAR(50) DEFAULT 'scheduled',
    fixtures_analyzed INTEGER DEFAULT 0,
    accumulators_generated INTEGER DEFAULT 0,
    accumulators_published INTEGER DEFAULT 0,
    errors JSONB DEFAULT '[]',
    stage_timings JSONB DEFAULT '{}',
    llm_tokens_used JSONB DEFAULT '{}',
    llm_cost_usd DECIMAL(8,4) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Delivery Log
CREATE TABLE delivery_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    accumulator_id UUID REFERENCES accumulators(id),
    user_id UUID REFERENCES users(id),
    channel VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    error_message TEXT,
    delivered_at TIMESTAMPTZ DEFAULT NOW()
);

-- Payment Transactions
CREATE TABLE payments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    provider VARCHAR(50) DEFAULT 'paystack',
    provider_reference VARCHAR(255),
    amount_ngn DECIMAL(12,2) NOT NULL,
    plan VARCHAR(50) NOT NULL,
    duration_days INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMIT;
