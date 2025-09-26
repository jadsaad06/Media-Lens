-- Run this once to bootstrap the Week 1 database schema.
CREATE EXTENSION IF NOT EXISTS pgcrypto; -- for gen_random_uuid on some Postgres installs

CREATE TABLE IF NOT EXISTS articles (
  id BIGSERIAL PRIMARY KEY,
  url TEXT UNIQUE NOT NULL,
  canonical_url TEXT,
  domain TEXT,
  source TEXT,
  title TEXT,
  published_at TIMESTAMPTZ,
  summary TEXT,
  text TEXT,
  hash TEXT UNIQUE,
  robots_ok BOOLEAN DEFAULT TRUE,
  text_limited BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS events (
  id BIGSERIAL PRIMARY KEY,
  label TEXT,
  centroid_embed VECTOR(384), -- if using all-MiniLM 384-dim; adjust if needed
  time_window_start TIMESTAMPTZ,
  time_window_stop TIMESTAMPTZ,
  size INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Bridge table mapping articles to events
CREATE TABLE IF NOT EXISTS article_events (
  article_id BIGINT REFERENCES articles(id) ON DELETE CASCADE,
  event_id BIGINT REFERENCES events(id) ON DELETE CASCADE,
  PRIMARY KEY (article_id, event_id)
);

-- Predictions for Week 1 baselines
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'stance_label') THEN
    CREATE TYPE stance_label AS ENUM ('pro','neutral','anti');
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS predictions (
  id BIGSERIAL PRIMARY KEY,
  article_id BIGINT REFERENCES articles(id) ON DELETE CASCADE,
  stance stance_label,
  stance_conf FLOAT,
  framing_tags TEXT[],
  framing_conf FLOAT[],
  model_version TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_articles_domain_pub ON articles(domain, published_at);
CREATE INDEX IF NOT EXISTS idx_predictions_article ON predictions(article_id);
