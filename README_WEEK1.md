# Week 1 — Data & Baselines: Handheld Start

This folder contains **ready-to-run scaffolding** for Week 1. Follow the steps below.

## 0) Install & Configure
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env to point DATABASE_URL to your Postgres
```

Create the DB tables:
```bash
psql "$DATABASE_URL" -f sql/schema.sql
```

## 1) Ingest a Few Feeds
```bash
python scripts/ingest_rss.py
```
- This normalizes articles and stores them in `articles`.
- It computes a basic **canonical_url**, sets **robots_ok**/**text_limited**.

## 2) Embed Titles (for quick clustering)
```bash
python scripts/embed_titles.py
```
- Saves `title_ids.npy` and `title_embeds.npy` to `/mnt/data`

## 3) Cluster Into Events
```bash
python scripts/cluster_events.py
```
- Creates `events` rows and links articles in `article_events`

## 4) Baseline Stance (Toy Example)
```bash
python scripts/evaluate_baselines.py
```
- Replace the dataset loader with FNC-1 / SemEval, then plug into your pipeline.

## What Next?
- Replace KMeans with time-windowed clustering (HDBSCAN or k-means per window).
- Add true datasets + proper evaluation & confusion matrices.
- Commit calibration curves once you swap in real models.

---
Generated: 2025-09-26T04:16:01.473026
