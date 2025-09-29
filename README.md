
# Media Lens — Stance & Coverage Explorer 📰

Media Lens is a **news stance & coverage explorer** that clusters related articles into events, quantifies **stance** (pro/neutral/anti) and **framing** (multi-label tags), and visualizes **coverage diversity** across outlets. It’s designed to be **auditable, explainable, and deployable** — no vague “bias scores,” just metrics you can interrogate.

---

## 🚀 Features
- **Ingestion & Clustering**  
  - RSS + approved API ingestion (robots.txt–respecting)  
  - Title/lede embeddings (Sentence-Transformers)  
  - Event clustering (time-windowed HDBSCAN / k-means)

- **ML Models**  
  - **Stance**: pro/neutral/anti (baseline linear classifier, fine-tuned DistilBERT stretch goal)  
  - **Framing**: 5–8 issue tags (keyword heuristics → multi-label classifier)  
  - **Confidence & Calibration**: show abstentions for low-confidence predictions

- **UI & Visualization**  
  - URL view: summary, stance bar, framing chips, confidence badge  
  - Event view: outlet table, stance histogram, diversity dial (Jensen–Shannon divergence)  
  - “Why?” drawer: token-level rationales (SHAP/attention)  

- **Ops & Observability**  
  - FastAPI backend with Redis caching & Postgres persistence  
  - Dockerized, deployed via GitHub Actions to Railway/Render  
  - Admin metrics panel (QPS, p95 latency, cache hit %, events indexed)

---

## 🏗️ Architecture

```text
┌────────────┐    RSS/APIs
│  Ingestor │───────▶ Articles (Postgres)
└─────┬──────┘
      │ Title/Lede Embedding
      ▼
 Event Clustering ─────▶ Events + Article-Events
      │
      ▼
  Stance/Framing Model ─────▶ Predictions (Postgres)
      │ Cache
      ▼
   Redis (7d TTL)
      │
      ▼
 FastAPI API Layer ─▶ React Frontend (URL View, Event View, Admin Panel)
```

---

## 📦 Tech Stack
| Layer        | Tech |
|--------------|------|
| **Frontend** | React + Vite, Tailwind, Recharts |
| **Backend**  | FastAPI (Uvicorn), Pydantic |
| **Models**   | HuggingFace Sentence-Transformers + Logistic Regression / DistilBERT |
| **Data**     | Postgres (articles/events/predictions), Redis (cache) |
| **Ops**      | Docker, GitHub Actions, Railway/Render deploy, Logtail/OTLP for logs |

---

## 📊 Metrics
| Metric | Goal |
|-------|------|
| **Stance** | Macro-F1 ≥ 0.65 (baseline), ≥ 0.75–0.80 (fine-tuned) |
| **Framing** | Micro-F1 ≥ 0.55 baseline, ≥ 0.65+ tuned |
| **Latency** | p95 < 500 ms (cached < 250 ms) |
| **Coverage Diversity** | Multi-distribution Jensen–Shannon divergence + headline similarity variance |

---

## 🏁 Getting Started

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/media-lens.git
cd media-lens
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit DATABASE_URL + RSS_FEEDS in .env
psql "$DATABASE_URL" -f sql/schema.sql
```

### 2. Ingest Articles
```bash
# Recommended: run as modules so imports resolve cleanly
python -m scripts.ingest_rss
```

### 3. Embed & Cluster
```bash
# Embeddings/cluster intermediates saved to DATA_DIR (default ./data)
export DATA_DIR=./data
python -m scripts.embed_titles
python -m scripts.cluster_events
```

### 4. Run Baselines (Stance + Framing)
```bash
# Writes artifacts under ./reports/ and persists models to ./outputs/
python -m scripts.evaluate_baselines
```

Artifacts written:
- reports/stance_classification_report.txt
- reports/stance_confusion_matrix.png
- reports/stance_reliability_plot.png (when available)
- reports/framing_report.txt (micro/macro F1, subset accuracy, thresholds)
- reports/framing_classification_report.txt (per-tag metrics)
- reports/params.json (embedder, seed, selected models/params)

Persisted models:
- outputs/stance_model.joblib (model + label encoder)
- outputs/framing_model.joblib (vectorizer + classifier + thresholds + labels)

### 5. Run API
```bash
uvicorn api.main:app --reload
```
Frontend will fetch from `localhost:8000`.

---

## ⚙️ Evaluation Options (env vars)

Stance/Framing evaluation accepts the following environment variables:

- EVAL_EMBEDDER: Sentence-Transformers model (default `sentence-transformers/all-MiniLM-L6-v2`).
- REPORTS_DIR: Directory for reports (default `./reports`).
- EVAL_DATA_DIR, EVAL_OUT_DIR: Data/output dirs (defaults `./data`, `./outputs`).

Framing-specific:
- FRAMING_TUNE_THRESHOLDS: `1` to tune per-tag thresholds on validation; `0` to skip and use a global threshold (default `1`).
- FRAMING_THRESHOLD: Global threshold when tuning is off (default `0.3`).
- FRAMING_ENSEMBLE: `1` to ensemble TF-IDF + SBERT logistic; `0` to use TF-IDF only (default `0`).

Stance-specific:
- STANCE_CALIBRATE: `1` to calibrate LinearSVC with sigmoid (enables probabilities); `0` to skip for speed (default `1`).

Examples:
```bash
# Fast-mode example
export STANCE_CALIBRATE=0
export FRAMING_TUNE_THRESHOLDS=0
export FRAMING_THRESHOLD=0.2
python -m scripts.evaluate_baselines

# Thorough-mode example
export EVAL_EMBEDDER=sentence-transformers/all-mpnet-base-v2
export FRAMING_TUNE_THRESHOLDS=1
export FRAMING_ENSEMBLE=1
python -m scripts.evaluate_baselines
```

---

## 📸 Screenshots
(Coming soon: URL view, Event view with stance bars, Diversity dial)

---

## 🔒 Ethics & Risk Controls
- Respect site TOS/robots.txt; fall back to headline/lede-only when needed  
- Show confidence and rationale; never a single “bias score”  
- Abstain when model is low-confidence (no forced labels)  
- Document sources and limitations (`DATA_SOURCES.md`, `LIMITATIONS.md`)

---

## 🧪 Testing & CI
- **Unit tests**: ingestion, caching, DB writes  
- **Integration tests**: cold→warm cache path, event diversity computation  
- **Load tests**: 100 rps burst, p95 < 500 ms  
- GitHub Actions: lint → test → build → deploy

---

## 🗺️ Roadmap
- [x] Week 1: Ingestor, embeddings, baseline stance/framing  
- [x] Week 2: FastAPI services, caching, DB migrations  
- [x] Week 3: React UI (URL + Event view, Why drawer)  
- [x] Week 4: Docker, CI/CD, deploy, observability  
- [ ] Week 5: Fine-tune stance, multi-label framing, load testing, demo dataset  

---

## 📜 License
MIT License — see `LICENSE`.
