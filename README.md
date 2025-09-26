
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
python scripts/ingest_rss.py
```

### 3. Embed & Cluster
```bash
python scripts/embed_titles.py
python scripts/cluster_events.py
```

### 4. Run Baselines
```bash
python scripts/evaluate_baselines.py
```
(Replace with real datasets for meaningful metrics)

### 5. Run API
```bash
uvicorn api.main:app --reload
```
Frontend will fetch from `localhost:8000`.

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
