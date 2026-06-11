# LexRAG 

**Production-grade Agentic RAG system for Indian Law**

[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-FF6B35?style=flat-square)](https://langchain-ai.github.io/langgraph)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=flat-square&logo=render&logoColor=white)](https://render.com)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?style=flat-square&logo=vercel&logoColor=white)](https://vercel.com)

---

LexRAG is an end-to-end intelligent legal research assistant that combines **InLegalBERT** (domain-specific Indian law embeddings) with a **7-node LangGraph agentic pipeline** to deliver grounded, cited answers from the Indian Constitution, IPC, CrPC, Evidence Act, and user-uploaded documents. It includes a standalone **Contract Intelligence** module with automated Gmail renewal reminders.

```
User Question → Rewrite → Strategist → [Legal KB | Document | Both] → Fusion → Answer → Critic
```

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [Knowledge Base Construction](#knowledge-base-construction)
  - [Running Locally](#running-locally)
  - [Docker](#docker)
- [Notebooks](#notebooks)
- [API Reference](#api-reference)
- [Deployment](#deployment)
  - [Backend → Render](#backend--render)
  - [Frontend → Vercel](#frontend--vercel)
  - [CI/CD → GitHub Actions](#cicd--github-actions)
- [ONNX Optimisation](#onnx-optimisation)
- [Evaluation](#evaluation)
- [Contract Intelligence](#contract-intelligence)
- [Contributing](#contributing)
- [License](#license)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                       │
│              Next.js 14  ·  Vercel CDN                      │
└────────────────────────────┬────────────────────────────────┘
                             │ REST
┌────────────────────────────▼────────────────────────────────┐
│                       FastAPI Backend                       │
│                    Render  ·  Docker                        │
│                                                             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │                  LangGraph Pipeline                  │  │
│   │                                                      │  │
│   │  [Rewrite] → [Strategist] → [Legal] ─┐               │  │
│   │                           → [Doc]   ─┤→ [Fusion]     │  │
│   │                           → [Both]  ─┘    │          │  │
│   │                                       [Answer]       │  │
│   │                                           │          │  │
│   │                                       [Critic] → END │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                             │
│     ┌──────────────┐  ┌────────────────┐                    │
│     │  BGE-Base    │  │  BGE-Reranker  │                    │
│     │  ONNX INT8   │  │  (lazy load)   │                    │
│     │  ~90MB RAM   │  │   ~90MB RAM    │                    │
│     └──────────────┘  └────────────────┘                    │
└────────────┬───────────────┬─────────────────────────────── ┘
             │               │
   ┌──────────▼───┐   ┌──────▼──────┐   ┌────────────────┐
   │   Pinecone   │   │  Supabase   │   │  Redis Cloud   │
   │ Vector Store │   │  Postgres   │   │    Cache       │
   │  (prod)      │   │  Contracts  │   │   1hr TTL      │
   └──────────────┘   └─────────────┘   └────────────────┘
   ┌──────────────┐
   │  ChromaDB    │
   │  (local dev) │
   └──────────────┘
```

### Agent Roles

| Node | Model | Role |
|------|-------|------|
| **Rewrite** | Llama-3.3-70b | Transforms query into retrieval-optimised form with legal terminology |
| **Strategist** | Llama-3.1-8b | Routes to LEGAL, DOCUMENT, or BOTH based on question type |
| **Legal** | Hybrid Search | BM25 + vector search over Indian law knowledge base |
| **Document** | Hybrid Search | BM25 + vector search over user-uploaded PDF |
| **Fusion** | BGE-Reranker | Merges results via RRF + cross-encoder reranking |
| **Answer** | Llama-3.3-70b | Generates grounded, cited answer from top-K chunks |
| **Critic** | Llama-3.3-70b | Self-evaluates faithfulness and completeness |

---

## Features

**Legal Knowledge Chat**
- Query the full Indian legal corpus — Constitution, IPC, CrPC, Evidence Act, Contract Act, BNS/BNSS/BSA
- Upload any PDF for combined document + legal knowledge search
- Hybrid BM25 + vector retrieval with Reciprocal Rank Fusion
- Cross-encoder reranking for precision retrieval
- Self-evaluation via critic agent (faithfulness score per answer)
- Redis caching — repeated questions return in ~50ms

**Contract Intelligence**
- Upload contract PDFs — LLM extracts parties, dates, obligations, auto-renewal clauses
- Structured storage in Supabase (Postgres)
- Automated Gmail renewal reminders at 30 / 7 / 1 day before expiry
- Natural language queries over all stored contracts
- Background thread test email fired 2 minutes after upload

**Production Engineering**
- InLegalBERT + BGE-Base converted to ONNX INT8 — fits in 512MB Render free tier
- ONNX model files split into <100MB parts for GitHub upload, reassembled at Docker build time
- Lazy cross-encoder loading — health check passes before full model load
- ChromaDB (local dev) ↔ Pinecone (production) automatic switch via `APP_ENV`
- RAGAS evaluation suite — faithfulness, answer relevancy, context recall, answer correctness
- Full CI/CD via GitHub Actions — tests must pass before Render deploys

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Query Embedding** | InLegalBERT (law-ai/InLegalBERT) via ONNX Runtime INT8 |
| **Passage Embedding** | BGE-Base-en-v1.5 via ONNX Runtime INT8 |
| **Reranker** | BGE-Reranker-Base (CrossEncoder, lazy loaded) |
| **LLM** | Groq · Llama-3.3-70b-versatile + Llama-3.1-8b-instant |
| **Agent Framework** | LangGraph 0.2.28 |
| **Vector DB (dev)** | ChromaDB 0.5.23 |
| **Vector DB (prod)** | Pinecone Free Tier |
| **Relational DB** | Supabase (Postgres) |
| **Cache** | Redis Cloud (30MB free) |
| **Backend** | FastAPI 0.115 · Uvicorn · Docker |
| **Frontend** | Next.js 14 App Router · Tailwind CSS |
| **Backend Hosting** | Render Free Tier |
| **Frontend Hosting** | Vercel |
| **CI/CD** | GitHub Actions |
| **Evaluation** | RAGAS 0.2.x |
| **Email** | Gmail SMTP via Python smtplib |

---

## Project Structure

```
lexrag/
├── notebooks/
│   ├── 01_processing_chunking.ipynb    # Build vector store from PDFs
│   ├── 02_agentic_rag.ipynb            # Test LangGraph pipeline
│   ├── 03_ragas_evaluation.ipynb       # RAGAS evaluation suite
│   └── 04_contract_agent.ipynb         # Contract intelligence pipeline
│
├── backend/
│   ├── main.py                         # FastAPI application
│   ├── agents/
│   │   ├── legal_graph.py              # LangGraph pipeline (canonical)
│   │   └── contract_agent.py           # Contract extraction agent
│   ├── models/
│   │   ├── onnx_embeddings.py          # ONNX Runtime embedding wrapper
│   │   ├── inlegalbert_onnx/           # InLegalBERT ONNX INT8 parts
│   │   │   ├── model_quantized.onnx.part1
│   │   │   ├── model_quantized.onnx.part2
│   │   │   ├── model_quantized.onnx.part3
│   │   │   ├── tokenizer.json
│   │   │   └── config.json
│   │   └── bge_onnx/                   # BGE-Base ONNX INT8 parts
│   ├── db/
│   │   └── supabase_client.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx                    # Landing page
│   │   ├── chat/page.tsx               # Legal chat interface
│   │   ├── contracts/page.tsx          # Contract intelligence dashboard
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── tailwind.config.ts
│   └── package.json
│
├── scripts/
│   ├── convert_inlegalbert_to_onnx.py  # One-time ONNX conversion
│   ├── convert_bge_to_onnx.py
│   ├── split_model.py                  # Split >100MB ONNX for GitHub
│   ├── reassemble_model.py             # Reassemble locally after clone
│   ├── migrate_to_pinecone.py          # One-time ChromaDB → Pinecone
│   └── verify_setup.py                 # Environment health check
│
├── data/
│   ├── raw/                            # Downloaded legal PDFs (gitignored)
│   └── vector_database/                # ChromaDB persist dir (gitignored)
│
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions CI pipeline
│
├── render.yaml                         # Render deployment config
├── docker-compose.yml                  # Local Docker development
├── .env.example                        # Environment variable template
└── .gitignore
```

---

## Getting Started

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11.9 | Managed via Conda |
| Node.js | 20.x LTS | Frontend only |
| Conda / Miniconda | 24.x | Recommended over venv |
| Docker Desktop | Latest | For local containerised dev |
| Git | Latest | |

GPU is optional — all models run on CPU. A GPU reduces embedding generation time during knowledge base construction (Notebook 01) but is not required.

### Environment Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/lexrag.git
cd lexrag
```

**2. Create and activate the Conda environment**

```bash
conda create -n lexrag python=3.11.9 -y
conda activate lexrag
```

**3. Install PyTorch (CPU build — install before everything else)**

```bash
# NVIDIA GPU with CUDA 12.1
pip install torch==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# CPU only (Render free tier, no GPU)
pip install torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon (M1/M2/M3)
pip install torch==2.4.0
```

**4. Install all backend dependencies**

```bash
pip install -r backend/requirements.txt
```

**5. Download NLTK data**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

**6. Register the Jupyter kernel**

```bash
pip install jupyter ipykernel notebook
python -m ipykernel install --user --name lexrag --display-name "LexRAG (Python 3.11)"
```

**7. Verify the environment**

```bash
python scripts/verify_setup.py
```

All items should show `PASS`. Fix any `FAIL` items before proceeding.

### Environment Variables

Copy `.env.example` to `.env` and fill in all values:

```bash
cp .env.example .env
```

```bash
# .env

# ── LLM (free at console.groq.com) ─────────────────────────────────────────
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx

# ── Vector DB (free at app.pinecone.io — needed for Render deployment) ──────
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
PINECONE_INDEX_NAME=legal-knowledge

# ── Supabase ────────────────────────────────────────────────────────────────
# Project URL: Supabase Dashboard → Connect modal → Project URL
SUPABASE_URL=https://xxxxxxxxxxxx.supabase.co
# Anon key: Project Settings → API → anon/public (starts with eyJ...)
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxx

# ── Gmail SMTP (replaces SendGrid — use App Password, not login password) ───
# Generate at: myaccount.google.com/apppasswords
EMAIL_SENDER=yourname@gmail.com
EMAIL_PASSWORD=xxxx xxxx xxxx xxxx
EMAIL_RECEIVER=yourname@gmail.com

# ── Redis (optional — free at app.redis.com) ────────────────────────────────
REDIS_URL=redis://default:password@host:port

# ── App config ──────────────────────────────────────────────────────────────
APP_ENV=development
CHROMA_PERSIST_DIR=../data/vector_database
```

> **Note:** `EMAIL_PASSWORD` must be a Gmail App Password — not your Gmail login password. Generate one at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) after enabling 2-Step Verification.

### Knowledge Base Construction

Download Indian legal documents and build the vector store. Run this **once** — the resulting ChromaDB is reused by all subsequent runs.

**1. Download legal corpus**

```bash
python scripts/download_corpus.py
# Downloads: Constitution, IPC, CrPC, Evidence Act, Contract Act, CPC
# Saved to: data/raw/
```

**2. Run Notebook 01**

```bash
jupyter notebook notebooks/01_processing_chunking.ipynb
```

Run all cells. This takes 10–20 minutes depending on hardware. On completion, `data/vector_database/` contains the ChromaDB with 5,000+ hybrid-tagged chunks.

**3. One-time Pinecone migration (required for Render deployment)**

```bash
python scripts/migrate_to_pinecone.py
```

### Running Locally

**Backend**

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

**Frontend**

```bash
cd frontend
npm install
cp .env.local.example .env.local
# Set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000).

### Docker

Run the full stack (backend + Redis) with a single command:

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Backend API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Frontend | http://localhost:3000 (run separately with `npm run dev`) |
| Redis | localhost:6379 |

Stop without deleting data:
```bash
docker-compose down
```

Full reset (deletes all volumes including model cache):
```bash
docker-compose down -v
```

---

## Notebooks

| Notebook | Purpose | Run When |
|----------|---------|----------|
| `01_processing_chunking` | Downloads, cleans, chunks, and embeds Indian legal PDFs into ChromaDB | Once — before any other notebook |
| `02_agentic_rag` | Interactive testing of the full LangGraph pipeline with sample questions | During development |
| `03_ragas_evaluation` | Evaluates pipeline quality across 4 RAGAS metrics with retry logic | After any change to retrieval or prompts |
| `04_contract_agent` | Contract PDF upload, LLM extraction, Supabase storage, Gmail reminders | During development of contract features |

Select `LexRAG (Python 3.11)` as the kernel in all notebooks.

---

## API Reference

### `GET /health`
Returns service status. Used by Render health checks.

```json
{ "status": "ok", "version": "2.0.0" }
```

### `POST /chat`
Run a legal question through the full LangGraph pipeline.

**Request**
```json
{
  "question": "What does Article 21 of the Indian Constitution guarantee?",
  "file_path": "/tmp/uploads/abc123.pdf"  // optional
}
```

**Response**
```json
{
  "answer": "Article 21 guarantees the right to life and personal liberty...",
  "strategy": "LEGAL",
  "critique": "FAITH:9 COMPLETE:8 ISSUES:None"
}
```

### `POST /upload`
Upload a PDF document. Returns the server-side file path to pass to `/chat`.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@contract.pdf"
```

**Response**
```json
{ "file_path": "/tmp/uploads/abc123.pdf", "file_id": "abc123" }
```

### `POST /contracts/analyze`
Upload and analyse a contract PDF. Extracts metadata and saves to Supabase.

```bash
curl -X POST http://localhost:8000/contracts/analyze \
  -F "file=@service_agreement.pdf" \
  -F "user_id=my_user"
```

### `GET /contracts`
List all contracts for a user.

```bash
curl "http://localhost:8000/contracts?user_id=my_user"
```

---

## Deployment

### Backend → Render

**1.** Push your repository to GitHub.

**2.** Go to [render.com](https://render.com) → New → Web Service → connect your repo.

**3.** Render auto-detects `render.yaml`. Confirm settings:
```
Runtime  : Docker
Plan     : Free
```

**4.** Add all environment variables from your `.env` under **Environment** in the Render dashboard.

**5.** Click **Deploy**. First build takes 8–12 minutes (model download + reassembly).

**6.** Health check endpoint: `https://your-service.onrender.com/health`

> **Free tier note:** Render free services sleep after 15 minutes of inactivity. Cold start takes ~30 seconds. Upgrade to the $7/month Starter plan to keep the service always-on.

### Frontend → Vercel

**1.** Go to [vercel.com](https://vercel.com) → New Project → import your repo.

**2.** Set **Root Directory** to `frontend`.

**3.** Add environment variable:
```
NEXT_PUBLIC_API_URL = https://your-service.onrender.com
```

**4.** Click **Deploy**. Your app is live at `https://your-app.vercel.app`.

**5.** Update `ALLOWED_ORIGINS` in Render to include your Vercel URL:
```
ALLOWED_ORIGINS = https://your-app.vercel.app
```

### CI/CD → GitHub Actions

Every push to `main` triggers the CI pipeline defined in `.github/workflows/ci.yml`:

```
Push to main
  └── Run tests (pytest)
  └── Start FastAPI, check /health
  └── If all pass → Render auto-deploys
  └── If any fail → Render does not deploy
```

Add the following secrets in **GitHub → Settings → Secrets and variables → Actions**:

```
GROQ_API_KEY
SUPABASE_URL
SUPABASE_ANON_KEY
PINECONE_API_KEY
EMAIL_SENDER
EMAIL_PASSWORD
EMAIL_RECEIVER
REDIS_URL
```

---

## ONNX Optimisation

InLegalBERT and BGE-Base are converted to ONNX INT8 to fit within Render's 512MB free tier.

| Model | PyTorch RAM | ONNX INT8 RAM | Saving |
|-------|-------------|---------------|--------|
| InLegalBERT | ~450 MB | ~90 MB | 80% |
| BGE-Base | ~450 MB | ~90 MB | 80% |
| BGE-Reranker | ~450 MB | ~90 MB (lazy) | 80% |
| **Total at startup** | **~900 MB → OOM** | **~180 MB → fits** | |

**Convert models (run once locally):**

```bash
python scripts/convert_inlegalbert_to_onnx.py
python scripts/convert_bge_to_onnx.py
```

**Split for GitHub (100MB file limit):**

```bash
python scripts/split_model.py
# Creates model_quantized.onnx.part1/2/3 — each under 100MB
# Delete the original model_quantized.onnx before committing
```

**Reassemble after cloning:**

```bash
python scripts/reassemble_model.py
```

The Dockerfile on Render runs reassembly automatically at build time.

---

## Evaluation

RAGAS evaluation runs against a curated 8-question ground truth dataset covering constitutional law, IPC, and procedural law.

```bash
# Run evaluation (takes ~15 minutes due to Groq rate limit delays)
jupyter notebook notebooks/03_ragas_evaluation.ipynb
```

**Baseline scores** (llama3-8b-8192 judge, BGE-Base embeddings):

| Metric | Target | Description |
|--------|--------|-------------|
| `faithfulness` | > 0.75 | Every claim traceable to retrieved context |
| `answer_relevancy` | > 0.75 | Answer addresses the actual question |
| `context_recall` | > 0.70 | Retrieved chunks contain the right information |
| `answer_correctness` | > 0.65 | Answer matches ground truth |

Results are saved to `evaluation_results.csv` after each run.

**Rate limit handling** — the evaluation notebook implements automatic retry with exact wait times parsed from Groq's error messages, plus 10-second delays between questions to stay under the 100K tokens/day free limit.

---

## Contract Intelligence

The contract agent (Notebook 04 + `/contracts/analyze` endpoint) implements a complete contract lifecycle management pipeline.

**Extraction fields:**

| Field | Description |
|-------|-------------|
| `party_a` / `party_b` | Full legal names of both parties |
| `contract_type` | Service Agreement, NDA, Lease, Employment, etc. |
| `start_date` / `end_date` | Contract term in YYYY-MM-DD format |
| `renewal_date` | Date reminders are calculated from |
| `notice_period_days` | Days of notice required (default 30) |
| `auto_renewal` | Whether contract renews automatically |
| `governing_law` | Jurisdiction (e.g. Laws of India, Karnataka) |
| `key_obligations` | 2–3 sentence summary of main obligations |

**Reminder schedule:**

```
renewal_date - 30 days → Email: "Renewing in 30 days"
renewal_date - 7 days  → Email: "Renewing in 7 days"
renewal_date - 1 day   → Email: "Renewing tomorrow"
```

Reminders are sent via Gmail SMTP using `smtplib` (Python standard library — no paid service required). The `reminder_sent_30/7/1` boolean columns in Supabase ensure each reminder fires exactly once.

**Test email:** Upload any contract through the UI or API — a test email fires automatically 2 minutes later via a background daemon thread, confirming the email pipeline is working before any real deadline arrives.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and add tests if applicable
4. Run the CI checks locally:
   ```bash
   python scripts/verify_setup.py
   pytest backend/tests/ -v
   ```
5. Commit with a descriptive message:
   ```bash
   git commit -m "feat: add multi-document comparison endpoint"
   ```
6. Push and open a Pull Request against `main`

**Commit message convention:**

| Prefix | Use for |
|--------|---------|
| `feat:` | New features |
| `fix:` | Bug fixes |
| `docs:` | Documentation changes |
| `refactor:` | Code restructuring without behaviour change |
| `deps:` | Dependency updates |
| `ci:` | CI/CD pipeline changes |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with [LangGraph](https://langchain-ai.github.io/langgraph) · [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT) · [Groq](https://groq.com) · [Supabase](https://supabase.com) · [Pinecone](https://pinecone.io)

Deployed on [Render](https://render.com) + [Vercel](https://vercel.com) · Entirely free infrastructure

</div>
