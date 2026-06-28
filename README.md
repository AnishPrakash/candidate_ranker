# 🏆 Redrob Intelligent Candidate Discovery & Ranking Engine
### Team Dev_NOVA — Redrob Hackathon Submission

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20Demo-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces/anishprakash/redrob-ranker)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What This Does

Given a pool of **100,000 candidate profiles**, this pipeline identifies and ranks the **top 100 best-fit candidates** for the role of *Senior AI Engineer — Founding Team* at Redrob AI.

The system is designed to defeat the adversarial traps in the dataset — keyword stuffers, behavioral twins, and chronologically impossible honeypot profiles — by combining semantic retrieval, behavioral signal analysis, and a learning-to-rank model trained to directly optimise NDCG.

---

## Architecture Overview

```
candidates.jsonl (100K)
        │
        ▼
┌─────────────────────────────────────────────────┐
│              OFFLINE PRE-COMPUTATION             │
│  (precompute.py — no time limit)                │
│                                                  │
│  1. Embed all 100K profiles → FAISS index       │
│  2. Build BM25 inverted index on skills +       │
│     career descriptions                          │
│  3. Compute tabular features → Parquet          │
│  4. Train XGBoost LambdaMART on synthetic       │
│     preference pairs                             │
└────────────────┬────────────────────────────────┘
                 │  artifacts/ (≤ 5 GB)
                 ▼
┌─────────────────────────────────────────────────┐
│           ONLINE RANKING  (rank.py)             │
│  (≤ 5 min · CPU only · 16 GB · no network)     │
│                                                  │
│  Step 1 │ Load artifacts          < 5s          │
│  Step 2 │ FAISS dense retrieval  → top 2000     │
│  Step 3 │ BM25 sparse retrieval  → top 2000     │
│  Step 4 │ Reciprocal Rank Fusion → top 1000     │
│  Step 5 │ Honeypot filter        → remove traps │
│  Step 6 │ Hard disqualifiers     → prune        │
│  Step 7 │ XGBoost LTR re-score  → top 100      │
│  Step 8 │ Deterministic NLG     → reasoning     │
│  Step 9 │ Write Dev_NOVA.csv                    │
└─────────────────────────────────────────────────┘
```

---

## Adversarial Defenses

### Honeypot Filter (`src/honeypot_filter.py`)
Detects chronologically impossible profiles using four checks:
- **Timeline impossibility** — `career_history` duration sum vs claimed `years_of_experience`
- **Founding paradox** — role duration exceeds company age
- **Skill duration paradox** — sum of `skill.duration_months` for advanced skills exceeds career length × 5
- **Education sanity** — graduation year precedes first career role start date

Any candidate triggering two or more checks is assigned score `-999` and removed before retrieval.

### Skill Authenticator (`src/skill_authenticator.py`)
Cross-validates claimed skills against career history descriptions using:

```
W_skill = α × min(1, D_career_match / D_skill_claimed) × log(1 + endorsements)
```

A backend engineer claiming 60 months of LLM fine-tuning with zero ML roles in their career history collapses to near-zero weight. This directly catches the CAND_0000001-style keyword stuffer pattern.

### Behavioral Signal Integration (`src/behavioral_signals.py`)

**Availability Index** — exponential time-decay on recruiter engagement:
```
A = recruiter_response_rate × exp(-0.01 × days_since_last_active)
```
Applied as a multiplicative modifier. A candidate inactive for 200 days with 50% response rate scores `0.5 × e^(-2) = 0.068` — effectively unreachable.

**Market Validation Index** — collaborative filtering signal from platform behaviour:
```
M = 0.3 × saved_norm + 0.2 × views_norm + 0.3 × interview_completion_rate + 0.2 × github_norm
```

---

## Repository Structure

```
candidate_ranker/
├── artifacts/                   # Pre-computed (committed to repo)
│   ├── faiss_index.bin          # FAISS IVFFlat index (100K embeddings)
│   ├── bm25_index.pkl           # BM25Okapi index on skills + career text
│   ├── candidate_features.parquet  # Tabular features for all 100K candidates
│   ├── jd_vector.npy            # JD query embedding (384-dim)
│   └── ltr_model.xgb            # Trained XGBoost LambdaMART model
│
├── src/                         # Pipeline modules
│   ├── ingest.py                # Streaming JSONL generator (memory-flat)
│   ├── honeypot_filter.py       # Chronological impossibility detector
│   ├── skill_authenticator.py   # Skill authenticity cross-validator
│   ├── behavioral_signals.py    # Availability + Market Validation indices
│   ├── embedder.py              # Sentence-transformer wrapper (all-MiniLM-L6-v2)
│   ├── hybrid_retriever.py      # FAISS + BM25 + Reciprocal Rank Fusion
│   ├── ltr_trainer.py           # XGBoost LambdaMART offline training
│   ├── reasoner.py              # Deterministic NLG template engine
│   └── jd_constraints.py       # Structured JD constraints (hard/soft/disqualifiers)
│
├── sandbox/                     # HuggingFace Spaces demo
│   ├── app.py                   # Streamlit app
│   └── requirements.txt
│
├── notebooks/
│   └── exploration.ipynb        # Data analysis on sample candidates
│
├── precompute.py                # Offline pre-computation (run once, no time limit)
├── rank.py                      # Main ranking script (≤ 5 min, CPU, no network)
├── validate_submission.py       # Format validator
├── Dev_NOVA.csv                 # Final submission (top 100 ranked candidates)
├── sample_candidates.json       # First 50 candidates (schema reference)
├── submission_metadata.yaml     # Submission metadata
├── requirements.txt             # Pinned dependencies
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/redrob-candidate-ranker
cd redrob-candidate-ranker
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pre-computation (run once — no time limit)

This builds the FAISS index, BM25 index, tabular features, and trains the LTR model. Artifacts are already committed to the repo so **you only need this if you want to rebuild from scratch**.

```bash
python precompute.py --candidates ./candidates.jsonl
```

Expected output in `artifacts/`:
```
artifacts/faiss_index.bin          (~147 MB)
artifacts/bm25_index.pkl           (~80 MB)
artifacts/candidate_features.parquet  (~12 MB)
artifacts/jd_vector.npy            (~3 KB)
artifacts/ltr_model.xgb            (~2 MB)
```

### 3. Reproduce the submission (≤ 5 minutes, CPU only)

```bash
python rank.py --candidates ./candidates.jsonl --out ./Dev_NOVA.csv
```

Expected runtime: **under 2 minutes** on a 16 GB CPU machine.

### 4. Validate the output

```bash
python validate_submission.py Dev_NOVA.csv
```

All checks must pass before submitting.

---

## Evaluation Alignment

The pipeline is explicitly tuned around the competition's composite metric:

```
Final Score = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

**NDCG@10 (50% weight)** — XGBoost trained with `rank:ndcg` objective directly optimises this. The top-10 slots are protected by hard disqualifiers that reject any candidate with honeypot signals, consulting-only careers, or availability index below 0.05.

**NDCG@50 (30% weight)** — Hybrid retrieval (FAISS + BM25 via RRF) ensures high recall at rank 50. BM25 catches exact tool names (`FAISS`, `Pinecone`, `NDCG`) that dense embeddings dilute; FAISS catches semantic matches that BM25 misses.

**MAP (15% weight)** — Every candidate in the top-100 passes a minimum utility threshold: `open_to_work_flag` considered, `recruiter_response_rate > 0.1`, and `availability_index > 0.05`.

**P@10 (5% weight)** — Micro-float tie-breaking (`score += 0.0001 × github_activity_score`) prevents alphabetical sorting from corrupting precision at top-10.

---

## Compute Constraints Compliance

| Constraint | Limit | Our system |
|---|---|---|
| Runtime | ≤ 5 minutes | ~2 minutes |
| RAM | ≤ 16 GB | ~3 GB peak |
| Compute | CPU only | ✅ CPU only |
| Network | Off | ✅ No API calls |
| Disk (artifacts) | ≤ 5 GB | ~242 MB total |

---

## Key Design Decisions

**Why FAISS + BM25 hybrid over pure dense search?**
BM25 catches exact tool names (`Qdrant`, `NDCG`, `sentence-transformers`) that dense models dilute into semantic neighbourhoods. A candidate who writes "Weaviate" gets full BM25 credit; dense search might blend them with candidates who only wrote "vector database". RRF fusion preserves both signals without manual weight tuning.

**Why XGBoost LambdaMART over a linear scoring formula?**
Linear formulas cannot capture interactions. A candidate with 8 years experience at a pure consulting firm should score below one with 5 years at a product company — even though 8 > 5 on the experience axis alone. LambdaMART learns these non-linear boundaries from preference pairs and directly optimises NDCG rather than a proxy loss.

**Why multiplicative availability, not additive?**
A technically perfect candidate who hasn't logged in for 8 months has functional hiring utility of near-zero. Additive scoring would still give them a high total. Multiplication correctly collapses their effective score: `final = base_score × availability_index`.

**Why deterministic NLG over an LLM reasoning call?**
The 5-minute compute constraint makes per-candidate LLM calls impossible at 100K scale. The NLG template engine pulls from 12+ signal-derived fact slots (github score, assessment scores, saves in 30d, offer acceptance rate, notice period, location, education tier, etc.) to generate reasoning that is specific, JD-connected, concern-honest, and zero-hallucination by construction.

---

## Team

**Dev_NOVA**

| Name | Role |
|---|---|
| Anish | ML Engineer and Developer / Team Lead |

---

## AI Tools Declaration

Used **Claude** for architectural discussion, code review, and debugging. All engineering decisions, scoring logic, pipeline design, and validation were implemented and verified by me. No candidate data was fed to any LLM during the ranking step. The ranking pipeline makes zero external API calls.