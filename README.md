# Job Search Agent

A personalized job-search agent. It loads your resume and preferences into a
persistent memory system, plans targeted searches, finds real job postings on
the internet, ranks them against your profile, and remembers what it found so
it never recommends the same job twice.

```
my_information/*.md
        │
        ▼
   ┌─────────┐   markdown → heading-scoped chunks (Qdrant, ns=profile)
   │ ingest  │
   └────┬────┘
        ▼
   ┌─────────┐   1 LLM call → search plan (queries, companies, skills)
   │ planner │
   └────┬────┘
        ▼
   ┌─────────┐   Tavily: search (budgeted, cached) → filter → extract ≤20 URLs
   │  SCOUT  │   ── the only stage that touches the network ──
   └────┬────┘
        ▼
   ┌─────────┐   SQLite (system of record) + Qdrant (vectors) + Neo4j (graph)
   │ persist │
   └────┬────┘
        ▼
   ┌─────────┐   deterministic: 0.5·vector + 0.35·skills + 0.15·preference
   │  rank   │   no LLM in the ranking path
   └────┬────┘
        ▼
     report  ── URL + description + match score + reasons
        │
        ▼
   ┌─────────┐   episodic log of plans, scouts and recommendations
   │ write-  │
   │  back   │
   └─────────┘
```

## Quick start

```bash
pip install -r requirements.txt
docker compose -f docker_compose.yml up -d

python -m src.cli doctor          # verify config + all backing services
python -m src.cli run             # full pipeline, prints a report
```

## Commands

| Command | What it does |
|---|---|
| `python -m src.cli run` | Ingest → plan → scout → rank → report |
| `python -m src.cli run --no-scout` | Re-rank stored jobs only. **0 Tavily credits.** |
| `python -m src.cli run --focus "backend, Go"` | Bias this run toward a focus |
| `python -m src.cli run --json` | Machine-readable output |
| `python -m src.cli ingest [--force]` | Load `my_information/` into memory only |
| `python -m src.cli list` | Every job found so far |
| `python -m src.cli status <job_id> applied` | Mark a job so it's never suggested again |
| `python -m src.cli stats` | Memory + database counters |
| `python -m src.cli history` | Past plans, scouts and recommendations |
| `python -m src.cli doctor` | Check config and every service |

## Memory system

Three tiers, following the HelloAgents reference architecture in `reference/`:

| Tier | Store | Holds | Used for |
|---|---|---|---|
| **Semantic** | Qdrant (1024-d) + Neo4j | Profile chunks, job postings, the knowledge graph | Retrieval and ranking |
| **Episodic** | SQLite | Every run: plan, scout result, recommendations | "What was I shown last Tuesday?" |
| **Working** | in-process, TTL 2h | Per-run scratchpad | Avoiding recomputation |

SQLite is the **system of record** (`memory_data/jobs.db`): `jobs`,
`companies`, `applications`, `search_cache`, `ingest_log`. Everything between
the Scout and the report reads only this file — which is why a Tavily outage
degrades a run instead of breaking it.

The Neo4j graph holds `(:Company)-[:POSTS]->(:Job)-[:REQUIRES]->(:Skill)`, so
you can ask relational questions a vector index can't answer:

```python
g.jobs_for_skill("python")        # every posting requiring Python
g.related_companies("Binance")    # companies sharing the most skills with Binance
g.company_job_counts()            # who is hiring the most
```

Namespaces (`profile` / `jobs`) keep the two kinds of content from bleeding
into each other's searches.

## How ranking works

Deterministic by design — ranking does not call the LLM, so it is fast,
reproducible, and debuggable.

| Signal | Weight | Source |
|---|---|---|
| Vector similarity | 0.50 | Job text vs. the centroid of your profile sections |
| Skill overlap | 0.35 | Technology terms found in both |
| Preferences | 0.15 | Location and seniority gates |

An **unknown** location is treated differently from a **wrong** one: a posting
that never states its city is scored neutrally, while one that states another
country is penalised. Each result carries `reasons` explaining its score.

Tune the weights in `.env`-independent config (`src/config.py` → `AgentConfig`).

## Cost control

Tavily is the only metered dependency.

- **Cache-first** — responses are cached in SQLite for 24h, keyed by query
  hash. A repeated query costs **0 credits**.
- **Budget cap** — `max_searches_per_run` (default 6) and a derived extract
  cap. When the budget is spent the Scout stops, it does not overspend.
- **Batched extract** — up to 20 URLs per call, Tavily's maximum.
- **Dedupe before extract** — listings are filtered out first, so credits are
  not spent fetching pages that contain no vacancy.
- **Board-index rejection** — company boards (`jobs.lever.co/binance`) and
  aggregator listing pages (`695 Graduate Jobs in Hong Kong`) are dropped
  before extraction; they describe no single role.

Typical run: **~7 credits**. Researcher's free tier is 1,000/month.

## Configuration

All settings live in `.env`:

```
LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_ID     DeepSeek
EMBED_API_KEY / EMBED_BASE_URL / EMBED_MODEL_NAME   DashScope text-embedding-v3
TAVILY_API_KEY                                  Tavily search + extract
QDRANT_URL / QDRANT_COLLECTION / QDRANT_VECTOR_SIZE
NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD
```

Two provider quirks this code works around, both verified against the live APIs:

1. **`deepseek-v4-flash` is a reasoning model.** It emits `reasoning_content`
   before the answer, and reasoning tokens are billed against `max_tokens`.
   A small budget returns an **empty** response. The client enforces a 1024
   token floor and retries with a larger budget when content comes back blank.
2. **DashScope caps embedding batches at 10 inputs.** Larger batches fail with
   `InvalidParameter`. `DashScopeEmbedder` chunks at that limit.

## Preventing duplicate applications

Mark a job once you have applied:

```bash
python -m src.cli status <job_id> applied      # also: rejected | dismissed | saved
```

Statuses `applied`, `rejected` and `dismissed` are **closed** — they are
excluded from every future report, and the status survives re-discovery of the
same posting. `python -m src.cli list` shows the id and current status of
everything found.

## Project layout

```
src/
  config.py            settings from .env
  cli.py               command line interface
  agent/
    ingest.py          markdown → heading-scoped chunks
    planner.py         LLM → search plan
    matcher.py         deterministic ranking
    report.py          output rendering
    pipeline.py        orchestration
  memory/
    semantic.py        Qdrant + Neo4j
    episodic.py        run history
    working.py         per-run scratchpad
    embedding.py       DashScope text-embedding-v3
    storage/           Qdrant / Neo4j adapters
  scout/
    tavily_scout.py    the only network boundary, budgeted + cached
    parse.py           web text → structured postings
  domain/
    models.py          JobPosting, SearchPlan, RankedJob
    sqlite_store.py    system of record
my_information/        your resume and preferences (markdown)
memory_data/           created at runtime
reference/             HelloAgents reference code — not imported at runtime
```

## Notes and limitations

- **ATS-targeted queries matter.** Generic queries surface aggregator pages
  that are login-walled and contain no vacancy. The planner is instructed to
  use `site:` filters against Greenhouse, Lever, Ashby and Workable; this was
  the single largest quality difference measured during development.
- **Location detection is region-level**, not city-level, and reads only the
  top of a posting. Postings that never state a location rank neutrally.
- **Company names come from the URL** (ATS path slugs) when the page does not
  state one clearly, so some are lightly normalised (`Valkyrietrading`).
- **Salary strings are regex-extracted** and occasionally pick up an unrelated
  figure from the body text. Treat them as hints, not facts.
- The reference NER pipeline (spaCy entity extraction) is not wired in; the
  graph is populated from structured fields instead, which is more reliable
  for job data.
