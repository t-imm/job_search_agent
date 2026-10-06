# Personalized Job Search AI Agent — Design Summary

> **Working name:** *JobPilot Local*
> **Source:** `plan/output.txt` (Camel-AI role-playing run: Chief Solution Architect → Enterprise AI Architect, 5 turns)
> **Goal:** A personalized job-search agent that discovers highly relevant jobs while minimizing API spend, cloud dependencies, and infrastructure — so the design can be defended to an Enterprise AI Architect.

---

## 1. Problem & Constraints

| Constraint | Meaning |
|---|---|
| **Zero cloud** | All inference and storage runs locally (workstation / edge). No hosted vector DB, no per-call memory vendor. |
| **Near-zero API cost** | Tavily is the *only* cost-bearing component, used sparingly for live job data — never for memory. |
| **Persistent profile** | User profile and application history survive across sessions. |
| **Real currency** | Not dollars, but: **number of local LLM inference passes + number of Tavily calls + wall-clock latency + human debugging effort.** |

---

## 2. Memory Systems — compared, then narrowed

### 2.1 Three memory types required

| Memory type | Holds | Access pattern | Primitive needed |
|---|---|---|---|
| **Structured profile** | Skills, titles, seniority, salary floor, locations, work-auth, constraints, application log | Exact/relational filters, updates | RDBMS tables + JSON |
| **Semantic / episodic** | Resume chunks, past JDs, cover-letter snippets, prior feedback | Approximate nearest-neighbor ("roles like this") | Vector index + keyword |
| **Relational / temporal** | "Remote *until* Q3", skill→role→company edges, preference drift | Multi-hop + time-aware traversal | Graph edges with validity |

> This mapping is the key design decision — it avoids the classic failure of forcing profile rules into a vector store.

### 2.2 Candidates evaluated

**Vector stores**

| System | Footprint | Offline | Verdict |
|---|---|---|---|
| **sqlite-vec** | Disk 1M ~6.0 GB; RSS 100K ~50 MB | Full (ships inside SQLite) | ✅ **Top candidate** — one file holds vector + relational + FTS5 |
| **LanceDB** | Disk 1M ~2.8 GB (~800 MB w/ PQ); RSS 100K ~120 MB | Full, in-process | ✅ Strong alternative if vectors dominate / >1M scale |
| ChromaDB | RSS 100K ~800 MB → 1M ~6.5 GB | Full in embedded mode | ⚠️ RAM blow-up; anti-scales (p99 up to 13 s) |
| FAISS | RAM only, manual serialization | Full | ⚠️ Library, not a DB — no persistence/ACID; use as accelerator only |
| Qdrant | Server-resident | Self-host, **not embedded** | ⚠️ Adds a daemon |
| Milvus | M2.25M on disk = 17 GB | Heavy | ❌ Overkill |

**Embedded / graph memory**

| System | Verdict |
|---|---|
| **Graphiti + FalkorDB** (temporal edges, lite/embedded mode) | ✅ Best if time-aware preferences are core — but LLM extraction per episode is a cost risk |
| **FalkorDB standalone** (Redis-based) | ✅ Pragmatic graph backend |
| Kuzu (embedded, ideal fit) | ❌ **Archived Oct 2025** after Apple acquisition — maintenance risk |
| Memgraph / Neo4j | ⚠️ Servers, not embedded |
| **DIY edge table (NetworkX / SQLite)** | ✅ Best *zero-infra* graph substitute — negligible footprint |

**Structured profile memory**

| System | Verdict |
|---|---|
| **SQLite (relational + JSON1 + FTS5)** | ✅ **Backbone of the design** — single ACID file |
| DuckDB | ✅ Great for analytics over history; not the OLTP profile store |
| memweave pattern (Markdown = source of truth) | ✅ Good for auditable profile notes |
| Postgres + pgvector | ⚠️ Best hybrid engine, but a server to run/size/backup |
| **Mem0 / Letta / Zep** | ❌ As-is these call an LLM on **write *and* read** → violates low-cost + no-cloud. Use as *patterns*, not dependencies. Zep CE deprecated 2025. |

### 2.3 Cross-cutting enablers
- **Local embeddings make "never for memory" airtight:** `nomic-embed-text` (274 MB, 62.28 MTEB) or `qwen3-embedding:0.6b` (639 MB, 64.33) via Ollama — embedding generation never touches an API.
- **Proven single-file hybrid pattern:** embed → `sqlite-vec` KNN → FTS5 BM25 → **Reciprocal Rank Fusion (RRF)** → rerank, all in one `.db` you can copy, back up, or inspect.
- **Footprint reality:** a single user's corpus (<100K vectors) is trivial — ~50 MB RSS (sqlite-vec) or ~120 MB (LanceDB) — next to nothing compared with the local LLM's own RAM.

**Decision → single-file SQLite core:** relational profile tables + FTS5 + `sqlite-vec` (hybrid RRF) + lightweight graph edge table, with local Ollama embeddings.

---

## 3. Agent Architectures — compared and scored

### 3.1 The three candidates

| | **A. ReAct tool-calling loop** | **B. Planner–Executor** | **C. Multi-agent role pipeline** |
|---|---|---|---|
| **Control flow** | User → thought → tool → observe → loop until model decides it's done | Planner emits explicit plan → deterministic executor runs steps → synthesizer answers → optional replan | Profiler → Scout → Matcher → Tailor → Critic, orchestrated by code |
| **LLM passes / request** | 4–9 (one per loop iteration, growing scratchpad) | **2–4, independent of task length** | 4–6 (one per role, narrow prompts) |
| **Tavily behaviour** | 1 call per scout iteration → re-fetch thrash | **Batched inside a planned step → fewest calls** | **Isolated to Scout → easiest to enforce** |
| **Token profile** | Grows ~linearly (full scratchpad re-sent each turn) | **Lowest** — expensive context built once, then code | Moderate — small per-role contexts |
| **Key strengths | Simplest, fastest to build, great MVP | Predictable, **plan is a replayable artifact**, unit-testable with no LLM | Clean separation of concerns, narrow roles → better on small models |
| **Key weaknesses** | Non-termination, thrashing, context overflow, silent profile drift | More engineering (plan schema, executor, verifier); brittle if task shape varies | Most moving parts, highest latency, over-engineering risk for one user |
| **Offline fit** | Highest conceptual simplicity | Strong — templatable, repeatable domain | Needs graceful degradation |

### 3.2 Scorecard (1 = bad, 5 = excellent)

| Criterion | A: ReAct | B: Planner–Executor | C: Role pipeline |
|---|:--:|:--:|:--:|
| Latency (lower better) | 3 | **4** | 3 |
| Debuggability | 3 | **5** | 4 |
| Offline resilience | **4** | **4** | 3 |
| Minimize API calls | 2 | 4 | **5** |
| Implementation effort (lower better) | **5** | 3 | 2 |
| Model-quality tolerance (small LLM) | 2 | 4 | **5** |
| Token volume on complex tasks | 2 | **5** | 4 |
| **TOTAL** | **21** | **29** | **26** |

**Reading:** B wins overall because job search is *templatable*, giving the fewest LLM passes and fewest Tavily calls with the best debuggability. C's Scout-isolation idea is worth **borrowing** even without adopting full multi-agent orchestration. A is the throwaway **spike/MVP**.

### 3.3 Cross-architecture controls (apply regardless of choice)
1. **Cache-first Tavily wrapper** — the single biggest saver; SHA-256 of the normalized query → SQLite `job_cache` → TTL hit = 0 API calls.
2. **Hybrid retrieval as one local function** — FTS5 BM25 → sqlite-vec KNN → RRF; deterministic, offline, loggable.
3. **Deterministic layers instead of prompts** — ranking, filtering, dedup, freshness, constraint matching belong in code. Every move from LLM to code removes latency and failure surface.
4. **Graceful degradation matrix** — Tavily unreachable → serve stale from cache; Ollama down → pre-computed rankings; graph unavailable → flat profile filters. Never hard-fail.
5. **Bounded loops + schema-typed artifacts** — hard caps on iterations/replans; JSON-schema validation on plans and inter-stage contracts.

---

## 4. Tool Layer

### 4.1 Tavily (the only paid component)

| Aspect | Detail |
|---|---|
| Endpoints | `POST /search` (basic/advanced depth), plus `Extract`, `Map`, `Crawl`, `Research` |
| Credit cost | basic search = **1 credit**; advanced = 2. Extract = 1 credit / 5 successful URLs (2 for advanced) |
| **Batching primitive** | **Extract accepts up to 20 URLs per call** |
| Rate limits | Dev 100 RPM, Prod 1,000 RPM — far above single-user need |
| Gotcha | HTTP 200 can still contain `failed_results`; check both fields |

**Pricing:** Researcher (free) = **1,000 credits/month**; pay-as-you-go = **$0.008/credit**; Project = 4,000 credits/month; Enterprise = custom.

**Cost reality check:** 2 basic searches/day + 20 JD extracts/day ≈ **180 credits/month** — comfortably inside the free tier. The danger is *unbounded agentic search* (the "20 searches/task" anti-pattern ≈ $0.16/task).

**Scout policy:** one `search` (basic) → one batched `extract` (≤20 URLs) → cache-first by query hash (12–24 h TTL) → dedupe by URL/`job_id` before extract → persist JDs so re-ranking and tailoring never re-call Tavily.

### 4.2 Job-data alternatives

| Source | Interface | Cost | Risk | Verdict |
|---|---|---|---|---|
| **ATS public APIs** — Greenhouse `boards-api.greenhouse.io`, Lever `api.lever.co`, Ashby `api.ashbyhq.com`, SmartRecruiters, Recruitee | Plain HTTP GET → JSON → normalize → upsert | **$0, no key, no proxy** | Low | ✅ **Primary** — same endpoints employer career pages call |
| **Himalayas public JSON feed** | REST GET, keyless, feed-shaped | $0 | Low | ✅ Excellent for offline cache seeding / breadth |
| Personio / Recruitee XML feeds | RSS/XML | $0 | Low (coverage is employer-dependent) | ✅ Supplement |
| **JobSpy (python-jobspy)** | `scrape_jobs()` → DataFrame; Indeed/LinkedIn/Glassdoor/Google | "Free" — but hidden residential-proxy costs | **High & fragile**; LinkedIn throttles unauthenticated scraping | ⚠️ Optional cache-warmer only, never critical path |
| Google Jobs | Only via scrapers/paid search API | — | Not ToS-clean | ⚠️ Fallback enrichment only |

**Source strategy:** ATS keyless APIs + one feed as primary → Tavily for the long tail / semantic discovery → JobSpy as an optional offline warmer.

### 4.3 Local utilities (all $0, fully offline)

| Utility | Libraries |
|---|---|
| Resume parsing | `pdfplumber` / `pypdf` + `python-docx`; or a local Qwen2.5 ResumeParser |
| ATS keyword matching | Resume-Matcher pattern, `rapidfuzz`, `scikit-learn`, skills taxonomy JSON — **deterministic, not LLM** |
| Semantic matching | `sqlite-vec` + Ollama embeddings + FTS5 RRF |
| Document generation | `python-docx` (DOCX), `ReportLab` (PDF), `open-resume` (ATS-readable HTML) |
| Feed/ATS fetching | `httpx` / `requests` + Pydantic schemas |
| Orchestration / state | Python + `apscheduler` + SQLite |

### 4.4 Recommended stack

```
SCOUT  (only cost-bearing, web-touching)
  1. Cache check   -> SQLite.job_cache (hash(query))              [free]
  2. Primary pull  -> ATS keyless APIs + Himalayas feed           [free, no key/proxy]
  3. Fallback      -> Tavily search (basic, batched, cached)      [1 cr]
  4. Content pull  -> Tavily Extract, <=20 URLs/call              [1 cr / 5 URLs]
  5. Upsert        -> SQLite jobs table (dedupe by url/job_id)
     (everything downstream reads SQLite -- NEVER Tavily again)

MATCHER (deterministic, offline)
  FTS5 BM25 -> sqlite-vec KNN -> RRF merge -> rule-based scoring
  + ATS keyword match (rapidfuzz) + graph edge lookups

TAILOR / RENDER (local LLM + local libs)
  Ollama (Qwen3-class) -> tailored bullets
  python-docx / ReportLab / open-resume -> DOCX + PDF + ATS-HTML
```

---

## 5. End-to-End Workflow

`Trigger → (det) Profile Load → [LLM Planner] → {Scout: Tavily/ATS} → (det) Matcher → [LLM Tailor] → [LLM Verify?] → (det) Render → Write-back`

| # | Stage | Memory R/W | LLM pass | Tool calls | Artifact |
|---|---|---|---|---|---|
| 0 | **Scheduler / Intake** (cron 07:00 or button) | — | none | none | `runs` record |
| 1 | **Profile Load** | R: `profile`, `graph_edges`, `constraints` | none (deterministic) | none | in-memory `ProfileContext` |
| 2 | **Planner** | R: `plan_templates` | **Pass 1** (~200–350 out-tok) | none | plan JSON in `runs.plan` |
| 3 | **Scout** — only web-touching stage | R/W: `job_cache`, `companies`, `jobs` | none | ATS keyless + Himalayas (primary) → Tavily search + Extract ≤20 URLs (cache miss / long tail only) | `jobs`, `job_cache`, `companies` |
| 4 | **Matcher** — deterministic | R: `jobs`, `sqlite-vec`, `fts5`, `graph_edges` | none (code, not prompts) | none | `matches` (scored, ranked, deduped) |
| 5 | **Tailor** | R: `profile`, top-K `matches` | **Pass 2** (~600–900 out-tok) | none | `drafts` + DOCX |
| 6 | **Verify / Critic** (optional, skippable) | R: `drafts`, JD keywords | **Pass 3** (~150–300 tok) | none | verdict + diff |
| 7 | **Render** | R: `drafts` | none | python-docx / ReportLab / open-resume | PDF + DOCX + ATS-HTML |
| 8 | **Memory Update** (write-back) | W: `applications`, `graph_edges`, `sqlite-vec`, `profile` deltas | none | local embeddings | updated single `.db` |

**Key invariant:** Tavily and the LLM appear only in *isolated, bounded* stages. Every stage between them reads/writes SQLite only. That is what keeps the cost surface tiny and the failure surface containable.

### Steady-state budget per run

| Dimension | Value |
|---|---|
| **Local LLM passes** | **2–3** (max 4) — Planner, Tailor, optional Verify |
| **Output tokens** | ~800–1,550 per run |
| **Tavily credits** | ~0–60 (warm cache) · ~180 (moderate) · ~540 (guarded upper bound) vs **1,000 free/month → $0** |
| **Wall clock** (7–8B Q4 @ 30–60 tok/s) | deterministic <1–2 s · LLM 25–55 s · network 2–15 s → **~30–75 s total** |
| **RAM** | Ollama 7–8B ~5–8 GB + embeddings 300–600 MB + sqlite-vec ~50 MB + app ~200–400 MB → **~6–9 GB total** |
| **Disk** | **~6–10 GB** (a single user's 10K–50K vector corpus = tens of MB) |

**Prompt-token optimization:** relevance-scoped injection (top-K + RRF, ~5 memories) instead of context stuffing (~24 memories) yields roughly a **72% prompt-token reduction** by design.

---

## 6. Persuasion Brief — answering the boss

### Recommendation
Build the Planner–Executor local agent on a **single-file SQLite core**, with **ATS/feed-primary + Tavily-fallback** retrieval and a **deterministic Matcher** — adopting the *patterns* of frameworks without their cost and dependency burden.

### Objection 1 — "Why not LangGraph / CrewAI / full multi-agent?"
**Framework *patterns* yes; framework *dependencies* and token multiplier no.**
- Anthropic's production multi-agent system beat single-agent by 90.2% — but at **~15× the tokens** of chat (agents alone ~4×). Multi-agent only wins when the task decomposes into **independent, parallel** threads; token usage alone explained 80% of performance variance. Job search is a **sequential, single-context** pipeline where every stage shares the same profile — you pay the multiplier without earning the parallelism.
- Benchmark deployments: LangGraph is cheapest per task (explicit structure eliminates redundant LLM calls); CrewAI hierarchical costs ~30% more than sequential; AutoGen without termination caps costs 2×; the same output ranged 2.7× across frameworks ($63 vs $171/mo). Planner–Executor *is* explicit structure — LangGraph's cost advantage without the 3–7-day production surface.
- **Compromise:** keep a thin **orchestration adapter seam** so LangGraph can drop in later for durable checkpointing / human-in-the-loop **without a rewrite**.

### Objection 2 — "Why DIY SQLite instead of pgvector / Mem0 / Zep?"
- **pgvector/Postgres:** performant and ACID, but it is a server to run, size, tune, back up, and monitor. For <100K vectors, sqlite-vec at ~50 MB RSS is sufficient, ACID, and one portable file.
- **Mem0 / Zep:** both run **LLM calls on the memory write path**, reintroducing the exact LLM-in-the-loop cost being avoided. Commercially: Mem0 $19–$249/mo, Zep $25/mo+ with the free tier deprecated (2026), and **Zep Community Edition was deprecated in 2025**. Mem0 self-hosted keeps memories in a Qdrant black box you cannot grep or edit.
- **Steal the good idea, not the dependency:** relevance-scoped memory injection (top-K + RRF) → the measured ~72% token saving.
- **Not a dead end:** because it is plain SQL, the schema migrates to Postgres+pgvector unchanged.

### Objection 3 — "How do you guarantee API cost stays near zero over 12 months?"
**It's an architectural guarantee, not a policy.**
1. **Single cost surface** — Tavily only; everything else is free.
2. **Four dampeners** — basic depth (1 cr) + batching (search = 1 call, extract ≤20 URLs) + TTL caching + dedupe → ~180 credits/mo vs 1,000 free; even the guarded upper bound (~540) stays inside the free tier.
3. **Steady-state trend to zero** once ATS + feed keep the cache warm.
4. **Circuit breakers** — a monthly credit counter; when the budget is hit, Scout serves cache and refuses paid calls. It degrades, it never surprises.
5. **Provider swappability** — Tavily sits behind a `SearchProvider` interface; swap to Brave / Perplexity / self-hosted SearXNG with zero pipeline changes.

### Objection 4 — "Is this maintainable/extensible at enterprise scale?"

| Concern | Seam | Enterprise path |
|---|---|---|
| More users | `tenant_id` on every table | Multi-tenant SQLite → Postgres+pgvector |
| More vectors / concurrency | SQL-backed access layer | Swap store behind the same interface |
| Durable orchestration | Orchestration adapter | Drop in LangGraph |
| Model/provider change | `EmbeddingProvider`, `LLMProvider` interfaces | Swap Ollama → hosted or larger local |
| Quality regression | Deterministic stages unit-testable with **no LLM**; LLM stages use a golden set | CI regression harness |
| Ops cost | **No memory server** | Add only when metrics demand |

**TCO:** ships as one process and one file — no server fleet, no per-call memory vendor, bounded dependency surface. That is *more* maintainable, not less, because there is less to operate.

### Rejected alternatives at a glance

| Category | **Chosen** | Rejected A | Rejected B | Decisive evidence |
|---|---|---|---|---|
| **Memory** | Single-file SQLite (kw + vector + graph) | pgvector/Postgres | Mem0/Zep | Server/ops overhead; LLM-on-write cost; Zep CE deprecated 2025 |
| **Architecture** | Planner–Executor + Scout isolation | ReAct loop | Multi-agent pipeline | ReAct = Tavily thrash + context re-send; multi-agent = ~15× tokens without parallelism payoff |
| **Tools** | ATS/feed primary, Tavily fallback | JobSpy-heavy scraping | Cloud job APIs | JobSpy = blocking + hidden proxy cost; cloud APIs = the exact dependency being avoided |

---

## 7. Exec Summary (one slide)

> **JobPilot Local — Zero-Cloud, Near-Zero-Cost Job Search Agent**
>
> - **What:** A personal job-search AI running entirely on the workstation (Ollama + one SQLite file), with free keyless ATS/feed APIs as the primary source and Tavily as the only possible cost.
> - **How:** `Plan → Scout (isolated web) → Deterministic Match → Tailor → Verify → Render → Write-back`, cache-first throughout.
> - **Budget per run:** 2–3 local LLM passes (~1.5K tokens), 0–2 Tavily credits, ~30–75 s, ~6–9 GB RAM. **Monthly: ~180 / 1,000 free credits → $0.**
> - **Why not frameworks:** multi-agent gains cost ~15× tokens and only pay off under parallel independent threads; job search is sequential and single-context. Adopt the *patterns*, not the *multiplier*.
> - **Why this memory:** ~50 MB @100K vectors, ACID, one portable file — no server, no per-call vendor.
> - **Cost control is architectural:** single cost surface + batching + TTL cache + dedupe + circuit breakers + swappable provider.
> - **Scale path:** same interfaces migrate to Postgres+pgvector and LangGraph when metrics demand — no rewrite.
>
> **The ask:** approve a 2-week spike to validate Planner–Executor on real hardware (measure actual LLM passes, Tavily hit-rate, and latency against this budget).

---

## 8. Risks & Open Validations

1. **Model capability** — confirm the local model reliably emits structured plans (requires a tool-calling Qwen3-class model).
2. **Memory freeze** — confirm SQLite edge table vs Graphiti/FalkorDB-lite before locking the memory layer.
3. **Live facts to re-verify at implementation time** — Tavily free-tier terms/pricing, and ATS board-token coverage (how many target employers disable their public board).
4. **Document risk** — test generated resume/PDF against a real ATS parser; keep layouts simple single-column.
5. **Spike validation** — measure real LLM pass count, Tavily hit-rate, and latency on actual hardware against the ~2–4 pass budget.

**Two data-quality flags carried through the research:** Kuzu was archived Oct 2025 (Apple acquisition); Zep Community Edition was deprecated 2025 (self-host = Graphiti engine only).