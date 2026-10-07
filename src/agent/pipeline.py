"""Planner-Executor pipeline with Scout isolation.

    ingest -> plan -> [SCOUT: search + extract] -> persist -> rank -> report -> write-back

The Scout is the only stage that touches the network, and it is bounded by a
credit budget. Everything between Scout and report reads SQLite and the memory
stores, so a Tavily outage degrades the run instead of breaking it.

Ranking is deterministic (`matcher.py`); the LLM is used exactly once, for the
plan. That keeps a run to ~1 LLM call and a handful of Tavily credits.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import AgentConfig, Settings, get_settings
from ..domain.models import (
    STATUS_DISCOVERED,
    JobPosting,
    RankedJob,
    SearchPlan,
)
from ..domain.sqlite_store import SQLiteStore
from ..memory import MemoryManager
from ..scout import ScoutBudget, TavilyScout, candidate_from_search_hit, parse_page
from ..scout.parse import is_listing_or_aggregator
from .ingest import Ingestor
from .matcher import Matcher, Profile, build_profile_from_text
from .planner import Planner
from .report import render_report

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Everything a caller (CLI or test) needs to know about one run."""

    plan: Optional[SearchPlan] = None
    ranked: List[RankedJob] = field(default_factory=list)
    new_jobs: List[JobPosting] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    report: str = ""
    duration_seconds: float = 0.0
    degraded: bool = False
    notes: List[str] = field(default_factory=list)


class JobSearchAgent:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.agent_config: AgentConfig = self.settings.agent

        self.memory = MemoryManager(
            storage_path=self.agent_config.storage_path,
            user_id=self.agent_config.user_id,
        )
        self.db: SQLiteStore = self.memory.sqlite
        self.ingestor = Ingestor(self.memory, self.agent_config.info_dir)
        self.planner = Planner(self.memory)
        self.matcher = Matcher(
            weight_vector=self.agent_config.weight_vector,
            weight_keyword=self.agent_config.weight_keyword,
            weight_preference=self.agent_config.weight_preference,
        )

    # ------------------------------------------------------------------- run

    def run(
        self,
        focus: Optional[str] = None,
        limit: Optional[int] = None,
        scout: bool = True,
        include_applied: bool = False,
    ) -> RunResult:
        started = time.time()
        result = RunResult()
        top_k = limit or self.agent_config.top_k_results

        # 1. Ingest the user's markdown into memory (idempotent).
        ingest_report = self.ingestor.ingest_all()
        logger.info(
            "Ingest: %s chunk(s) from %s (skipped unchanged: %s)",
            ingest_report["chunks_added"],
            [f["name"] for f in ingest_report["files"]] or "nothing",
            ingest_report["skipped"] or "none",
        )

        profile_text = self.ingestor.profile_summary()
        self.memory.set_scratch("profile_text", profile_text)
        profile = build_profile_from_text(profile_text)

        if not profile_text:
            result.notes.append(
                "No profile found in my_information/ - ranking relies on preferences only."
            )

        # 2. Plan. One LLM call.
        plan = self.planner.plan(focus=focus, max_queries=self.agent_config.max_searches_per_run)
        result.plan = plan
        self.memory.record_event(
            "plan",
            f"Planned {len(plan.queries)} queries: {'; '.join(plan.queries[:3])}",
            {"queries": plan.queries, "rationale": plan.rationale},
        )

        # 3. Scout (network-isolated, budget-bounded).
        budget = ScoutBudget(
            max_searches=self.agent_config.max_searches_per_run,
            max_extracts=max(1, self.agent_config.max_searches_per_run // 3),
        )
        new_jobs: List[JobPosting] = []
        if scout:
            new_jobs = self._scout(plan, budget, result)
        else:
            result.notes.append("Scout disabled; ranking previously discovered jobs only.")

        # 4. Rank everything we know, so history improves each run.
        pool = {job.job_id: job for job in self.db.all_jobs()}
        for job in new_jobs:
            pool[job.job_id] = job

        ranked = self.matcher.rank(
            list(pool.values()), profile, include_closed=include_applied
        )
        result.ranked = ranked[:top_k]
        result.new_jobs = new_jobs

        # Persist the score so later runs can show history.
        for item in result.ranked:
            self.db.set_match_score(item.job.job_id, item.score)

        # 5. Report.
        result.report = render_report(
            result.ranked,
            plan=plan,
            budget_summary=budget.summary(),
            profile=profile,
            ingest_report=ingest_report,
            degraded=result.degraded,
        )

        # 6. Write back: this run becomes an episode, so a future run knows
        #    what was already shown (prevents re-recommending the same thing).
        self._write_back(result)

        result.stats = {
            "ingest": ingest_report,
            "jobs_in_pool": len(pool),
            "new_jobs": len(new_jobs),
            "scout": budget.summary(),
            "memory": self.memory.get_stats(),
            "db": self.db.stats(),
        }
        result.duration_seconds = round(time.time() - started, 2)
        result.stats["duration_seconds"] = result.duration_seconds
        return result

    # ----------------------------------------------------------------- scout

    def _scout(self, plan: SearchPlan, budget: ScoutBudget, result: RunResult) -> List[JobPosting]:
        """Search, filter, extract, parse, persist. The only network stage."""
        try:
            scout = TavilyScout(cache=self.db, budget=budget)
        except Exception as exc:
            logger.error("Scout unavailable: %s", exc)
            result.degraded = True
            result.notes.append(f"Scout unavailable: {exc}")
            return []

        # -- search -----------------------------------------------------------
        candidates: Dict[str, JobPosting] = {}
        rejected_boards = 0
        for query in plan.queries:
            if budget.searches_left <= 0:
                result.notes.append("Search budget exhausted before all queries ran.")
                break
            try:
                response = scout.search(query)
            except Exception as exc:
                logger.error("Search failed for %r: %s", query, exc)
                result.notes.append(f"Search failed: {query[:50]}")
                continue

            for hit in response.get("results") or []:
                url = hit.get("url") or ""
                title = hit.get("title") or ""
                # Board indexes and aggregator listing pages describe no single
                # vacancy; keeping them yields fake companies and empty roles.
                if is_listing_or_aggregator(title, url):
                    rejected_boards += 1
                    continue
                candidate = candidate_from_search_hit(hit)
                if candidate is None:
                    continue
                # Keep the richer version if two queries surface one posting.
                existing = candidates.get(candidate.url)
                if existing is None or len(candidate.description) > len(existing.description):
                    candidates[candidate.url] = candidate

        logger.info(
            "Scout: %s posting candidates (%s board indexes skipped)",
            len(candidates), rejected_boards,
        )

        if not candidates:
            result.notes.append(
                "No individual postings found - results were listing pages or "
                "login-walled pages. Try re-running with a different focus."
            )
            return []

        # -- extract + parse --------------------------------------------------
        contents = scout.extract(list(candidates)[: self.settings.tavily.extract_max_urls])
        jobs: List[JobPosting] = []
        for url, content in contents.items():
            hint = candidates[url]
            job = parse_page(
                url, content, hint_title=hint.title, hint_company=hint.company
            )
            if job is None:
                continue
            # Prefer the hint's company when the page gave us a junk one.
            if hint.company and job.company in ("", "Unknown"):
                job.company = hint.company
            jobs.append(job)

        # Any candidate we could not extract still has snippet-level detail.
        for url, hint in candidates.items():
            if url not in contents and len(hint.description) > 150:
                jobs.append(hint)

        logger.info("Scout: parsed %s posting(s)", len(jobs))

        # -- persist ----------------------------------------------------------
        counts = self.db.upsert_jobs(jobs)
        stored: List[JobPosting] = []
        for job in jobs:
            # Only index genuinely new postings into the vector store.
            existing = self.db.get_job(job.job_id)
            if existing is None or existing.status == STATUS_DISCOVERED:
                self.memory.add_job(job)
            stored.append(job)

        logger.info("Scout: stored %s new, %s updated, %s duplicate",
                    counts["new"], counts["updated"], counts["duplicate"])

        for job in jobs:
            # Store the company once; a per-posting `url` here would overwrite
            # it on every subsequent posting from the same employer.
            self.db.upsert_company(
                job.company,
                location=job.location,
                url=job.url if job.company.lower().replace(" ", "") in job.url.lower() else "",
            )

        self.memory.record_event(
            "scout",
            f"Scouted {len(jobs)} postings ({counts['new']} new) via {len(plan.queries)} queries",
            {"queries": plan.queries, "counts": counts},
        )
        return stored

    # ------------------------------------------------------------- write-back

    def _write_back(self, result: RunResult) -> None:
        """Record this run so the next one can avoid repeating it."""
        if not result.ranked:
            return
        shown = [
            {"job_id": r.job.job_id, "title": r.job.title, "company": r.job.company,
             "url": r.job.url, "score": r.score}
            for r in result.ranked
        ]
        self.memory.record_event(
            "recommendation",
            f"Recommended {len(shown)} job(s): "
            + "; ".join(f"{s['title']} @ {s['company']}" for s in shown[:5]),
            {"jobs": shown},
        )
