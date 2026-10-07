"""The Planner: decides what to look for, from memory.

This is the only LLM call in the discovery path. It reads the *stored* profile
(not the raw files), so it reasons over exactly what the matcher will use.

Query construction leans on ATS-specific search syntax. That is not a stylistic
choice: generic queries surface aggregator listing pages, which are login-walled
and yield no usable posting, whereas `site:jobs.lever.co` returns one URL per
real vacancy. Verified against the live API during development.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..domain.models import SearchPlan
from ..llm import LLMError, get_llm_client
from ..memory import MemoryManager

logger = logging.getLogger(__name__)

#: ATS boards that expose one page per vacancy and are readable without auth.
ATS_DOMAINS = [
    "jobs.lever.co",
    "boards.greenhouse.io",
    "jobs.ashbyhq.com",
    "apply.workable.com",
]

SYSTEM_PROMPT = """You are a job-search strategist. You plan web searches for a candidate.

Given their profile and preferences, produce a JSON object:
{
  "queries": ["<search string>", ...],
  "target_companies": ["<company>", ...],
  "focus_skills": ["<skill>", ...],
  "rationale": "<one sentence>"
}

Rules for "queries":
- 4 to 6 queries.
- Each must be a real web search string a person could type.
- Most queries must target applicant-tracking sites, because company career
  portals are often login-walled. Use this form:
  <role keywords> site:jobs.lever.co OR site:boards.greenhouse.io <location>
- Include at most one broad query without a site: filter, to catch listings
  that are not on an ATS.
- Prefer the candidate's target seniority and location.
- Do not invent company names unless you are confident the company operates in
  their target city and hires for their skill set.
- Keep each query under about 120 characters.

"target_companies" should hold 0 to 5 real employers worth watching in the
candidate's city. "focus_skills" should hold 5 to 10 concrete technologies.

Return only the JSON object."""

#: Used when the LLM is unavailable - the run degrades instead of failing.
FALLBACK_QUERIES = [
    "graduate software engineer Hong Kong site:jobs.lever.co OR site:boards.greenhouse.io",
    "junior backend engineer Hong Kong site:jobs.lever.co OR site:jobs.ashbyhq.com",
    "software engineer intern Hong Kong site:apply.workable.com",
    "entry level software engineer Hong Kong site:boards.greenhouse.io",
]


class Planner:
    def __init__(self, memory: MemoryManager) -> None:
        self.memory = memory
        self.llm = get_llm_client()

    def plan(self, focus: Optional[str] = None, max_queries: int = 6) -> SearchPlan:
        profile_text = self.memory.get_scratch("profile_text", "") or self._load_profile_text()
        if not profile_text:
            logger.warning("No stored profile found; planning from preferences alone")

        context = self._build_context(profile_text, focus)

        try:
            payload = self.llm.chat_json(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                max_tokens=2048,
                temperature=0.4,
            )
            plan = self._coerce_plan(payload, max_queries)
            if plan.queries:
                logger.info("Planner produced %s queries", len(plan.queries))
                return plan
            logger.warning("Planner returned no usable queries; using fallback")
        except LLMError as exc:
            logger.error("Planner failed (%s); using fallback queries", exc)

        return SearchPlan(
            queries=FALLBACK_QUERIES[:max_queries],
            focus_skills=[],
            rationale="Fallback plan: the planner was unavailable.",
        )

    def _load_profile_text(self) -> str:
        items = self.memory.search_profile("skills experience education preferences", limit=30)
        if not items:
            return ""
        return "\n\n".join(
            f"{i.metadata.get('section', '')}\n{i.content}".strip() for i in items
        )

    def _build_context(self, profile_text: str, focus: Optional[str]) -> str:
        parts = ["## Candidate profile", profile_text[:6000] or "(no profile stored)"]
        if focus:
            parts += ["", "## Extra focus for this run", focus]

        # Nudge the planner toward ATS syntax; models default to generic queries.
        parts += [
            "",
            "## Reminder",
            "Target these applicant-tracking sites in your queries: "
            + ", ".join(ATS_DOMAINS),
            "A query with no site: filter usually returns login-walled listing "
            "pages that contain no actual job description.",
        ]
        return "\n".join(parts)

    @staticmethod
    def _coerce_plan(payload: Dict[str, Any], max_queries: int) -> SearchPlan:
        """Validate the model's output into a SearchPlan.

        Anything malformed is dropped rather than passed through: a bad query
        wastes a Tavily credit, which is the scarce resource here.
        """
        if not isinstance(payload, dict):
            return SearchPlan()

        def as_list(key: str, limit: int) -> List[str]:
            value = payload.get(key)
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                return []
            out = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    cleaned = " ".join(item.split())[:150]
                    if cleaned:
                        out.append(cleaned)
                if len(out) >= limit:
                    break
            return out

        rationale = payload.get("rationale")
        return SearchPlan(
            queries=as_list("queries", max_queries),
            target_companies=as_list("target_companies", 5),
            focus_skills=as_list("focus_skills", 12),
            rationale=rationale.strip()[:400] if isinstance(rationale, str) else "",
        )
