"""Typed domain models for jobs, companies and the user profile.

Deliberately LLM-free: these are the schemas the deterministic layers rely on.
Anything the model produces is validated back into these shapes before it is
allowed anywhere near storage.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# Status values used to avoid re-recommending anything already handled.
STATUS_DISCOVERED = "discovered"
STATUS_SAVED = "saved"
STATUS_APPLIED = "applied"
STATUS_REJECTED = "rejected"
STATUS_DISMISSED = "dismissed"

#: Statuses that must never be surfaced as a new recommendation again.
CLOSED_STATUSES = (STATUS_APPLIED, STATUS_REJECTED, STATUS_DISMISSED)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_url(url: str) -> str:
    """Strip tracking params so the same posting from two sources dedupes.

    `?utm_source=...`, `?gh_jid=...` style params and trailing slashes change
    constantly between job boards while pointing at one posting.
    """
    if not url:
        return ""
    url = url.strip()
    url = re.sub(r"[?&](utm_[a-z]+|gh_jid|lever-source|ref|src|trk\w*)=[^&#]*", "", url, flags=re.I)
    url = re.sub(r"[?&]$", "", url)
    return url.rstrip("/")


def derive_job_id(url: str, company: str, title: str) -> str:
    """Stable id for a posting.

    URL alone is not enough: some boards recycle the same URL across postings.
    Combining it with company+title gives a key that is stable across
    re-scrape runs but changes if the role is genuinely a different one.
    """
    basis = f"{normalise_url(url).lower()}|{company.strip().lower()}|{title.strip().lower()}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]


class JobPosting(BaseModel):
    """A single job posting discovered by the Scout."""

    job_id: str = ""
    title: str
    company: str
    url: str
    location: str = ""
    description: str = ""
    requirements: List[str] = Field(default_factory=list)
    seniority: str = ""
    remote: bool = False
    salary: str = ""
    posted_at: str = ""
    source: str = "tavily"
    discovered_at: str = Field(default_factory=utcnow_iso)
    status: str = STATUS_DISCOVERED
    raw: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def _clean_url(cls, value: str) -> str:
        return normalise_url(value)

    @field_validator("company", "title")
    @classmethod
    def _strip(cls, value: str) -> str:
        return (value or "").strip()

    def model_post_init(self, __context: Any) -> None:
        if not self.job_id:
            self.job_id = derive_job_id(self.url, self.company, self.title)
        # Split the description into discrete requirement bullets; used by the
        # deterministic keyword scorer.
        if not self.requirements:
            self.requirements = extract_requirement_bullets(self.description)

    @property
    def text_for_matching(self) -> str:
        parts = [self.title, self.company, self.location, self.seniority, self.description]
        parts.extend(self.requirements)
        return "\n".join(p for p in parts if p)

    def is_closed(self) -> bool:
        return self.status in CLOSED_STATUSES


class RankedJob(BaseModel):
    """A job plus its match evidence, used to render the final report."""

    job: JobPosting
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    preference_score: float = 0.0
    reasons: List[str] = Field(default_factory=list)
    already_seen: bool = False

    @property
    def pct(self) -> int:
        return int(round(self.score * 100))


class SearchPlan(BaseModel):
    """The Planner's output: what the Scout should go and look for."""

    queries: List[str] = Field(default_factory=list)
    target_companies: List[str] = Field(default_factory=list)
    focus_skills: List[str] = Field(default_factory=list)
    rationale: str = ""

    def all_terms(self) -> List[str]:
        return [*self.queries, *self.target_companies]


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def extract_requirement_bullets(description: str, limit: int = 25) -> List[str]:
    """Pull requirement-ish bullets out of a free-text job description."""
    if not description:
        return []
    bullets: List[str] = []
    for line in description.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) > 240:
            continue
        if _BULLET_RE.match(stripped):
            cleaned = _BULLET_RE.sub("", stripped).strip()
            if len(cleaned) > 2:
                bullets.append(cleaned)
        if len(bullets) >= limit:
            break
    return bullets
