"""Render the final report.

Plain text with an optional Markdown mode. Every recommendation carries a URL,
a short description, a match score, and the reasons behind it - the agent
should be able to justify each suggestion rather than assert it.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..domain.models import RankedJob, SearchPlan
from .matcher import Profile

SEPARATOR = "=" * 78
THIN = "-" * 78


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _job_summary(job) -> str:
    """One-line 'what is this job' description for the report."""
    if job.description:
        # Skip the boilerplate opening when we can find a meatier sentence.
        sentences = [s.strip() for s in job.description.split(".") if len(s.strip()) > 60]
        if sentences:
            return _truncate(sentences[0], 260)
        return _truncate(job.description, 260)
    if job.requirements:
        return _truncate(job.requirements[0], 260)
    return "(no description captured)"


def render_report(
    ranked: List[RankedJob],
    plan: Optional[SearchPlan] = None,
    budget_summary: Optional[Dict[str, Any]] = None,
    profile: Optional[Profile] = None,
    ingest_report: Optional[Dict[str, Any]] = None,
    degraded: bool = False,
    markdown: bool = False,
) -> str:
    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    header = f"Job Search Report - {now}"
    lines.append(f"# {header}" if markdown else header)
    lines.append(THIN if markdown else SEPARATOR)

    if ingest_report:
        added = ingest_report.get("chunks_added", 0)
        files = [f["name"] for f in ingest_report.get("files", [])]
        skipped = ingest_report.get("skipped", [])
        if added:
            lines.append(f"Profile ingested: {added} chunk(s) from {', '.join(files)}")
        elif skipped:
            lines.append(f"Profile up to date (unchanged: {', '.join(skipped)})")
        lines.append("")

    if plan:
        lines.append("Search plan")
        lines.append(THIN if markdown else "-" * 40)
        for query in plan.queries:
            lines.append(f"  - {query}")
        if plan.focus_skills:
            lines.append(f"  Focus skills: {', '.join(plan.focus_skills[:10])}")
        if plan.rationale:
            lines.append(f"  Rationale: {_truncate(plan.rationale, 220)}")
        lines.append("")

    lines.append(f"Top {len(ranked)} recommendation(s)")
    lines.append(THIN if markdown else SEPARATOR)

    if not ranked:
        lines.append("")
        lines.append("No new recommendations.")
        lines.append("")
        lines.append("Possible reasons:")
        lines.append("  - Every discovered job was already applied to or dismissed.")
        lines.append("  - The Scout found only listing pages, not individual postings.")
        lines.append("  - my_information/ is empty, so there is nothing to match against.")
    for index, item in enumerate(ranked, start=1):
        job = item.job
        lines.append("")
        badge = " (previously seen)" if item.already_seen else ""
        lines.append(f"{index}. {job.title} - {job.company}{badge}")
        lines.append(f"   Match: {item.pct}%   [vector {item.vector_score:.2f} | "
                     f"skills {item.keyword_score:.2f} | preference {item.preference_score:.2f}]")

        facts = []
        if job.location:
            facts.append(f"Location: {job.location}")
        if job.seniority:
            facts.append(f"Level: {job.seniority}")
        if job.remote:
            facts.append("Remote")
        if job.salary:
            facts.append(f"Salary: {job.salary}")
        facts.append(f"Source: {job.source}")
        if facts:
            lines.append("   " + " | ".join(facts))

        lines.append(f"   {job.url}")
        lines.append(f"   {_job_summary(job)}")

        if job.requirements:
            lines.append("   Key requirements:")
            for requirement in job.requirements[:4]:
                lines.append(f"     - {_truncate(requirement, 150)}")
        if item.reasons:
            lines.append(f"   Why: {'; '.join(item.reasons[:4])}")

    lines.append("")
    lines.append(THIN if markdown else SEPARATOR)
    lines.append("Run summary")
    if budget_summary:
        lines.append(
            f"  Tavily: {budget_summary.get('searches_used', 0)} search(es), "
            f"{budget_summary.get('extracts_used', 0)} extract batch(es), "
            f"{budget_summary.get('cache_hits', 0)} cache hit(s), "
            f"~{budget_summary.get('estimated_credits', 0)} credit(s)"
        )
        for error in budget_summary.get("errors", [])[:5]:
            lines.append(f"  Scout issue: {_truncate(error, 160)}")

    if degraded:
        lines.append("  NOTE: running in degraded mode (some stages unavailable).")

    lines.append("")
    lines.append("Already applied to, rejected or dismissed jobs are excluded automatically.")
    lines.append("Mark jobs with:  python -m src.cli status <job_id> applied")

    return "\n".join(lines)


def render_digest(ranked: List[RankedJob]) -> str:
    """A short summary suitable for chat or a quick scan."""
    if not ranked:
        return "No new job recommendations right now."
    parts = [f"Found {len(ranked)} new role(s). Top match: "
             f"{ranked[0].job.title} at {ranked[0].job.company} ({ranked[0].pct}%)."]
    for item in ranked[1:4]:
        parts.append(f"- {item.job.title} @ {item.job.company} ({item.pct}%)")
    return "\n".join(parts)
