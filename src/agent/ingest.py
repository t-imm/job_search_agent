"""Turn the user's markdown files into memory.

Reads `my_information/resume.md` and `job_preference.md`, splits them into
semantically meaningful chunks, and writes them to the `profile` namespace.

Two deliberate choices:

  * **Heading-aware chunking.** Splitting on blank lines would mix a degree
    with the bullet under it. Splitting on markdown headings keeps each
    experience / project / education entry intact, which is what makes
    retrieval precise later.
  * **Content-hash skip.** Re-running ingest with an unchanged file is a
    no-op, so the vector store does not accumulate duplicates.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..memory import PROFILE_NS, MemoryManager

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

#: Sections that describe who the user is, versus what they want.
_IDENTITY_SECTIONS = {
    "professional summary", "summary", "education", "work experience",
    "experience", "projects", "skills", "technical skills", "certifications",
    "awards", "research", "publications",
}
_PREFERENCE_SECTIONS = {
    "work location", "location", "job level", "seniority", "preferences",
    "job preferences", "compensation", "salary", "industry", "role",
    "roles", "job type", "availability", "work authorization", "language",
    "languages", "benefits", "company", "companies",
}

#: Importance by section - identity drives matching, preferences gate it.
_IMPORTANCE = {
    "professional summary": 0.85,
    "work experience": 0.8,
    "experience": 0.8,
    "projects": 0.75,
    "education": 0.7,
    "skills": 0.8,
    "technical skills": 0.8,
    "work location": 0.95,
    "job level": 0.9,
    "seniority": 0.9,
}


class Ingestor:
    def __init__(self, memory: MemoryManager, info_dir: Path) -> None:
        self.memory = memory
        self.info_dir = Path(info_dir)

    def discover_files(self) -> List[Path]:
        if not self.info_dir.exists():
            logger.warning("Info directory does not exist: %s", self.info_dir)
            return []
        # Ignore the directory's own README - it describes the folder, not the user.
        return sorted(
            p for p in self.info_dir.glob("*.md")
            if p.name.lower() != "readme.md"
        )

    def ingest_all(self, force: bool = False) -> Dict[str, Any]:
        report: Dict[str, Any] = {"files": [], "chunks_added": 0, "skipped": []}

        for path in self.discover_files():
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.error("Cannot read %s: %s", path, exc)
                continue

            if not force and not self.memory.sqlite.needs_ingest(str(path), content):
                report["skipped"].append(path.name)
                logger.info("Unchanged, skipping ingest: %s", path.name)
                continue

            chunks = chunk_markdown(content, source=path.name)
            if not chunks:
                continue

            self.memory.add_profile_chunks(chunks, path.name)
            self.memory.sqlite.record_ingest(str(path), content, len(chunks))

            report["files"].append({"name": path.name, "chunks": len(chunks)})
            report["chunks_added"] += len(chunks)
            logger.info("Ingested %s -> %s chunks", path.name, len(chunks))

        if not report["files"] and report["skipped"]:
            logger.info("Profile already up to date (%s)", ", ".join(report["skipped"]))
        return report

    def profile_summary(self, max_chars: int = 4000) -> str:
        """Fetch the whole stored profile for prompt construction.

        Reads back from Qdrant rather than an in-process cache, because the
        cache is empty on every fresh start and the profile must survive
        across runs. The agent's view of the user is therefore the *stored*
        memory - the same source the matcher uses.
        """
        profile_items = self.memory.semantic.list_namespace(PROFILE_NS)
        if not profile_items:
            return ""

        # Preserve document order (stored as `order`) rather than sorting by
        # name: a resume reads best summary -> experience -> projects.
        def sort_key(item) -> tuple:
            kind = item.metadata.get("kind", "identity")
            source_rank = 0 if item.metadata.get("source") == "resume.md" else 1
            return (source_rank, int(item.metadata.get("order", 0)))

        profile_items.sort(key=sort_key)

        parts = []
        seen_sections = set()
        for item in profile_items:
            section = item.metadata.get("section") or item.metadata.get("source", "")
            # Chunks from one section are already separated by \n\n; label
            # only when the section changes to avoid repeating every heading.
            if section and section not in seen_sections:
                parts.append(f"## {section}")
                seen_sections.add(section)
            parts.append(item.content)
        return "\n\n".join(parts)[:max_chars]


def chunk_markdown(markdown: str, source: str = "") -> List[Dict[str, Any]]:
    """Split markdown into heading-scoped chunks.

    A heading starts a new chunk; its body accumulates until the next heading
    of the same or higher level. Content before the first heading becomes its
    own chunk.

    Document order is preserved via `order` so the resume reads top-down
    (summary -> experience -> projects) rather than alphabetically.
    """
    lines = markdown.splitlines()
    chunks: List[Dict[str, Any]] = []
    current_section = ""
    current_level = 0
    buffer: List[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if not body:
            return
        # `buffer[0]` is the heading text itself; drop it when it simply
        # repeats, since profile_summary() re-adds the heading as a label.
        lines = body.splitlines()
        if lines and lines[0].strip() == current_section.strip():
            lines = lines[1:]
        body = "\n".join(lines).strip()
        if len(body) < 20:
            return
        lowered = current_section.lower().strip()
        kind = _classify_section(lowered)
        chunks.append(
            {
                "content": body,
                "section": current_section or source or "general",
                "kind": kind,
                "importance": _IMPORTANCE.get(
                    lowered, 0.6 if kind == "identity" else 0.75
                ),
                "order": len(chunks),
            }
        )

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            # A deeper heading stays inside the current chunk as a sub-section.
            if current_section and level > current_level:
                buffer.append("")
                buffer.append(f"{'#' * level} {heading_text}")
                continue
            flush()
            current_level = level
            current_section = heading_text
            buffer = [heading_text]
        else:
            buffer.append(line)
    flush()

    return chunks


def _classify_section(section: str) -> str:
    lowered = (section or "").lower().strip()
    if lowered in _PREFERENCE_SECTIONS:
        return "preference"
    if lowered in _IDENTITY_SECTIONS:
        return "identity"
    # Heuristic for headings we have not seen before.
    if any(word in lowered for word in ("prefer", "location", "salary", "level", "type")):
        return "preference"
    return "identity"
