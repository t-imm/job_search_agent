"""Deterministic job matcher.

Ranking lives in code, not in a prompt. Three signals are combined:

  * **vector**      - embedding similarity between the posting and the user's
                     stored profile (from Qdrant, via the same embedding model).
  * **keyword**     - overlap between posting skills and skills extracted from
                     the profile. Cheap, explainable, and catches exact tech
                     matches that embeddings blur together.
  * **preference**  - hard gates from the stated preferences: location and
                     seniority. These act as multipliers, because a perfect
                     skill match in the wrong city is not a good suggestion.

Every result carries human-readable `reasons` so the final report can explain
why a job was recommended, rather than asserting a number.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..domain.models import (
    STATUS_APPLIED,
    STATUS_DISMISSED,
    STATUS_REJECTED,
    JobPosting,
    RankedJob,
)
from ..memory.embedding import get_embedder

logger = logging.getLogger(__name__)

#: Closed statuses are never recommended again.
CLOSED = (STATUS_APPLIED, STATUS_REJECTED, STATUS_DISMISSED)

#: Multi-word tech names must win over their parts; longest match first.
_TECH_TERMS: Sequence[str] = (
    "machine learning", "deep learning", "computer vision", "natural language processing",
    "data engineering", "data analysis", "software engineering", "web development",
    "mobile development", "full stack", "back end", "front end", "backend", "frontend",
    "artificial intelligence", "generative ai", "large language model", "llm",
    "react", "react.js", "vue", "vue.js", "angular", "next.js", "nuxt", "svelte",
    "node.js", "node", "express", "django", "flask", "fastapi", "spring boot",
    "spring", "java", "kotlin", "scala", "python", "golang", "rust", "c++", "c#",
    "typescript", "javascript", "javaScript".lower(), "php", "ruby", "rails",
    "swift", "objective-c", "flutter", "react native", "android", "ios",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ci/cd", "jenkins", "github actions", "gitlab", "linux",
    "unix", "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
    "kafka", "rabbitmq", "graphql", "rest api", "rest", "grpc", "microservices",
    "distributed systems", "system design", "algorithms", "data structures",
    "tcp/ip", "networking", "network", "wireless", "5g", "iot", "embedded systems",
    "raspberry pi", "arduino", "stm32", "c/c++", "go", "golang",
    "pytorch", "tensorflow", "keras", "pandas", "numpy",
    "opencv", "yolo", "llm", "rag", "langchain", "power automate", "automation",
    "devops", "site reliability", "sre", "observability", "powershell",
    "sql", "nosql", "etl", "power bi", "tableau", "excel",
)

#: Single-letter and other tokens too generic to be evidence of a skill match.
#: Without this, "r" matches every posting containing the letter in any word.
_TOO_GENERIC_TERMS = frozenset({"go", "r", "rest"})

_TOKEN_RE = re.compile(r"[a-z][a-z0-9+#.\-]{1,20}")

_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those with without within for from to of in on at by
    as is are was were be been being do does did doing have has had having will would shall should can could
    may might must not no nor so such our yours their its it he she they we you i me my our us them who whom
    which what when where why how all any both each few more most other some only own same too very s t just
    don now also etc via per new role roles job jobs position positions company companies work working years
    year experience experienced ability strong excellent good great team teams work working role responsibilities
    requirements required preferred plus bonus nice good looking join joining opportunity opportunities""".split()
)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def content_tokens(text: str) -> Set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 2}


def extract_tech_terms(text: str) -> Set[str]:
    """Find known technology terms in free text, preserving multi-word names.

    Generic single words ("go", "r", "rest") are matched with word boundaries
    and excluded from *scoring*, because they match by accident far more often
    than by meaning.
    """
    lowered = f" {(text or '').lower()} "
    found: Set[str] = set()
    for term in _TECH_TERMS:
        if len(term) <= 3:
            # Short terms need boundaries to avoid substring false positives.
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                found.add(term)
        elif term in lowered:
            found.add(term)
        # Register component words so partial credit is possible.
        for part in term.split():
            if part not in _STOPWORDS and len(part) > 2 and part not in _TOO_GENERIC_TERMS:
                found.add(part)
    return found


def _scorable(terms: Set[str]) -> Set[str]:
    return {t for t in terms if t not in _TOO_GENERIC_TERMS}


class Profile:
    """The user's stored profile, distilled into scoring inputs."""

    def __init__(
        self,
        text: str,
        preferred_locations: Optional[Iterable[str]] = None,
        accepted_locations: Optional[Iterable[str]] = None,
        remote_ok: bool = True,
        target_seniority: Optional[Iterable[str]] = None,
        excluded_terms: Optional[Iterable[str]] = None,
    ) -> None:
        self.text = text or ""
        self.tech_terms = extract_tech_terms(self.text)
        self.tokens = content_tokens(self.text)
        self.preferred_locations = {l.lower() for l in (preferred_locations or [])}
        self.accepted_locations = {l.lower() for l in (accepted_locations or [])}
        self.remote_ok = remote_ok
        self.target_seniority = {s.lower() for s in (target_seniority or [])}
        self.excluded_terms = {t.lower() for t in (excluded_terms or [])}

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def summary_for_prompt(self, max_chars: int = 3000) -> str:
        return self.text[:max_chars]


class Matcher:
    """Ranks postings against a profile. No LLM calls."""

    def __init__(
        self,
        weight_vector: float = 0.5,
        weight_keyword: float = 0.35,
        weight_preference: float = 0.15,
    ) -> None:
        self.weight_vector = weight_vector
        self.weight_keyword = weight_keyword
        self.weight_preference = weight_preference
        self.embedder = get_embedder()
        # Cache of profile centroid embeddings keyed by job-batch id.
        self._profile_vector: Optional[List[float]] = None

    # ---------------------------------------------------------------- helpers

    def profile_vector(self, profile: Profile) -> List[float]:
        """Mean of the profile's section vectors - the 'ideal candidate' point."""
        if self._profile_vector is None:
            chunks = [c.strip() for c in profile.text.split("\n\n") if len(c.strip()) > 80]
            if not chunks:
                chunks = [profile.text[:1500] or "empty profile"]
            vectors = self.embedder.encode(chunks[:8])
            dim = len(vectors[0])
            centroid = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
            self._profile_vector = centroid
        return self._profile_vector

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))

    # ----------------------------------------------------------------- scoring

    def keyword_score(self, job: JobPosting, profile: Profile) -> Tuple[float, List[str]]:
        """Fraction of the profile's tech terms present in the posting.

        Also gives partial credit for raw content-word overlap so that a role
        matching on responsibilities rather than named technologies still scores.
        """
        if not profile.tech_terms:
            return 0.0, []

        profile_terms = _scorable(profile.tech_terms)
        job_terms = _scorable(extract_tech_terms(job.text_for_matching))
        if not profile_terms:
            return 0.0, []
        matched = profile_terms & job_terms
        tech_ratio = len(matched) / len(profile_terms)

        job_tokens = content_tokens(job.text_for_matching)
        token_overlap = len(profile.tokens & job_tokens)
        token_ratio = min(1.0, token_overlap / 40.0) if profile.tokens else 0.0

        score = 0.75 * tech_ratio + 0.25 * token_ratio

        matched_sorted = sorted(matched, key=len, reverse=True)[:6]
        reasons = [f"matches {m}" for m in matched_sorted]
        return min(1.0, score), reasons

    def preference_score(self, job: JobPosting, profile: Profile) -> Tuple[float, List[str]]:
        """Hard-ish gates for location and seniority."""
        reasons: List[str] = []
        score = 1.0

        if profile.preferred_locations:
            # Match against the canonical location first, then the raw text.
            location_text = f"{job.location} {job.description[:600]}".lower()
            canonical = (job.location or "").lower()
            hit = next(
                (loc for loc in profile.preferred_locations
                 if loc in canonical or loc in location_text),
                None,
            )
            if hit:
                reasons.append(f"location matches '{hit.title()}'")
            elif job.remote and profile.remote_ok:
                score *= 0.9
                reasons.append("remote, which you are open to")
            elif profile.accepted_locations and any(
                loc in location_text for loc in profile.accepted_locations
            ):
                score *= 0.95
                reasons.append("location in accepted locations")
            elif job.location:
                # A *known* wrong city is a real mismatch...
                score *= 0.6
                reasons.append(f"based in {job.location}, outside your preference")
            else:
                # ...but an unstated location is unknown, not wrong. Penalising
                # it would bury good roles whose posting omits the city.
                score *= 0.85
                reasons.append("location not stated in the posting")

        if job.remote and profile.remote_ok:
            reasons.append("remote-friendly")

        if profile.target_seniority:
            if job.seniority:
                if job.seniority.lower() in profile.target_seniority:
                    score *= 1.0
                    reasons.append(f"level '{job.seniority}' matches your target")
                else:
                    score *= 0.75
                    reasons.append(f"level '{job.seniority}' differs from your target")
            # Undetected seniority is neutral, not penalised.

        for term in profile.excluded_terms:
            if term and term in f"{job.title} {job.description[:400]}".lower():
                score *= 0.5
                reasons.append(f"mentions excluded term '{term}'")

        return max(0.0, min(1.0, score)), reasons

    def rank(
        self,
        jobs: List[JobPosting],
        profile: Profile,
        include_closed: bool = False,
        min_score: float = 0.0,
    ) -> List[RankedJob]:
        """Score and rank postings. Closed jobs are excluded unless asked for."""
        candidates = [j for j in jobs if include_closed or j.status not in CLOSED]
        if not candidates:
            return []
        if profile.is_empty:
            logger.warning("Profile is empty; ranking on preferences only")

        # One embedding round-trip per sub-batch for the whole job set.
        texts = [j.text_for_matching[:3000] or j.title for j in candidates]
        centroid: Optional[List[float]] = None
        job_vectors: List[List[float]] = []
        if not profile.is_empty:
            try:
                centroid = self.profile_vector(profile)
            except Exception as exc:
                logger.error("Could not embed the profile: %s", exc)
        try:
            job_vectors = self.embedder.encode(texts)
        except Exception as exc:
            # Degrade to keyword-only ranking, but say so loudly: a silent
            # fallback here looks like "poor matches" rather than "no vector
            # signal was available at all".
            logger.error(
                "Embedding %s job(s) failed (%s); ranking on skills and "
                "preferences only - vector scores will read 0.00.",
                len(texts), exc,
            )
            job_vectors = []

        ranked: List[RankedJob] = []
        for index, job in enumerate(candidates):
            v_score = 0.0
            if centroid is not None and index < len(job_vectors):
                v_score = self._cosine(job_vectors[index], centroid)

            k_score, k_reasons = self.keyword_score(job, profile)
            p_score, p_reasons = self.preference_score(job, profile)

            total = (
                self.weight_vector * v_score
                + self.weight_keyword * k_score
                + self.weight_preference * p_score
            )

            reasons = k_reasons[:3] + [r for r in p_reasons if "remote-friendly" not in r][:2]
            if job.salary:
                reasons.append(f"salary listed: {job.salary}")

            ranked.append(
                RankedJob(
                    job=job,
                    score=round(total, 4),
                    vector_score=round(v_score, 4),
                    keyword_score=round(k_score, 4),
                    preference_score=round(p_score, 4),
                    reasons=reasons,
                    already_seen=job.status != "discovered",
                )
            )

        ranked.sort(key=lambda r: r.score, reverse=True)
        return [r for r in ranked if r.score >= min_score]


def build_profile_from_text(text: str) -> Profile:
    """Derive scoring preferences from the ingested profile text.

    Kept deliberately simple and inspectable: the LLM is not trusted to emit
    these gates, because a wrong gate silently suppresses good jobs.

    Location labels must match the canonical names produced by
    `scout.parse.detect_location`, otherwise no job ever matches.
    """
    lowered = (text or "").lower()

    location_section = _section_text(lowered, ("work location", "location", "preferred location"))
    preferred: List[str] = []
    if location_section:
        # Ordered so the more specific label wins if several appear.
        for canonical, needles in (
            ("Hong Kong", ("hong kong", "hongkong")),
            ("Singapore", ("singapore",)),
            ("Mainland China", ("shenzhen", "beijing", "shanghai", "guangzhou")),
            ("Japan", ("tokyo", "osaka")),
            ("Taiwan, China", ("taipei", "taiwan")),
            ("United Kingdom", ("london", "united kingdom", " uk ", "uk based")),
            ("United States", ("united states", "usa", "new york", "san francisco")),
        ):
            if any(needle in location_section for needle in needles):
                preferred.append(canonical)

    # "open to remote work" anywhere in the preferences counts.
    remote_ok = "remote" in lowered

    seniority: List[str] = []
    level_section = _section_text(lowered, ("job level", "seniority", "preferred level"))
    if level_section:
        for known in ("entry", "junior", "graduate", "intern", "new grad", "early career"):
            if known in level_section:
                seniority.append(known)

    return Profile(
        text=text,
        preferred_locations=preferred,
        remote_ok=remote_ok,
        target_seniority=seniority,
    )


def _section_text(lowered_text: str, headings: Sequence[str]) -> str:
    """Return the body following any of `headings` in lowercased markdown."""
    for heading in headings:
        marker = f"## {heading}"
        if marker in lowered_text:
            tail = lowered_text.split(marker, 1)[1]
            # Stop at the next heading of the same level.
            return tail.split("\n## ", 1)[0]
    return ""
