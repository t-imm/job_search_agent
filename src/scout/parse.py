"""Turn raw web text into structured job postings.

This is where unstructured Tavily output becomes `JobPosting` objects. The
first pass is deterministic heuristics (cheap, no tokens, always runs); the LLM
is only consulted for postings the heuristics cannot confidently identify, so a
typical run spends zero LLM calls on parsing.

Keeping parsing out of `tavily_scout.py` means the Scout stays a pure I/O
boundary: everything between it and the report reads SQLite only.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from ..domain.models import JobPosting, extract_requirement_bullets, normalise_url

logger = logging.getLogger(__name__)

# Hostname fragments that indicate an aggregator or a job board.
_JOB_HOST_HINTS = (
    "jobs.", "job.", "careers.", "career.", "vacanc", "hiring", "workablejobs",
    "greenhouse.io", "lever.co", "ashbyhq.com", "smartrecruiters", "jobvite",
    "indeed.", "linkedin.", "glassdoor.", "hk.jobs", "ctgoodjobs", "jobsdb",
    "recruit", "apply", "boards.", "jobstreet", "jobsank", "themuse", "wellfound",
)

# Hostnames that are list pages or aggregators rather than single postings.
_AGGREGATOR_HOST_HINTS = (
    "indeed.", "linkedin.", "glassdoor.", "monster.", "SimplyHired", "ziprecruiter",
    "ctgoodjobs", "jobsdb", "jobstreet", "jobsank", "themuse", "wellfound",
    "efinancialcareers", "remoterocketship", "hku", "ust", "hkust",
)

_SENIORITY_PATTERNS = (
    ("intern", r"\bintern(ship)?\b|\binternship\b"),
    ("graduate", r"\bgraduate\b|\bnew grad\b|\bfresh grad\b|\bcampus\b|\bearly career\b"),
    ("junior", r"\bjunior\b|\bentry[- ]level\b|\bjr\.?\b"),
    ("mid", r"\bmid[- ]level\b"),
    ("senior", r"\bsenior\b|\bsr\.?\b|\blead\b|\bprincipal\b"),
    ("staff", r"\bstaff\b|\bdistinguished\b|\bfellow\b"),
)

#: Programs that are entry-level despite carrying the word "Program"/"Engineer".
_EARLY_CAREER_RE = re.compile(
    r"\b(accelerator|apprentice|academy|graduate|intern|campus|new grad|early career|"
    r"trainee|associate)\b",
    re.I,
)

#: Titles where a senior keyword describes the programme track, not the level.
_TRACK_WORD_RE = re.compile(r"\b(programme|program|track|scheme|initiative|academy)\b", re.I)

#: Regions a role can be based in, mapped to a canonical label.
_LOCATION_PATTERNS = (
    ("Hong Kong", r"\b(hong ?kong|hongkong|kowloon|causeway bay|central,? hong|cyberport|"
                 r"hkust|quarry bay|sai ying pun|tsim sha tsui)\b"),
    ("Singapore", r"\bsingapore\b"),
    ("Mainland China", r"\b(shenzhen|beijing|shanghai|guangzhou|hangzhou|shantou)\b"),
    ("Japan", r"\b(tokyo|osaka|kyoto)\b"),
    ("Taiwan, China", r"\b(taipei|taiwan)\b"),
    ("United Kingdom", r"\b(london|manchester|edinburgh|cambridge|oxford|uk|britain)\b"),
    ("United States", r"\b(san francisco|new york|nyc|seattle|austin|boston|chicago|"
                      r"los angeles|denver|united states|usa|\bu\.?s\.?a?\b|california)\b"),
    ("Europe", r"\b(berlin|amsterdam|paris|lisbon|dublin|zurich|ireland|netherlands|"
               r"france|germany|spain|poland)\b"),
    ("Australia", r"\b(sydney|melbourne|australia)\b"),
    ("Canada", r"\b(toronto|vancouver|ontario|canada)\b"),
)

#: Signals that a posting is open to candidates outside its listed city.
_REMOTE_SIGNAL_RE = re.compile(
    r"\bremote\b|\bwork from home\b|\bwfh\b|\banywhere\b|\bfully remote\b|"
    r"\bdistributed team\b|\bglobal team\b",
    re.I,
)
# Match a full amount: "$125,000", "HK$40,000 - HK$60,000", "USD 80,000 per year".
_SALARY_RE = re.compile(
    r"(?:HK\$|HKD|USD|SGD|GBP|EUR|AUD|CA\$|\$)\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?"
    r"(?:(?:-|–|—|to)\s?(?:(?:HK\$|HKD|USD|SGD|GBP|EUR|AUD|CA\$|\$)\s?)?\d{1,3}(?:,\d{3})*)?"
    r"(?:\s*(?:per|/)\s*(?:month|year|hour|yr|mo|annum))?",
    re.I,
)
_DATE_RE = re.compile(r"(?:posted|published)\s+(?:on\s+)?((?:\d{1,2}\s+\w+\s+\d{4})|(?:\w+\s+\d{1,2},?\s+\d{4})|(?:\d{4}-\d{2}-\d{2}))", re.I)

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+){1,7}$", re.I)

#: ATS boards that host one posting per URL. The company slug sits in the path.
_ATS_COMPANY_PATH = (
    "jobs.lever.co", "jobs.ashbyhq.com", "boards.greenhouse.io", "job-boards.greenhouse.io",
    "apply.workable.com", "jobs.smartrecruiters.com", "careers.smartrecruiters.com",
    "myworkdayjobs.com", "jobs.jobvite.com", "hire.lever.co", "boards-api.greenhouse.io",
)

#: Path segments that mean "this is a board index, not a posting".
_BOARD_MARKERS = re.compile(
    r"^(jobs?|careers?|vacancies|openings|positions|search|all|list|board|boards|"
    r"current|latest|home|index)$",
    re.I,
)
_EMPLOYEE_COUNT_RE = re.compile(r"\b([\d,]+)\s*[-–]?\s*(?:employees|staff|people)\b", re.I)

_NOISE_LINES = re.compile(
    r"^(cookie|privacy policy|terms of|all rights reserved|subscribe|sign in|log in|"
    r"©|copyright|skip to|share this|apply now|read more|back to top)",
    re.I,
)

# Markdown chrome that carries no signal for a job posting.
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")          # ![logo](url)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")            # [text](url)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*")             # ## Title
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1", re.S)
_MD_BULLET = re.compile(r"^\s*[-*•]\s+")
_MD_BLOCKQUOTE = re.compile(r"^\s*>\s?")
_MD_HR = re.compile(r"^\s*([-*_]\s*){3,}$")
_MD_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


def strip_markdown(text: str) -> str:
    """Reduce markdown to plain prose suitable for matching and display.

    ATS pages are markdown documents full of logo images and heading hashes;
    leaving them in pollutes both the title guess and the keyword scorer.
    """
    if not text:
        return ""
    out: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if _MD_HR.match(stripped) or _MD_TABLE_SEP.match(stripped):
            continue
        stripped = _MD_IMAGE.sub("", stripped)
        stripped = _MD_LINK.sub(r"\1", stripped)
        stripped = _MD_BLOCKQUOTE.sub("", stripped)
        stripped = _MD_HEADING.sub("", stripped)
        stripped = _MD_BULLET.sub("", stripped)
        out.append(stripped.strip())

    text = "\n".join(out)
    text = _MD_EMPHASIS.sub(r"\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_job_url(url: str) -> bool:
    """Cheap pre-filter: is this URL plausibly a single job posting?

    Distinguishes a *posting* from a *board index*. `jobs.lever.co/binance` is
    a company board listing dozens of roles; `jobs.lever.co/binance/<uuid>` is
    one role. Only the latter is worth extracting, because board pages blow
    the context budget and yield no per-job detail.
    """
    if not url:
        return False
    lowered = url.strip().lower()
    parsed = urlparse(lowered)
    host = parsed.netloc or ""
    segments = [s for s in (parsed.path or "").split("/") if s]

    # Query strings that clearly mean a filtered list, not one posting.
    if any(bad in lowered for bad in ("/jobs/search", "/search?", "f_tid=", "page=", "sortBy=")):
        return False

    if any(host.endswith(ats) or ats in host for ats in _ATS_COMPANY_PATH):
        # Expect /<company>/<id-or-slug>; anything shorter is a board index.
        if len(segments) >= 2 and not _BOARD_MARKERS.match(segments[-1]):
            return True
        return False

    if any(hint in lowered for hint in _JOB_HOST_HINTS):
        return True

    path = parsed.path or ""
    return bool(re.search(r"/(job|jobs|vacancy|career|position|opening)s?/", path))


def is_board_index(url: str) -> bool:
    """True when the URL is a company board or filtered list, not one posting."""
    parsed = urlparse((url or "").lower())
    host = parsed.netloc or ""
    segments = [s for s in (parsed.path or "").split("/") if s]
    if any(host.endswith(ats) or ats in host for ats in _ATS_COMPANY_PATH):
        return len(segments) < 2 or _BOARD_MARKERS.match(segments[-1]) is not None
    return False


def is_aggregator_url(url: str) -> bool:
    lowered = (url or "").lower()
    return any(hint in lowered for hint in _AGGREGATOR_HOST_HINTS)


def guess_company_from_url(url: str) -> str:
    """Derive a plausible company name from the URL.

    ATS boards put the company slug in the first path segment
    (`jobs.lever.co/lalamove/<id>` -> Lalamove), which is far more reliable
    than the hostname. Falls back to the registrable hostname otherwise.
    """
    parsed = urlparse((url or "").lower())
    host = parsed.netloc or ""
    segments = [s for s in (parsed.path or "").split("/") if s]

    if any(host.endswith(ats) or ats in host for ats in _ATS_COMPANY_PATH) and segments:
        slug = segments[0]
        if not _BOARD_MARKERS.match(slug) and slug not in ("jobs", "careers"):
            return _titlecase_slug(slug)

    if not host:
        return ""
    if host.startswith("www."):
        host = host[4:]
    parts = [
        p for p in host.split(".")
        if p not in ("com", "hk", "co", "io", "ai", "net", "org", "jobs", "job", "careers")
    ]
    if not parts:
        return ""
    return _titlecase_slug(parts[0])


def _titlecase_slug(slug: str) -> str:
    """'lalamove' -> 'Lalamove'; 'efinancial-careers' -> 'Efinancial Careers'."""
    cleaned = slug.replace("_", " ").replace("-", " ").strip()
    if not cleaned:
        return ""
    # Preserve deliberate casing such as "eFinancialCareers".
    if any(c.isupper() for c in cleaned):
        return cleaned
    return " ".join(word.capitalize() for word in cleaned.split())


def detect_seniority(text: str) -> str:
    """Infer seniority from a title or description.

    The title wins over the body: a posting's body almost always names
    "senior" somewhere in a sentence about mentoring, which would otherwise
    label a graduate programme as a senior role.
    """
    if not text:
        return ""
    for line in text.splitlines()[:3]:
        candidate = _seniority_in_line(line)
        if candidate:
            return candidate
    return _seniority_in_line(text) or ""


def _seniority_in_line(text: str) -> str:
    for label, pattern in _SENIORITY_PATTERNS:
        if re.search(pattern, text, re.I):
            # "Binance Accelerator Program - AI Agent Engineer" matched
            # "agent"->senior? No: it matched via the body. Here a track word
            # downgrades an otherwise senior-looking title.
            if label in ("senior", "staff") and _TRACK_WORD_RE.search(text):
                if _EARLY_CAREER_RE.search(text):
                    return "graduate"
            return label
    return ""


def clean_text(raw: str, max_chars: int = 4000) -> str:
    """Strip navigation noise and collapse whitespace."""
    if not raw:
        return ""
    kept = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or _NOISE_LINES.match(stripped):
            continue
        kept.append(stripped)
    text = "\n".join(kept)
    # A posting's useful content is at the top; cap the tail.
    return text[:max_chars]


def clean_description(raw: str, max_chars: int = 3000) -> str:
    """Extract the meat of a page: prefer lines that read like a job body."""
    if not raw:
        return ""
    lines = [l.strip() for l in strip_markdown(raw).splitlines()]
    # Drop the leading breadcrumb-ish lines before the first substantial one.
    body: List[str] = []
    seen_content = False
    for line in lines:
        if not line:
            continue
        if len(line) < 40 and not seen_content:
            continue
        if len(line) >= 40:
            seen_content = True
        body.append(line)
        if sum(len(b) for b in body) > max_chars:
            break
    text = "\n".join(body)
    return re.sub(r"\n{3,}", "\n\n", text)[:max_chars]


def extract_posted_at(text: str) -> str:
    match = _DATE_RE.search(text or "")
    return match.group(1) if match else ""


def extract_salary(text: str) -> str:
    match = _SALARY_RE.search(text or "")
    return match.group(0).strip() if match else ""


def detect_location(text: str, hint: str = "") -> str:
    """Return a canonical location label, or '' if none is stated.

    An empty result is meaningful: it means "unknown", which the matcher
    treats differently from "somewhere else".
    """
    if hint:
        canonical = _canonical_location(hint)
        if canonical:
            return canonical
    return _canonical_location(text or "")


def _canonical_location(text: str) -> str:
    lowered = text.lower()
    # Scan the first part of the page only: the legal boilerplate at the
    # bottom often lists unrelated offices.
    head = lowered[:1500]
    for label, pattern in _LOCATION_PATTERNS:
        if re.search(pattern, head):
            return label
    return ""


def infer_location(text: str, default: str = "") -> str:
    """Backwards-compatible helper: canonical location or the caller's default."""
    return detect_location(text) or default


def is_remote(text: str) -> bool:
    return bool(_REMOTE_SIGNAL_RE.search(text or ""))


_LISTING_TITLE_RE = re.compile(
    r"^\s*(?:\d[\d,]*\s+)?(?:[\w\s&/,'-]{0,40}?\s+)?"
    r"(?:jobs?|vacancies|openings|positions|listings?|results?|search)\b"
    r"(?:\s+(?:in|at|for|near)\b.*)?\s*$",
    re.I,
)


#: Path prefixes that always mean "filtered list", never a single posting.
_LISTING_PATH_RE = re.compile(
    r"/(search|search-jobs|jobs/search|job-search|find-jobs|all-jobs|latest-jobs|"
    r"current-openings|open-positions|vacancies)(/|$|\?)",
    re.I,
)


def is_listing_title(title: str) -> bool:
    """Detect a search-results page title rather than a vacancy title.

    "695 Graduate Software Engineer Jobs in Hong Kong SAR" is a listing, not
    a job; treating it as one produces recommendations with fake companies
    like "Linkedin" and no real description.
    """
    if not title:
        return False
    cleaned = title.strip()
    if len(cleaned) > 120:
        return False
    return bool(_LISTING_TITLE_RE.match(cleaned))


def is_listing_url(url: str) -> bool:
    """Detect a listing/browse URL from its path."""
    if not url:
        return False
    parsed = urlparse(url.lower())
    path = parsed.path or ""
    if _LISTING_PATH_RE.search(path):
        return True
    # A bare company board is handled by is_board_index().
    segments = [s for s in path.split("/") if s]
    if segments and _BOARD_MARKERS.match(segments[-1]) and len(segments) == 1:
        return True
    return False


def is_listing_or_aggregator(title: str, url: str) -> bool:
    """Single gate used everywhere a candidate is accepted or rejected.

    Centralised so the Scout's pre-filter and the parser cannot drift apart -
    a mismatch would let listing pages back in as jobs with fake companies.
    """
    return bool(
        is_listing_title(title)
        or is_listing_url(url)
        or is_board_index(url)
        or is_aggregator_url(url)
        or not looks_like_job_url(url)
    )


def candidate_from_search_hit(hit: Dict[str, Any]) -> Optional[JobPosting]:
    """Build a low-confidence posting from a search result alone.

    Search snippets carry a title and a URL but usually not the body, so these
    candidates exist to be fetched by `extract`, not to be reported directly.
    """
    url = normalise_url(hit.get("url") or "")
    title = (hit.get("title") or "").strip()
    if not url or not title:
        return None
    if is_listing_or_aggregator(title, url):
        return None

    snippet = (hit.get("content") or "").strip()
    company = guess_company_from_url(url)
    body = clean_description(snippet, max_chars=1200)
    # ATS search hits are usually "Company - Title"; reuse that when we can.
    title = _split_title_and_company(title, company)

    return JobPosting(
        title=title,
        company=company,
        url=url,
        description=body,
        location=detect_location(snippet, hint=title),
        seniority=detect_seniority(title),
        remote=is_remote(f"{title} {snippet}"),
        salary=extract_salary(snippet),
        posted_at=extract_posted_at(snippet),
        source="tavily",
        raw={"snippet": snippet, "search_score": hit.get("score")},
    )


_TITLE_NOISE = re.compile(
    r"\s*[-|–—]\s*(jobs?|job|vacancy|careers?|career|opening|listing|"
    r"hk|hong kong|apply|job board|myjobs|jobsdb|ctgoodjobs|indeed|linkedin)\s*$",
    re.I,
)

#: Leading "Company - " / "Company | " prefix on ATS search titles.
_LEADING_COMPANY_RE = re.compile(r"^\s*([A-Z][\w&.\-']*(?:\s+[\w&.\-']+){0,3})\s*[-|–—]\s+(?=\S)")


def _split_title_and_company(title: str, known_company: str) -> str:
    """Recover a clean job title from an ATS search-result title.

    Lever/Greenhouse style titles read "Lalamove - Software Engineer, Backend".
    The company part is already known from the URL, so strip it rather than
    duplicating it into the title.
    """
    cleaned = _tidy_title(title)

    # Case 1: title begins with the company we already know.
    if known_company and cleaned.lower().startswith(known_company.lower()):
        remainder = cleaned[len(known_company):].lstrip(" -–—|:")
        if len(remainder) >= 6:
            return remainder

    # Case 2: "Something - Actual Title"; keep the longer, more specific side.
    match = _LEADING_COMPANY_RE.match(cleaned)
    if match:
        head, tail = match.group(1).strip(), cleaned[match.end():].strip()
        if len(tail) >= 8 and tail.lower() != head.lower():
            return tail
    return cleaned


def _clean_title(title: str, company: str) -> str:
    return _split_title_and_company(title, company) or title.strip()


def parse_page(
    url: str,
    content: str,
    hint_title: str = "",
    hint_company: str = "",
) -> Optional[JobPosting]:
    """Build a posting from extracted page content (high confidence path)."""
    if not content or len(content.strip()) < 80:
        return None

    # An `/apply` URL often resolves to a bare upload form with no job body.
    if _is_form_page(content):
        logger.info("Skipping form-only page: %s", url[:80])
        return None

    text = clean_description(clean_text(content))
    if len(text) < 60:
        return None

    title = _tidy_title(hint_title) if hint_title else ""
    if not title:
        title = _guess_title_from_text(content)
    if not title:
        return None

    company = hint_company.strip() or _guess_company_from_text(text, url)
    if not company:
        company = guess_company_from_url(url) or "Unknown"

    requirements = extract_requirement_bullets(text)
    # Fall back to sentence-level skills if the page had no bullets.
    if not requirements:
        requirements = _guess_requirement_sentences(text)

    return JobPosting(
        title=title,
        company=company,
        url=url,
        description=text,
        requirements=requirements,
        location=detect_location(text, hint=title),
        seniority=detect_seniority(title),
        remote=is_remote(f"{title} {text[:1500]}"),
        salary=extract_salary(text[:3000]),
        posted_at=extract_posted_at(text[:2000]),
        source="tavily",
        raw={"extracted_chars": len(text)},
    )


_TITLE_LINE_RE = re.compile(
    r"^(?:job\s+title|position|role)\s*[:\-]\s*(.+)$", re.I | re.M
)


_FORM_MARKERS = re.compile(
    r"(file exceeds the maximum upload size|"
    r"please attach your resume|upload your (?:cv|resume)|"
    r"additional questions?\s*$|"
    r"submit your application)",
    re.I | re.M,
)


def _is_form_page(content: str) -> bool:
    """Detect an application form rather than a job description."""
    text = content or ""
    hits = len(_FORM_MARKERS.findall(text))
    # A real posting also contains form chrome, so require it to dominate.
    return hits >= 2 and len(text) < 2500


def _guess_title_from_text(raw: str) -> str:
    """Pull the job title out of a raw ATS page.

    Order matters. On Lever/Greenhouse/Workable pages the title is an H1/H2
    markdown heading, which is far more reliable than "first long line" -
    the latter lands on a requirement bullet once nav chrome is stripped.
    """
    if not raw:
        return ""

    # 1. Markdown heading, skipping logo/alt-text lines.
    for line in raw.splitlines():
        stripped = line.strip()
        if not _MD_HEADING.match(stripped):
            continue
        heading = strip_markdown(stripped)
        heading = heading.splitlines()[0].strip() if heading else ""
        if not heading or _MD_IMAGE.search(line):
            continue
        # "## Software Engineer, Backend"
        if 6 <= len(heading) <= 120 and not re.search(
            r"\b(responsibilit|requirement|qualification|about|what you|why|benefit|"
            r"how to apply|apply now|additional question)\b",
            heading,
            re.I,
        ):
            return _tidy_title(heading)

    # 2. Explicit "Job title:" style label.
    match = _TITLE_LINE_RE.search(strip_markdown(raw))
    if match:
        candidate = _tidy_title(match.group(1))
        if candidate:
            return candidate

    # 3. Fall back to the first substantial plain-text line.
    for line in strip_markdown(raw).splitlines():
        stripped = line.strip()
        if 6 <= len(stripped) <= 120 and not stripped.endswith((".", ":", ",")):
            return _tidy_title(stripped)
    return ""


def _tidy_title(raw: str) -> str:
    """Normalise a guessed title: drop markdown chrome and trailing punctuation."""
    cleaned = strip_markdown(raw or "")
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
    cleaned = _TITLE_NOISE.sub("", cleaned).strip()
    cleaned = cleaned.strip("*_`#-–—|: ")
    return cleaned[:120]


_COMPANY_PATTERNS = (
    re.compile(r"\b(?:company|employer|organisation|organization)\s*[:\-]\s*([^\n]{2,60})", re.I),
    re.compile(r"\babout\s+([A-Z][\w&.\- ]{2,40})\s+(?:is|we are)\b"),
    re.compile(r"\bat\s+([A-Z][\w&.\- ]{2,40})\s*,?\s*(?:we|is)\b"),
    re.compile(r"^([A-Z][\w&.\- ]{2,40})\s+is\s+(?:looking|hiring|seeking)", re.M),
)

_SKIP_COMPANY = {"we", "our", "the", "this", "you", "they", "it", "a", "an"}


def _guess_company_from_text(text: str, url: str) -> str:
    for pattern in _COMPANY_PATTERNS:
        match = pattern.search(text[:4000])
        if match:
            candidate = match.group(1).strip(" .,")
            if candidate.lower() not in _SKIP_COMPANY and len(candidate) > 1:
                return candidate
    return ""


_REQ_CUES = (
    "experience", "knowledge", "familiar", "proficient", "skilled", "ability",
    "degree", "must have", "requirement", "qualification", "competent",
    "working knowledge", "strong", "solid", "exposure to",
)


def _guess_requirement_sentences(text: str, limit: int = 20) -> List[str]:
    """Pull requirement-ish sentences when the page has no bullet list."""
    out: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        stripped = sentence.strip()
        if not (25 <= len(stripped) <= 240):
            continue
        lowered = stripped.lower()
        if any(cue in lowered for cue in _REQ_CUES):
            out.append(stripped)
        if len(out) >= limit:
            break
    return out
