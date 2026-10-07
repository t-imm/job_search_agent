"""The Scout: the only component allowed to touch the internet.

Wraps Tavily with three controls that keep the paid surface small:

  * **cache-first** - a SQLite cache keyed by query hash (24h TTL) means a
    re-run of the same plan costs zero credits.
  * **budget cap**  - a hard ceiling on search and extract calls per run, so a
    runaway loop degrades instead of billing.
  * **batch extract** - up to 20 URLs per Extract call (Tavily's limit).

Search returns *candidate* results. Turning those into structured postings is
`parse.py`'s job, keeping web I/O separate from interpretation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from ..config import TavilyConfig, get_settings

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


class ScoutError(RuntimeError):
    """Tavily could not be reached or returned nothing usable."""


@dataclass
class ScoutBudget:
    """Tracks credit spend so a run can never exceed its ceiling."""

    max_searches: int = 6
    max_extracts: int = 2
    searches_used: int = 0
    extracts_used: int = 0
    cache_hits: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def searches_left(self) -> int:
        return max(0, self.max_searches - self.searches_used)

    @property
    def extracts_left(self) -> int:
        return max(0, self.max_extracts - self.extracts_used)

    @property
    def estimated_credits(self) -> int:
        """basic search = 1 credit; extract = 1 credit per 5 successful URLs."""
        return self.searches_used + sum(
            1 for _ in range(self.extracts_used)
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "searches_used": self.searches_used,
            "searches_left": self.searches_left,
            "extracts_used": self.extracts_used,
            "extracts_left": self.extracts_left,
            "cache_hits": self.cache_hits,
            "estimated_credits": self.estimated_credits,
            "errors": self.errors,
        }


class TavilyScout:
    def __init__(
        self,
        config: TavilyConfig | None = None,
        cache: Any = None,
        budget: Optional[ScoutBudget] = None,
    ) -> None:
        self.config = config or get_settings().tavily
        if not self.config.api_key:
            raise ScoutError("TAVILY_API_KEY is not configured")
        # `cache` is a SQLiteStore; typed loosely to avoid a circular import.
        self.cache = cache
        self.budget = budget or ScoutBudget()
        self._session = requests.Session()

    # ----------------------------------------------------------------- search

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        use_cache: bool = True,
        ttl_hours: int = 24,
    ) -> Dict[str, Any]:
        """Run a Tavily search, serving from cache when possible."""
        if self.cache is not None and use_cache:
            cached = self.cache.get_cached_search(query, self.config.search_depth, ttl_hours)
            if cached is not None:
                self.budget.cache_hits += 1
                logger.info("Cache hit for search %r (0 credits)", query[:60])
                return cached

        if self.budget.searches_left <= 0:
            logger.warning("Search budget exhausted; skipping %r", query[:60])
            self.budget.errors.append(f"budget exhausted before: {query}")
            return {"results": [], "cached": False}

        payload: Dict[str, Any] = {
            "api_key": self.config.api_key,
            "query": query,
            "search_depth": self.config.search_depth,
            "max_results": max_results or self.config.max_results,
            "include_answer": False,
            "include_raw_content": self.config.include_raw_content,
            "include_images": False,
        }
        if include_domains:
            payload["include_domains"] = include_domains

        self.budget.searches_used += 1
        data = self._post(TAVILY_SEARCH_URL, payload, what=f"search {query[:50]!r}")

        # Tavily can answer 200 with per-result failures; keep what worked.
        results = data.get("results") or []
        if not isinstance(results, list):
            results = []

        response = {
            "query": query,
            "results": results,
            "response_time": data.get("response_time"),
            "cached": False,
        }
        if self.cache is not None and results:
            self.cache.set_cached_search(query, self.config.search_depth, response)
        return response

    def search_many(
        self,
        queries: List[str],
        max_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        out = []
        for query in queries:
            out.append(
                self.search(query, max_results=max_results, include_domains=include_domains)
            )
        return out

    # ---------------------------------------------------------------- extract

    def extract(self, urls: List[str]) -> Dict[str, str]:
        """Fetch page content for up to N URLs, batched 20 at a time.

        Returns a mapping of url -> raw markdown/text content. Unreachable URLs
        are simply absent from the result; they are not an error.
        """
        if not urls:
            return {}
        if self.budget.extracts_left <= 0:
            logger.warning("Extract budget exhausted; skipping %s url(s)", len(urls))
            self.budget.errors.append(f"extract budget exhausted for {len(urls)} urls")
            return {}

        contents: Dict[str, str] = {}
        # Preserve order while removing duplicates.
        unique_urls = list(dict.fromkeys(urls))

        for start in range(0, len(unique_urls), self.config.extract_max_urls):
            batch = unique_urls[start:start + self.config.extract_max_urls]
            self.budget.extracts_used += 1
            payload = {
                "api_key": self.config.api_key,
                "urls": batch,
                "extract_depth": "basic",
            }
            try:
                data = self._post(TAVILY_EXTRACT_URL, payload, what="extract")
            except ScoutError as exc:
                self.budget.errors.append(str(exc))
                continue

            for item in data.get("results") or []:
                url = item.get("url")
                text = item.get("raw_content") or item.get("content")
                if url and text:
                    contents[url] = text
            # 200 responses can still carry failed_results; record them.
            for failure in data.get("failed_results") or []:
                logger.info("Extract failed for %s: %s", failure.get("url"), failure.get("error"))

        return contents

    # ------------------------------------------------------------------ infra

    def _post(self, url: str, payload: Dict[str, Any], what: str, retries: int = 2) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = self._session.post(url, json=payload, timeout=90)
                if response.status_code >= 500 or response.status_code == 429:
                    raise ScoutError(f"Tavily {response.status_code} on {what}")
                if response.status_code >= 400:
                    raise ScoutError(
                        f"Tavily rejected {what}: {response.status_code} {response.text[:300]}"
                    )
                return response.json()
            except ScoutError:
                raise
            except Exception as exc:
                last_error = exc
                time.sleep(2 * (attempt + 1))
        raise ScoutError(f"Tavily {what} failed after {retries + 1} attempts: {last_error}")

    def health_check(self) -> bool:
        try:
            self._post(
                TAVILY_SEARCH_URL,
                {"api_key": self.config.api_key, "query": "health check", "max_results": 1},
                what="health check",
                retries=0,
            )
            return True
        except Exception:
            return False
