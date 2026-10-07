"""Scout layer: the only place the agent touches the internet."""
from .parse import (
    candidate_from_search_hit,
    guess_company_from_url,
    is_aggregator_url,
    is_board_index,
    is_listing_or_aggregator,
    is_listing_title,
    is_listing_url,
    looks_like_job_url,
    parse_page,
)
from .tavily_scout import ScoutBudget, ScoutError, TavilyScout

__all__ = [
    "TavilyScout",
    "ScoutBudget",
    "ScoutError",
    "candidate_from_search_hit",
    "parse_page",
    "looks_like_job_url",
    "is_board_index",
    "is_aggregator_url",
    "is_listing_title",
    "is_listing_url",
    "is_listing_or_aggregator",
    "guess_company_from_url",
]
