"""Central configuration for the job-search agent.

All secrets and endpoints come from the project `.env` file. Nothing here reads
from `/reference` — that directory is reference material only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMConfig:
    """DeepSeek chat configuration.

    `deepseek-v4-flash` is a *reasoning* model: it emits `reasoning_content`
    first, then the real answer. Reasoning tokens are billed against
    `max_tokens`, so a tight budget yields an EMPTY content string. Keep
    max_tokens generous (>= 1024) or the caller gets nothing back.
    """
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.deepseek.com"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL_ID", "deepseek-v4-flash"))
    timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 60))
    max_tokens: int = 2048
    temperature: float = 0.3

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Aliyun DashScope text-embedding-v3 via the OpenAI-compatible REST API."""
    api_key: str = field(default_factory=lambda: _env("EMBED_API_KEY"))
    base_url: str = field(default_factory=lambda: _env("EMBED_BASE_URL"))
    model: str = field(default_factory=lambda: _env("EMBED_MODEL_NAME", "text-embedding-v3"))
    dimension: int = field(default_factory=lambda: _env_int("EMBED_DIMENSION", 1024))
    timeout: int = 30


@dataclass(frozen=True)
class QdrantConfig:
    url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY"))
    collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "job_search_vectors"))
    vector_size: int = field(default_factory=lambda: _env_int("QDRANT_VECTOR_SIZE", 1024))
    distance: str = field(default_factory=lambda: _env("QDRANT_DISTANCE", "cosine"))
    timeout: int = field(default_factory=lambda: _env_int("QDRANT_TIMEOUT", 30))


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    username: str = field(default_factory=lambda: _env("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "hello-agents-password"))
    database: str = field(default_factory=lambda: _env("NEO4J_DATABASE", "neo4j"))
    max_connection_pool_size: int = field(default_factory=lambda: _env_int("NEO4J_MAX_CONNECTION_POOL_SIZE", 50))
    connection_timeout: int = field(default_factory=lambda: _env_int("NEO4J_CONNECTION_TIMEOUT", 60))


@dataclass(frozen=True)
class TavilyConfig:
    api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))
    max_results: int = 6
    search_depth: str = "basic"      # 1 credit; "advanced" costs 2
    include_raw_content: bool = True
    extract_max_urls: int = 20       # Tavily caps Extract at 20 URLs/call


@dataclass(frozen=True)
class AgentConfig:
    """Knobs for ranking and scout budget."""
    user_id: str = "job_seeker"
    storage_path: str = field(default_factory=lambda: _env("MEMORY_STORAGE_PATH", "./memory_data"))
    # Deterministic scoring weights (kept in code, not prompts, per design).
    weight_vector: float = 0.5
    weight_keyword: float = 0.35
    weight_preference: float = 0.15
    top_k_results: int = 10
    max_searches_per_run: int = 6
    # Jobs older than this many days are treated as stale in the report.
    freshness_days: int = 30
    info_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "my_information")


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty means OK."""
        problems = []
        if not self.llm.api_key:
            problems.append("LLM_API_KEY is missing from .env")
        if not self.embedding.api_key:
            problems.append("EMBED_API_KEY is missing from .env")
        if not self.tavily.api_key:
            problems.append("TAVILY_API_KEY is missing from .env")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
