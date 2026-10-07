"""Neo4j graph store adapter.

Holds the job-search knowledge graph:

    (:Company)-[:POSTS]->(:Job)-[:REQUIRES]->(:Skill)
    (:Job)-[:MATCHES]->(:Profile)

Company and skill nodes are reused across postings, so the graph answers
questions a vector index cannot, e.g. "which companies keep posting roles
that match this skill?" or "what have I already applied to at this company?".
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from ...config import Neo4jConfig, get_settings

logger = logging.getLogger(__name__)


class Neo4jUnavailable(RuntimeError):
    """Neo4j could not be reached."""


class Neo4jStore:
    def __init__(self, config: Neo4jConfig | None = None) -> None:
        self.config = config or get_settings().neo4j
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout,
            )
            self.driver.verify_connectivity()
        except Exception as exc:
            raise Neo4jUnavailable(
                f"Cannot reach Neo4j at {self.config.uri}. "
                f"Start it with: docker compose -f docker_compose.yml up -d ({exc})"
            ) from exc
        self.database = self.config.database
        self._ensure_constraints()

    def _run(self, query: str, **params) -> List[Any]:
        """Execute a query and materialise records inside the session.

        The result must be consumed before the session closes; returning the
        live `Result` would raise ResultConsumedError on iteration.
        """
        with self.driver.session(database=self.database) as session:
            return list(session.run(query, **params))

    def _ensure_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT company_key IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT skill_key IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT job_key IF NOT EXISTS FOR (j:Job) REQUIRE j.job_id IS UNIQUE",
            "CREATE INDEX job_posted_at IF NOT EXISTS FOR (j:Job) ON (j.posted_at)",
        ]
        for statement in statements:
            try:
                self._run(statement)
            except Exception as exc:
                # Older Neo4j versions lack IF NOT EXISTS on constraints.
                logger.debug("Constraint setup skipped: %s", exc)

    def close(self) -> None:
        try:
            self.driver.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ writes

    def upsert_company(self, name: str, properties: Optional[Dict[str, Any]] = None) -> None:
        props = {k: v for k, v in (properties or {}).items() if v is not None}
        self._run(
            """
            MERGE (c:Company {name: $name})
            SET c += $props
            """,
            name=name,
            props=props,
        )

    def upsert_job(
        self,
        job_id: str,
        company: str,
        title: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        props = {k: v for k, v in (properties or {}).items() if v is not None}
        self._run(
            """
            MERGE (c:Company {name: $company})
            MERGE (j:Job {job_id: $job_id})
            SET j += $props, j.title = $title, j.company = $company
            MERGE (c)-[:POSTS]->(j)
            """,
            job_id=job_id,
            company=company,
            title=title,
            props=props,
        )

    def link_skills(self, job_id: str, skills: List[str]) -> None:
        if not skills:
            return
        self._run(
            """
            MATCH (j:Job {job_id: $job_id})
            UNWIND $skills AS skill_name
            MERGE (s:Skill {name: toLower(skill_name)})
            MERGE (j)-[:REQUIRES]->(s)
            """,
            job_id=job_id,
            skills=skills,
        )

    def mark_applied(self, job_id: str, status: str = "applied") -> None:
        self._run(
            """
            MATCH (j:Job {job_id: $job_id})
            SET j.status = $status
            """,
            job_id=job_id,
            status=status,
        )

    # ------------------------------------------------------------------- reads

    def company_job_counts(self, limit: int = 50) -> List[Dict[str, Any]]:
        result = self._run(
            """
            MATCH (c:Company)-[:POSTS]->(j:Job)
            RETURN c.name AS company, count(j) AS job_count
            ORDER BY job_count DESC, company ASC
            LIMIT $limit
            """,
            limit=limit,
        )
        return [record.data() for record in result]

    def jobs_for_skill(self, skill: str, limit: int = 50) -> List[Dict[str, Any]]:
        result = self._run(
            """
            MATCH (j:Job)-[:REQUIRES]->(s:Skill {name: $skill})
            RETURN j.job_id AS job_id, j.title AS title, j.url AS url, j.company AS company
            LIMIT $limit
            """,
            skill=skill.lower(),
            limit=limit,
        )
        return [record.data() for record in result]

    def related_companies(self, company: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Companies sharing the most skills with `company` (2-hop traversal)."""
        result = self._run(
            """
            MATCH (target:Company {name: $company})-[:POSTS]->(j:Job)-[:REQUIRES]->(s:Skill)<-[:REQUIRES]-(j2:Job)<-[:POSTS]-(other:Company)
            WHERE other.name <> $company
            RETURN other.name AS company, count(DISTINCT s) AS shared_skills
            ORDER BY shared_skills DESC
            LIMIT $limit
            """,
            company=company,
            limit=limit,
        )
        return [record.data() for record in result]

    def get_stats(self) -> Dict[str, Any]:
        queries = {
            "companies": "MATCH (c:Company) RETURN count(c) AS n",
            "jobs": "MATCH (j:Job) RETURN count(j) AS n",
            "skills": "MATCH (s:Skill) RETURN count(s) AS n",
            "relationships": "MATCH ()-[r]->() RETURN count(r) AS n",
        }
        stats: Dict[str, Any] = {}
        for key, query in queries.items():
            try:
                records = self._run(query)
                record = records[0] if records else None
                stats[key] = record["n"] if record else 0
            except Exception:
                stats[key] = 0
        return stats

    def health_check(self) -> bool:
        try:
            records = self._run("RETURN 1 AS ok")
            return bool(records) and records[0]["ok"] == 1
            return True
        except Exception as exc:
            logger.error("Neo4j health check failed: %s", exc)
            return False

    def clear(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")
