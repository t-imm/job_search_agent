"""Command line interface.

    python -m src.cli run                  # full pipeline: ingest, plan, scout, report
    python -m src.cli run --no-scout       # re-rank what is already stored (0 credits)
    python -m src.cli ingest               # load my_information/ into memory only
    python -m src.cli status <job_id> applied
    python -m src.cli list                 # everything discovered so far
    python -m src.cli stats                # memory + database counters
    python -m src.cli history              # past runs
    python -m src.cli doctor               # check config and backing services
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python src/cli.py` as well as `python -m src.cli`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings

logger = logging.getLogger("cli")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    # These libraries are chatty at INFO and drown out our own progress lines.
    for noisy in ("neo4j", "neo4j.notifications", "urllib3", "qdrant_client", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_run(args) -> int:
    from src.agent.pipeline import JobSearchAgent

    agent = JobSearchAgent()
    result = agent.run(
        focus=args.focus,
        limit=args.limit,
        scout=not args.no_scout,
        include_applied=args.include_applied,
    )

    if args.json:
        import json

        print(json.dumps(
            {
                "plan": result.plan.model_dump() if result.plan else None,
                "recommendations": [
                    {
                        "rank": i,
                        "title": r.job.title,
                        "company": r.job.company,
                        "url": r.job.url,
                        "score": r.score,
                        "pct": r.pct,
                        "location": r.job.location,
                        "seniority": r.job.seniority,
                        "remote": r.job.remote,
                        "salary": r.job.salary,
                        "summary": r.job.description[:400],
                        "requirements": r.job.requirements[:6],
                        "reasons": r.reasons,
                    }
                    for i, r in enumerate(result.ranked, 1)
                ],
                "stats": result.stats,
                "notes": result.notes,
                "degraded": result.degraded,
            },
            indent=2,
            ensure_ascii=False,
        ))
    else:
        print(result.report)
        for note in result.notes:
            print(f"[note] {note}")

    return 0


def cmd_ingest(args) -> int:
    from src.agent.ingest import Ingestor
    from src.memory import MemoryManager
    from src.config import get_settings

    settings = get_settings()
    memory = MemoryManager(
        storage_path=settings.agent.storage_path, user_id=settings.agent.user_id
    )
    ingestor = Ingestor(memory, settings.agent.info_dir)
    report = ingestor.ingest_all(force=args.force)

    if report["files"]:
        for entry in report["files"]:
            print(f"  ingested {entry['name']}: {entry['chunks']} chunks")
    if report["skipped"]:
        print(f"  unchanged (skipped): {', '.join(report['skipped'])}")
    if not report["files"] and not report["skipped"]:
        print("  nothing to ingest")
    print(f"\nTotal chunks added: {report['chunks_added']}")

    summary = ingestor.profile_summary(max_chars=600)
    if summary:
        print("\nStored profile preview:")
        print(summary[:600])
    return 0


def cmd_status(args) -> int:
    from src.domain.sqlite_store import SQLiteStore
    from src.config import get_settings
    from src.domain.models import derive_job_id

    settings = get_settings()
    db = SQLiteStore(Path(settings.agent.storage_path) / "jobs.db")

    job_id = args.job_id
    # Accept a job id, or a URL from which we can derive one.
    if job_id.startswith("http"):
        job_id = derive_job_id(job_id, args.company or "", args.title or "")

    if not db.set_status(job_id, args.status):
        print(f"Could not update job '{job_id}'. Run 'list' to see stored job ids.")
        return 1
    print(f"Marked {job_id} as {args.status}. It will not be recommended again.")
    return 0


def cmd_list(args) -> int:
    from src.domain.sqlite_store import SQLiteStore
    from src.config import get_settings

    settings = get_settings()
    db = SQLiteStore(Path(settings.agent.storage_path) / "jobs.db")
    jobs = db.all_jobs()

    if args.status:
        jobs = [j for j in jobs if j.status == args.status]
    if not jobs:
        print("No jobs stored yet. Run: python -m src.cli run")
        return 0

    print(f"{len(jobs)} job(s):\n")
    for job in jobs:
        match = ""
        print(f"- [{job.status:9}] {job.title} @ {job.company}")
        print(f"    id     : {job.job_id}")
        print(f"    url    : {job.url}")
        if job.location or job.seniority:
            print(f"    detail : {job.location} {job.seniority}".rstrip())
    return 0


def cmd_stats(args) -> int:
    from src.memory import MemoryManager
    from src.config import get_settings
    import json

    settings = get_settings()
    memory = MemoryManager(
        storage_path=settings.agent.storage_path, user_id=settings.agent.user_id
    )
    print(json.dumps(
        {"memory": memory.get_stats(), "database": memory.sqlite.stats()},
        indent=2,
        default=str,
    ))
    return 0


def cmd_history(args) -> int:
    from src.memory import MemoryManager
    from src.config import get_settings

    settings = get_settings()
    memory = MemoryManager(
        storage_path=settings.agent.storage_path, user_id=settings.agent.user_id
    )
    episodes = memory.recent_events(kind=args.kind, limit=args.limit)
    if not episodes:
        print("No history yet.")
        return 0
    for episode in episodes:
        stamp = episode.timestamp.strftime("%Y-%m-%d %H:%M")
        print(f"[{stamp}] {episode.metadata.get('kind', 'event')}: {episode.content}")
    return 0


def cmd_doctor(args) -> int:
    """Check configuration and that every backing service is reachable."""
    from src.config import get_settings
    from src.memory.embedding import get_embedder
    from src.memory.storage.neo4j_store import Neo4jStore
    from src.memory.storage.qdrant_store import QdrantStore
    from src.llm import get_llm_client
    from src.scout import TavilyScout

    settings = get_settings()
    ok = True

    print("Configuration")
    problems = settings.validate()
    for problem in problems:
        print(f"  MISSING  {problem}")
        ok = False
    if not problems:
        print("  OK       all API keys present")
    print(f"  info     storage: {settings.agent.storage_path}")
    print(f"  info     info_dir: {settings.agent.info_dir}")

    print("\nEmbedding (DashScope)")
    try:
        dimension = get_embedder().dimension
        print(f"  OK       text-embedding-v3, dimension {dimension}")
    except Exception as exc:
        print(f"  FAIL     {exc}")
        ok = False

    print("\nVector store (Qdrant)")
    try:
        store = QdrantStore()
        print(f"  OK       {settings.qdrant.url}, collection '{store.collection}', "
              f"{store.count()} vector(s)")
    except Exception as exc:
        print(f"  FAIL     {exc}")
        ok = False

    print("\nGraph store (Neo4j)")
    try:
        graph = Neo4jStore()
        print(f"  OK       {settings.neo4j.uri}, stats: {graph.get_stats()}")
        graph.close()
    except Exception as exc:
        print(f"  WARN     {exc}")
        print("          continuing without the graph layer")

    print("\nLLM (DeepSeek)")
    try:
        client = get_llm_client()
        reply = client.chat([{"role": "user", "content": "Say OK"}], max_tokens=1024)
        print(f"  OK       {settings.llm.model} replied: {reply.strip()[:40]!r}")
    except Exception as exc:
        print(f"  FAIL     {exc}")
        ok = False

    print("\nScout (Tavily)")
    try:
        TavilyScout().health_check()
        print("  OK       reachable")
    except Exception as exc:
        print(f"  FAIL     {exc}")
        ok = False

    print("\n" + ("All checks passed." if ok else "Some checks failed - see above."))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-search-agent",
        description="Personalized job-search agent with a persistent memory system.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full pipeline and print a report")
    run.add_argument("--focus", help="extra focus for this run, e.g. 'backend, Go, fintech'")
    run.add_argument("--limit", type=int, help="max recommendations (default 10)")
    run.add_argument("--no-scout", action="store_true",
                     help="skip the network stage; re-rank stored jobs only (0 credits)")
    run.add_argument("--include-applied", action="store_true",
                     help="also show jobs already marked applied")
    run.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    run.set_defaults(func=cmd_run)

    ingest = sub.add_parser("ingest", help="load my_information/ into memory")
    ingest.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    ingest.set_defaults(func=cmd_ingest)

    status = sub.add_parser("status", help="mark a job so it is not recommended again")
    status.add_argument("job_id", help="job id, or a job URL")
    status.add_argument("status", choices=["applied", "rejected", "dismissed", "saved", "discovered"])
    status.add_argument("--company", default="", help="company name, if passing a URL")
    status.add_argument("--title", default="", help="job title, if passing a URL")
    status.set_defaults(func=cmd_status)

    listing = sub.add_parser("list", help="list stored jobs")
    listing.add_argument("--status", help="filter by status")
    listing.set_defaults(func=cmd_list)

    stats = sub.add_parser("stats", help="memory and database counters")
    stats.set_defaults(func=cmd_stats)

    history = sub.add_parser("history", help="past runs and events")
    history.add_argument("--kind", help="plan | scout | recommendation")
    history.add_argument("--limit", type=int, default=15)
    history.set_defaults(func=cmd_history)

    doctor = sub.add_parser("doctor", help="check config and backing services")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
