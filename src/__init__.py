"""Job-search agent package.

Layout:
    config.py       central settings loaded from .env
    memory/         HelloAgents-style memory system (semantic/episodic/working)
    llm/            DeepSeek client
    scout/          Tavily-backed internet access with caching
    domain/         job/company/profile models + SQLite stores
    agent/          Planner-Executor pipeline with Scout isolation
    cli.py          command line entry point
"""
