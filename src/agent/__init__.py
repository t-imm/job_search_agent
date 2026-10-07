"""Agent layer: Planner-Executor pipeline with Scout isolation."""
from .matcher import Matcher, Profile, build_profile_from_text
from .pipeline import JobSearchAgent, RunResult

__all__ = [
    "JobSearchAgent",
    "RunResult",
    "Matcher",
    "Profile",
    "build_profile_from_text",
]
