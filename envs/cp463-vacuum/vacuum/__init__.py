"""vacuum_gridworld — environment ของ CP463 Competition 1

build spec: docs/competitions/CP463/1-2026/vacuum-robot/environment-spec.md
"""

__version__ = "1.0.0"

from vacuum.config import Config, load_config
from vacuum.env import VacuumEnv
from vacuum.scoring import EpisodeStats, episode_score, submission_score

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "VacuumEnv",
    "EpisodeStats",
    "episode_score",
    "submission_score",
]
