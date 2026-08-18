"""ปลั๊กที่ทำให้ Arena runner รันโจทย์นี้ได้ — สัญญาอยู่ที่ `runners/agent_env/plugin.py`

ไฟล์นี้เป็น **ที่เดียว** ที่ผูก environment ตัวนี้เข้ากับแพลตฟอร์ม
ถ้าจะเพิ่มโจทย์ใหม่ก็เขียนไฟล์แบบนี้ในแพ็กเกจของโจทย์นั้น โดยไม่ต้องแตะ `runners/` หรือ `core/`

ประกาศใน TaskSpec ว่า

    env_plugin: "vacuum.arena:PLUGIN"

⚠️ ไม่ import อะไรจาก `runners/` เพื่อให้ starter kit ที่นิสิตติดตั้งไม่ต้องมีโค้ดแพลตฟอร์มติดไปด้วย
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vacuum import __version__, load_config
from vacuum.env import VacuumEnv
from vacuum.replay import encode as encode_replay
from vacuum.replay import header_from_env
from vacuum.rollout import agent_config as _agent_config
from vacuum.scoring import ScoreBreakdown, episode_score, submission_score


class VacuumPlugin:
    name = "vacuum_gridworld"
    version = __version__

    def load_config(self, path: str) -> Any:
        return load_config(path)

    def config_hash(self, config: Any) -> str:
        return config.config_hash

    def apply_overrides(self, config: Any, overrides: dict[str, Any]) -> Any:
        return config.replace(**overrides) if overrides else config

    def make_env(self, config: Any) -> VacuumEnv:
        return VacuumEnv(config)

    def agent_config(self, config: Any) -> dict[str, Any]:
        return _agent_config(config)

    def episode_score(self, env: VacuumEnv, config: Any) -> ScoreBreakdown:
        return episode_score(
            env.stats(),
            config.episode.max_steps,
            completion_bonus=config.scoring.completion_bonus,
            max_penalty=config.scoring.max_penalty,
            w_collision=config.penalties.collision,
            w_redundant=config.penalties.redundant_suck,
        )

    def zero_score(self, config: Any) -> ScoreBreakdown:
        """episode ที่ agent ล้มเหลว — template §7.3 กำหนดว่าได้ 0 ไม่ใช่ "คะแนนน้อยลง" """
        return ScoreBreakdown(
            score=0.0, auc=0.0, completed=False, penalty=0.0,
            coverage=0.0, t_end=0, reason="agent_failed",
        )

    def aggregate(self, breakdowns: list[ScoreBreakdown]) -> Any:
        return submission_score(breakdowns)

    def write_replay(self, path: str, env: VacuumEnv) -> int:
        blob = encode_replay(header_from_env(env), env.events)
        Path(path).write_bytes(blob)
        return len(blob)

    def step_timeout_ms(self, config: Any) -> int:
        return config.episode.step_timeout_ms


PLUGIN = VacuumPlugin()
