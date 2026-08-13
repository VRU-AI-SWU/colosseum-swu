"""รัน agent กับ environment แล้วคิดคะแนน — ใช้ใน starter kit และตอน calibrate

⚠️ **ห้ามใช้ module นี้เป็นตัวรันของ grader** — มันรัน agent ใน process เดียวกับ environment
ซึ่งเปิดให้โค้ดนิสิตเอื้อมไปอ่านผังห้อง เฉลย และ seed ได้ตรงๆ ด้วย `gc` ไม่กี่บรรทัด
(README §10.4) runner จริงต้องแยก process: environment อยู่ฝั่ง runner ที่เชื่อถือได้
agent อยู่ใน sandbox คุยกันผ่าน pipe ทีละ step

module นี้มีไว้สำหรับ (ก) นิสิตรันในเครื่องตัวเอง (ข) การทดลอง calibrate ของผู้สอน
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from vacuum.config import Config
from vacuum.env import VacuumEnv
from vacuum.scoring import ScoreBreakdown, SubmissionScore, episode_score, submission_score


class Agent(Protocol):
    def __init__(self, config: dict): ...
    def reset(self, episode_info: dict) -> None: ...
    def act(self, observation) -> int: ...


def agent_config(config: Config) -> dict[str, Any]:
    """ข้อมูลที่ agent ได้รู้ตอนสร้าง — **ไม่มีผังห้อง ไม่มี seed**"""
    return {
        "width": config.room.width,
        "height": config.room.height,
        "observation": config.robot.observation,
        "observation_window": config.robot.observation_window,
        "max_steps": config.episode.max_steps,
        "action_noise": config.dynamics.action_noise,
        "sticky_dirt": config.dynamics.sticky_dirt,
        "sensor_noise": config.dynamics.sensor_noise,
    }


@dataclass
class EpisodeResult:
    seed: int
    breakdown: ScoreBreakdown
    events: list[tuple[int, int, int]]


def run_episode(
    config: Config,
    agent: Agent,
    seed: int,
    *,
    env: VacuumEnv | None = None,
    keep_events: bool = False,
) -> EpisodeResult:
    env = env or VacuumEnv(config)
    obs, info = env.reset(seed=seed)
    agent.reset({k: v for k, v in agent_config(config).items()})

    done = False
    while not done:
        action = agent.act(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    breakdown = episode_score(
        env.stats(),
        config.episode.max_steps,
        completion_bonus=config.scoring.completion_bonus,
        max_penalty=config.scoring.max_penalty,
        w_collision=config.penalties.collision,
        w_redundant=config.penalties.redundant_suck,
    )
    return EpisodeResult(seed=seed, breakdown=breakdown, events=env.events if keep_events else [])


def evaluate(
    config: Config,
    agent_factory: Callable[[dict], Agent],
    seeds: Iterable[int],
) -> tuple[SubmissionScore, list[EpisodeResult]]:
    """รันทุก seed ด้วย agent instance ใหม่ต่อ episode แล้วรวมคะแนน

    สร้าง agent ใหม่ต่อ episode โดยตั้งใจ — ถ้าคะแนนต่างจากการเรียก `reset()`
    บน instance เดิม แปลว่ามี state รั่วข้าม episode (§13 การตรวจ submission)
    """
    env = VacuumEnv(config)
    cfg = agent_config(config)
    results = [run_episode(config, agent_factory(cfg), seed, env=env) for seed in seeds]
    return submission_score([r.breakdown for r in results]), results
