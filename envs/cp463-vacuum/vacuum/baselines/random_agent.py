"""🥉 Bronze — RandomAgent (environment-spec §10)

    if cell ปัจจุบันสกปรก: SUCK
    else: uniform random จาก {UP, DOWN, LEFT, RIGHT}

ไม่เลือก IDLE เลย — IDLE ไม่มีประโยชน์ในโจทย์นี้ และจะทำให้ baseline อ่อนเกินจนไม่มีความหมาย

ความหมายบน leaderboard: "โค้ดทำงานได้แล้ว"
"""

from __future__ import annotations

import numpy as np

from vacuum.baselines.common import MOVES, SUCK, WorldModel


class RandomAgent:
    def __init__(self, config: dict):
        self.W = config["width"]
        self.H = config["height"]
        # RNG ของ agent เอง — ตรึง seed ไว้เพื่อให้ผลทำซ้ำได้ (template §6 "ต้อง deterministic")
        # และห้ามใช้ global RNG เด็ดขาด ไม่งั้นจะไปกวน environment (env-spec §2 ข้อ 1)
        self._seed = int(config.get("agent_seed", 0))
        self.model = WorldModel(self.W, self.H, config["observation"], config.get("observation_window"))
        self.rng = np.random.Generator(np.random.PCG64(self._seed))

    def reset(self, episode_info: dict) -> None:
        self.model.reset()
        self.rng = np.random.Generator(np.random.PCG64(self._seed))

    def act(self, observation) -> int:
        self.model.update(observation)
        if self.model.dirty_here():
            return SUCK
        return int(MOVES[int(self.rng.integers(0, 4))])
