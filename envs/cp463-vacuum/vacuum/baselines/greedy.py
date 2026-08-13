"""🥈 Silver — GreedyAgent (environment-spec §10)

    if cell ปัจจุบันสกปรก: SUCK
    if มี cell สกปรกอยู่ในหน้าต่างที่มองเห็น:
        เดินหนึ่งก้าวเข้าหา cell สกปรกที่ใกล้ที่สุด (Manhattan, เสมอ → flat_index ต่ำสุด)
    else:
        เดินสุ่มโดยเลี่ยงทิศที่รู้ว่าเป็นกำแพง

**มองแค่ในหน้าต่าง ไม่วางแผนระยะไกล** — จุดนี้คือสิ่งที่แยก Silver ออกจาก Gold
(Gold จำแผนที่สะสมทั้งหมดและ BFS ไปหาเป้าที่มองไม่เห็นแล้วได้)

ความหมายบน leaderboard: "agent มีกลยุทธ์แล้ว"
"""

from __future__ import annotations

import numpy as np

from vacuum.baselines.common import DX, DY, MOVES, SUCK, WorldModel, decode_pos


class GreedyAgent:
    def __init__(self, config: dict):
        self.W = config["width"]
        self.H = config["height"]
        self.mode = config["observation"]
        self.window = config.get("observation_window")
        self._seed = int(config.get("agent_seed", 0))
        self.model = WorldModel(self.W, self.H, self.mode, self.window)
        self.rng = np.random.Generator(np.random.PCG64(self._seed))

    def reset(self, episode_info: dict) -> None:
        self.model.reset()
        self.rng = np.random.Generator(np.random.PCG64(self._seed))

    def _visible_radius(self) -> int | None:
        """รัศมีของสิ่งที่ "มองเห็นตอนนี้" — None = เห็นทั้งห้อง"""
        if self.mode == "full":
            return None
        if self.mode == "local":
            return self.window // 2
        return 1  # sensor

    def act(self, observation) -> int:
        self.model.update(observation)
        if self.model.dirty_here():
            return SUCK

        x, y = self.model.pos
        r = self._visible_radius()

        # หา cell สกปรกที่ใกล้ที่สุด "ในหน้าต่างที่มองเห็น" เท่านั้น
        best: tuple[int, int] | None = None  # (ระยะ Manhattan, flat_index)
        for gy in range(self.H):
            if r is not None and abs(gy - y) > r:
                continue
            for gx in range(self.W):
                if r is not None and abs(gx - x) > r:
                    continue
                i = self.model.flat(gx, gy)
                if not self.model.dirty[i] or self.model.obstacle[i]:
                    continue
                key = (abs(gx - x) + abs(gy - y), i)
                if best is None or key < best:
                    best = key

        if best is not None:
            ty, tx = divmod(best[1], self.W)
            step = self._step_toward(x, y, tx, ty)
            if step is not None:
                return step

        return self._random_step(x, y)

    def _step_toward(self, x: int, y: int, tx: int, ty: int) -> int | None:
        """หนึ่งก้าวเข้าหาเป้า โดยเลี่ยงกำแพงที่รู้แล้ว — เสมอ → action id ต่ำสุด"""
        candidates = []
        for d in MOVES:
            nx, ny = x + DX[d], y + DY[d]
            if not self.model.in_bounds(nx, ny) or self.model.known_wall(nx, ny):
                continue
            candidates.append((abs(nx - tx) + abs(ny - ty), d))
        if not candidates:
            return None
        return min(candidates)[1]

    def _random_step(self, x: int, y: int) -> int:
        options = [
            d
            for d in MOVES
            if self.model.in_bounds(x + DX[d], y + DY[d])
            and not self.model.known_wall(x + DX[d], y + DY[d])
        ]
        if not options:
            options = list(MOVES)
        return int(options[int(self.rng.integers(0, len(options)))])
