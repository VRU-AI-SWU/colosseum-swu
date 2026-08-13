"""VacuumEnv — environment-spec §5 (transition), §6 (termination), §8 (Gymnasium API)

`reward` คืน 0.0 เสมอโดยตั้งใจ — score เป็นเรื่องภายนอก (`vacuum.scoring`)
และ **reward เป็นสิ่งที่นิสิตออกแบบเอง** ตอนเทรน (ดู examples/reward_wrappers.py)
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from vacuum.config import Config
from vacuum.generator import DX, DY, Layout, NoiseTape, generate_layout, make_noise_tape
from vacuum.observation import build_observation, observation_space, sensed_size
from vacuum.scoring import EpisodeStats

UP, DOWN, LEFT, RIGHT, SUCK, IDLE = range(6)
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "SUCK", "IDLE")

# ทิศตั้งฉาก ใช้ตอนลื่น — UP/DOWN ลื่นไป LEFT/RIGHT และกลับกัน
PERPENDICULAR = {UP: (LEFT, RIGHT), DOWN: (LEFT, RIGHT), LEFT: (UP, DOWN), RIGHT: (UP, DOWN)}

# bit ของ flags ใน replay body (§9)
F_MOVED = 1 << 0
F_COLLISION = 1 << 1
F_SLIPPED = 1 << 2
F_CLEANED = 1 << 3
F_STICKY_FAIL = 1 << 4
F_REDUNDANT = 1 << 5


class VacuumEnv(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 8}

    def __init__(self, config: Config, render_mode: str | None = None):
        self.config = config
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(6)
        self.observation_space = observation_space(config)

        self._sensed_size = sensed_size(config)
        self._seed: int | None = None
        self.layout: Layout | None = None
        self.tape: NoiseTape | None = None

    # ── Gymnasium API ───────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is None:
            # กันการทดลองที่ทำซ้ำไม่ได้ — ไม่สุ่ม seed ให้เอง
            raise ValueError(
                "VacuumEnv.reset() ต้องระบุ seed เสมอ เช่น env.reset(seed=20001) "
                "— environment นี้ห้ามสุ่ม seed เอง เพื่อให้ทุก run ทำซ้ำได้"
            )
        super().reset(seed=seed)
        self._seed = int(seed)

        self.layout = generate_layout(self.config, self._seed)
        self.tape = make_noise_tape(self.config, self._seed, self._sensed_size)

        H, W = self.config.room.height, self.config.room.width
        self.dirt = self.layout.dirt0.copy()
        self.sticky_hit = np.zeros((H, W), dtype=bool)
        self.visited = np.zeros((H, W), dtype=bool)
        self.x, self.y = self.layout.start
        self.visited[self.y, self.x] = True  # visited[start] = True ก่อน timestep แรก
        self.t = 0
        self.battery = self.config.robot.battery

        self.cleaned = 0
        self.collisions = 0
        self.redundant_sucks = 0
        self.sticky_fails = 0
        self.slips = 0
        self.reason: str | None = None

        self._cleaned_at_t = [0]
        self._events: list[tuple[int, int, int]] = []  # (action, flags, flat_index)

        info = self._info()
        info.update(
            D0=self.layout.D0,
            free_count=self.layout.free_count,
            effective_density=self.layout.effective_density,
            config_hash=self.config.config_hash,
            env_version=__import__("vacuum").__version__,
            seed=self._seed,
            start=self.layout.start,
        )
        return self._observation(), info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.layout is None:
            raise RuntimeError("ต้องเรียก reset(seed=...) ก่อน step()")
        if self.reason is not None:
            raise RuntimeError(f"episode จบไปแล้ว (reason={self.reason}) — ต้อง reset ก่อน")

        a = _validate_action(action)
        t = self.t  # index ของ noise tape — ใช้ค่าของ timestep นี้เสมอ ไม่ว่า action จะเป็นอะไร
        flags = 0

        # ── 1. movement ────────────────────────────────────────────
        if a in (UP, DOWN, LEFT, RIGHT):
            d = a
            if self.config.dynamics.action_noise > 0 and self.tape.slip[t] < self.config.dynamics.action_noise:
                d = PERPENDICULAR[a][int(self.tape.slip_dir[t])]
                self.slips += 1
                flags |= F_SLIPPED
            nx, ny = self.x + DX[d], self.y + DY[d]
            if not self._in_bounds(nx, ny) or self.layout.obstacle[ny, nx]:
                self.collisions += 1  # ชน = อยู่ที่เดิม แต่ timestep ยังเดิน
                flags |= F_COLLISION
            else:
                self.x, self.y = nx, ny
                self.visited[ny, nx] = True
                flags |= F_MOVED
            if self.battery is not None:
                self.battery -= self.config.robot.move_cost

        # ── 2. suck ────────────────────────────────────────────────
        elif a == SUCK:
            if self.dirt[self.y, self.x]:
                if self.layout.sticky[self.y, self.x] and not self.sticky_hit[self.y, self.x]:
                    # ครั้งแรกไม่ติด — ไม่นับเป็น redundant_suck เพราะ agent ไม่มีทางรู้ล่วงหน้า
                    self.sticky_hit[self.y, self.x] = True
                    self.sticky_fails += 1
                    flags |= F_STICKY_FAIL
                else:
                    self.dirt[self.y, self.x] = False
                    self.cleaned += 1
                    flags |= F_CLEANED
            else:
                self.redundant_sucks += 1
                flags |= F_REDUNDANT
            if self.battery is not None:
                self.battery -= self.config.robot.suck_cost

        # ── 3. idle ────────────────────────────────────────────────
        else:
            pass  # ไม่เสียแบต ไม่มี penalty แต่เสีย 1 timestep ซึ่งกดค่า AUC เอง

        self.t += 1
        self._cleaned_at_t.append(self.cleaned)
        self._events.append((a, flags, self._flat_index()))

        terminated, truncated = self._check_termination()
        return self._observation(), 0.0, terminated, truncated, self._info()

    # ── ภายใน ───────────────────────────────────────────────────────

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.config.room.width and 0 <= y < self.config.room.height

    def _flat_index(self) -> int:
        return self.y * self.config.room.width + self.x

    def _check_termination(self) -> tuple[bool, bool]:
        """ตรวจ *หลังจาก* transition เสร็จ ตามลำดับใน §6"""
        if self.config.episode.stop_on_full_coverage and self.cleaned == self.layout.D0:
            self.reason = "complete"
            return True, False
        if self.battery is not None and self.battery <= 0:
            self.reason = "battery"
            return True, False
        if self.t >= self.config.episode.max_steps:
            self.reason = "max_steps"
            return False, True
        return False, False

    def _observation(self) -> dict[str, np.ndarray]:
        draw = None
        if self.tape.sensor is not None:
            draw = self.tape.sensor[min(self.t, len(self.tape.sensor) - 1)]
        return build_observation(
            self.config,
            obstacle=self.layout.obstacle,
            dirt=self.dirt,
            visited=self.visited,
            pos=(self.x, self.y),
            t=self.t,
            battery_left=self.battery,
            sensor_draw=draw,
        )

    def _info(self) -> dict[str, Any]:
        return {
            "cleaned": self.cleaned,
            "collisions": self.collisions,
            "redundant_sucks": self.redundant_sucks,
            "sticky_fails": self.sticky_fails,
            "slips": self.slips,
            "coverage": self.cleaned / self.layout.D0,
            "t": self.t,
            "battery_left": self.battery,
            "reason": self.reason,
        }

    # ── สิ่งที่ runner กับ starter kit เอาไปใช้ต่อ ────────────────────

    def stats(self) -> EpisodeStats:
        """สรุป episode สำหรับส่งให้ `vacuum.scoring`"""
        return EpisodeStats(
            D0=self.layout.D0,
            cleaned_at_t=np.array(self._cleaned_at_t, dtype=np.int64),
            collisions=self.collisions,
            redundant_sucks=self.redundant_sucks,
            sticky_fails=self.sticky_fails,
            slips=self.slips,
            reason=self.reason,
        )

    @property
    def events(self) -> list[tuple[int, int, int]]:
        """(action, flags, flat_index หลัง transition) ต่อ timestep — ใช้เขียน replay (§9)"""
        return list(self._events)

    # ── render ──────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        return None

    def _render_ansi(self) -> str:
        rows = []
        for y in range(self.config.room.height):
            row = []
            for x in range(self.config.room.width):
                if (x, y) == (self.x, self.y):
                    row.append("@")
                elif self.layout.obstacle[y, x]:
                    row.append("#")
                elif self.dirt[y, x]:
                    row.append("*" if self.layout.sticky[y, x] else ".")
                else:
                    row.append(" ")
            rows.append("".join(row))
        header = f"t={self.t} coverage={self.cleaned}/{self.layout.D0}"
        return header + "\n" + "\n".join(rows)

    def _render_rgb(self, cell: int = 8) -> np.ndarray:
        H, W = self.config.room.height, self.config.room.width
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (235, 235, 235)  # พื้นสะอาด
        img[self.layout.obstacle] = (60, 60, 70)
        img[self.dirt] = (170, 130, 80)
        img[self.dirt & self.layout.sticky] = (120, 80, 40)
        img[self.y, self.x] = (40, 110, 220)
        return np.repeat(np.repeat(img, cell, axis=0), cell, axis=1)


def _validate_action(action: Any) -> int:
    if isinstance(action, (np.integer,)):
        action = int(action)
    if isinstance(action, np.ndarray) and action.shape == ():
        action = int(action)
    if not isinstance(action, int) or isinstance(action, bool):
        raise ValueError(
            f"action ต้องเป็น int ในช่วง [0, 5] — ได้ {type(action).__name__} ({action!r})"
        )
    if not 0 <= action <= 5:
        raise ValueError(f"action ต้องอยู่ในช่วง [0, 5] — ได้ {action}")
    return action
