"""แผนที่สะสม + การสร้าง feature ให้ policy — **ใช้ไฟล์นี้ทั้งตอนเทรนและตอน inference**

ทำไมต้องมี: observation ที่ environment ให้คือหน้าต่าง 5×5 รอบตัว (POMDP)
policy แบบ MLP ที่ไม่มีความจำมองเห็นแค่นั้นจึงทำ coverage ไม่ได้เลยโดยหลักการ
— มันไม่รู้ว่าเคยไปตรงไหนมาแล้ว

ทางเลือกมาตรฐานมีสองทาง

| ทาง | ข้อดี | ข้อเสีย |
|---|---|---|
| Recurrent policy (LSTM) เรียนรู้ที่จะจำเอง | ไม่ต้องออกแบบ state | กินตัวอย่างมหาศาลเมื่อ episode ยาว 1,500 step |
| **แยก state estimation ออกมาเขียนเอง แล้วให้ policy ตัดสินใจบน belief** | เรียนรู้ได้เร็วกว่ามาก · เป็นสิ่งที่ Silver/Gold ทำอยู่แล้ว | ต้องออกแบบ feature เอง (ซึ่งก็เป็นส่วนหนึ่งของการบ้าน) |

ที่นี่เลือกทางที่สอง — และนี่คือรูปแบบที่นิสิตทำได้เหมือนกัน เพราะแพลตฟอร์มเห็นแค่ `act()`
จะเก็บ state อะไรไว้ข้างในก็เรื่องของ agent

⚠️ **feature ที่สร้างตอนเทรนกับตอนประเมินต้องเหมือนกันเป๊ะ** ถ้าต่างกันแม้แต่การเรียงแกน
policy จะทำงานผิดแบบเงียบๆ — การใช้ module เดียวกันทั้งสองฝั่งคือสิ่งที่การันตีข้อนี้
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

CROP = 15  # ขนาดหน้าต่าง ego-centric ที่ตัดจากแผนที่สะสม (ต้องเป็นเลขคี่)
N_CHANNELS = 4  # unknown · obstacle · dirty · visited
N_SCALARS = 5

SENSOR_OFFSETS = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))  # current, UP, DOWN, LEFT, RIGHT


def feature_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "map": spaces.Box(0.0, 1.0, shape=(N_CHANNELS, CROP, CROP), dtype=np.float32),
            "scalars": spaces.Box(0.0, 1.0, shape=(N_SCALARS,), dtype=np.float32),
        }
    )


class MapMemory:
    """แผนที่สะสมที่ agent สร้างเองจาก observation ที่เห็นมาทั้งหมด

    เก็บ *ความเชื่อ* ไม่ใช่ความจริง — เมื่อ `sensor_noise > 0` ค่าที่อ่านมาผิดได้
    `confidence` นับจำนวนครั้งที่เห็นแต่ละช่อง แล้วใช้ถ่วงน้ำหนักการเขียนทับ
    (ช่องที่เพิ่งเห็นครั้งแรกเปลี่ยนใจง่าย ช่องที่เห็นซ้ำหลายครั้งเปลี่ยนใจยาก)
    — เป็นตัวกรอง noise แบบถูกที่สุดที่ยังได้ผล และเป็นจุดที่ planner แบบ Gold ไม่มี
    """

    def __init__(self, config: dict):
        self.W = int(config["width"])
        self.H = int(config["height"])
        self.mode = config["observation"]
        self.window = config.get("observation_window")
        self.max_steps = int(config["max_steps"])
        self.reset()

    def reset(self) -> None:
        shape = (self.H, self.W)
        self.seen = np.zeros(shape, dtype=np.float32)  # จำนวนครั้งที่เคยเห็นช่องนี้
        self.obstacle = np.zeros(shape, dtype=np.float32)  # หลักฐานสะสม ∈ [0,1]
        self.dirty = np.zeros(shape, dtype=np.float32)
        self.visited = np.zeros(shape, dtype=np.float32)
        self.pos = (0, 0)
        self.t = 0

    # ── ดูดข้อมูลจาก observation ────────────────────────────────────

    def update(self, obs: dict) -> None:
        grid = np.asarray(obs["grid"], dtype=np.float32)
        px, py = np.asarray(obs["pos"], dtype=np.float64)
        x = int(round(float(px) * max(self.W - 1, 1)))
        y = int(round(float(py) * max(self.H - 1, 1)))
        self.pos = (x, y)
        self.t = int(round(float(obs["scalars"][0]) * self.max_steps))
        self.visited[y, x] = 1.0

        if self.mode == "full":
            ys, xs = np.mgrid[0 : self.H, 0 : self.W]
            self._absorb(xs.ravel(), ys.ravel(), grid[0].ravel(), grid[1].ravel())

        elif self.mode == "local":
            k = self.window
            r = k // 2
            gx, gy = np.meshgrid(np.arange(x - r, x + r + 1), np.arange(y - r, y + r + 1))
            inside = (gx >= 0) & (gx < self.W) & (gy >= 0) & (gy < self.H)
            self._absorb(gx[inside], gy[inside], grid[0][inside], grid[1][inside])

        else:  # sensor
            xs, ys, obs_v, dirt_v = [], [], [], []
            for i, (dx, dy) in enumerate(SENSOR_OFFSETS):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    xs.append(nx); ys.append(ny)
                    obs_v.append(grid[i, 0]); dirt_v.append(grid[i, 1])
            self._absorb(np.array(xs), np.array(ys), np.array(obs_v), np.array(dirt_v))

        # หุ่นยืนอยู่ตรงนี้ได้ = ไม่ใช่กำแพงแน่นอน · ความเชื่ออื่นสู้หลักฐานนี้ไม่ได้
        self.obstacle[y, x] = 0.0
        self.seen[y, x] = max(self.seen[y, x], 3.0)

    def _absorb(self, xs, ys, obstacle, dirt) -> None:
        """รวมหลักฐานใหม่เข้ากับของเดิมแบบค่าเฉลี่ยเคลื่อนที่ถ่วงด้วยจำนวนครั้งที่เห็น"""
        if len(xs) == 0:
            return
        xs = np.asarray(xs, dtype=int); ys = np.asarray(ys, dtype=int)
        n = self.seen[ys, xs]
        w = 1.0 / (n + 1.0)  # เห็นครั้งแรก w=1 (เชื่อเต็ม) · เห็นครั้งที่ 4 w=0.25
        self.obstacle[ys, xs] += w * (np.asarray(obstacle, dtype=np.float32) - self.obstacle[ys, xs])
        self.dirty[ys, xs] += w * (np.asarray(dirt, dtype=np.float32) - self.dirty[ys, xs])
        self.seen[ys, xs] = n + 1.0

    # ── แปลงเป็น feature ให้ policy ─────────────────────────────────

    def features(self) -> dict[str, np.ndarray]:
        x, y = self.pos
        r = CROP // 2
        planes = np.stack(
            [
                (self.seen == 0).astype(np.float32),  # unknown
                self.obstacle,
                self.dirty,
                self.visited,
            ]
        )
        # นอกขอบ grid: unknown=0 · obstacle=1 (เป็นกำแพง) · dirty=0 · visited=0
        pad = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        crop = np.repeat(pad[:, None, None], CROP, axis=1).repeat(CROP, axis=2).copy()

        y0, y1 = max(0, y - r), min(self.H, y + r + 1)
        x0, x1 = max(0, x - r), min(self.W, x + r + 1)
        crop[:, y0 - (y - r) : y1 - (y - r), x0 - (x - r) : x1 - (x - r)] = planes[:, y0:y1, x0:x1]

        known = self.seen > 0
        n_known = float(known.sum())
        scalars = np.array(
            [
                x / max(self.W - 1, 1),
                y / max(self.H - 1, 1),
                min(self.t / self.max_steps, 1.0),
                float(self.dirty[known].sum()) / max(n_known, 1.0),  # สัดส่วนที่ยังเชื่อว่าสกปรก
                n_known / float(self.W * self.H),  # สำรวจไปแล้วกี่ส่วน
            ],
            dtype=np.float32,
        )
        return {"map": crop.astype(np.float32), "scalars": np.clip(scalars, 0.0, 1.0)}
