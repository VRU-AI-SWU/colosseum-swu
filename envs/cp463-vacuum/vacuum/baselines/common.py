"""เครื่องมือร่วมของ baseline agents — **ส่วนที่แจกให้นิสิต**

ในนี้มีแต่สิ่งที่เป็น *โครงสร้างข้อมูลและการแปลงรูปแบบ* ไม่ใช่ *อัลกอริทึม*

| แจก | ไม่แจก |
|---|---|
| `decode_pos` — แปลง `obs["pos"]` เป็นพิกัด (ผิดง่ายและไม่มีอะไรสอน) | การวางแผนเส้นทาง (BFS / frontier) |
| `WorldModel` — สะสมสิ่งที่เห็นเป็นแผนที่ โดยเขียนทับด้วยค่าล่าสุด | การกรอง noise ด้วย belief (log-odds) |

**เหตุผล** — สองอย่างที่ไม่แจกคือก้าว Silver → Gold (+0.70) และ Gold → Diamond
ซึ่งเป็นบทเรียนที่โจทย์นี้ตั้งใจให้ฝึก ถ้าแจกโค้ดไป งานที่เหลือของนิสิตคือพิมพ์ตาม
([README §10.4](../../../../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries) ระบุไว้ตั้งแต่แรกว่า
baseline ระดับ Diamond เป็นของลับ)

ส่วน `WorldModel` แจกได้เพราะมันคือการอ่าน observation 3 โหมดให้ถูกรูปแบบ — งานที่ต้อง
เปิด spec แล้วพิมพ์ตาม ไม่ใช่การตัดสินใจเชิงอัลกอริทึม
"""

from __future__ import annotations

import math
from collections import deque

UP, DOWN, LEFT, RIGHT, SUCK, IDLE = range(6)
DX = (0, 0, -1, 1)
DY = (-1, 1, 0, 0)
MOVES = (UP, DOWN, LEFT, RIGHT)


def decode_pos(obs: dict, W: int, H: int) -> tuple[int, int]:
    """คืนตำแหน่งสัมบูรณ์จาก `obs["pos"]` ที่ถูก normalize ไว้"""
    px, py = float(obs["pos"][0]), float(obs["pos"][1])
    return int(round(px * max(W - 1, 1))), int(round(py * max(H - 1, 1)))


class WorldModel:
    """แผนที่สะสมจาก observation — known_free / known_obstacle / known_dirty / unknown

    หมายเหตุเรื่องความไม่แน่นอน: เมื่อ `sensor_noise > 0` ค่าที่อ่านมาผิดได้
    โมเดลนี้ใช้ค่าที่เห็นล่าสุดทับของเดิมเสมอ (ไม่มีการกรอง) ซึ่งเป็นพฤติกรรม
    ที่ทำให้ baseline อ่อนลงเมื่อมี noise — และนั่นคือสิ่งที่ตั้งใจ (§15 การทดลองที่ 1)
    """

    def __init__(self, W: int, H: int, mode: str, window: int | None):
        self.W, self.H = W, H
        self.mode = mode
        self.window = window
        self.reset()

    def reset(self) -> None:
        n = self.W * self.H
        self.known = bytearray(n)
        self.obstacle = bytearray(n)
        self.dirty = bytearray(n)
        self.pos = (0, 0)

    def flat(self, x: int, y: int) -> int:
        return y * self.W + x

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H

    def _mark(self, x: int, y: int, obstacle: float, dirt: float) -> None:
        i = self.flat(x, y)
        self.known[i] = 1
        self.obstacle[i] = 1 if obstacle >= 0.5 else 0
        self.dirty[i] = 1 if dirt >= 0.5 else 0

    def update(self, obs: dict) -> None:
        x, y = decode_pos(obs, self.W, self.H)
        self.pos = (x, y)
        grid = obs["grid"]

        if self.mode == "full":
            for gy in range(self.H):
                for gx in range(self.W):
                    self._mark(gx, gy, grid[0][gy][gx], grid[1][gy][gx])

        elif self.mode == "local":
            k = self.window
            r = k // 2
            for j in range(k):
                for i in range(k):
                    gx, gy = x - r + i, y - r + j
                    if self.in_bounds(gx, gy):  # นอกขอบไม่ต้องจำ (เป็นกำแพงโดยปริยาย)
                        self._mark(gx, gy, grid[0][j][i], grid[1][j][i])

        else:  # sensor — [current, UP, DOWN, LEFT, RIGHT]
            offsets = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))
            for idx, (dx, dy) in enumerate(offsets):
                gx, gy = x + dx, y + dy
                if self.in_bounds(gx, gy):
                    self._mark(gx, gy, grid[idx][0], grid[idx][1])

        # หุ่นยืนอยู่ตรงนี้ได้ แปลว่าช่องนี้ไม่ใช่กำแพงแน่นอน — แก้ค่าที่ sensor อ่านผิด
        self.obstacle[self.flat(x, y)] = 0
        self._pin_free(x, y)

    def _pin_free(self, x: int, y: int) -> None:
        """hook ให้ subclass ที่เก็บหลักฐานสะสมล้างความเชื่อเรื่องกำแพงของช่องที่ยืนอยู่ด้วย"""

    # ── query ───────────────────────────────────────────────────────

    def dirty_here(self) -> bool:
        return bool(self.dirty[self.flat(*self.pos)])

    def is_known_free(self, x: int, y: int) -> bool:
        i = self.flat(x, y)
        return bool(self.known[i]) and not self.obstacle[i]

    def is_unknown(self, x: int, y: int) -> bool:
        return not self.known[self.flat(x, y)]

    def known_wall(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return True
        i = self.flat(x, y)
        return bool(self.known[i]) and bool(self.obstacle[i])


__all__ = [
    "UP", "DOWN", "LEFT", "RIGHT", "SUCK", "IDLE", "DX", "DY", "MOVES",
    "WorldModel", "decode_pos",
]
