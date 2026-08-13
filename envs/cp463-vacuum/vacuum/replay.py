"""Replay format `.vrp` — environment-spec §9

หลักการ: **เซิร์ฟเวอร์เก็บแค่ log ของสิ่งที่เกิดขึ้น การวาดภาพเกิดบนเบราว์เซอร์ทั้งหมด**
ไฟล์เป็น delta ต่อ timestep (4 ไบต์) ไม่ใช่ snapshot ของ state เต็ม

ข้อกำหนดสำคัญ: เล่น header + body ตั้งแต่ต้นต้องสร้าง state ทุกเฟรมขึ้นมาใหม่ได้ครบ
โดย **client ไม่ต้องรู้จัก RNG เลย** — นี่คือเหตุผลที่ต้องบันทึกผลของการสุ่ม
(`slipped` และตำแหน่งจริง) ลงไป ไม่ใช่หวังว่า client จะสุ่มซ้ำได้เหมือนกัน

    [magic "VRP1"][uint32 LE: ขนาด header ที่บีบแล้ว][header zstd][body zstd]
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import zstandard as zstd

MAGIC = b"VRP1"
FORMAT = "vrp/1"
ZSTD_LEVEL = 10
BODY_ITEM = struct.Struct("<BBH")  # action, flags, flat_index

F_MOVED = 1 << 0
F_COLLISION = 1 << 1
F_SLIPPED = 1 << 2
F_CLEANED = 1 << 3
F_STICKY_FAIL = 1 << 4
F_REDUNDANT = 1 << 5


def _pack_bits(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.reshape(-1))).decode("ascii")


def _unpack_bits(blob: str, H: int, W: int) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(blob), dtype=np.uint8)
    return np.unpackbits(raw)[: H * W].astype(bool).reshape(H, W)


@dataclass(frozen=True)
class ReplayHeader:
    env_version: str
    config_hash: str
    seed: int
    W: int
    H: int
    obstacle_b64: str
    dirt0_b64: str
    sticky_b64: str
    start: tuple[int, int]
    max_steps: int
    D0: int
    format: str = FORMAT

    @property
    def obstacle(self) -> np.ndarray:
        return _unpack_bits(self.obstacle_b64, self.H, self.W)

    @property
    def dirt0(self) -> np.ndarray:
        return _unpack_bits(self.dirt0_b64, self.H, self.W)

    @property
    def sticky(self) -> np.ndarray:
        return _unpack_bits(self.sticky_b64, self.H, self.W)


@dataclass(frozen=True)
class Frame:
    """state ที่สร้างขึ้นใหม่จากการเล่น replay — ต้องตรงกับตอนรันจริงทุกเฟรม"""

    t: int
    pos: tuple[int, int]
    action: int | None  # None ที่เฟรม 0 (ก่อนทำ action แรก)
    flags: int
    dirt: np.ndarray
    visited: np.ndarray
    cleaned: int
    collisions: int
    redundant_sucks: int
    sticky_fails: int
    slips: int


def header_from_env(env) -> ReplayHeader:
    from vacuum import __version__

    layout = env.layout
    return ReplayHeader(
        env_version=__version__,
        config_hash=env.config.config_hash,
        seed=env._seed,
        W=env.config.room.width,
        H=env.config.room.height,
        obstacle_b64=_pack_bits(layout.obstacle),
        dirt0_b64=_pack_bits(layout.dirt0),
        sticky_b64=_pack_bits(layout.sticky),
        start=layout.start,
        max_steps=env.config.episode.max_steps,
        D0=layout.D0,
    )


def encode(header: ReplayHeader, events: list[tuple[int, int, int]]) -> bytes:
    if header.W * header.H > 65536:
        raise ValueError(
            f"grid {header.W}x{header.H} ใหญ่เกินกว่าที่ flat_index แบบ uint16 จะเก็บได้ "
            f"— ต้องขึ้นเวอร์ชันของ replay format"
        )
    cctx = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    head = cctx.compress(json.dumps(asdict(header), sort_keys=True).encode("utf-8"))
    body = cctx.compress(b"".join(BODY_ITEM.pack(a, f, i) for a, f, i in events))
    return MAGIC + struct.pack("<I", len(head)) + head + body


def decode(data: bytes) -> tuple[ReplayHeader, list[tuple[int, int, int]]]:
    if data[:4] != MAGIC:
        raise ValueError("ไม่ใช่ไฟล์ .vrp (magic ไม่ตรง)")
    (head_len,) = struct.unpack("<I", data[4:8])
    dctx = zstd.ZstdDecompressor()
    raw_head = json.loads(dctx.decompress(data[8 : 8 + head_len]).decode("utf-8"))
    if raw_head.get("format") != FORMAT:
        raise ValueError(f"replay format ไม่รองรับ: {raw_head.get('format')!r}")
    raw_head["start"] = tuple(raw_head["start"])
    header = ReplayHeader(**raw_head)

    raw_body = dctx.decompress(data[8 + head_len :])
    events = [tuple(item) for item in BODY_ITEM.iter_unpack(raw_body)]
    return header, events


def write_replay(path: str | Path, env) -> int:
    """เขียน replay ของ episode ล่าสุดของ env — คืนขนาดไฟล์เป็นไบต์"""
    blob = encode(header_from_env(env), env.events)
    Path(path).write_bytes(blob)
    return len(blob)


def read_replay(path: str | Path) -> tuple[ReplayHeader, list[tuple[int, int, int]]]:
    return decode(Path(path).read_bytes())


def frames(header: ReplayHeader, events: list[tuple[int, int, int]]) -> Iterator[Frame]:
    """เล่น replay แล้วคืน state ทุกเฟรม ตั้งแต่ t=0 ถึง t=len(events)

    ไม่แตะ RNG เลย — ทุกอย่างอ่านจาก flags และตำแหน่งที่บันทึกไว้
    """
    W = header.W
    dirt = header.dirt0.copy()
    visited = np.zeros((header.H, W), dtype=bool)
    x, y = header.start
    visited[y, x] = True
    counters = dict(cleaned=0, collisions=0, redundant_sucks=0, sticky_fails=0, slips=0)

    yield Frame(
        t=0, pos=(x, y), action=None, flags=0,
        dirt=dirt.copy(), visited=visited.copy(), **counters,
    )

    for t, (action, flags, flat) in enumerate(events, start=1):
        y, x = divmod(flat, W)
        if flags & F_MOVED:
            visited[y, x] = True
        if flags & F_CLEANED:
            dirt[y, x] = False
            counters["cleaned"] += 1
        if flags & F_COLLISION:
            counters["collisions"] += 1
        if flags & F_SLIPPED:
            counters["slips"] += 1
        if flags & F_STICKY_FAIL:
            counters["sticky_fails"] += 1
        if flags & F_REDUNDANT:
            counters["redundant_sucks"] += 1

        yield Frame(
            t=t, pos=(x, y), action=action, flags=flags,
            dirt=dirt.copy(), visited=visited.copy(), **counters,
        )
