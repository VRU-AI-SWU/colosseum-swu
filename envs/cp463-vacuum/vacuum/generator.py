"""สร้างห้องจาก seed — environment-spec §2 (RNG) และ §3 (อัลกอริทึมสร้างห้อง)

    seed → ① วางสิ่งกีดขวาง → ② ถมช่องที่เข้าไม่ถึง → ③ โปรยฝุ่น → ④ เลือกช่องเหนียว → ⑤ วางหุ่น

ลำดับสลับไม่ได้ และลำดับการ draw จาก layout_rng ห้ามสลับ (§2)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from vacuum.config import Config

# ── สายสุ่ม (§2) ────────────────────────────────────────────────────
# แยกสายเพราะการเปลี่ยน max_steps (ซึ่งเปลี่ยนจำนวน draw ของ noise)
# ต้องไม่ทำให้ผังห้องของ seed เดิมเปลี่ยน
LAYOUT_STREAM = 0x5EED
NOISE_STREAM = 0xA11CE
SENSOR_STREAM = 0x53E4

# ทิศตาม action 0..3 — UP, DOWN, LEFT, RIGHT
DX = (0, 0, -1, 1)
DY = (-1, 1, 0, 0)


def _stream(seed: int, stream_id: int) -> np.random.Generator:
    """`Generator(PCG64)` เท่านั้น — ห้ามใช้ `random` ของ Python หรือ global numpy RNG (§2 ข้อ 1)"""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), stream_id])))


def layout_rng(seed: int) -> np.random.Generator:
    return _stream(seed, LAYOUT_STREAM)


def noise_rng(seed: int) -> np.random.Generator:
    return _stream(seed, NOISE_STREAM)


def sensor_rng(seed: int) -> np.random.Generator:
    return _stream(seed, SENSOR_STREAM)


@dataclass(frozen=True)
class Layout:
    """ผังห้องที่ generate จาก seed — ทุกฟิลด์คงที่ตลอด episode ยกเว้นที่ env คัดลอกไปแก้"""

    obstacle: np.ndarray  # bool[H, W]
    dirt0: np.ndarray  # bool[H, W] — ฝุ่นตอนเริ่ม
    sticky: np.ndarray  # bool[H, W] — subset ของ dirt0
    start: tuple[int, int]  # (x, y)
    D0: int  # จำนวน cell สกปรกตอนเริ่ม (ตัวหารของ coverage)
    free_count: int
    effective_density: float  # ความหนาแน่นจริงหลังถมช่องที่เข้าไม่ถึง (§3.2)


# ── §3.1 obstacle ──────────────────────────────────────────────────


def generate_obstacles(
    rng: np.random.Generator, W: int, H: int, density: float, generator: str
) -> np.ndarray:
    obs = np.zeros((H, W), dtype=bool)
    target = int(round(density * W * H))
    if target == 0:
        return obs

    if generator == "random":
        idx = rng.permutation(W * H)[:target]
        obs.reshape(-1)[idx] = True

    elif generator == "clustered":
        # random walk: หย่อนจุดเริ่มลงมั่ว แล้วเดินสุ่มต่อ 3–8 ก้าว ระบายช่องที่เดินผ่านเป็นกำแพง
        # เดินทับช่องเดิมได้ (ไม่นับซ้ำ) → ก้อนแน่นขึ้นแทนที่จะยืดยาว
        placed, guard = 0, 0
        while placed < target and guard < 10_000:
            guard += 1
            x = int(rng.integers(0, W))
            y = int(rng.integers(0, H))
            run_len = int(rng.integers(3, 9))  # 3..8
            for _ in range(run_len):
                if not obs[y, x]:
                    obs[y, x] = True
                    placed += 1
                    if placed >= target:
                        break
                d = int(rng.integers(0, 4))
                nx, ny = x + DX[d], y + DY[d]
                if 0 <= nx < W and 0 <= ny < H:
                    x, y = nx, ny
    else:
        raise ValueError(generator)
    return obs


# ── §3.2 บังคับ connectivity (ไม่ใช้ RNG) ──────────────────────────


def largest_free_component(obstacle: np.ndarray) -> np.ndarray:
    """คืน mask ของก้อนพื้นที่ว่างที่เชื่อมต่อกันแบบ 4 ทิศที่ใหญ่ที่สุด

    เสมอ → เลือก component ที่มี flat_index ต่ำสุด ซึ่งได้มาฟรีจากการไล่ scan
    flat index จากน้อยไปมาก (component ที่เจอก่อนมี flat_index ต่ำกว่าเสมอ)
    """
    H, W = obstacle.shape
    free = ~obstacle
    seen = np.zeros((H, W), dtype=bool)
    best: np.ndarray | None = None
    best_size = -1

    for flat in range(H * W):
        y, x = divmod(flat, W)
        if not free[y, x] or seen[y, x]:
            continue
        comp = np.zeros((H, W), dtype=bool)
        queue = deque([(x, y)])
        seen[y, x] = True
        comp[y, x] = True
        size = 0
        while queue:
            cx, cy = queue.popleft()
            size += 1
            for d in range(4):
                nx, ny = cx + DX[d], cy + DY[d]
                if 0 <= nx < W and 0 <= ny < H and free[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    comp[ny, nx] = True
                    queue.append((nx, ny))
        if size > best_size:  # `>` ไม่ใช่ `>=` → เสมอแล้วเก็บอันที่เจอก่อน = flat_index ต่ำสุด
            best_size = size
            best = comp

    return best if best is not None else np.zeros((H, W), dtype=bool)


def enforce_connectivity(obstacle: np.ndarray) -> np.ndarray:
    """ถมช่องว่างที่เข้าไม่ถึงให้เป็นกำแพง — จบใน pass เดียว ไม่ใช้ค่าสุ่ม ไม่มีทางวนไม่จบ"""
    free = ~obstacle
    largest = largest_free_component(obstacle)
    return obstacle | (free & ~largest)


# ── §3.3 dirt ──────────────────────────────────────────────────────


def generate_dirt(
    rng: np.random.Generator,
    obstacle: np.ndarray,
    dirt_ratio: float,
    distribution: str,
) -> tuple[np.ndarray, int]:
    H, W = obstacle.shape
    free_idx = np.flatnonzero(~obstacle.reshape(-1))  # เรียงจากน้อยไปมากอยู่แล้ว
    n_free = len(free_idx)
    # เลือกแบบไม่คืนที่ด้วยจำนวนตายตัว ไม่ใช่ Bernoulli รายช่อง → D0 ไม่แกว่งระหว่าง seed
    n_dirt = max(1, int(round(dirt_ratio * n_free)))

    if distribution == "uniform":
        w = np.ones(n_free, dtype=np.float64)
    elif distribution == "clustered":
        k = max(1, int(round(n_free * dirt_ratio / 25)))
        centers = rng.choice(free_idx, size=k, replace=False)
        cy, cx = np.divmod(centers, W)
        fy, fx = np.divmod(free_idx, W)
        # ระยะ Manhattan ถึงศูนย์กลางที่ใกล้ที่สุด
        d = np.min(
            np.abs(fx[:, None] - cx[None, :]) + np.abs(fy[:, None] - cy[None, :]),
            axis=1,
        ).astype(np.float64)
        w = np.exp(-d / 3.0)  # 3 = "รัศมีที่รู้สึกได้" — ห่าง 3 ช่อง เหลือโอกาสราว 37%
    else:
        raise ValueError(distribution)

    chosen = rng.choice(free_idx, size=n_dirt, replace=False, p=w / w.sum())
    dirt = np.zeros((H, W), dtype=bool)
    dirt.reshape(-1)[chosen] = True
    return dirt, n_dirt


# ── §3.4 sticky ────────────────────────────────────────────────────


def select_sticky(
    rng: np.random.Generator, dirt: np.ndarray, sticky_ratio: float, D0: int
) -> np.ndarray:
    sticky = np.zeros_like(dirt)
    n_sticky = int(round(sticky_ratio * D0))
    if n_sticky > 0:
        dirty_idx = np.flatnonzero(dirt.reshape(-1))
        chosen = rng.choice(dirty_idx, size=n_sticky, replace=False)
        sticky.reshape(-1)[chosen] = True
    return sticky


# ── §3.5 robot start ───────────────────────────────────────────────


def choose_start(
    rng: np.random.Generator, obstacle: np.ndarray, mode: str
) -> tuple[int, int]:
    H, W = obstacle.shape
    free_idx = np.flatnonzero(~obstacle.reshape(-1))

    if mode == "random":
        flat = int(rng.choice(free_idx))
    else:
        fy, fx = np.divmod(free_idx, W)
        if mode == "corner":
            key = fx + fy
        elif mode == "center":
            key = np.abs(fx - W // 2) + np.abs(fy - H // 2)
        else:
            raise ValueError(mode)
        # เสมอ → flat_index ต่ำสุด (free_idx เรียงจากน้อยไปมาก, argmin คืนตัวแรกที่น้อยที่สุด)
        flat = int(free_idx[int(np.argmin(key))])

    y, x = divmod(flat, W)
    return x, y


# ── ประกอบทั้งหมด ──────────────────────────────────────────────────


def generate_layout(config: Config, seed: int) -> Layout:
    """สร้างผังห้องจาก seed — ลำดับการ draw ตาม §2 ห้ามสลับ"""
    W, H = config.room.width, config.room.height
    rng = layout_rng(seed)

    obstacle = generate_obstacles(
        rng, W, H, config.room.obstacle_density, config.room.obstacle_generator
    )
    obstacle = enforce_connectivity(obstacle)

    free_count = int((~obstacle).sum())
    if free_count < 4:
        raise ValueError(
            f"config นี้ให้ห้องที่มีช่องว่างเหลือ {free_count} ช่อง (seed={seed}) — "
            f"ลด room.obstacle_density หรือเพิ่มขนาดห้อง"
        )

    dirt0, D0 = generate_dirt(rng, obstacle, config.room.dirt_ratio, config.room.dirt_distribution)
    sticky = select_sticky(rng, dirt0, config.dynamics.sticky_dirt, D0)
    start = choose_start(rng, obstacle, config.robot.start)

    return Layout(
        obstacle=obstacle,
        dirt0=dirt0,
        sticky=sticky,
        start=start,
        D0=D0,
        free_count=free_count,
        effective_density=float(obstacle.sum()) / float(W * H),
    )


# ── noise tape (§2 common random numbers) ──────────────────────────


@dataclass(frozen=True)
class NoiseTape:
    """สุ่มไว้ล่วงหน้าทั้งม้วน แล้วอ้างด้วย `tape[t]` ตาม index ของ timestep

    *อ้างตาม index* ไม่ใช่ *ดึงค่าถัดไป* — ทำให้สองทีมที่ทำ action ต่างกัน
    ยังเจอ "ดวง" ก้อนเดียวกันที่ timestep เดียวกัน (common random numbers)
    """

    slip: np.ndarray  # float64[max_steps]
    slip_dir: np.ndarray  # int[max_steps] — 0 หรือ 1
    sensor: np.ndarray | None  # float64[max_steps + 1, n_sensed] หรือ None ถ้า sensor_noise = 0


def make_noise_tape(config: Config, seed: int, n_sensed: int) -> NoiseTape:
    T = config.episode.max_steps
    rng = noise_rng(seed)
    slip = rng.random(T)
    slip_dir = rng.integers(0, 2, T)

    sensor = None
    if config.dynamics.sensor_noise > 0:
        # max_steps + 1 แถว เพราะมี observation ที่ t = 0 (จาก reset) ไปจนถึง t = max_steps
        sensor = sensor_rng(seed).random((T + 1, n_sensed))

    return NoiseTape(slip=slip, slip_dir=slip_dir, sensor=sensor)
