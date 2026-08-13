"""Observation encoding — environment-spec §4

ทุกโหมดคืน dict โครงสร้างเดียวกัน ต่างกันแค่ `grid`

    {"grid": ndarray, "pos": float32[2], "scalars": float32[2]}

`scalars` ไม่มี coverage โดยตั้งใจ — ในโหมด local/sensor การบอก coverage รวม
เท่ากับแอบให้ข้อมูลทั้งแผนที่

── การตัดสินใจที่ไม่ได้อยู่ใน spec: sensor_noise กับโหมดที่ไม่ใช่ `sensor` ─────
env-spec §4.3 นิยาม `sensor_noise` ไว้ใต้หัวข้อโหมด `sensor` เท่านั้น แต่ §11
ตั้ง phase Final เป็น `observation: local` + `sensor_noise: 0.05` จึงต้องนิยามเพิ่ม

ที่ implement ไว้: **พลิกเฉพาะ channel ที่เป็นค่าจากเซนเซอร์ คือ obstacle กับ dirt**
ทุกโหมด ไม่พลิก `visited` (เป็นร่องรอยที่หุ่นจำเอง ไม่ใช่การอ่านค่าจากเซนเซอร์)
และไม่พลิก channel ตำแหน่งหุ่นในโหมด `full` (ไม่งั้นหุ่นจะ "หายไป" หรือโผล่สองที่)
ลำดับของค่าที่ถูกพลิกตรึงไว้เป็น obstacle ทั้งผืนก่อน แล้วตามด้วย dirt (row-major)
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from vacuum.config import Config

# ลำดับ cell ในโหมด sensor — ตายตัว
SENSOR_ORDER = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))  # current, UP, DOWN, LEFT, RIGHT


def sensed_size(config: Config) -> int:
    """จำนวนค่าที่ `sensor_noise` มีสิทธิ์พลิกต่อหนึ่ง observation"""
    mode = config.robot.observation
    if mode == "full":
        return 2 * config.room.height * config.room.width
    if mode == "local":
        k = config.robot.observation_window
        return 2 * k * k
    return 2 * len(SENSOR_ORDER)


def grid_shape(config: Config) -> tuple[int, ...]:
    mode = config.robot.observation
    if mode == "full":
        return (4, config.room.height, config.room.width)
    if mode == "local":
        k = config.robot.observation_window
        return (3, k, k)
    return (len(SENSOR_ORDER), 2)


def observation_space(config: Config) -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(low=0.0, high=1.0, shape=grid_shape(config), dtype=np.float32),
            "pos": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "scalars": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        }
    )


def _window(plane: np.ndarray, x: int, y: int, k: int, outside: float) -> np.ndarray:
    """ตัดหน้าต่าง k×k รอบ (x, y) โดยเติมค่านอกขอบด้วย `outside`"""
    H, W = plane.shape
    r = k // 2
    out = np.full((k, k), outside, dtype=np.float32)
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    out[y0 - (y - r) : y1 - (y - r), x0 - (x - r) : x1 - (x - r)] = plane[y0:y1, x0:x1]
    return out


def build_observation(
    config: Config,
    *,
    obstacle: np.ndarray,
    dirt: np.ndarray,
    visited: np.ndarray,
    pos: tuple[int, int],
    t: int,
    battery_left: int | None,
    sensor_draw: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """สร้าง observation ที่ timestep `t`

    `sensor_draw` คือแถวที่ `t` ของ sensor tape (หรือ None ถ้าไม่มี sensor noise)
    — ต้องเป็นแถวที่ index ด้วย t เสมอ ไม่ว่า agent จะทำ action อะไร (§2 common random numbers)
    """
    W, H = config.room.width, config.room.height
    x, y = pos
    mode = config.robot.observation

    obs_f = obstacle.astype(np.float32)
    dirt_f = dirt.astype(np.float32)

    if mode == "full":
        robot = np.zeros((H, W), dtype=np.float32)
        robot[y, x] = 1.0
        sensed = np.stack([obs_f, dirt_f])  # (2, H, W)
        sensed = _apply_sensor_noise(sensed, sensor_draw, config)
        grid = np.concatenate([sensed, visited.astype(np.float32)[None], robot[None]])

    elif mode == "local":
        k = config.robot.observation_window
        sensed = np.stack(
            [
                _window(obs_f, x, y, k, outside=1.0),  # นอกขอบคือกำแพง
                _window(dirt_f, x, y, k, outside=0.0),
            ]
        )
        sensed = _apply_sensor_noise(sensed, sensor_draw, config)
        seen = _window(visited.astype(np.float32), x, y, k, outside=0.0)
        grid = np.concatenate([sensed, seen[None]])

    else:  # sensor
        cells = np.zeros((len(SENSOR_ORDER), 2), dtype=np.float32)
        for i, (dx, dy) in enumerate(SENSOR_ORDER):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                cells[i, 0] = obs_f[ny, nx]
                cells[i, 1] = dirt_f[ny, nx]
            else:
                cells[i, 0] = 1.0  # นอกขอบคือกำแพง
                cells[i, 1] = 0.0
        # จัดให้ลำดับการพลิกเป็น "obstacle ทั้งชุดก่อน แล้ว dirt" เหมือนโหมดอื่น
        sensed = np.stack([cells[:, 0], cells[:, 1]])
        sensed = _apply_sensor_noise(sensed, sensor_draw, config)
        grid = sensed.T.copy()

    battery_init = config.robot.battery
    battery_frac = 1.0 if battery_init is None else float(battery_left) / float(battery_init)

    return {
        "grid": np.ascontiguousarray(grid, dtype=np.float32),
        "pos": np.array(
            [x / max(W - 1, 1), y / max(H - 1, 1)],
            dtype=np.float32,
        ),
        "scalars": np.array(
            [t / config.episode.max_steps, max(0.0, battery_frac)],
            dtype=np.float32,
        ),
    }


def _apply_sensor_noise(
    sensed: np.ndarray, draw: np.ndarray | None, config: Config
) -> np.ndarray:
    """พลิกค่า 0↔1 อย่างเป็นอิสระต่อกันด้วยความน่าจะเป็น `sensor_noise`"""
    if draw is None or config.dynamics.sensor_noise <= 0.0:
        return sensed
    flip = (draw < config.dynamics.sensor_noise).reshape(sensed.shape)
    return np.where(flip, 1.0 - sensed, sensed).astype(np.float32)
