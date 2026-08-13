"""โหลด + validate config และคำนวณ config_hash — environment-spec §12

การ validate ทั้งหมดเกิดที่นี่ที่เดียว และต้อง raise ตั้งแต่ตอนโหลด config
ไม่ใช่ตอนรัน episode (env-spec §3.2: "raise error ตอนสร้าง competition ไม่ใช่ตอนรัน")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

OBSERVATION_MODES = ("full", "local", "sensor")
OBSTACLE_GENERATORS = ("random", "clustered")
DIRT_DISTRIBUTIONS = ("uniform", "clustered")
START_MODES = ("random", "corner", "center")

# generator ที่ประกาศไว้ใน overview.md แต่ยังไม่ implement ใน v1.0.0
NOT_IMPLEMENTED_GENERATORS = ("rooms", "fixed")
NOT_IMPLEMENTED_DISTRIBUTIONS = ("patchy",)


class ConfigError(ValueError):
    """config ที่ใช้สร้าง environment ไม่ได้ — ข้อความต้องบอกวิธีแก้เสมอ"""


def _check_prob(value: float, path: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{path} ต้องอยู่ในช่วง [0, 1] — ได้ {value}")
    return value


@dataclass(frozen=True)
class Room:
    width: int = 20
    height: int = 20
    obstacle_density: float = 0.15
    obstacle_generator: str = "clustered"
    dirt_distribution: str = "uniform"
    dirt_ratio: float = 0.60
    guarantee_connected: bool = True

    def validate(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ConfigError(f"room.width/height ต้อง >= 2 — ได้ {self.width}x{self.height}")
        if self.obstacle_generator in NOT_IMPLEMENTED_GENERATORS:
            raise ConfigError(
                f"room.obstacle_generator: {self.obstacle_generator!r} ยังไม่ implement ใน v1.0.0 "
                f"(env-spec §3.1) — ถ้าต้องใช้ต้องขึ้น env_version พร้อม conformance test ใหม่"
            )
        if self.obstacle_generator not in OBSTACLE_GENERATORS:
            raise ConfigError(
                f"room.obstacle_generator ต้องเป็นหนึ่งใน {OBSTACLE_GENERATORS} — ได้ {self.obstacle_generator!r}"
            )
        if self.dirt_distribution in NOT_IMPLEMENTED_DISTRIBUTIONS:
            raise ConfigError(
                f"room.dirt_distribution: {self.dirt_distribution!r} ยังไม่ implement ใน v1.0.0 (env-spec §3.3)"
            )
        if self.dirt_distribution not in DIRT_DISTRIBUTIONS:
            raise ConfigError(
                f"room.dirt_distribution ต้องเป็นหนึ่งใน {DIRT_DISTRIBUTIONS} — ได้ {self.dirt_distribution!r}"
            )
        _check_prob(self.obstacle_density, "room.obstacle_density")
        if not 0.0 < float(self.dirt_ratio) <= 1.0:
            raise ConfigError(f"room.dirt_ratio ต้องอยู่ในช่วง (0, 1] — ได้ {self.dirt_ratio}")
        if not self.guarantee_connected:
            raise ConfigError(
                "room.guarantee_connected: false ทำให้ coverage แตะ 100% ไม่ได้ในบาง seed "
                "และ completion_bonus กลายเป็นของที่แจกไม่ได้ (env-spec §3.2) — v1.0.0 ไม่รองรับ"
            )


@dataclass(frozen=True)
class Robot:
    start: str = "random"
    observation: str = "local"
    observation_window: int | None = 5
    battery: int | None = None
    move_cost: int = 1
    suck_cost: int = 2

    def validate(self) -> None:
        if self.start not in START_MODES:
            raise ConfigError(f"robot.start ต้องเป็นหนึ่งใน {START_MODES} — ได้ {self.start!r}")
        if self.observation not in OBSERVATION_MODES:
            raise ConfigError(
                f"robot.observation ต้องเป็นหนึ่งใน {OBSERVATION_MODES} — ได้ {self.observation!r}"
            )
        if self.observation == "local":
            k = self.observation_window
            if k is None:
                raise ConfigError("robot.observation: local ต้องระบุ robot.observation_window ด้วย")
            if k < 3 or k % 2 == 0:
                raise ConfigError(
                    f"robot.observation_window ต้องเป็นเลขคี่ >= 3 (หุ่นต้องอยู่กลางหน้าต่างพอดี) — ได้ {k}"
                )
        if self.battery is not None and self.battery <= 0:
            raise ConfigError(f"robot.battery ต้องเป็น null หรือจำนวนเต็มบวก — ได้ {self.battery}")
        if self.move_cost < 0 or self.suck_cost < 0:
            raise ConfigError("robot.move_cost / robot.suck_cost ต้องไม่ติดลบ")


@dataclass(frozen=True)
class Dynamics:
    action_noise: float = 0.10
    sticky_dirt: float = 0.15
    sensor_noise: float = 0.0

    def validate(self) -> None:
        _check_prob(self.action_noise, "dynamics.action_noise")
        _check_prob(self.sticky_dirt, "dynamics.sticky_dirt")
        _check_prob(self.sensor_noise, "dynamics.sensor_noise")


@dataclass(frozen=True)
class Episode:
    max_steps: int = 1500
    stop_on_full_coverage: bool = True
    step_timeout_ms: int = 1000

    def validate(self) -> None:
        if self.max_steps < 1:
            raise ConfigError(f"episode.max_steps ต้อง >= 1 — ได้ {self.max_steps}")
        if self.step_timeout_ms < 1:
            raise ConfigError("episode.step_timeout_ms ต้อง >= 1 (เป็นตัวกันงานค้าง ไม่มีผลต่อคะแนน)")


@dataclass(frozen=True)
class Penalties:
    collision: float = 1.0
    redundant_suck: float = 0.2

    def validate(self) -> None:
        if self.collision < 0 or self.redundant_suck < 0:
            raise ConfigError("penalties.* ต้องไม่ติดลบ")


@dataclass(frozen=True)
class Scoring:
    metric: str = "coverage_auc"
    completion_bonus: float = 1.0
    max_penalty: float = 0.2

    def validate(self) -> None:
        if self.metric != "coverage_auc":
            raise ConfigError(f"scoring.metric รองรับเฉพาะ 'coverage_auc' ใน v1.0.0 — ได้ {self.metric!r}")
        if self.completion_bonus < 0:
            raise ConfigError("scoring.completion_bonus ต้องไม่ติดลบ")
        if self.max_penalty < 0:
            raise ConfigError("scoring.max_penalty ต้องไม่ติดลบ")


@dataclass(frozen=True)
class Config:
    task: str = "vacuum_gridworld"
    version: str = "1.0.0"
    phase: str | None = None
    room: Room = field(default_factory=Room)
    robot: Robot = field(default_factory=Robot)
    dynamics: Dynamics = field(default_factory=Dynamics)
    episode: Episode = field(default_factory=Episode)
    penalties: Penalties = field(default_factory=Penalties)
    scoring: Scoring = field(default_factory=Scoring)

    def __post_init__(self) -> None:
        if self.task != "vacuum_gridworld":
            raise ConfigError(f"task ต้องเป็น 'vacuum_gridworld' — ได้ {self.task!r}")
        for section in (self.room, self.robot, self.dynamics, self.episode, self.penalties, self.scoring):
            section.validate()
        if not self.episode.stop_on_full_coverage:
            # ไม่ raise เพราะ spec อนุญาตให้ใช้ debug ได้ แต่ห้ามใช้ในคอนฟิกที่ตัดสินคะแนน
            import warnings

            warnings.warn(
                "episode.stop_on_full_coverage: false เปลี่ยนความหมายของ AUC — "
                "ใช้ debug เท่านั้น ห้ามใช้ใน config ที่ตัดสินคะแนน (env-spec §6)",
                stacklevel=3,
            )

    # ── การคำนวณ hash ───────────────────────────────────────────────
    def normalized(self) -> dict[str, Any]:
        """dict ที่ normalize แล้ว (เรียง key, ไม่มี comment) ใช้คำนวณ hash

        `phase` ไม่เข้า hash เพราะเป็นแค่ชื่อเรียกของมนุษย์ ไม่มีผลต่อพฤติกรรมของ environment
        ส่วนค่า seed ไม่เคยอยู่ใน config ของ environment (เป็นเรื่องของ runner) จึงไม่เข้า hash
        """
        data = asdict(self)
        data.pop("phase", None)
        return data

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def replace(self, **overrides: Any) -> "Config":
        """สร้าง config ใหม่โดยแทนที่ค่าบางตัว — ใช้กับ Phase.config_override และการ calibrate

        รับ key แบบ dotted เช่น `replace(**{"dynamics.action_noise": 0.2})`
        """
        data = asdict(self)
        for key, value in overrides.items():
            if "." in key:
                section, name = key.split(".", 1)
                if section not in data or not isinstance(data[section], dict):
                    raise ConfigError(f"ไม่รู้จัก section {section!r} ใน config")
                if name not in data[section]:
                    raise ConfigError(f"ไม่รู้จักฟิลด์ {key!r} ใน config")
                data[section][name] = value
            else:
                if key not in data:
                    raise ConfigError(f"ไม่รู้จักฟิลด์ {key!r} ใน config")
                data[key] = value
        return from_dict(data)


_SECTIONS = {
    "room": Room,
    "robot": Robot,
    "dynamics": Dynamics,
    "episode": Episode,
    "penalties": Penalties,
    "scoring": Scoring,
}


def from_dict(data: dict[str, Any]) -> Config:
    """สร้าง Config จาก dict — ฟิลด์ที่ไม่รู้จักถือเป็น error ไม่ใช่เพิกเฉย

    เหตุผล: การพิมพ์ชื่อฟิลด์ผิด (`action_noize`) แล้วระบบเงียบๆ ใช้ค่า default
    คือบั๊กที่หาไม่เจอจนกว่าจะเปิดเทอม
    """
    data = dict(data)
    # `evaluation` เป็นเรื่องของ runner (public/private seeds) ไม่ใช่ของ environment
    data.pop("evaluation", None)

    kwargs: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        raw = data.pop(name, None) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"section {name!r} ต้องเป็น mapping")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ConfigError(
                f"ไม่รู้จักฟิลด์ {sorted(unknown)} ใน section {name!r} — "
                f"ฟิลด์ที่รองรับคือ {sorted(known)}"
            )
        kwargs[name] = cls(**raw)

    unknown_top = set(data) - {"task", "version", "phase"}
    if unknown_top:
        raise ConfigError(f"ไม่รู้จักฟิลด์ระดับบนสุด {sorted(unknown_top)}")
    kwargs.update({k: v for k, v in data.items() if k in ("task", "version", "phase")})
    return Config(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: ไฟล์ config ต้องเป็น mapping ที่ระดับบนสุด")
    return from_dict(raw)
