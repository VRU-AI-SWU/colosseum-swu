"""ปลั๊กที่ทำให้ Arena runner รันโจทย์นี้ได้ — สัญญาอยู่ที่ `runners/agent_env/plugin.py`

ไฟล์นี้เป็น **ที่เดียว** ที่ผูก environment ตัวนี้เข้ากับแพลตฟอร์ม
ถ้าจะเพิ่มโจทย์ใหม่ก็เขียนไฟล์แบบนี้ในแพ็กเกจของโจทย์นั้น โดยไม่ต้องแตะ `runners/` หรือ `core/`

ประกาศใน TaskSpec ว่า

    env_plugin: "vacuum.arena:PLUGIN"

⚠️ ไม่ import อะไรจาก `runners/` เพื่อให้ starter kit ที่นิสิตติดตั้งไม่ต้องมีโค้ดแพลตฟอร์มติดไปด้วย
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from runners.sandbox.schema import Limit, as_dicts, derive
from vacuum import __version__, load_config
from vacuum.config import (
    DIRT_DISTRIBUTIONS,
    OBSTACLE_GENERATORS,
    Config,
)
from vacuum.env import VacuumEnv
from vacuum.replay import encode as encode_replay
from vacuum.replay import header_from_env
from vacuum.rollout import agent_config as _agent_config
from vacuum.scoring import ScoreBreakdown, episode_score, submission_score


class VacuumPlugin:
    name = "vacuum_gridworld"
    version = __version__

    def load_config(self, path: str) -> Any:
        return load_config(path)

    def config_hash(self, config: Any) -> str:
        return config.config_hash

    def apply_overrides(self, config: Any, overrides: dict[str, Any]) -> Any:
        return config.replace(**overrides) if overrides else config

    def make_env(self, config: Any) -> VacuumEnv:
        return VacuumEnv(config)

    def agent_config(self, config: Any) -> dict[str, Any]:
        return _agent_config(config)

    def episode_score(self, env: VacuumEnv, config: Any) -> ScoreBreakdown:
        return episode_score(
            env.stats(),
            config.episode.max_steps,
            completion_bonus=config.scoring.completion_bonus,
            max_penalty=config.scoring.max_penalty,
            w_collision=config.penalties.collision,
            w_redundant=config.penalties.redundant_suck,
        )

    def zero_score(self, config: Any) -> ScoreBreakdown:
        """episode ที่ agent ล้มเหลว — template §7.3 กำหนดว่าได้ 0 ไม่ใช่ "คะแนนน้อยลง" """
        return ScoreBreakdown(
            score=0.0, auc=0.0, completed=False, penalty=0.0,
            coverage=0.0, t_end=0, reason="agent_failed",
        )

    def aggregate(self, breakdowns: list[ScoreBreakdown]) -> Any:
        return submission_score(breakdowns)

    def write_replay(self, path: str, env: VacuumEnv) -> int:
        blob = encode_replay(header_from_env(env), env.events)
        Path(path).write_bytes(blob)
        return len(blob)

    def config_schema(self) -> list[dict[str, Any]]:
        """หน้าตาของ config สำหรับสร้างฟอร์ม — **อนุมานจาก dataclass ไม่ได้เขียนมือ**

        รายการที่เขียนมือจะเพี้ยนจากโค้ดในวันที่มีคนเพิ่มฟิลด์ใหม่แล้วลืมแก้
        """
        return as_dicts(derive(Config, CONFIG_LIMITS))

    def step_timeout_ms(self, config: Any) -> int:
        return config.episode.step_timeout_ms


#: ขอบเขตที่อนุมานจาก dataclass ไม่ได้ — มันอยู่ใน `Config.validate()` แบบคำสั่ง
#:
#: ⚠️ **สิ่งที่ประกาศตรงนี้ต้องตรงกับสิ่งที่ `validate()` บังคับจริง** ไม่งั้นฟอร์ม
#: จะรับค่าที่ loader ปฏิเสธ แล้วผู้สอนจะกรอกครบแต่กดบันทึกไม่ได้ ·
#: `test_schema.py` ยิงค่านอกขอบเขตเข้า loader จริงเพื่อยืนยันข้อนี้
# ⚠️ **ประกาศเฉพาะขอบเขตที่ `validate()` บังคับจริง** — เคยใส่เพดานที่คิดว่าน่าจะมี
# (ห้องไม่เกิน 100×100, step ไม่เกินแสน) แล้วเทสต์จับได้ว่า loader ไม่ได้บังคับ
# ฟอร์มที่แคบกว่าของจริงจะทำให้ผู้สอนตั้งค่าที่ระบบรองรับไม่ได้โดยไม่มีเหตุผล
# ถ้าเห็นว่าควรมีเพดานจริง ต้องไปเพิ่มใน `validate()` ก่อน แล้วค่อยประกาศที่นี่
CONFIG_LIMITS = {
    "task": Limit(fixed=True, help="ชนิดของ environment — เปลี่ยนไม่ได้"),
    "version": Limit(fixed=True, help="เวอร์ชันของ environment — ขึ้นเมื่อกติกาเปลี่ยน"),
    "phase": Limit(fixed=True, help="ชื่อช่วง — ระบบตั้งให้ตามปฏิทิน"),

    "room.width": Limit(minimum=2, help="ห้องใหญ่ขึ้น = สำรวจยากขึ้น"),
    "room.height": Limit(minimum=2),
    "room.obstacle_density": Limit(minimum=0.0, maximum=1.0, help="สัดส่วนช่องที่เป็นสิ่งกีดขวาง"),
    "room.obstacle_generator": Limit(choices=OBSTACLE_GENERATORS),
    "room.dirt_distribution": Limit(choices=DIRT_DISTRIBUTIONS),
    "room.dirt_ratio": Limit(minimum=0.0, maximum=1.0, help="สัดส่วนช่องว่างที่มีฝุ่น — ต้องมากกว่า 0"),
    "room.guarantee_connected": Limit(
        fixed=True, help="v1.0.0 บังคับให้เป็นจริง ไม่งั้น coverage แตะ 100% ไม่ได้บางเมล็ด"
    ),

    "robot.start": Limit(choices=("random", "corner"), help="จุดเริ่มของหุ่น"),
    "robot.observation": Limit(choices=("local", "full"), help="เห็นทั้งห้อง หรือเห็นแค่รอบตัว"),
    "robot.observation_window": Limit(minimum=1, maximum=99, help="ขนาดหน้าต่างที่เห็น (เลขคี่)"),
    "robot.battery": Limit(help="จำนวนพลังงาน — ว่าง = ไม่จำกัด"),
    "robot.move_cost": Limit(minimum=0),
    "robot.suck_cost": Limit(minimum=0),

    "dynamics.action_noise": Limit(minimum=0.0, maximum=1.0, help="โอกาสที่หุ่นไปผิดทิศ"),
    "dynamics.sticky_dirt": Limit(minimum=0.0, maximum=1.0, help="โอกาสที่ดูดครั้งเดียวไม่ขึ้น"),
    "dynamics.sensor_noise": Limit(minimum=0.0, maximum=1.0, help="โอกาสที่เซ็นเซอร์อ่านผิด"),

    "episode.max_steps": Limit(minimum=1, help="เพดานจำนวน step ต่อ episode"),
    "episode.stop_on_full_coverage": Limit(help="ดูดครบแล้วจบทันที"),
    "episode.step_timeout_ms": Limit(
        minimum=1, help="เพดานเวลาต่อ step — กันงานค้าง ไม่มีผลต่อคะแนน"
    ),

    "penalties.collision": Limit(minimum=0.0, help="หักเมื่อชนสิ่งกีดขวาง"),
    "penalties.redundant_suck": Limit(minimum=0.0, help="หักเมื่อดูดช่องที่สะอาดแล้ว"),

    "scoring.metric": Limit(fixed=True, help="วิธีคิดคะแนน — เปลี่ยนแล้วคะแนนเก่าเทียบไม่ได้"),
    "scoring.completion_bonus": Limit(minimum=0.0),
    "scoring.max_penalty": Limit(minimum=0.0, help="เพดานของค่าปรับรวม"),
}


PLUGIN = VacuumPlugin()
