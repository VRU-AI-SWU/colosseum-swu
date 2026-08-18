"""สัญญาระหว่าง runner กับ environment ของโจทย์

runner ตัวนี้ **ไม่รู้จักโจทย์ใดๆ** — มันรู้แค่ว่า environment ตัวหนึ่งต้องทำอะไรได้
การเพิ่ม competition ใหม่จึงแตะแค่ `envs/` ไม่ต้องแก้ `runners/` หรือ `core/`
([README §10.5](../../README.md#105-โครงสร้าง-repository))

env package ต้อง export วัตถุที่มี attribute ตามนี้ แล้วประกาศชื่อไว้ใน TaskSpec เช่น

    env_plugin: "vacuum.arena:PLUGIN"
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnvPlugin(Protocol):
    """สิ่งที่ environment ของโจทย์ต้องให้ runner ได้"""

    name: str
    version: str

    def load_config(self, path: str) -> Any:
        """โหลด config ของ phase จากไฟล์ YAML"""

    def config_hash(self, config: Any) -> str:
        """ลายนิ้วมือของ config — บันทึกลงทุก run และทุก replay"""

    def apply_overrides(self, config: Any, overrides: dict[str, Any]) -> Any:
        """ทับค่าบางตัวใน config — รองรับ `Phase.config_override` ตาม README §12
        คีย์เป็นแบบ dotted เช่น `{"dynamics.sensor_noise": 0.05}`"""

    def make_env(self, config: Any) -> Any:
        """สร้าง environment ที่มี `reset(seed=...)` / `step(action)` แบบ Gymnasium"""

    def agent_config(self, config: Any) -> dict[str, Any]:
        """ข้อมูลที่ agent ได้รู้ตอนสร้าง — **ห้ามมี seed หรือผังห้อง**"""

    def episode_score(self, env: Any, config: Any) -> Any:
        """คะแนนของ episode ที่เพิ่งจบ — ต้องเป็นตัวเดียวกับที่ starter kit ใช้"""

    def zero_score(self, config: Any) -> Any:
        """คะแนนของ episode ที่ agent ล้มเหลว (template §7.3 กำหนดว่าเป็น 0)"""

    def aggregate(self, breakdowns: list[Any]) -> Any:
        """รวมคะแนนทุก episode เป็นผลของ submission พร้อมเกณฑ์ตัดสินเสมอ"""

    def write_replay(self, path: str, env: Any) -> int:
        """เขียน replay ของ episode ล่าสุด — คืนขนาดไฟล์เป็นไบต์"""

    def step_timeout_ms(self, config: Any) -> int:
        """เพดานเวลาต่อ 1 step — **ตัวกันงานค้าง ไม่มีผลต่อคะแนน**"""


def resolve(spec: str) -> EnvPlugin:
    """แปลง `"module:attr"` ให้เป็น plugin จริง"""
    if ":" not in spec:
        raise ValueError(f"env_plugin ต้องอยู่ในรูป 'module:attr' — ได้ {spec!r}")
    module_name, attr = spec.split(":", 1)
    plugin = getattr(importlib.import_module(module_name), attr)
    missing = [
        m
        for m in (
            "load_config", "config_hash", "apply_overrides", "make_env", "agent_config",
            "episode_score", "zero_score", "aggregate", "write_replay", "step_timeout_ms",
        )
        if not hasattr(plugin, m)
    ]
    if missing:
        raise TypeError(f"{spec} ขาด {missing} — ดูสัญญาที่ runners/agent_env/plugin.py")
    return plugin
