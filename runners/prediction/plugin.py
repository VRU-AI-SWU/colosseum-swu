"""สัญญาระหว่าง runner กับ env ของโจทย์ทำนาย

runner ตัวนี้ **ไม่รู้จักโจทย์ใดๆ** — ไม่รู้ว่าเป็น churn หรือราคาบ้าน ไม่รู้ว่า
คะแนนหลักคือ macro-F1 หรือ R² · การเพิ่ม competition ใหม่จึงแตะแค่ `envs/`

env package ต้อง export วัตถุที่มี attribute ตามนี้ แล้วประกาศชื่อไว้ที่ competition เช่น

    env_plugin: "tabular.arena:PLUGIN"

**เส้นแบ่งที่สำคัญที่สุดอยู่ที่ `grading_data` กับ `predictor_config`** — ตัวแรก
คืนเฉลย ตัวหลังคืนสิ่งที่ส่งเข้ากล่องได้ ถ้าวันหนึ่งมีใครเผลอให้ `predictor_config`
คืนอะไรที่มาจากชุดที่ใช้ตัดสิน การแข่งจบทันทีโดยไม่มีใครรู้
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

REQUIRED = (
    "load_spec",
    "apply_overrides",
    "config_hash",
    "env_version",
    "grading_data",
    "predictor_config",
    "score",
    "predict_timeout_s",
)


@runtime_checkable
class PredictionPlugin(Protocol):
    """สิ่งที่ env ของโจทย์ทำนายต้องให้ runner ได้"""

    name: str

    def load_spec(self, path: str) -> Any:
        """โหลดสเปคของโจทย์จากไฟล์ YAML"""

    def apply_overrides(self, spec: Any, overrides: dict[str, Any]) -> Any:
        """ทับค่าบางตัวในสเปค — รองรับ `Phase.config_override`"""

    def config_hash(self, spec: Any) -> str:
        """ลายนิ้วมือของสเปค — บันทึกลงทุก run · เปลี่ยนเมื่อไรคะแนนเก่าเทียบไม่ได้"""

    def env_version(self, spec: Any) -> str:
        """เวอร์ชันของแพ็กเกจโจทย์ — บันทึกคู่กับคะแนน"""

    def grading_data(self, spec: Any, kind: str) -> Any:
        """🔒 ชุดที่ใช้ตัดสิน — คืนวัตถุที่มี `.X` (DataFrame) และ `.y`

        `kind` เป็น `"public"` ระหว่างเทอม · `"private"` ตอนปิดรับ
        **ค่าที่คืนมาต้องไม่มีทางเดินทางเข้ากล่อง** นอกจากส่วน `.X`
        """

    def predictor_config(self, spec: Any) -> dict[str, Any]:
        """ข้อมูลที่โค้ดนิสิตได้รู้ตอนสร้าง `Predictor` — **ห้ามมีอะไรจากชุดที่ใช้ตัดสิน**"""

    def score(self, spec: Any, y_true: Any, y_pred: Any) -> Any:
        """ให้คะแนน — คืนวัตถุที่มี `.primary`, `.primary_name`, `.as_dict()`"""

    def predict_timeout_s(self, spec: Any) -> float:
        """เพดานเวลาต่อการเรียก `predict` หนึ่งครั้ง — **ตัวกันงานค้าง ไม่มีผลต่อคะแนน**"""


def resolve(spec: str) -> PredictionPlugin:
    """แปลง `"module:attr"` ให้เป็น plugin จริง"""
    if ":" not in spec:
        raise ValueError(f"env_plugin ต้องอยู่ในรูป 'module:attr' — ได้ {spec!r}")
    module_name, attr = spec.split(":", 1)
    plugin = getattr(importlib.import_module(module_name), attr)
    missing = [m for m in REQUIRED if not hasattr(plugin, m)]
    if missing:
        raise TypeError(f"{spec} ขาด {missing} — ดูสัญญาที่ runners/prediction/plugin.py")
    return plugin
