"""ปลั๊กที่ทำให้ Arena runner รันโจทย์นี้ได้ — สัญญาอยู่ที่ `runners/prediction/plugin.py`

ไฟล์นี้เป็น **จุดเดียว** ที่แพลตฟอร์มแตะโจทย์นี้ · runner ไม่รู้ว่านี่คือ churn
หรือราคาบ้าน ไม่รู้ว่าคะแนนหลักคือ macro-F1 หรือ R² มันรู้แค่ว่าเรียกอะไรได้บ้าง

⚠️ **`grading_data` คืนเฉลย** — ผู้เรียก (runner) ส่งเข้ากล่องได้เฉพาะ `.X`
ส่วน `.y` ใช้คิดคะแนนฝั่ง trusted เท่านั้น
"""

from __future__ import annotations

from typing import Any

from tabular import __version__
from tabular.config import ConfigError, TaskSpec, load_config
from tabular.dataset import grading_data
from tabular.metrics import score

#: เพดานเวลาต่อการเรียก `predict` หนึ่งครั้ง — **ตัวกันงานค้าง ไม่มีผลต่อคะแนน**
#: กว้างพอสำหรับ pipeline ที่หนัก (ensemble ใหญ่ๆ) บนชุดหลักพันแถว
PREDICT_TIMEOUT_S = 300.0


class TabularPlugin:
    name = "cp462-tabular"

    def load_spec(self, path: str) -> TaskSpec:
        return load_config(path)

    def apply_overrides(self, spec: TaskSpec, overrides: dict[str, Any]) -> TaskSpec:
        return spec.replace(**overrides) if overrides else spec

    def config_hash(self, spec: TaskSpec) -> str:
        return spec.config_hash

    def env_version(self, spec: TaskSpec) -> str:
        return __version__

    def grading_data(self, spec: TaskSpec, kind: str):
        """🔒 ชุดที่ใช้ตัดสิน พร้อมเฉลย"""
        return grading_data(spec, kind)

    def predictor_config(self, spec: TaskSpec) -> dict[str, Any]:
        """สิ่งที่โค้ดนิสิตได้รู้ตอนสร้าง `Predictor`

        **มีแค่สิ่งที่ประกาศต่อสาธารณะอยู่แล้ว** — ชื่อโจทย์ ชนิด และคะแนนหลัก
        ไม่มีเมล็ด ไม่มีขนาดของชุดที่ใช้ตัดสิน ไม่มีสัดส่วนของคลาสในชุดนั้น
        (สัดส่วนของคลาสในชุดลับเป็นข้อมูลที่ใช้เดาเฉลยได้จริงถ้าโจทย์ไม่สมดุล)
        """
        return {"task": spec.slug, "kind": spec.kind, "primary": spec.primary}

    def score(self, spec: TaskSpec, y_true, y_pred):
        return score(
            y_true, y_pred,
            kind=spec.kind, primary=spec.primary,
            seed=spec.bootstrap_seed, labels=spec.labels or None,
        )

    def predict_timeout_s(self, spec: TaskSpec) -> float:
        return PREDICT_TIMEOUT_S


PLUGIN = TabularPlugin()

__all__ = ["PLUGIN", "ConfigError", "TabularPlugin"]
