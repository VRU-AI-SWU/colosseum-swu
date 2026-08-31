"""ปลั๊กที่ทำให้ Arena runner รันโจทย์นี้ได้ — สัญญาอยู่ที่ `runners/prediction/plugin.py`

ไฟล์นี้เป็น **จุดเดียว** ที่แพลตฟอร์มแตะโจทย์นี้ · runner ไม่รู้ว่านี่คือ churn
หรือราคาบ้าน ไม่รู้ว่าคะแนนหลักคือ macro-F1 หรือ R² มันรู้แค่ว่าเรียกอะไรได้บ้าง

⚠️ **`grading_data` คืนเฉลย** — ผู้เรียก (runner) ส่งเข้ากล่องได้เฉพาะ `.X`
ส่วน `.y` ใช้คิดคะแนนฝั่ง trusted เท่านั้น
"""

from __future__ import annotations

import os
from typing import Any

from runners.sandbox.schema import Limit, as_dicts, derive
from tabular import __version__
from tabular.config import KINDS, PRIMARY_BY_KIND, ConfigError, TaskSpec, load_config
from tabular.dataset import grading_data
from tabular.metrics import score
from tabular.secrets import load_grading_seed

#: เพดานเวลาต่อการเรียก `predict` หนึ่งครั้ง — **ตัวกันงานค้าง ไม่มีผลต่อคะแนน**
#: กว้างพอสำหรับ pipeline ที่หนัก (ensemble ใหญ่ๆ) บนชุดหลักพันแถว
PREDICT_TIMEOUT_S = 300.0

#: ตัวแปรแวดล้อมที่ยอมให้ใช้เมล็ดสำรองแทนของจริง — **dev กับเทสต์เท่านั้น**
#:
#: อยู่ใน environment ไม่ใช่ในโค้ด ด้วยเหตุผลเดียวกับ `allow_seed_fallback` ของ
#: CP463: เครื่องที่ไม่มี `ARENA_SECRETS` ต้องพัฒนาและรันเทสต์ได้ แต่การเปิดมัน
#: ต้องเป็นการกระทำที่มองเห็นได้ ไม่ใช่ค่าเริ่มต้น · `load_grading_seed` เตือนดังๆ
#: ทุกครั้งที่ใช้ และ worker ของจริงไม่เคยตั้งค่านี้
ALLOW_FALLBACK_ENV = "ARENA_CP462_ALLOW_SEED_FALLBACK"

#: ขอบเขตที่อนุมานจาก dataclass ไม่ได้ — มันอยู่ใน `TaskSpec.__post_init__`
#:
#: ⚠️ **ต้องตรงกับสิ่งที่ `__post_init__` บังคับจริง** ไม่งั้นฟอร์มจะรับค่าที่
#: loader ปฏิเสธ · `test_schema.py` ยิงค่านอกขอบเขตเข้า loader จริงเพื่อยืนยัน
# ⚠️ ประกาศเฉพาะขอบเขตที่ `__post_init__` บังคับจริง — ดูเหตุผลที่ vacuum/arena.py
CONFIG_LIMITS = {
    "slug": Limit(help="ชื่อสั้นของโจทย์ — ใช้อ้างในคำสั่งและใน URL"),
    "task": Limit(help="ชื่อชุดข้อมูลใน `generator.TASKS` — จุดที่จะสลับเป็นข้อมูลจริง"),
    "title": Limit(help="ชื่อที่นิสิตเห็น — แก้ได้ตลอด ไม่กระทบคะแนนเก่า"),
    "kind": Limit(choices=KINDS, help="clustering ยังไม่รองรับโดยตั้งใจ"),
    "primary": Limit(
        choices=tuple(sorted({m for ms in PRIMARY_BY_KIND.values() for m in ms})),
        help="คะแนนหลัก — ต้องเข้าคู่กับชนิดโจทย์ และ 'มากกว่าดีกว่า' เสมอ",
    ),
    "n_rows": Limit(minimum=100, help="จำนวนแถวของชุดที่แจกนิสิต"),
    "data_seed": Limit(help="เมล็ดของชุดที่แจก — สาธารณะ นิสิตใช้สร้างข้อมูลเดียวกัน"),
    "split_seed": Limit(help="เมล็ดการแบ่ง train/val/test"),
    "bootstrap_seed": Limit(help="เมล็ดของช่วงความเชื่อมั่น — ตรึงให้ทุกทีมเทียบกันได้"),
    "ratios": Limit(fixed=True, help="สัดส่วน train/val/test — แก้ผ่านฟอร์มยังไม่รองรับ"),
    "labels": Limit(fixed=True, help="ลำดับคลาสที่ตรึงไว้ — classification เท่านั้น"),
    "grading_rows": Limit(minimum=100, help="จำนวนแถวของชุดที่ใช้ตัดสิน"),
    "grading_public_ratio": Limit(
        minimum=0.01, maximum=0.99, help="สัดส่วนที่เป็น test_public ที่เหลือเป็น private"
    ),
    "grading_seed": Limit(
        fixed=True, help="🔒 เมล็ดของชุดลับ — อยู่ใน ARENA_SECRETS ไม่ใช่ในฟอร์ม"
    ),
}


class TabularPlugin:
    name = "cp462-tabular"

    @property
    def allow_seed_fallback(self) -> bool:
        return os.environ.get(ALLOW_FALLBACK_ENV) == "1"

    def load_spec(self, path: str) -> TaskSpec:
        """โหลดสเปคสาธารณะ **แล้วฉีดเมล็ดของชุดที่ใช้ตัดสินเข้าไป**

        นี่คือจุดเดียวที่เมล็ดลับเข้าสู่ระบบ และมันอยู่ฝั่ง trusted เสมอ —
        `predictor_host` ไม่เคยเรียกฟังก์ชันนี้ และ `tabular` ก็ไม่ได้อยู่ใน image
        """
        spec = load_config(path)
        return spec.replace(
            grading_seed=load_grading_seed(spec.slug, allow_fallback=self.allow_seed_fallback)
        )

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

    def config_schema(self) -> list[dict[str, Any]]:
        """หน้าตาของ config สำหรับสร้างฟอร์ม — **อนุมานจาก dataclass ไม่ได้เขียนมือ**"""
        return as_dicts(derive(TaskSpec, CONFIG_LIMITS))

    def predict_timeout_s(self, spec: TaskSpec) -> float:
        return PREDICT_TIMEOUT_S


PLUGIN = TabularPlugin()

__all__ = ["PLUGIN", "ConfigError", "TabularPlugin"]
