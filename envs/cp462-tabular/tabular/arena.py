"""ปลั๊กที่ทำให้ Arena runner รันโจทย์นี้ได้ — สัญญาอยู่ที่ `runners/prediction/plugin.py`

ไฟล์นี้เป็น **จุดเดียว** ที่แพลตฟอร์มแตะโจทย์นี้ · runner ไม่รู้ว่านี่คือ churn
หรือราคาบ้าน ไม่รู้ว่าคะแนนหลักคือ macro-F1 หรือ R² มันรู้แค่ว่าเรียกอะไรได้บ้าง

⚠️ **`grading_data` คืนเฉลย** — ผู้เรียก (runner) ส่งเข้ากล่องได้เฉพาะ `.X`
ส่วน `.y` ใช้คิดคะแนนฝั่ง trusted เท่านั้น
"""

from __future__ import annotations

import os
from typing import Any

from runners.sandbox.schema import Limit, Offer, as_dicts, derive
from tabular import __version__
from tabular.config import KINDS, PRIMARY_BY_KIND, ConfigError, TaskSpec, load_config
from tabular.generator import TASKS
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
    "slug": Limit(
        label="รหัสชุดโจทย์",
        help="ผูกกับเมล็ดของชุดที่ใช้ตัดสิน — ต้องมีไฟล์ชื่อนี้ใน ARENA_SECRETS ก่อน "
             "· คนละอันกับรหัส competition ข้างบน",
    ),
    "task": Limit(
        label="ชุดข้อมูล",
        choices=tuple(sorted(TASKS)),
        help="ชุดข้อมูลที่ระบบสร้างให้ได้ — ยังอัปโหลดไฟล์ของตัวเองไม่ได้",
    ),
    "title": Limit(label="ชื่อโจทย์ที่นิสิตเห็น", help="แก้ได้ตลอด ไม่กระทบคะแนนเก่า"),
    "kind": Limit(label="ชนิด", choices=KINDS),
    "primary": Limit(
        label="คะแนนหลัก",
        choices=tuple(sorted({m for ms in PRIMARY_BY_KIND.values() for m in ms})),
        help="ใช้จัดอันดับบนกระดาน — ทุกตัว 'มากกว่าดีกว่า'",
    ),
    "n_rows": Limit(label="จำนวนแถวที่แจกนิสิต", minimum=100),
    "ratios": Limit(
        label="สัดส่วน train / val / test",
        help="ของนิสิตล้วน · คั่นด้วยจุลภาค รวมกันต้องได้ 1.0",
    ),
    "labels": Limit(
        label="คลาสทั้งหมด",
        help="คั่นด้วยจุลภาค · ลำดับตรึงไว้ทั้งเทอม เพราะ confusion matrix อ้างลำดับนี้",
    ),
    "data_seed": Limit(label="เมล็ดของชุดที่แจก", help="สาธารณะ — นิสิตใช้สร้างข้อมูลชุดเดียวกัน"),
    "split_seed": Limit(label="เมล็ดการแบ่ง train/val/test"),
    "bootstrap_seed": Limit(
        label="เมล็ดของช่วงความเชื่อมั่น", help="ตรึงไว้ให้ทุกทีมเทียบกันได้และรันซ้ำได้ค่าเดิม"
    ),
    "grading_rows": Limit(label="จำนวนแถวของชุดที่ใช้ตัดสิน", minimum=100),
    "grading_public_ratio": Limit(
        label="สัดส่วนที่เป็นชุดสาธารณะ", minimum=0.01, maximum=0.99,
        help="ที่เหลือเป็นชุดลับที่ใช้ตัดสินตอนปิดรับ",
    ),
    "grading_seed": Limit(
        fixed=True, label="เมล็ดของชุดลับ",
        help="🔒 อยู่ใน ARENA_SECRETS ไม่ใช่ในฟอร์ม",
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

    def offers(self) -> list[dict[str, Any]]:
        """โจทย์สองแบบที่ env นี้เสิร์ฟได้ — **ผู้สอนเลือกจากตรงนี้**

        `kind` ถูกซ่อนเพราะตัวเลือกตอบให้แล้ว และ `primary` ถูกจำกัดให้เหลือเฉพาะ
        ที่เข้าคู่กับ kind นั้น — เดิมฟอร์มโชว์คะแนนหลักทั้ง 5 ตัวรวมกัน แล้วผู้สอน
        ที่เลือก classification คู่กับ `r2` จะโดน loader ปฏิเสธหลังกดบันทึก
        ทั้งที่ฟอร์มเสนอให้เลือกเอง

        **clustering ไม่อยู่ในรายการโดยตั้งใจ** — ดู template §6
        """
        return as_dicts([
            Offer(
                id="classification",
                label="Classification",
                blurb="ทำนาย label ของแต่ละแถว — ผู้ป่วยกลุ่มไหน · อีเมลนี้สแปมไหม",
                # `labels` **ไม่ได้ซ่อน** — classification ต้องประกาศคลาสเอง และค่านี้
                # ต่างกันทุกโจทย์ · เดิมซ่อนไว้แล้วไม่มีใครเติมให้ ผลคือสร้าง
                # classification ไม่ได้เลยสักครั้ง
                defaults={"kind": "classification", "primary": "macro_f1",
                          "labels": [0, 1], "ratios": [0.6, 0.15, 0.25]},
                hide=("kind",),
                narrow={"primary": PRIMARY_BY_KIND["classification"]},
            ),
            Offer(
                id="regression",
                label="Regression",
                blurb="ทำนายตัวเลขของแต่ละแถว — ราคาบ้าน · ปริมาณการใช้ไฟ",
                # regression **ต้องไม่มี** labels — ซ่อนช่องนั้นและเติมค่าว่างให้
                defaults={"kind": "regression", "primary": "r2", "labels": [],
                          "ratios": [0.6, 0.15, 0.25]},
                hide=("kind", "labels"),
                narrow={"primary": PRIMARY_BY_KIND["regression"]},
            ),
        ])

    def config_schema(self) -> list[dict[str, Any]]:
        """หน้าตาของ config สำหรับสร้างฟอร์ม — **อนุมานจาก dataclass ไม่ได้เขียนมือ**"""
        return as_dicts(derive(TaskSpec, CONFIG_LIMITS))

    def predict_timeout_s(self, spec: TaskSpec) -> float:
        return PREDICT_TIMEOUT_S


PLUGIN = TabularPlugin()

__all__ = ["PLUGIN", "ConfigError", "TabularPlugin"]
