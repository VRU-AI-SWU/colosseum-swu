"""ปลั๊กที่ทำให้ Arena runner รันโจทย์นี้ได้ — สัญญาอยู่ที่ `runners/prediction/plugin.py`

ไฟล์นี้เป็น **จุดเดียว** ที่แพลตฟอร์มแตะโจทย์นี้ · runner ไม่รู้ว่านี่คือ churn
หรือราคาบ้าน ไม่รู้ว่าคะแนนหลักคือ macro-F1 หรือ R² มันรู้แค่ว่าเรียกอะไรได้บ้าง

⚠️ **`grading_data` คืนเฉลย** — ผู้เรียก (runner) ส่งเข้ากล่องได้เฉพาะ `.X`
ส่วน `.y` ใช้คิดคะแนนฝั่ง trusted เท่านั้น
"""

from __future__ import annotations

from typing import Any

from runners.sandbox.schema import Limit, Offer, as_dicts, derive
from tabular import __version__
from tabular import splits, store
from tabular.config import (
    KINDS,
    MAX_STUDENT_RATIO,
    MIN_STUDENT_RATIO,
    PRIMARY_BY_KIND,
    ConfigError,
    TaskSpec,
    load_config,
)
from tabular.dataset import grading_data, parts, student_csv, to_dataset
from tabular.metrics import score

#: เพดานเวลาต่อการเรียก `predict` หนึ่งครั้ง — **ตัวกันงานค้าง ไม่มีผลต่อคะแนน**
#: กว้างพอสำหรับ pipeline ที่หนัก (ensemble ใหญ่ๆ) บนชุดหลักพันแถว
PREDICT_TIMEOUT_S = 300.0

#: ขอบเขตที่อนุมานจาก dataclass ไม่ได้ — มันอยู่ใน `TaskSpec.__post_init__`
#:
#: ⚠️ **ต้องตรงกับสิ่งที่ `__post_init__` บังคับจริง** ไม่งั้นฟอร์มจะรับค่าที่
#: loader ปฏิเสธ · `test_config_schema.py` ยิงค่านอกขอบเขตเข้า loader จริงเพื่อยืนยัน
CONFIG_LIMITS = {
    "title": Limit(label="ชื่อโจทย์ที่นิสิตเห็น", help="แก้ได้ตลอด ไม่กระทบคะแนนเก่า"),
    "kind": Limit(label="ชนิด", choices=KINDS),
    "primary": Limit(
        label="คะแนนหลัก",
        choices=tuple(sorted({m for ms in PRIMARY_BY_KIND.values() for m in ms})),
        help="ใช้จัดอันดับบนกระดาน — ทุกตัว 'มากกว่าดีกว่า'",
    ),
    "dataset": Limit(
        label="ไฟล์ข้อมูล",
        widget="upload",
        help="CSV ที่มีหัวคอลัมน์ · ทั้งไฟล์อยู่บนเซิร์ฟเวอร์ นิสิตได้เฉพาะส่วนที่แจก",
    ),
    "target": Limit(
        label="คอลัมน์เฉลย",
        widget="column",
        help="คอลัมน์ที่โมเดลต้องทำนาย — ถูกตัดออกจากข้อมูลที่ส่งเข้ากล่องเสมอ",
    ),
    "drop": Limit(
        label="คอลัมน์ที่ไม่ให้โมเดลเห็น",
        widget="columns",
        help="รหัสแถว ชื่อคน วันที่ดึงข้อมูล — อะไรที่รู้แล้วทำนายได้โดยไม่ต้องเรียนรู้",
    ),
    "student_ratio": Limit(
        label="สัดส่วนที่แจกนิสิต",
        minimum=MIN_STUDENT_RATIO, maximum=MAX_STUDENT_RATIO,
        help="ของทั้งไฟล์ · ที่เหลือคือชุดที่ใช้ตัดสิน ซึ่งนิสิตไม่เคยเห็น",
    ),
    "grading_public_ratio": Limit(
        label="ในชุดที่ใช้ตัดสิน ส่วนที่โชว์บนกระดาน",
        minimum=0.01, maximum=0.99,
        help="ที่เหลือซ่อนไว้ตัดสินรอบสุดท้าย — กันการจูนเข้าหากระดานตลอดเทอม",
    ),
    "split_seed": Limit(
        label="เมล็ดการแบ่งสามกอง",
        help="คุมการแบ่งของ*ระบบ* — ไม่เกี่ยวกับ train/val ที่นิสิตแบ่งเองด้วยเมล็ดของเขา",
    ),
    "bootstrap_seed": Limit(
        label="เมล็ดของช่วงความเชื่อมั่น", help="ตรึงไว้ให้ทุกทีมเทียบกันได้และรันซ้ำได้ค่าเดิม"
    ),
    "labels": Limit(
        label="คลาสทั้งหมด",
        help="ระบบอ่านจากไฟล์ให้แล้ว · ลำดับตรึงไว้ทั้งเทอม เพราะ confusion matrix อ้างลำดับนี้",
    ),
}


class TabularPlugin:
    name = "cp462-tabular"

    def load_spec(self, path: str) -> TaskSpec:
        """โหลดสเปคจากไฟล์ — **ไม่มีอะไรลับต้องฉีดเพิ่มอีกแล้ว**

        เดิมตรงนี้อ่าน `grading_seed` จาก `ARENA_SECRETS` มาใส่ เพราะชุดที่ใช้
        ตัดสินถูก *สร้าง* จากเมล็ด · ตอนนี้มันเป็นส่วนหนึ่งของไฟล์ที่อยู่ในคลัง
        ความลับจึงเป็นเรื่องของ "ใครอ่านคลังได้" ไม่ใช่ "ใครรู้ตัวเลข"
        """
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

        **มีแค่สิ่งที่ประกาศต่อสาธารณะอยู่แล้ว** — ชนิดกับคะแนนหลัก
        ไม่มีขนาดของชุดที่ใช้ตัดสิน ไม่มีสัดส่วนของคลาสในชุดนั้น
        (สัดส่วนของคลาสในชุดลับเป็นข้อมูลที่ใช้เดาเฉลยได้จริงถ้าโจทย์ไม่สมดุล)
        """
        return {"kind": spec.kind, "primary": spec.primary}

    def score(self, spec: TaskSpec, y_true, y_pred):
        return score(
            y_true, y_pred,
            kind=spec.kind, primary=spec.primary,
            seed=spec.bootstrap_seed, labels=spec.labels or None,
        )

    # ── สิ่งที่ผู้สอนใช้ตอนสร้างโจทย์ (ไม่เกี่ยวกับการให้คะแนน) ──────────────

    def inspect_dataset(self, blob: bytes) -> dict[str, Any]:
        """ตรวจไฟล์ที่เพิ่งอัปโหลดแล้วสรุปให้หน้าเว็บ — **ยังไม่เก็บลงคลัง**

        ผู้สอนต้องเห็นว่ามีคอลัมน์อะไรบ้าง คอลัมน์ไหนเป็นเฉลยได้ และแต่ละคลาส
        มีกี่แถว *ก่อน* ที่จะตั้งค่าอะไร · ฟอร์มที่ให้พิมพ์ชื่อคอลัมน์จากความจำ
        คือฟอร์มที่พิมพ์ผิดได้ แล้วความผิดจะไปโผล่ตอนนิสิตส่งงานเข้ามาแล้ว
        """
        _, profile = store.inspect(blob)
        return profile.as_dict()

    def save_dataset(self, blob: bytes) -> str:
        """เก็บไฟล์ลงคลังแล้วคืนลายนิ้วมือ — ค่าที่ใส่ในช่อง `dataset`"""
        store.inspect(blob)  # ตรวจก่อนเก็บเสมอ — คลังต้องไม่มีไฟล์ที่ใช้ไม่ได้
        return store.put(blob)

    def preview(self, spec: TaskSpec) -> dict[str, Any]:
        """สามกองจะออกมาหน้าตายังไง — **ตอบคำถาม "สัดส่วนในข้อมูลจริงเป็นเท่าไร"**

        ผู้สอนกรอกสัดส่วนเป็นตัวเลข 0–1 แต่สิ่งที่เขาต้องตัดสินใจจริงคือ "กองที่
        ใช้ตัดสินรอบสุดท้ายจะมีกี่แถว และคลาสที่พบน้อยจะเหลือกี่แถวในนั้น" ·
        ตัวเลขสองแบบนี้ต่างกันมากเมื่อข้อมูลไม่สมดุล และแบบหลังคือแบบที่บอกได้ว่า
        อันดับสุดท้ายจะมีความหมายไหม
        """
        split = parts(spec)
        out: dict[str, Any] = {"sizes": split.sizes(), "rows": len(split.student)
                               + len(split.test_public) + len(split.test_private)}
        if spec.kind == "classification":
            out["classes"] = {
                name: {
                    str(label): int(count)
                    for label, count in getattr(split, name).y.value_counts().sort_index().items()
                }
                for name in splits.PARTS
            }
            out["thin"] = splits.thin_strata(
                _target_series(spec),
                kind=spec.kind,
                student_ratio=spec.student_ratio,
                grading_public_ratio=spec.grading_public_ratio,
            )
        return out

    def student_bytes(self, spec: TaskSpec) -> bytes:
        """ไฟล์ที่นิสิตดาวน์โหลด — กองที่แจกเท่านั้น พร้อมเฉลยของกองนั้น"""
        return student_csv(spec)

    def offers(self) -> list[dict[str, Any]]:
        """โจทย์สองแบบที่ env นี้เสิร์ฟได้ — **ผู้สอนเลือกจากตรงนี้**

        `kind` ถูกซ่อนเพราะตัวเลือกตอบให้แล้ว และ `primary` ถูกจำกัดให้เหลือเฉพาะ
        ที่เข้าคู่กับ kind นั้น — เดิมฟอร์มโชว์คะแนนหลักทั้ง 5 ตัวรวมกัน แล้วผู้สอน
        ที่เลือก classification คู่กับ `r2` จะโดน loader ปฏิเสธหลังกดบันทึก
        ทั้งที่ฟอร์มเสนอให้เลือกเอง

        **`labels` ไม่ได้ซ่อนแต่ก็ไม่ต้องกรอก** — หน้าเว็บเติมให้จากไฟล์ที่อัปโหลด
        ผู้สอนเห็นค่าและแก้ลำดับได้ แต่ไม่ต้องพิมพ์เอง · เคยซ่อนไว้แล้วไม่มีใคร
        เติมให้ ผลคือสร้าง classification ไม่ได้เลยสักครั้ง

        **clustering ไม่อยู่ในรายการโดยตั้งใจ** — ดู template §6
        """
        return as_dicts([
            Offer(
                id="classification",
                label="Classification",
                blurb="ทำนาย label ของแต่ละแถว — ผู้ป่วยกลุ่มไหน · อีเมลนี้สแปมไหม",
                defaults={"kind": "classification", "primary": "macro_f1"},
                hide=("kind",),
                narrow={"primary": PRIMARY_BY_KIND["classification"]},
            ),
            Offer(
                id="regression",
                label="Regression",
                blurb="ทำนายตัวเลขของแต่ละแถว — ราคาบ้าน · ปริมาณการใช้ไฟ",
                # regression **ต้องไม่มี** labels — ซ่อนช่องนั้นและเติมค่าว่างให้
                defaults={"kind": "regression", "primary": "r2", "labels": []},
                hide=("kind", "labels"),
                narrow={"primary": PRIMARY_BY_KIND["regression"]},
            ),
        ])

    def config_schema(self) -> list[dict[str, Any]]:
        """หน้าตาของ config สำหรับสร้างฟอร์ม — **อนุมานจาก dataclass ไม่ได้เขียนมือ**"""
        return as_dicts(derive(TaskSpec, CONFIG_LIMITS))

    def predict_timeout_s(self, spec: TaskSpec) -> float:
        return PREDICT_TIMEOUT_S


def _target_series(spec: TaskSpec):
    """คอลัมน์เฉลยของไฟล์เต็ม — ใช้นับว่าคลาสไหนจะบางเกินไป"""
    return to_dataset(
        store.read(spec.dataset), target=spec.target, drop=spec.drop, source=spec.dataset
    ).y


PLUGIN = TabularPlugin()

__all__ = ["PLUGIN", "ConfigError", "TabularPlugin"]
