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

#: สิ่งที่ **runner** เรียกตอนให้คะแนน — ขาดตัวใดตัวหนึ่งแปลว่าให้คะแนนไม่ได้
REQUIRED = (
    "offers",
    "config_schema",
    "load_spec",
    "apply_overrides",
    "config_hash",
    "env_version",
    "grading_data",
    "predictor_config",
    "score",
    "predict_timeout_s",
)

#: สิ่งที่ **หน้าเว็บของผู้สอน** เรียกตอนสร้างโจทย์ — runner ไม่เคยเรียกเลย
#:
#: แยกจาก `REQUIRED` โดยตั้งใจ · การรวมกันทำให้ runner ปฏิเสธ plugin ที่มันรันได้
#: จริง เพียงเพราะ plugin นั้นยังไม่รองรับการสร้างโจทย์ผ่านหน้าเว็บ — เส้นสองเส้นนี้
#: มีผู้เรียกคนละคนและพังคนละเวลา
AUTHORING = (
    "inspect_dataset",
    "save_dataset",
    "preview",
    "student_bytes",
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

    def offers(self) -> list[dict[str, Any]]:
        """โจทย์แต่ละแบบที่ env นี้เสิร์ฟได้ — สิ่งที่ผู้สอนเลือกบนหน้าเว็บ

        ผู้สอนไม่ควรต้องรู้ว่าระบบเรียกมันว่า `task_type` อะไร หรือมีฟิลด์ `kind`
        อยู่ใน config · ดู `runners/sandbox/schema.py:Offer`
        """

    def config_schema(self) -> list[dict[str, Any]]:
        """หน้าตาของ config สำหรับสร้างฟอร์ม — **อนุมานจาก dataclass ไม่เขียนมือ**"""

    # ── สิ่งที่ผู้สอนใช้ตอนสร้างโจทย์ ────────────────────────────────────────
    #
    # โจทย์ทำนายต่างจาก agent env ตรงที่ **ข้อมูลมาจากผู้สอน ไม่ได้มาจากโค้ด** ·
    # env จึงต้องรับไฟล์ ตรวจไฟล์ และบอกได้ว่าไฟล์นั้นจะถูกแบ่งออกมาหน้าตายังไง
    # ก่อนที่ใครจะกดสร้าง · ทั้งสี่เมธอดนี้อยู่ฝั่ง trusted ทั้งหมด

    def inspect_dataset(self, blob: bytes) -> dict[str, Any]:
        """ตรวจไฟล์ที่เพิ่งอัปโหลดแล้วสรุปคอลัมน์ให้หน้าเว็บ — **ยังไม่เก็บลงคลัง**"""

    def save_dataset(self, blob: bytes) -> str:
        """เก็บไฟล์ลงคลังแล้วคืนรหัสที่ใช้อ้างใน config"""

    def preview(self, spec: Any) -> dict[str, Any]:
        """🔒 ไฟล์จะถูกแบ่งเป็นกี่แถวต่อกอง และคลาสไหนจะบางเกินไป

        ผู้สอนต้องเห็นตัวเลขนี้**ก่อน**กดสร้าง — สัดส่วน `0.15` ไม่ได้บอกอะไรเลย
        ส่วน "คลาสที่พบน้อยจะเหลือ 3 แถวในกองที่ตัดสินรอบสุดท้าย" บอกได้ทันทีว่า
        อันดับสุดท้ายจะมีความหมายหรือเป็นเรื่องของโชค
        """

    def student_bytes(self, spec: Any) -> bytes:
        """ไฟล์ที่นิสิตดาวน์โหลด — **กองที่แจกเท่านั้น**

        นี่คือทางออกทางเดียวของข้อมูลจากเซิร์ฟเวอร์ · ทุกไบต์ที่ผ่านฟังก์ชันนี้
        ถือว่านิสิตเห็นแล้ว — ถ้าวันหนึ่งมันคืนกองอื่นมาด้วย การแข่งจบทันทีโดย
        ไม่มีใครรู้ · คู่กับ `grading_data` ที่อยู่คนละฝั่งของเส้นเดียวกัน
        """


def resolve(spec: str, *, also: tuple[str, ...] = ()) -> PredictionPlugin:
    """แปลง `"module:attr"` ให้เป็น plugin จริง

    `also=AUTHORING` เมื่อผู้เรียกคือหน้าเว็บของผู้สอน ซึ่งต้องการเมธอดชุดที่
    runner ไม่ต้องการ
    """
    if ":" not in spec:
        raise ValueError(f"env_plugin ต้องอยู่ในรูป 'module:attr' — ได้ {spec!r}")
    module_name, attr = spec.split(":", 1)
    plugin = getattr(importlib.import_module(module_name), attr)
    missing = [m for m in (*REQUIRED, *also) if not hasattr(plugin, m)]
    if missing:
        raise TypeError(f"{spec} ขาด {missing} — ดูสัญญาที่ runners/prediction/plugin.py")
    return plugin
