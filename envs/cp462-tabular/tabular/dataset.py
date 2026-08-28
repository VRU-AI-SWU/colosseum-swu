"""ทางเข้าข้อมูล — **แยกให้ชัดว่าอะไรนิสิตได้ อะไรอยู่ในระบบ**

    open_data(spec)     train + val + test พร้อมเฉลย   นิสิตเรียกได้
    grading_data(spec)  test_public + test_private     🔒 ต้องมีเมล็ดลับ

**สองชุดนี้เป็น dataset คนละใบ ไม่ใช่ชุดเดียวแบ่งกัน**

  ชุดของนิสิต   สร้างจาก `data_seed` ที่อยู่ในไฟล์ config ที่แจกไปกับแพ็กเกจ
  ชุดที่ตัดสิน  สร้างจาก `grading_seed` ที่อยู่ใน `ARENA_SECRETS` เท่านั้น

⚠️ **เดิมทั้งสองชุดมาจาก dataset เดียวกัน** ซึ่งแปลว่านิสิตรัน
`grading_data(load('churn'), 'private')` บนเครื่องตัวเองแล้วได้เฉลยครบทุกแถว
— ทดสอบแล้วทำคะแนนได้ 1.0000 โดยจำเฉลยไว้แล้วจับคู่ด้วย `account_id`
ตอนนี้ฟังก์ชันนั้นปฏิเสธถ้าไม่มีเมล็ดลับ และเมล็ดลับไม่ได้อยู่ในแพ็กเกจ

**ทำไมนิสิตสร้างข้อมูลเองแทนที่จะดาวน์โหลดไฟล์** — ตัวสร้างทำซ้ำได้ทุกบิต
นิสิตจึงได้ชุดเดียวกับที่ผู้สอนใช้เป๊ะ โดยไม่ต้องแจกไฟล์ ไม่ต้องกังวลว่าใครโหลด
ผิดเวอร์ชัน · `selfcheck` ยืนยันว่าเครื่องนิสิตได้ข้อมูลตรงกับ grader จริง

> 🔻 **จุดสลับเป็นข้อมูลจริง** — เมื่อวิชามี dataset จริงแล้ว แก้แค่ `_full()`
> กับ `_grading_full()` ให้อ่านไฟล์แทนการเรียก generator · ส่วนที่เหลือไม่ต้องแตะ
> **ชุดที่ใช้ตัดสินต้องเป็นไฟล์ที่ไม่เคยถูกแจก** ไม่ใช่ส่วนหนึ่งของไฟล์ที่แจกไปแล้ว
"""

from __future__ import annotations

from tabular import generator, splits
from tabular.config import TaskSpec
from tabular.secrets import GradingSeedUnavailable

#: เลื่อนช่วง `account_id` ของชุดที่ใช้ตัดสินให้พ้นจากของนิสิตเสมอ
#: ถ้าไม่เลื่อน สอง dataset จะมี id ทับกันทั้งหมดเพราะทั้งคู่เริ่มที่ 100_000
GRADING_ID_OFFSET = 10_000_000


def _full(spec: TaskSpec) -> generator.Dataset:
    """ชุดของนิสิตทั้งชุดก่อนแบ่ง — **จุดสลับเป็นข้อมูลจริงจุดที่ 1**"""
    return generator.make(spec.task, seed=spec.data_seed, n=spec.n_rows)


def _grading_full(spec: TaskSpec) -> generator.Dataset:
    """🔒 ชุดที่ใช้ตัดสินทั้งชุดก่อนแบ่ง — **จุดสลับเป็นข้อมูลจริงจุดที่ 2**"""
    if spec.grading_seed is None:
        raise GradingSeedUnavailable(
            f"โจทย์ {spec.slug!r} ยังไม่มีเมล็ดของชุดที่ใช้ตัดสิน\n"
            "  ฝั่ง trusted ต้องฉีดค่านี้เข้ามาผ่าน `tabular.arena.PLUGIN.load_spec`\n"
            "  (ถ้าคุณเป็นนิสิต: ค่านี้ไม่ได้อยู่ในแพ็กเกจโดยตั้งใจ — ชุดที่ใช้ตัดสิน\n"
            "   ต้องเป็นข้อมูลที่คุณไม่เคยเห็น ไม่งั้นคะแนนไม่มีความหมาย)"
        )
    return generator.make(
        spec.task, seed=spec.grading_seed, n=spec.grading_rows, id_offset=GRADING_ID_OFFSET
    )


def all_parts(spec: TaskSpec) -> splits.Split:
    """สามส่วนของชุดที่แจกนิสิต — **ทุกส่วนนิสิตมีอยู่แล้ว**"""
    return splits.split(_full(spec), seed=spec.split_seed, ratios=spec.ratios)


def grading_parts(spec: TaskSpec) -> splits.GradingSplit:
    """🔒 สองส่วนของชุดที่ใช้ตัดสิน"""
    return splits.grading_split(
        _grading_full(spec), seed=spec.split_seed, public_ratio=spec.grading_public_ratio
    )


def open_data(spec: TaskSpec) -> dict[str, generator.Dataset]:
    """`train` `val` `test` พร้อมเฉลย — สิ่งที่นิสิตได้รับ"""
    return all_parts(spec).open_parts()


def grading_data(spec: TaskSpec, kind: str) -> generator.Dataset:
    """ชุดที่ใช้ตัดสิน — `"public"` ระหว่างเทอม · `"private"` ตอนปิดรับ

    รับ `kind` เป็นสตริงแทนการให้เลือกฟิลด์เอง เพื่อให้จุดที่หยิบชุดลับมีชื่อชัดเจน
    และหาได้ด้วยการค้นหาคำเดียวตอนตรวจว่ามีใครเรียกผิดที่ไหม
    """
    if kind not in splits.GRADING_PARTS and kind not in ("public", "private"):
        raise ValueError(f"kind ต้องเป็น 'public' หรือ 'private' — ได้ {kind!r}")
    parts = grading_parts(spec)
    return parts.test_public if kind in ("public", "test_public") else parts.test_private


def features_only(dataset: generator.Dataset):
    """ฟีเจอร์ล้วนสำหรับส่งเข้า sandbox — **เฉลยไม่ตามไปด้วย**

    ฟังก์ชันบางๆ ที่ดูเหมือนไม่จำเป็น แต่ทำให้จุดที่ตัดเฉลยออกมีชื่อ และทำให้
    การส่ง `dataset.X` ตรงๆ กลายเป็นสิ่งที่สังเกตเห็นได้ตอนรีวิว
    """
    return dataset.X
