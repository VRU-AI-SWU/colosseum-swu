"""ทางเข้าข้อมูล — **แยกให้ชัดว่าอะไรนิสิตได้ อะไรอยู่ในระบบ**

    open_data(spec)     train + val พร้อมเฉลย        นิสิตเรียกได้
    grading_data(spec)  test_public + test_private   🔒 ฝั่ง trusted เท่านั้น

**ทำไมนิสิตสร้างข้อมูลเองแทนที่จะดาวน์โหลดไฟล์** — ตัวสร้างทำซ้ำได้ทุกบิต
นิสิตจึงได้ `train`/`val` ชุดเดียวกับที่ผู้สอนใช้เป๊ะ โดยไม่ต้องแจกไฟล์
ไม่ต้องกังวลว่าใครโหลดผิดเวอร์ชัน และไม่มีไฟล์ให้หลุด · `selfcheck` ยืนยันว่า
เครื่องนิสิตได้ข้อมูลตรงกับ grader จริง

`grading_data` อยู่ไฟล์เดียวกันเพราะมันมาจากการแบ่งชุดเดียวกัน — **แต่ถูกเรียก
เฉพาะฝั่ง trusted** โค้ดนิสิตไม่มีทางเรียกได้เพราะรันคนละ process และไม่ได้รับ
`spec` ที่มีเมล็ด (template §5)

> 🔻 **จุดสลับเป็นข้อมูลจริง** — เมื่อวิชามี dataset จริงแล้ว แก้แค่ `_full()`
> ให้อ่านไฟล์แทนการเรียก generator · ส่วนที่เหลือทั้งหมดไม่ต้องแตะ
"""

from __future__ import annotations

from tabular import generator, splits
from tabular.config import TaskSpec


def _full(spec: TaskSpec) -> generator.Dataset:
    """ข้อมูลทั้งชุดก่อนแบ่ง — **จุดเดียวที่ต้องแก้ตอนสลับเป็นข้อมูลจริง**"""
    return generator.make(spec.task, seed=spec.data_seed, n=spec.n_rows)


def all_parts(spec: TaskSpec) -> splits.Split:
    """สี่ส่วนของโจทย์นี้ — ฝั่ง trusted เท่านั้น"""
    return splits.split(_full(spec), seed=spec.split_seed, ratios=spec.ratios)


def open_data(spec: TaskSpec) -> dict[str, generator.Dataset]:
    """`train` กับ `val` พร้อมเฉลย — สิ่งเดียวที่นิสิตได้รับ"""
    return all_parts(spec).open_parts()


def grading_data(spec: TaskSpec, kind: str) -> generator.Dataset:
    """ชุดที่ใช้ตัดสิน — `"public"` ระหว่างเทอม · `"private"` ตอนปิดรับ

    รับ `kind` เป็นสตริงแทนการให้เลือกฟิลด์เอง เพื่อให้จุดที่หยิบชุดลับมีชื่อชัดเจน
    และหาได้ด้วยการค้นหาคำเดียวตอนตรวจว่ามีใครเรียกผิดที่ไหม
    """
    parts = all_parts(spec)
    if kind == "public":
        return parts.test_public
    if kind == "private":
        return parts.test_private
    raise ValueError(f"kind ต้องเป็น 'public' หรือ 'private' — ได้ {kind!r}")


def features_only(dataset: generator.Dataset):
    """ฟีเจอร์ล้วนสำหรับส่งเข้า sandbox — **เฉลยไม่ตามไปด้วย**

    ฟังก์ชันบางๆ ที่ดูเหมือนไม่จำเป็น แต่ทำให้จุดที่ตัดเฉลยออกมีชื่อ และทำให้
    การส่ง `dataset.X` ตรงๆ กลายเป็นสิ่งที่สังเกตเห็นได้ตอนรีวิว
    """
    return dataset.X
