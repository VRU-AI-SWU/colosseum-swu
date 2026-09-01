"""ทางเข้าข้อมูล — **แยกให้ชัดว่าอะไรนิสิตได้ อะไรอยู่ในระบบ**

    student_data(spec)  กองที่แจก พร้อมเฉลย   → API เสิร์ฟเป็นไฟล์ให้ดาวน์โหลด
    grading_data(spec)  test_public/private   🔒 ไม่เคยออกจากเซิร์ฟเวอร์

**ทั้งสามกองมาจากไฟล์เดียวกัน แต่ไฟล์นั้นอยู่บนเซิร์ฟเวอร์ที่เดียว**

เดิมนิสิต *สร้างข้อมูลเอง* จากเมล็ดที่แจกไปในแพ็กเกจ ซึ่งเคยพังมาแล้วครั้งหนึ่ง
(นิสิตสร้างชุดที่ใช้ตัดสินเองได้ครบทุกแถว ทำคะแนน 1.0000) แล้วถูกปะด้วยการแยก
เป็นสอง dataset จากเมล็ดลับ · ตอนนี้เลิกใช้เมล็ดทั้งหมด:

  ก่อน   ความลับ = ตัวเลขที่ใช้สร้างข้อมูล   → รั่วเมื่อไร สร้างเฉลยได้ทั้งชุด
  ตอนนี้ ความลับ = ตัวข้อมูลเอง              → ไม่มีตัวเลขไหนที่ย้อนกลับไปได้

**นิสิตแบ่ง train/val/test เอง** ด้วยเมล็ดของเขาเอง ระบบไม่ยุ่งและไม่ต้องรู้ —
หน้าที่ของระบบคือรับ pipeline กับโมเดลเข้ามาแล้ววัดกับข้อมูลที่เขาไม่เคยเห็น
ซึ่งตรงกับที่เขาจะเจอจริงนอกห้องเรียน

⚠️ **`grading_data` คืนเฉลย** — ผู้เรียกส่งเข้ากล่องได้เฉพาะ `.X`
"""

from __future__ import annotations

import pandas as pd

from tabular import splits, store
from tabular.config import TaskSpec
from tabular.table import Dataset


def _full(spec: TaskSpec) -> Dataset:
    """🔒 ไฟล์เต็มพร้อมเฉลยทุกแถว — จุดเดียวที่อ่านคลัง"""
    frame = store.read(spec.dataset)
    return to_dataset(frame, target=spec.target, drop=spec.drop, source=spec.dataset)


def to_dataset(
    frame: pd.DataFrame, *, target: str, drop: list[str], source: str = "ชุดข้อมูล"
) -> Dataset:
    """แยกคอลัมน์เฉลยออกจากฟีเจอร์ แล้วตัดคอลัมน์ที่ไม่ให้โมเดลเห็น

    แยกเป็นฟังก์ชันสาธารณะเพราะหน้าเว็บต้องเรียกตอนตรวจไฟล์ที่เพิ่งอัปโหลด
    ก่อนจะมี `TaskSpec` — และการตรวจตอนนั้นต้องเดินทางเดียวกับตอนให้คะแนนเป๊ะ
    """
    if target not in frame.columns:
        raise store.DatasetError(
            f"{source}: ไม่มีคอลัมน์เฉลย {target!r} — ที่มีคือ {list(frame.columns)}"
        )
    missing = [name for name in drop if name not in frame.columns]
    if missing:
        raise store.DatasetError(f"{source}: จะตัดคอลัมน์ {missing} แต่ไม่มีในไฟล์")

    y = frame[target]
    if y.isna().any():
        raise store.DatasetError(
            f"{source}: คอลัมน์เฉลย {target!r} มีค่าว่าง {int(y.isna().sum())} แถว\n"
            "  แถวที่ไม่มีเฉลยให้คะแนนไม่ได้ — ลบออกจากไฟล์ก่อนอัปโหลด"
        )
    X = frame.drop(columns=[target, *drop])
    if X.empty or not len(X.columns):
        raise store.DatasetError(f"{source}: ตัดแล้วไม่เหลือคอลัมน์ฟีเจอร์เลย")
    return Dataset(X=X, y=y)


def parts(spec: TaskSpec) -> splits.ThreeWay:
    """🔒 สามกองของไฟล์ — ผู้เรียกต้องอยู่ฝั่ง trusted"""
    return splits.three_way(
        _full(spec),
        kind=spec.kind,
        seed=spec.split_seed,
        student_ratio=spec.student_ratio,
        final_ratio=spec.final_ratio,
    )


def student_data(spec: TaskSpec) -> Dataset:
    """กองที่แจกนิสิต พร้อมเฉลย — สิ่งเดียวที่ออกจากเซิร์ฟเวอร์ได้"""
    return parts(spec).student


def student_csv(spec: TaskSpec) -> bytes:
    """ไฟล์ที่นิสิตดาวน์โหลด — ฟีเจอร์ + เฉลย ของกองที่แจกเท่านั้น"""
    return splits.as_frame(student_data(spec)).to_csv(index=False).encode("utf-8")


def grading_data(spec: TaskSpec, kind: str) -> Dataset:
    """🔒 ชุดที่ใช้ตัดสิน — `"public"` ระหว่างเทอม · `"private"` ตอนปิดรับ

    รับ `kind` เป็นสตริงแทนการให้เลือกฟิลด์เอง เพื่อให้จุดที่หยิบชุดลับมีชื่อชัดเจน
    และหาได้ด้วยการค้นหาคำเดียวตอนตรวจว่ามีใครเรียกผิดที่ไหม
    """
    if kind not in ("public", "private", *splits.GRADING_PARTS):
        raise ValueError(f"kind ต้องเป็น 'public' หรือ 'private' — ได้ {kind!r}")
    split = parts(spec)
    return split.test_public if kind in ("public", "test_public") else split.test_private


def features_only(dataset: Dataset) -> pd.DataFrame:
    """ฟีเจอร์ล้วนสำหรับส่งเข้า sandbox — **เฉลยไม่ตามไปด้วย**

    ฟังก์ชันบางๆ ที่ดูเหมือนไม่จำเป็น แต่ทำให้จุดที่ตัดเฉลยออกมีชื่อ และทำให้
    การส่ง `dataset.X` ตรงๆ กลายเป็นสิ่งที่สังเกตเห็นได้ตอนรีวิว
    """
    return dataset.X
