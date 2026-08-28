"""แบ่งข้อมูล — **สอง dataset คนละชุด ไม่ใช่ชุดเดียวแบ่งสี่ส่วน**

    ชุดของนิสิต (เมล็ดสาธารณะ · อยู่ในแพ็กเกจที่แจก)
      train   เอาไปเทรน
      val     เอาไปเลือกโมเดล/จูน
      test    เอาไปวัดเองก่อนส่ง

    ชุดที่ใช้ตัดสิน (🔒 เมล็ดอยู่ใน ARENA_SECRETS — คนละ dataset กันคนละใบ)
      test_public    ให้คะแนนบน leaderboard ระหว่างเทอม
      test_private   ตัดสินเกรดตอนปิดรับ

⚠️ **เดิมทั้งห้าส่วนมาจาก dataset ชุดเดียวที่สร้างจากเมล็ดในไฟล์ config ที่แจก
ให้นิสิต ซึ่งแปลว่านิสิตคำนวณเฉลยของชุดที่ใช้ตัดสินเองได้ทั้งหมด** — ทดสอบแล้วได้
macro-F1 = 1.0000 โดยจำเฉลยไว้แล้วจับคู่ด้วย `account_id` ที่ส่งเข้ากล่องอยู่แล้ว
การแยกเป็นคนละ dataset ที่สร้างจากเมล็ดลับคือสิ่งที่ปิดช่องนั้น และตรงกับที่
ผู้สอนออกแบบไว้ตั้งแต่ต้นว่า "มี dataset อีกชุดหนึ่งเก็บไว้ในระบบเป็น unseen data"

**ทำไมชุดตัดสินต้องแยกเป็นสองส่วน** — ถ้ามีส่วนเดียว ทีมที่ส่งวันละ 5 ครั้งตลอดเทอม
จะค่อยๆ จูนเข้าหาส่วนนั้นจนคะแนนบน leaderboard สูงเกินความสามารถจริง
(template §1.1) · `test_private` ไม่เคยให้ผลกลับเลยจนถึงวันปิดรับ

**การแบ่งต้องทำซ้ำได้ทุกบิต** — นิสิตแบ่งด้วยเมล็ดเดียวกับที่ผู้สอนใช้ จึงได้
`train`/`val`/`test` ชุดเดียวกันเป๊ะ · ถ้าต่างกัน คะแนนที่นิสิตวัดเองจะเทียบกับ
leaderboard ไม่ได้ และไม่มีใครรู้ว่าทำไม

ใช้ `numpy.random.Generator.permutation` ตัวเดียว **ไม่ใช้ `train_test_split`
ของ sklearn** ซึ่งผูกกับ `RandomState` แบบเก่าและเปลี่ยนพฤติกรรมข้ามเวอร์ชันได้
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tabular.generator import Dataset

#: ส่วนของชุดที่แจกนิสิต เรียงตามลำดับที่ตัดจากข้อมูลที่สับแล้ว
PARTS = ("train", "val", "test")

#: ส่วนของชุดที่ใช้ตัดสิน
GRADING_PARTS = ("test_public", "test_private")


@dataclass(frozen=True)
class Split:
    """สามส่วนของชุดที่แจกนิสิต — **ทุกส่วนในนี้นิสิตมีอยู่แล้ว**

    ต่างจากเดิมที่คลาสนี้ถือชุดที่ใช้ตัดสินไว้ด้วย · ตอนนี้ชุดนั้นเป็น `GradingSplit`
    ที่สร้างได้เฉพาะเมื่อมีเมล็ดลับ ทำให้ "เผลอส่งเฉลยออกไป" กลายเป็นสิ่งที่
    เขียนไม่ได้ ไม่ใช่แค่สิ่งที่ห้ามเขียน
    """

    train: Dataset
    val: Dataset
    test: Dataset

    def open_parts(self) -> dict[str, Dataset]:
        return {name: getattr(self, name) for name in PARTS}

    def sizes(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in PARTS}


@dataclass(frozen=True)
class GradingSplit:
    """🔒 สองส่วนของชุดที่ใช้ตัดสิน — มาจาก dataset คนละใบกับของนิสิต"""

    test_public: Dataset
    test_private: Dataset

    def sizes(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in GRADING_PARTS}


def _cut(total: int, ratios: tuple[float, ...]) -> list[int]:
    """แปลงสัดส่วนเป็นจำนวนแถว โดยให้ผลรวมเท่ากับ `total` เป๊ะ

    ปัดลงทุกส่วนแล้วโยนเศษที่เหลือให้ `train` — ถ้าปัดแบบธรรมดา ผลรวมอาจขาด
    หรือเกินไปหนึ่งถึงสองแถว แล้วจำนวนแถวจะขึ้นกับการปัดของแต่ละเครื่อง
    """
    counts = [int(total * r) for r in ratios]
    counts[0] += total - sum(counts)
    return counts


def _parts(dataset: Dataset, *, seed: int, ratios, names) -> dict[str, Dataset]:
    """สับแล้วตัดตามสัดส่วน — คืนเป็น dict ตามชื่อที่ให้มา"""
    if len(ratios) != len(names):
        raise ValueError(f"ต้องมี {len(names)} สัดส่วน — ได้ {len(ratios)}")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"สัดส่วนต้องรวมกันได้ 1.0 — ได้ {sum(ratios)}")
    if any(r <= 0 for r in ratios):
        raise ValueError("ทุกส่วนต้องมีข้อมูลอย่างน้อยเล็กน้อย — สัดส่วนต้องเป็นบวก")

    total = len(dataset)
    counts = _cut(total, tuple(ratios))
    if min(counts) < 1:
        raise ValueError(
            f"ข้อมูล {total} แถวน้อยเกินไปสำหรับสัดส่วนนี้ — "
            f"จะได้ {dict(zip(names, counts))} ซึ่งมีส่วนที่ว่าง"
        )

    order = np.random.default_rng(seed).permutation(total)
    parts: dict[str, Dataset] = {}
    start = 0
    for name, count in zip(names, counts):
        idx = order[start : start + count]
        parts[name] = Dataset(
            # `reset_index` เพื่อให้ index เป็น 0..n-1 เสมอ — index ที่กระโดด
            # จะทำให้นิสิตที่ใช้ `.loc` ได้ผลต่างจากที่คาด และทำให้ fingerprint
            # ขึ้นกับลำดับเดิมโดยไม่จำเป็น
            X=dataset.X.iloc[idx].reset_index(drop=True),
            y=dataset.y.iloc[idx].reset_index(drop=True),
        )
        start += count
    return parts


def grading_split(dataset: Dataset, *, seed: int, public_ratio: float) -> GradingSplit:
    """🔒 แบ่งชุดลับเป็น public/private"""
    return GradingSplit(
        **_parts(dataset, seed=seed, ratios=(public_ratio, 1.0 - public_ratio),
                 names=GRADING_PARTS)
    )


def split(dataset: Dataset, *, seed: int, ratios: tuple[float, float, float]) -> Split:
    """สับแล้วตัดเป็นสามส่วนสำหรับนิสิต

    สับด้วย `permutation` แล้วตัดตามลำดับ — **ไม่ stratify** เพราะการ stratify
    ต้องรู้เป้าหมาย ซึ่งทำให้การแบ่งของ classification กับ regression ต่างกัน
    แล้วโค้ดจะแตกเป็นสองทาง · ขนาดชุดที่ใช้ (หลักพัน) ใหญ่พอที่การสับธรรมดา
    จะได้สัดส่วนคลาสใกล้เคียงกันอยู่แล้ว — มีเทสต์ยืนยันข้อนี้
    """
    return Split(**_parts(dataset, seed=seed, ratios=ratios, names=PARTS))


def as_frame(dataset: Dataset) -> pd.DataFrame:
    """รวม X กับ y เป็นตารางเดียวสำหรับเขียนไฟล์ให้นิสิต

    ใช้กับชุดของนิสิตเท่านั้น — ชุดที่ใช้ตัดสินต้องไม่มีเฉลยติดไปด้วย
    """
    return pd.concat([dataset.X, dataset.y], axis=1)
