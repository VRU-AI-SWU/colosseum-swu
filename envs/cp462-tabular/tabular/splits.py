"""แบ่งข้อมูลเป็น 4 ส่วน — template §1

    train          นิสิตได้ พร้อมเฉลย        เอาไปเทรน
    val            นิสิตได้ พร้อมเฉลย        เอาไปเลือกโมเดล/จูน
    test_public    🔒 อยู่ในระบบ             ให้คะแนนบน leaderboard ระหว่างเทอม
    test_private   🔒 อยู่ในระบบ             ตัดสินเกรดตอนปิดรับ

**ทำไม test ต้องแยกเป็นสองชุด** — ถ้ามีชุดเดียว ทีมที่ส่งวันละ 5 ครั้งตลอดเทอม
จะค่อยๆ จูนเข้าหาชุดนั้นจนคะแนนบน leaderboard สูงเกินความสามารถจริง
(template §1.1) · `test_private` ไม่เคยให้ผลกลับเลยจนถึงวันปิดรับ จึงเป็น
ตัวเดียวที่บอกว่าโมเดลทำงานกับข้อมูลที่ไม่เคยเห็นจริงๆ ได้แค่ไหน

**การแบ่งต้องทำซ้ำได้ทุกบิต** — นิสิตแบ่งด้วยเมล็ดเดียวกับที่ผู้สอนใช้ จึงได้
`train`/`val` ชุดเดียวกันเป๊ะ · ถ้าต่างกัน คะแนนที่นิสิตวัดเองจะเทียบกับ
leaderboard ไม่ได้ และไม่มีใครรู้ว่าทำไม

ใช้ `numpy.random.Generator.permutation` ตัวเดียว **ไม่ใช้ `train_test_split`
ของ sklearn** ซึ่งผูกกับ `RandomState` แบบเก่าและเปลี่ยนพฤติกรรมข้ามเวอร์ชันได้
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tabular.generator import Dataset

#: ชื่อของแต่ละส่วน เรียงตามลำดับที่ตัดจากข้อมูลที่สับแล้ว
PARTS = ("train", "val", "test_public", "test_private")


@dataclass(frozen=True)
class Split:
    """สี่ส่วนของข้อมูลชุดเดียวกัน

    **`open_parts` คือสิ่งเดียวที่นิสิตได้รับ** — เมธอดนี้มีไว้ให้เรียกแทนการหยิบ
    ฟิลด์เอง เพราะการเผลอส่ง `test_private` ออกไปคือความผิดพลาดที่กู้ไม่ได้
    """

    train: Dataset
    val: Dataset
    test_public: Dataset
    test_private: Dataset

    def open_parts(self) -> dict[str, Dataset]:
        return {"train": self.train, "val": self.val}

    def sizes(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in PARTS}


def _cut(total: int, ratios: tuple[float, ...]) -> list[int]:
    """แปลงสัดส่วนเป็นจำนวนแถว โดยให้ผลรวมเท่ากับ `total` เป๊ะ

    ปัดลงทุกส่วนแล้วโยนเศษที่เหลือให้ `train` — ถ้าปัดแบบธรรมดา ผลรวมอาจขาด
    หรือเกินไปหนึ่งถึงสองแถว แล้วจำนวนแถวจะขึ้นกับการปัดของแต่ละเครื่อง
    """
    counts = [int(total * r) for r in ratios]
    counts[0] += total - sum(counts)
    return counts


def split(
    dataset: Dataset, *, seed: int, ratios: tuple[float, float, float, float]
) -> Split:
    """สับแล้วตัดเป็นสี่ส่วน

    สับด้วย `permutation` แล้วตัดตามลำดับ — **ไม่ stratify** เพราะการ stratify
    ต้องรู้เป้าหมาย ซึ่งทำให้การแบ่งของ classification กับ regression ต่างกัน
    แล้วโค้ดจะแตกเป็นสองทาง · ขนาดชุดที่ใช้ (หลักพัน) ใหญ่พอที่การสับธรรมดา
    จะได้สัดส่วนคลาสใกล้เคียงกันอยู่แล้ว — มีเทสต์ยืนยันข้อนี้
    """
    if len(ratios) != len(PARTS):
        raise ValueError(f"ต้องมี {len(PARTS)} สัดส่วน — ได้ {len(ratios)}")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"สัดส่วนต้องรวมกันได้ 1.0 — ได้ {sum(ratios)}")
    if any(r <= 0 for r in ratios):
        raise ValueError("ทุกส่วนต้องมีข้อมูลอย่างน้อยเล็กน้อย — สัดส่วนต้องเป็นบวก")

    total = len(dataset)
    counts = _cut(total, ratios)
    if min(counts) < 1:
        raise ValueError(
            f"ข้อมูล {total} แถวน้อยเกินไปสำหรับสัดส่วนนี้ — "
            f"จะได้ {dict(zip(PARTS, counts))} ซึ่งมีส่วนที่ว่าง"
        )

    order = np.random.default_rng(seed).permutation(total)
    parts: dict[str, Dataset] = {}
    start = 0
    for name, count in zip(PARTS, counts):
        idx = order[start : start + count]
        parts[name] = Dataset(
            # `reset_index` เพื่อให้ index เป็น 0..n-1 เสมอ — index ที่กระโดด
            # จะทำให้นิสิตที่ใช้ `.loc` ได้ผลต่างจากที่คาด และทำให้ fingerprint
            # ขึ้นกับลำดับเดิมโดยไม่จำเป็น
            X=dataset.X.iloc[idx].reset_index(drop=True),
            y=dataset.y.iloc[idx].reset_index(drop=True),
        )
        start += count
    return Split(**parts)


def as_frame(dataset: Dataset) -> pd.DataFrame:
    """รวม X กับ y เป็นตารางเดียวสำหรับเขียนไฟล์ให้นิสิต

    ใช้กับ `train`/`val` เท่านั้น — ชุดที่ใช้ตัดสินต้องไม่มีเฉลยติดไปด้วย
    """
    return pd.concat([dataset.X, dataset.y], axis=1)
