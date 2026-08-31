"""ชนิดข้อมูลพื้นฐานที่ทุกโมดูลใช้ร่วมกัน — ไม่มี dependency ไปหาโมดูลอื่นในแพ็กเกจ

แยกออกมาเพื่อตัดวงกลมของการ import: `store` → `table`, `splits` → `table`,
`dataset` → ทั้งสอง · เดิม `Dataset` อยู่ใน `generator` ซึ่งกลายเป็นเรื่องแปลก
ตั้งแต่ตัวสร้างข้อมูลสังเคราะห์เลิกเป็นแหล่งข้อมูลของการให้คะแนน
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Dataset:
    """ข้อมูลหนึ่งชุดพร้อมเป้าหมาย — `X` กับ `y` แยกกันเสมอ

    แยกเพราะ **`y` ของชุดที่ใช้ตัดสินไม่เคยเข้าไปใน sandbox** (template §5)
    การเก็บรวมใน DataFrame เดียวทำให้เผลอส่งเข้าไปทั้งก้อนได้ง่ายเกินไป
    """

    X: pd.DataFrame
    y: pd.Series

    def __len__(self) -> int:
        return len(self.X)


def fingerprint(frame: pd.DataFrame | pd.Series) -> str:
    """ลายนิ้วมือของข้อมูล — ใช้ตรวจว่าเครื่องนิสิตได้ไฟล์ตรงกับที่ grader ใช้

    ใช้ CSV เป็นตัวกลางเพราะมันเสถียรข้ามเวอร์ชัน pandas มากกว่า pickle/parquet
    และตรึง `float_format` เพราะ repr ของ float เปลี่ยนได้ระหว่างเวอร์ชัน
    """
    blob = frame.to_csv(index=False, float_format="%.10g").encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]
