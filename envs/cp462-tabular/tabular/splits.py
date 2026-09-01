"""แบ่งไฟล์เดียวเป็นสามกอง — **แบบ stratified เสมอ**

    student        แจกนิสิต · เขาแบ่ง train/val/test เองด้วยเมล็ดของเขาเอง
    test_public    🔒 leaderboard ระหว่างเทอม
    test_private   🔒 ตัดสินตอนปิดรับ

**ทำไมต้อง stratify** — เดิมใช้ `permutation` ธรรมดาโดยให้เหตุผลว่า "ชุดหลักพัน
แถวใหญ่พอที่การสับธรรมดาจะได้สัดส่วนใกล้เคียงกันอยู่แล้ว" ซึ่งจริงสำหรับข้อมูล
สังเคราะห์ที่สมดุล แต่ **ชุดข้อมูลจริงของวิชานี้ไม่สมดุล** · โจทย์แบบ churn หรือ
การคัดกรองโรคมักมีคลาสบวกอยู่ 5–15% พอสุ่มธรรมดาแล้วตัด 15% เป็นชุดตัดสินสุดท้าย
สัดส่วนคลาสบวกในกองนั้นแกว่งได้หลายจุดเปอร์เซ็นต์ตามเมล็ด — ซึ่งแปลว่า **เมล็ด
ที่ผู้สอนเลือกมีผลต่ออันดับสุดท้าย** และไม่มีใครมองเห็น

**regression ก็ stratify** โดยแบ่ง `y` เป็นช่วงตามควอนไทล์ก่อน — ถ้าไม่ทำ กองที่
ใช้ตัดสินอาจไม่มีบ้านราคาแพงเลยสักหลัง แล้ว R² จะวัดคนละเรื่องกับที่ตั้งใจ

**การแบ่งต้องทำซ้ำได้ทุกบิต** — ใช้ `numpy.random.Generator` สายเดียวและ
**ไม่ใช้ `train_test_split` ของ sklearn** ซึ่งผูกกับ `RandomState` แบบเก่าและ
เปลี่ยนพฤติกรรมข้ามเวอร์ชันได้
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tabular.table import Dataset

#: ชื่อของสามกอง เรียงตามลำดับที่ตัด
PARTS = ("student", "test_public", "test_private")

#: ส่วนของชุดที่ใช้ตัดสิน — สองกองหลัง
GRADING_PARTS = ("test_public", "test_private")

#: จำนวนช่วงที่ใช้ stratify ของ regression — 10 ควอนไทล์
#: มากกว่านี้แต่ละช่วงจะมีน้อยแถวจนการแบ่งไม่ต่างจากสุ่มธรรมดา
REGRESSION_BINS = 10


class SplitError(Exception):
    """แบ่งไม่ได้ — ต้องบอกตั้งแต่ตอนสร้างโจทย์ ไม่ใช่ตอนให้คะแนน"""


@dataclass(frozen=True)
class ThreeWay:
    """สามกองของไฟล์เดียว — **`student` เท่านั้นที่ออกจากเซิร์ฟเวอร์ได้**"""

    student: Dataset
    test_public: Dataset
    test_private: Dataset

    def sizes(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in PARTS}


def strata_of(y: pd.Series, *, kind: str) -> pd.Series:
    """ป้ายกลุ่มที่ใช้ stratify — คลาสสำหรับ classification, ช่วงควอนไทล์สำหรับ regression

    คืนเป็น `Series` ของสตริงเสมอ เพื่อให้ลำดับของกลุ่มเรียงได้แน่นอนข้ามชนิดข้อมูล
    (ค่า `None` กับตัวเลขเรียงเทียบกันไม่ได้ใน Python 3)
    """
    if kind == "classification":
        # ค่าว่างเป็นกลุ่มของตัวเอง — ถ้าปล่อยเป็น NaN มันจะหายไปจากการ groupby
        # แล้วแถวนั้นตกหล่นจากทั้งสามกองโดยไม่มีใครรู้
        #
        # ป้ายต้องอ่านเหมือนที่ผู้สอนเห็นในไฟล์ · คอลัมน์ที่มีค่าว่างจะถูก pandas
        # อ่านเป็น float ทำให้คลาส `0` กลายเป็น `"0.0"` แล้วรายงาน "คลาสไหนบางไป"
        # จะเรียกชื่อคลาสด้วยคำที่ไม่มีอยู่ในไฟล์ของเขา
        return y.map(_label).astype(str)

    ranks = y.rank(method="first", na_option="bottom")
    bins = min(REGRESSION_BINS, max(1, y.nunique(dropna=True)))
    return pd.Series(
        np.floor((ranks - 1) / len(y) * bins).astype(int).astype(str), index=y.index
    )


def _label(value) -> str:
    """ชื่อของกลุ่มหนึ่ง — ตัด `.0` ที่ pandas เติมให้คอลัมน์จำนวนเต็มที่มีค่าว่าง"""
    if value is None or value != value:  # NaN
        return "<ว่าง>"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _cut(total: int, student_ratio: float, final_ratio: float) -> list[int]:
    """แปลงสัดส่วนสองตัวเป็นจำนวนแถวสามกอง — ผลรวมเท่ากับ `total` เป๊ะเสมอ

    **ทั้งสองสัดส่วนวัดจากทั้งไฟล์เหมือนกัน** กองที่สามเป็นเศษที่เหลือ · เดิม
    ตัวที่สองเป็นสัดส่วน*ภายในส่วนที่กันไว้* ซึ่งอธิบายให้ผู้สอนเข้าใจไม่ได้จริง
    — เขาถามตรงๆ ว่าเลขนั้นมาจากไหน

    ปัดแบบครึ่งขึ้น (`+ 0.5`) ไม่ใช่ `round()` ของ Python ซึ่งปัดเข้าหาเลขคู่
    (`round(12.5) == 12` แต่ `round(37.5) == 38`) — พฤติกรรมที่อธิบายยากมาก
    เวลาผู้สอนถามว่าทำไมสองคลาสที่ตั้งค่าเหมือนกันได้ผลคนละแบบ

    กองกลาง (leaderboard) รับเศษทั้งหมด เพราะมันเป็นกองที่คำนวณจากอีกสองกอง
    อยู่แล้ว — การให้เศษไปกองที่ผู้สอนกรอกเองจะทำให้ตัวเลขที่เห็นไม่ตรงกับที่พิมพ์
    """
    student = int(total * student_ratio + 0.5)
    final = int(total * final_ratio + 0.5)
    return [student, total - student - final, final]


def _spread(sizes: dict[str, int]) -> list[tuple[float, str, int]]:
    """ลำดับที่กลุ่มทุกกลุ่มกระจายสม่ำเสมอตลอดสาย — **หัวใจของการ stratify ที่นี่**

    ให้สมาชิกลำดับที่ `j` ของกลุ่มขนาด `m` อยู่ที่ตำแหน่ง `(j + 0.5) / m` แล้ว
    เรียงทุกกลุ่มรวมกันด้วยค่านั้น · ผลคือ **ตัดตรงไหนก็ได้สัดส่วนคลาสใกล้เคียง
    ทั้งไฟล์** เพราะแต่ละกลุ่มถูกโรยเท่าๆ กันตลอดสาย

    ทำแบบนี้แทนการตัดทีละกลุ่มแยกกัน เพราะการตัดทีละกลุ่มต้องปัดหนึ่งครั้งต่อ
    กลุ่ม แล้วเศษสะสมเข้าทางเดียวกันหมด — regression แบ่งเป็น 10 ช่วง ขนาดของ
    กองจึงเลื่อนได้ถึง 10 แถวจากที่สัดส่วนบอกไว้ ซึ่งอธิบายให้ผู้สอนไม่ได้
    ตัดครั้งเดียวบนสายที่เรียงแล้วปัดแค่ครั้งเดียว ขนาดจึงตรงเป๊ะเสมอ

    คืน `(ตำแหน่ง, ชื่อกลุ่ม, ลำดับในกลุ่ม)` ยังไม่ผูกกับแถวจริง เพื่อให้
    `thin_strata` นับล่วงหน้าได้โดยไม่ต้องมีข้อมูลหรือเมล็ด
    """
    out = [
        ((j + 0.5) / size, label, j)
        for label, size in sorted(sizes.items())
        for j in range(size)
    ]
    # `label` เป็นตัวตัดสินเมื่อตำแหน่งเท่ากัน — ไม่งั้นลำดับจะขึ้นกับ hash ของ dict
    out.sort(key=lambda item: (item[0], item[1]))
    return out


def _allocate(
    sizes: dict[str, int], student_ratio: float, final_ratio: float
) -> dict[str, dict[str, int]]:
    """แต่ละกลุ่มจะมีกี่แถวในแต่ละกอง — คำนวณจากขนาดกลุ่มล้วน ไม่ต้องใช้ข้อมูลจริง

    ใช้ร่วมกันระหว่างการแบ่งจริงกับการรายงานล่วงหน้า **เพื่อให้ตัวเลขที่ผู้สอน
    เห็นก่อนกดสร้างเป็นตัวเลขเดียวกับที่เกิดขึ้นจริง** ไม่ใช่ค่าประมาณคนละสูตร
    """
    order = _spread(sizes)
    counts = {label: dict.fromkeys(PARTS, 0) for label in sizes}
    start = 0
    for name, count in zip(PARTS, _cut(len(order), student_ratio, final_ratio)):
        for _, label, _j in order[start : start + count]:
            counts[label][name] += 1
        start += count
    return counts


def three_way(
    dataset: Dataset,
    *,
    kind: str,
    seed: int,
    student_ratio: float,
    final_ratio: float,
) -> ThreeWay:
    """แบ่งสามกองแบบ stratified

    สัดส่วนทั้งสองตัววัดจาก**ทั้งไฟล์** — กอง leaderboard คือส่วนที่เหลือ
    """
    strata = strata_of(dataset.y, kind=kind)
    rng = np.random.default_rng(seed)

    # เรียงชื่อกลุ่มเสมอ — ลำดับที่ `unique()` คืนมาขึ้นกับลำดับที่พบในข้อมูล
    # ซึ่งจะทำให้การแบ่งเปลี่ยนไปตามการเรียงของไฟล์โดยไม่มีใครตั้งใจ
    labels = strata.to_numpy()
    positions = np.arange(len(dataset.y))
    members = {
        label: positions[labels == label][rng.permutation(int((labels == label).sum()))]
        for label in sorted(strata.unique())
    }

    # โรยทุกกลุ่มให้กระจายทั่วสาย แล้วตัดครั้งเดียว — ขนาดของกองจึงตรงกับสัดส่วนเป๊ะ
    order = np.array(
        [members[label][j] for _, label, j in _spread({k: len(v) for k, v in members.items()})],
        dtype=int,
    )

    parts = {}
    start = 0
    for name, count in zip(PARTS, _cut(len(order), student_ratio, final_ratio)):
        idx = order[start : start + count]
        start += count
        # สับอีกครั้งภายในกอง — ไม่งั้นแถวจะเรียงสลับคลาสเป็นจังหวะตายตัว แล้วโค้ด
        # ที่เผลอใช้ `head()` แทนการสุ่มจะได้ผลที่ดูดีเกินจริง
        idx = idx[rng.permutation(len(idx))]
        parts[name] = Dataset(
            # `reset_index` เพื่อให้ index เป็น 0..n-1 เสมอ — index ที่กระโดด
            # จะทำให้นิสิตที่ใช้ `.loc` ได้ผลต่างจากที่คาด
            X=dataset.X.iloc[idx].reset_index(drop=True),
            y=dataset.y.iloc[idx].reset_index(drop=True),
        )

    result = ThreeWay(**parts)
    empty = [name for name, size in result.sizes().items() if size == 0]
    if empty:
        raise SplitError(
            f"แบ่งแล้วมีกองที่ว่าง: {empty} — ข้อมูล {len(dataset)} แถวน้อยเกินไป"
            f"สำหรับสัดส่วนนี้ (ได้ {result.sizes()})"
        )
    return result


def thin_strata(
    y: pd.Series,
    *,
    kind: str,
    student_ratio: float,
    final_ratio: float,
    floor: int = 5,
) -> dict[str, dict[str, int]]:
    """กลุ่มที่จะเหลือน้อยเกินไปในกองที่ใช้ตัดสิน — **ตรวจก่อนสร้างโจทย์**

    คืน `{ชื่อกลุ่ม: {ชื่อกอง: จำนวน}}` เฉพาะกลุ่มที่มีกองไหนต่ำกว่า `floor`
    ผู้สอนควรเห็นตัวเลขนี้ตอนกดสร้าง ไม่ใช่ไปเจอว่า macro-F1 แกว่งทั้งเทอม
    เพราะคลาสหนึ่งมีอยู่ 3 แถวในชุดที่ใช้ตัดสินสุดท้าย
    """
    strata = strata_of(y, kind=kind)
    sizes = {label: int((strata == label).sum()) for label in sorted(strata.unique())}
    allocated = _allocate(sizes, student_ratio, final_ratio)
    return {
        label: counts
        for label, counts in allocated.items()
        if min(counts[name] for name in GRADING_PARTS) < floor
    }


def as_frame(dataset: Dataset) -> pd.DataFrame:
    """รวม X กับ y เป็นตารางเดียวสำหรับเขียนไฟล์ให้นิสิต

    ใช้กับกอง `student` เท่านั้น — สองกองที่ใช้ตัดสินต้องไม่มีเฉลยติดไปด้วย
    """
    return pd.concat([dataset.X, dataset.y], axis=1)
