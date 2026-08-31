"""การแบ่งสามกอง — **ข้อที่พังแล้วการแข่งจบทันทีอยู่ข้างล่างสุด**

เรียงจากคุณสมบัติทั่วไปไปหาคุณสมบัติเรื่องความปลอดภัย · ข้อสุดท้ายคือข้อที่
เคยพังจริงมาแล้วในรูปแบบอื่น (นิสิตสร้างชุดที่ใช้ตัดสินเองได้จากเมล็ดที่แจก)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabular.splits import (
    GRADING_PARTS,
    PARTS,
    SplitError,
    as_frame,
    strata_of,
    thin_strata,
    three_way,
)
from tabular.table import Dataset, fingerprint


def _frame(n: int, *, positives: float = 0.25, seed: int = 0) -> Dataset:
    """ข้อมูลจำลองที่คุมสัดส่วนคลาสได้ — ใช้ทดสอบเคสไม่สมดุลโดยเฉพาะ"""
    rng = np.random.default_rng(seed)
    y = (np.arange(n) < int(n * positives)).astype(int)
    return Dataset(
        X=pd.DataFrame({"row": np.arange(n), "noise": rng.normal(size=n)}),
        y=pd.Series(rng.permutation(y), name="label"),
    )


def _split(data: Dataset, **kw):
    options = {"kind": "classification", "seed": 3,
               "student_ratio": 0.7, "grading_public_ratio": 0.5}
    options.update(kw)
    return three_way(data, **options)


# ── โครงของผลลัพธ์ ────────────────────────────────────────────────────────


def test_every_row_lands_in_exactly_one_part():
    """ไม่มีแถวหาย ไม่มีแถวซ้ำ — สิ่งแรกที่ต้องจริงก่อนจะพูดถึงอย่างอื่น"""
    data = _frame(1000)
    split = _split(data)

    seen = pd.concat([getattr(split, name).X["row"] for name in PARTS])
    assert len(seen) == 1000
    assert sorted(seen.tolist()) == list(range(1000))


def test_sizes_follow_the_two_nested_ratios():
    """`student_ratio` ตัดจากทั้งไฟล์ · `grading_public_ratio` ตัดจากส่วนที่เหลือ

    นี่คือจุดที่คนอ่านผิดได้ง่ายที่สุด — `grading_public_ratio=0.5` ไม่ได้แปลว่า
    ครึ่งหนึ่งของไฟล์ แต่แปลว่าครึ่งหนึ่งของ 30% ที่กันไว้ = 15% ของไฟล์
    """
    split = _split(_frame(1000), student_ratio=0.7, grading_public_ratio=0.5)
    assert split.sizes() == {"student": 700, "test_public": 150, "test_private": 150}

    split = _split(_frame(1000), student_ratio=0.8, grading_public_ratio=0.25)
    assert split.sizes() == {"student": 800, "test_public": 50, "test_private": 150}


@pytest.mark.parametrize("n", [997, 1000, 4321])
@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_part_sizes_never_drift_with_the_number_of_strata(n, kind):
    """ขนาดของกองต้องไม่ขึ้นกับว่ามีกี่กลุ่ม — regression มี 10 ช่วง

    เดิมตัดทีละกลุ่มแล้วปัดหนึ่งครั้งต่อกลุ่ม เศษจึงสะสมเข้าทางเดียว: 2 คลาส
    เลื่อนได้ 1–2 แถว แต่ 10 ช่วงเลื่อนได้ถึง 10 แถว · ข้อนี้ตรึงว่าเลื่อนไม่ได้เลย
    """
    data = (
        _frame(n, positives=0.3)
        if kind == "classification"
        else Dataset(X=pd.DataFrame({"row": np.arange(n)}),
                     y=pd.Series(np.linspace(0, 1, n), name="v"))
    )
    sizes = three_way(data, kind=kind, seed=5,
                      student_ratio=0.7, grading_public_ratio=0.5).sizes()

    student = int(n * 0.7 + 0.5)
    public = int((n - student) * 0.5 + 0.5)
    assert sizes == {"student": student, "test_public": public,
                     "test_private": n - student - public}


def test_the_preview_numbers_are_the_numbers_that_actually_happen():
    """`thin_strata` ต้องทำนายด้วยสูตรเดียวกับที่แบ่งจริง — ไม่ใช่ค่าประมาณ

    ผู้สอนตัดสินใจจากตัวเลขที่เห็นก่อนกดสร้าง · ถ้ามันเป็นคนละสูตรกับของจริง
    คำเตือน "คลาสนี้จะเหลือ 3 แถว" จะเชื่อไม่ได้ และคำที่เชื่อไม่ได้คือคำที่
    ทุกคนเรียนรู้ที่จะข้าม
    """
    from tabular.splits import _allocate

    data = _frame(3000, positives=0.11)
    split = three_way(data, kind="classification", seed=13,
                      student_ratio=0.75, grading_public_ratio=0.4)

    predicted = _allocate(
        {"0": int((data.y == 0).sum()), "1": int((data.y == 1).sum())},
        0.75, 0.4,
    )
    for name in PARTS:
        actual = getattr(split, name).y.value_counts()
        assert predicted["1"][name] == int(actual.get(1, 0)), name
        assert predicted["0"][name] == int(actual.get(0, 0)), name


def test_a_file_too_small_for_the_ratios_fails_loudly():
    with pytest.raises(SplitError, match="ว่าง"):
        _split(_frame(4, positives=0.5), student_ratio=0.9, grading_public_ratio=0.5)


# ── stratify ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("positives", [0.5, 0.25, 0.08, 0.03])
def test_class_balance_is_preserved_in_every_part(positives):
    """**เหตุผลที่ต้อง stratify** — สัดส่วนคลาสต้องเท่ากันทุกกอง แม้ข้อมูลจะเบ้มาก

    เดิมใช้ `permutation` ธรรมดาโดยให้เหตุผลว่าชุดใหญ่พอ · ข้อนี้ยิงที่ 3%
    ซึ่งเป็นระดับที่โจทย์จริงอย่างการคัดกรองโรคเป็น แล้วการสุ่มธรรมดาแกว่ง
    """
    data = _frame(4000, positives=positives)
    split = _split(data)

    for name in PARTS:
        got = getattr(split, name).y.mean()
        assert abs(got - positives) < 0.01, f"{name} ได้ {got:.3f} ควรใกล้ {positives}"


def test_regression_strata_cover_the_whole_range():
    """regression ก็ต้อง stratify — ไม่งั้นกองที่ใช้ตัดสินอาจไม่มีค่าสูงเลย"""
    n = 3000
    data = Dataset(
        X=pd.DataFrame({"row": np.arange(n)}),
        # y เบ้ขวาแรงๆ — ค่าสูงมีน้อย ซึ่งเป็นเคสที่การสุ่มธรรมดาพลาด
        y=pd.Series(np.exp(np.linspace(0, 6, n)), name="price"),
    )
    split = three_way(data, kind="regression", seed=1,
                      student_ratio=0.7, grading_public_ratio=0.5)

    whole = data.y.quantile([0.05, 0.95])
    for name in GRADING_PARTS:
        part = getattr(split, name).y
        assert part.max() >= whole[0.95], f"{name} ไม่มีค่าสูงเลย"
        assert part.min() <= whole[0.05], f"{name} ไม่มีค่าต่ำเลย"


def test_missing_labels_become_their_own_stratum():
    """ค่าว่างในคอลัมน์เฉลยต้องไม่ทำให้แถวหายไปเงียบๆ จาก `groupby`"""
    y = pd.Series([0, 1, None, 1, 0], name="label")
    assert strata_of(y, kind="classification").tolist() == ["0", "1", "<ว่าง>", "1", "0"]


def test_thin_strata_reports_classes_that_will_be_too_small():
    """**สิ่งที่ผู้สอนต้องเห็นก่อนกดสร้าง** — คลาสที่จะเหลือไม่กี่แถวตอนตัดสิน"""
    y = pd.Series([0] * 980 + [1] * 20, name="label")
    thin = thin_strata(y, kind="classification", student_ratio=0.7,
                       grading_public_ratio=0.5)

    assert "1" in thin, "คลาสที่มี 20 แถวจะเหลือ 3 แถวต่อกอง ต้องเตือน"
    assert "0" not in thin
    assert thin["1"]["test_private"] == 3


# ── ทำซ้ำได้ ──────────────────────────────────────────────────────────────


def test_the_same_seed_gives_the_same_split_bit_for_bit():
    data = _frame(2000)
    a, b = _split(data, seed=42), _split(data, seed=42)
    for name in PARTS:
        assert fingerprint(as_frame(getattr(a, name))) == fingerprint(as_frame(getattr(b, name)))


def test_a_different_seed_moves_the_rows():
    data = _frame(2000)
    a, b = _split(data, seed=1), _split(data, seed=2)
    assert set(a.test_private.X["row"]) != set(b.test_private.X["row"])


def test_row_order_in_the_file_does_not_change_the_split():
    """ไฟล์ที่เรียงใหม่แต่เนื้อเหมือนเดิม **ต้อง**ให้การแบ่งคนละแบบ

    ฟังดูย้อนแย้ง แต่ถูกต้อง — การแบ่งอ้างตำแหน่งแถว ไม่ใช่เนื้อแถว · สิ่งที่
    ตรึงความหมายของ "ไฟล์นี้" คือ `digest` ของเนื้อไฟล์ ซึ่งเปลี่ยนไปแล้วเมื่อ
    เรียงใหม่ → `config_hash` เปลี่ยน → แพลตฟอร์มปฏิเสธการเทียบคะแนนเก่ากับใหม่
    ข้อนี้จึงยืนยันว่าเราไม่ได้เผลอ "แก้" ให้มันทนต่อการเรียงใหม่ ซึ่งจะกลบสัญญาณ
    """
    data = _frame(1000)
    flipped = Dataset(X=data.X.iloc[::-1].reset_index(drop=True),
                      y=data.y.iloc[::-1].reset_index(drop=True))
    assert set(_split(data).test_private.X["row"]) != set(_split(flipped).test_private.X["row"])


def test_rows_are_shuffled_across_classes_inside_each_part():
    """แต่ละกองต้องไม่เรียงเป็นก้อนตามคลาส — ไม่งั้น `head()` จะได้คลาสเดียวล้วน"""
    part = _split(_frame(2000, positives=0.5)).student.y
    assert 0.3 < part.head(100).mean() < 0.7


# ── 🔒 เส้นแบ่งความไว้ใจ ──────────────────────────────────────────────────


def test_the_student_part_shares_no_row_with_the_grading_parts():
    """**ข้อที่พังแล้วการแข่งจบ** — สิ่งที่นิสิตได้ต้องไม่ทับกับสิ่งที่ใช้ตัดสินเลย

    เคยพังมาแล้วในรูปแบบอื่น: ทั้งห้าส่วนเคยมาจาก dataset ชุดเดียวที่สร้างจาก
    เมล็ดในไฟล์ที่แจกนิสิต ทำให้คำนวณเฉลยของชุดตัดสินได้ครบทุกแถว (macro-F1
    = 1.0000) · โครงเปลี่ยนไปแล้วแต่คุณสมบัติที่ต้องจริงยังเป็นข้อเดิม
    """
    split = _split(_frame(3000))

    student = set(split.student.X["row"])
    for name in GRADING_PARTS:
        grading = set(getattr(split, name).X["row"])
        assert not (student & grading), f"{name} มีแถวที่นิสิตได้ไปแล้ว {len(student & grading)} แถว"

    assert not (set(split.test_public.X["row"]) & set(split.test_private.X["row"])), (
        "กองที่โชว์บนกระดานทับกับกองที่ตัดสินรอบสุดท้าย — ทีมที่จูนเข้าหากระดาน "
        "จะได้เปรียบตอนตัดสินโดยอัตโนมัติ"
    )


def test_as_frame_carries_the_answer_and_is_only_used_for_the_student_part():
    """`as_frame` รวมเฉลยเข้าไปด้วย — ยืนยันว่ามันทำอย่างนั้นจริง

    จุดประสงค์คือทำให้เห็นชัดว่าฟังก์ชันนี้อันตรายถ้าเรียกกับกองที่ใช้ตัดสิน
    ผู้เรียกที่ถูกต้องมีที่เดียวคือ `dataset.student_csv`
    """
    split = _split(_frame(500))
    assert "label" in as_frame(split.student).columns
