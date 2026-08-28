"""การแบ่งข้อมูล — ต้องทำซ้ำได้ ไม่ทับกัน และไม่ทำให้ชุดที่ใช้ตัดสินหลุด"""

from __future__ import annotations

import pytest

from tabular.generator import make
from tabular.splits import GRADING_PARTS, PARTS, as_frame, grading_split, split

RATIOS = (0.60, 0.15, 0.25)


@pytest.fixture
def churn():
    return make("churn", seed=1, n=5000)


def test_same_seed_gives_the_same_split(churn):
    a = split(churn, seed=9, ratios=RATIOS)
    b = split(churn, seed=9, ratios=RATIOS)
    for name in PARTS:
        assert getattr(a, name).X.equals(getattr(b, name).X)
        assert getattr(a, name).y.equals(getattr(b, name).y)


def test_different_seed_gives_a_different_split(churn):
    a = split(churn, seed=9, ratios=RATIOS)
    b = split(churn, seed=10, ratios=RATIOS)
    assert not a.train.X.equals(b.train.X)


def test_every_row_lands_in_exactly_one_part(churn):
    """ไม่ทับกันและไม่หาย — ถ้าทับ แถวของ test จะไปโผล่ใน train ที่นิสิตได้"""
    parts = split(churn, seed=9, ratios=RATIOS)
    ids = [set(getattr(parts, name).X["account_id"]) for name in PARTS]

    assert sum(len(s) for s in ids) == len(churn), "มีแถวหายหรือซ้ำ"
    assert set().union(*ids) == set(churn.X["account_id"])
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            assert not (left & right), "มีแถวอยู่สองส่วนพร้อมกัน"


def test_sizes_match_the_ratios_and_sum_exactly(churn):
    parts = split(churn, seed=9, ratios=RATIOS)
    sizes = parts.sizes()
    assert sum(sizes.values()) == len(churn), "ผลรวมต้องเท่ากับข้อมูลทั้งหมดเป๊ะ"
    for name, ratio in zip(PARTS, RATIOS):
        assert abs(sizes[name] / len(churn) - ratio) < 0.01


def test_class_balance_survives_a_plain_shuffle(churn):
    """ไม่ stratify โดยตั้งใจ — เทสต์นี้ยืนยันว่าขนาดชุดใหญ่พอที่จะไม่ต้อง

    ถ้าวันหนึ่งชุดข้อมูลเล็กลงมากจนข้อนี้แดง แปลว่าถึงเวลาต้อง stratify จริงๆ
    ไม่ใช่แก้เกณฑ์ให้ผ่าน
    """
    parts = split(churn, seed=9, ratios=RATIOS)
    overall = churn.y.mean()
    for name in PARTS:
        rate = getattr(parts, name).y.mean()
        assert abs(rate - overall) < 0.05, f"{name}: สัดส่วนคลาสเพี้ยนไป {rate - overall:+.3f}"


def test_the_student_split_cannot_hold_a_grading_set(churn):
    """**ด่านสำคัญ** — `Split` ต้องไม่มีที่ให้ชุดที่ใช้ตัดสินอยู่เลย

    เดิมคลาสนี้ถือ `test_public`/`test_private` ไว้ด้วย แล้วกันการหลุดด้วย
    `open_parts()` ที่เลือกคืนแค่บางฟิลด์ · การกันแบบนั้นพึ่งวินัยของคนเขียน
    ตอนนี้ชุดที่ใช้ตัดสินอยู่คนละคลาสและมาจากคนละ dataset — เผลอส่งออกไปไม่ได้
    เพราะมันไม่ได้อยู่ในนี้ตั้งแต่แรก
    """
    parts = split(churn, seed=9, ratios=RATIOS)
    assert set(parts.open_parts()) == {"train", "val", "test"}
    for forbidden in GRADING_PARTS:
        assert not hasattr(parts, forbidden), f"Split ไม่ควรมีฟิลด์ {forbidden}"


def test_grading_split_cuts_public_and_private(churn):
    graded = grading_split(churn, seed=9, public_ratio=0.4)
    assert graded.sizes() == {"test_public": 2000, "test_private": 3000}
    assert not (set(graded.test_public.X["account_id"])
                & set(graded.test_private.X["account_id"]))


def test_index_is_reset_so_students_get_predictable_rows(churn):
    parts = split(churn, seed=9, ratios=RATIOS)
    for name in PARTS:
        part = getattr(parts, name)
        assert list(part.X.index) == list(range(len(part)))
        assert list(part.y.index) == list(range(len(part)))


def test_as_frame_puts_the_target_last(churn):
    parts = split(churn, seed=9, ratios=RATIOS)
    frame = as_frame(parts.train)
    assert list(frame.columns)[:-1] == list(parts.train.X.columns)
    assert list(frame.columns)[-1] == parts.train.y.name


@pytest.mark.parametrize(
    "ratios,reason",
    [
        ((0.6, 0.4), "จำนวนส่วนไม่ครบ"),
        ((0.6, 0.2, 0.1), "รวมกันไม่ได้ 1.0"),
        ((0.7, 0.3, 0.0), "มีส่วนที่เป็นศูนย์"),
    ],
)
def test_bad_ratios_are_rejected(churn, ratios, reason):
    with pytest.raises(ValueError):
        split(churn, seed=1, ratios=ratios)


def test_too_little_data_says_what_it_would_produce():
    """ข้อความต้องบอกว่าจะได้อะไร ไม่ใช่แค่ว่าผิด — §13 ของ template"""
    tiny = make("churn", seed=1, n=8)
    with pytest.raises(ValueError, match="test"):
        split(tiny, seed=1, ratios=(0.96, 0.02, 0.02))
