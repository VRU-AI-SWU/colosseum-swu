"""ข้อมูลต้องเหมือนกันทุกบิตบนทุกเครื่อง และโจทย์ต้องยากพอให้แข่งกันได้

สองเรื่องนี้คือสิ่งที่ทำให้การแข่งมีความหมาย

  · **ทำซ้ำได้** — นิสิตเทรนบนข้อมูลชุดเดียวกับที่ grader ใช้แบ่ง ถ้าต่างกัน
    คะแนนที่วัดเองเทียบกับ leaderboard ไม่ได้ และไม่มีใครรู้ว่าทำไม
  · **มีบันได** — ถ้าทายเดาก็ได้คะแนนใกล้เคียงคนที่ตั้งใจทำ ก็ไม่มีอะไรให้แข่ง
    เป็นบทเรียนเดียวกับที่ cp463-vacuum ต้อง calibrate baseline ladder
"""

from __future__ import annotations

import numpy as np
import pytest

from tabular.generator import MISSING_RATE, PLANS, TASKS, fingerprint, make
from tabular.table import Dataset

@pytest.mark.parametrize("task", sorted(TASKS))
def test_same_seed_gives_identical_data(task):
    a, b = make(task, seed=7, n=500), make(task, seed=7, n=500)
    assert fingerprint(a) == fingerprint(b)
    assert a.X.equals(b.X)
    assert a.y.equals(b.y)


@pytest.mark.parametrize("task", sorted(TASKS))
def test_different_seeds_give_different_data(task):
    assert fingerprint(make(task, seed=1, n=500)) != fingerprint(make(task, seed=2, n=500))


@pytest.mark.parametrize("task", sorted(TASKS))
def test_global_rng_cannot_change_the_data(task):
    """นิสิตที่เรียก `np.random.seed()` ในโค้ดตัวเองต้องไม่ทำให้ข้อมูลเปลี่ยน

    บทเรียนตรงกับ cp463-vacuum — ตัวสร้างต้องใช้ Generator ของตัวเองเท่านั้น
    """
    before = fingerprint(make(task, seed=7, n=500))
    np.random.seed(999)
    _ = np.random.random(1000)
    assert fingerprint(make(task, seed=7, n=500)) == before


@pytest.mark.parametrize("task", sorted(TASKS))
def test_shape_and_columns_are_stable(task):
    d = make(task, seed=3, n=400)
    assert len(d) == 400 == len(d.y)
    assert list(d.X.columns) == [
        "account_id", "tenure_months", "monthly_spend",
        "support_tickets", "plan", "region",
    ]
    assert "target" not in d.X.columns, "เป้าหมายต้องไม่ปนอยู่ใน X"
    assert d.y.name in {"churned", "monthly_value"}


@pytest.mark.parametrize("task", sorted(TASKS))
def test_missing_values_appear_where_intended(task):
    """ค่าว่างเป็นส่วนหนึ่งของโจทย์ ไม่ใช่ความบังเอิญ — ถ้าหายไปแปลว่าโจทย์ง่ายลง"""
    d = make(task, seed=5, n=4000)
    for column, rate in MISSING_RATE.items():
        seen = d.X[column].isna().mean()
        assert 0.5 * rate < seen < 2.0 * rate, f"{column}: ว่าง {seen:.3f} ควรราว {rate}"
    assert not d.y.isna().any(), "เป้าหมายต้องไม่มีค่าว่าง"
    assert not d.X["account_id"].isna().any()


@pytest.mark.parametrize("task", sorted(TASKS))
def test_the_rare_category_stays_rare_but_present(task):
    """หมวดที่พบน้อยคือสิ่งที่บังคับให้จัดการหมวดที่ไม่เคยเห็น — ต้องมีแต่ต้องน้อย"""
    counts = make(task, seed=5, n=4000).X["plan"].value_counts()
    assert set(counts.index) == set(PLANS)
    assert 0.005 < counts["legacy"] / 4000 < 0.05


def test_missing_values_do_not_predict_the_target():
    """ค่าว่างต้องถูกเจาะ **หลัง** สร้างเป้าหมาย

    ถ้าเจาะก่อน ค่าว่างจะกลายเป็นสัญญาณที่ทำนายเป้าหมายได้ — นิสิตจะหาเจอแล้ว
    ได้คะแนนสูงเกินจริงจากสิ่งที่ไม่มีในข้อมูลจริง
    """
    d = make("churn", seed=11, n=8000)
    for column in MISSING_RATE:
        missing = d.X[column].isna()
        gap = abs(d.y[missing].mean() - d.y[~missing].mean())
        assert gap < 0.06, f"{column}: อัตราคลาสบวกต่างกัน {gap:.3f} เมื่อค่าว่าง/ไม่ว่าง"


def test_churn_is_imbalanced_enough_to_matter():
    """ไม่สมดุลคือส่วนหนึ่งของโจทย์ — ถ้าสมดุลเป๊ะ accuracy ก็ใช้ได้และบทเรียนหาย"""
    rate = make("churn", seed=1, n=6000).y.mean()
    assert 0.15 < rate < 0.35, f"สัดส่วนคลาสบวก {rate:.1%} ไม่อยู่ในช่วงที่ตั้งใจ"


def test_housing_target_is_right_skewed():
    """เบ้ขวาคือสิ่งที่ทำให้การแปลงเป้าหมายมีผล — เป็นบทเรียนที่อยากให้เจอ"""
    y = make("housing", seed=1, n=6000).y
    assert y.skew() > 0.25
    assert (y > 0).all(), "มูลค่าติดลบไม่มีความหมาย"


def test_unknown_task_says_what_exists():
    with pytest.raises(ValueError, match="ที่มีคือ"):
        make("clustering", seed=1, n=10)


def test_dataset_keeps_x_and_y_apart():
    """`y` ของชุดที่ใช้ตัดสินต้องไม่หลุดเข้า sandbox — โครงสร้างต้องบังคับตั้งแต่ต้น"""
    d = make("churn", seed=1, n=10)
    assert isinstance(d, Dataset)
    assert d.y.name not in d.X.columns
