"""ของกลางสำหรับเทสต์ของ env นี้

**เทสต์ต้องมีคลังชุดข้อมูลของตัวเอง** — เดิมที่นี่ตั้ง `ARENA_CP462_ALLOW_SEED_FALLBACK`
เพื่อยืมเมล็ดสำรองมาสร้างชุดที่ใช้ตัดสิน · ตอนนี้ไม่มีเมล็ดแล้ว ข้อมูลเป็นไฟล์
เทสต์จึงปั๊มไฟล์ตัวอย่างลงคลังชั่วคราวแล้วชี้ `ARENA_DATASETS` ไปที่นั่น

ผลพลอยได้ที่สำคัญ: **เทสต์เดินเส้นทางเดียวกับผู้สอนจริง** — อัปโหลดไฟล์ แล้ว
สร้าง spec ที่ชี้ไปหาไฟล์นั้น · เดิมเทสต์เดินเส้นทางพิเศษที่ผู้สอนไม่เคยใช้
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def datasets_root(tmp_path_factory):
    """คลังชุดข้อมูลชั่วคราว — session เดียวใช้ร่วมกันเพื่อไม่ต้องปั๊มไฟล์ซ้ำ"""
    return tmp_path_factory.mktemp("datasets")


@pytest.fixture(autouse=True)
def _point_at_the_store(monkeypatch, datasets_root):
    from tabular import store

    monkeypatch.setenv(store.DATASETS_ENV, str(datasets_root))


@pytest.fixture(scope="session")
def sample_csv():
    """ไฟล์ตัวอย่างของแต่ละชนิดโจทย์ — `{"churn": bytes, "housing": bytes}`"""
    from tabular.generator import sample_csv as make_csv

    return {task: make_csv(task, seed=20260101, n=2000) for task in ("churn", "housing")}


@pytest.fixture
def stored(sample_csv, _point_at_the_store):
    """เก็บไฟล์ตัวอย่างลงคลังแล้วคืน `{ชื่อโจทย์: digest}`"""
    from tabular import store

    return {task: store.put(blob) for task, blob in sample_csv.items()}


@pytest.fixture
def churn_spec(stored):
    """สเปคของโจทย์ classification ที่ชี้ไปหาไฟล์ในคลัง"""
    from tabular.config import TaskSpec

    return TaskSpec(
        title="Churn", kind="classification", primary="macro_f1",
        dataset=stored["churn"], target="churned",
        split_seed=7, bootstrap_seed=11, labels=[0, 1],
        drop=["account_id"],
    )


@pytest.fixture
def housing_spec(stored):
    """สเปคของโจทย์ regression ที่ชี้ไปหาไฟล์ในคลัง"""
    from tabular.config import TaskSpec

    return TaskSpec(
        title="Housing", kind="regression", primary="r2",
        dataset=stored["housing"], target="monthly_value",
        split_seed=7, bootstrap_seed=11, drop=["account_id"],
    )


@pytest.fixture(params=["churn_spec", "housing_spec"])
def any_spec(request):
    """ทั้งสองชนิดโจทย์ — ใช้กับข้อที่ต้องจริงกับทั้งคู่"""
    return request.getfixturevalue(request.param)
