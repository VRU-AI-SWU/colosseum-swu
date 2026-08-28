"""TaskSpec, การเข้าถึงข้อมูล และ starter kit ที่แจกจริง

สองเรื่องที่ผิดแล้วเจ็บที่สุด

  · **เฉลยหลุดเข้า sandbox** — ถ้า `open_data` คืนชุดที่ใช้ตัดสินมาด้วย
    ทั้งการแข่งจบทันทีโดยไม่มีใครรู้
  · **`config_hash` ไม่นิ่ง** — hash ที่เปลี่ยนเองแปลว่าคะแนนเก่าเทียบไม่ได้
    ทั้งที่ไม่มีใครแก้อะไร
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tabular.config import CONFIG_DIR, ConfigError, TaskSpec, load, load_config
from tabular.dataset import all_parts, features_only, grading_data, open_data
from tabular.secrets import FALLBACK_SEED, GradingSeedUnavailable

SLUGS = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
STARTER = Path(__file__).resolve().parent.parent / "tabular" / "starter"


def base(**kw) -> TaskSpec:
    args = dict(
        slug="t", task="churn", title="ชื่อ", kind="classification", primary="macro_f1",
        n_rows=1000, data_seed=1, split_seed=2, bootstrap_seed=3,
        ratios=(0.6, 0.15, 0.25), labels=[0, 1],
        grading_rows=500, grading_public_ratio=0.4,
    )
    args.update(kw)
    return TaskSpec(**args)


# ── config ที่แจกจริง ───────────────────────────────────────────────


@pytest.mark.parametrize("slug", SLUGS)
def test_shipped_configs_load(slug):
    spec = load(slug)
    assert spec.slug == slug
    assert spec.config_hash.startswith("sha256:")


def test_both_kinds_are_covered():
    """โจทย์ที่แจกต้องครอบทั้งสองชนิด — ไม่งั้นทางของ regression ไม่เคยถูกรัน"""
    assert {load(s).kind for s in SLUGS} == {"classification", "regression"}


# ── config_hash คือสัญญา ───────────────────────────────────────────


def test_hash_is_stable_across_calls():
    assert base().config_hash == base().config_hash


def test_title_does_not_change_the_hash():
    """ชื่อเป็นข้อความให้คนอ่าน — แก้ได้โดยไม่ทำให้คะแนนเก่าเทียบไม่ได้"""
    assert base().config_hash == base(title="ชื่อใหม่").config_hash


@pytest.mark.parametrize(
    "field,value",
    [
        ("ratios", (0.5, 0.2, 0.3)),
        ("grading_rows", 800),
        ("grading_public_ratio", 0.5),
        ("data_seed", 999),
        ("split_seed", 999),
        ("bootstrap_seed", 999),
        ("n_rows", 2000),
        ("primary", "accuracy"),
        ("labels", [1, 0]),          # ลำดับคลาสมีผลต่อ confusion matrix
    ],
)
def test_anything_that_affects_scoring_changes_the_hash(field, value):
    assert base().config_hash != base(**{field: value}).config_hash


# ── config ที่ผิดต้องล้มตั้งแต่โหลด ────────────────────────────────


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"kind": "clustering"}, "kind"),
        ({"primary": "roc_auc"}, "ที่ใช้ได้คือ"),
        ({"kind": "regression", "primary": "macro_f1"}, "ที่ใช้ได้คือ"),
        ({"labels": []}, "labels"),
        ({"kind": "regression", "primary": "r2", "labels": [0, 1]}, "ต้องไม่มี"),
        ({"ratios": (0.6, 0.4)}, "3 ค่า"),
        ({"ratios": (0.6, 0.2, 0.1)}, "1.0"),
        ({"n_rows": 50}, "น้อยเกินไป"),
        ({"grading_rows": 50}, "น้อยเกินไป"),
        ({"grading_public_ratio": 1.0}, "ระหว่าง 0 กับ 1"),
    ],
)
def test_bad_config_fails_at_load_not_at_scoring(kw, match):
    with pytest.raises(ConfigError, match=match):
        base(**kw)


def test_unknown_field_in_yaml_is_rejected(tmp_path):
    """พิมพ์ชื่อฟิลด์ผิดต้องรู้ทันที ไม่ใช่ถูกเมินแล้วใช้ค่าเริ่มต้นเงียบๆ"""
    path = tmp_path / "bad.yaml"
    path.write_text("slug: x\ntyop: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="typo|tyop"):
        load_config(path)


def test_unknown_slug_lists_what_exists():
    with pytest.raises(ConfigError, match="ที่มีคือ"):
        load("ไม่มีจริง")


def test_replace_rejects_unknown_fields():
    with pytest.raises(ConfigError, match="ไม่รู้จักฟิลด์"):
        base().replace(nope=1)


# ── เฉลยต้องไม่หลุด ────────────────────────────────────────────────


@pytest.mark.parametrize("slug", SLUGS)
def test_open_data_gives_the_three_student_parts(slug):
    assert set(open_data(load(slug))) == {"train", "val", "test"}


# ── ชุดที่ใช้ตัดสินต้องเข้าถึงไม่ได้จากฝั่งนิสิต ────────────────────


@pytest.mark.parametrize("slug", SLUGS)
def test_students_cannot_compute_the_grading_answers(slug):
    """**ด่านสำคัญที่สุดของทั้งวิชา**

    เดิมทุกส่วนมาจาก dataset ชุดเดียวที่สร้างจากเมล็ดในไฟล์ config ที่แจก แปลว่า
    นิสิตรัน `grading_data(load('churn'), 'private')` แล้วได้เฉลยครบทุกแถว
    วัดแล้วทำ macro-F1 ได้ 1.0000 ด้วยการจำเฉลยแล้วจับคู่ด้วย `account_id`

    ตอนนี้เมล็ดของชุดที่ใช้ตัดสินไม่ได้อยู่ในไฟล์ config และไม่ได้อยู่ในแพ็กเกจ
    `load(slug)` จึงคืนสเปคที่ไม่มีมัน และ `grading_data` ต้องปฏิเสธ
    """
    spec = load(slug)
    assert spec.grading_seed is None, "สเปคที่โหลดจากไฟล์ที่แจกต้องไม่มีเมล็ดลับ"
    for kind in ("public", "private"):
        with pytest.raises(GradingSeedUnavailable):
            grading_data(spec, kind)


@pytest.mark.parametrize("slug", SLUGS)
def test_the_secret_seed_is_not_in_the_shipped_config(slug):
    """อ่านไฟล์ตรงๆ ด้วย — เผื่อวันหนึ่งมีคนใส่กลับเข้าไปแล้ว loader เมินมันเงียบๆ"""
    text = (CONFIG_DIR / f"{slug}.yaml").read_text(encoding="utf-8")
    for line in text.splitlines():
        assert not line.strip().startswith("grading_seed:"), (
            f"configs/{slug}.yaml มี grading_seed — นั่นคือการแจกเฉลยให้นิสิต"
        )


@pytest.mark.parametrize("slug", SLUGS)
def test_grading_sets_do_not_overlap_what_students_get(slug):
    spec = load(slug).replace(grading_seed=FALLBACK_SEED)
    student_ids = set()
    for part in all_parts(spec).open_parts().values():
        student_ids |= set(part.X["account_id"])

    for kind in ("public", "private"):
        graded = set(grading_data(spec, kind).X["account_id"])
        assert not (graded & student_ids), f"{kind}: มีแถวซ้ำกับที่นิสิตได้รับ"


def test_a_different_grading_seed_gives_different_answers():
    """ชุดที่ใช้ตัดสินต้องขึ้นกับเมล็ดลับจริงๆ — ไม่ใช่มีฟิลด์ไว้เฉยๆ"""
    spec = load(SLUGS[0])
    a = grading_data(spec.replace(grading_seed=FALLBACK_SEED), "private")
    b = grading_data(spec.replace(grading_seed=FALLBACK_SEED + 1), "private")
    assert not a.X.equals(b.X)


def test_grading_data_rejects_a_bad_kind():
    spec = load(SLUGS[0]).replace(grading_seed=FALLBACK_SEED)
    with pytest.raises(ValueError, match="public"):
        grading_data(spec, "ไม่มีจริง")


@pytest.mark.parametrize("slug", SLUGS)
def test_features_only_drops_the_answer(slug):
    spec = load(slug).replace(grading_seed=FALLBACK_SEED)
    test = grading_data(spec, "public")
    X = features_only(test)
    assert test.y.name not in X.columns
    assert len(X) == len(test)


@pytest.mark.parametrize("slug", SLUGS)
def test_the_same_spec_always_gives_the_same_split(slug):
    spec = load(slug)
    assert all_parts(spec).train.X.equals(all_parts(spec).train.X)


# ── starter kit ที่แจกจริงต้องใช้งานได้ ────────────────────────────


def test_starter_predictor_satisfies_the_contract(tmp_path):
    """เทรนด้วย `train.py` ที่แจก แล้วโหลดด้วย `predictor.py` ที่แจก

    ถ้าสองไฟล์นี้ไม่เข้ากัน นิสิตทุกคนจะติดตั้งแต่ก้าวแรกโดยไม่ใช่ความผิดของเขา
    """
    for name in ("predictor.py", "train.py"):
        (tmp_path / name).write_text((STARTER / name).read_text(encoding="utf-8"),
                                     encoding="utf-8")

    run = subprocess.run(
        [sys.executable, "train.py", "--task", "housing"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "pipeline.pkl").is_file(), "train.py ต้องบันทึก pipeline.pkl"
    assert "r2 บน val" in run.stdout

    check = subprocess.run(
        [sys.executable, "-c",
         "from predictor import Predictor\n"
         "from tabular.config import load\n"
         "from tabular.dataset import open_data\n"
         "spec = load('housing')\n"
         "test = open_data(spec)['test']\n"
         "y = Predictor({}).predict(test.X)\n"
         "assert len(y) == len(test), (len(y), len(test))\n"
         "print('OK', len(y))"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert check.returncode == 0, check.stderr
    assert "OK 3000" in check.stdout


def test_starter_predictor_does_not_import_the_answers():
    """`predictor.py` รันใน sandbox — ห้ามแตะโมดูลที่เห็นเฉลย"""
    source = (STARTER / "predictor.py").read_text(encoding="utf-8")
    for forbidden in ("tabular.dataset", "tabular.metrics", "grading_data"):
        assert forbidden not in source, f"predictor.py อ้างถึง {forbidden}"
