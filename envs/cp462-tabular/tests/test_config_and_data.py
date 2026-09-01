"""config · คลังชุดข้อมูล · เส้นแบ่งความไว้ใจ

**สองอย่างที่พังแล้วการแข่งจบทันที**

  · **เฉลยหลุดไปกับไฟล์ที่แจก** — ถ้าไฟล์ที่นิสิตดาวน์โหลดมีแถวของชุดที่ใช้ตัดสิน
  · **คลังอ่านได้จากเครื่องนิสิต** — ถ้า `store.read` ทำงานได้โดยไม่มีคลังจริง

เคยพังมาแล้วในรูปแบบก่อนหน้า: ทั้งห้าส่วนมาจาก dataset ชุดเดียวที่สร้างจากเมล็ด
ในไฟล์ที่แจกนิสิต ทำให้คำนวณเฉลยเองได้ครบทุกแถว (macro-F1 = 1.0000) · โครง
เปลี่ยนไปแล้วแต่คำถามที่ต้องตอบให้ได้ทุกครั้งยังเป็นข้อเดิม
"""

from __future__ import annotations

import pytest

from tabular import store
from tabular.config import ConfigError, TaskSpec, from_mapping
from tabular.dataset import grading_data, parts, student_csv, to_dataset
from tabular.splits import as_frame
from tabular.store import DatasetError


def _spec(**kw) -> TaskSpec:
    base = dict(
        title="ทดสอบ", kind="classification", primary="macro_f1",
        dataset="sha256:" + "ab" * 32, target="churned",
        split_seed=1, bootstrap_seed=2, labels=[0, 1],
    )
    base.update(kw)
    return TaskSpec(**base)


# ── คลังชุดข้อมูล ─────────────────────────────────────────────────────────


def test_the_same_file_uploaded_twice_gets_the_same_id(sample_csv, datasets_root):
    """อ้างด้วยลายนิ้วมือของเนื้อไฟล์ — อัปโหลดซ้ำจึงไม่เกิดของซ้ำในคลัง"""
    first, second = store.put(sample_csv["churn"]), store.put(sample_csv["churn"])
    assert first == second
    assert len(list(datasets_root.glob(f"{first.split(':')[1]}.csv"))) == 1


def test_changing_one_byte_changes_the_id(sample_csv):
    other = sample_csv["churn"].replace(b"churned", b"CHURNED", 1)
    assert store.digest_of(other) != store.digest_of(sample_csv["churn"])


@pytest.mark.parametrize("bad", [
    "churn", "sha256:zz", "../../etc/passwd", "sha256:" + "g" * 64,
    "sha256:" + "ab" * 32 + "/../../etc/passwd",
])
def test_a_malformed_id_never_reaches_the_filesystem(bad):
    """`dataset` แก้ได้ผ่านหน้าเว็บ — ค่าที่มี `../` ต้องไม่พาไปอ่านไฟล์นอกคลัง"""
    with pytest.raises(DatasetError, match="ผิดรูปแบบ"):
        store.path_of(bad)


def test_a_file_that_is_not_csv_is_refused():
    with pytest.raises(DatasetError, match="CSV"):
        store.inspect(b"\x00\x01\x02\xff\xfe not a csv at all")


def test_a_file_with_too_few_rows_is_refused():
    blob = b"a,b,y\n" + b"1,2,0\n" * 50
    with pytest.raises(DatasetError, match="น้อยกว่าขั้นต่ำ"):
        store.inspect(blob)


def test_duplicate_column_names_are_refused():
    blob = b"a,a,y\n" + b"1,2,0\n" * 400
    with pytest.raises(DatasetError, match="ซ้ำ"):
        store.inspect(blob)


def test_a_stray_index_column_is_named_and_refused():
    """ไฟล์ที่ export มาโดยติด index — บอกวิธีแก้ ไม่ใช่แค่บอกว่าผิด"""
    blob = b",a,y\n" + b"0,1,0\n" * 400
    with pytest.raises(DatasetError, match="index=False"):
        store.inspect(blob)


def test_the_profile_lists_the_values_of_columns_that_could_be_classes(sample_csv):
    """หน้าเว็บใช้ค่านี้เติม `labels` ให้ — ผู้สอนจึงไม่ต้องพิมพ์เอง"""
    _, profile = store.inspect(sample_csv["churn"])

    assert profile.column("churned").values == [0, 1]
    assert profile.column("plan").values == ["basic", "legacy", "premium", "standard"]
    # `account_id` ไม่ซ้ำเลย — ไม่ใช่คลาส และต้องไม่ถูกเสนอเป็นคลาส
    assert profile.column("account_id").values is None
    assert profile.column("monthly_spend").dtype == "numeric"


def test_the_profile_values_are_plain_python_types(sample_csv):
    """ค่าจาก numpy เขียนลง YAML ไม่ได้ — `labels` ที่เก็บไปจะอ่านกลับไม่ได้"""
    _, profile = store.inspect(sample_csv["churn"])
    for value in profile.column("churned").values:
        assert type(value) in (int, float, str, bool), type(value)


# ── TaskSpec ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("field, value, message", [
    ("kind", "clustering", "kind ต้องเป็น"),
    ("primary", "r2", "ใช้คะแนนหลัก"),
    ("dataset", "churn", "ลายนิ้วมือ"),
    ("target", "", "คอลัมน์ไหนเป็นเฉลย"),
    ("labels", [], "ต้องประกาศ `labels`"),
    ("student_ratio", 0.05, "แจกนิสิต"),
    ("student_ratio", 0.98, "แจกนิสิต"),
    ("final_ratio", 0.0, "ตัดสินรอบสุดท้าย"),
    ("final_ratio", 0.9, "ตัดสินรอบสุดท้าย"),
])
def test_a_bad_field_is_refused_at_load_time(field, value, message):
    """ทุกข้อต้องล้มตอนโหลด ไม่ใช่ไปพังตอนนิสิตส่งงานเข้ามาแล้ว"""
    with pytest.raises(ConfigError, match=message):
        _spec(**{field: value})


def test_the_two_ratios_must_leave_room_for_the_leaderboard():
    """**สองค่านี้วัดจากทั้งไฟล์** รวมกันต้องน้อยกว่า 100% ไม่งั้นกระดานไม่มีข้อมูล

    ข้อความต้องบอกทั้งสองค่าที่กรอกมาและผลรวม — ผู้สอนที่เห็นแค่ "ค่าไม่ถูกต้อง"
    จะไม่รู้ว่าต้องลดตัวไหน
    """
    with pytest.raises(ConfigError, match="ไม่เหลือข้อมูลให้ leaderboard"):
        _spec(student_ratio=0.9, final_ratio=0.5)


def test_the_leaderboard_share_is_derived_not_typed():
    """กองที่สามเป็นผลลบ — ไม่มีช่องให้กรอก จึงไม่มีอะไรให้กรอกผิด"""
    assert _spec(student_ratio=0.7, final_ratio=0.15).leaderboard_ratio == pytest.approx(0.15)
    assert _spec(student_ratio=0.6, final_ratio=0.1).leaderboard_ratio == pytest.approx(0.3)


def test_the_target_cannot_also_be_dropped():
    with pytest.raises(ConfigError, match="อย่างใดอย่างหนึ่ง"):
        _spec(drop=["churned", "account_id"])


def test_regression_must_not_declare_classes():
    with pytest.raises(ConfigError, match="ต้องไม่มี"):
        _spec(kind="regression", primary="r2", labels=[0, 1])


def test_a_config_from_the_old_shape_explains_what_changed():
    """โจทย์ที่สร้างก่อนเดือน ส.ค. 2026 มีฟิลด์ที่ไม่มีแล้ว — ต้องบอกว่าทำไม"""
    with pytest.raises(ConfigError, match="นิสิตแบ่ง train/val/test เอง"):
        from_mapping({
            "slug": "churn", "task": "churn", "title": "x", "kind": "classification",
            "primary": "macro_f1", "n_rows": 12000, "data_seed": 1, "split_seed": 2,
            "bootstrap_seed": 3, "ratios": [0.6, 0.15, 0.25], "grading_rows": 3000,
        })


# ── config_hash คือสัญญา ──────────────────────────────────────────────────


@pytest.mark.parametrize("field, value", [
    ("dataset", "sha256:" + "cd" * 32),
    ("target", "region"),
    ("drop", ["account_id"]),
    ("split_seed", 999),
    ("bootstrap_seed", 999),
    ("student_ratio", 0.6),
    ("final_ratio", 0.3),
    ("labels", [1, 0]),
    ("kind", "regression"),
])
def test_anything_that_changes_the_score_changes_the_hash(field, value):
    base = _spec()
    if field == "kind":
        changed = _spec(kind="regression", primary="r2", labels=[])
    else:
        changed = _spec(**{field: value})
    assert base.config_hash != changed.config_hash, f"{field} ไม่ได้เข้า hash"


def test_renaming_the_task_does_not_break_old_scores():
    """`title` เป็นข้อความให้คนอ่าน — แก้แล้วคะแนนเก่าต้องยังเทียบได้"""
    assert _spec().config_hash == _spec(title="ชื่อใหม่").config_hash


def test_swapping_the_data_file_can_never_be_silent():
    """**เหตุผลที่อ้างไฟล์ด้วยลายนิ้วมือ ไม่ใช่ชื่อ**

    ถ้าอ้างด้วยชื่อ การเปลี่ยนไฟล์ใต้ชื่อเดิมกลางเทอมจะไม่ขยับอะไรเลย และ
    leaderboard จะยังดูเหมือนเทียบกันได้ทั้งที่คะแนนก่อนกับหลังมาจากคนละข้อมูล
    """
    before = _spec(dataset="sha256:" + "11" * 32)
    after = _spec(dataset="sha256:" + "22" * 32)
    assert before.config_hash != after.config_hash


# ── ข้อมูลที่ออกจากเซิร์ฟเวอร์ ────────────────────────────────────────────


def test_the_answer_column_never_reaches_the_sandbox(any_spec):
    for kind in ("public", "private"):
        assert any_spec.target not in grading_data(any_spec, kind).X.columns


def test_dropped_columns_never_reach_the_sandbox(churn_spec):
    """`account_id` ถูกตัด — ถ้ามันไปถึงกล่อง โมเดลจะจำเฉลยด้วย id ได้"""
    assert "account_id" not in grading_data(churn_spec, "private").X.columns
    assert "account_id" not in student_csv(churn_spec).decode().splitlines()[0]


def test_the_downloaded_file_holds_only_the_student_part(churn_spec):
    """**ข้อที่พังแล้วการแข่งจบ** — ไฟล์ที่แจกต้องไม่มีแถวของชุดที่ใช้ตัดสินเลย"""
    split = parts(churn_spec)
    handed_out = set(map(tuple, as_frame(split.student).itertuples(index=False)))

    for name in ("test_public", "test_private"):
        secret = set(map(tuple, as_frame(getattr(split, name)).itertuples(index=False)))
        assert not (handed_out & secret), (
            f"{len(handed_out & secret)} แถวของ {name} อยู่ในไฟล์ที่แจกนิสิต"
        )


def test_the_downloaded_file_has_the_answer_for_its_own_rows(churn_spec):
    """นิสิตต้องเทรนได้ — ไฟล์ที่แจกมีเฉลยของกองตัวเอง"""
    header = student_csv(churn_spec).decode().splitlines()[0]
    assert "churned" in header.split(",")


def test_a_missing_answer_in_the_source_file_is_refused():
    """แถวที่ไม่มีเฉลยให้คะแนนไม่ได้ — ต้องบอกตอนอัปโหลด ไม่ใช่ตอนตัดสิน"""
    import pandas as pd

    frame = pd.DataFrame({"a": range(400), "y": [1, 0, None] * 133 + [1]})
    with pytest.raises(DatasetError, match="ค่าว่าง"):
        to_dataset(frame, target="y", drop=[])


def test_dropping_every_feature_is_refused():
    import pandas as pd

    frame = pd.DataFrame({"id": range(400), "y": [0, 1] * 200})
    with pytest.raises(DatasetError, match="ฟีเจอร์"):
        to_dataset(frame, target="y", drop=["id"])


# ── 🔒 เครื่องนิสิตต้องเข้าไม่ถึงคลัง ─────────────────────────────────────


def test_without_the_store_nothing_can_read_the_grading_answers(churn_spec, monkeypatch):
    """เครื่องนิสิตไม่มี `ARENA_DATASETS` — ต้องล้มพร้อมบอกว่านี่คือความตั้งใจ

    นี่คือข้อที่แทนที่ข้อเดิมเรื่องเมล็ดลับ · เดิมความลับคือตัวเลขที่ใช้สร้าง
    ข้อมูล ตอนนี้คือตัวข้อมูลเอง — ซึ่งไม่มีทางย้อนกลับไปได้จากอะไรที่นิสิตมี
    """
    monkeypatch.delenv(store.DATASETS_ENV, raising=False)

    with pytest.raises(DatasetError, match="ถ้าคุณเป็นนิสิต"):
        grading_data(churn_spec, "private")
    with pytest.raises(DatasetError, match="ถ้าคุณเป็นนิสิต"):
        student_csv(churn_spec)


def test_a_dataset_id_that_is_not_in_the_store_says_which_machine_to_look_at(churn_spec):
    """worker ที่ชี้คลังคนละที่กับ API — ข้อความต้องพาไปหาสาเหตุนั้น"""
    with pytest.raises(DatasetError, match="ARENA_DATASETS"):
        grading_data(churn_spec.replace(dataset="sha256:" + "ff" * 32), "private")


def test_the_package_ships_no_dataset_and_no_seed():
    """แพ็กเกจที่นิสิตติดตั้งต้องไม่มีข้อมูลหรือของลับติดไปด้วย"""
    from pathlib import Path

    import tabular

    root = Path(tabular.__file__).resolve().parent
    assert not (root / "secrets.py").exists(), "โมดูลเมล็ดลับต้องถูกลบไปแล้ว"
    assert not (root / "configs").exists(), "config ของโจทย์อยู่ในฐานข้อมูล ไม่ใช่ในแพ็กเกจ"
    assert not list(root.rglob("*.csv")), "ไม่มีไฟล์ข้อมูลใดถูกแพ็กไปกับแพ็กเกจ"
