"""ปฏิทินของ competition — เขตของวันและสัญญาเรื่อง config_hash

เครื่องมือ `tools/setup_competition.py` เขียนทับ record ที่ใช้ตัดสินคะแนนจริง
สองอย่างที่ผิดแล้วเจ็บและมองไม่เห็นตอนรีวิว

  · **เขตของวัน** — `2026-09-30` ต้องหมายถึงถึงสิ้นวันนั้นตามเวลาไทย
    ถ้าเผลอใช้ UTC ตรงๆ นิสิตจะเสียวันสุดท้ายไป 17 ชั่วโมง และจะรู้ตอนที่สายไปแล้ว

  · **config_hash** — `config_override` ที่ประกอบแล้วต้องให้ hash ตรงกับตอน
    generate seed ไม่งั้น worker โยน `ConfigDrift` ตอนให้คะแนน ซึ่งเป็นจังหวะ
    ที่แย่ที่สุดที่จะรู้ · เครื่องมือตรวจก่อนเขียน เทสต์นี้ตรวจว่ามันตรวจจริง
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from runners.seeds import expected_config_hash  # noqa: E402
from tools.setup_competition import TASK_TYPES, ICT, PHASES, parse_range  # noqa: E402

VACUUM = TASK_TYPES["agent_env"]
sys.path.insert(0, str(VACUUM.env_root))

BASE = REPO / "envs" / "cp463-vacuum" / "vacuum" / "configs" / "main.yaml"
SLUG = "cp463-vacuum-1-2026"


def config_override_for(phase, base):
    return VACUUM.overrides(phase, base)


def verify(base, overrides, slug):
    return VACUUM.verify(base, overrides, slug)

#: `verify()` ต้องอ่าน config_hash ที่ตรึงไว้ใน seeds.yaml ซึ่งอยู่ใน repo ลับ
#: เทสต์ที่เหลือไม่แตะของลับเลย ตามธรรมเนียมของ core/tests ที่ตัด ARENA_SECRETS ทิ้ง
needs_secrets = pytest.mark.skipif(
    expected_config_hash(competition_slug=SLUG, phase="main") is None,
    reason="ต้องตั้ง ARENA_SECRETS ให้ชี้ไป clone ของ colosseum-hypogeum",
)


# ── เขตของวัน ──────────────────────────────────────────────────────


def test_range_covers_the_last_day_completely():
    """วันจบต้องรวมทั้งวัน — `Phase.contains` ใช้ `start <= when < end`"""
    start, end = parse_range("2026-09-15..2026-09-30")
    assert start == datetime(2026, 9, 15, 0, 0, tzinfo=ICT)
    assert end == datetime(2026, 10, 1, 0, 0, tzinfo=ICT)

    last_moment = datetime(2026, 9, 30, 23, 59, 59, tzinfo=ICT)
    assert start <= last_moment < end, "วันสุดท้ายของช่วงถูกตัดออกไป"


def test_range_is_thai_time_not_utc():
    """ถ้าเผลอใช้ UTC เที่ยงคืนไทยจะกลายเป็น 17:00 ของวันก่อนหน้า"""
    start, _ = parse_range("2026-09-15..2026-09-30")
    assert start.utcoffset() == timedelta(hours=7)
    # เที่ยงคืนไทย = 17:00 UTC ของวันก่อนหน้า
    assert start.astimezone(timezone.utc) == datetime(2026, 9, 14, 17, 0, tzinfo=timezone.utc)


def test_consecutive_ranges_leave_no_gap_and_no_overlap():
    """ช่วงที่ติดกันต้องไม่มีรูโหว่ ไม่งั้นงานที่ส่งคาบเกี่ยวจะหา phase ไม่เจอ"""
    _, warmup_end = parse_range("2026-09-15..2026-09-30")
    main_start, _ = parse_range("2026-10-01..2026-10-31")
    assert warmup_end == main_start


@pytest.mark.parametrize("bad", ["2026-09-15", "2026-09-30..2026-09-15", "15/09/2026..30/09/2026"])
def test_bad_ranges_are_rejected(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_range(bad)


# ── สัญญาเรื่อง config ─────────────────────────────────────────────


def test_override_reproduces_each_phase_config_exactly():
    """`main.yaml` + override ต้องได้ config ที่เหมือน YAML ของ phase นั้นทุกบิต

    override คำนวณจากไฟล์จริง ไม่ได้เขียนมือ — เทสต์นี้ยืนยันว่าการคำนวณนั้นถูก
    """
    from vacuum import load_config
    from vacuum.config import CONFIG_DIR

    base = load_config(BASE)
    for phase in PHASES:
        override = config_override_for(phase, BASE)
        got = base.replace(**override)
        want = load_config(CONFIG_DIR / f"{phase}.yaml")
        assert got.config_hash == want.config_hash, f"{phase}: ประกอบแล้วไม่ตรงกับ {phase}.yaml"


def test_main_needs_no_override():
    """competition ชี้ที่ main.yaml อยู่แล้ว — phase main จึงต้องไม่มี override

    ถ้ามีค่าโผล่มา แปลว่า base ที่ใช้ไม่ใช่ main.yaml จริง ซึ่งเป็นสัญญาณว่ามีอะไรผิด
    """
    assert config_override_for("main", BASE) == {}


def test_overrides_are_json_safe():
    """`config_override` ถูกเก็บลงฐานข้อมูลเป็น JSON — ค่าที่ serialize ไม่ได้จะพังตอนบันทึก"""
    import json

    for phase in PHASES:
        json.dumps(config_override_for(phase, BASE))


@needs_secrets
def test_verify_catches_a_config_that_drifted_from_the_seeds():
    """ถ้า override ผิด เครื่องมือต้องจับได้ **ก่อน** เขียนลงฐานข้อมูล

    เป็นด่านเดียวที่กันไม่ให้ ConfigDrift ไปโผล่ตอนให้คะแนนจริง
    """
    good = {phase: config_override_for(phase, BASE) for phase in PHASES}
    assert verify(BASE, good, SLUG) == []

    tampered = {phase: dict(o) for phase, o in good.items()}
    tampered["final"]["room.width"] = 999
    problems = verify(BASE, tampered, SLUG)
    assert any("final" in p and "config_hash" in p for p in problems), problems


# ── โจทย์ทำนาย (CP462) ─────────────────────────────────────────────


PREDICTION = TASK_TYPES["prediction"]

pytest.importorskip("tabular", reason="ต้องติดตั้ง envs/cp462-tabular ก่อน")

#: ชื่อโจทย์ที่ผู้สอนตั้ง — ใช้ยืนยันว่าชื่อเดินทางจาก config ไปถึง competition
CHURN_TITLE = "ทำนายการเลิกใช้บริการ"


@pytest.fixture(scope="module")
def churn(tmp_path_factory):
    """คลังชุดข้อมูล + ไฟล์ config ของโจทย์ churn — `(config_path, datasets_root)`

    เดิมชี้ไปที่ `tabular/configs/churn.yaml` ที่แพ็กไปกับแพ็กเกจ · ไฟล์นั้นไม่มี
    แล้ว เพราะ config อยู่ในฐานข้อมูลและชุดข้อมูลเป็นไฟล์ที่ผู้สอนอัปโหลดเข้าคลัง
    fixture นี้จึงจำลองสิ่งที่ผู้สอนทำจริง: อัปโหลดไฟล์ แล้วเขียน config ที่ชี้ไปหามัน
    """
    import os

    import yaml
    from tabular import store
    from tabular.arena import PLUGIN
    from tabular.generator import sample_csv

    root = tmp_path_factory.mktemp("datasets")
    previous = os.environ.get(store.DATASETS_ENV)
    os.environ[store.DATASETS_ENV] = str(root)

    digest = PLUGIN.save_dataset(sample_csv("churn", seed=20260101, n=4000))
    path = tmp_path_factory.mktemp("configs") / "churn.yaml"
    path.write_text(
        yaml.safe_dump({
            "title": CHURN_TITLE, "kind": "classification", "primary": "macro_f1",
            "dataset": digest, "target": "churned", "drop": ["account_id"],
            "labels": [0, 1], "split_seed": 7, "bootstrap_seed": 11,
        }, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )

    yield path, root

    if previous is None:
        os.environ.pop(store.DATASETS_ENV, None)
    else:
        os.environ[store.DATASETS_ENV] = previous


def test_prediction_has_no_per_phase_config_change(churn):
    """โจทย์เดียวตรึงทั้งเทอม — การเปลี่ยนสเปคกลางเทอมทำให้คะแนนเก่าเทียบไม่ได้

    ต่างจาก CP463 ที่ตั้งใจให้แต่ละ phase ยากขึ้น · ที่นี่สิ่งที่ต่างกันระหว่าง
    phase คือปฏิทินกับชุดที่ใช้ตัดสิน ซึ่งไม่ได้อยู่ใน config
    """
    for phase in PHASES:
        assert PREDICTION.overrides(phase, churn[0]) == {}


def test_prediction_whitelist_excludes_the_package_that_sees_the_answers():
    """`tabular` เห็นเฉลยและจงใจไม่อยู่ใน image — ใส่ใน whitelist = ปล่อยให้กินโควตาฟรี"""
    assert "tabular" not in PREDICTION.whitelist
    assert {"numpy", "pandas", "sklearn", "joblib"} <= PREDICTION.whitelist


def test_prediction_title_comes_from_the_config_file(churn):
    """ชื่อโจทย์มาจาก config ไม่ใช่พิมพ์ซ้ำใน CLI — พิมพ์ซ้ำแล้วจะไม่ตรงกันวันหนึ่ง"""
    assert PREDICTION.title(churn[0]) == CHURN_TITLE


def test_prediction_verify_refuses_when_the_data_file_is_not_on_this_machine(churn, monkeypatch):
    """**ด่านสำคัญ** — ไฟล์ไม่อยู่ในคลัง = worker จะล้มทุก run โดยที่นิสิตไม่ผิด

    เจตนาเดียวกับการตรวจ config_hash ของ CP463: อะไรที่จะทำให้การให้คะแนนล้ม
    ต้องรู้ตอนตั้งค่า ไม่ใช่ตอนนิสิตส่งงานแล้ว
    """
    from tabular import store

    monkeypatch.setenv(store.DATASETS_ENV, str(churn[1] / "ที่ไม่มีอยู่"))
    problems = PREDICTION.verify(churn[0], {p: {} for p in PHASES}, "cp462-churn-1-2026")
    assert problems, "ไฟล์ไม่อยู่ในคลังแล้วต้องปฏิเสธ"
    assert "ARENA_DATASETS" in problems[0]


def test_prediction_verify_passes_when_the_file_is_there(churn):
    assert PREDICTION.verify(churn[0], {p: {} for p in PHASES}, "cp462-churn-1-2026") == []


def test_prediction_verify_warns_about_a_class_too_thin_to_rank_on(tmp_path, churn):
    """**สิ่งที่ผู้สอนต้องรู้ก่อนเปิดรับ** — คลาสที่เหลือไม่กี่แถวตอนตัดสิน

    ไม่ใช่ข้อผิดพลาดของระบบ แต่เป็นข้อผิดพลาดของการออกแบบโจทย์ที่มองไม่เห็น
    จนกว่าจะถึงวันตัดเกรด · เครื่องมือต้องพูดตอนที่ยังแก้ได้
    """
    import yaml
    from tabular import store

    rows = ["a,b,y"] + [f"{i},{i % 7},{1 if i < 12 else 0}" for i in range(1200)]
    digest = store.put(("\n".join(rows) + "\n").encode("utf-8"))
    path = tmp_path / "thin.yaml"
    path.write_text(
        yaml.safe_dump({
            "title": "บาง", "kind": "classification", "primary": "macro_f1",
            "dataset": digest, "target": "y", "drop": [], "labels": [0, 1],
            "split_seed": 1, "bootstrap_seed": 2,
        }, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )

    problems = PREDICTION.verify(path, {p: {} for p in PHASES}, "zz-thin")
    assert any("บางเกินไป" in p for p in problems), problems


def test_the_created_competition_is_one_the_worker_can_actually_run(tmp_path, churn):
    """สร้างจริงผ่าน CLI แล้วให้ worker รันงานจนได้คะแนน

    เขียนแถวลงฐานข้อมูลได้ไม่ได้แปลว่าใช้งานได้ — เทสต์นี้เดินเส้นเดียวกับของจริง
    ตั้งแต่ `--create` จนถึงคะแนนขึ้น run
    """
    import io
    import os
    import shutil
    import subprocess
    import zipfile

    from core.domain import Course, RunStatus
    from core.service import build_arena
    from core.wiring import VALIDATORS
    from runners.worker import Worker
    from tabular.arena import PLUGIN
    from tabular.config import load_config

    config_path, datasets_root = churn

    db_path = tmp_path / "arena.db"
    arena = build_arena(tmp_path / "artifacts", validators=VALIDATORS, db_path=db_path)
    arena.store.save_course(Course(id="cp462-1-2026", name="CP462", join_code="CP462X"))
    arena.store.db.close()

    env = {**os.environ, "ARENA_DATASETS": str(datasets_root)}
    run_tool = subprocess.run(
        [sys.executable, str(REPO / "tools" / "setup_competition.py"),
         "--db", str(db_path), "--slug", "cp462-churn-1-2026", "--create", "--yes",
         "--course", "cp462-1-2026", "--task-type", "prediction",
         "--env-plugin", "tabular.arena:PLUGIN", "--config", str(config_path),
         "--warmup", "2026-09-15..2026-09-30",
         "--main", "2026-10-01..2026-10-31",
         "--final", "2026-11-01..2026-11-30",
         "--opens-now"],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert run_tool.returncode == 0, run_tool.stdout + run_tool.stderr
    assert "สร้างแล้ว" in run_tool.stdout

    arena = build_arena(tmp_path / "artifacts", validators=VALIDATORS, db_path=db_path)
    competition = arena.store.competition_by_slug("cp462-churn-1-2026")
    assert competition.task_type == "prediction"
    assert competition.paradigm == "supervised-learning"
    assert "tabular" not in competition.effective_whitelist()
    assert [p.name for p in competition.phases] == list(PHASES)

    # submission จริง — **ดาวน์โหลดไฟล์แล้วเทรนจากไฟล์ เหมือนที่นิสิตทำ**
    spec = load_config(config_path)
    starter = REPO / "envs" / "cp462-tabular" / "tabular" / "starter"
    work = tmp_path / "student"
    work.mkdir()
    for name in ("predictor.py", "train.py"):
        shutil.copy2(starter / name, work / name)
    (work / "data.csv").write_bytes(PLUGIN.student_bytes(spec))

    trained = subprocess.run(
        [sys.executable, "train.py", "--data", "data.csv",
         "--target", spec.target, "--kind", spec.kind],
        cwd=work, capture_output=True, text=True, timeout=900,
    )
    assert trained.returncode == 0, trained.stderr

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path in sorted(work.iterdir()):
            if path.is_file() and path.name not in ("train.py", "data.csv"):
                zf.writestr(path.name, path.read_bytes())

    user = arena.sign_in(email="a@example.invalid", name="นิสิต", google_sub="s1")
    team = arena.enroll(user=user, join_code="CP462X")
    _submission, run = arena.submit(
        slug="cp462-churn-1-2026", team=team, user_id=user.id, archive=buf.getvalue()
    )

    worker = Worker(
        runner_id="t", store=arena.store, queue=arena.queue,
        artifacts=arena.artifacts, workdir=tmp_path / "work",
    )
    assert worker.drain(limit=2) >= 1

    done = arena.queue.runs[run.id]
    assert done.status is RunStatus.DONE, f"{done.status}: {done.error_message}"
    assert done.score > 0.4
    assert done.metrics["n_rows"] == PLUGIN.preview(spec)["sizes"]["test_public"]
