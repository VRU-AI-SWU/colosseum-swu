"""ต่อ `tabular` เข้ากับ runner จริง — ทั้งเส้นตั้งแต่ starter kit ถึงคะแนน

`runners/tests/test_prediction_runner.py` พิสูจน์ว่า **runner** ทำงานถูก โดยใช้
โจทย์ปลอมที่ไม่ต้องมี scikit-learn · ไฟล์นี้พิสูจน์อีกครึ่งหนึ่ง: ว่า **โจทย์จริง**
เสียบเข้ากับ runner นั้นได้ และสิ่งที่นิสิตได้รับไปจริงๆ ผ่านทั้งเส้นทาง

ถ้าไฟล์นี้ล้ม แปลว่านิสิตทำตาม starter kit ทุกอย่างแล้วยังส่งงานไม่ผ่าน —
ซึ่งไม่ใช่ความผิดของเขา

**เทสต์เดินเส้นทางเดียวกับของจริงทุกขั้น** — ผู้สอนอัปโหลด CSV เข้าคลัง · ระบบ
แบ่งสามกอง · นิสิตดาวน์โหลดกองของตัวเองเป็นไฟล์ · เทรนจากไฟล์นั้น · ส่งเข้ามา
วัดกับกองที่เขาไม่เคยเห็น · เดิมเทสต์ลัดด้วยการให้ทั้งสองฝั่งสร้างข้อมูลจากเมล็ด
ซึ่งเป็นเส้นทางที่ไม่มีใครใช้จริงอีกแล้ว
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tabular", reason="ต้องติดตั้ง envs/cp462-tabular ก่อน")

from runners.prediction.plugin import AUTHORING, REQUIRED, resolve
from runners.prediction.runner import run_submission
from runners.sandbox.launcher import SubprocessLauncher
from tabular.arena import PLUGIN

PLUGIN_SPEC = "tabular.arena:PLUGIN"
REPO = Path(__file__).resolve().parent.parent.parent
STARTER = REPO / "envs" / "cp462-tabular" / "tabular" / "starter"

from runners.tests.conftest import TABULAR_TASKS as TASKS


def launcher() -> SubprocessLauncher:
    return SubprocessLauncher(host_module="runners.prediction.predictor_host")


@pytest.fixture(scope="module")
def tasks(tabular_tasks):
    """โจทย์ทดสอบ — นิยามอยู่ที่ `conftest.py` เพราะเทสต์ Docker ใช้ตัวเดียวกัน"""
    return tabular_tasks


@pytest.fixture(scope="module")
def trained(tmp_path_factory, tasks):
    """submission จริง — **ดาวน์โหลดไฟล์แล้วเทรนจากไฟล์ เหมือนที่นิสิตทำ**

    `scope="module"` เพราะการเทรนสองโจทย์กินเวลาหลายสิบวินาที และผลของมันคงที่
    """
    built = {}
    for name, (_path, spec) in tasks.items():
        work = tmp_path_factory.mktemp(name)
        for filename in ("predictor.py", "train.py"):
            shutil.copy2(STARTER / filename, work / filename)
        # นี่คือไฟล์ที่นิสิตกดดาวน์โหลดจากหน้าโจทย์ — ทางออกทางเดียวของข้อมูล
        (work / "data.csv").write_bytes(PLUGIN.student_bytes(spec))

        run = subprocess.run(
            [sys.executable, "train.py", "--data", "data.csv",
             "--target", spec.target, "--kind", spec.kind, "--primary", spec.primary],
            cwd=work, capture_output=True, text=True, timeout=900,
        )
        assert run.returncode == 0, f"{name}: train.py ล้ม\n{run.stderr}"
        for leftover in ("train.py", "data.csv"):
            (work / leftover).unlink()  # นิสิตส่งแค่ predictor.py + pipeline.pkl
        built[name] = work
    return built


# ── สัญญาของ plugin ────────────────────────────────────────────────


def test_the_plugin_satisfies_both_contracts():
    """`tabular` ต้องทำได้ทั้งฝั่งให้คะแนนและฝั่งสร้างโจทย์"""
    assert resolve(PLUGIN_SPEC, also=AUTHORING) is PLUGIN
    for name in (*REQUIRED, *AUTHORING):
        assert callable(getattr(PLUGIN, name)), f"{name} ต้องเรียกได้"


def test_predictor_config_carries_nothing_from_the_grading_set(tasks):
    """**ด่านสำคัญที่สุดของไฟล์นี้** — ตรวจรายการคีย์แบบเป๊ะ ไม่ใช่ blacklist

    blacklist พลาดทุกครั้งที่มีคนเพิ่มฟิลด์ใหม่ที่ตั้งชื่อไม่ตรงคำต้องห้าม
    การล็อกรายการทำให้ฟิลด์ใหม่ที่รั่วต้องผ่านการแก้เทสต์นี้ก่อนเสมอ
    """
    for name, (_path, spec) in tasks.items():
        config = PLUGIN.predictor_config(spec)
        assert set(config) == {"kind", "primary"}, f"{name}: {sorted(config)}"
        assert spec.dataset not in config.values(), "รหัสชุดข้อมูลต้องไม่เข้าไปในกล่อง"
        assert spec.split_seed not in config.values()
        assert spec.bootstrap_seed not in config.values()


@pytest.mark.parametrize("name", sorted(TASKS))
def test_config_hash_survives_the_round_trip_through_yaml(name, tasks):
    """hash ที่ runner บันทึกลง run ต้องเป็นตัวเดียวกับที่คำนวณจากสเปคในหน่วยความจำ"""
    path, spec = tasks[name]
    assert PLUGIN.config_hash(PLUGIN.load_spec(str(path))) == spec.config_hash


@pytest.mark.parametrize("name", sorted(TASKS))
def test_what_students_download_shares_no_row_with_the_grading_set(name, tasks):
    """**ข้อที่พังแล้วการแข่งจบ** — ไฟล์ที่แจกต้องไม่ทับกับกองที่ใช้ตัดสิน

    เคยพังจริงในรูปแบบก่อนหน้า: ทั้งสองชุดสร้างจากเมล็ดที่อยู่ในไฟล์ที่แจก
    นิสิตจึงคำนวณเฉลยเองได้ครบทุกแถว (macro-F1 = 1.0000) · ตอนนี้กองที่ใช้
    ตัดสินเป็นแถวที่ไม่เคยถูกส่งออกไป ไม่ใช่แถวที่สร้างใหม่จากตัวเลขลับ
    """
    import io

    import pandas as pd

    _path, spec = tasks[name]
    handed_out = pd.read_csv(io.BytesIO(PLUGIN.student_bytes(spec)))
    features = [c for c in handed_out.columns if c != spec.target]
    seen = set(map(tuple, handed_out[features].astype(str).itertuples(index=False)))

    for kind in ("public", "private"):
        graded = PLUGIN.grading_data(spec, kind)
        secret = set(map(tuple, graded.X[features].astype(str).itertuples(index=False)))
        assert not (seen & secret), f"{kind}: {len(seen & secret)} แถวอยู่ในไฟล์ที่แจกแล้ว"


def test_the_download_never_carries_a_grading_row_even_after_a_reload(tasks):
    """เรียกซ้ำต้องได้ไฟล์เดิมเป๊ะ — ไม่งั้นนิสิตสองคนได้ข้อมูลคนละชุด"""
    _path, spec = tasks["churn"]
    assert PLUGIN.student_bytes(spec) == PLUGIN.student_bytes(spec)


# ── ทั้งเส้นด้วย starter kit ที่แจกจริง ─────────────────────────────


@pytest.mark.parametrize("name", sorted(TASKS))
def test_the_shipped_starter_kit_passes_every_check(name, tasks, trained):
    """**สิ่งที่นิสิตได้รับไปต้องผ่านการตรวจทั้งสามชั้น**

    ถ้าข้อนี้ล้ม แปลว่าเราแจก pipeline ที่ระบบของเราเองปฏิเสธ
    """
    path, _spec = tasks[name]
    result = run_submission(
        env_plugin=PLUGIN_SPEC,
        config_path=path,
        submission_dir=trained[name],
        launcher=launcher(),
    )
    assert result.ok, f"{result.status}: {result.detail}\n{result.log[-2000:]}"
    assert result.checks == {
        "determinism": True,
        "row_permutation": True,
        "subset_consistency": True,
    }


@pytest.mark.parametrize("name", sorted(TASKS))
def test_the_score_is_the_same_metric_students_measure_themselves(name, tasks, trained):
    """คะแนนจาก runner ต้องตรงกับที่คิดตรงๆ ทุกหลัก — ไม่งั้น leaderboard โกหก"""
    path, spec = tasks[name]
    result = run_submission(
        env_plugin=PLUGIN_SPEC,
        config_path=path,
        submission_dir=trained[name],
        launcher=launcher(),
    )
    assert result.ok, f"{result.status}: {result.detail}"
    assert result.score.primary_name == spec.primary
    assert result.score.ci_low <= result.score.primary <= result.score.ci_high
    assert result.score.primary > 0.4, f"starter kit ควรได้คะแนนพอใช้ — ได้ {result.score.primary}"


def test_public_and_private_are_different_sets(tasks, trained):
    """คะแนนสองชุดต้องมาจากข้อมูลคนละก้อน — ไม่งั้นชุดลับไม่มีความหมาย"""
    path, spec = tasks["churn"]
    expected = PLUGIN.preview(spec)["sizes"]

    scores = {}
    for kind in ("public", "private"):
        result = run_submission(
            env_plugin=PLUGIN_SPEC,
            config_path=path,
            submission_dir=trained["churn"],
            kind=kind,
            launcher=launcher(),
        )
        assert result.ok, f"{kind}: {result.status}: {result.detail}"
        scores[kind] = result.score.primary
        assert result.n_rows == expected[f"test_{kind}"]
    assert scores["public"] != scores["private"], "สองชุดให้คะแนนเท่ากันเป๊ะ — น่าสงสัย"


def test_a_pipeline_that_refits_on_predict_is_caught(tmp_path, tasks):
    """leakage แบบที่นิสิตทำโดยไม่รู้ตัวบ่อยที่สุด — normalize ใหม่ทุกครั้งที่ทำนาย"""
    (tmp_path / "predictor.py").write_text(
        "import numpy as np\n"
        "\n"
        "class Predictor:\n"
        "    def __init__(self, config):\n"
        "        pass\n"
        "\n"
        "    def predict(self, X):\n"
        "        col = X['monthly_spend'].to_numpy(dtype='float64')\n"
        "        return (col > np.nanmean(col)).astype('int64')\n",
        encoding="utf-8",
    )
    result = run_submission(
        env_plugin=PLUGIN_SPEC,
        config_path=tasks["churn"][0],
        submission_dir=tmp_path,
        launcher=launcher(),
    )
    assert result.status == "batch_dependent", f"{result.status}: {result.detail}"


def test_predictions_outside_the_declared_labels_are_rejected(tmp_path, tasks):
    """คลาสที่ไม่มีในโจทย์ต้องถูกปฏิเสธ ไม่ใช่ถูกนับเป็นทายผิดเฉยๆ"""
    (tmp_path / "predictor.py").write_text(
        "import numpy as np\n"
        "\n"
        "class Predictor:\n"
        "    def __init__(self, config):\n"
        "        pass\n"
        "\n"
        "    def predict(self, X):\n"
        "        return np.full(len(X), 7, dtype='int64')\n",
        encoding="utf-8",
    )
    result = run_submission(
        env_plugin=PLUGIN_SPEC,
        config_path=tasks["churn"][0],
        submission_dir=tmp_path,
        launcher=launcher(),
    )
    assert result.status == "bad_prediction"
    assert "7" in result.detail


def test_nan_predictions_are_rejected(tmp_path, tasks):
    (tmp_path / "predictor.py").write_text(
        "import numpy as np\n"
        "\n"
        "class Predictor:\n"
        "    def __init__(self, config):\n"
        "        pass\n"
        "\n"
        "    def predict(self, X):\n"
        "        return np.full(len(X), np.nan)\n",
        encoding="utf-8",
    )
    result = run_submission(
        env_plugin=PLUGIN_SPEC,
        config_path=tasks["housing"][0],
        submission_dir=tmp_path,
        launcher=launcher(),
    )
    assert result.status == "bad_prediction"
    assert "NaN" in result.detail


def test_the_answers_never_reach_the_sandbox(tmp_path, tasks):
    """โค้ดในกล่องต้องไม่มีทางเห็นเฉลย — ไล่หาทุกทางที่ `SubprocessLauncher` ตอบได้

    ต่างจากโจทย์ RL ตรงที่ของลับไม่ใช่ seed แต่เป็น `y` ของชุดที่ใช้ตัดสิน
    **และตอนนี้รวมถึงรหัสของไฟล์ในคลังด้วย** — ใครที่ได้รหัสนั้นไปพร้อมกับสิทธิ์
    อ่านคลัง จะอ่านไฟล์เต็มได้ทั้งใบ

    ⚠️ ข้อที่ว่า "`import tabular` จากในกล่องไม่ได้" ตรวจที่นี่ไม่ได้ — `SubprocessLauncher`
    ใช้ interpreter ตัวเดียวกับ runner จึงเห็นทุกอย่างที่ runner เห็น มันเป็น
    สมบัติของ **image** ไม่ใช่ของ runner · อยู่ที่ `test_docker_sandbox.py` แทน
    """
    (tmp_path / "predictor.py").write_text(
        "import numpy as np\n"
        "\n"
        "FORBIDDEN = ('churned', 'monthly_value', 'y', 'target', 'label')\n"
        "\n"
        "class Predictor:\n"
        "    def __init__(self, config):\n"
        "        import gc\n"
        "        for obj in gc.get_objects():\n"
        "            if type(obj).__name__ in ('TaskSpec', 'ThreeWay', 'Dataset'):\n"
        "                raise AssertionError('เอื้อมถึง ' + type(obj).__name__ + ' ได้')\n"
        "        leaked = [k for k, v in config.items() if 'sha256' in str(v)]\n"
        "        if leaked:\n"
        "            raise AssertionError('รหัสชุดข้อมูลหลุดเข้ากล่อง: ' + str(leaked))\n"
        "\n"
        "    def predict(self, X):\n"
        "        leaked = [c for c in X.columns if c in FORBIDDEN]\n"
        "        if leaked:\n"
        "            raise AssertionError('เฉลยหลุดมาในตาราง: ' + str(leaked))\n"
        "        return np.zeros(len(X), dtype='int64')\n",
        encoding="utf-8",
    )
    result = run_submission(
        env_plugin=PLUGIN_SPEC,
        config_path=tasks["churn"][0],
        submission_dir=tmp_path,
        launcher=launcher(),
    )
    assert result.ok, f"{result.status}: {result.detail}"


# ── ครบวงจร: ส่งงาน → คิว → worker → คะแนน ──────────────────────────


def test_a_prediction_competition_flows_through_the_worker(tmp_path, tasks, trained):
    """เกณฑ์เดียวกับ README §14 M1 แต่สำหรับโจทย์ทำนาย

    พิสูจน์ว่า `task_type="prediction"` เดินทางครบเส้น — ตัวตรวจไฟล์ใน `core`
    หยิบตัวที่ถูก · `Worker` เลือก runner ที่ถูก · คะแนนถูกบันทึกในรูปที่
    leaderboard ใช้ได้
    """
    import io
    import zipfile

    from core.domain import Competition, Course, Phase, RunStatus, new_id
    from core.service import build_arena
    from core.wiring import VALIDATORS
    from runners.worker import Worker

    path, spec = tasks["churn"]
    arena = build_arena(tmp_path / "artifacts", validators=VALIDATORS)
    arena.store.save_course(
        Course(id="cp462-1-2026", name="CP462 Machine Learning", join_code="CP462TEST")
    )
    now = datetime.now(timezone.utc)
    competition = Competition(
        id=new_id(),
        course_id="cp462-1-2026",
        slug="cp462-churn-1-2026",
        title="ทำนายการเลิกใช้บริการ",
        task_type="prediction",
        env_plugin=PLUGIN_SPEC,
        config_path=str(path),
        opens_at=now - timedelta(days=1),
        closes_at=now + timedelta(days=30),
        quota_per_day=5,
        # **ไม่มี `tabular`** — มันจงใจไม่อยู่ใน image ของโจทย์นี้
        import_whitelist=frozenset({"numpy", "pandas", "sklearn", "scipy", "joblib"}),
        phases=[Phase(id=new_id(), name="main",
                      starts_at=now - timedelta(days=1), ends_at=now + timedelta(days=30))],
    )
    arena.store.save_competition(competition)

    user = arena.sign_in(email="นิสิต@example.invalid", name="นิสิต", google_sub="sub-cp462")
    team = arena.enroll(user=user, join_code="CP462TEST")

    # เอาเฉพาะไฟล์ — เทสต์ก่อนหน้ารัน predictor ในโฟลเดอร์นี้จนเกิด `__pycache__`
    # (ในกล่องจริงไม่เกิด เพราะ image ตั้ง PYTHONDONTWRITEBYTECODE และ mount แบบอ่านอย่างเดียว)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for item in sorted(trained["churn"].iterdir()):
            if item.is_file():
                zf.writestr(item.name, item.read_bytes())

    _submission, run = arena.submit(
        slug=competition.slug, team=team, user_id=user.id, archive=buf.getvalue(),
    )

    worker = Worker(
        runner_id="runner-test",
        store=arena.store,
        queue=arena.queue,
        artifacts=arena.artifacts,
        workdir=tmp_path / "work",
    )
    assert worker.drain(limit=2) >= 1

    done = arena.queue.runs[run.id]
    assert done.status is RunStatus.DONE, f"{done.status}: {done.error_message}"
    assert done.score is not None and done.score > 0.4
    assert done.metrics["checks"] == {
        "determinism": True, "row_permutation": True, "subset_consistency": True,
    }
    assert done.metrics["n_rows"] == PLUGIN.preview(spec)["sizes"]["test_public"]
    assert done.config_hash and done.env_version
    assert np.isfinite(done.score)
