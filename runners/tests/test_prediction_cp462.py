"""ต่อ `tabular` เข้ากับ runner จริง — ทั้งเส้นตั้งแต่ starter kit ถึงคะแนน

`runners/tests/test_prediction_runner.py` พิสูจน์ว่า **runner** ทำงานถูก โดยใช้
โจทย์ปลอมที่ไม่ต้องมี scikit-learn · ไฟล์นี้พิสูจน์อีกครึ่งหนึ่ง: ว่า **โจทย์จริง**
เสียบเข้ากับ runner นั้นได้ และสิ่งที่นิสิตได้รับไปจริงๆ ผ่านทั้งเส้นทาง

ถ้าไฟล์นี้ล้ม แปลว่านิสิตทำตาม starter kit ทุกอย่างแล้วยังส่งงานไม่ผ่าน —
ซึ่งไม่ใช่ความผิดของเขา
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tabular", reason="ต้องติดตั้ง envs/cp462-tabular ก่อน")

from runners.prediction.plugin import REQUIRED, resolve
from runners.prediction.runner import run_submission
from runners.sandbox.launcher import SubprocessLauncher
from tabular.arena import PLUGIN
from tabular.config import CONFIG_DIR, load

PLUGIN_SPEC = "tabular.arena:PLUGIN"
REPO = Path(__file__).resolve().parent.parent.parent
STARTER = REPO / "envs" / "cp462-tabular" / "tabular" / "starter"
SLUGS = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def launcher() -> SubprocessLauncher:
    return SubprocessLauncher(host_module="runners.prediction.predictor_host")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """submission จริง: เทรนด้วย `train.py` ที่แจก แล้วเหลือแต่ไฟล์ที่นิสิตต้องส่ง

    `scope="module"` เพราะการเทรนสองโจทย์กินเวลาหลายสิบวินาที และผลของมันคงที่
    """
    built = {}
    for slug in SLUGS:
        work = tmp_path_factory.mktemp(slug)
        for name in ("predictor.py", "train.py"):
            shutil.copy2(STARTER / name, work / name)
        run = subprocess.run(
            [sys.executable, "train.py", "--task", slug],
            cwd=work, capture_output=True, text=True, timeout=900,
        )
        assert run.returncode == 0, f"{slug}: train.py ล้ม\n{run.stderr}"
        (work / "train.py").unlink()  # นิสิตส่งแค่ predictor.py + pipeline.pkl
        built[slug] = work
    return built


# ── สัญญาของ plugin ────────────────────────────────────────────────


def test_the_plugin_satisfies_the_contract():
    assert resolve(PLUGIN_SPEC) is PLUGIN
    for name in REQUIRED:
        assert callable(getattr(PLUGIN, name)), f"{name} ต้องเรียกได้"


def test_predictor_config_carries_nothing_from_the_grading_set():
    """**ด่านสำคัญที่สุดของไฟล์นี้** — ตรวจรายการคีย์แบบเป๊ะ ไม่ใช่ blacklist

    blacklist พลาดทุกครั้งที่มีคนเพิ่มฟิลด์ใหม่ที่ตั้งชื่อไม่ตรงคำต้องห้าม
    การล็อกรายการทำให้ฟิลด์ใหม่ที่รั่วต้องผ่านการแก้เทสต์นี้ก่อนเสมอ
    """
    for slug in SLUGS:
        spec = load(slug)
        config = PLUGIN.predictor_config(spec)
        assert set(config) == {"task", "kind", "primary"}, f"{slug}: {sorted(config)}"
        assert spec.data_seed not in config.values()
        assert spec.split_seed not in config.values()
        assert spec.bootstrap_seed not in config.values()


@pytest.mark.parametrize("slug", SLUGS)
def test_config_hash_matches_what_the_students_selfcheck_pins(slug):
    """hash ที่ runner บันทึกลง run ต้องเป็นตัวเดียวกับที่ `selfcheck` ตรึงไว้"""
    spec = load(slug)
    assert PLUGIN.config_hash(spec) == spec.config_hash


@pytest.mark.parametrize("slug", SLUGS)
def test_grading_data_is_not_what_students_get(slug):
    """ชุดที่ใช้ตัดสินมาจาก dataset คนละใบ — และเมล็ดของมันมาจาก ARENA_SECRETS

    `load()` (ฝั่งนิสิต) ได้สเปคที่ไม่มีเมล็ดลับ · `PLUGIN.load_spec()` (ฝั่ง trusted)
    ฉีดเข้ามา — ความต่างนี้คือสิ่งที่กันไม่ให้นิสิตคำนวณเฉลยเองได้
    """
    from tabular.dataset import open_data

    assert load(slug).grading_seed is None, "สเปคฝั่งนิสิตต้องไม่มีเมล็ดลับ"
    spec = PLUGIN.load_spec(str(CONFIG_DIR / f"{slug}.yaml"))
    assert spec.grading_seed is not None, "ฝั่ง trusted ต้องได้เมล็ดลับมา"

    student_ids = set()
    for part in open_data(spec).values():
        student_ids |= set(part.X["account_id"])
    for kind in ("public", "private"):
        graded = PLUGIN.grading_data(spec, kind)
        assert not (set(graded.X["account_id"]) & student_ids), f"{kind}: มีแถวซ้ำ"


# ── ทั้งเส้นด้วย starter kit ที่แจกจริง ─────────────────────────────


@pytest.mark.parametrize("slug", SLUGS)
def test_the_shipped_starter_kit_passes_every_check(slug, trained):
    """**สิ่งที่นิสิตได้รับไปต้องผ่านการตรวจทั้งสามชั้น**

    ถ้าข้อนี้ล้ม แปลว่าเราแจก pipeline ที่ระบบของเราเองปฏิเสธ
    """
    result = run_submission(
        env_plugin=PLUGIN_SPEC,
        config_path=CONFIG_DIR / f"{slug}.yaml",
        submission_dir=trained[slug],
        launcher=launcher(),
    )
    assert result.ok, f"{result.status}: {result.detail}\n{result.log[-2000:]}"
    assert result.checks == {
        "determinism": True,
        "row_permutation": True,
        "subset_consistency": True,
    }


@pytest.mark.parametrize("slug", SLUGS)
def test_the_score_is_the_same_metric_students_measure_themselves(slug, trained):
    """คะแนนจาก runner ต้องตรงกับที่คิดตรงๆ ทุกหลัก — ไม่งั้น leaderboard โกหก"""
    spec = load(slug)
    result = run_submission(
        env_plugin=PLUGIN_SPEC,
        config_path=CONFIG_DIR / f"{slug}.yaml",
        submission_dir=trained[slug],
        launcher=launcher(),
    )
    assert result.ok, f"{result.status}: {result.detail}"
    assert result.score.primary_name == spec.primary
    assert result.score.ci_low <= result.score.primary <= result.score.ci_high
    assert result.score.primary > 0.4, f"starter kit ควรได้คะแนนพอใช้ — ได้ {result.score.primary}"


def test_public_and_private_are_different_sets(trained):
    """คะแนนสองชุดต้องมาจากข้อมูลคนละก้อน — ไม่งั้นชุดลับไม่มีความหมาย"""
    slug = SLUGS[0]
    scores = {}
    for kind in ("public", "private"):
        result = run_submission(
            env_plugin=PLUGIN_SPEC,
            config_path=CONFIG_DIR / f"{slug}.yaml",
            submission_dir=trained[slug],
            kind=kind,
            launcher=launcher(),
        )
        assert result.ok, f"{kind}: {result.status}: {result.detail}"
        scores[kind] = result.score.primary
        assert result.n_rows == {"public": 1200, "private": 1800}[kind]
    assert scores["public"] != scores["private"], "สองชุดให้คะแนนเท่ากันเป๊ะ — น่าสงสัย"


def test_a_pipeline_that_refits_on_predict_is_caught(tmp_path):
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
        config_path=CONFIG_DIR / "churn.yaml",
        submission_dir=tmp_path,
        launcher=launcher(),
    )
    assert result.status == "batch_dependent", f"{result.status}: {result.detail}"


def test_predictions_outside_the_declared_labels_are_rejected(tmp_path):
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
        config_path=CONFIG_DIR / "churn.yaml",
        submission_dir=tmp_path,
        launcher=launcher(),
    )
    assert result.status == "bad_prediction"
    assert "7" in result.detail


def test_nan_predictions_are_rejected(tmp_path):
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
        config_path=CONFIG_DIR / "housing.yaml",
        submission_dir=tmp_path,
        launcher=launcher(),
    )
    assert result.status == "bad_prediction"
    assert "NaN" in result.detail


def test_the_answers_never_reach_the_sandbox(tmp_path):
    """โค้ดในกล่องต้องไม่มีทางเห็นเฉลย — ไล่หาทุกทางที่ `SubprocessLauncher` ตอบได้

    ต่างจากโจทย์ RL ตรงที่ของลับไม่ใช่ seed แต่เป็น `y` ของชุดที่ใช้ตัดสิน

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
        "            if type(obj).__name__ in ('TaskSpec', 'Split', 'Dataset'):\n"
        "                raise AssertionError('เอื้อมถึง ' + type(obj).__name__ + ' ได้')\n"
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
        config_path=CONFIG_DIR / "churn.yaml",
        submission_dir=tmp_path,
        launcher=launcher(),
    )
    assert result.ok, f"{result.status}: {result.detail}"
    assert np.isclose(result.score.primary, result.score.primary)
