"""runner ของโจทย์ทำนาย — การตรวจสามชั้นและเส้นแบ่ง trust boundary

ใช้ plugin ปลอมที่ประกาศในไฟล์นี้ ไม่ใช่ `tabular` ของ CP462 โดยตั้งใจ

  · runner ต้องไม่รู้จักโจทย์ใดโจทย์หนึ่ง — เทสต์ที่ผูกกับ CP462 จะกลายเป็น
    เทสต์ของ CP462 แทนที่จะเป็นเทสต์ของ runner
  · เขียน predictor ที่**โกงแบบเจาะจง**ได้ง่าย ซึ่งเป็นสิ่งที่ต้องพิสูจน์ว่าจับได้
  · ไม่ต้องมี scikit-learn — `runners/` ตั้งใจให้ติดตั้งได้โดยไม่มีมัน

เทสต์ที่ใช้ของจริงทั้งเส้นอยู่ที่ `envs/cp462-tabular/tests/test_arena_plugin.py`
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from runners.prediction.runner import run_submission
from runners.sandbox.launcher import SubprocessLauncher

HERE = Path(__file__).resolve().parent
FAKE_PLUGIN = "runners.tests.test_prediction_runner:PLUGIN"
N_ROWS = 60


# ── โจทย์ปลอมที่เล็กที่สุดที่ยังครบสัญญา ────────────────────────────


@dataclass
class FakeData:
    X: pd.DataFrame
    y: np.ndarray


@dataclass
class FakeScore:
    primary_name: str
    primary: float

    def as_dict(self) -> dict:
        return {"primary_name": self.primary_name, "primary": self.primary}


class FakePlugin:
    """โจทย์ทำนายที่เล็กที่สุด — คะแนนคือสัดส่วนที่ทายถูก"""

    name = "fake"

    def load_spec(self, path: str) -> dict:
        return {"kind": "classification", "hash": "sha256:00000000000000ff"}

    def apply_overrides(self, spec: dict, overrides: dict) -> dict:
        return {**spec, **overrides}

    def config_hash(self, spec: dict) -> str:
        return spec["hash"]

    def env_version(self, spec: dict) -> str:
        return "0.0.1"

    def grading_data(self, spec: dict, kind: str) -> FakeData:
        rng = np.random.default_rng(7 if kind == "public" else 8)
        return FakeData(
            X=pd.DataFrame(
                {
                    "num": rng.normal(size=N_ROWS),
                    "grp": pd.Categorical(rng.choice(["a", "b", "c"], size=N_ROWS)),
                }
            ),
            y=rng.integers(0, 2, size=N_ROWS),
        )

    def predictor_config(self, spec: dict) -> dict:
        return {"kind": spec["kind"]}

    def score(self, spec: dict, y_true, y_pred) -> FakeScore:
        y_pred = np.asarray(y_pred)
        if y_pred.shape != np.asarray(y_true).shape:
            raise ValueError(f"รูปร่างไม่ตรง: {y_pred.shape} vs {np.asarray(y_true).shape}")
        return FakeScore("accuracy", float(np.mean(np.asarray(y_true) == y_pred)))

    def predict_timeout_s(self, spec: dict) -> float:
        return 30.0


PLUGIN = FakePlugin()


# ── ตัวช่วย ────────────────────────────────────────────────────────


@pytest.fixture
def make_submission(tmp_path):
    """เขียน `predictor.py` ลงโฟลเดอร์แล้วคืน path — เหมือน `make_submission` ของโจทย์ RL"""

    def build(body: str) -> Path:
        directory = tmp_path / "submission"
        directory.mkdir(exist_ok=True)
        (directory / "predictor.py").write_text(textwrap.dedent(body), encoding="utf-8")
        return directory

    return build


def run(submission, **kwargs):
    return run_submission(
        env_plugin=FAKE_PLUGIN,
        config_path="ไม่ได้ใช้ — plugin ปลอมไม่อ่านไฟล์",
        submission_dir=submission,
        launcher=SubprocessLauncher(host_module="runners.prediction.predictor_host"),
        **kwargs,
    )


CONSTANT = """
    import numpy as np

    class Predictor:
        def __init__(self, config):
            self.config = config
        def predict(self, X):
            return np.zeros(len(X), dtype="int64")
"""


# ── ทางที่ถูกต้อง ──────────────────────────────────────────────────


def test_a_well_behaved_predictor_is_scored(make_submission):
    result = run(make_submission(CONSTANT))
    assert result.ok, f"{result.status}: {result.detail}"
    assert result.score.primary_name == "accuracy"
    assert 0.0 <= result.score.primary <= 1.0
    assert result.n_rows == N_ROWS
    assert result.config_hash == "sha256:00000000000000ff"
    assert result.env_version == "0.0.1"


def test_all_three_checks_are_reported_as_passed(make_submission):
    """ผลของการตรวจต้องถูกรายงาน ไม่ใช่แค่ "ไม่ล้ม" — ผู้สอนต้องเห็นว่าตรวจแล้วจริง"""
    result = run(make_submission(CONSTANT))
    assert result.checks == {
        "determinism": True,
        "row_permutation": True,
        "subset_consistency": True,
    }


def test_predictions_can_be_strings(make_submission):
    """label ที่เป็นข้อความต้องเดินทางกลับได้ — ไม่ใช่ทุกโจทย์ใช้ตัวเลข"""
    result = run(
        make_submission(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config): pass
                def predict(self, X):
                    return np.array(["ก"] * len(X), dtype=object)
            """
        )
    )
    # plugin ปลอมเทียบกับเฉลยที่เป็นตัวเลข จึงได้ 0 คะแนน — ประเด็นคือมันไปถึงขั้นให้คะแนนได้
    assert result.ok, f"{result.status}: {result.detail}"
    assert result.score.primary == 0.0


def test_predictor_sees_the_config_but_not_the_answers(make_submission):
    """สิ่งที่เข้ากล่องได้มีแค่ `predictor_config` — ต้องไม่มีอะไรจากชุดที่ใช้ตัดสิน"""
    result = run(
        make_submission(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config):
                    assert set(config) == {"kind"}, f"config มีฟิลด์เกิน: {sorted(config)}"
                def predict(self, X):
                    assert list(X.columns) == ["num", "grp"], list(X.columns)
                    assert "y" not in X.columns and "target" not in X.columns
                    return np.zeros(len(X), dtype="int64")
            """
        )
    )
    assert result.ok, f"{result.status}: {result.detail}"


def test_the_frame_arrives_with_its_dtypes_intact(make_submission):
    """สิ่งที่กล่องเห็นต้องเป็นตารางเดียวกับที่ฝั่ง trusted ส่งไป ไม่ใช่ของที่แปลงแล้ว"""
    result = run(
        make_submission(
            """
            import numpy as np
            import pandas as pd

            class Predictor:
                def __init__(self, config): pass
                def predict(self, X):
                    assert str(X["num"].dtype) == "float64", X["num"].dtype
                    assert isinstance(X["grp"].dtype, pd.CategoricalDtype), X["grp"].dtype
                    assert list(X.index) == list(range(len(X))), "index ต้องถูกรีเซ็ต"
                    return np.zeros(len(X), dtype="int64")
            """
        )
    )
    assert result.ok, f"{result.status}: {result.detail}"


# ── การตรวจสามชั้นต้องจับของจริงได้ ────────────────────────────────


def test_a_nondeterministic_predictor_is_rejected(make_submission):
    result = run(
        make_submission(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config):
                    self.rng = np.random.default_rng()
                def predict(self, X):
                    return self.rng.integers(0, 2, size=len(X))
            """
        )
    )
    assert result.status == "nondeterministic", result.detail
    assert "random_state" in result.detail
    assert result.score is None, "การตรวจไม่ผ่านต้องไม่มีคะแนนติดออกไป"


def test_a_row_order_dependent_predictor_is_rejected(make_submission):
    """ทำนายตามตำแหน่งของแถวในก้อน — ผ่านการตรวจความคงที่ แต่ต้องตกข้อนี้"""
    result = run(
        make_submission(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config): pass
                def predict(self, X):
                    return (np.arange(len(X)) % 2).astype("int64")
            """
        )
    )
    assert result.status == "row_order_dependent", result.detail
    assert "แถวที่" in result.detail, "ต้องบอกด้วยว่าแถวไหนไม่ตรง"


def test_a_batch_dependent_predictor_is_rejected(make_submission):
    """ใช้สถิติของก้อนที่รับเข้ามาแทนค่าที่จำไว้ตอน fit — leakage แบบที่เจอบ่อยที่สุด

    ตัวนี้ทนต่อการสลับแถว (ค่าเฉลี่ยของทั้งก้อนไม่ขึ้นกับลำดับ) จึงผ่านสองข้อแรก
    ได้ — ต้องมีข้อที่สามถึงจะจับได้
    """
    result = run(
        make_submission(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config): pass
                def predict(self, X):
                    return (X["num"].to_numpy() > X["num"].mean()).astype("int64")
            """
        )
    )
    assert result.status == "batch_dependent", result.detail
    assert "สถิติ" in result.detail


def test_check_failures_name_the_rows_of_the_full_set(make_submission):
    """เลขแถวที่รายงานต้องเป็นเลขของชุดเต็ม ไม่ใช่ตำแหน่งในก้อนที่สลับแล้ว

    ถ้ารายงานเป็นตำแหน่งในก้อนย่อย ผู้สอนจะไปดูแถวผิดตัวตอนไล่ปัญหากับนิสิต
    """
    result = run(
        make_submission(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config): pass
                def predict(self, X):
                    return (np.arange(len(X)) % 2).astype("int64")
            """
        )
    )
    rows = [int(line.split("แถวที่ ")[1].split(":")[0])
            for line in result.detail.splitlines() if "แถวที่ " in line]
    assert rows, result.detail
    assert all(0 <= r < N_ROWS for r in rows), rows


def test_checks_can_be_skipped_for_a_smoke_test(make_submission):
    """smoke test แค่อยากรู้ว่ากล่องเปิดติด — แต่ต้องไม่มีคะแนนออกไปโดยไม่ถูกตรวจ"""
    result = run(
        make_submission(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config):
                    self.rng = np.random.default_rng()
                def predict(self, X):
                    return self.rng.integers(0, 2, size=len(X))
            """
        ),
        run_checks=False,
    )
    assert result.ok, f"{result.status}: {result.detail}"
    assert result.checks == {}, "ไม่ได้ตรวจก็ต้องไม่รายงานว่าผ่าน"


# ── submission ที่พัง ──────────────────────────────────────────────


def test_crash_in_init_is_reported_with_the_traceback(make_submission):
    result = run(
        make_submission(
            """
            class Predictor:
                def __init__(self, config):
                    raise ValueError("โหลด pipeline.pkl ไม่ได้")
                def predict(self, X): ...
            """
        )
    )
    assert result.status == "predictor_init_failed"
    assert "โหลด pipeline.pkl ไม่ได้" in result.detail


def test_missing_predictor_py_is_reported_clearly(tmp_path):
    (tmp_path / "ว่าง").mkdir()
    result = run(tmp_path / "ว่าง")
    assert result.status == "predictor_init_failed"
    assert "predictor.py" in result.detail


def test_crash_in_predict_is_reported_with_the_traceback(make_submission):
    result = run(
        make_submission(
            """
            class Predictor:
                def __init__(self, config): pass
                def predict(self, X):
                    raise RuntimeError("ลืมแปลงคอลัมน์")
            """
        )
    )
    assert result.status == "predict_failed"
    assert "ลืมแปลงคอลัมน์" in result.detail


def test_wrong_number_of_predictions_is_rejected(make_submission):
    result = run(
        make_submission(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config): pass
                def predict(self, X):
                    return np.zeros(len(X) - 1, dtype="int64")
            """
        )
    )
    assert result.status == "bad_prediction"
    assert str(N_ROWS) in result.detail


def test_a_slow_predictor_times_out_instead_of_hanging(make_submission, monkeypatch):
    monkeypatch.setattr(PLUGIN, "predict_timeout_s", lambda spec: 0.5, raising=False)
    result = run(
        make_submission(
            """
            import time

            class Predictor:
                def __init__(self, config): pass
                def predict(self, X):
                    time.sleep(30)
            """
        )
    )
    assert result.status == "predict_timeout"


def test_student_print_does_not_break_the_protocol(make_submission):
    """นิสิต `print()` ระหว่างทำนายได้ตามปกติ และ log ต้องถูกเก็บไว้ให้เขาดู"""
    result = run(
        make_submission(
            """
            import os
            import numpy as np

            class Predictor:
                def __init__(self, config):
                    print("โหลดโมเดลแล้ว")
                def predict(self, X):
                    os.write(1, b"raw write to fd 1\\n")
                    return np.zeros(len(X), dtype="int64")
            """
        )
    )
    assert result.ok, f"{result.status}: {result.detail}"
    assert "โหลดโมเดลแล้ว" in result.log
    assert "raw write to fd 1" in result.log


def test_predictor_runs_with_its_own_folder_as_cwd(make_submission):
    """starter kit สอนให้เขียน `joblib.load("pipeline.pkl")` แบบ path สัมพัทธ์"""
    submission = make_submission(
        """
        import numpy as np

        class Predictor:
            def __init__(self, config):
                self.n = int(open("weights.txt", encoding="utf-8").read())
            def predict(self, X):
                return np.full(len(X), self.n, dtype="int64")
        """
    )
    (submission / "weights.txt").write_text("1", encoding="utf-8")
    result = run(submission)
    assert result.ok, f"{result.status}: {result.detail}"


def test_helper_modules_next_to_predictor_can_be_imported(make_submission):
    submission = make_submission(
        """
        import numpy as np
        from helper import LABEL

        class Predictor:
            def __init__(self, config): pass
            def predict(self, X):
                return np.full(len(X), LABEL, dtype="int64")
        """
    )
    (submission / "helper.py").write_text("LABEL = 1\n", encoding="utf-8")
    result = run(submission)
    assert result.ok, f"{result.status}: {result.detail}"


# ── ความทำซ้ำได้ของตัวตรวจเอง ──────────────────────────────────────


def test_the_checks_use_the_same_rows_every_time(make_submission):
    """การตรวจที่ไม่ผ่านต้องเกิดซ้ำได้ — ผู้สอนกับนิสิตต้องคุยกันบนชุดเดียวกัน"""
    body = """
        import numpy as np

        class Predictor:
            def __init__(self, config): pass
            def predict(self, X):
                return (np.arange(len(X)) % 2).astype("int64")
    """
    first = run(make_submission(body)).detail
    second = run(make_submission(body)).detail
    assert first == second


def test_the_tiny_subset_is_what_catches_a_mild_batch_leak(make_submission, monkeypatch):
    """พิสูจน์ว่าก้อนจิ๋วเป็นตัวจับจริง ไม่ใช่ของประดับ

    predictor ตัวเดียวกับข้อ `batch_dependent` ข้างบน · ปิดก้อนจิ๋วแล้วมันรอด
    เพราะค่าเฉลี่ยของ subset 30% บังเอิญใกล้ค่าเฉลี่ยของชุดเต็มพอที่ไม่มีแถวไหน
    ตกคนละฝั่ง — นี่คือเหตุการณ์จริงที่ทำให้ `TINY_SUBSET_ROWS` ถูกเพิ่มเข้ามา
    """
    from runners.prediction import runner as runner_mod

    monkeypatch.setattr(runner_mod, "TINY_SUBSET_ROWS", 10**9)  # ปิดก้อนจิ๋ว
    result = run(
        make_submission(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config): pass
                def predict(self, X):
                    return (X["num"].to_numpy() > X["num"].mean()).astype("int64")
            """
        )
    )
    assert result.ok, (
        "ถ้าข้อนี้เริ่มล้ม แปลว่า subset 30% จับได้เองแล้ว — ตรวจว่ามีอะไรเปลี่ยน "
        f"(ขนาดชุด · เมล็ด · สัดส่วน) ก่อนจะสรุปว่าก้อนจิ๋วไม่จำเป็น · ได้ {result.status}"
    )
