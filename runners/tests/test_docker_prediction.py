"""พิสูจน์ว่า sandbox ของโจทย์ทำนายปิดจริง — ไม่ใช่แค่ใส่ธงไว้ใน command line

    docker build -f runners/prediction/images/Dockerfile.cpu -t arena/tabular:cpu .
    pytest runners/tests/test_docker_prediction.py

**สิ่งที่ตรวจได้เฉพาะที่นี่** คือข้อที่เป็นสมบัติของ *image* ไม่ใช่ของ runner —
`SubprocessLauncher` ใช้ interpreter ตัวเดียวกับ runner จึงเห็นทุกอย่างที่ runner เห็น
เทสต์ leakage ที่เหลืออยู่ที่ `test_prediction_runner.py` และ `test_prediction_cp462.py`

ของลับของโจทย์นี้ไม่ใช่ seed แต่เป็น **เฉลยของชุดที่ใช้ตัดสิน** — ซึ่งอยู่ใน
`tabular.dataset` ที่ตั้งใจไม่ให้เข้า image เลย
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from runners.prediction.runner import run_submission
from runners.sandbox.launcher import DockerLauncher

pytest.importorskip("tabular", reason="ต้องติดตั้ง envs/cp462-tabular ก่อน")

IMAGE = "arena/tabular:cpu"

pytestmark = pytest.mark.docker


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


requires_docker = pytest.mark.skipif(
    not _docker_ready(),
    reason=f"ต้องมี Docker และ image {IMAGE} — build ด้วย Dockerfile.cpu ก่อน",
)


@pytest.fixture
def churn(tabular_tasks):
    """config ของโจทย์ churn ที่ชี้ไปหาไฟล์ในคลังชั่วคราว — นิยามที่ `conftest.py`"""
    return tabular_tasks["churn"][0]


@pytest.fixture
def predictor(tmp_path):
    """เขียน submission ลง**โฟลเดอร์ย่อย** ไม่ใช่ `tmp_path` ตรงๆ

    `tmp_path` ของ pytest เป็น `drwx------` — container รันด้วย uid 10001 จึง
    `chdir` เข้าไปไม่ได้ แล้วทุกข้อในไฟล์นี้ล้มด้วย `PermissionError: '/submission'`
    ซึ่งไม่ใช่บั๊กของ sandbox · ของจริงโฟลเดอร์นี้มาจาก `ArtifactStore.extract`
    ที่เปิดสิทธิ์ให้ผู้ใช้อื่นอ่านได้เสมอ เทสต์จึงต้องจำลองสภาพเดียวกัน
    """

    def build(body: str) -> Path:
        directory = tmp_path / "submission"
        directory.mkdir(exist_ok=True)
        (directory / "predictor.py").write_text(textwrap.dedent(body), encoding="utf-8")
        directory.chmod(0o755)
        (directory / "predictor.py").chmod(0o644)
        return directory

    return build


def _run(submission, churn, **kwargs):
    return run_submission(
        env_plugin="tabular.arena:PLUGIN",
        config_path=churn,
        submission_dir=submission,
        launcher=DockerLauncher(image=IMAGE),
        **kwargs,
    )


ZEROS = """
    import numpy as np

    class Predictor:
        def __init__(self, config): pass
        def predict(self, X):
            return np.zeros(len(X), dtype="int64")
"""


# ── กล่องต้องใช้งานได้จริงก่อน ─────────────────────────────────────


@requires_docker
def test_scoring_works_end_to_end(predictor, churn):
    """ถ้าข้อนี้ล้ม ข้ออื่นในไฟล์นี้ไม่มีความหมาย — image ต้องรันได้ก่อน"""
    result = _run(predictor(ZEROS), churn)
    assert result.ok, f"{result.status}: {result.detail}\n{result.log[-2000:]}"
    assert result.checks["determinism"] is True
    assert result.n_rows > 0


@requires_docker
def test_the_libraries_students_need_are_in_the_image(predictor, churn):
    """นิสิตต้อง `joblib.load` โมเดล sklearn ได้ — ไม่มีเน็ตให้ติดตั้งเพิ่มตอนรัน"""
    result = _run(churn=churn, submission=predictor(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config):
                    import joblib, pandas, sklearn, scipy  # noqa: F401
                def predict(self, X):
                    return np.zeros(len(X), dtype="int64")
            """
        )
    )
    assert result.ok, f"{result.status}: {result.detail}\n{result.log[-2000:]}"


# ── สิ่งที่ต้องเข้าถึงไม่ได้จากในกล่อง ──────────────────────────────


@requires_docker
def test_the_answers_package_is_not_in_the_image(predictor, churn):
    """**ข้อสำคัญที่สุดของไฟล์นี้** — `tabular` เห็นเฉลย จึงต้องไม่อยู่ใน image

    ถ้ามันหลุดเข้าไป โค้ดนิสิตเรียก `tabular.dataset.grading_data(spec, "private")`
    ได้ตรงๆ แล้วการแข่งจบทันทีโดยไม่มีใครรู้
    """
    result = _run(churn=churn, submission=predictor(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config):
                    for name in ("tabular", "tabular.dataset", "tabular.metrics"):
                        try:
                            __import__(name)
                        except ImportError:
                            continue
                        raise AssertionError(f"import {name} ได้จากในกล่อง")
                def predict(self, X):
                    return np.zeros(len(X), dtype="int64")
            """
        )
    )
    assert result.ok, f"{result.status}: {result.detail}"


@requires_docker
def test_the_trusted_runner_is_not_in_the_image(predictor, churn):
    result = _run(churn=churn, submission=predictor(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config):
                    for name in ("runners.prediction.runner", "runners.prediction.plugin",
                                 "runners.sandbox.launcher"):
                        try:
                            __import__(name)
                        except ImportError:
                            continue
                        raise AssertionError(f"import {name} ได้จากในกล่อง")
                def predict(self, X):
                    return np.zeros(len(X), dtype="int64")
            """
        )
    )
    assert result.ok, f"{result.status}: {result.detail}"


@requires_docker
def test_no_network(predictor, churn):
    result = _run(churn=churn, submission=predictor(
            """
            import socket
            import numpy as np

            class Predictor:
                def __init__(self, config):
                    socket.setdefaulttimeout(5)
                    try:
                        socket.create_connection(("1.1.1.1", 53), timeout=5)
                    except OSError:
                        return
                    raise AssertionError("ต่อเน็ตออกไปได้ — --network none ไม่ทำงาน")
                def predict(self, X):
                    return np.zeros(len(X), dtype="int64")
            """
        )
    )
    assert result.ok, f"{result.status}: {result.detail}"


@requires_docker
def test_not_root_and_rootfs_is_read_only(predictor, churn):
    result = _run(churn=churn, submission=predictor(
            """
            import os
            import numpy as np

            class Predictor:
                def __init__(self, config):
                    assert os.geteuid() != 0, "รันเป็น root"
                    try:
                        open("/opt/arena/ฝังไว้.txt", "w").close()
                    except OSError:
                        pass
                    else:
                        raise AssertionError("เขียน rootfs ได้")
                def predict(self, X):
                    return np.zeros(len(X), dtype="int64")
            """
        )
    )
    assert result.ok, f"{result.status}: {result.detail}"


@requires_docker
def test_the_submission_mount_is_read_only(predictor, churn):
    """แก้ไฟล์ตัวเองระหว่างรันไม่ได้ — ไม่งั้นการ rejudge จะไม่ตรงกับที่ส่งมา"""
    result = _run(churn=churn, submission=predictor(
            """
            import numpy as np

            class Predictor:
                def __init__(self, config):
                    try:
                        open("/submission/predictor.py", "a").close()
                    except OSError:
                        return
                    raise AssertionError("เขียนทับ submission ได้")
                def predict(self, X):
                    return np.zeros(len(X), dtype="int64")
            """
        )
    )
    assert result.ok, f"{result.status}: {result.detail}"


@requires_docker
def test_tmp_is_writable_because_sklearn_needs_it(predictor, churn):
    """ต้องมีที่เขียนบ้าง — joblib/sklearn เขียนไฟล์ชั่วคราว แต่ต้องไม่ลงดิสก์จริง"""
    result = _run(churn=churn, submission=predictor(
            """
            import tempfile
            import numpy as np

            class Predictor:
                def __init__(self, config):
                    with tempfile.NamedTemporaryFile(mode="w") as fh:
                        fh.write("ok")
                def predict(self, X):
                    return np.zeros(len(X), dtype="int64")
            """
        )
    )
    assert result.ok, f"{result.status}: {result.detail}"
