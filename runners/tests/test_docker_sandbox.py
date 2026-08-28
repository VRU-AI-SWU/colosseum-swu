"""พิสูจน์ว่า sandbox ปิดจริง — ไม่ใช่แค่ใส่ธงไว้ใน command line

ทุกข้อในนี้ทดสอบโดย **ให้ agent พยายามทำสิ่งที่ห้ามจริงๆ** แล้วดูว่ามันทำไม่ได้
การอ่าน `docker run` แล้วเชื่อว่าธงทำงานเป็นคนละเรื่องกับการยืนยันว่ามันทำงาน

    docker build -f runners/agent_env/images/Dockerfile.cpu -t arena/vacuum:cpu .
    pytest runners/tests/test_docker_sandbox.py
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from runners.sandbox.launcher import DockerLauncher
from runners.agent_env.runner import run_submission
from runners.tests.conftest import CONFIGS

IMAGE = "arena/vacuum:cpu"
MAIN = CONFIGS / "main.yaml"

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


def _run(submission, seeds=(70001,), **kwargs):
    return run_submission(
        env_plugin="vacuum.arena:PLUGIN",
        config_path=MAIN,
        submission_dir=submission,
        seeds=list(seeds),
        launcher=DockerLauncher(image=IMAGE),
        **kwargs,
    )


# ── ต้องให้ผลเหมือนกันทุกประการกับการรันแบบอื่น ────────────────────


@requires_docker
def test_docker_scores_match_in_process(baseline_submission):
    """คะแนนจาก sandbox ต้องตรงกับที่นิสิตได้ในเครื่องตัวเอง

    ถ้าไม่ตรง นิสิตจะจูนบนตัวเลขชุดหนึ่งแล้วถูกตัดสินด้วยอีกชุด
    """
    from vacuum import load_config
    from vacuum.baselines import BASELINES
    from vacuum.rollout import evaluate

    seeds = [70001, 70002]
    expected, _ = evaluate(load_config(MAIN), BASELINES["silver"], seeds)
    result = _run(baseline_submission("silver"), seeds=seeds)

    assert result.ok, f"{result.status}: {result.detail}\n{result.log}"
    assert result.summary.score == pytest.approx(expected.score, abs=1e-12)


# ── สิ่งที่ sandbox ต้องปิด ─────────────────────────────────────────


@requires_docker
def test_no_network(make_submission):
    """`--network none` — ไม่มีทางส่งเฉลยออกไปหรือโหลดอะไรเข้ามา"""
    sub = make_submission(
        """
        import socket

        class Agent:
            def __init__(self, config):
                try:
                    socket.create_connection(("1.1.1.1", 53), timeout=3)
                except OSError:
                    self.blocked = True
                else:
                    raise AssertionError("ต่อเน็ตออกไปได้ — sandbox พัง")

            def reset(self, episode_info): pass
            def act(self, observation): return 4
        """
    )
    result = _run(sub)
    assert result.ok, f"{result.status}: {result.detail}\n{result.log}"


@requires_docker
def test_rootfs_is_read_only(make_submission):
    sub = make_submission(
        """
        from pathlib import Path

        class Agent:
            def __init__(self, config):
                for target in ("/opt/arena/pwned", "/usr/pwned", "/pwned"):
                    try:
                        Path(target).write_text("x")
                    except OSError:
                        continue
                    raise AssertionError(f"เขียน {target} ได้ — rootfs ไม่ได้เป็น read-only")

            def reset(self, episode_info): pass
            def act(self, observation): return 4
        """
    )
    result = _run(sub)
    assert result.ok, f"{result.status}: {result.detail}\n{result.log}"


@requires_docker
def test_submission_is_read_only(make_submission):
    """แก้ไฟล์ตัวเองระหว่างรันไม่ได้ — กันการฝัง state ข้าม run"""
    sub = make_submission(
        """
        from pathlib import Path

        class Agent:
            def __init__(self, config):
                try:
                    (Path("/submission") / "sneaky.txt").write_text("x")
                except OSError:
                    return
                raise AssertionError("เขียนลง /submission ได้ — mount ไม่ได้เป็น :ro")

            def reset(self, episode_info): pass
            def act(self, observation): return 4
        """
    )
    result = _run(sub)
    assert result.ok, f"{result.status}: {result.detail}\n{result.log}"


@requires_docker
def test_runs_as_non_root(make_submission):
    sub = make_submission(
        """
        import os

        class Agent:
            def __init__(self, config):
                if os.geteuid() == 0:
                    raise AssertionError("รันเป็น root — sandbox พัง")

            def reset(self, episode_info): pass
            def act(self, observation): return 4
        """
    )
    result = _run(sub)
    assert result.ok, f"{result.status}: {result.detail}\n{result.log}"


@requires_docker
def test_trusted_side_code_is_absent(make_submission):
    """ไฟล์ฝั่ง runner ต้องไม่อยู่ใน container

    ถ้าวันหนึ่งมีคนเผลอ COPY runner.py เข้าไป trust boundary จะพังแบบเงียบๆ
    เพราะ agent จะ import environment ขึ้นมาเองแล้วอ่าน state ได้
    """
    sub = make_submission(
        """
        import importlib

        class Agent:
            def __init__(self, config):
                for module in ("runners.agent_env.runner", "runners.agent_env.plugin"):
                    try:
                        importlib.import_module(module)
                    except ImportError:
                        continue
                    raise AssertionError(f"import {module} ได้ใน sandbox")

            def reset(self, episode_info): pass
            def act(self, observation): return 4
        """
    )
    result = _run(sub)
    assert result.ok, f"{result.status}: {result.detail}\n{result.log}"


# ── สิ่งที่ sandbox ต้อง *ไม่* ปิด ──────────────────────────────────


@requires_docker
def test_tmp_is_writable(make_submission):
    """ต้องมีที่เขียนบ้าง — numpy/torch ใช้ temp file และนิสิตอาจ cache อะไรระหว่าง episode"""
    sub = make_submission(
        """
        from pathlib import Path

        class Agent:
            def __init__(self, config):
                Path("/tmp/scratch.txt").write_text("ok")
                assert Path("/tmp/scratch.txt").read_text() == "ok"

            def reset(self, episode_info): pass
            def act(self, observation): return 4
        """
    )
    result = _run(sub)
    assert result.ok, f"{result.status}: {result.detail}\n{result.log}"


@requires_docker
def test_student_can_import_the_environment_package(make_submission):
    """นิสิตต้อง import `vacuum` ได้ เพราะ starter kit มี helper ที่เขาใช้เขียน agent"""
    sub = make_submission(
        """
        from vacuum.baselines.common import WorldModel

        class Agent:
            def __init__(self, config):
                self.model = WorldModel(
                    config["width"], config["height"],
                    config["observation"], config.get("observation_window"),
                )
            def reset(self, episode_info):
                self.model.reset()
            def act(self, observation):
                self.model.update(observation)
                return 4 if self.model.dirty_here() else 3
        """
    )
    result = _run(sub)
    assert result.ok, f"{result.status}: {result.detail}\n{result.log}"
