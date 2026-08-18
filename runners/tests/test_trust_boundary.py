"""ทดสอบว่า trust boundary เป็นจริง และ agent ที่พังไม่ทำให้ทั้ง run ล่ม

[README §10.4](../../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries) บอกว่าถ้า agent
อยู่ใน process เดียวกับ environment นิสิตเขียนแค่ `import gc` แล้วไล่หา object ของ env
ก็อ่านผังห้องทั้งใบและค่า seed ได้ทันที → เล่นได้สมบูรณ์แบบโดยไม่ต้องเรียนรู้อะไรเลย

ไฟล์นี้พิสูจน์ว่าช่องนั้นถูกปิดจริง ไม่ใช่แค่เขียนไว้ในเอกสาร
"""

from __future__ import annotations

import pytest

from runners.agent_env.runner import run_submission
from runners.tests.conftest import CONFIGS

MAIN = CONFIGS / "main.yaml"


def _run(submission, **kwargs):
    return run_submission(
        env_plugin="vacuum.arena:PLUGIN",
        config_path=MAIN,
        submission_dir=submission,
        seeds=kwargs.pop("seeds", [70001]),
        **kwargs,
    )


# ── trust boundary ──────────────────────────────────────────────────


def test_agent_cannot_reach_the_environment(make_submission):
    """ไล่หา object ของ env ผ่าน gc แล้วต้องไม่เจอ — นี่คือช่องที่ §10.4 ปิดไว้"""
    sub = make_submission(
        """
        import gc

        class Agent:
            def __init__(self, config):
                pass

            def reset(self, episode_info):
                pass

            def act(self, observation):
                for obj in gc.get_objects():
                    if type(obj).__name__ == "VacuumEnv":
                        raise AssertionError("เอื้อมถึง VacuumEnv ได้ — trust boundary พัง")
                    if type(obj).__name__ == "Layout":
                        raise AssertionError("เอื้อมถึงผังห้องได้ — trust boundary พัง")
                return 4
        """
    )
    result = _run(sub)
    assert result.ok, f"{result.status}: {result.detail}\n{result.log}"
    assert result.episodes[0].status == "ok"


def test_agent_never_receives_seed_or_layout(make_submission):
    """ตรวจ **รายการคีย์ที่อนุญาตแบบเป๊ะ** ไม่ใช่แค่ blacklist คำต้องห้าม

    blacklist พลาดสองทาง — ฟิลด์ลับที่ตั้งชื่อใหม่จะรอด และค่า config ที่ประกาศ
    ต่อสาธารณะอยู่แล้ว (`sticky_dirt`, `sensor_noise` — ซึ่ง agent ที่กรอง noise **ต้องใช้**)
    จะถูกฟ้องผิด · การล็อกรายการคีย์ทำให้ฟิลด์ใหม่ที่รั่วต้องผ่านการแก้เทสต์นี้ก่อนเสมอ
    """
    sub = make_submission(
        """
        ALLOWED_CONFIG = {
            "width", "height", "observation", "observation_window",
            "max_steps", "action_noise", "sticky_dirt", "sensor_noise",
        }
        ALLOWED_OBS = {"grid", "pos", "scalars"}

        class Agent:
            def __init__(self, config):
                extra = set(config) - ALLOWED_CONFIG
                if extra:
                    raise AssertionError(f"agent_config มีฟิลด์เกิน: {sorted(extra)}")

            def reset(self, episode_info):
                extra = set(episode_info) - ALLOWED_CONFIG
                if extra:
                    raise AssertionError(f"episode_info มีฟิลด์เกิน: {sorted(extra)}")

            def act(self, observation):
                extra = set(observation) - ALLOWED_OBS
                if extra:
                    raise AssertionError(f"observation มีฟิลด์เกิน: {sorted(extra)}")
                # scalars ต้องมีแค่ 2 ค่า (t, battery) — ไม่มี coverage (env-spec §4)
                if len(observation["scalars"]) != 2:
                    raise AssertionError("scalars ยาวเกิน 2 — อาจมีข้อมูลที่ไม่ควรให้")
                return 4
        """
    )
    result = _run(sub)
    assert result.ok, f"{result.status}: {result.detail}\n{result.log}"


# ── agent ที่พัง ────────────────────────────────────────────────────


def test_exception_in_act_fails_only_that_episode(make_submission):
    """`act()` โยน exception → episode นั้นได้ 0 · run เดินต่อ (template §7.3)"""
    sub = make_submission(
        """
        class Agent:
            def __init__(self, config):
                self.episode = -1

            def reset(self, episode_info):
                self.episode += 1
                self.t = 0

            def act(self, observation):
                self.t += 1
                if self.episode == 0 and self.t == 5:
                    raise RuntimeError("พังตั้งใจ")
                return 4
        """
    )
    result = _run(sub, seeds=[70001, 70002])
    assert result.ok, f"{result.status}: {result.detail}"
    assert result.episodes[0].status == "agent_error"
    assert result.episodes[0].breakdown.score == 0.0
    assert "พังตั้งใจ" in result.episodes[0].detail
    assert result.episodes[1].status == "ok", "episode ถัดไปต้องยังรันได้"


def test_timeout_fails_the_episode_not_the_score(make_submission):
    """เกิน `step_timeout_ms` → **ล้มเหลว ไม่ใช่ได้คะแนนน้อยลง**

    wall-clock ไม่มีผลต่อคะแนนตามหลัก hardware-independent scoring — มันเป็นแค่ตัวกันงานค้าง
    """
    sub = make_submission(
        """
        import time

        class Agent:
            def __init__(self, config):
                self.t = 0

            def reset(self, episode_info):
                self.t = 0

            def act(self, observation):
                self.t += 1
                if self.t == 3:
                    time.sleep(5)
                return 4
        """
    )
    result = _run(sub, config_overrides={"episode.step_timeout_ms": 300})
    assert result.episodes[0].status == "agent_timeout"
    assert result.episodes[0].breakdown.score == 0.0


def test_invalid_action_is_rejected(make_submission):
    sub = make_submission(
        """
        class Agent:
            def __init__(self, config): pass
            def reset(self, episode_info): pass
            def act(self, observation): return 99
        """
    )
    result = _run(sub)
    assert result.episodes[0].status == "invalid_action"
    assert "[0, 5]" in result.episodes[0].detail


def test_crash_in_init_fails_the_whole_run(make_submission):
    """ล้มตอนสร้าง Agent = submission ใช้ไม่ได้ทั้งอัน ไม่ใช่แค่ episode เดียว"""
    sub = make_submission(
        """
        class Agent:
            def __init__(self, config):
                raise ValueError("โหลด weights ไม่ได้")
            def reset(self, episode_info): pass
            def act(self, observation): return 4
        """
    )
    result = _run(sub)
    assert result.status == "agent_init_failed"
    assert "โหลด weights ไม่ได้" in result.detail


def test_missing_agent_py_is_reported_clearly(tmp_path):
    (tmp_path / "empty").mkdir()
    result = _run(tmp_path / "empty")
    assert result.status == "agent_init_failed"
    assert "agent.py" in result.detail


# ── สิ่งที่นิสิตทำแล้วต้องไม่พัง ──────────────────────────────────────


def test_student_print_does_not_break_the_protocol(make_submission):
    """นิสิต `print()` เยอะแค่ไหนก็ต้องไม่ทำ stream ของโปรโตคอลพัง

    นี่คือเหตุผลที่ agent host ย้าย fd 1 ไปที่ stderr ก่อนโหลด agent.py
    ถ้าไม่ทำ การ print ตัวเดียวจะทำให้ทั้ง run พังแบบที่อธิบายไม่ได้
    """
    sub = make_submission(
        """
        import os, sys

        class Agent:
            def __init__(self, config):
                print("สร้าง agent")
                os.write(1, b"raw write to fd 1\\n")

            def reset(self, episode_info):
                print("reset")

            def act(self, observation):
                print("กำลังคิด", observation["pos"])
                sys.stdout.write("เขียนผ่าน sys.stdout\\n")
                return 4
        """
    )
    result = _run(sub)
    assert result.ok, f"{result.status}: {result.detail}"
    assert result.episodes[0].status == "ok"
    assert "กำลังคิด" in result.log, "log ของนิสิตต้องถูกเก็บไว้ให้เขาดู"
    assert "raw write to fd 1" in result.log, "เขียนลง fd 1 ตรงๆ ก็ต้องไปออก stderr"


def test_log_is_capped(make_submission):
    """log ที่ใหญ่เกินต้องถูกตัด ไม่ใช่กิน RAM ของ runner หรือทำ pipe เต็มจนค้าง"""
    sub = make_submission(
        """
        class Agent:
            def __init__(self, config): pass
            def reset(self, episode_info): pass
            def act(self, observation):
                print("x" * 4000)
                return 4
        """
    )
    result = _run(sub, config_overrides={"episode.max_steps": 1500})
    assert result.ok
    assert len(result.log.encode()) <= 1024 * 1024 + 200
    assert "ถูกตัดที่ 1 MB" in result.log


@pytest.mark.parametrize("seeds", [[70001], [70001, 70002, 70003]])
def test_reset_is_called_once_per_episode(make_submission, seeds):
    sub = make_submission(
        """
        class Agent:
            def __init__(self, config):
                self.resets = 0
            def reset(self, episode_info):
                self.resets += 1
            def act(self, observation):
                if self.resets < 1:
                    raise AssertionError("act ถูกเรียกก่อน reset")
                return 4
        """
    )
    result = _run(sub, seeds=seeds)
    assert result.ok
    assert all(e.status == "ok" for e in result.episodes)
