"""**การทดสอบที่สำคัญที่สุดของ runner**: รันแยก process แล้วต้องได้คะแนนตรงกับรันใน process เดียว

ถ้าสองเส้นทางนี้ให้ผลไม่ตรงกัน แปลว่าโปรโตคอลทำข้อมูลเพี้ยน (dtype หาย, ปัดเลข,
observation คลาดไปหนึ่ง step) ซึ่งจะทำให้คะแนนที่นิสิตเห็นตอน `arena eval --local`
ไม่ตรงกับคะแนนบน leaderboard — เป็นบั๊กที่ทำลายความเชื่อถือของทั้งระบบและหาสาเหตุยากมาก
"""

from __future__ import annotations

import pytest

from runners.agent_env.runner import run_submission
from runners.tests.conftest import CONFIGS
from vacuum import load_config
from vacuum.baselines import BASELINES
from vacuum.rollout import agent_config, evaluate

SEEDS = [70001, 70002, 70003]


@pytest.mark.parametrize("level", list(BASELINES))
@pytest.mark.parametrize("phase", ["warmup", "main"])
def test_out_of_process_matches_in_process(level, phase, baseline_submission):
    config = load_config(CONFIGS / f"{phase}.yaml")
    expected, _ = evaluate(config, BASELINES[level], SEEDS)

    result = run_submission(
        env_plugin="vacuum.arena:PLUGIN",
        config_path=CONFIGS / f"{phase}.yaml",
        submission_dir=baseline_submission(level),
        seeds=SEEDS,
    )

    assert result.ok, f"{result.status}: {result.detail}\n{result.log}"
    assert [e.status for e in result.episodes] == ["ok"] * len(SEEDS)
    assert result.summary.score == pytest.approx(expected.score, abs=1e-12), (
        f"{phase}/{level}: คะแนนข้าม process ไม่ตรงกัน "
        f"({result.summary.score} vs {expected.score})"
    )
    assert result.summary.n_completed == expected.n_completed
    for got, want in zip(result.episodes, expected.per_episode):
        assert got.breakdown.score == pytest.approx(want.score, abs=1e-12)
        assert got.breakdown.t_end == want.t_end
        assert got.breakdown.collisions == want.collisions
        assert got.breakdown.redundant_sucks == want.redundant_sucks


def test_observation_dtype_survives_the_wire(make_submission):
    """dtype ของ observation ต้องข้าม process มาแบบไม่เปลี่ยน

    ถ้ากลายเป็น float64 ตอนรันจริงแต่เป็น float32 ตอนเทรน policy จะเจอ input
    ที่ต่างจากที่ฝึกมาแบบเงียบๆ — จับได้ยากเพราะคะแนนแค่ "แย่ลงเฉยๆ"
    """
    sub = make_submission(
        """
        import numpy as np

        class Agent:
            def __init__(self, config):
                self.problems = []

            def reset(self, episode_info):
                pass

            def act(self, observation):
                for key in ("grid", "pos", "scalars"):
                    arr = observation[key]
                    if not isinstance(arr, np.ndarray):
                        raise TypeError(f"{key} ไม่ใช่ ndarray: {type(arr)}")
                    if arr.dtype != np.float32:
                        raise TypeError(f"{key} dtype = {arr.dtype} ไม่ใช่ float32")
                if observation["grid"].ndim != 3:
                    raise ValueError("grid ต้องเป็น 3 มิติในโหมด local")
                return 4
        """
    )
    result = run_submission(
        env_plugin="vacuum.arena:PLUGIN",
        config_path=CONFIGS / "main.yaml",
        submission_dir=sub,
        seeds=[70001],
    )
    assert result.ok, f"{result.status}: {result.detail}\n{result.log}"
    assert result.episodes[0].status == "ok"


def test_replay_is_written(tmp_path, baseline_submission):
    result = run_submission(
        env_plugin="vacuum.arena:PLUGIN",
        config_path=CONFIGS / "main.yaml",
        submission_dir=baseline_submission("gold"),
        seeds=[70001, 70002],
        replay_dir=tmp_path / "replays",
    )
    assert result.ok
    files = sorted(p.name for p in (tmp_path / "replays").glob("*.vrp"))
    assert files == ["70001.vrp", "70002.vrp"]
    assert all(e.replay_bytes > 0 for e in result.episodes)
    # งบตาม README §10.3 — ~1–2 KB ต่อ episode หลังบีบอัด
    assert max(e.replay_bytes for e in result.episodes) < 4096
