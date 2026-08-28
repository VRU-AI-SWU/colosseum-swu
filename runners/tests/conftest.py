from __future__ import annotations

import os

import textwrap
from pathlib import Path

import pytest

import vacuum

REPO = Path(__file__).resolve().parents[2]
CONFIGS = Path(vacuum.__file__).resolve().parent / "configs"


def write_submission(directory: Path, body: str) -> Path:
    """สร้างโฟลเดอร์ submission ที่มี agent.py ตามที่ส่งเข้ามา"""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return directory


@pytest.fixture
def make_submission(tmp_path: Path):
    counter = {"n": 0}

    def _make(body: str) -> Path:
        counter["n"] += 1
        return write_submission(tmp_path / f"sub{counter['n']}", body)

    return _make


@pytest.fixture
def baseline_submission(make_submission):
    """submission ที่ห่อ baseline ตัวใดตัวหนึ่ง — ใช้เทียบผลกับการรันแบบ in-process"""

    def _make(level: str) -> Path:
        return make_submission(
            f"""
            from vacuum.baselines import BASELINES

            class Agent:
                def __init__(self, config):
                    self._inner = BASELINES[{level!r}](config)

                def reset(self, episode_info):
                    self._inner.reset(episode_info)

                def act(self, observation):
                    return self._inner.act(observation)
            """
        )

    return _make


# ── เมล็ดของชุดที่ใช้ตัดสินของ CP462 ────────────────────────────────
# เครื่องที่ไม่มี `ARENA_SECRETS` ต้องรันเทสต์ได้ — เปิดเมล็ดสำรองให้เฉพาะตอนนั้น
# ถ้ามีของจริงอยู่ก็ใช้ของจริง ซึ่งเป็นการตรวจที่แข็งแรงกว่า
if not os.environ.get("ARENA_SECRETS"):
    os.environ.setdefault("ARENA_CP462_ALLOW_SEED_FALLBACK", "1")
