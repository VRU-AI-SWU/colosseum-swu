"""การเลือก sandbox ของ `arena serve` — README §4.1

ก่อนหน้านี้ `arena serve` ใช้ `SubprocessLauncher` เสมอและไม่มีธงให้เลือก แปลว่า
ถ้าเปิดให้นิสิตส่งงานจริง โค้ดของนิสิตจะรันบนเครื่องที่มีเฉลยอยู่โดยไม่มี container ห่อ
ทั้งที่ `DockerLauncher` เขียนครบและผ่านเทสต์ 8 ข้อแล้ว
"""

from __future__ import annotations

from argparse import Namespace

import pytest

from core.cli import _pick_launcher
from runners.agent_env.launcher import DockerLauncher, SubprocessLauncher


class FakeDocker(DockerLauncher):
    ready = True

    @classmethod
    def available(cls, image: str = "", docker: str = "") -> bool:
        return cls.ready


def args(sandbox="auto", real_seeds=False) -> Namespace:
    return Namespace(sandbox=sandbox, real_seeds=real_seeds)


@pytest.fixture(autouse=True)
def _reset():
    FakeDocker.ready = True
    yield


def test_real_seeds_refuses_subprocess():
    """โหมดให้คะแนนจริงต้องไม่ยอมรันโค้ดนิสิตนอก container ไม่ว่าจะสั่งยังไง"""
    launcher, _ = _pick_launcher(args("subprocess", real_seeds=True), FakeDocker, SubprocessLauncher)
    assert launcher is None, "ต้องหยุด ไม่ใช่เตือนแล้วรันต่อ"


def test_real_seeds_requires_docker_to_be_ready():
    """ขอ docker แล้ว docker ไม่พร้อม = หยุด ไม่ใช่ถอยไป subprocess เงียบๆ"""
    FakeDocker.ready = False
    launcher, _ = _pick_launcher(args("auto", real_seeds=True), FakeDocker, SubprocessLauncher)
    assert launcher is None


def test_real_seeds_picks_docker_when_ready():
    launcher, note = _pick_launcher(args("auto", real_seeds=True), FakeDocker, SubprocessLauncher)
    assert isinstance(launcher, DockerLauncher)
    assert "network none" in note


def test_dev_falls_back_to_subprocess_when_docker_missing():
    """เครื่องที่ยังไม่ได้ build image ต้องพัฒนาต่อได้ — แค่ต้องบอกให้ชัดว่าไม่มี sandbox"""
    FakeDocker.ready = False
    launcher, note = _pick_launcher(args("auto"), FakeDocker, SubprocessLauncher)
    assert isinstance(launcher, SubprocessLauncher)
    assert "⚠️" in note


def test_dev_prefers_docker_when_available():
    launcher, _ = _pick_launcher(args("auto"), FakeDocker, SubprocessLauncher)
    assert isinstance(launcher, DockerLauncher)


def test_explicit_docker_fails_loudly_when_unavailable():
    FakeDocker.ready = False
    launcher, _ = _pick_launcher(args("docker"), FakeDocker, SubprocessLauncher)
    assert launcher is None
