"""ทดสอบ submission validation — environment-spec §13

เกณฑ์ที่ §13 ตั้งไว้คือ **error ต้องบอกวิธีแก้** ไม่ใช่แค่บอกว่าผิด
ภาระ support ที่ใหญ่ที่สุดของโจทย์แบบนี้คือ "ส่งแล้วพัง แต่ไม่รู้ว่าพังเพราะอะไร"
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from runners.agent_env.validate import (
    check_import_whitelist,
    inspect_archive,
    smoke_test,
)
from runners.tests.conftest import CONFIGS

MAIN = CONFIGS / "main.yaml"

GOOD_AGENT = """
from vacuum.baselines import BASELINES  # noqa: F401

class Agent:
    def __init__(self, config):
        self._inner = BASELINES["silver"](config)
    def reset(self, episode_info):
        self._inner.reset(episode_info)
    def act(self, observation):
        return self._inner.act(observation)
"""


def make_zip(tmp_path: Path, files: dict[str, str | bytes], name="sub.zip") -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        for filename, content in files.items():
            zf.writestr(filename, content)
    return path


# ── ชั้น static ─────────────────────────────────────────────────────


def test_good_archive_passes(tmp_path):
    report = inspect_archive(make_zip(tmp_path, {"agent.py": GOOD_AGENT}))
    assert report.ok, str(report)
    assert "vacuum" in report.imports


def test_missing_agent_py(tmp_path):
    report = inspect_archive(make_zip(tmp_path, {"model.pt": b"weights"}))
    assert not report.ok
    assert report.problems[0].code == "missing_agent_py"
    assert "ระดับบนสุด" in report.problems[0].fix


def test_missing_agent_class(tmp_path):
    report = inspect_archive(make_zip(tmp_path, {"agent.py": "class MyBot:\n    pass\n"}))
    assert [p.code for p in report.problems] == ["missing_agent_class"]
    assert "Agent" in report.problems[0].fix


def test_missing_methods_lists_them(tmp_path):
    report = inspect_archive(
        make_zip(tmp_path, {"agent.py": "class Agent:\n    def act(self, o):\n        return 4\n"})
    )
    assert report.problems[0].code == "missing_methods"
    assert "reset" in report.problems[0].message and "__init__" in report.problems[0].message


def test_syntax_error_reports_line(tmp_path):
    report = inspect_archive(make_zip(tmp_path, {"agent.py": "class Agent:\n  def act(self)\n"}))
    assert report.problems[0].code == "syntax_error"
    assert "บรรทัด" in report.problems[0].message


def test_path_traversal_is_rejected(tmp_path):
    report = inspect_archive(make_zip(tmp_path, {"../evil.py": "x=1", "agent.py": GOOD_AGENT}))
    assert any(p.code == "unsafe_path" for p in report.problems)


def test_import_whitelist(tmp_path):
    report = inspect_archive(
        make_zip(tmp_path, {"agent.py": "import requests\nimport os\n" + GOOD_AGENT})
    )
    check_import_whitelist(report)
    codes = [p.code for p in report.problems]
    assert "import_not_allowed" in codes
    offending = [p for p in report.problems if p.code == "import_not_allowed"]
    assert len(offending) == 1, "os เป็น stdlib ต้องไม่ถูกฟ้อง"
    assert "requests" in offending[0].message


def test_static_checks_never_execute_submission_code(tmp_path):
    """ชั้น static รันบน cloud API — ห้ามรันโค้ดที่อัพโหลดมาเด็ดขาด"""
    marker = tmp_path / "executed"
    evil = f"""
    from pathlib import Path
    Path({str(marker)!r}).write_text("boom")

    class Agent:
        def __init__(self, config): pass
        def reset(self, episode_info): pass
        def act(self, observation): return 4
    """
    import textwrap

    inspect_archive(make_zip(tmp_path, {"agent.py": textwrap.dedent(evil)}))
    assert not marker.exists(), "inspect_archive รันโค้ดนิสิต — ผิดหลักการของชั้น static"


# ── ชั้น dynamic ────────────────────────────────────────────────────


def test_smoke_test_passes_for_good_agent(baseline_submission):
    assert smoke_test(
        env_plugin="vacuum.arena:PLUGIN",
        config_path=MAIN,
        submission_dir=baseline_submission("silver"),
    ).ok


def test_smoke_test_catches_state_leak(make_submission):
    """agent ที่ไม่ล้าง state ใน reset() ต้องถูกจับได้

    อาการจริงคือคะแนนขึ้นกับลำดับที่ระบบสุ่ม episode มาให้ ไม่ใช่ฝีมือ
    ซึ่งถ้าไม่ตรวจจะกลายเป็นความไม่ยุติธรรมที่มองไม่เห็น
    """
    sub = make_submission(
        """
        class Agent:
            def __init__(self, config):
                self.total_steps = 0   # ไม่ถูกล้างใน reset — state รั่วข้าม episode

            def reset(self, episode_info):
                pass

            def act(self, observation):
                self.total_steps += 1
                # ทำงาน 30 step แรกแล้วเลิก — episode ที่สองในกระบวนการเดียวกัน
                # จะไม่ได้ทำอะไรเลยเพราะตัวนับเกิน 30 ไปแล้ว
                return 4 if self.total_steps <= 30 else 5
        """
    )
    result = smoke_test(
        env_plugin="vacuum.arena:PLUGIN", config_path=MAIN, submission_dir=sub
    )
    assert not result.ok
    assert result.problems[0].code == "state_leak"
    assert "reset()" in result.problems[0].fix


def test_smoke_test_reports_crash_with_traceback(make_submission):
    sub = make_submission(
        """
        class Agent:
            def __init__(self, config): pass
            def reset(self, episode_info): pass
            def act(self, observation):
                raise KeyError("ลืมใส่ weights")
        """
    )
    result = smoke_test(
        env_plugin="vacuum.arena:PLUGIN", config_path=MAIN, submission_dir=sub
    )
    assert not result.ok
    assert "ลืมใส่ weights" in result.detail


@pytest.mark.parametrize("bad_action", ["'up'", "99", "-1", "None"])
def test_smoke_test_catches_bad_actions(make_submission, bad_action):
    sub = make_submission(
        f"""
        class Agent:
            def __init__(self, config): pass
            def reset(self, episode_info): pass
            def act(self, observation): return {bad_action}
        """
    )
    result = smoke_test(
        env_plugin="vacuum.arena:PLUGIN", config_path=MAIN, submission_dir=sub
    )
    assert not result.ok
    assert result.problems[0].code in ("invalid_action", "agent_error")


def test_smoke_test_is_fast(baseline_submission):
    """§13 กำหนดว่าการตรวจต้องเร็ว — smoke test รัน 2 รอบ × 2 episode สั้นๆ"""
    import time

    t0 = time.perf_counter()
    smoke_test(
        env_plugin="vacuum.arena:PLUGIN",
        config_path=MAIN,
        submission_dir=baseline_submission("silver"),
    )
    assert time.perf_counter() - t0 < 15.0


# ── การตรวจต้องยอมรับเท่าที่ตอนรันรับได้จริง ────────────────────────
# ถ้าการตรวจยอมกว้างกว่า submission จะผ่าน **กินโควตาไปหนึ่งครั้ง** แล้วค่อยไปตาย
# ตอนรันด้วย `agent_init_failed` ซึ่งคือการเสียโควตาให้ความผิดพลาดที่บอกได้ตั้งแต่แรก
# เคสที่พบบ่อยคือรัน `arena submit` จากโฟลเดอร์แม่แทนที่จะ cd เข้าโฟลเดอร์งาน

AGENT_SRC = (
    "class Agent:\n"
    "    def __init__(self, config): pass\n"
    "    def reset(self, info): pass\n"
    "    def act(self, obs): return 0\n"
)


def _zip(tmp_path: Path, entries: dict[str, str]) -> Path:
    path = tmp_path / "submission.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, body in entries.items():
            zf.writestr(name, body)
    return path


@pytest.mark.parametrize(
    "label,entries,ok,code",
    [
        ("ราก zip", {"agent.py": AGENT_SRC}, True, None),
        ("ซ้อนหนึ่งชั้น (zip ทั้งโฟลเดอร์มา)", {"my-agent/agent.py": AGENT_SRC}, True, None),
        (
            "ซ้อนหนึ่งชั้น + ไฟล์อื่นที่ราก",
            {"my-agent/agent.py": AGENT_SRC, "notes.txt": "x"},
            True,
            None,
        ),
        (
            "สองโฟลเดอร์งาน — ตอนรันเลือกไม่ได้",
            {"my-agent/agent.py": AGENT_SRC, "my-agent-v2/agent.py": AGENT_SRC},
            False,
            "ambiguous_agent_py",
        ),
        ("ซ้อนสองชั้น — ตอนรันหาไม่เจอ", {"a/b/agent.py": AGENT_SRC}, False, "missing_agent_py"),
        ("ไม่มีเลย", {"README.md": "x"}, False, "missing_agent_py"),
    ],
)
def test_archive_is_accepted_exactly_when_it_can_actually_run(
    tmp_path, label, entries, ok, code
):
    report = inspect_archive(_zip(tmp_path, entries))
    assert report.ok is ok, f"{label}: {report}"
    if code:
        assert code in {p.code for p in report.problems}


def test_ambiguous_archive_tells_the_student_what_to_do(tmp_path):
    """§13 — error ต้องบอกวิธีแก้ ไม่ใช่แค่บอกว่าผิด"""
    report = inspect_archive(
        _zip(tmp_path, {"my-agent/agent.py": AGENT_SRC, "old/agent.py": AGENT_SRC})
    )
    problem = next(p for p in report.problems if p.code == "ambiguous_agent_py")
    assert "my-agent" in problem.message and "old" in problem.message  # บอกว่าเจอที่ไหนบ้าง
    assert "--dir" in problem.fix or "cd" in problem.fix  # บอกวิธีแก้


def test_validation_agrees_with_what_extract_resolves(tmp_path):
    """ผูกสองฝั่งไว้ด้วยกัน — ถ้าใครแก้ข้างเดียวเทสต์นี้จะจับได้

    นี่คือสัญญาที่เทสต์ข้างบนตั้งอยู่บนมัน: `inspect_archive` ต้องยอมรับ
    เท่าที่ `ArtifactStore.extract` หา `agent.py` เจอเท่านั้น
    """
    from core.store import ArtifactStore

    for entries in (
        {"agent.py": AGENT_SRC},
        {"my-agent/agent.py": AGENT_SRC},
        {"my-agent/agent.py": AGENT_SRC, "my-agent-v2/agent.py": AGENT_SRC},
        {"a/b/agent.py": AGENT_SRC},
    ):
        archive = _zip(tmp_path, entries)
        report = inspect_archive(archive)
        resolved = ArtifactStore(tmp_path / "store").extract(str(archive), tmp_path / "work")
        runnable = (resolved / "agent.py").is_file()
        assert report.ok is runnable, f"{sorted(entries)} — ตรวจว่า {report.ok} แต่รันได้ {runnable}"
        for path in (tmp_path / "work",):
            if path.exists():
                __import__("shutil").rmtree(path)
