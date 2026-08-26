"""ส่วนของ CLI ที่มีตรรกะจริง — การแพ็กไฟล์และการอ่านช่วง seed

`pack()` สำคัญกว่าที่ดู: `models/` กับ `.venv/` เป็นสาเหตุที่พบบ่อยที่สุดที่ทำให้ zip
เกินเพดาน 95 MB โดยที่นิสิตไม่รู้ตัว — ถ้า CLI ไม่ข้ามให้ ภาระจะไปตกที่ error ตอนอัพโหลด
"""

from __future__ import annotations

import io
import zipfile

import pytest

from core.cli import _require_agent_dir, pack, parse_seeds


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("1-5", [1, 2, 3, 4, 5]),
        ("1,5,9", [1, 5, 9]),
        ("1-3,10", [1, 2, 3, 10]),
        ("20001", [20001]),
        (" 1 - 3 ", [1, 2, 3]),
    ],
)
def test_parse_seeds(spec, expected):
    assert parse_seeds(spec) == expected


def test_pack_skips_heavy_and_irrelevant_files(tmp_path):
    (tmp_path / "agent.py").write_text("class Agent: pass")
    (tmp_path / "weights.npz").write_bytes(b"w" * 100)
    for junk in ("models", ".venv", "__pycache__", ".git"):
        (tmp_path / junk).mkdir()
        (tmp_path / junk / "big.bin").write_bytes(b"x" * 10_000)

    with zipfile.ZipFile(__import__("io").BytesIO(pack(tmp_path))) as zf:
        names = set(zf.namelist())

    assert names == {"agent.py", "weights.npz"}


def test_pack_keeps_nested_student_modules(tmp_path):
    (tmp_path / "agent.py").write_text("class Agent: pass")
    (tmp_path / "mylib").mkdir()
    (tmp_path / "mylib" / "planner.py").write_text("x = 1")

    with zipfile.ZipFile(__import__("io").BytesIO(pack(tmp_path))) as zf:
        assert "mylib/planner.py" in zf.namelist()


# ── ยืนผิดโฟลเดอร์ ─────────────────────────────────────────────────
# `arena init` สร้าง *โฟลเดอร์ใหม่* แต่ `--dir` ของ eval/submit เริ่มต้นที่ "."
# คนที่รัน init แล้วรัน eval ต่อโดยไม่ `cd` เคยเจอ traceback ของ agent_host
# ซึ่งอ่านแล้วเหมือน agent ตัวเองพัง ทั้งที่แค่ยืนผิดที่


def test_missing_agent_py_says_so_instead_of_raising_a_traceback(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _require_agent_dir(str(tmp_path))
    assert "ไม่พบ agent.py" in str(exc.value)
    assert "arena init" in str(exc.value)


def test_error_points_at_the_nested_folder_when_there_is_one(tmp_path):
    """เคสที่พบบ่อยที่สุด — ยืนอยู่ข้างนอก `my-agent/` ที่เพิ่ง init มา"""
    (tmp_path / "my-agent").mkdir()
    (tmp_path / "my-agent" / "agent.py").write_text("class Agent: pass\n")
    with pytest.raises(SystemExit) as exc:
        _require_agent_dir(str(tmp_path))
    assert "cd my-agent" in str(exc.value)


def test_correct_folder_passes_and_returns_an_absolute_path(tmp_path):
    (tmp_path / "agent.py").write_text("class Agent: pass\n")
    got = _require_agent_dir(str(tmp_path))
    assert got.is_absolute()
    assert (got / "agent.py").is_file()


def test_zip_names_always_use_forward_slashes(tmp_path):
    """ชื่อใน zip ต้องเป็น `/` เสมอ ไม่ว่าจะแพ็กจากเครื่องอะไร

    รูปแบบ zip กำหนดไว้แบบนั้น และฝั่งเซิร์ฟเวอร์แยกโฟลเดอร์ด้วย `/` ตรงๆ
    (`core/store.py` `extract` · `validate._resolve_agent_py`) ถ้าไฟล์จากเครื่อง
    Windows หลุดมาเป็น `my-agent\\agent.py` เซิร์ฟเวอร์จะเห็นเป็นชื่อไฟล์เดียว
    ยาวๆ แล้วหา agent.py ไม่เจอ
    """
    (tmp_path / "agent.py").write_text("class Agent: pass\n")
    (tmp_path / "helpers").mkdir()
    (tmp_path / "helpers" / "world.py").write_text("X = 1\n")
    (tmp_path / "helpers" / "deep").mkdir()
    (tmp_path / "helpers" / "deep" / "more.py").write_text("Y = 2\n")

    names = zipfile.ZipFile(io.BytesIO(pack(tmp_path))).namelist()

    assert not any("\\" in n for n in names), names
    assert "helpers/world.py" in names
    assert "helpers/deep/more.py" in names
