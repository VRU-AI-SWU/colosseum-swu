"""ส่วนของ CLI ที่มีตรรกะจริง — การแพ็กไฟล์และการอ่านช่วง seed

`pack()` สำคัญกว่าที่ดู: `models/` กับ `.venv/` เป็นสาเหตุที่พบบ่อยที่สุดที่ทำให้ zip
เกินเพดาน 95 MB โดยที่นิสิตไม่รู้ตัว — ถ้า CLI ไม่ข้ามให้ ภาระจะไปตกที่ error ตอนอัพโหลด
"""

from __future__ import annotations

import zipfile

import pytest

from core.cli import pack, parse_seeds


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
