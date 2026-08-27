"""ปฏิทินของ competition — เขตของวันและสัญญาเรื่อง config_hash

เครื่องมือ `tools/setup_competition.py` เขียนทับ record ที่ใช้ตัดสินคะแนนจริง
สองอย่างที่ผิดแล้วเจ็บและมองไม่เห็นตอนรีวิว

  · **เขตของวัน** — `2026-09-30` ต้องหมายถึงถึงสิ้นวันนั้นตามเวลาไทย
    ถ้าเผลอใช้ UTC ตรงๆ นิสิตจะเสียวันสุดท้ายไป 17 ชั่วโมง และจะรู้ตอนที่สายไปแล้ว

  · **config_hash** — `config_override` ที่ประกอบแล้วต้องให้ hash ตรงกับตอน
    generate seed ไม่งั้น worker โยน `ConfigDrift` ตอนให้คะแนน ซึ่งเป็นจังหวะ
    ที่แย่ที่สุดที่จะรู้ · เครื่องมือตรวจก่อนเขียน เทสต์นี้ตรวจว่ามันตรวจจริง
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from runners.seeds import expected_config_hash  # noqa: E402
from tools.setup_competition import (  # noqa: E402
    ICT,
    PHASES,
    config_override_for,
    parse_range,
    verify,
)

BASE = REPO / "envs" / "cp463-vacuum" / "vacuum" / "configs" / "main.yaml"
SLUG = "cp463-vacuum-1-2026"

#: `verify()` ต้องอ่าน config_hash ที่ตรึงไว้ใน seeds.yaml ซึ่งอยู่ใน repo ลับ
#: เทสต์ที่เหลือไม่แตะของลับเลย ตามธรรมเนียมของ core/tests ที่ตัด ARENA_SECRETS ทิ้ง
needs_secrets = pytest.mark.skipif(
    expected_config_hash(competition_slug=SLUG, phase="main") is None,
    reason="ต้องตั้ง ARENA_SECRETS ให้ชี้ไป clone ของ colosseum-hypogeum",
)


# ── เขตของวัน ──────────────────────────────────────────────────────


def test_range_covers_the_last_day_completely():
    """วันจบต้องรวมทั้งวัน — `Phase.contains` ใช้ `start <= when < end`"""
    start, end = parse_range("2026-09-15..2026-09-30")
    assert start == datetime(2026, 9, 15, 0, 0, tzinfo=ICT)
    assert end == datetime(2026, 10, 1, 0, 0, tzinfo=ICT)

    last_moment = datetime(2026, 9, 30, 23, 59, 59, tzinfo=ICT)
    assert start <= last_moment < end, "วันสุดท้ายของช่วงถูกตัดออกไป"


def test_range_is_thai_time_not_utc():
    """ถ้าเผลอใช้ UTC เที่ยงคืนไทยจะกลายเป็น 17:00 ของวันก่อนหน้า"""
    start, _ = parse_range("2026-09-15..2026-09-30")
    assert start.utcoffset() == timedelta(hours=7)
    # เที่ยงคืนไทย = 17:00 UTC ของวันก่อนหน้า
    assert start.astimezone(timezone.utc) == datetime(2026, 9, 14, 17, 0, tzinfo=timezone.utc)


def test_consecutive_ranges_leave_no_gap_and_no_overlap():
    """ช่วงที่ติดกันต้องไม่มีรูโหว่ ไม่งั้นงานที่ส่งคาบเกี่ยวจะหา phase ไม่เจอ"""
    _, warmup_end = parse_range("2026-09-15..2026-09-30")
    main_start, _ = parse_range("2026-10-01..2026-10-31")
    assert warmup_end == main_start


@pytest.mark.parametrize("bad", ["2026-09-15", "2026-09-30..2026-09-15", "15/09/2026..30/09/2026"])
def test_bad_ranges_are_rejected(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_range(bad)


# ── สัญญาเรื่อง config ─────────────────────────────────────────────


def test_override_reproduces_each_phase_config_exactly():
    """`main.yaml` + override ต้องได้ config ที่เหมือน YAML ของ phase นั้นทุกบิต

    override คำนวณจากไฟล์จริง ไม่ได้เขียนมือ — เทสต์นี้ยืนยันว่าการคำนวณนั้นถูก
    """
    from vacuum import load_config
    from vacuum.config import CONFIG_DIR

    base = load_config(BASE)
    for phase in PHASES:
        override = config_override_for(phase, BASE)
        got = base.replace(**override)
        want = load_config(CONFIG_DIR / f"{phase}.yaml")
        assert got.config_hash == want.config_hash, f"{phase}: ประกอบแล้วไม่ตรงกับ {phase}.yaml"


def test_main_needs_no_override():
    """competition ชี้ที่ main.yaml อยู่แล้ว — phase main จึงต้องไม่มี override

    ถ้ามีค่าโผล่มา แปลว่า base ที่ใช้ไม่ใช่ main.yaml จริง ซึ่งเป็นสัญญาณว่ามีอะไรผิด
    """
    assert config_override_for("main", BASE) == {}


def test_overrides_are_json_safe():
    """`config_override` ถูกเก็บลงฐานข้อมูลเป็น JSON — ค่าที่ serialize ไม่ได้จะพังตอนบันทึก"""
    import json

    for phase in PHASES:
        json.dumps(config_override_for(phase, BASE))


@needs_secrets
def test_verify_catches_a_config_that_drifted_from_the_seeds():
    """ถ้า override ผิด เครื่องมือต้องจับได้ **ก่อน** เขียนลงฐานข้อมูล

    เป็นด่านเดียวที่กันไม่ให้ ConfigDrift ไปโผล่ตอนให้คะแนนจริง
    """
    good = {phase: config_override_for(phase, BASE) for phase in PHASES}
    assert verify(BASE, good, SLUG) == []

    tampered = {phase: dict(o) for phase, o in good.items()}
    tampered["final"]["room.width"] = 999
    problems = verify(BASE, tampered, SLUG)
    assert any("final" in p and "config_hash" in p for p in problems), problems
