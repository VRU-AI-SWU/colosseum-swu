"""ตัวเลขที่หน้าเว็บบอกนิสิต ต้องตรงกับ config ที่ใช้ตัดสินจริง

หน้าเว็บอธิบายว่าแต่ละ phase ต่างกันยังไง — ห้องกี่ช่อง เห็นรอบตัวแค่ไหน กี่ step
ข้อความพวกนั้นเขียนไว้ใน `web/index.html` ซึ่งแปลว่า **มีแหล่งข้อมูลชุดที่สอง
ซ้อนกับ YAML** และแหล่งข้อมูลสองชุดที่พูดเรื่องเดียวกันจะค่อยๆ ไม่ตรงกันเสมอ

ถ้าวันหนึ่งมีคน calibrate config ใหม่แล้วลืมแก้หน้าเว็บ นิสิตจะวางแผนจากตัวเลข
ที่ผิด — ซึ่งแย่กว่าการไม่บอกอะไรเลย เพราะเขาจะเชื่อมันจนกว่าคะแนนจะไม่เข้าท่า

เทสต์นี้ทำหน้าที่แทนการเตือนความจำ · แก้ YAML แล้วมันจะแดงจนกว่าจะแก้หน้าเว็บด้วย
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "envs" / "cp463-vacuum"))

import pytest  # noqa: E402

from vacuum import load_config  # noqa: E402
from vacuum.config import CONFIG_DIR  # noqa: E402

INDEX = REPO / "web" / "index.html"
PHASES = ("warmup", "main", "final")


@pytest.fixture(scope="module")
def blurbs() -> dict[str, str]:
    """ข้อความอธิบาย phase ที่ฝังอยู่ในหน้าเว็บ"""
    block = re.search(r"const PHASE_BLURB = \{(.*?)\};", INDEX.read_text(), re.S)
    assert block, "ไม่พบ PHASE_BLURB ในหน้าเว็บ — โครงของไฟล์เปลี่ยนไปแล้ว"
    found = dict(re.findall(r'(\w+):\s*"([^"]*)"', block.group(1)))
    assert set(found) == set(PHASES), f"PHASE_BLURB ไม่ครบทุก phase: {sorted(found)}"
    return found


def _int(text: str) -> int:
    """`1,500` → 1500 — หน้าเว็บใส่ลูกน้ำให้คนอ่าน"""
    return int(text.replace(",", ""))


@pytest.mark.parametrize("phase", PHASES)
def test_room_size_matches_config(phase, blurbs):
    config = load_config(CONFIG_DIR / f"{phase}.yaml")
    m = re.search(r"ห้อง (\d+)×(\d+)", blurbs[phase])
    assert m, f"{phase}: หน้าเว็บไม่ได้บอกขนาดห้อง"
    assert (_int(m.group(1)), _int(m.group(2))) == (config.room.width, config.room.height)


@pytest.mark.parametrize("phase", PHASES)
def test_max_steps_matches_config(phase, blurbs):
    config = load_config(CONFIG_DIR / f"{phase}.yaml")
    m = re.search(r"([\d,]+) step", blurbs[phase])
    assert m, f"{phase}: หน้าเว็บไม่ได้บอกจำนวน step"
    assert _int(m.group(1)) == config.episode.max_steps


@pytest.mark.parametrize("phase", PHASES)
def test_observation_matches_config(phase, blurbs):
    """`full` ต้องเขียนว่าเห็นทั้งห้อง ส่วน `local` ต้องบอกขนาดหน้าต่างให้ตรง"""
    config = load_config(CONFIG_DIR / f"{phase}.yaml")
    text = blurbs[phase]

    if config.robot.observation == "full":
        assert "เห็นทั้งห้อง" in text, f"{phase}: config เป็น full แต่หน้าเว็บไม่ได้บอกแบบนั้น"
        return

    m = re.search(r"รอบตัว (\d+)×(\d+)", text)
    assert m, f"{phase}: config เป็น local แต่หน้าเว็บไม่ได้บอกขนาดหน้าต่าง"
    window = config.robot.observation_window
    assert (_int(m.group(1)), _int(m.group(2))) == (window, window)


@pytest.mark.parametrize("phase", PHASES)
def test_noise_claim_matches_config(phase, blurbs):
    """ช่วงที่ไม่มี noise ต้องบอกว่าไม่มี ช่วงที่มีต้องไม่บอกว่าไม่มี

    เป็นความต่างที่กระทบวิธีเขียน agent มากที่สุด — ไม่มี noise แปลว่าเชื่อ
    เซนเซอร์ตรงๆ ได้ ซึ่งเป็นสมมติฐานที่พังทันทีเมื่อเข้าช่วง Main
    """
    config = load_config(CONFIG_DIR / f"{phase}.yaml")
    clean = config.dynamics.sensor_noise == 0 and config.dynamics.action_noise == 0
    says_none = "ไม่มี noise" in blurbs[phase]
    assert says_none == clean, (
        f"{phase}: config มี noise={not clean} แต่หน้าเว็บบอกว่าไม่มี noise={says_none}"
    )


@pytest.mark.parametrize("phase", PHASES)
def test_sticky_claim_matches_config(phase):
    config = load_config(CONFIG_DIR / f"{phase}.yaml")
    blurb = re.search(
        rf'{phase}:\s*"([^"]*)"', re.search(r"const PHASE_BLURB = \{(.*?)\};", INDEX.read_text(), re.S).group(1)
    ).group(1)
    has_sticky = config.dynamics.sticky_dirt > 0
    says_none = "ไม่มีช่องเหนียว" in blurb
    assert says_none == (not has_sticky), (
        f"{phase}: config มีช่องเหนียว={has_sticky} แต่หน้าเว็บบอกว่าไม่มี={says_none}"
    )


def test_phase_labels_cover_every_phase_the_tool_writes():
    """ชื่อที่หน้าเว็บแสดงต้องครบทุก phase ที่ setup_competition เขียนลงฐานข้อมูล

    ถ้าขาด หน้าเว็บจะแสดงชื่อดิบอย่าง `warmup` แทน `Warm-up` ซึ่งไม่ถึงกับพัง
    แต่แปลว่ามี phase ที่ไม่มีใครคิดถึงตอนทำหน้าเว็บ
    """
    from tools.setup_competition import PHASES as TOOL_PHASES

    labels = re.search(r"const PHASE_LABEL = \{(.*?)\};", INDEX.read_text(), re.S)
    assert labels
    known = set(re.findall(r"(\w+):", labels.group(1)))
    assert set(TOOL_PHASES) <= known, f"หน้าเว็บไม่รู้จัก phase: {set(TOOL_PHASES) - known}"


def test_calendar_fetch_needs_no_token():
    """หน้าเว็บต้องเรียกปฏิทินแบบไม่แนบโทเคน ไม่งั้นคนที่ยังไม่ล็อกอินจะไม่เห็น

    `api()` ของหน้าเว็บใส่ `Authorization` ให้อัตโนมัติ — ปฏิทินจึงต้องใช้ `fetch`
    ตรงๆ ซึ่งเป็นรายละเอียดที่คนแก้ทีหลังมองข้ามได้ง่ายมาก
    """
    source = INDEX.read_text()
    body = re.search(r"async function loadCalendar\(\) \{(.*?)\n\}", source, re.S)
    assert body, "ไม่พบ loadCalendar()"
    assert "fetch(API" in body.group(1), "ปฏิทินต้องเรียกด้วย fetch ตรงๆ"
    assert "api(" not in body.group(1), "ปฏิทินต้องไม่เรียกผ่าน api() ซึ่งแนบโทเคนไปด้วย"


def test_blurbs_are_valid_json_strings(blurbs):
    """ข้อความถูกฝังใน JS — อักขระที่ escape ไม่ถูกจะทำให้ทั้งหน้าไม่ทำงาน"""
    for phase, text in blurbs.items():
        json.dumps(text)
        assert '"' not in text, f"{phase}: มีเครื่องหมายคำพูดซ้อนในข้อความ"


# ── คำอธิบายของโจทย์ทำนาย ──────────────────────────────────────────


def prediction_blurbs() -> dict[str, str]:
    block = re.search(r"const PREDICTION_BLURB = \{(.*?)\};", INDEX.read_text(), re.S)
    assert block, "ไม่พบ PREDICTION_BLURB ในหน้าเว็บ"
    found = dict(re.findall(r'(\w+):\s*"([^"]*)"', block.group(1)))
    assert set(found) == set(PHASES), f"PREDICTION_BLURB ไม่ครบทุก phase: {sorted(found)}"
    return found


def test_prediction_blurbs_do_not_promise_a_harder_task_later():
    """โจทย์ทำนาย **ไม่เปลี่ยนกติกาเลยระหว่าง phase** — ข้อความต้องไม่อ้างว่าเปลี่ยน

    ผูกกับสิ่งที่เครื่องมือรับประกันจริง (`PredictionTask.overrides` คืน `{}` เสมอ)
    ไม่ใช่กับความจำของคนเขียน · ถ้าวันหนึ่งมีคนทำให้ config ต่างกันต่อ phase
    เทสต์นี้จะเตือนว่าข้อความบนหน้าเว็บต้องแก้ตามด้วย
    """
    from tools.setup_competition import TASK_TYPES

    task = TASK_TYPES["prediction"]
    churn = REPO / "envs" / "cp462-tabular" / "tabular" / "configs" / "churn.yaml"
    assert all(task.overrides(p, churn) == {} for p in PHASES), (
        "โจทย์ทำนายเริ่มมี config ต่างกันต่อ phase แล้ว — ต้องแก้ PREDICTION_BLURB ด้วย"
    )

    for phase, text in prediction_blurbs().items():
        for claim in ("ยากขึ้น", "ห้อง ", "step", "noise"):
            assert claim not in text, f"{phase}: อ้างถึง {claim!r} ทั้งที่โจทย์ไม่เปลี่ยน"


def test_the_page_picks_blurbs_by_task_type():
    """ถ้าเลือกไม่ถูกชนิด CP462 จะเห็นข้อความเรื่องห้อง 10×10 ของ CP463"""
    text = INDEX.read_text()
    assert re.search(r"const BLURBS = \{[^}]*agent_env[^}]*prediction[^}]*\}", text), \
        "ไม่พบตารางเลือกคำอธิบายตามชนิดโจทย์"
    assert "BLURBS[cal.task_type]" in text, "ตอนวาดไม่ได้เลือกตามชนิดโจทย์"


def test_the_api_sends_the_task_type_the_page_needs():
    """ผูกสองฝั่ง — หน้าเว็บอ่าน `cal.task_type` ถ้า API ไม่ส่ง ทุกโจทย์จะได้ข้อความของ CP463"""
    from core.api import create_app
    from core.wiring import demo_arena
    from fastapi.testclient import TestClient

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        arena, _ = demo_arena(Path(tmp))
        client = TestClient(create_app(arena))
        body = client.get("/api/competitions/cp463-vacuum-1-2026").json()
    assert body["task_type"] == "agent_env"
