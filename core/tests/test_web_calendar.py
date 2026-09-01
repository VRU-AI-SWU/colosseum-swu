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


# ── JS ต้องไม่อ้าง element ที่ไม่มีอยู่จริง ─────────────────────────


def test_every_element_the_script_looks_up_exists():
    """**บั๊กที่เคยเกิดจริง** — `$("cal-edit")` คืน null เพราะลืมใส่ element ลง HTML

    ผลไม่ใช่แค่ "แผงไม่โผล่" แต่เป็น TypeError ที่โยนกลางทาง แล้วลาก**ปฏิทินทั้งแถบ**
    หายไปสำหรับทุกคนรวมทั้งนิสิต · ไม่มี console error ให้เห็นด้วยเพราะหน้าตอนยัง
    ไม่ล็อกอินไม่เดินมาถึงบรรทัดนั้น

    ตรวจแบบ static: ทุก `$("ชื่อ")` ที่เป็นข้อความตรงๆ ต้องมี `id="ชื่อ"` อยู่ในไฟล์
    (นับทั้ง element ที่เขียนไว้ใน HTML และที่ JS สร้างด้วย innerHTML)
    """
    html = INDEX.read_text()
    declared = set(re.findall(r'id="([\w-]+)"', html))
    looked_up = set(re.findall(r'\$\("([\w-]+)"\)', html))

    missing = sorted(looked_up - declared)
    assert not missing, (
        f"JS หา element ที่ไม่มี id นี้อยู่ที่ไหนเลย: {missing}\n"
        "  `$()` จะคืน null แล้วบรรทัดถัดไปโยน TypeError ซึ่งลากส่วนอื่นพังไปด้วย"
    )


def test_the_instructor_panel_cannot_break_the_student_calendar():
    """แผงของผู้สอนต้องวาด**ท้ายสุด**และถูกห่อ try — นิสิตต้องเห็นปฏิทินเสมอ

    หลักเดียวกับที่ `loadCalendar` ห่อ fetch ไว้แล้ว ("ออฟไลน์ — leaderboard สำคัญกว่า")
    """
    html = INDEX.read_text()
    # ดูเฉพาะใน `loadCalendar` — ตัว `renderCalendarEditor` เองก็เรียกตัวเองตอนกด
    # ปุ่มเปิด/ปิด ซึ่งอยู่ก่อนหน้าในไฟล์ · ถ้าค้นทั้งไฟล์จะไปเจอการเรียกอันนั้นแทน
    loader = html[html.index("async function loadCalendar()"):]
    unhide = loader.index('$("cal").hidden = false;')
    call = loader.index("renderCalendarEditor(cal);")
    assert call > unhide, "วาดแผงผู้สอนก่อนปฏิทินจะโผล่ — ถ้ามันโยน ปฏิทินจะหายทั้งแถบ"

    after = loader[unhide:call]
    assert "try {" in after, "การเรียก renderCalendarEditor ไม่ได้ถูกห่อด้วย try"


# ── การรีเฟรชอัตโนมัติต้องไม่ล้างสิ่งที่กำลังกรอก ──────────────────


def test_every_panel_guards_against_being_redrawn_while_in_use():
    """**บั๊กที่เกิดจริง** — หน้ารีเฟรชเองทุก 30 วินาที แล้ววาด `innerHTML` ใหม่ทั้งก้อน

    ผู้สอนที่เปิดฟอร์มสร้างโจทย์แล้วพิมพ์ไปครึ่งทาง จะโดนล้างทิ้งกลางคันโดยไม่มี
    อะไรบอก — `<details>` ปิดเอง ตัวเลือกเด้งกลับอันแรก ค่าที่กรอกหายหมด
    อาการที่ผู้ใช้เห็นคือ "เลือกโจทย์แล้วสักพักมันเด้งกลับหน้าสร้างโจทย์ใหม่"

    ตรวจแบบ static: ตัววาดทุกแผงที่รับค่าจากผู้ใช้ ต้องเรียก `isBusy` ก่อนวาด
    """
    text = INDEX.read_text()
    assert "function isBusy(" in text, "ไม่มีตัวตรวจว่าแผงกำลังถูกใช้อยู่"

    for name in ("renderAuthoring", "renderStaffPanel", "renderCalendarEditor", "renderTeam"):
        start = text.index(f"function {name}(")
        body = text[start:text.index("innerHTML", start)]
        assert "isBusy(" in body, f"{name} วาดทับได้โดยไม่เช็คว่าผู้ใช้กำลังกรอกอยู่"


def test_the_guard_covers_both_focus_and_an_open_section():
    """โฟกัสอย่างเดียวไม่พอ — ผู้สอนที่กางฟอร์มไว้แล้วเลื่อนไปอ่านอย่างอื่น
    ยังไม่ได้พิมพ์ต่อ แต่ค่าที่กรอกไปแล้วต้องไม่หาย"""
    text = INDEX.read_text()
    guard = text[text.index("function isBusy("):]
    guard = guard[:guard.index("\n}")]
    assert "activeElement" in guard
    assert "details[open]" in guard


# ── แผงของคนที่ล็อกอินแล้ว ต้องหายไปตอนออกจากระบบ ──────────────────


def test_every_signed_in_panel_is_hidden_when_leaving_the_board():
    """**บั๊กที่เคยเกิดจริง** — หลังล็อกเอาต์ แผงสร้างโจทย์ยังค้างอยู่บนหน้าจอ

    `show()` ซ่อนแค่ `gate` `enroll` `board` `picker` · `#team` กับ `#staff`
    อยู่ใน `#board` จึงหายตาม แต่ `#cal` กับ `#authoring` อยู่**นอก** และไม่มี
    ใครซ่อนมันเลย · ผลคือคนถัดไปที่มานั่งที่เครื่องนั้นเห็นรหัสวิชา รหัสโจทย์
    ชื่อไฟล์ข้อมูล และชื่อคอลัมน์ทั้งหมด

    ตรวจว่าใช้ **คลาสใน markup** ไม่ใช่รายชื่อใน JS — รายชื่อคือสิ่งที่ต้องจำไป
    เติมทุกครั้งที่เพิ่มแผงใหม่ ซึ่งเป็นสาเหตุของบั๊กนี้ตั้งแต่แรก
    """
    html = INDEX.read_text()

    for panel in ("cal", "authoring"):
        tag = re.search(rf'<section[^>]*id="{panel}"[^>]*>', html)
        assert tag, f"ไม่เจอ element #{panel}"
        assert "signed-in" in tag.group(0), (
            f"#{panel} อยู่นอก #board จึงต้องติดคลาส signed-in ไม่งั้นมันจะค้างหลังล็อกเอาต์"
        )

    assert 'querySelectorAll(".signed-in")' in html, (
        "show() ต้องซ่อนจากคลาสใน markup — รายชื่อที่เขียนใน JS คือสิ่งที่คนลืมอัปเดต"
    )


def test_leaving_the_board_only_hides_never_shows():
    """แต่ละแผงมีเงื่อนไขของตัวเอง — `show()` สั่งแสดงจะทับเงื่อนไขนั้น

    ปฏิทินต้องมีข้อมูลก่อนถึงจะโผล่ · แผงสร้างโจทย์เฉพาะผู้สอน · ถ้า `show()`
    ตั้ง `hidden = false` ให้ทุกอันตอนกลับเข้าหน้ากระดาน นิสิตจะเห็นแผงเปล่า
    """
    html = INDEX.read_text()
    block = html[html.index("function show(view)"):]
    block = block[: block.index("\n}")]
    assert 'el.hidden = true' in block
    assert "el.hidden = false" not in block, "show() ต้องไม่สั่งแสดงแผงพวกนี้"


def test_logging_out_forgets_the_uploaded_dataset():
    """รหัสไฟล์กับชื่อคอลัมน์ที่อัปโหลดค้างในหน่วยความจำของหน้า — ต้องล้างด้วย"""
    html = INDEX.read_text()
    block = html[html.index("function logout(msg)"):]
    block = block[: block.index("\n}")]
    assert "DATASET = null" in block
    assert 'querySelectorAll(".signed-in")' in block and 'innerHTML = ""' in block, (
        "ต้องล้างเนื้อของแผงด้วย ไม่ใช่แค่ซ่อน — `hidden` ไม่กันคนที่เปิด devtools"
    )
