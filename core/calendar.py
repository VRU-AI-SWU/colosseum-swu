"""ปฏิทินของ competition — **กติกาเรื่องวันอยู่ที่นี่ที่เดียว**

ใช้ร่วมกันระหว่าง `tools/setup_competition.py` (ผู้สอนรันบนเครื่อง) กับ
`POST /api/competitions/{slug}/calendar` (ผู้สอนกดจากหน้าเว็บ) · สองทางนี้เขียนทับ
ค่าที่ใช้ตัดสินคะแนนจริงทั้งคู่ ถ้ากติกาเรื่องวันแยกกันอยู่คนละที่ มันจะเพี้ยนกัน
แล้วผลคือ deadline ที่หน้าเว็บบอกกับที่ระบบใช้จริงไม่ตรงกัน

**เขตของวันเป็นเวลาไทย และวันจบรวมทั้งวัน** — `2026-09-30` หมายถึงถึง 23:59:59
ของวันนั้นตามเวลาไทย ไม่ใช่ 07:00 ซึ่งเป็นสิ่งที่จะได้ถ้าใช้ UTC ตรงๆ
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.domain import Phase, new_id

#: เวลาไทย — นิสิตอ่าน deadline เป็นวันตามปฏิทินของตัวเอง ไม่ใช่ UTC
ICT = timezone(timedelta(hours=7))

#: ชื่อ phase ตามลำดับ · ลำดับนี้มีความหมาย — ช่วงต้องเรียงตามนี้และห้ามทับกัน
PHASES = ("warmup", "main", "final")


class CalendarInvalid(Exception):
    """ปฏิทินที่ส่งมาใช้ไม่ได้ — ข้อความต้องบอกว่าอะไรผิดและควรเป็นอย่างไร"""


def parse_day(text: str) -> datetime:
    """`2026-09-15` → เที่ยงคืนของวันนั้นตามเวลาไทย"""
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").replace(tzinfo=ICT)
    except (ValueError, AttributeError) as exc:
        raise CalendarInvalid(f"รูปแบบวันต้องเป็น YYYY-MM-DD — ได้ {text!r}") from exc


def day_range(first: str, last: str) -> tuple[datetime, datetime]:
    """ช่วงครึ่งเปิดที่ **รวมวันสุดท้ายทั้งวัน**

    คืนเที่ยงคืนของวันถัดจากวันจบ เพราะ `Phase.contains` ใช้
    `starts_at <= when < ends_at` — ถ้าคืนเที่ยงคืนของวันจบเอง นิสิตจะเสียวัน
    สุดท้ายไปทั้งวันโดยไม่มีใครตั้งใจ
    """
    start, end = parse_day(first), parse_day(last)
    if end < start:
        raise CalendarInvalid(f"วันจบมาก่อนวันเริ่ม: {first} .. {last}")
    return start, end + timedelta(days=1)


def parse_range(text: str) -> tuple[datetime, datetime]:
    """`2026-09-15..2026-09-30` → ช่วงครึ่งเปิดตามเวลาไทย"""
    if not isinstance(text, str) or ".." not in text:
        raise CalendarInvalid(f"รูปแบบต้องเป็น YYYY-MM-DD..YYYY-MM-DD (ได้ {text!r})")
    first, last = text.split("..", 1)
    return day_range(first, last)


def check_order(ranges: dict[str, tuple[datetime, datetime]]) -> None:
    """ช่วงต้องเรียงตาม `PHASES` และห้ามทับกัน

    **ยอมให้มีช่องว่างได้** (เช่นเว้นสัปดาห์สอบกลางภาค) แต่ห้ามทับ เพราะ
    `Phase.contains` คืนช่วงแรกที่เจอ — งานที่ส่งในช่วงที่ทับกันจะถูกจัดเข้า phase
    ตามลำดับที่บังเอิญเก็บไว้ ซึ่งเป็นพฤติกรรมที่อธิบายให้นิสิตไม่ได้
    """
    missing = [p for p in PHASES if p not in ranges]
    if missing:
        raise CalendarInvalid(f"ขาดช่วงของ {', '.join(missing)}")
    for earlier, later in zip(PHASES, PHASES[1:]):
        if ranges[earlier][1] > ranges[later][0]:
            raise CalendarInvalid(
                f"ช่วง {earlier} กับ {later} ทับกัน — "
                f"{earlier} จบ {(ranges[earlier][1] - timedelta(days=1)):%d %b %Y} "
                f"แต่ {later} เริ่ม {ranges[later][0]:%d %b %Y}"
            )


def build_phases(
    ranges: dict[str, tuple[datetime, datetime]],
    overrides: dict[str, dict] | None = None,
) -> list[Phase]:
    """สร้าง `Phase` ครบชุด — ตรวจลำดับให้ก่อนเสมอ"""
    check_order(ranges)
    overrides = overrides or {}
    return [
        Phase(
            id=new_id(),
            name=name,
            starts_at=ranges[name][0],
            ends_at=ranges[name][1],
            config_override=dict(overrides.get(name, {})),
        )
        for name in PHASES
    ]


def as_days(phase: Phase) -> tuple[str, str]:
    """`Phase` → คู่ของวันแบบที่คนกรอก (วันจบรวมทั้งวัน)

    ตรงข้ามกับ `day_range` — หน้าเว็บใช้เติมค่าเดิมลงในช่องกรอก ถ้าแปลงกลับไม่ตรง
    ผู้สอนที่เปิดฟอร์มแล้วกดบันทึกโดยไม่แก้อะไร จะเลื่อนปฏิทินไปหนึ่งวันทุกครั้ง
    """
    return (
        phase.starts_at.astimezone(ICT).strftime("%Y-%m-%d"),
        (phase.ends_at.astimezone(ICT) - timedelta(days=1)).strftime("%Y-%m-%d"),
    )
