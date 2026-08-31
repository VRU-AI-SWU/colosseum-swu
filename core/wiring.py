"""ประกอบชิ้นส่วนเข้าด้วยกัน — **ที่เดียวที่ core กับ runners มาเจอกัน**

`core/` ไม่ import `runners/` และ `runners/agent_env/` ไม่ import `core/`
การผูกทั้งสองฝั่งเกิดที่นี่ ทำให้เกณฑ์ตรวจใน [README §10.5](../README.md#105-โครงสร้าง-repository)
ยังเป็นจริง: การเพิ่ม competition ใหม่แตะแค่ `envs/` กับไฟล์นี้
"""

from __future__ import annotations

import json

from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.domain import Competition, Phase, Team, User, new_id
from core.leaderboard import BaselineMark
from core.service import Arena, build_arena
from vacuum import config_path as vacuum_config_path

REPO = Path(__file__).resolve().parent.parent


def agent_env_validator(archive_url: str, whitelist: frozenset[str]):
    """ตัวตรวจไฟล์ของ task template `agent_env` — core เรียกผ่าน registry ไม่ import ตรงๆ"""
    from runners.agent_env.validate import check_import_whitelist, inspect_archive

    return check_import_whitelist(inspect_archive(archive_url), whitelist)


def prediction_validator(archive_url: str, whitelist: frozenset[str]):
    """ตัวตรวจไฟล์ของ task template `prediction`"""
    from runners.prediction.validate import check_import_whitelist, inspect_archive

    return check_import_whitelist(inspect_archive(archive_url), whitelist)


#: `Competition.task_type` → ตัวตรวจไฟล์ · competition ที่ประกาศ task_type ที่ไม่มีในนี้
#: จะถูกปฏิเสธตอนรับไฟล์พร้อมข้อความที่บอกว่าต้องลงทะเบียนที่ไหน (`core/service.py`)
VALIDATORS = {
    "agent_env": agent_env_validator,
    "prediction": prediction_validator,
}

PIN_DIR = REPO / "core" / "baseline_pins"

LABELS = {
    "bronze": "🥉 Bronze",
    "silver": "🥈 Silver",
    "gold": "🥇 Gold",
    "diamond": "💎 Diamond",
}


def baseline_ladder(slug: str, phase: str) -> list[BaselineMark]:
    """หมุด baseline ของ phase หนึ่ง — อ่านจากไฟล์ที่ `tools/pin_baselines.py` ตรึงไว้

    ค่าเหล่านี้วัดบน **public seeds ชุดจริง** ซึ่งเป็นชุดเดียวกับที่ให้คะแนนนิสิต
    จึงเทียบกับ leaderboard ได้ตรงๆ (ค่าชุดเดิมมาจากชุด conformance ซึ่งเทียบไม่ได้)

    README §6.2 สั่งให้ตรึงไว้ทั้งเทอม — การอ่านจากไฟล์แทนการคำนวณสดคือสิ่งที่ทำให้
    หมุดไม่ขยับ ถ้า environment เปลี่ยนจน `--check` ไม่ผ่าน แปลว่าต้องขึ้น env_version
    และ rejudge ไม่ใช่แก้ตัวเลขในไฟล์เฉยๆ
    """
    data = json.loads((PIN_DIR / f"{slug}.json").read_text(encoding="utf-8"))
    entry = data["phases"][phase]
    return [
        BaselineMark(level, LABELS[level], values["score"], entry["config_hash"])
        for level, values in entry["levels"].items()
    ]


#: ladder ของ phase ที่กำลังแข่ง — README §6.2 บอกว่าต้อง **ตรึงไว้ทั้งเทอม**
CP463_VACUUM_LADDER = baseline_ladder("cp463-vacuum-1-2026", "main")


def google_auth_from_env() -> "GoogleAuth | None":
    """อ่านค่าตั้ง OAuth จากตัวแปรแวดล้อม — คืน `None` ถ้ายังไม่ได้ตั้ง

    **client secret อยู่ใน environment ไม่ใช่ในไฟล์ใน repo** · คืน None เงียบๆ
    แทนที่จะล้ม เพราะเครื่อง dev กับเทสต์ไม่จำเป็นต้องมี — endpoint ที่ต้องใช้มัน
    จะตอบ 503 พร้อมบอกว่าต้องตั้งตัวแปรไหน ซึ่งชัดกว่าการที่บริการไม่ยอมเริ่มเลย

        ARENA_GOOGLE_CLIENT_ID      จาก Google Cloud console
        ARENA_GOOGLE_CLIENT_SECRET  🔒
        ARENA_GOOGLE_REDIRECT_URI   ต้องตรงกับที่ลงทะเบียนไว้ใน console เป๊ะ
        ARENA_WEB_ORIGIN            ที่ที่จะส่งนิสิตกลับไปพร้อมโทเคน
    """
    import os

    from core.auth import GoogleAuth

    client_id = os.environ.get("ARENA_GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("ARENA_GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None

    web_origin = os.environ.get("ARENA_WEB_ORIGIN", "https://colosseum.vru-ai.com").rstrip("/")
    redirect_uri = os.environ.get(
        "ARENA_GOOGLE_REDIRECT_URI", "https://colosseum-api.vru-ai.com/auth/google/callback"
    )
    return GoogleAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        web_origin=web_origin,
        allowed_domain=os.environ.get("ARENA_ALLOWED_DOMAIN", "g.swu.ac.th"),
    )


def staff_emails_from_env() -> frozenset[str]:
    """อีเมลของผู้สอน/TA จาก `ARENA_STAFF_EMAILS` — คั่นด้วยจุลภาค

        ARENA_STAFF_EMAILS=aj@g.swu.ac.th,ta@g.swu.ac.th

    **ว่างไว้ = ไม่มีใครเป็นผู้สอน** ซึ่งเป็นค่าเริ่มต้นที่ถูกต้อง · การเดาว่า
    "คนแรกที่ล็อกอินคือผู้สอน" จะทำให้ใครก็ตามที่รู้ URL ก่อนเพื่อนยึดสิทธิ์ไปได้

    อยู่ใน environment ไม่ใช่ในฐานข้อมูล เหมือน sudoers — ถ้าแก้ผ่านหน้าเว็บได้
    คนที่ยึดสิทธิ์ได้ครั้งเดียวจะแต่งตั้งตัวเองถาวรและถอดคนอื่นออกได้
    """
    import os

    raw = os.environ.get("ARENA_STAFF_EMAILS", "")
    return frozenset(e.strip().lower() for e in raw.split(",") if e.strip())


#: คำนำหน้าของตัวแปรที่ระบุผู้สอน "ของวิชานั้น"
COURSE_STAFF_PREFIX = "ARENA_COURSE_STAFF_"


def env_key_for_course(course_id: str) -> str:
    """`cp462-1-2026` → `ARENA_COURSE_STAFF_CP462_1_2026`

    id ของวิชาใช้ `-` ได้แต่ชื่อตัวแปรแวดล้อมใช้ไม่ได้ จึงแปลงเป็น `_` และตัวพิมพ์ใหญ่
    """
    return COURSE_STAFF_PREFIX + course_id.upper().replace("-", "_")


def course_staff_from_env() -> dict[str, frozenset[str]]:
    """ผู้สอน**ของแต่ละวิชา** — คั่นด้วยจุลภาค เหมือน `ARENA_STAFF_EMAILS`

        ARENA_COURSE_STAFF_CP462_1_2026=aj2@g.swu.ac.th,ta@g.swu.ac.th

    อยู่ใน environment ด้วยเหตุผลเดียวกับ `ARENA_STAFF_EMAILS` — ถ้าเก็บในฐานข้อมูล
    แล้วแก้ผ่านหน้าเว็บได้ คนที่ยึดสิทธิ์ผู้สอนได้ครั้งเดียวจะแต่งตั้งตัวเองถาวร

    **คีย์ที่คืนเป็น id ของวิชาตามที่สะกดในตัวแปร** (แปลง `_` กลับเป็น `-` และเป็น
    ตัวพิมพ์เล็ก) · ถ้าไม่ตรงกับวิชาที่มีจริง ตัวแปรนั้นจะไม่มีผลกับใครเลย —
    `arena serve` จึงพิมพ์รายการที่อ่านได้ออกมาให้เห็นตอนเริ่ม
    """
    import os

    out: dict[str, frozenset[str]] = {}
    for key, raw in os.environ.items():
        if not key.startswith(COURSE_STAFF_PREFIX):
            continue
        course_id = key[len(COURSE_STAFF_PREFIX):].lower().replace("_", "-")
        emails = frozenset(e.strip().lower() for e in raw.split(",") if e.strip())
        if emails:
            out[course_id] = emails
    return out


def demo_arena(
    root: Path, *, teams: int = 3, db_path: Path | str | None = None
) -> tuple[Arena, list[Team]]:
    """Arena ที่พร้อมใช้สำหรับ dev และเทสต์ — มี CP463 Competition 1 ลงทะเบียนไว้แล้ว

    โทเคนของ**คน**คือ `team-1`, `team-2`, ... (ชื่อเดิมไว้เพื่อไม่ให้เทสต์เดิมพัง)
    แต่ละคนมีทีมเดี่ยวของตัวเองในวิชา `cp463-1-2026`

    **idempotent** — เรียกซ้ำบนฐานข้อมูลเดิมจะใช้ competition กับทีมชุดเดิม ไม่สร้างใหม่
    ถ้าสร้างใหม่ทุกครั้ง submission ที่ส่งไปก่อนรีสตาร์ทจะชี้ไป competition id ที่ไม่มีใครใช้
    แล้ว leaderboard จะว่างทั้งที่ข้อมูลยังอยู่ครบ
    """
    arena = build_arena(root, validators=VALIDATORS, db_path=db_path)
    now = datetime.now(timezone.utc)

    existing = arena.store.competition_by_slug("cp463-vacuum-1-2026")
    if existing is not None:
        return arena, [
            arena.store.teams[f"team-{i}"]
            for i in range(1, teams + 1)
            if f"team-{i}" in arena.store.teams
        ]

    competition = Competition(
        id=new_id(),
        course_id="cp463-1-2026",
        slug="cp463-vacuum-1-2026",
        title="Vacuum Robot Challenge",
        task_type="agent_env",
        env_plugin="vacuum.arena:PLUGIN",
        config_path=str(vacuum_config_path("main")),
        opens_at=now - timedelta(days=1),
        closes_at=now + timedelta(days=30),
        quota_per_day=5,
        phases=[
            Phase(
                id=new_id(),
                name="main",
                starts_at=now - timedelta(days=1),
                ends_at=now + timedelta(days=30),
            )
        ],
    )
    arena.store.save_competition(competition)

    created = []
    for i in range(1, teams + 1):
        # ⚠️ **โทเคนที่เดาได้ — dev กับเทสต์เท่านั้น**
        # ของจริงได้โทเคนสุ่มจาก `new_token()` เสมอ ตรงนี้ตั้งค่าตายตัวเพื่อให้เทสต์
        # และการลองใช้ในเครื่องไม่ต้องไปอ่านค่าสุ่มมาก่อนทุกครั้ง
        #
        # ต้องสร้าง **User** ด้วย ไม่ใช่แค่ Team — โทเคนย้ายมาอยู่ที่คนแล้ว
        # ทีมที่ไม่มีคนอยู่จะยืนยันตัวตนไม่ได้เลย
        user = User(
            id=f"user-{i}",
            email=f"user{i}@example.invalid",
            name=f"นิสิตที่ {i}",
            google_sub=f"demo-sub-{i}",
            token=f"team-{i}",
        )
        arena.store.save_user(user)
        team = Team(
            id=f"team-{i}",
            course_id=competition.course_id,
            name=f"ทีมที่ {i}",
            member_ids=[user.id],
            token=f"team-{i}",
        )
        arena.store.save_team(team)
        created.append(team)

    return arena, created
