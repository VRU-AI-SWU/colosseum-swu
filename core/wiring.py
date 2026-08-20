"""ประกอบชิ้นส่วนเข้าด้วยกัน — **ที่เดียวที่ core กับ runners มาเจอกัน**

`core/` ไม่ import `runners/` และ `runners/agent_env/` ไม่ import `core/`
การผูกทั้งสองฝั่งเกิดที่นี่ ทำให้เกณฑ์ตรวจใน [README §10.5](../README.md#105-โครงสร้าง-repository)
ยังเป็นจริง: การเพิ่ม competition ใหม่แตะแค่ `envs/` กับไฟล์นี้
"""

from __future__ import annotations

import json

from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.domain import Competition, Phase, Team, new_id
from core.leaderboard import BaselineMark
from core.service import Arena, build_arena
from vacuum import config_path as vacuum_config_path

REPO = Path(__file__).resolve().parent.parent


def agent_env_validator(archive_url: str, whitelist: frozenset[str]):
    """ตัวตรวจไฟล์ของ task template `agent_env` — core เรียกผ่าน registry ไม่ import ตรงๆ"""
    from runners.agent_env.validate import check_import_whitelist, inspect_archive

    return check_import_whitelist(inspect_archive(archive_url), whitelist)


VALIDATORS = {"agent_env": agent_env_validator}

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


def demo_arena(
    root: Path, *, teams: int = 3, db_path: Path | str | None = None
) -> tuple[Arena, list[Team]]:
    """Arena ที่พร้อมใช้สำหรับ dev และเทสต์ — มี CP463 Competition 1 ลงทะเบียนไว้แล้ว

    โทเคนของทีมคือ `team-1`, `team-2`, ... (ของชั่วคราวจนกว่าจะมี OAuth)

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
        team = Team(
            id=f"team-{i}",
            course_id=competition.course_id,
            name=f"ทีมที่ {i}",
            member_ids=[f"user-{i}"],
        )
        arena.store.save_team(team)  # id ทำหน้าที่เป็นโทเคนไปก่อน
        created.append(team)

    return arena, created
