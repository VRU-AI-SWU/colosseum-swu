"""ประกอบชิ้นส่วนเข้าด้วยกัน — **ที่เดียวที่ core กับ runners มาเจอกัน**

`core/` ไม่ import `runners/` และ `runners/agent_env/` ไม่ import `core/`
การผูกทั้งสองฝั่งเกิดที่นี่ ทำให้เกณฑ์ตรวจใน [README §10.5](../README.md#105-โครงสร้าง-repository)
ยังเป็นจริง: การเพิ่ม competition ใหม่แตะแค่ `envs/` กับไฟล์นี้
"""

from __future__ import annotations

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

#: หมุด baseline ที่วัดจริงแล้ว — README §6.2 บอกว่าต้อง **ตรึงไว้ทั้งเทอม**
#: ค่าชุดนี้มาจากชุด conformance · ค่าที่ใช้ตัดสินเกรดต้องรันบน public seeds อีกครั้ง
CP463_VACUUM_LADDER = [
    BaselineMark("bronze", "🥉 Bronze", 0.243796, "sha256:f58cc4e51a1e4f8"),
    BaselineMark("silver", "🥈 Silver", 0.810008, "sha256:f58cc4e51a1e4f8"),
    BaselineMark("gold", "🥇 Gold", 1.716052, "sha256:f58cc4e51a1e4f8"),
    BaselineMark("diamond", "💎 Diamond", 1.803565, "sha256:f58cc4e51a1e4f8"),
]


def demo_arena(root: Path, *, teams: int = 3) -> tuple[Arena, list[Team]]:
    """Arena ที่พร้อมใช้สำหรับ dev และเทสต์ — มี CP463 Competition 1 ลงทะเบียนไว้แล้ว

    โทเคนของทีมคือ `team-1`, `team-2`, ... (ของชั่วคราวจนกว่าจะมี OAuth)
    """
    arena = build_arena(root, validators=VALIDATORS)
    now = datetime.now(timezone.utc)

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
    arena.store.competitions[competition.id] = competition

    created = []
    for i in range(1, teams + 1):
        team = Team(
            id=f"team-{i}",
            course_id=competition.course_id,
            name=f"ทีมที่ {i}",
            member_ids=[f"user-{i}"],
        )
        arena.store.teams[team.id] = team  # id ทำหน้าที่เป็นโทเคนไปก่อน
        created.append(team)

    return arena, created
