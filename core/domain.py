"""แบบจำลองโดเมนของแพลตฟอร์ม — README §2 และ §12

**core ไม่รู้จักโจทย์ใดๆ** ทั้งไฟล์นี้ไม่มีคำว่า vacuum, grid, coverage หรือ RL เลย
ถ้าวันหนึ่งต้องเขียน `if competition == "..."` ลงใน core แปลว่าออกแบบผิด
([README §10.5](../README.md#105-โครงสร้าง-repository))

ข้อกำหนดสำคัญสามข้อจาก §2 ที่แบบจำลองนี้บังคับให้เป็นจริง

1. **Submission ≠ Run** — submission หนึ่งครั้งถูกรันได้หลายครั้ง (public ตอนส่ง,
   private ตอนปิดเทอม, rejudge เมื่อพบบั๊ก) ประวัติทั้งหมดเก็บไว้ ไม่ทับกัน
2. **Team ไม่ใช่ User** — คะแนนผูกกับทีม แต่ log ว่าใครเป็นคนกดส่ง
3. **Course-scoped ทุกอย่าง** — ผู้สอนแต่ละวิชาเห็นเฉพาะวิชาตัวเอง
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, Enum):
    STUDENT = "student"
    TA = "ta"
    INSTRUCTOR = "instructor"


class RunKind(str, Enum):
    """ทำไม run หนึ่งถึงถูกสร้าง — ตัวนี้กำหนดว่าใช้ seed ชุดไหนและขึ้น leaderboard ไหนไหม"""

    PUBLIC = "public"  # ตอนส่ง — ขึ้น public leaderboard
    PRIVATE = "private"  # ตอนปิดรับ — ตัดสินเกรด
    DRYRUN = "dryrun"  # ทดสอบว่าแพ็กไฟล์ถูก — ไม่กินโควตา ไม่ขึ้น leaderboard
    REJUDGE = "rejudge"  # รันซ้ำเมื่อแก้บั๊กใน environment — เก็บผลเดิมไว้เทียบ


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: run ที่ยังไม่จบ — ใช้บังคับกติกา "1 งานพร้อมกันต่อทีม"
ACTIVE_RUN_STATUSES = frozenset({RunStatus.QUEUED, RunStatus.RUNNING})

#: kind ที่ขึ้น leaderboard — dryrun กับ rejudge ไม่ขึ้น
SCORING_RUN_KINDS = frozenset({RunKind.PUBLIC, RunKind.PRIVATE})


@dataclass
class Team:
    id: str
    course_id: str
    name: str
    alias: str | None = None  # ชื่อนิรนามบน leaderboard (README §6.1) ผู้สอนเห็นชื่อจริงเสมอ
    member_ids: list[str] = field(default_factory=list)

    def display_name(self, *, reveal: bool) -> str:
        """`reveal=True` สำหรับผู้สอน/TA เท่านั้น"""
        return self.name if reveal or not self.alias else self.alias


@dataclass
class Phase:
    """ช่วงของเทอมที่ใช้ config ต่างกัน — README §6.5"""

    id: str
    name: str
    starts_at: datetime
    ends_at: datetime
    config_override: dict[str, Any] = field(default_factory=dict)

    def contains(self, when: datetime) -> bool:
        return self.starts_at <= when < self.ends_at


@dataclass
class Competition:
    id: str
    course_id: str
    slug: str
    title: str
    task_type: str  # เช่น "agent_env" — ตัวเลือก runner
    env_plugin: str  # เช่น "vacuum.arena:PLUGIN" — core ไม่รู้ว่ามันคืออะไร
    config_path: str
    opens_at: datetime
    closes_at: datetime
    quota_per_day: int = 5
    max_final_submissions: int = 2
    phases: list[Phase] = field(default_factory=list)
    #: package ที่นิสิต import ได้ นอกเหนือจาก stdlib — ประกาศตอนเปิดเทอม (§13)
    #: ว่างไว้ = ใช้ค่าจาก `default_whitelist()`
    import_whitelist: frozenset[str] = frozenset()

    def effective_whitelist(self) -> frozenset[str]:
        """whitelist ที่ใช้จริง — รวม **แพ็กเกจของ environment เอง** ให้เสมอ

        นิสิตต้อง import แพ็กเกจนั้นได้โดยนิยาม เพราะ starter kit ทั้งชุดอยู่ในนั้น
        (baseline ที่ใช้เป็นตัวอย่าง, helper ของแผนที่, ตัวคิดคะแนนที่รันในเครื่องตัวเอง)
        core ดึงชื่อมาจาก `env_plugin` ได้โดยไม่ต้องรู้ว่าแพ็กเกจนั้นทำอะไร
        """
        env_package = self.env_plugin.split(":", 1)[0].split(".", 1)[0]
        base = self.import_whitelist or frozenset({"numpy", "torch"})
        return base | {env_package}

    def phase_at(self, when: datetime) -> Phase | None:
        return next((p for p in self.phases if p.contains(when)), None)

    def is_open(self, when: datetime) -> bool:
        return self.opens_at <= when < self.closes_at


@dataclass
class Submission:
    """สิ่งที่ทีมส่ง 1 ครั้ง — **ไม่ใช่** ผลการรัน"""

    id: str
    competition_id: str
    team_id: str
    submitted_by: str  # user id — ใช้ดูการกระจายงานในกลุ่ม (§7)
    artifact_url: str
    artifact_sha256: str
    note: str = ""
    is_final_pick: bool = False
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class EpisodeResult:
    run_id: str
    seed: int
    score: float
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    replay_url: str | None = None


@dataclass
class Run:
    """การประเมิน 1 ครั้งของ submission หนึ่ง"""

    id: str
    submission_id: str
    competition_id: str
    team_id: str
    kind: RunKind
    status: RunStatus = RunStatus.QUEUED
    lane: str = "cpu"
    config_hash: str | None = None
    env_version: str | None = None
    score: float | None = None
    tiebreak: tuple = ()  # คีย์ตัดสินเสมอที่ env plugin ประกาศ — core ไม่ตีความ
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    attempts: int = 0
    runner_id: str | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    episodes: list[EpisodeResult] = field(default_factory=list)

    @property
    def counts_for_leaderboard(self) -> bool:
        return self.status is RunStatus.DONE and self.kind in SCORING_RUN_KINDS


@dataclass
class AuditEvent:
    """append-only — README §7 ต้องย้อนดูได้เสมอว่าใครทำอะไร"""

    id: str
    actor_id: str | None
    action: str
    target_type: str
    target_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)


class QuotaExceeded(Exception):
    """ทีมส่งเกินโควตาของวัน — §7 กันการยิงมั่วจนบังเอิญได้คะแนนดี"""


class CompetitionClosed(Exception):
    pass
