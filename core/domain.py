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

import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def new_id() -> str:
    return uuid.uuid4().hex


def new_token() -> str:
    """โทเคนของทีม — **ต้องเดาไม่ได้**

    เดิมใช้ `team-1`, `team-2` ซึ่งเดาถูกตั้งแต่ครั้งแรก ตอนที่ระบบรันบน localhost
    เรื่องนี้ไม่สำคัญ แต่ API อยู่บนอินเทอร์เน็ตแล้ว ใครที่รู้ URL จะส่งงานในนามทีมใด
    ก็ได้ กินโควตาของทีมนั้น และเห็นคะแนนรายตอนของเขา
    """
    return secrets.token_urlsafe(24)


def new_invite_code() -> str:
    """รหัสเชิญเข้าทีม — สั้นพอที่จะบอกกันด้วยปากได้

    ตัดอักษรที่สับสนออก (0/O, 1/I/L) เพราะนิสิตจะอ่านรหัสนี้ให้เพื่อนฟังในห้องเรียน
    26 ตัวอักษร ยกกำลัง 6 ≈ 300 ล้าน — เดาสุ่มไม่คุ้มเมื่อมีเพดานการลองผิด
    """
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(6))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, Enum):
    STUDENT = "student"
    TA = "ta"
    INSTRUCTOR = "instructor"


@dataclass(frozen=True)
class Paradigm:
    """แนวทางการเรียนรู้ที่โจทย์หนึ่งอยู่ในนั้น — ใช้จัดกลุ่มให้คนอ่านเข้าใจ

    **ติดที่ competition ไม่ใช่ที่วิชา** เพราะวิชาหนึ่งคร่อมหลาย paradigm โดยธรรมชาติ —
    วิชา AI มีทั้งโจทย์ RL และ agent ที่ใช้ LLM ส่วนวิชา ML มีทั้ง supervised
    และ unsupervised · ถ้าเอา paradigm ครอบวิชา จะต้องเลือกให้วิชาหนึ่งอันเดียว
    หรือแตกวิชาเป็นหลายอัน ซึ่งผิดทั้งคู่

    **แยกจาก `task_type` โดยตั้งใจ** — paradigm บอกว่า*นิสิตกำลังเรียนอะไร*
    ส่วน task_type บอกว่า*ระบบให้คะแนนยังไง* สองอย่างนี้ไม่ตรงกันหนึ่งต่อหนึ่ง:
    supervised กับ unsupervised ใช้ runner ตัวเดียวกัน (prediction) แต่เป็นคนละ
    paradigm ในสายตานิสิต

    เก็บเป็นทะเบียนในโค้ดไม่ใช่ตาราง เพราะเป็นคำศัพท์ที่นิ่ง (เหมือน `RunKind`)
    และหน้าเว็บอ้างถึงมันตรงๆ ตอนจัดกลุ่ม — การเพิ่มอันใหม่คือการแก้โค้ดสามบรรทัด
    ไม่ใช่งานที่ต้องทำกลางเทอม
    """

    id: str
    name: str
    blurb: str


PARADIGMS: dict[str, Paradigm] = {
    p.id: p
    for p in (
        Paradigm(
            "reinforcement-learning",
            "Reinforcement Learning",
            "agent ตัดสินใจเป็นลำดับในสภาพแวดล้อมที่ตอบกลับ — วัดที่ผลของทั้งเส้นทาง",
        ),
        Paradigm(
            "supervised-learning",
            "Supervised Learning",
            "เรียนจากตัวอย่างที่มีเฉลย แล้วทำนายข้อมูลที่ไม่เคยเห็น",
        ),
        Paradigm(
            "unsupervised-learning",
            "Unsupervised Learning",
            "หาโครงสร้างในข้อมูลที่ไม่มีเฉลย — จัดกลุ่ม ลดมิติ หาสิ่งผิดปกติ",
        ),
    )
}


class ParadigmUnknown(Exception):
    """competition อ้าง paradigm ที่ไม่มีในทะเบียน — สะกดผิดหรือลืมเพิ่ม"""


def require_paradigm(paradigm_id: str) -> Paradigm:
    """แปลง id เป็น Paradigm — ล้มทันทีถ้าไม่รู้จัก

    ยอมให้ค่าที่สะกดผิดผ่านไป แปลว่าหน้าเว็บจะมีกลุ่มลอยๆ ที่ไม่มีคำอธิบาย
    และไม่มีใครสังเกตจนกว่านิสิตจะถาม
    """
    try:
        return PARADIGMS[paradigm_id]
    except KeyError:
        raise ParadigmUnknown(
            f"ไม่รู้จัก paradigm {paradigm_id!r} — ที่มีคือ {sorted(PARADIGMS)}"
        ) from None


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
class User:
    """นิสิตหนึ่งคน — ยืนยันตัวตนด้วย Google Workspace ของมหาวิทยาลัย

    `google_sub` คือรหัสถาวรที่ Google ให้มา ใช้เป็นตัวจับคู่แทนอีเมล เพราะอีเมล
    เปลี่ยนได้ (เปลี่ยนชื่อ, เปลี่ยนนามสกุล) แต่ `sub` ไม่เปลี่ยน
    """

    id: str
    email: str
    name: str
    google_sub: str
    #: credential ของ **คน** ไม่ใช่ของทีม
    #:
    #: เดิมโทเคนเป็นของทีม ซึ่งพังทันทีที่มีหลายวิชา — คนหนึ่งอยู่หลายทีม (ทีมละวิชา)
    #: จึงมีหลายโทเคน และตอนล็อกอินครั้งแรกยังไม่ได้เข้าวิชาไหน จึงยังไม่มีทีม
    #: จึงไม่มีโทเคนให้ส่งกลับไปให้หน้าเว็บใช้เรียก "เข้าวิชาด้วยรหัส"
    #:
    #: ของแถมที่ตามมา: โทเคนหลุดกระทบคนเดียวไม่ใช่ทั้งทีม · สิทธิ์ผู้สอนถามตรงๆ ได้
    #: ว่าคนนี้เป็นผู้สอนไหม (เดิมต้องมีกฎ "ทั้งทีมต้องเป็นผู้สอน" เพราะใช้โทเคนร่วมกัน)
    #: · `actor_id` ใน audit รู้จริงว่าใครกด ไม่ใช่เดาจากสมาชิกคนแรก
    token: str = field(default_factory=new_token)
    created_at: datetime = field(default_factory=utcnow)


#: ขนาดทีมของวิชาที่เพิ่งสร้าง — ผู้สอนเปลี่ยนได้จากหน้าเว็บทีหลัง
DEFAULT_MAX_TEAM_SIZE = 6

#: เพดานที่ผู้สอนตั้งได้ · ไม่ได้มาจากข้อจำกัดทางเทคนิค แต่เป็นการกันการพิมพ์ผิด
#: (ใส่ 60 แทน 6) ซึ่งจะกลายเป็น "ทั้งห้องเป็นทีมเดียว" โดยไม่มีใครทันสังเกต
MAX_TEAM_SIZE_CEILING = 20


class TeamSizeInvalid(Exception):
    """ขนาดทีมที่ตั้งใหม่ใช้ไม่ได้ — ข้อความต้องบอกว่าทำไมและต้องทำอะไรต่อ"""


#: ความยาวสูงสุดของชื่อบนกระดาน — ยาวกว่านี้ตารางจะเสียรูปบนมือถือ
MAX_ALIAS_LENGTH = 24

#: คำที่จองไว้ให้หมุด baseline — ทีมที่ตั้งชื่อว่า "Gold" จะอ่านเหมือนหมุดของผู้สอน
#: ซึ่งเป็นการเข้าใจผิดที่มีผลต่อการตัดสินใจของคนอื่น ไม่ใช่แค่เรื่องมารยาท
RESERVED_ALIASES = frozenset({"bronze", "silver", "gold", "diamond", "baseline"})


#: ชื่อวิชายาวได้กว่าชื่อทีม เพราะมันมีทั้งรหัสวิชา ชื่อเต็ม และภาคเรียน
MAX_COURSE_NAME_LENGTH = 60


class CourseNameInvalid(Exception):
    """ชื่อวิชาที่ตั้งใหม่ใช้ไม่ได้"""


class AliasInvalid(Exception):
    """ชื่อบนกระดานที่ตั้งใหม่ใช้ไม่ได้"""


def clean_alias(raw: str | None) -> str | None:
    """ตรวจและทำความสะอาดชื่อที่จะขึ้นกระดาน — คืน `None` ถ้าขอกลับไปใช้ชื่อจริง

    **ตัดอักขระควบคุมทิ้ง** เพราะ leaderboard วาดด้วย HTML และชื่อเดินทางผ่าน JSON
    อักขระอย่าง zero-width space ทำให้สองทีมมีชื่อที่ตาเห็นเหมือนกันได้
    """
    if raw is None:
        return None
    text = "".join(ch for ch in raw if ch.isprintable() and not ch.isspace() or ch == " ")
    text = " ".join(text.split())  # ยุบช่องว่างซ้ำ ตัดหัวท้าย
    if not text:
        return None  # ส่งค่าว่างมา = ขอใช้ชื่อจริง
    if len(text) > MAX_ALIAS_LENGTH:
        raise AliasInvalid(f"ชื่อยาวเกินไป — ไม่เกิน {MAX_ALIAS_LENGTH} ตัวอักษร (ตอนนี้ {len(text)})")
    if text.lower() in RESERVED_ALIASES:
        raise AliasInvalid(
            f"{text!r} เป็นชื่อของหมุด baseline บนกระดาน — เลือกชื่ออื่นที่ไม่ทำให้คนอื่นสับสน"
        )
    return text


@dataclass
class Course:
    """วิชาหนึ่งวิชาในเทอมหนึ่ง — เจ้าของกติกาที่ใช้ร่วมกันทุก competition ในวิชานั้น

    เกิดขึ้นเพราะ `MAX_TEAM_SIZE` เคยเป็นค่าคงที่ในโค้ด การจะเปลี่ยนขนาดทีมจึงต้อง
    แก้โค้ดแล้ว deploy ใหม่ ซึ่งไม่ใช่สิ่งที่ผู้สอนควรต้องทำกลางเทอม

    **ไม่มี `staff_emails` ที่นี่โดยตั้งใจ** — ว่าใครเป็นผู้สอนเป็นเรื่องของการตั้งค่า
    เครื่อง ไม่ใช่ข้อมูลที่แก้ผ่านหน้าเว็บได้ ไม่งั้นใครที่ยึดสิทธิ์ผู้สอนได้ครั้งเดียว
    จะแต่งตั้งตัวเองถาวร · มันอยู่ที่ `Arena.staff_emails` ซึ่งมาจาก `ARENA_STAFF_EMAILS`
    ใน `/etc/arena.env` เหมือน sudoers
    """

    id: str
    name: str
    max_team_size: int = DEFAULT_MAX_TEAM_SIZE
    #: รหัสที่ผู้สอนแจกในคาบ — นิสิตใส่เพื่อเข้าวิชา
    #:
    #: ใช้รูปแบบเดียวกับรหัสเชิญทีม เพราะทำหน้าที่เดียวกัน (อ่านให้ฟังในห้องได้)
    #: และเป็นสิ่งที่กันคนนอกวิชาหลุดเข้ามาโดยไม่ต้องอัปโหลดรายชื่อ
    join_code: str = field(default_factory=new_invite_code)
    #: วิชาที่จบเทอมแล้ว — ไม่รับคนเข้าใหม่ แต่ข้อมูลยังอยู่ครบ
    archived_at: datetime | None = None

    @property
    def is_open(self) -> bool:
        return self.archived_at is None

    def validated_team_size(self, size: int) -> int:
        """ตรวจค่าที่ผู้สอนกรอก — คืนค่าที่ใช้ได้ หรือโยนพร้อมเหตุผล"""
        if size < 1:
            raise TeamSizeInvalid("ขนาดทีมต้องอย่างน้อย 1 คน")
        if size > MAX_TEAM_SIZE_CEILING:
            raise TeamSizeInvalid(
                f"ขนาดทีมสูงสุดที่ตั้งได้คือ {MAX_TEAM_SIZE_CEILING} คน "
                f"— ถ้าต้องการมากกว่านี้จริง ต้องแก้ `MAX_TEAM_SIZE_CEILING` ในโค้ด"
            )
        return size


@dataclass
class Team:
    id: str
    course_id: str
    name: str
    alias: str | None = None  # ชื่อนิรนามบน leaderboard (README §6.1) ผู้สอนเห็นชื่อจริงเสมอ
    member_ids: list[str] = field(default_factory=list)
    #: credential ที่ `arena submit` ใช้ — แยกจาก `id` โดยตั้งใจ เดิมสองอย่างนี้เป็นตัวเดียวกัน
    #: ทำให้ id ที่เดาได้กลายเป็นรหัสผ่านที่เดาได้ไปด้วย
    token: str = field(default_factory=new_token)
    #: รหัสให้เพื่อนใช้เข้าทีม
    invite_code: str = field(default_factory=new_invite_code)
    #: ทีมเดี่ยวที่สมาชิกย้ายไปอยู่ทีมอื่นแล้ว — เก็บไว้เพื่อ audit แต่ไม่ขึ้น leaderboard
    dissolved_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.dissolved_at is None

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
    #: id ใน `PARADIGMS` — บอกว่านิสิต*กำลังเรียนอะไร* ต่างจาก `task_type`
    #: ที่บอกว่าระบบ*ให้คะแนนยังไง* · ดูเหตุผลที่ `class Paradigm`
    paradigm: str = "reinforcement-learning"
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
