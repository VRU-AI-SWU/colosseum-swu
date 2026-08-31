"""ตรรกะทางธุรกิจของแพลตฟอร์ม — กติกาทั้งหมดอยู่ที่นี่ที่เดียว

ทำไมต้องมีชั้นนี้แยกจาก API: กติกาอย่างโควตา, หน้าต่างเวลา, 1 งานพร้อมกันต่อทีม
และ final pick สูงสุด 2 ชุด **ต้องบังคับได้เหมือนกันไม่ว่าคำสั่งจะมาจาก REST, CLI,
หรือหน้าเว็บของผู้สอน** ถ้ากระจายอยู่ใน handler ของแต่ละเส้นทาง มันจะเพี้ยนกันสักวัน

**core ยังไม่รู้จักโจทย์** — มันไม่รู้ว่า `env_plugin` คืออะไร รู้แค่ต้องส่งต่อให้ runner
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from core.domain import (
    DEFAULT_MAX_TEAM_SIZE,
    MAX_COURSE_NAME_LENGTH,
    AliasInvalid,
    CourseIdInvalid,
    CourseNameInvalid,
    PreferredNameInvalid,
    TeamNameInvalid,
    Competition,
    CompetitionClosed,
    Course,
    TeamSizeInvalid,
    QuotaExceeded,
    Run,
    RunKind,
    RunStatus,
    Submission,
    Team,
    User,
    new_id,
    clean_alias,
    clean_course_name,
    clean_preferred_name,
    clean_team_name,
    new_invite_code,
    new_token,
    require_paradigm,
    utcnow,
    valid_course_id,
)
from core.db import Database
from core.queue import JobQueue, check_quota
from core.store import ArtifactStore, Store, runs_of


class SubmissionRejected(Exception):
    """ไฟล์ไม่ผ่านการตรวจตอนอัพโหลด — ข้อความต้องบอกวิธีแก้เสมอ (§13)"""

    def __init__(self, problems: list):
        super().__init__("\n".join(str(p) for p in problems))
        self.problems = problems


class TooManyFinalPicks(Exception):
    pass


class InviteInvalid(Exception):
    """รหัสเชิญผิดหรือหมดอายุ — ข้อความต้องบอกให้ไปขอรหัสใหม่จากเพื่อน"""


class NotEnrolled(Exception):
    """คนนี้ยังไม่ได้เข้าวิชาที่กำลังจะทำงานด้วย — ข้อความต้องบอกวิธีเข้า"""


class TeamFull(Exception):
    pass


#: ขนาดทีมสูงสุด — **นโยบายของวิชา ไม่ใช่ข้อจำกัดทางเทคนิค**
#: ผู้สอนกำหนด 6 คน (ส.ค. 2026) · ตั้งไว้ที่นี่ที่เดียวเพื่อให้เปลี่ยนง่าย
#: เก็บชื่อเดิมไว้ให้โค้ดที่ import อยู่ — แต่ตอนนี้มันเป็นแค่ **ค่าเริ่มต้นของวิชาใหม่**
#: ไม่ใช่กฎที่ใช้ตัดสินอีกต่อไป · ตัวที่ใช้จริงคือ `Course.max_team_size`
MAX_TEAM_SIZE = DEFAULT_MAX_TEAM_SIZE


class ArchiveValidator(Protocol):
    """การตรวจไฟล์แบบ static — **เป็นของ task template ไม่ใช่ของ core**

    สัญญาว่าต้องมี `agent.py` ที่นิยาม `class Agent` เป็นเรื่องของ template
    agent-vs-environment ส่วนโจทย์ prediction-based ตรวจ `model.pkl` กับ `preprocess.py` แทน
    core จึงต้องไม่รู้จักทั้งคู่ — มันแค่เรียกตัวที่ถูกฉีดเข้ามาตาม `task_type` ของ competition
    """

    def __call__(self, archive_url: str, whitelist: frozenset[str]) -> Any: ...


@dataclass
class Arena:
    """หน้าตาเดียวที่ API และ CLI คุยด้วย"""

    store: Store
    queue: JobQueue
    artifacts: ArtifactStore
    #: task_type → ตัวตรวจไฟล์ · ประกอบตอน wiring ไม่ใช่ import ตรงๆ ใน core
    validators: dict[str, ArchiveValidator] = field(default_factory=dict)
    #: อีเมลของผู้สอน/TA — มาจาก `ARENA_STAFF_EMAILS` ใน `/etc/arena.env`
    #:
    #: **เป็นการตั้งค่าเครื่อง ไม่ใช่ข้อมูลในฐานข้อมูล** โดยตั้งใจ เหมือน sudoers —
    #: ถ้าเก็บในฐานข้อมูลแล้วแก้ผ่านหน้าเว็บได้ ใครที่ยึดสิทธิ์ผู้สอนได้ครั้งเดียว
    #: จะแต่งตั้งตัวเองถาวรและถอดคนอื่นออกได้ · ว่างไว้ = ไม่มีใครเป็นผู้สอน
    #: ซึ่งเป็นค่าเริ่มต้นที่ถูกต้อง (ปลอดภัยโดยปริยาย)
    staff_emails: frozenset[str] = frozenset()
    #: `course_id` → อีเมลของผู้สอนเฉพาะวิชานั้น · มาจาก `ARENA_COURSE_STAFF_<COURSE_ID>`
    #:
    #: อยู่ใน environment ด้วยเหตุผลเดียวกับ `staff_emails` — ถ้าเก็บในฐานข้อมูล
    #: แล้วแก้ผ่านหน้าเว็บได้ คนที่ยึดสิทธิ์ได้ครั้งเดียวจะแต่งตั้งตัวเองถาวร
    #:
    #: **คนใน `staff_emails` จัดการได้ทุกวิชาเสมอ** ไม่ต้องใส่ซ้ำที่นี่ — ไม่งั้น
    #: ผู้ดูแลระบบจะล็อกตัวเองออกจากวิชาที่ตัวเองไม่ได้สอนแต่ต้องเข้าไปแก้ตอนมีปัญหา
    course_staff: dict[str, frozenset[str]] = field(default_factory=dict)

    def is_staff(self, email: str) -> bool:
        """ผู้สอนระดับทั้งระบบ — จัดการได้ทุกวิชา"""
        return bool(email) and email.strip().lower() in self.staff_emails

    def can_manage_course(self, email: str, course_id: str) -> bool:
        """แก้ค่าของวิชานี้ได้ไหม — ผู้สอนของวิชานั้น หรือผู้สอนระดับทั้งระบบ"""
        if self.is_staff(email):
            return True
        if not email:
            return False
        return email.strip().lower() in self.course_staff.get(course_id, frozenset())

    def managed_courses(self, email: str) -> list[str]:
        """วิชาที่คนนี้จัดการได้ — หน้าเว็บใช้ตัดสินว่าจะแสดงแผงผู้สอนของวิชาไหน"""
        if self.is_staff(email):
            return sorted(self.store.courses)
        return sorted(c for c in self.store.courses if self.can_manage_course(email, c))

    def rotate_user_token(self, *, user: User) -> User:
        """ออกโทเคนใหม่ให้คนคนเดียว

        ยืนยันด้วยโทเคน*เดิม* ซึ่งฟังดูย้อนแย้งแต่ถูกต้อง: คนที่ยังถือโทเคนอยู่คือ
        เจ้าของโดยนิยาม และถ้ามีคนอื่นถือด้วยก็ยิ่งต้องรีบเปลี่ยน

        ต่างจากตอนโทเคนเป็นของทีม — ตอนนั้นเปลี่ยนทีเดียวกระทบทุกคนในทีม
        ทั้งที่หลุดคนเดียว เพื่อนต้องมาตั้งค่าใหม่โดยไม่ได้ทำอะไรผิด
        """
        user.token = new_token()
        self.store.save_user(user)
        self.store.record("user.token_rotated", "user", user.id, actor_id=user.id)
        return user

    # ── ตัวตนและทีม ─────────────────────────────────────────────────

    def sign_in(self, *, google_sub: str, email: str, name: str) -> User:
        """ล็อกอิน — คืน `User` เท่านั้น ไม่ผูกกับวิชา

        **เดิมรับ `course_id` แล้วสร้างทีมให้ทันที** ซึ่งใช้ได้ตอนมีวิชาเดียวเพราะ
        เดาถูกเสมอ · พอมีหลายวิชา การเดาจะพาคนไปอยู่ผิดวิชาโดยไม่มีใครรู้
        จึงแยกเป็นสองขั้น: ล็อกอินรู้ว่าเป็นใคร แล้วค่อย `enroll()` ด้วยรหัสเข้าวิชา

        ผลข้างเคียงที่ตั้งใจ: คนที่ล็อกอินแล้วยังไม่ได้ใส่รหัส จะยังไม่มีทีมและ
        ไม่โผล่ที่ไหนเลย ซึ่งถูกต้อง — คนนอกวิชาที่หลุดเข้ามาไม่ควรมีตัวตนในระบบ
        """
        user = self.store.user_by_google_sub(google_sub)
        if user is None:
            user = User(id=new_id(), email=email, name=name, google_sub=google_sub)
            self.store.save_user(user)
            self.store.record("user.created", "user", user.id, email=email)
        elif (user.email, user.name) != (email, name):
            # อีเมลกับชื่อเปลี่ยนได้ (เปลี่ยนนามสกุล ฯลฯ) — จับคู่ด้วย sub จึงตามได้
            user.email, user.name = email, name
            self.store.save_user(user)

        return user

    def enroll(self, *, user: User, join_code: str) -> Team:
        """เข้าวิชาด้วยรหัสที่ผู้สอนแจกในคาบ — ได้ทีมเดี่ยวทันที

        **สร้างทีมเดี่ยวให้ตั้งแต่แรก** ไม่มีหน้าจอ "กรุณาเลือกทีม" ให้ติด เพราะนิสิต
        ที่หากลุ่มไม่ได้มักลงเอยด้วยการทำคนเดียวอยู่แล้ว — สถานะนั้นต้องเป็นเรื่องปกติ
        ที่ใช้งานได้ทันที ไม่ใช่ข้อผิดพลาดที่ต้องแก้ก่อนถึงจะเริ่มได้
        ส่วนคนที่จับกลุ่มได้ค่อยกดเข้าทีมเพื่อนทีหลัง
        """
        course = self.store.course_by_join_code(join_code)
        if course is None:
            raise InviteInvalid(
                "ไม่พบวิชาจากรหัสนี้ — ตรวจตัวอักษรอีกครั้ง หรือขอรหัสจากผู้สอน"
            )

        existing = self.store.team_of(user.id, course.id)
        if existing is not None:
            return existing  # อยู่ในวิชานี้อยู่แล้ว — ไม่ใช่ข้อผิดพลาด

        team = Team(
            id=new_id(), course_id=course.id, name=user.name, member_ids=[user.id]
        )
        self.store.save_team(team)
        self.store.record(
            "team.created", "team", team.id, actor_id=user.id, solo=True, course=course.id
        )
        return team

    def team_for(self, *, user: User, course_id: str) -> Team:
        """ทีมของคนนี้ในวิชานั้น — ใช้แปลง "โทเคนของคน" เป็น "ทีมที่กำลังทำงานแทน"

        โยนเมื่อยังไม่ได้เข้าวิชา แทนที่จะสร้างทีมให้เงียบๆ — การสร้างทีมโดยไม่มีใคร
        ขอ แปลว่าคนที่พิมพ์ slug ผิดจะได้ทีมในวิชาที่ไม่ได้เรียนโดยไม่รู้ตัว
        """
        team = self.store.team_of(user.id, course_id)
        if team is None:
            course = self.store.courses.get(course_id)
            raise NotEnrolled(
                f"ยังไม่ได้เข้าวิชา {course.name if course else course_id} — "
                "ใส่รหัสเข้าวิชาที่หน้าเว็บก่อน (ขอรหัสจากผู้สอน)"
            )
        return team

    def set_preferred_name(self, *, user: User, raw: str | None, actor_id: str | None) -> User:
        """ตั้งชื่อที่อยากให้เพื่อนร่วมชั้นเรียก — ส่งค่าว่างมาเพื่อกลับไปใช้ชื่อจาก Google

        **ไม่ทับชื่อจริง แค่บังไว้จากเพื่อน** — `User.name` จาก Google ยังอยู่ครบ
        และผู้สอนของวิชานั้นเห็นเสมอ · ถ้าให้ทับได้จริง นิสิตจะตั้งชื่อเป็นชื่อเพื่อน
        แล้วผู้สอนแยกไม่ออกว่าใครส่งงาน ซึ่งกระทบการตัดเกรด ไม่ใช่แค่ความเป็นส่วนตัว
        """
        name = clean_preferred_name(raw)
        before, user.preferred_name = user.preferred_name, name
        if before == name:
            return user
        self.store.save_user(user)
        self.store.record(
            "user.renamed", "user", user.id, actor_id=actor_id, before=before, after=name,
        )
        return user

    def rename_team(self, *, team: Team, raw: str, actor_id: str | None) -> Team:
        """ทีมเปลี่ยนชื่อตัวเอง

        ชื่อทีมเริ่มต้นเป็นชื่อ-นามสกุลของคนที่เข้าวิชาคนแรก ซึ่งอ่านแปลกทันทีที่
        มีเพื่อนเข้ามาร่วม — ทีมสามคนที่ชื่อว่าคนหนึ่งในสามคนนั้น

        ใช้กติกาเรื่องชื่อซ้ำเดียวกับ `set_alias` เพราะชื่อทีมขึ้นกระดานได้เหมือนกัน
        (ทีมที่ยังไม่ได้ตั้ง alias) — สองแถวที่ชื่อเหมือนกันทำให้คนอ่านผิดได้จริง
        """
        name = clean_team_name(raw)
        taken = {
            other_name.casefold()
            for other in self.store.teams.values()
            if other.is_active and other.course_id == team.course_id and other.id != team.id
            for other_name in (other.name, other.alias)
            if other_name
        }
        if name.casefold() in taken:
            raise TeamNameInvalid(f"มีทีมอื่นใช้ชื่อ {name!r} อยู่แล้ว — เลือกชื่อที่ไม่ซ้ำ")

        before, team.name = team.name, name
        if before == name:
            return team
        self.store.save_team(team)
        self.store.record(
            "team.renamed", "team", team.id, actor_id=actor_id, before=before, after=name,
        )
        return team

    def set_alias(self, *, team: Team, raw: str | None, actor_id: str | None) -> Team:
        """ทีมตั้งชื่อที่จะแสดงบนกระดานเอง — README §6.1 "ลดแรงกดดันของทีมท้ายตาราง"

        ปฏิเสธชื่อที่ซ้ำกับทีมอื่นในวิชาเดียวกัน **โดยเทียบทั้งชื่อจริงและชื่อบนกระดาน**
        ของทีมที่ยังใช้งานอยู่ · ไม่ใช่เรื่องมารยาท — กระดานคือที่ที่คนใช้ตัดสินใจว่า
        ตัวเองอยู่ตรงไหน สองแถวที่ชื่อเหมือนกันทำให้อ่านผิดได้จริง

        ผู้สอนยังเห็นชื่อจริงเสมอ ตัวนี้จึงไม่ใช่การซ่อนตัวจากการตรวจ
        """
        alias = clean_alias(raw)
        if alias is not None:
            taken = {
                name.casefold()
                for other in self.store.teams.values()
                if other.is_active and other.course_id == team.course_id and other.id != team.id
                for name in (other.name, other.alias)
                if name
            }
            if alias.casefold() in taken:
                raise AliasInvalid(
                    f"มีทีมอื่นใช้ชื่อ {alias!r} อยู่แล้ว — เลือกชื่อที่ไม่ซ้ำ"
                )

        before, team.alias = team.alias, alias
        if before == alias:
            return team
        self.store.save_team(team)
        self.store.record(
            "team.alias", "team", team.id, actor_id=actor_id,
            before=before, after=alias,
        )
        return team

    def create_course(
        self, *, course_id: str, name: str, max_team_size: int, actor_id: str | None
    ) -> Course:
        """สร้างวิชาใหม่ — กติกาของ id และชื่ออยู่ที่ `core/domain.py` ที่เดียว

        **ไม่ตั้งใครเป็นผู้สอนของวิชานี้** — สิทธิ์มาจาก `ARENA_COURSE_STAFF_<ID>`
        ใน environment เท่านั้น (เหมือน sudoers) คนที่สร้างจึงต้องเป็นผู้สอนระดับ
        ทั้งระบบอยู่แล้ว ซึ่งจัดการทุกวิชาได้โดยนิยาม · ถ้าให้คนสร้างกลายเป็นผู้สอน
        อัตโนมัติ ใครก็ตามที่สร้างวิชาได้จะแต่งตั้งตัวเองได้ ซึ่งคือช่องที่ทั้ง
        การออกแบบพยายามปิด
        """
        course_id = valid_course_id(course_id)
        if course_id in self.store.courses:
            raise CourseIdInvalid(f"มีวิชา {course_id!r} อยู่แล้ว")

        course = Course(
            id=course_id,
            name=clean_course_name(name),
            max_team_size=Course(id=course_id, name="x").validated_team_size(max_team_size),
            join_code=new_invite_code(),
        )
        self.store.save_course(course)
        self.store.record("course.created", "course", course.id, actor_id=actor_id, name=course.name)
        return course

    def create_competition(
        self,
        *,
        slug: str,
        course_id: str,
        title: str,
        task_type: str,
        env_plugin: str,
        config_text: str,
        paradigm: str,
        ranges: dict[str, tuple[str, str]],
        quota_per_day: int = 5,
        import_whitelist: frozenset[str] = frozenset(),
        actor_id: str | None,
    ) -> Competition:
        """สร้าง competition ใหม่ — **config เดินทางมาเป็นเนื้อหา ไม่ใช่ path**

        ผู้เรียกต้องตรวจ config กับ plugin จริงมาก่อนแล้ว (`core/wiring.prepare_config`)
        ที่นี่ไม่รู้จัก environment ใดเลยตามหลักของ `core/`
        """
        from core.calendar import build_phases, day_range

        slug = valid_course_id(slug)  # กติกาเดียวกัน — slug ก็โผล่ใน URL
        if self.store.competition_by_slug(slug) is not None:
            raise CourseIdInvalid(f"มี competition {slug!r} อยู่แล้ว")
        if course_id not in self.store.courses:
            raise CourseIdInvalid(f"ไม่รู้จักวิชา {course_id!r}")
        require_paradigm(paradigm)

        phases = build_phases({name: day_range(*pair) for name, pair in ranges.items()})
        competition = Competition(
            id=new_id(),
            course_id=course_id,
            slug=slug,
            title=title.strip() or slug,
            task_type=task_type,
            env_plugin=env_plugin,
            config_path="",          # ไม่มีไฟล์บนเครื่อง — นี่คือจุดที่ v5 ปลดล็อก
            config_text=config_text,
            opens_at=phases[0].starts_at,
            closes_at=phases[-1].ends_at,
            quota_per_day=quota_per_day,
            import_whitelist=frozenset(import_whitelist),
            paradigm=paradigm,
            phases=phases,
        )
        self.store.save_competition(competition)
        self.store.record(
            "competition.created", "competition", competition.id,
            actor_id=actor_id, slug=slug, course=course_id, task_type=task_type,
        )
        return competition

    def update_course(
        self, *, course_id: str, size: int | None, name: str | None, actor_id: str | None
    ) -> Course:
        """แก้ค่าของวิชาเท่าที่ส่งมา — `None` แปลว่า "ไม่แตะ" ไม่ใช่ "ล้างทิ้ง"

        แยกความหมายนี้ให้ชัด เพราะฟอร์มที่ส่งเฉพาะฟิลด์ที่แก้เป็นเรื่องปกติ
        แต่การตีความค่าที่ไม่ได้ส่งว่า "ล้าง" จะลบชื่อวิชาทิ้งโดยไม่มีใครขอ
        """
        course = self.store.course(course_id)
        if name is not None:
            cleaned = clean_course_name(name)
            if cleaned != course.name:
                before, course.name = course.name, cleaned
                self.store.save_course(course)
                self.store.record(
                    "course.renamed", "course", course.id,
                    actor_id=actor_id, before=before, after=cleaned,
                )
        if size is not None:
            self.set_max_team_size(course_id=course_id, size=size, actor_id=actor_id)
        return self.store.course(course_id)

    def set_calendar(
        self,
        *,
        slug: str,
        ranges: dict[str, tuple[str, str]],
        actor_id: str | None,
    ) -> Competition:
        """ตั้งปฏิทินของ competition ใหม่ — กติกาเรื่องวันมาจาก `core/calendar.py`

        **คง `id` ของ competition ไว้เสมอ** — สร้างใหม่จะทำให้ run ที่ส่งไปแล้วกำพร้า

        ⚠️ **ไม่ตรวจว่า run เก่ายังอยู่ในช่วงไหม** — ปล่อยให้ผู้เรียกเป็นคนเตือน
        เพราะการเลื่อนปฏิทินหลังมีคนส่งงานแล้วเป็นการตัดสินใจของผู้สอน ไม่ใช่ข้อผิดพลาด
        (งานที่หลุดออกนอกทุกช่วงจะถูก `phase_at` มองว่าไม่มี phase แล้วถอยไปใช้ 'main')
        """
        from core.calendar import PHASES, build_phases, day_range

        competition = self._competition(slug)
        parsed = {name: day_range(*ranges[name]) for name in PHASES if name in ranges}
        # เก็บ `config_override` เดิมของแต่ละ phase ไว้ — ปฏิทินกับ config เป็นคนละเรื่อง
        # ผู้สอนที่แค่เลื่อนวันต้องไม่เผลอล้างค่าที่ทำให้แต่ละ phase ยากต่างกัน
        keep = {p.name: dict(p.config_override) for p in competition.phases}
        phases = build_phases(parsed, overrides=keep)

        competition.phases = phases
        competition.opens_at = min(competition.opens_at, phases[0].starts_at)
        competition.closes_at = phases[-1].ends_at
        self.store.save_competition(competition)
        self.store.record(
            "competition.calendar_changed", "competition", competition.id,
            actor_id=actor_id,
            phases={name: list(ranges[name]) for name in PHASES if name in ranges},
        )
        return competition

    def set_max_team_size(self, *, course_id: str, size: int, actor_id: str | None) -> Course:
        """ผู้สอนเปลี่ยนขนาดทีมของวิชา — **ผู้เรียกต้องตรวจสิทธิ์มาก่อนแล้ว**

        ปฏิเสธถ้ามีทีมที่ใหญ่เกินค่าใหม่อยู่แล้ว แทนที่จะยอมแล้วปล่อยให้ทีมนั้น
        เกินโควตาต่อไปเงียบๆ · สถานะที่ข้อมูลขัดกับกฎของตัวเองเป็นสิ่งที่อธิบาย
        ให้นิสิตฟังไม่ได้ และจะกลายเป็นข้อโต้แย้งตอนตัดเกรด · ข้อความบอกชื่อทีม
        ที่เป็นปัญหาไปเลย ผู้สอนจะได้ตัดสินใจได้โดยไม่ต้องไปไล่หาเอง
        """
        course = self.store.course(course_id)
        size = course.validated_team_size(size)

        too_big = sorted(
            (t for t in self.store.teams.values()
             if t.is_active and t.course_id == course_id and len(t.member_ids) > size),
            key=lambda t: -len(t.member_ids),
        )
        if too_big:
            names = ", ".join(f"{t.name} ({len(t.member_ids)} คน)" for t in too_big[:3])
            more = f" และอีก {len(too_big) - 3} ทีม" if len(too_big) > 3 else ""
            raise TeamSizeInvalid(
                f"ลดเหลือ {size} คนไม่ได้ เพราะมีทีมที่ใหญ่กว่านั้นอยู่แล้ว — {names}{more}\n"
                "ให้สมาชิกย้ายออกจนทีมเล็กพอก่อน หรือตั้งค่าที่ไม่ต่ำกว่าทีมที่ใหญ่ที่สุด"
            )

        before = course.max_team_size
        if before == size:
            return course
        course.max_team_size = size
        self.store.save_course(course)
        self.store.record(
            "course.max_team_size", "course", course.id,
            actor_id=actor_id, before=before, after=size,
        )
        return course

    def join_team(self, *, user: User, invite_code: str, course_id: str) -> Team:
        """ย้ายเข้าทีมของเพื่อนด้วยรหัสเชิญ

        ทีมเดิมที่ว่างลงจะถูก**ยุบ** ไม่ใช่ลบ — งานที่เคยส่งไปยังอยู่ใน audit trail
        ตรวจย้อนหลังได้ แต่หายจาก leaderboard เพราะมันคือผลงานของคนเดียว
        ไม่ใช่ของทีมใหม่ ([README §7](../README.md))
        """
        target = self.store.team_by_invite_code(invite_code)
        if target is None or target.course_id != course_id:
            raise InviteInvalid(
                "ไม่พบทีมจากรหัสนี้ — ตรวจตัวอักษรอีกครั้ง หรือขอรหัสใหม่จากเพื่อนในทีม"
            )

        current = self.store.team_of(user.id, course_id)
        if current is not None and current.id == target.id:
            return target  # อยู่ทีมนี้อยู่แล้ว — ไม่ใช่ข้อผิดพลาด

        limit = self.store.course(course_id).max_team_size
        if len(target.member_ids) >= limit:
            raise TeamFull(f"ทีมนี้เต็มแล้ว (สูงสุด {limit} คน)")

        if current is not None:
            current.member_ids = [m for m in current.member_ids if m != user.id]
            if not current.member_ids:
                current.dissolved_at = utcnow()
                self.store.record(
                    "team.dissolved", "team", current.id, actor_id=user.id,
                    reason="สมาชิกคนสุดท้ายย้ายไปทีมอื่น", moved_to=target.id,
                )
            self.store.save_team(current)

        target.member_ids = [*target.member_ids, user.id]
        self.store.save_team(target)
        self.store.record("team.joined", "team", target.id, actor_id=user.id)
        return target

    def rotate_token(self, *, team: Team, actor_id: str | None = None) -> Team:
        """ออกโทเคนใหม่ให้ทีม — โทเคนเดิมใช้ไม่ได้ทันที

        มีไว้สำหรับกรณีที่โทเคนหลุด ซึ่งเรื่องที่พบบ่อยที่สุดคือนิสิตเผลอ commit
        ค่ามันขึ้น GitHub · ก่อนมีปุ่มนี้ ทางแก้เดียวคือผู้สอนเข้าไปแก้ฐานข้อมูลเอง

        ⚠️ **กระทบทั้งทีม** — เพื่อนร่วมทีมที่ตั้ง `ARENA_TOKEN` ไว้แล้วจะใช้ไม่ได้
        จนกว่าจะมาเอาค่าใหม่ · ผลนี้ย้อนกลับไม่ได้ จึงต้องเตือนก่อนกด ไม่ใช่หลังกด
        """
        team.token = new_token()
        self.store.save_team(team)
        self.store.record("team.token_rotated", "team", team.id, actor_id=actor_id)
        return team

    # ── ส่งงาน ──────────────────────────────────────────────────────

    def submit(
        self,
        *,
        slug: str,
        team: Team,
        user_id: str,
        archive: bytes,
        note: str = "",
        dry_run: bool = False,
        now: datetime | None = None,
    ) -> tuple[Submission, Run]:
        """อัพโหลด + ตรวจแบบ static + เข้าคิว

        ⚠️ **ไม่รันโค้ดนิสิตที่นี่** — เมธอดนี้ทำงานอยู่บน cloud API ซึ่งรับไฟล์จาก
        อินเทอร์เน็ต การตรวจที่ต้องรันโค้ดเกิดที่ runner ในภายหลัง (smoke test)
        """
        now = now or utcnow()
        competition = self._competition(slug)

        if not competition.is_open(now):
            raise CompetitionClosed(
                f"competition {slug} รับ submission ระหว่าง "
                f"{competition.opens_at:%Y-%m-%d %H:%M} ถึง {competition.closes_at:%Y-%m-%d %H:%M} เท่านั้น"
            )

        kind = RunKind.DRYRUN if dry_run else RunKind.PUBLIC
        if kind is RunKind.PUBLIC:
            remaining = check_quota(
                runs_of(self.queue.runs, team.id, competition.id),
                team.id,
                competition.quota_per_day,
                now=now,
            )
            if remaining <= 0:
                raise QuotaExceeded(
                    f"ทีมนี้ส่งครบ {competition.quota_per_day} ครั้งของวันนี้แล้ว "
                    f"— dry run ไม่กินโควตา ใช้ทดสอบว่าแพ็กไฟล์ถูกได้"
                )

        url, digest = self.artifacts.put_submission(archive)
        validator = self.validators.get(competition.task_type)
        if validator is None:
            raise KeyError(
                f"ไม่มีตัวตรวจไฟล์สำหรับ task_type {competition.task_type!r} — "
                f"ต้องลงทะเบียนตอน wiring"
            )
        report = validator(url, competition.effective_whitelist())
        if not report.ok:
            self.store.record(
                "submission.rejected", "team", team.id,
                actor_id=user_id, sha256=digest, problems=[p.code for p in report.problems],
            )
            raise SubmissionRejected(report.problems)

        submission = Submission(
            id=new_id(),
            competition_id=competition.id,
            team_id=team.id,
            submitted_by=user_id,
            artifact_url=url,
            artifact_sha256=digest,
            note=note,
            created_at=now,
        )
        self.store.save_submission(submission)

        run = self.queue.enqueue(
            Run(
                id=new_id(),
                submission_id=submission.id,
                competition_id=competition.id,
                team_id=team.id,
                kind=kind,
                created_at=now,
            )
        )
        self.store.record(
            "submission.created", "submission", submission.id,
            actor_id=user_id, run_id=run.id, kind=kind.value, sha256=digest,
        )
        return submission, run

    # ── final pick ──────────────────────────────────────────────────

    def set_final_pick(
        self, *, submission_id: str, team: Team, picked: bool, user_id: str
    ) -> Submission:
        """เลือก submission ที่จะเอาไปรันบน private — **กติกาที่ขาดไม่ได้** (template)

        ถ้าให้ระบบเป็นคนหยิบโดยดูคะแนน private เท่ากับให้ทีมที่ส่งเยอะจับฉลากเยอะกว่า
        ซึ่งกลับไปเป็นปัญหาเดิมที่ private leaderboard มีไว้แก้พอดี
        """
        submission = self.store.submissions[submission_id]
        if submission.team_id != team.id:
            raise PermissionError("เลือก submission ของทีมอื่นไม่ได้")

        competition = self.store.competitions[submission.competition_id]
        if picked:
            current = [
                s
                for s in self.store.submissions_of(team.id, competition.id)
                if s.is_final_pick and s.id != submission_id
            ]
            if len(current) >= competition.max_final_submissions:
                raise TooManyFinalPicks(
                    f"เลือกได้สูงสุด {competition.max_final_submissions} ชุด "
                    f"— เอาชุดเดิมออกก่อนถ้าจะเปลี่ยน"
                )

        submission.is_final_pick = picked
        self.store.save_submission(submission)
        self.store.record(
            "submission.final_pick", "submission", submission_id, actor_id=user_id, picked=picked
        )
        return submission

    def final_picks(self, *, team_id: str, competition_id: str) -> list[Submission]:
        """ทีมที่ไม่เลือกเอง → ระบบใช้ตัวที่คะแนน public สูงสุดเป็นค่าเริ่มต้น"""
        chosen = [s for s in self.store.submissions_of(team_id, competition_id) if s.is_final_pick]
        if chosen:
            return chosen

        best = self._best_public_run(team_id, competition_id)
        return [self.store.submissions[best.submission_id]] if best else []

    def _best_public_run(self, team_id: str, competition_id: str) -> Run | None:
        candidates = [
            r
            for r in runs_of(self.queue.runs, team_id, competition_id)
            if r.kind is RunKind.PUBLIC and r.status is RunStatus.DONE and r.score is not None
        ]
        return max(candidates, key=lambda r: (r.score, *r.tiebreak), default=None)

    # ── ปิด competition แล้วรัน private ─────────────────────────────

    def close_and_enqueue_private(self, *, slug: str, actor_id: str) -> list[Run]:
        """รันเฉพาะ final pick ของแต่ละทีมบน private seeds

        **ห้ามรันทุก submission แล้วเลือกอันที่ดีที่สุด** — เท่ากับให้ทีมที่ส่งเยอะ
        จับฉลากเยอะกว่า ซึ่งทำให้ private พังด้วยเหตุผลเดียวกับ public เป๊ะ
        """
        competition = self._competition(slug)
        created: list[Run] = []
        for team_id in {s.team_id for s in self.store.submissions.values()
                        if s.competition_id == competition.id}:
            for submission in self.final_picks(team_id=team_id, competition_id=competition.id):
                run = Run(
                    id=new_id(),
                    submission_id=submission.id,
                    competition_id=competition.id,
                    team_id=team_id,
                    kind=RunKind.PRIVATE,
                )
                # ข้ามกติกา 1 งาน/ทีม เพราะรอบนี้ผู้สอนเป็นคนสั่ง ไม่ใช่นิสิตส่งเอง
                self.queue.adopt(run)
                created.append(run)
        self.store.record(
            "competition.private_run", "competition", competition.id,
            actor_id=actor_id, runs=len(created),
        )
        return created

    # ── อ่านสถานะ ───────────────────────────────────────────────────

    def submission_status(self, submission_id: str) -> dict:
        submission = self.store.submissions[submission_id]
        runs = [r for r in self.queue.runs.values() if r.submission_id == submission_id]
        return {
            "submission": submission,
            "runs": sorted(runs, key=lambda r: r.created_at),
            "queue_position": next(
                (self.queue.position_of(r.id) for r in runs if r.status is RunStatus.QUEUED), None
            ),
        }

    def quota_left(self, *, slug: str, team_id: str, now: datetime | None = None) -> int:
        competition = self._competition(slug)
        return check_quota(
            runs_of(self.queue.runs, team_id, competition.id),
            team_id,
            competition.quota_per_day,
            now=now or utcnow(),
        )

    def _competition(self, slug: str) -> Competition:
        competition = self.store.competition_by_slug(slug)
        if competition is None:
            raise KeyError(f"ไม่รู้จัก competition {slug!r}")
        return competition


def build_arena(
    root: Path,
    validators: dict[str, Callable] | None = None,
    *,
    db_path: Path | str | None = None,
    staff_emails: frozenset[str] = frozenset(),
    course_staff: dict[str, frozenset[str]] | None = None,
) -> Arena:
    """ประกอบ Arena — ถ้าให้ `db_path` มา สถานะจะอยู่รอดข้ามการรีสตาร์ท

    ไม่ให้ `db_path` = ทำงานในหน่วยความจำล้วน ซึ่งเป็นสิ่งที่เทสต์ส่วนใหญ่ต้องการ
    (เร็วกว่า และไม่ต้องเก็บกวาดไฟล์)
    """
    db = Database(db_path) if db_path is not None else None
    store = Store(db=db)
    queue = JobQueue(db=db)
    if db is not None:
        store.courses = db.load_courses()
        store.teams = db.load_teams()
        store.users = db.load_users()
        store.competitions = db.load_competitions()
        store.submissions = db.load_submissions()
        store.audit = db.load_audit()
        queue.runs = db.load_runs()
        queue._served = db.load_served()
    return Arena(
        store=store,
        queue=queue,
        artifacts=ArtifactStore(Path(root)),
        validators=dict(validators or {}),
        staff_emails=staff_emails,
        course_staff=dict(course_staff or {}),
    )
