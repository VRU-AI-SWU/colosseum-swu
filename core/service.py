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
    Competition,
    CompetitionClosed,
    QuotaExceeded,
    Run,
    RunKind,
    RunStatus,
    Submission,
    Team,
    User,
    new_id,
    utcnow,
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


class TeamFull(Exception):
    pass


#: ขนาดทีมสูงสุด — **นโยบายของวิชา ไม่ใช่ข้อจำกัดทางเทคนิค**
#: ผู้สอนกำหนด 6 คน (ส.ค. 2026) · ตั้งไว้ที่นี่ที่เดียวเพื่อให้เปลี่ยนง่าย
MAX_TEAM_SIZE = 6


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

    # ── ตัวตนและทีม ─────────────────────────────────────────────────

    def sign_in(self, *, google_sub: str, email: str, name: str, course_id: str) -> tuple:
        """ล็อกอินสำเร็จแล้วต้องมีทีมเสมอ — คืน `(user, team)`

        **สร้างทีมเดี่ยวให้อัตโนมัติตั้งแต่ครั้งแรก** ไม่มีหน้าจอ "กรุณาเลือกทีม" ให้ติด
        เพราะนิสิตที่หากลุ่มไม่ได้มักลงเอยด้วยการทำคนเดียวอยู่แล้ว — สถานะนั้นต้องเป็น
        เรื่องปกติที่ใช้งานได้ทันที ไม่ใช่ข้อผิดพลาดที่ต้องแก้ก่อนถึงจะเริ่มได้
        ส่วนคนที่จับกลุ่มได้ค่อยกดเข้าทีมเพื่อนทีหลัง
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

        team = self.store.team_of(user.id, course_id)
        if team is None:
            team = Team(
                id=new_id(), course_id=course_id, name=name, member_ids=[user.id]
            )
            self.store.save_team(team)
            self.store.record("team.created", "team", team.id, actor_id=user.id, solo=True)
        return user, team

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

        if len(target.member_ids) >= MAX_TEAM_SIZE:
            raise TeamFull(f"ทีมนี้เต็มแล้ว (สูงสุด {MAX_TEAM_SIZE} คน)")

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
) -> Arena:
    """ประกอบ Arena — ถ้าให้ `db_path` มา สถานะจะอยู่รอดข้ามการรีสตาร์ท

    ไม่ให้ `db_path` = ทำงานในหน่วยความจำล้วน ซึ่งเป็นสิ่งที่เทสต์ส่วนใหญ่ต้องการ
    (เร็วกว่า และไม่ต้องเก็บกวาดไฟล์)
    """
    db = Database(db_path) if db_path is not None else None
    store = Store(db=db)
    queue = JobQueue(db=db)
    if db is not None:
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
    )
