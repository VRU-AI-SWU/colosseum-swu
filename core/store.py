"""ที่เก็บข้อมูลในหน่วยความจำ (write-through ลง SQLite) + artifact ลงดิสก์

**ตั้งใจให้เป็นของที่ถอดออกได้** — ตรรกะทางธุรกิจทั้งหมดอยู่ที่ `core/service.py`
ซึ่งคุยกับที่นี่ผ่านเมธอดไม่กี่ตัว การย้ายไป Postgres จึงเป็นการเขียน class ใหม่ที่มี
เมธอดชุดเดียวกัน ไม่ใช่การรื้อ service

ถ้าส่ง `db` เข้ามา ทุกการเปลี่ยนแปลงจะถูกเขียนลง SQLite ทันทีและอ่านกลับได้ตอนเริ่ม
([`core/db.py`](db.py)) · ถ้าไม่ส่ง มันทำงานในหน่วยความจำล้วนเหมือนเดิม ซึ่งเป็น
สิ่งที่เทสต์ส่วนใหญ่ต้องการ

⚠️ **การแก้ object โดยตรงจะไม่ถูกบันทึก** — ต้องเรียก `save_*` ทุกครั้งหลังแก้
เป็นราคาของการเลือก write-through แทน query layer (เหตุผลอยู่ใน `core/db.py`)
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from core.db import Database
from core.domain import (
    AuditEvent,
    Competition,
    Course,
    Run,
    Submission,
    Team,
    User,
    new_id,
    utcnow,
)


@dataclass
class ArtifactStore:
    """เก็บ zip ที่นิสิตอัพโหลด + ไฟล์ replay ที่ runner สร้าง

    บนของจริงเป็น S3-compatible (README §11) — อินเทอร์เฟซเหมือนกัน
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        (self.root / "submissions").mkdir(parents=True, exist_ok=True)
        (self.root / "replays").mkdir(parents=True, exist_ok=True)

    def put_submission(self, data: bytes) -> tuple[str, str]:
        """คืน `(url, sha256)` — hash ใช้ตรวจสอบย้อนหลังว่าคะแนนมาจากไฟล์ไหน (§7)"""
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / "submissions" / f"{digest}.zip"
        if not path.exists():
            path.write_bytes(data)
        return str(path), digest

    def extract(self, url: str, into: Path, entry: str = "agent.py") -> Path:
        """แตก zip ไปยังโฟลเดอร์ที่จะ mount เข้า sandbox

        กัน path traversal ที่นี่อีกชั้น ถึงแม้ `validate.inspect_archive` จะตรวจไปแล้ว —
        ชั้นนี้เป็นตัวที่เขียนไฟล์ลงดิสก์จริง จึงเป็นด่านสุดท้ายที่ต้องปลอดภัยด้วยตัวเอง

        `entry` คือชื่อไฟล์ทางเข้าของโจทย์ (`agent.py` · `predictor.py`) — **ต้องตรงกับ
        ที่ตัวตรวจ zip ใช้เสมอ** เพราะสองฝั่งนี้เป็นกติกาเดียวกัน ถ้าที่นี่หาไม่เจอแต่
        ตัวตรวจยอมรับ submission จะผ่านการตรวจ กินโควตา แล้วไปตายตอนรัน
        (`runners/tests/test_validate.py` ผูกสองฝั่งไว้ด้วยกัน)
        """
        into.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(url) as zf:
            for info in zf.infolist():
                target = (into / info.filename).resolve()
                if not str(target).startswith(str(into.resolve())):
                    raise ValueError(f"path ใน zip ออกนอกโฟลเดอร์: {info.filename}")
            zf.extractall(into)
        self._make_readable_by_the_sandbox(into)
        # รองรับ zip ที่ห่อด้วยโฟลเดอร์ชั้นเดียว (นิสิต zip ทั้งโฟลเดอร์มา)
        if not (into / entry).exists():
            nested = [p for p in into.iterdir() if p.is_dir() and (p / entry).exists()]
            if len(nested) == 1:
                return nested[0]
        return into

    @staticmethod
    def _make_readable_by_the_sandbox(root: Path) -> None:
        """เปิดสิทธิ์อ่านให้ **ผู้ใช้อื่น** เพราะ container รันด้วย uid คนละตัว (10001)

        ⚠️ **ไม่ใช่การหย่อนความปลอดภัย** — โฟลเดอร์นี้เป็นที่ทำงานชั่วคราวที่มีแต่
        โค้ดของนิสิตเอง ซึ่งกำลังจะถูก mount เข้า container ให้มันอ่านอยู่แล้ว

        เดิมพึ่ง umask ของ process ที่รัน worker ล้วนๆ ซึ่งบังเอิญใช้ได้ (0002 → 0775)
        แต่แปลว่าการตั้ง `UMask=0077` ใน systemd unit วันหนึ่งจะทำให้ submission
        **ทุกอัน**ล้มพร้อมกันด้วย `PermissionError: '/submission'` โดยไม่มีใครแก้โค้ด
        ตัวเอง · เจอตอนรันเทสต์ sandbox ของ CP462 บนเครื่องจริงครั้งแรก

        โหมดของไฟล์ใน zip ก็เชื่อไม่ได้เหมือนกัน — zip เก็บ mode มาด้วยได้ และ
        `extractall` เอามาใช้ ไฟล์ที่นิสิต zip มาแบบ 0600 จะอ่านไม่ได้ในกล่อง
        """
        for path in [root, *root.rglob("*")]:
            try:
                mode = path.stat().st_mode
                # ให้สิทธิ์อ่านตามที่เจ้าของมี และให้สิทธิ์เข้าโฟลเดอร์ตามไปด้วย
                path.chmod(mode | (0o755 if path.is_dir() else 0o444))
            except OSError:
                continue  # ไฟล์แปลกๆ ใน zip ไม่ควรทำให้ทั้ง run ล้ม

    def replay_path(self, run_id: str) -> Path:
        path = self.root / "replays" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def clear_workdir(self, path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)


@dataclass
class Store:
    courses: dict[str, Course] = field(default_factory=dict)
    teams: dict[str, Team] = field(default_factory=dict)
    competitions: dict[str, Competition] = field(default_factory=dict)
    submissions: dict[str, Submission] = field(default_factory=dict)
    users: dict[str, User] = field(default_factory=dict)
    audit: list[AuditEvent] = field(default_factory=list)
    #: ผู้สอน/TA ที่**แต่งตั้งผ่านหน้าเว็บ** — `{course_id: {อีเมล}}`
    #:
    #: เก็บเป็นอีเมลไม่ใช่ `user_id` เพราะแต่งตั้งคนที่ยังไม่เคยล็อกอินได้ ซึ่งเป็น
    #: กรณีปกติ: ผู้สอนเพิ่ม TA ไว้ก่อนเปิดเทอม แล้ว TA ค่อยล็อกอินทีหลัง
    course_staff: dict[str, set[str]] = field(default_factory=dict)
    db: Database | None = None

    # ── เขียน (ต้องเรียกทุกครั้งที่แก้ ไม่งั้นของหายตอนรีสตาร์ท) ──────

    def save_course(self, course: Course) -> Course:
        self.courses[course.id] = course
        if self.db:
            self.db.save_course(course)
        return course

    def add_course_staff(self, course_id: str, email: str, *, added_by: str = "") -> None:
        from datetime import datetime, timezone

        email = email.strip().lower()
        self.course_staff.setdefault(course_id, set()).add(email)
        if self.db:
            self.db.add_course_staff(
                course_id, email, added_by, datetime.now(timezone.utc).isoformat()
            )

    def remove_course_staff(self, course_id: str, email: str) -> None:
        email = email.strip().lower()
        self.course_staff.get(course_id, set()).discard(email)
        if self.db:
            self.db.remove_course_staff(course_id, email)

    def save_team(self, team: Team) -> Team:
        self.teams[team.id] = team
        if self.db:
            self.db.save_team(team)
        return team

    def save_competition(self, competition: Competition) -> Competition:
        self.competitions[competition.id] = competition
        if self.db:
            self.db.save_competition(competition)
        return competition

    def delete_competition(self, competition_id: str) -> None:
        self.competitions.pop(competition_id, None)
        if self.db:
            self.db.delete_competition(competition_id)

    def save_submission(self, submission: Submission) -> Submission:
        self.submissions[submission.id] = submission
        if self.db:
            self.db.save_submission(submission)
        return submission

    def save_user(self, user: User) -> User:
        self.users[user.id] = user
        if self.db:
            self.db.save_user(user)
        return user

    # ── lookup ──────────────────────────────────────────────────────

    def course(self, course_id: str) -> Course:
        """คืนวิชานั้น — สร้างให้ด้วยค่าเริ่มต้นถ้ายังไม่มี

        **ไม่โยนเมื่อหาไม่เจอโดยตั้งใจ** · วิชาเป็นข้อมูลที่เพิ่งเพิ่มเข้ามาใน schema v3
        deployment ที่ตั้ง competition ไว้ก่อนหน้านั้นจึงอาจยังไม่มีแถวของตัวเอง
        การปฏิเสธการเข้าทีมเพราะเหตุนี้คือการทำให้นิสิตรับผลของ migration ที่ค้าง
        """
        course = self.courses.get(course_id)
        if course is None:
            course = self.save_course(Course(id=course_id, name=course_id))
        return course

    def competition_by_slug(self, slug: str) -> Competition | None:
        return next((c for c in self.competitions.values() if c.slug == slug), None)

    def user_by_token(self, token: str) -> User | None:
        """โทเคนบอกว่าเป็น**ใคร** ไม่ใช่ว่าอยู่ทีมไหน

        ทีมที่จะใช้ทำงานหาจากคนคนนี้ + วิชาของ competition ที่กำลังยุ่งด้วย
        (`team_of`) · ผลคือนิสิตที่เรียนหลายวิชาใช้โทเคนอันเดียวได้ทุกวิชา
        """
        if not token:
            return None
        return next((u for u in self.users.values() if u.token == token), None)

    def course_by_join_code(self, code: str) -> Course | None:
        want = (code or "").strip().upper()
        return next(
            (c for c in self.courses.values() if c.is_open and c.join_code.upper() == want), None
        ) if want else None

    def courses_of(self, user_id: str) -> list[str]:
        """วิชาที่คนนี้อยู่ — เรียงตามชื่อวิชาให้หน้าเว็บแสดงได้คงที่"""
        return sorted(
            {t.course_id for t in self.teams.values() if t.is_active and user_id in t.member_ids},
            key=lambda cid: self.courses[cid].name if cid in self.courses else cid,
        )

    def user_by_google_sub(self, sub: str) -> User | None:
        return next((u for u in self.users.values() if u.google_sub == sub), None)

    def team_by_invite_code(self, code: str) -> Team | None:
        """หาทีมจากรหัสเชิญ — ไม่สนตัวพิมพ์เล็กใหญ่เพราะนิสิตพิมพ์เอง"""
        code = (code or "").strip().upper()
        if not code:
            return None
        return next(
            (t for t in self.teams.values() if t.invite_code == code and t.is_active), None
        )

    def team_of(self, user_id: str, course_id: str) -> Team | None:
        """ทีมที่นิสิตคนนี้อยู่ในวิชานี้ — ทีมที่ยุบแล้วไม่นับ"""
        return next(
            (
                t
                for t in self.teams.values()
                if t.course_id == course_id and user_id in t.member_ids and t.is_active
            ),
            None,
        )

    def submissions_of(self, team_id: str, competition_id: str) -> list[Submission]:
        return sorted(
            (
                s
                for s in self.submissions.values()
                if s.team_id == team_id and s.competition_id == competition_id
            ),
            key=lambda s: s.created_at,
        )

    # ── audit ───────────────────────────────────────────────────────

    def record(
        self,
        action: str,
        target_type: str,
        target_id: str,
        *,
        actor_id: str | None = None,
        **payload,
    ) -> AuditEvent:
        """append-only — README §7 ต้องย้อนดูได้เสมอว่าใครทำอะไรเมื่อไร"""
        event = AuditEvent(
            id=new_id(),
            actor_id=actor_id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            payload=payload,
            created_at=utcnow(),
        )
        self.audit.append(event)
        if self.db:
            self.db.save_audit(event)
        return event

    def events_for(self, target_id: str) -> list[AuditEvent]:
        return [e for e in self.audit if e.target_id == target_id]


def runs_of(queue_runs: dict[str, Run], team_id: str, competition_id: str) -> list[Run]:
    return sorted(
        (
            r
            for r in queue_runs.values()
            if r.team_id == team_id and r.competition_id == competition_id
        ),
        key=lambda r: r.created_at,
    )
