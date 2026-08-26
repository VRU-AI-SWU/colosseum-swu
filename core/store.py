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
from core.domain import AuditEvent, Competition, Run, Submission, Team, User, new_id, utcnow


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

    def extract(self, url: str, into: Path) -> Path:
        """แตก zip ไปยังโฟลเดอร์ที่จะ mount เข้า sandbox

        กัน path traversal ที่นี่อีกชั้น ถึงแม้ `validate.inspect_archive` จะตรวจไปแล้ว —
        ชั้นนี้เป็นตัวที่เขียนไฟล์ลงดิสก์จริง จึงเป็นด่านสุดท้ายที่ต้องปลอดภัยด้วยตัวเอง
        """
        into.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(url) as zf:
            for info in zf.infolist():
                target = (into / info.filename).resolve()
                if not str(target).startswith(str(into.resolve())):
                    raise ValueError(f"path ใน zip ออกนอกโฟลเดอร์: {info.filename}")
            zf.extractall(into)
        # รองรับ zip ที่ห่อด้วยโฟลเดอร์ชั้นเดียว (นิสิต zip ทั้งโฟลเดอร์มา)
        if not (into / "agent.py").exists():
            nested = [p for p in into.iterdir() if p.is_dir() and (p / "agent.py").exists()]
            if len(nested) == 1:
                return nested[0]
        return into

    def replay_path(self, run_id: str) -> Path:
        path = self.root / "replays" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def clear_workdir(self, path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)


@dataclass
class Store:
    teams: dict[str, Team] = field(default_factory=dict)
    competitions: dict[str, Competition] = field(default_factory=dict)
    submissions: dict[str, Submission] = field(default_factory=dict)
    users: dict[str, User] = field(default_factory=dict)
    audit: list[AuditEvent] = field(default_factory=list)
    db: Database | None = None

    # ── เขียน (ต้องเรียกทุกครั้งที่แก้ ไม่งั้นของหายตอนรีสตาร์ท) ──────

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

    def competition_by_slug(self, slug: str) -> Competition | None:
        return next((c for c in self.competitions.values() if c.slug == slug), None)

    def team_by_token(self, token: str) -> Team | None:
        """หาทีมจาก `Authorization: Bearer <token>`

        เดิมเป็น `self.teams.get(token)` เพราะ id กับ token เป็นตัวเดียวกัน — id ที่
        เดาได้จึงกลายเป็นรหัสผ่านที่เดาได้ · ตอนนี้แยกกันแล้ว

        **ทีมที่ยุบแล้วใช้โทเคนไม่ได้** — ไม่งั้นการยุบทีมจะซ่อนมันจากกระดานเฉยๆ
        แต่ยังส่งงานในนามทีมนั้นได้อยู่ ซึ่งทำให้การยุบไม่ได้ปิดอะไรเลย
        """
        if not token:
            return None
        return next(
            (t for t in self.teams.values() if t.token == token and t.is_active), None
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
