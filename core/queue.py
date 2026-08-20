"""คิวงานแบบ fair-share พร้อม lease + heartbeat — README §10.2 และ §7

    submit → enqueue → runner claim (lease) → heartbeat → report → leaderboard

**สามข้อที่ไฟล์นี้มีไว้แก้โดยเฉพาะ**

1. **Fair-share ไม่ใช่ FIFO** — ถ้าใช้ FIFO ล้วน ทีมที่ส่ง 5 งานรวดจะบล็อกทุกทีมที่ส่งทีหลัง
   คืนก่อน deadline เลยกลายเป็นการแข่งว่าใครกดส่งเร็วกว่า ไม่ใช่ใครเก่งกว่า
2. **runner หายไปแล้วงานต้องกลับเข้าคิวเอง** — ไฟดับ เน็ตหลุด process ถูก kill
   ถ้างานค้างอยู่ที่ `running` ตลอดกาล ทีมนั้นจะส่งอะไรไม่ได้อีกเลยเพราะติดกติกา 1 งาน/ทีม
3. **รายงานผลซ้ำต้องไม่นับซ้ำ** — runner อาจส่งผลแล้วเน็ตหลุดก่อนได้ ack แล้วส่งใหม่

working set อยู่ในหน่วยความจำ และเขียนทะลุลง SQLite ทุกครั้งที่สถานะเปลี่ยน
([`core/db.py`](db.py)) — ตรรกะการเลือกงานในไฟล์นี้จึงไม่ต้องเปลี่ยนเลย
โครงสร้างนี้ย้ายไป Postgres ได้ตรงๆ (`SELECT ... FOR UPDATE SKIP LOCKED`) เมื่อถึงวันนั้น

⚠️ **ทุก transition ของ run ต้องจบด้วย `self._persist(run)`** — ถ้าลืม งานที่กำลัง
รันอยู่จะกลับมาเป็น `queued` หลังรีสตาร์ท หรือแย่กว่านั้นคือคะแนนที่รายงานแล้วหายไป
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

from core.db import Database
from core.domain import (
    ACTIVE_RUN_STATUSES,
    Run,
    RunKind,
    RunStatus,
    utcnow,
)

DEFAULT_LEASE = timedelta(minutes=5)
MAX_ATTEMPTS = 3


class LeaseExpired(Exception):
    """runner รายงานผลของงานที่ lease หมดอายุไปแล้ว — งานอาจถูกแจกให้คนอื่นไปแล้ว"""


@dataclass
class JobQueue:
    """คิวเดียวรองรับหลายเลน (cpu / gpu) — README §10.2 บอกว่างาน RL ไม่ควรไปแย่ 3090"""

    lease_duration: timedelta = DEFAULT_LEASE
    max_attempts: int = MAX_ATTEMPTS
    runs: dict[str, Run] = field(default_factory=dict)
    #: ทีมไหนถูกเสิร์ฟไปแล้วกี่งาน — ใช้จัดลำดับแบบ round-robin
    _served: dict[str, int] = field(default_factory=dict)
    db: Database | None = None

    def _persist(self, run: Run) -> Run:
        if self.db:
            self.db.save_run(run)
        return run

    def _bump_served(self, team_id: str, delta: int = 1) -> None:
        self._served[team_id] = self._served.get(team_id, 0) + delta
        if self.db:
            self.db.save_served(team_id, self._served[team_id])

    # ── ฝั่งผู้ส่งงาน ────────────────────────────────────────────────

    def enqueue(self, run: Run) -> Run:
        if run.id in self.runs:
            return self.runs[run.id]  # idempotent: enqueue ซ้ำด้วย id เดิมไม่สร้างงานใหม่
        if self.active_run_for(run.team_id, run.competition_id) is not None:
            raise RuntimeError(
                f"ทีม {run.team_id} มีงานที่ยังไม่จบอยู่แล้ว — 1 งานพร้อมกันต่อทีม (README §7)"
            )
        run.status = RunStatus.QUEUED
        self.runs[run.id] = run
        if run.team_id not in self._served:
            self._bump_served(run.team_id, 0)
        return self._persist(run)

    def adopt(self, run: Run) -> Run:
        """ใส่งานเข้าคิว**โดยข้ามกติกา 1 งานพร้อมกันต่อทีม**

        มีทางเข้านี้ทางเดียวและตั้งชื่อให้อ่านออก เพราะเดิม `service.start_private_run`
        เขียนลง `queue.runs` กับ `queue._served` ตรงๆ ซึ่งเลี่ยงทั้งการตรวจและการบันทึก
        ลงฐานข้อมูล — งาน private ทั้งชุดจะหายไปถ้ารีสตาร์ทระหว่างรอบตัดเกรด
        """
        self.runs[run.id] = run
        if run.team_id not in self._served:
            self._bump_served(run.team_id, 0)
        return self._persist(run)

    def active_run_for(self, team_id: str, competition_id: str) -> Run | None:
        return next(
            (
                r
                for r in self.runs.values()
                if r.team_id == team_id
                and r.competition_id == competition_id
                and r.status in ACTIVE_RUN_STATUSES
            ),
            None,
        )

    # ── ฝั่ง runner ─────────────────────────────────────────────────

    def claim(self, runner_id: str, *, lanes: Iterable[str] = ("cpu",), now: datetime | None = None) -> Run | None:
        """หยิบงานถัดไปแบบ fair-share แล้วจอง lease ไว้

        ลำดับการเลือก: **ทีมที่ถูกเสิร์ฟน้อยที่สุดก่อน** แล้วค่อยเก่าสุดก่อนภายในทีมเดียวกัน
        ทีมที่ส่งรัวจึงไม่ได้เปรียบ — งานที่ 2 ของทีม A จะรอจนทุกทีมได้คิวแรกก่อนเสมอ
        """
        now = now or utcnow()
        self.requeue_expired(now)

        lanes = set(lanes)
        waiting = [r for r in self.runs.values() if r.status is RunStatus.QUEUED and r.lane in lanes]
        if not waiting:
            return None

        run = min(waiting, key=lambda r: (self._served.get(r.team_id, 0), r.created_at, r.id))

        run.status = RunStatus.RUNNING
        run.runner_id = runner_id
        run.attempts += 1
        run.started_at = run.started_at or now
        run.lease_expires_at = now + self.lease_duration
        self._bump_served(run.team_id)
        return self._persist(run)

    def heartbeat(self, run_id: str, runner_id: str, *, now: datetime | None = None) -> None:
        """ต่ออายุ lease — runner ต้องเรียกถี่กว่า `lease_duration` เสมอ"""
        now = now or utcnow()
        run = self.runs[run_id]
        if run.status is not RunStatus.RUNNING or run.runner_id != runner_id:
            raise LeaseExpired(f"run {run_id} ไม่ได้อยู่ในมือของ runner {runner_id} แล้ว")
        run.lease_expires_at = now + self.lease_duration
        self._persist(run)

    def requeue_expired(self, now: datetime | None = None) -> list[Run]:
        """งานที่ runner หายไปกลางคัน → กลับเข้าคิว

        งานที่ล้มซ้ำเกิน `max_attempts` ถูกทำเครื่องหมายว่าล้มเหลวแทนที่จะวนไม่รู้จบ —
        ไม่งั้น submission ที่ทำ runner ตายจะฆ่า runner ทุกตัวที่หยิบมันขึ้นมาไปเรื่อยๆ
        """
        now = now or utcnow()
        requeued = []
        for run in self.runs.values():
            if run.status is not RunStatus.RUNNING or run.lease_expires_at is None:
                continue
            if run.lease_expires_at > now:
                continue
            if run.attempts >= self.max_attempts:
                run.status = RunStatus.FAILED
                run.error_message = (
                    f"runner หยุดตอบ {run.attempts} ครั้งติด — งานนี้อาจทำให้ runner ล่ม"
                )
                run.finished_at = now
            else:
                run.status = RunStatus.QUEUED
                run.runner_id = None
                run.lease_expires_at = None
                requeued.append(run)
            self._persist(run)
        return requeued

    def report(
        self,
        run_id: str,
        runner_id: str,
        *,
        status: RunStatus,
        score: float | None = None,
        tiebreak: tuple = (),
        metrics: dict | None = None,
        episodes: list | None = None,
        config_hash: str | None = None,
        env_version: str | None = None,
        error_message: str | None = None,
        now: datetime | None = None,
    ) -> Run:
        """รับผลจาก runner — **idempotent**: เรียกซ้ำด้วย run_id เดิมไม่ทำให้คะแนนซ้ำซ้อน"""
        now = now or utcnow()
        run = self.runs[run_id]

        if run.status in (RunStatus.DONE, RunStatus.FAILED):
            return run  # รายงานมาแล้ว — เงียบๆ แล้วคืนของเดิม (runner อาจ retry หลังเน็ตหลุด)
        if run.runner_id != runner_id:
            raise LeaseExpired(
                f"run {run_id} ถูกแจกให้ runner {run.runner_id} ไปแล้ว — ผลจาก {runner_id} ถูกทิ้ง"
            )

        run.status = status
        run.score = score
        run.tiebreak = tiebreak
        run.metrics = metrics or {}
        run.episodes = episodes or []
        run.config_hash = config_hash
        run.env_version = env_version
        run.error_message = error_message
        run.finished_at = now
        run.lease_expires_at = None
        return self._persist(run)

    # ── สถานะสำหรับหน้าเว็บ ─────────────────────────────────────────

    def position_of(self, run_id: str) -> int | None:
        """ลำดับในคิวตามที่จะถูกหยิบจริง — ใช้แสดง "รออีกกี่งาน" ให้นิสิต"""
        run = self.runs[run_id]
        if run.status is not RunStatus.QUEUED:
            return None
        waiting = sorted(
            (r for r in self.runs.values() if r.status is RunStatus.QUEUED and r.lane == run.lane),
            key=lambda r: (self._served.get(r.team_id, 0), r.created_at, r.id),
        )
        return next(i for i, r in enumerate(waiting) if r.id == run.id)

    def depth(self, lane: str = "cpu") -> int:
        return sum(1 for r in self.runs.values() if r.status is RunStatus.QUEUED and r.lane == lane)


def check_quota(
    runs: Iterable[Run], team_id: str, quota_per_day: int, *, now: datetime | None = None
) -> int:
    """คืนจำนวนที่เหลือของวันนี้ — **dry run ไม่กินโควตา** (README §9)"""
    now = now or utcnow()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    used = sum(
        1
        for r in runs
        if r.team_id == team_id
        and r.kind is RunKind.PUBLIC
        and r.created_at >= start_of_day
    )
    return max(0, quota_per_day - used)
