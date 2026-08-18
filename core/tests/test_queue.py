"""คิวงาน — README §10.2 และ §7

เน้นสามสถานการณ์ที่เจ็บที่สุดถ้าพลาด: ทีมหนึ่งบล็อกทั้งคิว · งานค้างเพราะ runner หาย ·
คะแนนซ้ำเพราะ runner รายงานสองครั้ง
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.domain import Run, RunKind, RunStatus, new_id
from core.queue import JobQueue, LeaseExpired, check_quota

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def make_run(team: str, *, kind=RunKind.PUBLIC, created_at=T0, lane="cpu") -> Run:
    return Run(
        id=new_id(),
        submission_id=new_id(),
        competition_id="comp",
        team_id=team,
        kind=kind,
        lane=lane,
        created_at=created_at,
    )


# ── fair-share ──────────────────────────────────────────────────────


def test_one_team_cannot_block_the_queue():
    """ทีม A ส่งก่อน 3 งาน แต่ B กับ C ต้องได้คิวก่อนงานที่ 2 ของ A

    ถ้าเป็น FIFO ล้วน คืนก่อน deadline จะกลายเป็นการแข่งว่าใครกดส่งเร็วกว่า
    """
    q = JobQueue()
    a1 = q.enqueue(make_run("A", created_at=T0))
    # ทีมเดียวส่งได้ทีละงาน — งานที่สองของ A จึงต้องรอให้งานแรกจบก่อน
    with pytest.raises(RuntimeError, match="1 งานพร้อมกันต่อทีม"):
        q.enqueue(make_run("A", created_at=T0 + timedelta(seconds=1)))

    b1 = q.enqueue(make_run("B", created_at=T0 + timedelta(seconds=2)))
    c1 = q.enqueue(make_run("C", created_at=T0 + timedelta(seconds=3)))

    order = [q.claim("runner-1", now=T0 + timedelta(seconds=10)).id for _ in range(3)]
    assert order == [a1.id, b1.id, c1.id]


def test_round_robin_across_teams_after_first_round():
    """เมื่อ A จบงานแรกแล้วส่งใหม่ทันที มันต้องไปต่อท้ายทีมที่ยังไม่เคยถูกเสิร์ฟ"""
    q = JobQueue()
    a1 = q.enqueue(make_run("A", created_at=T0))
    b1 = q.enqueue(make_run("B", created_at=T0 + timedelta(seconds=5)))

    claimed_a = q.claim("r1", now=T0)
    q.report(claimed_a.id, "r1", status=RunStatus.DONE, score=1.0, now=T0)

    # A ส่งงานใหม่ทันที ก่อนที่ B จะถูกหยิบด้วยซ้ำ
    a2 = q.enqueue(make_run("A", created_at=T0 + timedelta(seconds=6)))

    nxt = q.claim("r1", now=T0 + timedelta(seconds=7))
    assert nxt.id == b1.id, "B ที่ยังไม่เคยถูกเสิร์ฟต้องมาก่อนงานที่สองของ A"
    assert a1.id != a2.id


def test_lanes_are_separate():
    q = JobQueue()
    q.enqueue(make_run("A", lane="gpu"))
    cpu_run = q.enqueue(make_run("B", lane="cpu"))
    assert q.claim("r1", lanes=["cpu"], now=T0).id == cpu_run.id
    assert q.claim("r1", lanes=["cpu"], now=T0) is None
    assert q.depth("gpu") == 1


# ── lease + heartbeat ───────────────────────────────────────────────


def test_dead_runner_returns_the_job_to_the_queue():
    """runner หายไป → งานกลับเข้าคิว ไม่ค้างจนทีมนั้นส่งอะไรไม่ได้อีกเลย"""
    q = JobQueue(lease_duration=timedelta(minutes=5))
    run = q.enqueue(make_run("A"))
    q.claim("runner-ตาย", now=T0)
    assert run.status is RunStatus.RUNNING

    requeued = q.requeue_expired(now=T0 + timedelta(minutes=6))
    assert [r.id for r in requeued] == [run.id]
    assert run.status is RunStatus.QUEUED and run.runner_id is None

    assert q.claim("runner-ใหม่", now=T0 + timedelta(minutes=6)).id == run.id


def test_heartbeat_keeps_the_lease_alive():
    q = JobQueue(lease_duration=timedelta(minutes=5))
    run = q.enqueue(make_run("A"))
    q.claim("r1", now=T0)

    for minute in (4, 8, 12):
        q.heartbeat(run.id, "r1", now=T0 + timedelta(minutes=minute))
        assert q.requeue_expired(now=T0 + timedelta(minutes=minute)) == []
    assert run.status is RunStatus.RUNNING


def test_heartbeat_from_a_stale_runner_is_rejected():
    q = JobQueue(lease_duration=timedelta(minutes=5))
    run = q.enqueue(make_run("A"))
    q.claim("r1", now=T0)
    q.requeue_expired(now=T0 + timedelta(minutes=6))
    q.claim("r2", now=T0 + timedelta(minutes=6))

    with pytest.raises(LeaseExpired):
        q.heartbeat(run.id, "r1", now=T0 + timedelta(minutes=7))


def test_poison_job_stops_after_max_attempts():
    """งานที่ทำ runner ตายซ้ำๆ ต้องหยุด ไม่ใช่ไล่ฆ่า runner ทุกตัวไปเรื่อยๆ"""
    q = JobQueue(lease_duration=timedelta(minutes=1), max_attempts=3)
    run = q.enqueue(make_run("A"))
    now = T0
    for _ in range(3):
        q.claim("r1", now=now)
        now += timedelta(minutes=2)
        q.requeue_expired(now=now)

    assert run.status is RunStatus.FAILED
    assert "runner หยุดตอบ" in run.error_message
    assert q.claim("r2", now=now) is None


# ── idempotency ─────────────────────────────────────────────────────


def test_reporting_twice_does_not_double_count():
    """runner ส่งผลแล้วเน็ตหลุดก่อนได้ ack แล้วส่งใหม่ — ต้องไม่กลายเป็นสองผล"""
    q = JobQueue()
    run = q.enqueue(make_run("A"))
    q.claim("r1", now=T0)

    first = q.report(run.id, "r1", status=RunStatus.DONE, score=1.5, now=T0)
    second = q.report(run.id, "r1", status=RunStatus.DONE, score=99.0, now=T0)

    assert first is second
    assert run.score == 1.5, "ผลครั้งแรกต้องชนะ ไม่ใช่ถูกทับด้วยค่าที่ส่งมาทีหลัง"


def test_enqueue_is_idempotent_by_run_id():
    q = JobQueue()
    run = make_run("A")
    assert q.enqueue(run) is q.enqueue(run)
    assert len(q.runs) == 1


def test_report_from_a_runner_that_lost_the_lease_is_rejected():
    q = JobQueue(lease_duration=timedelta(minutes=1))
    run = q.enqueue(make_run("A"))
    q.claim("r1", now=T0)
    q.requeue_expired(now=T0 + timedelta(minutes=2))
    q.claim("r2", now=T0 + timedelta(minutes=2))

    with pytest.raises(LeaseExpired):
        q.report(run.id, "r1", status=RunStatus.DONE, score=1.0, now=T0 + timedelta(minutes=3))


# ── โควตา ───────────────────────────────────────────────────────────


def test_dry_run_does_not_consume_quota():
    """README §9 — dry run มีไว้เช็คว่าแพ็กไฟล์ถูก ไม่ควรกินสิทธิ์ส่งจริง"""
    runs = [make_run("A", kind=RunKind.PUBLIC) for _ in range(2)]
    runs += [make_run("A", kind=RunKind.DRYRUN) for _ in range(5)]
    assert check_quota(runs, "A", quota_per_day=5, now=T0) == 3


def test_quota_resets_next_day():
    runs = [make_run("A", created_at=T0 - timedelta(days=1)) for _ in range(5)]
    assert check_quota(runs, "A", quota_per_day=5, now=T0) == 5


def test_quota_is_per_team():
    runs = [make_run("A") for _ in range(5)]
    assert check_quota(runs, "A", quota_per_day=5, now=T0) == 0
    assert check_quota(runs, "B", quota_per_day=5, now=T0) == 5


# ── สถานะที่แสดงให้นิสิต ────────────────────────────────────────────


def test_queue_position_follows_fair_share_not_arrival():
    q = JobQueue()
    q.enqueue(make_run("A", created_at=T0))
    q.claim("r1", now=T0)
    q.report(list(q.runs)[0], "r1", status=RunStatus.DONE, score=1.0, now=T0)

    a2 = q.enqueue(make_run("A", created_at=T0 + timedelta(seconds=1)))
    b1 = q.enqueue(make_run("B", created_at=T0 + timedelta(seconds=2)))

    assert q.position_of(b1.id) == 0, "B มาทีหลังแต่ยังไม่เคยถูกเสิร์ฟ จึงอยู่หน้า"
    assert q.position_of(a2.id) == 1
