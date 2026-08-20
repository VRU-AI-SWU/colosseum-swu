"""รีสตาร์ทแล้วต้องไม่มีอะไรหาย — README §11

เทสต์ทุกข้อในไฟล์นี้มีรูปแบบเดียวกัน: ทำอะไรบางอย่าง → **ทิ้ง Arena ทั้งก้อน** →
ประกอบใหม่จากไฟล์เดียวกัน → ยืนยันว่าสิ่งที่ควรอยู่ยังอยู่

การ "ทิ้งทั้งก้อน" สำคัญกว่าที่คิด — ถ้าใช้ object เดิมต่อ เทสต์จะผ่านแม้ว่าจะไม่มี
การเขียนลงดิสก์เลยสักครั้ง เพราะทุกอย่างยังอยู่ในหน่วยความจำ
"""

from __future__ import annotations

import io
import textwrap
import zipfile

import pytest
from fastapi.testclient import TestClient

from core.api import create_app
from core.db import Database, SchemaMismatch
from core.domain import RunKind, RunStatus
from core.wiring import CP463_VACUUM_LADDER, demo_arena
from runners.worker import Worker

SLUG = "cp463-vacuum-1-2026"

AGENT = """
from vacuum.baselines import BASELINES

class Agent:
    def __init__(self, config):
        self._inner = BASELINES["silver"](config)
    def reset(self, episode_info):
        self._inner.reset(episode_info)
    def act(self, observation):
        return self._inner.act(observation)
"""


def zip_bytes(source: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("agent.py", textwrap.dedent(source))
    return buf.getvalue()


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """คืนฟังก์ชันที่ "บูตระบบขึ้นมาใหม่" จากดิสก์ชุดเดิมได้เรื่อยๆ"""
    monkeypatch.delenv("ARENA_SECRETS", raising=False)
    root = tmp_path / "data"

    def boot():
        arena, teams = demo_arena(root / "artifacts", teams=3, db_path=root / "arena.db")
        client = TestClient(create_app(arena, baselines={SLUG: CP463_VACUUM_LADDER}))
        worker = Worker(
            runner_id="runner-test",
            store=arena.store,
            queue=arena.queue,
            artifacts=arena.artifacts,
            workdir=root / "work",
            allow_seed_fallback=True,
        )
        return arena, teams, client, worker

    return boot


def auth(team_id: str) -> dict:
    return {"Authorization": f"Bearer {team_id}"}


def submit(client, team_id: str, source: str = AGENT, **form):
    return client.post(
        f"/api/competitions/{SLUG}/submissions",
        headers=auth(team_id),
        files={"file": ("sub.zip", zip_bytes(source), "application/zip")},
        data=form,
    )


# ── วงจรหลัก ────────────────────────────────────────────────────────


def test_score_survives_restart(lab):
    """เหตุผลทั้งหมดที่ทำ persistence — "รีสตาร์ททีคะแนนหายหมด" ต้องไม่จริงอีกต่อไป"""
    arena, teams, client, worker = lab()
    body = submit(client, "team-1").json()
    worker.run_once()
    before = client.get(f"/api/competitions/{SLUG}/leaderboard", headers=auth("team-1")).json()
    score_before = [r for r in before["rows"] if r["type"] == "team"][0]["score"]
    arena.store.db.close()

    _arena2, _teams2, client2, _worker2 = lab()
    after = client2.get(f"/api/competitions/{SLUG}/leaderboard", headers=auth("team-1")).json()
    rows = [r for r in after["rows"] if r["type"] == "team"]
    assert len(rows) == 1
    assert rows[0]["score"] == score_before

    status = client2.get(
        f"/api/submissions/{body['submission_id']}", headers=auth("team-1")
    ).json()
    assert status["runs"][0]["status"] == "done"
    assert status["runs"][0]["score"] == score_before
    assert status["runs"][0]["env_version"] == "1.0.0"


def test_episode_detail_survives_restart(lab):
    arena, _teams, client, worker = lab()
    body = submit(client, "team-1").json()
    worker.run_once()
    run_id = client.get(
        f"/api/submissions/{body['submission_id']}", headers=auth("team-1")
    ).json()["runs"][0]["id"]
    arena.store.db.close()

    _a2, _t2, client2, _w2 = lab()
    episodes = client2.get(f"/api/runs/{run_id}/episodes", headers=auth("team-1")).json()
    assert len(episodes["episodes"]) == 10
    assert [e["episode"] for e in episodes["episodes"]] == list(range(1, 11))
    assert '"seed"' not in str(episodes)


def test_queued_run_is_picked_up_after_restart(lab):
    """งานที่ยังไม่ได้รันต้องยังอยู่ในคิว — ไม่งั้นนิสิตส่งแล้วเงียบหายตลอดกาล"""
    arena, _teams, client, _worker = lab()
    body = submit(client, "team-1").json()
    assert (
        client.get(f"/api/submissions/{body['submission_id']}", headers=auth("team-1"))
        .json()["runs"][0]["status"] == "queued"
    )
    arena.store.db.close()

    arena2, _t2, client2, worker2 = lab()
    assert worker2.run_once() is not None, "งานที่ค้างคิวไว้ต้องถูกหยิบขึ้นมาทำต่อ"
    assert (
        client2.get(f"/api/submissions/{body['submission_id']}", headers=auth("team-1"))
        .json()["runs"][0]["status"] == "done"
    )


def test_in_flight_run_survives_and_can_be_requeued(lab):
    """งานที่ runner คว้าไปแล้วแต่ยังไม่จบ — ต้องกลับมาเป็น `running` พร้อม lease เดิม

    ถ้ามันกลับมาเป็น `queued` เฉยๆ งานจะถูกรันซ้ำสองรอบ และถ้ามันหายไปเลย
    ทีมนั้นจะติดกติกา 1 งาน/ทีม จนส่งอะไรไม่ได้อีก
    """
    arena, _teams, client, _worker = lab()
    submit(client, "team-1")
    claimed = arena.queue.claim("runner-a")
    assert claimed.status is RunStatus.RUNNING
    lease = claimed.lease_expires_at
    arena.store.db.close()

    arena2, *_ = lab()
    restored = arena2.queue.runs[claimed.id]
    assert restored.status is RunStatus.RUNNING
    assert restored.runner_id == "runner-a"
    assert restored.lease_expires_at == lease
    assert restored.attempts == 1

    # lease หมดอายุแล้วต้องกลับเข้าคิวได้ตามปกติ ไม่ใช่ค้างเป็น running ตลอดกาล
    requeued = arena2.queue.requeue_expired(now=lease.replace(year=lease.year + 1))
    assert [r.id for r in requeued] == [claimed.id]
    assert arena2.queue.runs[claimed.id].status is RunStatus.QUEUED


def test_fair_share_counters_survive_restart(lab):
    """ตัวนับ round-robin ต้องอยู่รอด ไม่งั้นรีสตาร์ท = ล้างสิทธิ์ของทีมที่เพิ่งได้คิวไป

    ทีมที่ส่งรัวแล้วโดนจัดไว้ท้ายแถว จะกลับมาเท่ากับทีมที่ยังไม่เคยได้คิวเลย
    ซึ่งทำให้ fair-share ที่ทั้งไฟล์ `queue.py` มีไว้แก้ พังเงียบๆ
    """
    arena, _teams, client, worker = lab()
    for _ in range(2):
        submit(client, "team-1")
        worker.run_once()
    submit(client, "team-2")
    worker.run_once()
    served_before = dict(arena.queue._served)
    assert served_before["team-1"] == 2 and served_before["team-2"] == 1
    arena.store.db.close()

    arena2, *_ = lab()
    assert arena2.queue._served == served_before


def test_quota_is_counted_across_restart(lab):
    """โควตาต่อวันต้องไม่รีเซ็ตเพราะรีสตาร์ท — ไม่งั้นมันเลี่ยงได้ด้วยการรอ deploy"""
    arena, _teams, client, worker = lab()
    for _ in range(5):
        assert submit(client, "team-1").status_code == 201
        worker.run_once()
    arena.store.db.close()

    _a2, _t2, client2, _w2 = lab()
    over = submit(client2, "team-1")
    assert over.status_code == 429


def test_final_pick_and_private_runs_survive_restart(lab):
    """รอบตัดเกรดเป็นจุดที่การสูญหายเจ็บที่สุด — ทั้ง final pick และงาน private"""
    arena, _teams, client, worker = lab()
    body = submit(client, "team-1").json()
    worker.run_once()
    picked = client.post(
        f"/api/submissions/{body['submission_id']}/final-pick",
        headers=auth("team-1"),
        data={"picked": "true"},
    )
    assert picked.status_code == 200
    created = arena.close_and_enqueue_private(slug=SLUG, actor_id="instructor")
    assert len(created) == 1
    arena.store.db.close()

    arena2, *_ = lab()
    assert arena2.store.submissions[body["submission_id"]].is_final_pick
    private = [r for r in arena2.queue.runs.values() if r.kind is RunKind.PRIVATE]
    assert len(private) == 1
    assert private[0].id == created[0].id


def test_audit_trail_survives_restart(lab):
    """README §7 บอกว่าต้องย้อนดูได้ *เสมอ* ว่าใครทำอะไรเมื่อไร — รวมถึงข้ามการรีสตาร์ท"""
    arena, _teams, client, worker = lab()
    body = submit(client, "team-1", note="ครั้งแรก").json()
    worker.run_once()
    before = [(e.action, e.target_id) for e in arena.store.audit]
    assert any(a == "run.completed" for a, _ in before)
    arena.store.db.close()

    arena2, *_ = lab()
    assert [(e.action, e.target_id) for e in arena2.store.audit] == before
    assert arena2.store.events_for(body["submission_id"])


# ── ความปลอดภัยของไฟล์ฐานข้อมูล ─────────────────────────────────────


def test_schema_version_mismatch_fails_loudly(tmp_path):
    """เปิดไฟล์เก่าด้วย schema ใหม่ต้องล้มทันที ไม่ใช่ทำงานต่อแล้วข้อมูลหายบางส่วน"""
    path = tmp_path / "arena.db"
    db = Database(path)
    db.close()

    import sqlite3

    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    conn.commit()
    conn.close()

    with pytest.raises(SchemaMismatch, match="migration"):
        Database(path)


def test_ephemeral_mode_writes_nothing(tmp_path, monkeypatch):
    """ไม่ส่ง db_path = ไม่มีไฟล์ไหนถูกสร้าง — เทสต์ส่วนใหญ่พึ่งพฤติกรรมนี้"""
    monkeypatch.delenv("ARENA_SECRETS", raising=False)
    root = tmp_path / "data"
    arena, _teams = demo_arena(root / "artifacts", teams=1)
    assert arena.store.db is None
    assert arena.queue.db is None
    assert not list(root.glob("*.db"))


def test_reboot_does_not_duplicate_competition_or_teams(lab):
    """demo_arena ต้อง idempotent — ไม่งั้น competition id ใหม่ทำให้ leaderboard ว่างเปล่า"""
    arena, teams, _client, _worker = lab()
    ids = (sorted(arena.store.competitions), sorted(arena.store.teams))
    assert len(teams) == 3
    arena.store.db.close()

    arena2, teams2, *_ = lab()
    assert (sorted(arena2.store.competitions), sorted(arena2.store.teams)) == ids
    assert [t.id for t in teams2] == [t.id for t in teams]
