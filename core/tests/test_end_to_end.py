"""ทดสอบครบวงจร: อัพโหลด zip จริง → worker รันใน sandbox → คะแนนขึ้น leaderboard

นี่คือเกณฑ์ที่ [README §14 M1](../../README.md#14-ขอบเขต-mvp-และ-roadmap) ใช้วัดว่าพร้อมหรือยัง

> "ทีมทดสอบส่ง agent แล้วเห็นคะแนนขึ้น leaderboard ได้ครบวงจร โดยไม่มีใครต้องเข้า SSH"
"""

from __future__ import annotations

import io
import textwrap
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.api import create_app
from core.wiring import CP463_VACUUM_LADDER, demo_arena
from runners.worker import Worker

SLUG = "cp463-vacuum-1-2026"

STRATEGY_AGENT = """
from vacuum.baselines import BASELINES

class Agent:
    def __init__(self, config):
        self._inner = BASELINES["silver"](config)
    def reset(self, episode_info):
        self._inner.reset(episode_info)
    def act(self, observation):
        return self._inner.act(observation)
"""

BRONZE_AGENT = STRATEGY_AGENT.replace('"silver"', '"bronze"')


def zip_bytes(agent_source: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("agent.py", textwrap.dedent(agent_source))
    return buf.getvalue()


@pytest.fixture
def system(tmp_path, monkeypatch):
    # ตัด ARENA_SECRETS ทิ้งเสมอ ไม่ว่าเครื่องที่รันจะตั้งไว้หรือไม่ — สองเหตุผล
    #   1. จำนวน episode ที่เทสต์ยืนยันจะขึ้นกับว่าเครื่องนั้นมีของลับหรือเปล่า → flaky
    #   2. ถ้าอ่าน seed จริง ค่ามันจะโผล่ใน assertion message ตอนเทสต์ล้ม แล้วไหลไป CI log
    monkeypatch.delenv("ARENA_SECRETS", raising=False)
    arena, teams = demo_arena(tmp_path / "artifacts", teams=3)
    client = TestClient(create_app(arena, baselines={SLUG: CP463_VACUUM_LADDER}))
    worker = Worker(
        runner_id="runner-test",
        store=arena.store,
        queue=arena.queue,
        artifacts=arena.artifacts,
        workdir=tmp_path / "work",
        allow_seed_fallback=True,  # ⚠️ dev เท่านั้น — ของจริงต้องมี ARENA_SECRETS
    )
    return arena, teams, client, worker


def auth(team) -> dict:
    return {"Authorization": f"Bearer {team.id}"}


def submit(client, team, source, **form):
    return client.post(
        f"/api/competitions/{SLUG}/submissions",
        headers=auth(team),
        files={"file": ("sub.zip", zip_bytes(source), "application/zip")},
        data=form,
    )


# ── วงจรหลัก ────────────────────────────────────────────────────────


def test_submit_run_and_see_score_on_leaderboard(system):
    arena, teams, client, worker = system
    team = teams[0]

    created = submit(client, team, STRATEGY_AGENT, note="ลองครั้งแรก")
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["quota_left"] == 4
    assert body["queue_position"] == 0

    # ก่อน worker ทำงาน — ยังไม่มีคะแนน
    pending = client.get(f"/api/submissions/{body['submission_id']}", headers=auth(team)).json()
    assert pending["runs"][0]["status"] == "queued"

    assert worker.run_once() is not None

    done = client.get(f"/api/submissions/{body['submission_id']}", headers=auth(team)).json()
    run = done["runs"][0]
    assert run["status"] == "done", run["error"]
    assert run["score"] > 0.4, "agent ที่มีกลยุทธ์ควรได้คะแนนชัดเจนกว่าเดินสุ่ม"
    assert run["env_version"] == "1.0.0"
    assert run["config_hash"].startswith("sha256:")

    board = client.get(f"/api/competitions/{SLUG}/leaderboard", headers=auth(team)).json()
    mine = [r for r in board["rows"] if r["type"] == "team" and r["is_you"]]
    assert len(mine) == 1
    assert mine[0]["score"] == pytest.approx(run["score"])


def test_episodes_and_replays_are_available(system):
    arena, teams, client, worker = system
    body = submit(client, teams[0], STRATEGY_AGENT).json()
    worker.run_once()

    run_id = client.get(
        f"/api/submissions/{body['submission_id']}", headers=auth(teams[0])
    ).json()["runs"][0]["id"]

    episodes = client.get(f"/api/runs/{run_id}/episodes", headers=auth(teams[0])).json()
    assert len(episodes["episodes"]) == 10
    assert all(e["status"] == "ok" for e in episodes["episodes"])
    assert all(e["replay_bytes"] > 0 for e in episodes["episodes"])
    assert [e["episode"] for e in episodes["episodes"]] == list(range(1, 11))

    replays = list((arena.artifacts.replay_path(run_id)).glob("*.vrp"))
    assert len(replays) == 10


def test_api_never_reveals_seed_values(system):
    """ค่า seed ต้องไม่เดินทางข้ามเส้น worker → API → นิสิต

    README §10.4 จัดค่า public seed เป็นความลับรองจาก private เพราะรู้แล้ว overfit ได้
    เดิม worker ใส่ `"seed": e.seed` ลง metrics ทำให้ค่าจริงโผล่ทั้งใน leaderboard
    และ endpoint รายตอน — พบตอน dry-run เต็มระบบ ไม่ใช่ตอนรีวิวโค้ด
    """
    from runners.seeds import FALLBACK_SEEDS

    arena, teams, client, worker = system
    body = submit(client, teams[0], STRATEGY_AGENT).json()
    worker.run_once()
    run_id = client.get(
        f"/api/submissions/{body['submission_id']}", headers=auth(teams[0])
    ).json()["runs"][0]["id"]

    surfaces = [
        client.get(f"/api/runs/{run_id}/episodes", headers=auth(teams[0])).text,
        client.get(f"/api/submissions/{body['submission_id']}", headers=auth(teams[0])).text,
        client.get(
            f"/api/competitions/{SLUG}/leaderboard", headers=auth(teams[0])
        ).text,
    ]
    for payload in surfaces:
        assert '"seed"' not in payload
        leaked = [s for s in FALLBACK_SEEDS if str(s) in payload]
        assert not leaked, f"ค่า seed หลุดออกทาง API: {leaked[:3]}"


def test_cors_allows_only_the_configured_web_origin(system, tmp_path):
    """หน้าเว็บอยู่คนละ origin กับ API จึงต้องมี CORS — แต่ต้องเปิดให้โดเมนเดียว

    ทุก endpoint ยืนยันตัวตนด้วย Bearer token การเปิด `*` แปลว่าหน้าเว็บใดก็ตามที่
    นิสิตเปิดอยู่ ยิง request ในนามของทีมได้ถ้าดักโทเคนไปได้
    """
    arena, _teams, _client, _worker = system
    web = "https://colosseum.vru-ai.com"
    guarded = TestClient(create_app(arena, allow_origins=[web]))

    assert guarded.get("/api/health", headers={"Origin": web}).headers.get(
        "access-control-allow-origin"
    ) == web
    assert (
        guarded.get("/api/health", headers={"Origin": "https://evil.example"}).headers.get(
            "access-control-allow-origin"
        )
        is None
    )

    # ไม่ตั้งค่า = ไม่เปิดให้ใครเลย ซึ่งถูกต้องสำหรับ dev ที่เรียกผ่าน localhost หรือ CLI
    assert (
        TestClient(create_app(arena))
        .get("/api/health", headers={"Origin": web})
        .headers.get("access-control-allow-origin")
        is None
    )


def test_leaderboard_ranks_teams_and_shows_next_target(system):
    arena, teams, client, worker = system
    submit(client, teams[0], STRATEGY_AGENT)
    submit(client, teams[1], BRONZE_AGENT)
    worker.drain()

    board = client.get(f"/api/competitions/{SLUG}/leaderboard", headers=auth(teams[1])).json()
    team_rows = [r for r in board["rows"] if r["type"] == "team"]
    assert [r["rank"] for r in team_rows] == [1, 2]
    assert team_rows[0]["score"] > team_rows[1]["score"]

    # ทีมที่ใช้ Bronze ต้องเห็นเป้าหมายถัดไปที่ทำได้ ไม่ใช่แค่ไล่ที่ 1 (README §6.2)
    assert board["next_target"]["level"] in ("bronze", "silver")

    # หมุด baseline ต้องปรากฏบนตารางด้วย
    assert {r["level"] for r in board["rows"] if r["type"] == "baseline"} == {
        "bronze", "silver", "gold", "diamond"
    }


# ── กติกาที่ต้องบังคับได้ ────────────────────────────────────────────


def test_broken_submission_is_rejected_with_a_fix(system):
    """§13 — error ต้องบอกวิธีแก้ ไม่ใช่แค่บอกว่าผิด"""
    _arena, teams, client, _worker = system
    response = submit(client, teams[0], "class NotAnAgent:\n    pass\n")
    assert response.status_code == 422
    problem = response.json()["detail"][0]
    assert problem["code"] == "missing_agent_class"
    assert "Agent" in problem["fix"]


def test_quota_is_enforced_and_dry_run_is_free(system):
    arena, teams, client, worker = system
    team = teams[0]

    for _ in range(5):
        assert submit(client, team, STRATEGY_AGENT).status_code == 201
        worker.run_once()  # ต้องรันให้จบก่อน ไม่งั้นติดกติกา 1 งาน/ทีม

    over = submit(client, team, STRATEGY_AGENT)
    assert over.status_code == 429
    assert "dry run ไม่กินโควตา" in over.json()["detail"]

    assert submit(client, team, STRATEGY_AGENT, dry_run="true").status_code == 201


def test_one_running_job_per_team(system):
    _arena, teams, client, _worker = system
    assert submit(client, teams[0], STRATEGY_AGENT).status_code == 201
    second = submit(client, teams[0], STRATEGY_AGENT)
    assert second.status_code == 409
    assert "1 งานพร้อมกันต่อทีม" in second.json()["detail"]


def test_cannot_read_another_teams_submission(system):
    _arena, teams, client, _worker = system
    body = submit(client, teams[0], STRATEGY_AGENT).json()
    assert (
        client.get(f"/api/submissions/{body['submission_id']}", headers=auth(teams[1])).status_code
        == 403
    )


def test_requires_a_token(system):
    _arena, _teams, client, _worker = system
    assert client.get(f"/api/competitions/{SLUG}/leaderboard").status_code == 401


# ── final pick + private ────────────────────────────────────────────


def test_final_pick_is_capped_at_two(system):
    arena, teams, client, worker = system
    team = teams[0]
    ids = []
    for _ in range(3):
        ids.append(submit(client, team, STRATEGY_AGENT).json()["submission_id"])
        worker.run_once()

    for sid in ids[:2]:
        assert client.post(f"/api/submissions/{sid}/final-pick", headers=auth(team)).status_code == 200
    third = client.post(f"/api/submissions/{ids[2]}/final-pick", headers=auth(team))
    assert third.status_code == 409
    assert "สูงสุด 2 ชุด" in third.json()["detail"]


def test_private_run_uses_only_final_picks(system):
    """**ห้ามรันทุก submission แล้วเลือกอันที่ดีที่สุด** — เท่ากับให้ทีมที่ส่งเยอะจับฉลากเยอะกว่า
    ซึ่งทำให้ private พังด้วยเหตุผลเดียวกับ public เป๊ะ"""
    arena, teams, client, worker = system
    for _ in range(2):
        submit(client, teams[0], STRATEGY_AGENT)
        worker.run_once()
    submit(client, teams[1], BRONZE_AGENT)
    worker.run_once()

    private_runs = arena.close_and_enqueue_private(slug=SLUG, actor_id="instructor")
    assert len(private_runs) == 2, "หนึ่งชุดต่อทีม (ทีมที่ไม่เลือกเอง → ใช้ตัวที่ public สูงสุด)"

    worker.drain()
    board = client.get(
        f"/api/competitions/{SLUG}/leaderboard?kind=private", headers=auth(teams[0])
    ).json()
    assert len([r for r in board["rows"] if r["type"] == "team"]) == 2


def test_public_and_private_boards_are_separate(system):
    _arena, teams, client, worker = system
    submit(client, teams[0], STRATEGY_AGENT)
    worker.run_once()

    public = client.get(f"/api/competitions/{SLUG}/leaderboard", headers=auth(teams[0])).json()
    private = client.get(
        f"/api/competitions/{SLUG}/leaderboard?kind=private", headers=auth(teams[0])
    ).json()
    assert len([r for r in public["rows"] if r["type"] == "team"]) == 1
    assert len([r for r in private["rows"] if r["type"] == "team"]) == 0


# ── audit ───────────────────────────────────────────────────────────


def test_everything_is_audited(system):
    arena, teams, client, worker = system
    body = submit(client, teams[0], STRATEGY_AGENT).json()
    worker.run_once()

    actions = {e.action for e in arena.store.audit}
    assert {"submission.created", "run.completed"} <= actions
    assert any(e.payload.get("sha256") for e in arena.store.events_for(body["submission_id"]))
