"""ชื่อบนกระดาน — ทีมตั้งเอง ผู้สอนเห็นชื่อจริงเสมอ

README §6.1: *"โหมด alias นิรนาม (ทีมเลือกเองได้) — ลดแรงกดดันของทีมท้ายตาราง
ผู้สอนยังเห็นชื่อจริงเสมอ"* · ฟิลด์ `Team.alias` กับ `display_name(reveal=)` มีมา
ตั้งแต่ต้น แต่ไม่มีทางตั้งค่า และ **ไม่มีใครเรียกด้วย `reveal_names=True` เลย**
แปลว่าผู้สอนก็ไม่เห็นชื่อจริงเหมือนกัน

กระดานคือที่ที่คนใช้ตัดสินใจว่าตัวเองอยู่ตรงไหน ชื่อที่ซ้ำหรือชื่อที่ปลอมเป็นหมุด
baseline จึงไม่ใช่เรื่องมารยาท แต่ทำให้อ่านผิดได้จริง
"""

from __future__ import annotations

import pytest

from core.domain import (
    MAX_ALIAS_LENGTH,
    AliasInvalid,
    RunKind,
    clean_alias,
)
from core.leaderboard import build
from core.wiring import demo_arena

COURSE = "cp463-1-2026"
STAFF = "aj@g.swu.ac.th"


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.delenv("ARENA_SECRETS", raising=False)
    a, _teams = demo_arena(tmp_path / "artifacts", teams=0)
    return a


def _member(arena, email: str, name: str):
    return arena.sign_in(google_sub=f"sub-{email}", email=email, name=name, course_id=COURSE)


# ── การทำความสะอาดชื่อ ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,want",
    [
        ("  ทีม   ลุยเลย  ", "ทีม ลุยเลย"),   # ยุบช่องว่างซ้ำ ตัดหัวท้าย
        ("", None),                            # ว่าง = ขอกลับไปใช้ชื่อจริง
        ("   ", None),
        (None, None),
        ("ทีม​ปลอม", "ทีมปลอม"),          # zero-width space ถูกตัดทิ้ง
        ("a" * MAX_ALIAS_LENGTH, "a" * MAX_ALIAS_LENGTH),
    ],
)
def test_clean_alias(raw, want):
    assert clean_alias(raw) == want


def test_zero_width_characters_cannot_fake_a_duplicate_name(arena):
    """สองทีมที่ตาเห็นชื่อเหมือนกันคือปัญหาของกระดาน ไม่ใช่แค่ความน่ารำคาญ"""
    _u, first = _member(arena, "a@g.swu.ac.th", "ทีมหนึ่ง")
    arena.set_alias(team=first, raw="ทีมจริง", actor_id=None)

    _u2, second = _member(arena, "b@g.swu.ac.th", "ทีมสอง")
    with pytest.raises(AliasInvalid, match="มีทีมอื่นใช้ชื่อ"):
        arena.set_alias(team=second, raw="ทีม​จริง", actor_id=None)


@pytest.mark.parametrize("reserved", ["Gold", "gold", "💎 diamond".split()[1], "baseline"])
def test_baseline_names_are_reserved(reserved):
    """ทีมที่ชื่อ "Gold" จะอ่านเหมือนหมุดของผู้สอนบนกระดาน"""
    with pytest.raises(AliasInvalid, match="หมุด baseline"):
        clean_alias(reserved)


def test_too_long_says_how_long_it_is(arena):
    with pytest.raises(AliasInvalid) as exc:
        clean_alias("ก" * (MAX_ALIAS_LENGTH + 5))
    assert str(MAX_ALIAS_LENGTH) in str(exc.value)
    assert str(MAX_ALIAS_LENGTH + 5) in str(exc.value), "ต้องบอกว่าตอนนี้ยาวเท่าไร"


# ── กติกาของชื่อซ้ำ ────────────────────────────────────────────────


def test_alias_cannot_collide_with_another_teams_real_name(arena):
    """ชื่อจริงของทีมอื่นก็ขึ้นกระดานได้ (ถ้าเขาไม่ตั้ง alias) จึงต้องกันด้วย"""
    _u, _first = _member(arena, "a@g.swu.ac.th", "ทีมหนึ่ง")
    _u2, second = _member(arena, "b@g.swu.ac.th", "ทีมสอง")
    with pytest.raises(AliasInvalid):
        arena.set_alias(team=second, raw="ทีมหนึ่ง", actor_id=None)


def test_a_dissolved_team_does_not_reserve_its_name(arena):
    """ทีมที่ยุบแล้วไม่ขึ้นกระดาน จึงไม่มีเหตุให้จองชื่อไว้"""
    from core.domain import utcnow

    _u, first = _member(arena, "a@g.swu.ac.th", "ทีมหนึ่ง")
    first.dissolved_at = utcnow()
    arena.store.save_team(first)

    _u2, second = _member(arena, "b@g.swu.ac.th", "ทีมสอง")
    arena.set_alias(team=second, raw="ทีมหนึ่ง", actor_id=None)
    assert second.alias == "ทีมหนึ่ง"


def test_keeping_your_own_alias_is_not_a_collision(arena):
    """กดบันทึกซ้ำด้วยชื่อเดิมต้องไม่กลายเป็น "ชื่อนี้มีคนใช้แล้ว" """
    _u, team = _member(arena, "a@g.swu.ac.th", "ทีมหนึ่ง")
    arena.set_alias(team=team, raw="ชื่อเท่ๆ", actor_id=None)
    arena.set_alias(team=team, raw="ชื่อเท่ๆ", actor_id=None)
    assert team.alias == "ชื่อเท่ๆ"


def test_clearing_the_alias_goes_back_to_the_real_name(arena):
    _u, team = _member(arena, "a@g.swu.ac.th", "ทีมหนึ่ง")
    arena.set_alias(team=team, raw="ชื่อเท่ๆ", actor_id=None)
    arena.set_alias(team=team, raw="", actor_id=None)
    assert team.alias is None
    assert team.display_name(reveal=False) == "ทีมหนึ่ง"


def test_every_change_is_audited(arena):
    _u, team = _member(arena, "a@g.swu.ac.th", "ทีมหนึ่ง")
    arena.set_alias(team=team, raw="ชื่อแรก", actor_id=team.member_ids[0])
    arena.set_alias(team=team, raw="ชื่อสอง", actor_id=team.member_ids[0])
    events = [e for e in arena.store.audit if e.action == "team.alias"]
    assert [(e.payload["before"], e.payload["after"]) for e in events] == [
        (None, "ชื่อแรก"),
        ("ชื่อแรก", "ชื่อสอง"),
    ]


# ── ใครเห็นชื่ออะไรบนกระดาน ─────────────────────────────────────────


def _board(arena, team, kind=RunKind.PUBLIC):
    return build(
        list(arena.queue.runs.values()), arena.store.teams,
        kind=kind, reveal_names=arena.team_acts_as_staff(team),
    )


def test_students_see_the_alias_and_staff_see_the_real_name(arena):
    """**หัวใจของ §6.1** — ก่อนหน้านี้ไม่มีใครเห็นชื่อจริงเลย แม้แต่ผู้สอน"""
    from core.domain import Run, RunStatus, new_id

    arena.staff_emails = frozenset({STAFF})
    _s, staff_team = _member(arena, STAFF, "อาจารย์")
    _u, team = _member(arena, "a@g.swu.ac.th", "สมชาย ใจดี")
    arena.set_alias(team=team, raw="ทีมลับ", actor_id=None)

    competition = next(iter(arena.store.competitions.values()))
    arena.queue.runs["r1"] = Run(
        id="r1", submission_id="s1", team_id=team.id, competition_id=competition.id,
        kind=RunKind.PUBLIC, status=RunStatus.DONE, score=1.0,
    )

    student_view = _board(arena, team)
    staff_view = _board(arena, staff_team)
    assert [r.display_name for r in student_view] == ["ทีมลับ"]
    assert [r.display_name for r in staff_view] == ["สมชาย ใจดี"]


# ── ผ่าน API ────────────────────────────────────────────────────────


@pytest.fixture
def client(arena):
    from fastapi.testclient import TestClient

    from core.api import create_app

    return TestClient(create_app(arena))


def auth(team):
    return {"Authorization": f"Bearer {team.token}"}


def test_any_team_can_set_its_own_alias(arena, client):
    """ไม่ต้องเป็นผู้สอน — §6.1 ให้เป็นสิทธิ์ของทีม"""
    _u, team = _member(arena, "a@g.swu.ac.th", "สมชาย ใจดี")
    res = client.post("/api/teams/alias", data={"alias": " ทีม  ลุยเลย "}, headers=auth(team))
    assert res.status_code == 200, res.text
    assert res.json() == {"alias": "ทีม ลุยเลย", "shown_as": "ทีม ลุยเลย"}


def test_bad_alias_comes_back_with_the_reason(arena, client):
    _u, team = _member(arena, "a@g.swu.ac.th", "สมชาย")
    res = client.post("/api/teams/alias", data={"alias": "Gold"}, headers=auth(team))
    assert res.status_code == 422
    assert "baseline" in res.json()["detail"]


def test_me_reports_what_the_board_shows(arena, client):
    _u, team = _member(arena, "a@g.swu.ac.th", "สมชาย ใจดี")
    before = client.get("/api/me", headers=auth(team)).json()["team"]
    assert before["alias"] is None and before["shown_as"] == "สมชาย ใจดี"

    client.post("/api/teams/alias", data={"alias": "ทีมลับ"}, headers=auth(team))
    after = client.get("/api/me", headers=auth(team)).json()["team"]
    assert after["alias"] == "ทีมลับ" and after["shown_as"] == "ทีมลับ"


def test_alias_needs_a_token(client):
    assert client.post("/api/teams/alias", data={"alias": "x"}).status_code == 401
