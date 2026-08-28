"""ล็อกอินด้วย Google Workspace และการจัดทีม — README §11

นิสิตจัดทีมกันเอง และคนที่หากลุ่มไม่ได้ลงเอยด้วยการทำคนเดียว **สถานะ solo จึงต้อง
เป็นเรื่องปกติที่ใช้งานได้ทันที ไม่ใช่ข้อผิดพลาดที่ต้องแก้ก่อนถึงจะเริ่มได้**
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from core.api import create_app
from core.auth import AuthError, GoogleAuth, GoogleIdentity
from core.domain import RunKind, Run, RunStatus
from core.leaderboard import build
from core.service import MAX_TEAM_SIZE, InviteInvalid, TeamFull
from core.wiring import demo_arena

COURSE = "cp463-1-2026"


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.delenv("ARENA_SECRETS", raising=False)
    a, _teams = demo_arena(tmp_path / "artifacts", teams=0)
    return a


def sign_in(arena, sub, email, name):
    """ล็อกอิน + เข้าวิชา — คืน `(user, team)` เหมือนที่ `sign_in` เคยคืน

    ตอนนี้เป็นสองขั้น: ล็อกอินรู้ว่าเป็นใคร แล้วค่อยใส่รหัสเข้าวิชา
    เทสต์ยังอยากได้ทั้งคู่ในบรรทัดเดียว จึงห่อไว้ที่นี่
    """
    user = arena.sign_in(google_sub=sub, email=email, name=name)
    team = arena.enroll(user=user, join_code=arena.store.course(COURSE).join_code)
    return user, team


# ── ล็อกอิน ─────────────────────────────────────────────────────────


def test_first_login_gets_a_working_team_immediately(arena):
    """ไม่มีหน้าจอ "กรุณาเลือกทีม" ให้ติด — ล็อกอินเสร็จส่งงานได้เลย"""
    user, team = sign_in(arena, "sub-1", "a@g.swu.ac.th", "นิสิต ก")
    assert team.member_ids == [user.id]
    assert team.is_active
    assert team.token and team.token != team.id, "โทเคนต้องแยกจาก id และเดาไม่ได้"
    assert len(team.invite_code) == 6


def test_second_login_returns_the_same_team(arena):
    _u1, t1 = sign_in(arena, "sub-1", "a@g.swu.ac.th", "นิสิต ก")
    _u2, t2 = sign_in(arena, "sub-1", "a@g.swu.ac.th", "นิสิต ก")
    assert t1.id == t2.id
    assert len(arena.store.users) == 1


def test_changed_email_still_matches_the_same_person(arena):
    """จับคู่ด้วย google_sub ไม่ใช่อีเมล — นิสิตเปลี่ยนนามสกุลแล้วอีเมลเปลี่ยนได้"""
    user, team = sign_in(arena, "sub-1", "old@g.swu.ac.th", "ชื่อเดิม")
    user2, team2 = sign_in(arena, "sub-1", "new@g.swu.ac.th", "ชื่อใหม่")
    assert user2.id == user.id and team2.id == team.id
    assert user2.email == "new@g.swu.ac.th"
    assert len(arena.store.users) == 1, "ต้องไม่กลายเป็นคนใหม่"


# ── เข้าทีม ─────────────────────────────────────────────────────────


def test_joining_a_team_dissolves_the_empty_solo_team(arena):
    owner, host = sign_in(arena, "sub-1", "a@g.swu.ac.th", "เจ้าของทีม")
    joiner, solo = sign_in(arena, "sub-2", "b@g.swu.ac.th", "คนเข้าร่วม")

    joined = arena.join_team(user=joiner, invite_code=host.invite_code, course_id=COURSE)

    assert joined.id == host.id
    assert set(joined.member_ids) == {owner.id, joiner.id}
    assert not solo.is_active, "ทีมเดี่ยวที่ว่างลงต้องถูกยุบ"
    assert solo.id in arena.store.teams, "แต่ยังต้องอยู่ใน store เพื่อ audit"


def test_invite_code_is_case_insensitive(arena):
    _o, host = sign_in(arena, "sub-1", "a@g.swu.ac.th", "เจ้าของ")
    joiner, _ = sign_in(arena, "sub-2", "b@g.swu.ac.th", "ผู้เข้าร่วม")
    joined = arena.join_team(
        user=joiner, invite_code=host.invite_code.lower(), course_id=COURSE
    )
    assert joined.id == host.id


def test_bad_invite_code_says_what_to_do(arena):
    joiner, _ = sign_in(arena, "sub-2", "b@g.swu.ac.th", "ผู้เข้าร่วม")
    with pytest.raises(InviteInvalid, match="ขอรหัสใหม่"):
        arena.join_team(user=joiner, invite_code="ZZZZZZ", course_id=COURSE)


def test_joining_the_team_you_are_already_in_is_not_an_error(arena):
    owner, host = sign_in(arena, "sub-1", "a@g.swu.ac.th", "เจ้าของ")
    same = arena.join_team(user=owner, invite_code=host.invite_code, course_id=COURSE)
    assert same.id == host.id
    assert same.member_ids == [owner.id], "ต้องไม่ถูกเพิ่มซ้ำ"


def test_team_size_is_capped(arena):
    _o, host = sign_in(arena, "sub-0", "h@g.swu.ac.th", "เจ้าของ")
    for i in range(1, MAX_TEAM_SIZE):
        u, _ = sign_in(arena, f"sub-{i}", f"m{i}@g.swu.ac.th", f"สมาชิก {i}")
        arena.join_team(user=u, invite_code=host.invite_code, course_id=COURSE)

    extra, _ = sign_in(arena, "sub-x", "x@g.swu.ac.th", "คนเกิน")
    with pytest.raises(TeamFull):
        arena.join_team(user=extra, invite_code=host.invite_code, course_id=COURSE)


def test_dissolved_team_drops_off_the_leaderboard(arena):
    """คะแนน solo ต้องหายจากกระดานเมื่อเจ้าตัวไปเข้าทีมอื่น

    เพราะมันคือผลงานของคนเดียว ไม่ใช่ของทีมใหม่ · แต่ยังต้องตรวจย้อนหลังได้
    """
    _o, host = sign_in(arena, "sub-1", "a@g.swu.ac.th", "เจ้าของ")
    joiner, solo = sign_in(arena, "sub-2", "b@g.swu.ac.th", "คนเข้าร่วม")

    runs = [
        Run(id="r1", submission_id="s1", competition_id="c", team_id=solo.id,
            kind=RunKind.PUBLIC, status=RunStatus.DONE, score=1.5),
        Run(id="r2", submission_id="s2", competition_id="c", team_id=host.id,
            kind=RunKind.PUBLIC, status=RunStatus.DONE, score=0.5),
    ]
    before = build(runs, arena.store.teams)
    assert [r.team_id for r in before] == [solo.id, host.id]

    arena.join_team(user=joiner, invite_code=host.invite_code, course_id=COURSE)

    after = build(runs, arena.store.teams)
    assert [r.team_id for r in after] == [host.id], "ทีมที่ยุบแล้วต้องไม่อยู่บนกระดาน"
    assert any(e.action == "team.dissolved" for e in arena.store.audit)


# ── เปลี่ยนโทเคน ────────────────────────────────────────────────────


def test_rotating_the_token_kills_the_old_one(arena):
    """โทเคนเดิมต้องใช้ไม่ได้ทันที ไม่งั้นการเปลี่ยนไม่ได้แก้ปัญหาที่มันหลุดไปแล้ว"""
    user, _team = sign_in(arena, "sub-1", "a@g.swu.ac.th", "นิสิต")
    old = user.token
    client = TestClient(create_app(arena))

    res = client.post(
        "/api/users/rotate-token", headers={"Authorization": f"Bearer {old}"}
    )
    assert res.status_code == 200, res.text
    new = res.json()["token"]

    assert new != old
    assert len(new) >= 24, "โทเคนใหม่ต้องยาวและสุ่มเหมือนเดิม"
    assert client.get(
        "/api/me", headers={"Authorization": f"Bearer {old}"}
    ).status_code == 401
    assert client.get(
        "/api/me", headers={"Authorization": f"Bearer {new}"}
    ).status_code == 200


def test_rotation_keeps_the_team_and_its_history(arena):
    """เปลี่ยนแค่โทเคน — ทีม สมาชิก และรหัสเชิญต้องไม่ขยับ"""
    user, team = sign_in(arena, "sub-1", "a@g.swu.ac.th", "นิสิต")
    team_id, members, invite = team.id, list(team.member_ids), team.invite_code

    arena.rotate_user_token(user=user)

    assert team.id == team_id
    assert team.member_ids == members
    assert team.invite_code == invite, "รหัสเชิญเป็นคนละเรื่อง ต้องไม่เปลี่ยนตาม"
    assert team.is_active
    assert any(e.action == "user.token_rotated" for e in arena.store.audit)


def test_rotation_only_affects_the_person_who_asked(arena):
    """**ของแถมจากการย้ายโทเคนมาที่คน** — เพื่อนร่วมทีมไม่ต้องตั้งค่าใหม่

    เดิมโทเคนเป็นของทีม โทเคนหลุดของคนเดียวจึงบังคับให้ทุกคนในทีมเปลี่ยนพร้อมกัน
    ทั้งที่ไม่ได้ทำอะไรผิด
    """
    host_user, host = sign_in(arena, "sub-1", "a@g.swu.ac.th", "เจ้าของ")
    mate, _ = sign_in(arena, "sub-2", "b@g.swu.ac.th", "เพื่อน")
    arena.join_team(user=mate, invite_code=host.invite_code, course_id=COURSE)
    mate_token_before = mate.token

    arena.rotate_user_token(user=host_user)

    assert mate.token == mate_token_before, "โทเคนของเพื่อนต้องไม่ถูกแตะ"
    client = TestClient(create_app(arena))
    assert client.get(
        "/api/me", headers={"Authorization": f"Bearer {mate.token}"}
    ).status_code == 200


def test_rotation_is_recorded_for_audit(arena):
    """README §7 — ต้องย้อนดูได้ว่าใครเปลี่ยนเมื่อไร"""
    user, _team = sign_in(arena, "sub-1", "a@g.swu.ac.th", "นิสิต")
    arena.rotate_user_token(user=user)
    event = next(e for e in arena.store.audit if e.action == "user.token_rotated")
    assert event.target_id == user.id
    assert event.actor_id == user.id


# ── ตัวตนจาก Google ─────────────────────────────────────────────────


def cfg(**kw) -> GoogleAuth:
    return GoogleAuth(
        client_id="cid", client_secret="secret",
        redirect_uri="https://api.example/auth/google/callback",
        web_origin="https://web.example", **kw,
    )


def test_outside_domain_is_rejected_with_a_useful_message():
    with pytest.raises(AuthError, match="g.swu.ac.th"):
        cfg()._require_allowed(
            GoogleIdentity(sub="s", email="someone@gmail.com", name="ใคร", hd=None)
        )


def test_hd_claim_wins_over_the_email_suffix():
    """บัญชีในองค์กรที่ตั้ง alias เป็นโดเมนอื่นต้องยังผ่าน — `hd` คือของจริง"""
    cfg()._require_allowed(
        GoogleIdentity(sub="s", email="a@alias.example", name="นิสิต", hd="g.swu.ac.th")
    )


def test_state_round_trips_and_expires():
    c = cfg()
    state = c.make_state()
    c.check_state(state)  # ไม่ throw

    with pytest.raises(AuthError, match="หมดอายุ"):
        c.check_state(state, now=time.time() + 3600)


def test_tampered_state_is_rejected():
    c = cfg()
    raw, _, sig = c.make_state().partition(".")
    with pytest.raises(AuthError):
        c.check_state(f"{raw}x.{sig}")
    with pytest.raises(AuthError):
        c.check_state(raw)


def test_state_signed_by_another_instance_is_rejected():
    """คีย์สุ่มใหม่ทุกครั้งที่บริการเริ่ม — state ที่ค้างข้าม restart ต้องใช้ไม่ได้"""
    with pytest.raises(AuthError):
        cfg().check_state(cfg().make_state())


def test_token_is_returned_through_the_url_fragment():
    """ต้องเป็น `#` ไม่ใช่ `?` — fragment ไม่ถูกส่งไปเซิร์ฟเวอร์จึงไม่ไปโผล่ใน access log"""
    url = cfg().redirect_back("secret-token")
    assert url.startswith("https://web.example/#token=")
    assert "?" not in url


# ── ผ่าน HTTP จริง ──────────────────────────────────────────────────


def test_login_endpoint_is_clear_when_google_is_not_configured(arena):
    client = TestClient(create_app(arena), follow_redirects=False)
    res = client.get("/auth/google/login")
    assert res.status_code == 503
    assert "ARENA_GOOGLE_CLIENT_ID" in res.json()["detail"]


def test_login_redirects_to_google_with_the_domain_restriction(arena):
    client = TestClient(create_app(arena, google=cfg()), follow_redirects=False)
    res = client.get("/auth/google/login")
    assert res.status_code == 302
    location = res.headers["location"]
    assert location.startswith("https://accounts.google.com/")
    assert "hd=g.swu.ac.th" in location
    assert "client_id=cid" in location


def test_callback_failure_sends_the_student_back_to_the_page(arena):
    """ปลายทางคือเบราว์เซอร์ของนิสิต — ห้ามโชว์ JSON ดิบใส่หน้าคนที่แค่กดปุ่มล็อกอิน"""
    client = TestClient(create_app(arena, google=cfg()), follow_redirects=False)
    res = client.get("/auth/google/callback", params={"error": "access_denied"})
    assert res.status_code == 302
    assert res.headers["location"].startswith("https://web.example/#error=")


def test_me_and_join_over_http(arena):
    _o, host = sign_in(arena, "sub-1", "a@g.swu.ac.th", "เจ้าของ")
    joiner, solo = sign_in(arena, "sub-2", "b@g.swu.ac.th", "ผู้เข้าร่วม")
    client = TestClient(create_app(arena))
    joiner_auth = {"Authorization": f"Bearer {joiner.token}"}

    me = client.get("/api/me", headers=joiner_auth).json()
    assert len(me["enrollments"]) == 1, "อยู่วิชาเดียว"
    mine = me["enrollments"][0]["team"]
    assert mine["is_solo"] is True
    assert mine["members"] == [{"name": "ผู้เข้าร่วม", "email": "b@g.swu.ac.th"}]
    assert mine["invite_code"] == solo.invite_code

    res = client.post(
        "/api/teams/join",
        headers=joiner_auth,
        data={"course_id": COURSE, "invite_code": host.invite_code},
    )
    assert res.status_code == 200, res.text
    assert "token" not in res.json(), "โทเคนเป็นของคน ไม่เปลี่ยนเมื่อย้ายทีม"

    after = client.get("/api/me", headers=joiner_auth).json()["enrollments"][0]["team"]
    assert len(after["members"]) == 2
    assert after["is_solo"] is False
    assert after["id"] == host.id, "ย้ายเข้าทีมเจ้าของแล้ว"


def test_a_person_can_be_in_two_courses_with_one_token(arena):
    """**เหตุผลทั้งหมดของการย้ายโทเคนมาที่คน**

    นิสิตที่เรียนทั้ง AI และ ML มีทีมคนละทีม แต่ตั้ง `ARENA_TOKEN` ครั้งเดียว
    """
    from core.domain import Course

    other = arena.store.save_course(Course(id="ml-1-2026", name="Machine Learning 1/2026"))
    user, _team = sign_in(arena, "sub-1", "a@g.swu.ac.th", "นิสิต")
    arena.enroll(user=user, join_code=other.join_code)

    client = TestClient(create_app(arena))
    me = client.get("/api/me", headers={"Authorization": f"Bearer {user.token}"}).json()
    assert {e["course"]["id"] for e in me["enrollments"]} == {COURSE, other.id}
    assert len({e["team"]["id"] for e in me["enrollments"]}) == 2, "คนละทีมกัน"


def test_a_dissolved_team_stops_being_yours(arena):
    """ยุบทีมแล้วต้องทำอะไรในวิชานั้นไม่ได้อีก

    **ความหมายเปลี่ยนไปจากตอนโทเคนเป็นของทีม** — เดิมการยุบฆ่าโทเคนทิ้ง
    ตอนนี้โทเคนเป็นของคน คนยังอยู่และยังล็อกอินได้ สิ่งที่หายไปคือ*ทีมในวิชานั้น*
    ผลที่ต้องการเหมือนเดิม: ไม่ขึ้นกระดาน และส่งงานในนามทีมนั้นไม่ได้อีก
    """
    _o, host = sign_in(arena, "sub-1", "a@g.swu.ac.th", "เจ้าของ")
    joiner, solo = sign_in(arena, "sub-2", "b@g.swu.ac.th", "ผู้เข้าร่วม")
    client = TestClient(create_app(arena))
    joiner_auth = {"Authorization": f"Bearer {joiner.token}"}

    arena.join_team(user=joiner, invite_code=host.invite_code, course_id=COURSE)
    assert not solo.is_active, "ทีมเดี่ยวเดิมถูกยุบ"

    # คนยังอยู่และยังใช้โทเคนเดิมได้ — แต่ทีมที่โผล่คือทีมใหม่ ไม่ใช่ทีมที่ยุบไปแล้ว
    me = client.get("/api/me", headers=joiner_auth).json()
    assert me["enrollments"][0]["team"]["id"] == host.id
    assert solo.id not in {e["team"]["id"] for e in me["enrollments"]}

    # และทีมที่ยุบแล้วต้องไม่ใช่ทีมที่ระบบจะใช้ทำงานแทนเราอีก
    assert arena.team_for(user=joiner, course_id=COURSE).id == host.id
