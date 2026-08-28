"""ขนาดทีมเป็นข้อมูลของวิชา ไม่ใช่ค่าคงที่ในโค้ด — และใครเปลี่ยนได้บ้าง

เดิม `MAX_TEAM_SIZE = 6` อยู่ในโค้ด การจะเปลี่ยนขนาดทีมจึงต้องแก้โค้ดแล้ว deploy ใหม่
ซึ่งไม่ใช่สิ่งที่ผู้สอนควรต้องทำกลางเทอม

เรื่องที่ต้องระวังที่สุดคือ **สิทธิ์** — โทเคนใช้ร่วมกันทั้งทีม ถ้าตรวจสิทธิ์แบบ
"มีผู้สอนอยู่ในทีมก็พอ" นิสิตที่อยู่ทีมเดียวกับผู้สอนจะถือโทเคนที่เปลี่ยนกติกาของ
ทั้งวิชาได้ · และการซ่อนปุ่มบนหน้าเว็บไม่ใช่ความปลอดภัย เพราะ endpoint ยิงตรงได้
"""

from __future__ import annotations

import pytest

from core.domain import (
    DEFAULT_MAX_TEAM_SIZE,
    MAX_TEAM_SIZE_CEILING,
    Course,
    Team,
    TeamSizeInvalid,
    User,
    new_id,
)
from core.service import TeamFull
from core.wiring import demo_arena

COURSE = "cp463-1-2026"
STAFF = "aj@g.swu.ac.th"
STUDENT = "nisit@g.swu.ac.th"


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.delenv("ARENA_SECRETS", raising=False)
    a, _teams = demo_arena(tmp_path / "artifacts", teams=0)
    return a


def _member(arena, email: str, name: str) -> tuple[User, Team]:
    """ล็อกอิน + เข้าวิชา — คืน `(user, team)` เหมือนที่ `sign_in` เคยคืน

    ตอนนี้เป็นสองขั้น: ล็อกอินรู้ว่าเป็นใคร แล้วค่อยใส่รหัสเข้าวิชา
    เทสต์ยังอยากได้ทั้งคู่ในบรรทัดเดียว จึงห่อไว้ที่นี่
    """
    sub = f"sub-{email}"
    user = arena.sign_in(google_sub=sub, email=email, name=name)
    team = arena.enroll(user=user, join_code=arena.store.course(COURSE).join_code)
    return user, team


# ── วิชาเป็นเจ้าของกติกา ────────────────────────────────────────────


def test_course_is_created_on_demand_with_the_default_size(arena):
    """deployment ที่ตั้ง competition ไว้ก่อน schema v3 ต้องไม่พังเพราะยังไม่มีแถววิชา"""
    course = arena.store.course("วิชาที่ยังไม่เคยมี")
    assert course.max_team_size == DEFAULT_MAX_TEAM_SIZE
    assert arena.store.courses["วิชาที่ยังไม่เคยมี"] is course


def test_join_uses_the_course_limit_not_the_constant(arena):
    """ลดขนาดทีมแล้ว การเข้าทีมต้องใช้ค่าใหม่ทันที ไม่ใช่ค่าที่คอมไพล์มา"""
    arena.staff_emails = frozenset({STAFF})
    _staff_user, staff_team = _member(arena, STAFF, "อาจารย์")
    arena.set_max_team_size(course_id=COURSE, size=2, actor_id=None)

    owner, owner_team = _member(arena, "a@g.swu.ac.th", "เอ")
    second, _ = _member(arena, "b@g.swu.ac.th", "บี")
    arena.join_team(user=second, invite_code=owner_team.invite_code, course_id=COURSE)
    assert len(owner_team.member_ids) == 2

    third, _ = _member(arena, "c@g.swu.ac.th", "ซี")
    with pytest.raises(TeamFull, match="สูงสุด 2 คน"):
        arena.join_team(user=third, invite_code=owner_team.invite_code, course_id=COURSE)


@pytest.mark.parametrize("bad", [0, -1, MAX_TEAM_SIZE_CEILING + 1])
def test_absurd_sizes_are_rejected(arena, bad):
    """เพดานมีไว้กันพิมพ์ผิด — 60 แทน 6 จะกลายเป็นทั้งห้องเป็นทีมเดียว"""
    with pytest.raises(TeamSizeInvalid):
        arena.set_max_team_size(course_id=COURSE, size=bad, actor_id=None)


def test_cannot_shrink_below_an_existing_team(arena):
    """ปฏิเสธ ดีกว่ายอมแล้วปล่อยให้มีทีมที่ผิดกติกาของตัวเองอยู่

    สถานะแบบนั้นอธิบายให้นิสิตฟังไม่ได้ และจะกลายเป็นข้อโต้แย้งตอนตัดเกรด
    """
    owner, owner_team = _member(arena, "a@g.swu.ac.th", "เอ")
    for i in range(3):
        mate, _ = _member(arena, f"m{i}@g.swu.ac.th", f"เพื่อน {i}")
        arena.join_team(user=mate, invite_code=owner_team.invite_code, course_id=COURSE)
    assert len(owner_team.member_ids) == 4

    with pytest.raises(TeamSizeInvalid) as exc:
        arena.set_max_team_size(course_id=COURSE, size=2, actor_id=None)
    assert owner_team.name in str(exc.value), "ต้องบอกชื่อทีมที่เป็นปัญหา"
    assert "4 คน" in str(exc.value), "ต้องบอกว่าทีมนั้นใหญ่แค่ไหน"
    assert arena.store.course(COURSE).max_team_size == DEFAULT_MAX_TEAM_SIZE, "ต้องไม่เปลี่ยน"


def test_dissolved_teams_do_not_block_shrinking(arena):
    """ทีมที่ยุบแล้วยังอยู่ในฐานข้อมูลเพื่อ audit — แต่ต้องไม่มีสิทธิ์ขวางกติกาใหม่"""
    owner, owner_team = _member(arena, "a@g.swu.ac.th", "เอ")
    for i in range(3):
        mate, _ = _member(arena, f"m{i}@g.swu.ac.th", f"เพื่อน {i}")
        arena.join_team(user=mate, invite_code=owner_team.invite_code, course_id=COURSE)
    owner_team.dissolved_at = __import__("core.domain", fromlist=["utcnow"]).utcnow()
    arena.store.save_team(owner_team)

    arena.set_max_team_size(course_id=COURSE, size=2, actor_id=None)
    assert arena.store.course(COURSE).max_team_size == 2


def test_change_is_written_to_the_audit_trail(arena):
    _user, team = _member(arena, STAFF, "อาจารย์")
    arena.set_max_team_size(course_id=COURSE, size=4, actor_id=team.member_ids[0])
    event = next(e for e in arena.store.audit if e.action == "course.max_team_size")
    assert event.payload["before"] == DEFAULT_MAX_TEAM_SIZE
    assert event.payload["after"] == 4


# ── ใครเปลี่ยนได้บ้าง ───────────────────────────────────────────────


def test_nobody_is_staff_by_default(arena):
    """ค่าเริ่มต้นต้องปลอดภัย — ไม่ตั้ง ARENA_STAFF_EMAILS = ไม่มีใครเป็นผู้สอน"""
    assert arena.staff_emails == frozenset()
    assert arena.is_staff(STAFF) is False


def test_staff_is_decided_by_the_person_not_the_team(arena):
    """พอโทเคนเป็นของคน สิทธิ์ก็ถามจากคนตรงๆ ได้

    เดิมต้องมีกฎ "ทั้งทีมต้องเป็นผู้สอน" เพราะโทเคนใช้ร่วมกันทั้งทีม — ผู้สอนที่ไป
    อยู่ทีมเดียวกับนิสิตจะมอบสิทธิ์ให้นิสิตคนนั้นโดยไม่ตั้งใจ · ข้อจำกัดนั้นหายไปแล้ว
    """
    arena.staff_emails = frozenset({STAFF})
    staff_user, staff_team = _member(arena, STAFF, "อาจารย์")
    student, _ = _member(arena, STUDENT, "นิสิต")
    arena.join_team(user=student, invite_code=staff_team.invite_code, course_id=COURSE)

    assert len(staff_team.member_ids) == 2, "อยู่ทีมเดียวกันแล้ว"
    assert arena.is_staff(staff_user.email) is True
    assert arena.is_staff(student.email) is False


def test_staff_email_matching_ignores_case_and_spaces(arena):
    arena.staff_emails = frozenset({STAFF})
    assert arena.is_staff("  AJ@G.SWU.AC.TH ") is True
    assert arena.is_staff("") is False


# ── ค่าที่ตั้งได้ ───────────────────────────────────────────────────


def test_validated_team_size_accepts_the_whole_allowed_range():
    course = Course(id=COURSE, name=COURSE)
    for size in (1, DEFAULT_MAX_TEAM_SIZE, MAX_TEAM_SIZE_CEILING):
        assert course.validated_team_size(size) == size


def test_setting_the_same_size_is_not_an_error(arena):
    """กดบันทึกซ้ำต้องไม่พัง และต้องไม่เขียน audit ซ้ำโดยไม่จำเป็น"""
    before = len(arena.store.audit)
    arena.set_max_team_size(course_id=COURSE, size=DEFAULT_MAX_TEAM_SIZE, actor_id=None)
    assert len(arena.store.audit) == before


# ── ด่านจริงอยู่ที่ endpoint ไม่ใช่ที่การซ่อนปุ่ม ────────────────────
# หน้าเว็บซ่อนแผงผู้สอนเมื่อ `/api/me` บอกว่าไม่ใช่ผู้สอน แต่นั่นเป็นแค่ความสะอาด
# ของหน้าจอ · โทเคนของทีมอยู่ในมือนิสิตทุกคน และ endpoint ยิงตรงด้วย curl ได้


@pytest.fixture
def client(arena):
    from fastapi.testclient import TestClient

    from core.api import create_app

    return TestClient(create_app(arena))


def auth(user):
    """โทเคนเป็นของ **คน** แล้ว ไม่ใช่ของทีม"""
    return {"Authorization": f"Bearer {user.token}"}


def test_student_token_is_refused_even_though_the_ui_hides_the_panel(arena, client):
    arena.staff_emails = frozenset({STAFF})
    student, _team = _member(arena, STUDENT, "นิสิต")

    res = client.post(
        f"/api/courses/{COURSE}/settings", data={"size": 20}, headers=auth(student)
    )
    assert res.status_code == 403, res.text
    assert arena.store.course(COURSE).max_team_size == DEFAULT_MAX_TEAM_SIZE


def test_staff_token_can_change_it(arena, client):
    arena.staff_emails = frozenset({STAFF})
    staff, _staff_team = _member(arena, STAFF, "อาจารย์")

    res = client.post(
        f"/api/courses/{COURSE}/settings", data={"size": 3}, headers=auth(staff)
    )
    assert res.status_code == 200, res.text
    assert res.json()["course"]["max_team_size"] == 3
    assert arena.store.course(COURSE).max_team_size == 3


def test_unknown_course_is_404_not_a_silent_no_op(arena, client):
    """พิมพ์ชื่อวิชาผิดต้องรู้ทันที ไม่ใช่ตอบ 200 แล้วไม่มีอะไรเปลี่ยน"""
    arena.staff_emails = frozenset({STAFF})
    staff, _staff_team = _member(arena, STAFF, "อาจารย์")

    res = client.post(
        "/api/courses/วิชาที่ไม่มีจริง/settings", data={"size": 3}, headers=auth(staff)
    )
    assert res.status_code == 404, res.text


def test_staff_authority_is_deployment_wide_not_per_course(arena, client):
    """**ข้อจำกัดที่รู้ตัว** — `ARENA_STAFF_EMAILS` เป็นรายชื่อของทั้งเครื่อง

    วันนี้ผู้สอนคนเดียวสอนทุกวิชาบนเครื่องนี้ จึงยังไม่เป็นปัญหา · ถ้าวันหนึ่งมี
    ผู้สอนหลายคนแบ่งกันคนละวิชา ต้องมีรายชื่อผู้สอนรายวิชา ไม่ใช่รายเครื่อง
    เทสต์นี้เขียนไว้ให้เห็นพฤติกรรมจริง ไม่ใช่เพื่อรับรองว่ามันถูกในระยะยาว
    """
    from core.domain import Course

    arena.staff_emails = frozenset({STAFF})
    staff, _staff_team = _member(arena, STAFF, "อาจารย์")
    other = arena.store.save_course(Course(id="วิชาของคนอื่น", name="วิชาของคนอื่น"))

    res = client.post(
        f"/api/courses/{other.id}/settings", data={"size": 3}, headers=auth(staff)
    )
    assert res.status_code == 200, res.text


def test_bad_size_comes_back_with_the_reason_not_just_a_number(arena, client):
    """422 ต้องบอกว่าทีมไหนใหญ่เกินไป — หน้าเว็บเอาข้อความนี้ไปโชว์ตรงๆ"""
    arena.staff_emails = frozenset({STAFF})
    staff, _staff_team = _member(arena, STAFF, "อาจารย์")
    owner, owner_team = _member(arena, "a@g.swu.ac.th", "ทีมใหญ่")
    for i in range(3):
        mate, _ = _member(arena, f"m{i}@g.swu.ac.th", f"เพื่อน {i}")
        arena.join_team(user=mate, invite_code=owner_team.invite_code, course_id=COURSE)

    res = client.post(
        f"/api/courses/{COURSE}/settings", data={"size": 2}, headers=auth(staff)
    )
    assert res.status_code == 422, res.text
    assert "ทีมใหญ่" in res.json()["detail"]


def test_me_reports_the_course_and_whether_you_are_staff(arena, client):
    arena.staff_emails = frozenset({STAFF})
    staff, _staff_team = _member(arena, STAFF, "อาจารย์")
    student, _student_team = _member(arena, STUDENT, "นิสิต")

    staff_view = client.get("/api/me", headers=auth(staff)).json()
    assert staff_view["is_staff"] is True
    assert staff_view["enrollments"][0]["course"]["max_team_size"] == DEFAULT_MAX_TEAM_SIZE

    student_view = client.get("/api/me", headers=auth(student)).json()
    assert student_view["is_staff"] is False
    assert student_view["enrollments"][0]["course"]["id"] == COURSE


def test_no_token_gets_401_not_403(client):
    """ไม่มีโทเคนเลย = ยังไม่รู้ว่าเป็นใคร ซึ่งต่างจาก 'รู้แล้วว่าไม่ใช่ผู้สอน'"""
    assert client.post(f"/api/courses/{COURSE}/settings", data={"size": 3}).status_code == 401


# ── ชื่อวิชา ────────────────────────────────────────────────────────
# วิชาที่ migrate มาจาก schema เก่าได้ชื่อเป็น id ของเครื่อง (`cp463-1-2026`)
# ซึ่งนิสิตต้องอ่าน · ต้องมีทางแก้ที่ไม่ใช่การเข้าไปยุ่งกับฐานข้อมูลตรงๆ


def test_staff_can_rename_a_course(arena, client):
    arena.staff_emails = frozenset({STAFF})
    staff, _t = _member(arena, STAFF, "อาจารย์")
    res = client.post(
        f"/api/courses/{COURSE}/settings",
        data={"name": "CP463 · Artificial Intelligence 1/2026"},
        headers=auth(staff),
    )
    assert res.status_code == 200, res.text
    assert arena.store.course(COURSE).name == "CP463 · Artificial Intelligence 1/2026"


def test_omitted_fields_are_left_alone_not_cleared(arena, client):
    """ฟอร์มที่ส่งเฉพาะฟิลด์ที่แก้เป็นเรื่องปกติ — การตีความว่า "ล้าง" จะลบชื่อทิ้ง"""
    arena.staff_emails = frozenset({STAFF})
    staff, _t = _member(arena, STAFF, "อาจารย์")
    arena.update_course(course_id=COURSE, size=None, name="ชื่อดี", actor_id=None)

    client.post(f"/api/courses/{COURSE}/settings", data={"size": 3}, headers=auth(staff))
    course = arena.store.course(COURSE)
    assert course.name == "ชื่อดี", "ไม่ได้ส่ง name มา = ไม่แตะ"
    assert course.max_team_size == 3


@pytest.mark.parametrize("bad", ["", "   ", "ก" * 61])
def test_bad_course_names_are_rejected(arena, bad):
    from core.domain import CourseNameInvalid

    with pytest.raises(CourseNameInvalid):
        arena.update_course(course_id=COURSE, size=None, name=bad, actor_id=None)


def test_rename_is_audited(arena):
    arena.update_course(course_id=COURSE, size=None, name="ชื่อใหม่", actor_id="u1")
    event = next(e for e in arena.store.audit if e.action == "course.renamed")
    assert event.payload["after"] == "ชื่อใหม่"
    assert event.actor_id == "u1"


def test_students_do_not_see_the_join_code(arena, client):
    """รหัสเข้าวิชาเป็นของผู้สอนไว้แจก — ไม่ใช่ของที่ทุกคนหยิบไปส่งต่อได้เอง"""
    arena.staff_emails = frozenset({STAFF})
    staff, _s = _member(arena, STAFF, "อาจารย์")
    student, _t = _member(arena, STUDENT, "นิสิต")

    staff_view = client.get("/api/me", headers=auth(staff)).json()
    student_view = client.get("/api/me", headers=auth(student)).json()
    assert staff_view["enrollments"][0]["course"]["join_code"]
    assert student_view["enrollments"][0]["course"]["join_code"] is None
