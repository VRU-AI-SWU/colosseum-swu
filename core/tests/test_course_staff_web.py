"""แต่งตั้งผู้สอน/TA ของวิชาจากหน้าเว็บ — และสิ่งที่หน้าเว็บ**ทำไม่ได้**

เดิมสิทธิ์ทั้งหมดอยู่ใน `/etc/arena.env` ที่เดียว ซึ่งปลอดภัยแต่แปลว่าการเพิ่ม TA
หนึ่งคนต้อง ssh + sudo + restart · ตอนนี้แยกเป็นสามชั้น

    1. ARENA_STAFF_EMAILS            ทั้งระบบ · ต้องมี root
    2. ARENA_COURSE_STAFF_<COURSE>   รายวิชา · ต้องมี root
    3. ตาราง course_staff            รายวิชา · แต่งตั้งจากหน้าเว็บได้

**ชั้นบนไม่มีทางถูกชั้นล่างลบทิ้ง** — นั่นคือสมอที่ทำให้มีทางกู้คืนเสมอ

สองอย่างที่ผิดแล้วเจ็บ

  · **ยึดวิชาถาวร** — ถ้าถอดคนที่ตั้งจากไฟล์ได้ คนที่ยึดบัญชีผู้สอนได้ครั้งเดียว
    จะถอดคนอื่นออกจนหมดแล้วไม่มีใครเข้าไปแก้ได้อีกเลย
  · **วิชากำพร้า** — ถ้าถอดคนสุดท้ายได้ วิชานั้นจะไม่มีใครแก้อะไรได้ ต้องให้คนที่
    มี root ลงมือ ซึ่งแพงกว่าการกันไว้ตั้งแต่แรกมาก
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core.api import create_app
from core.domain import Course
from core.service import StaffChangeRejected, build_arena, clean_staff_email

OWNER = "owner@g.swu.ac.th"          # ARENA_STAFF_EMAILS — ทั้งระบบ
FILE_AJ = "file-aj@g.swu.ac.th"      # ARENA_COURSE_STAFF_CP462_1_2026
WEB_TA = "web-ta@g.swu.ac.th"        # แต่งตั้งผ่านหน้าเว็บ
STUDENT = "student@g.swu.ac.th"
OTHER_AJ = "other@g.swu.ac.th"       # ผู้สอนของอีกวิชา

COURSE = "cp462-1-2026"
OTHER_COURSE = "cp463-1-2026"


@pytest.fixture
def arena(tmp_path):
    a = build_arena(
        tmp_path / "artifacts",
        db_path=tmp_path / "arena.db",
        staff_emails=frozenset({OWNER}),
        course_staff={COURSE: frozenset({FILE_AJ}), OTHER_COURSE: frozenset({OTHER_AJ})},
    )
    a.store.save_course(Course(id=COURSE, name="CP462", join_code="AAAAAA"))
    a.store.save_course(Course(id=OTHER_COURSE, name="CP463", join_code="BBBBBB"))
    return a


@pytest.fixture
def client(arena):
    return TestClient(create_app(arena))


def sign_in(arena, email: str):
    return arena.sign_in(email=email, name=email.split("@")[0], google_sub=email)


def auth(user) -> dict:
    return {"Authorization": f"Bearer {user.token}"}


def add(client, user, email, course=COURSE):
    return client.post(f"/api/courses/{course}/staff", headers=auth(user),
                       data={"email": email})


def drop(client, user, email, course=COURSE):
    return client.post(f"/api/courses/{course}/staff/remove", headers=auth(user),
                       data={"email": email})


def emails(reply) -> list[str]:
    return [s["email"] for s in reply.json()["staff"]]


# ── อีเมลที่รับได้ ────────────────────────────────────────────────


@pytest.mark.parametrize("raw, expected", [
    ("  AJ@G.SWU.AC.TH ", "aj@g.swu.ac.th"),
    ("ta@g.swu.ac.th", "ta@g.swu.ac.th"),
])
def test_emails_are_normalised_to_lower_case(raw, expected):
    """การจับคู่สิทธิ์ใช้ตัวพิมพ์เล็ก — ถ้าเก็บตามที่พิมพ์ รายการจะมีคนเดียวสองแถว"""
    assert clean_staff_email(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", "ไม่ใช่อีเมล", "a@b", "a b@c.d", "@g.swu.ac.th"])
def test_a_string_that_is_not_an_email_is_refused(bad):
    with pytest.raises(StaffChangeRejected):
        clean_staff_email(bad)


# ── ใครแต่งตั้งใครได้ ─────────────────────────────────────────────


def test_the_course_instructor_can_appoint_a_ta_without_touching_the_server(arena, client):
    """**เหตุผลทั้งหมดของฟีเจอร์นี้** — เพิ่ม TA ต้องไม่ต้อง ssh + sudo + restart"""
    aj = sign_in(arena, FILE_AJ)
    reply = add(client, aj, WEB_TA)
    assert reply.status_code == 200, reply.text
    assert WEB_TA in emails(reply)

    ta = sign_in(arena, WEB_TA)
    assert arena.can_manage_course(ta.email, COURSE)


def test_a_student_cannot_appoint_themselves(arena, client):
    student = sign_in(arena, STUDENT)
    assert add(client, student, STUDENT).status_code == 422
    assert not arena.can_manage_course(STUDENT, COURSE)


def test_an_instructor_of_another_course_cannot_reach_into_this_one(arena, client):
    """สิทธิ์ขยายได้เฉพาะในวิชาที่ตัวเองดูแล — รัศมีความเสียหายคือหนึ่งวิชา"""
    other = sign_in(arena, OTHER_AJ)
    assert add(client, other, other.email).status_code == 422
    assert not arena.can_manage_course(OTHER_AJ, COURSE)


def test_the_system_owner_can_appoint_in_any_course(arena, client):
    owner = sign_in(arena, OWNER)
    assert add(client, owner, WEB_TA, course=OTHER_COURSE).status_code == 200


def test_a_newly_appointed_ta_can_appoint_others_in_that_course(arena, client):
    """TA ที่ถูกตั้งผ่านเว็บมีสิทธิ์เท่าผู้สอนของวิชานั้น — รวมถึงเพิ่มคนต่อ

    ตั้งใจให้เป็นแบบนี้ · การแยกระดับ "ผู้สอน" กับ "TA" ภายในวิชาเดียวกันเพิ่ม
    ความซับซ้อนโดยที่ยังไม่มีใครขอ และวิชาหนึ่งมีคนไม่กี่คนที่เชื่อใจกันอยู่แล้ว
    """
    aj = sign_in(arena, FILE_AJ)
    add(client, aj, WEB_TA)
    ta = sign_in(arena, WEB_TA)
    assert add(client, ta, "another@g.swu.ac.th").status_code == 200


def test_appointing_someone_who_never_signed_in_works(arena, client):
    """กรณีปกติ — ผู้สอนเพิ่ม TA ไว้ก่อนเปิดเทอม TA ค่อยล็อกอินทีหลัง"""
    aj = sign_in(arena, FILE_AJ)
    add(client, aj, "ยังไม่เคยเข้า@g.swu.ac.th")

    later = sign_in(arena, "ยังไม่เคยเข้า@g.swu.ac.th")
    assert arena.can_manage_course(later.email, COURSE)


def test_appointing_the_same_person_twice_is_not_an_error(arena, client):
    aj = sign_in(arena, FILE_AJ)
    add(client, aj, WEB_TA)
    reply = add(client, aj, WEB_TA)
    assert reply.status_code == 200
    assert emails(reply).count(WEB_TA) == 1


# ── 🔒 สิ่งที่หน้าเว็บทำไม่ได้ ────────────────────────────────────


def test_the_list_says_which_rows_the_web_cannot_remove(arena, client):
    """หน้าเว็บต้องรู้ก่อนวาดปุ่ม — ปุ่มที่กดแล้วโดนปฏิเสธอ่านเหมือนระบบพัง"""
    aj = sign_in(arena, FILE_AJ)
    add(client, aj, WEB_TA)

    rows = {s["email"]: s for s in client.get(
        f"/api/courses/{COURSE}/staff", headers=auth(aj)).json()["staff"]}
    assert rows[OWNER] == {"email": OWNER, "source": "system", "removable": False}
    assert rows[FILE_AJ] == {"email": FILE_AJ, "source": "file", "removable": False}
    assert rows[WEB_TA] == {"email": WEB_TA, "source": "web", "removable": True}


def test_someone_configured_in_the_env_file_cannot_be_removed_from_the_web(arena, client):
    """**สมอที่ทำให้มีทางกู้คืนเสมอ** — ถอดได้เฉพาะที่ไฟล์บนเซิร์ฟเวอร์

    ถ้าถอดได้จากหน้าเว็บ คนที่ยึดบัญชีผู้สอนได้ครั้งเดียวจะถอดคนอื่นออกจนหมด
    แล้วยึดวิชาไว้ถาวรโดยไม่มีใครเข้าไปแก้ได้อีก
    """
    aj = sign_in(arena, FILE_AJ)
    for victim in (OWNER, FILE_AJ):
        reply = drop(client, aj, victim)
        assert reply.status_code == 422
        assert "/etc/arena.env" in reply.json()["detail"]
    assert arena.can_manage_course(OWNER, COURSE)
    assert arena.can_manage_course(FILE_AJ, COURSE)


def test_the_last_manager_cannot_be_removed(arena, tmp_path):
    """วิชาที่ไม่มีผู้ดูแลเลยต้องให้คนที่มี root มาแก้ — กันไว้ถูกกว่ามาก"""
    lonely = build_arena(
        tmp_path / "a2", db_path=tmp_path / "a2.db",
        course_staff={COURSE: frozenset()},
    )
    lonely.store.save_course(Course(id=COURSE, name="CP462", join_code="AAAAAA"))
    client = TestClient(create_app(lonely))

    # ตั้งคนแรกผ่านเมธอดตรงๆ เพราะยังไม่มีใครมีสิทธิ์จะกดผ่าน API ได้
    lonely.store.add_course_staff(COURSE, WEB_TA)
    only = sign_in(lonely, WEB_TA)

    reply = drop(client, only, WEB_TA)
    assert reply.status_code == 422
    assert "คนสุดท้าย" in reply.json()["detail"]
    assert lonely.can_manage_course(WEB_TA, COURSE)


def test_removing_yourself_works_when_someone_else_remains(arena, client):
    """ลาออกจากวิชาเองได้ ตราบใดที่ยังมีคนอื่นดูแลอยู่"""
    aj = sign_in(arena, FILE_AJ)
    add(client, aj, WEB_TA)
    ta = sign_in(arena, WEB_TA)

    assert drop(client, ta, WEB_TA).status_code == 200
    assert not arena.can_manage_course(WEB_TA, COURSE)


def test_removing_someone_who_is_not_on_the_list_says_so(arena, client):
    aj = sign_in(arena, FILE_AJ)
    reply = drop(client, aj, "ไม่มีคนนี้@g.swu.ac.th")
    assert reply.status_code == 422
    assert "ไม่ได้อยู่ในรายชื่อ" in reply.json()["detail"]


def test_a_student_cannot_read_the_staff_list(arena, client):
    """ไม่ใช่ความลับ แต่เป็นอีเมลของคนจริง ซึ่งนิสิตทั้งชั้นไม่ต้องเห็นเป็นรายการ"""
    student = sign_in(arena, STUDENT)
    assert client.get(f"/api/courses/{COURSE}/staff", headers=auth(student)).status_code == 403


def test_an_unknown_course_is_a_404(arena, client):
    owner = sign_in(arena, OWNER)
    assert client.get("/api/courses/ไม่มี/staff", headers=auth(owner)).status_code == 404
    assert add(client, owner, WEB_TA, course="ไม่มี").status_code == 404


# ── อยู่รอดข้ามการรีสตาร์ท ────────────────────────────────────────


def test_appointments_survive_a_restart(arena, tmp_path):
    """สิทธิ์ที่หายตอนรีสตาร์ทแย่กว่าไม่มีฟีเจอร์นี้เลย — TA จะงงว่าทำไมใช้ไม่ได้"""
    client = TestClient(create_app(arena))
    aj = sign_in(arena, FILE_AJ)
    add(client, aj, WEB_TA)
    arena.store.db.close()

    again = build_arena(
        tmp_path / "artifacts", db_path=tmp_path / "arena.db",
        staff_emails=frozenset({OWNER}),
        course_staff={COURSE: frozenset({FILE_AJ})},
    )
    assert again.can_manage_course(WEB_TA, COURSE)
    assert [s["email"] for s in again.course_managers(COURSE)] == [OWNER, FILE_AJ, WEB_TA]


def test_every_change_is_written_to_the_audit_log(arena, client):
    """README §7 — ต้องย้อนดูได้เสมอว่าใครให้สิทธิ์ใครเมื่อไร"""
    aj = sign_in(arena, FILE_AJ)
    add(client, aj, WEB_TA)
    drop(client, aj, WEB_TA)

    actions = [(e.action, e.payload.get("email"), e.actor_id) for e in arena.store.audit]
    assert ("course.staff_added", WEB_TA, aj.id) in actions
    assert ("course.staff_removed", WEB_TA, aj.id) in actions
