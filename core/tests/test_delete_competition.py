"""ลบโจทย์ที่สร้างผิด — และสิ่งที่ลบ**ไม่ได้**

ผู้สอนที่กรอกฟอร์มผิดต้องมีทางแก้ ไม่งั้นวิชาจะสะสมโจทย์ร้างที่นิสิตเห็นแล้วสับสน

**แต่พอมีคนส่งงานแล้ว การลบคือคนละเรื่อง** — มันทำลายงานของนิสิตและประวัติการ
ให้คะแนนไปพร้อมกัน ซึ่งไม่ใช่สิ่งที่ปุ่มบนหน้าเว็บควรทำได้ · ด่านอยู่ที่ `Arena`
ไม่ใช่ที่ endpoint เพราะกติกาต้องเหมือนกันไม่ว่าคำสั่งจะมาจากไหน
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from core.api import create_app
from core.calendar import build_phases, day_range
from core.domain import Competition, Course, RunKind, new_id
from core.service import CompetitionInUse, build_arena

AJ = "aj@g.swu.ac.th"
STUDENT = "student@g.swu.ac.th"
COURSE = "cp462-1-2026"
SLUG = "cp462-typo-1-2026"


@pytest.fixture
def arena(tmp_path):
    a = build_arena(tmp_path / "artifacts", db_path=tmp_path / "arena.db",
                    course_staff={COURSE: frozenset({AJ})})
    a.store.save_course(Course(id=COURSE, name="CP462", join_code="AAAAAA"))
    now = datetime.now(timezone.utc)
    a.store.save_competition(Competition(
        id=new_id(), course_id=COURSE, slug=SLUG, title="พิมพ์ผิด",
        task_type="prediction", env_plugin="x:PLUGIN",
        config_text="kind: classification\n", config_path="",
        opens_at=now - timedelta(days=1), closes_at=now + timedelta(days=30),
        phases=build_phases({
            "warmup": day_range("2026-09-15", "2026-09-30"),
            "main": day_range("2026-10-01", "2026-10-31"),
            "final": day_range("2026-11-01", "2026-11-30"),
        }),
    ))
    return a


@pytest.fixture
def client(arena):
    return TestClient(create_app(arena))


def sign_in(arena, email):
    return arena.sign_in(email=email, name=email.split("@")[0], google_sub=email)


def auth(user):
    return {"Authorization": f"Bearer {user.token}"}


def add_submission(arena, slug):
    """งานที่นิสิตส่งเข้ามาหนึ่งชิ้น พร้อม run ของมัน"""
    from core.domain import Run, Submission

    competition = arena.store.competition_by_slug(slug)
    student = sign_in(arena, STUDENT)
    team = arena.enroll(user=student, join_code="AAAAAA")
    sub = Submission(id=new_id(), competition_id=competition.id, team_id=team.id,
                     submitted_by=student.id, artifact_url="file:///ไม่มี.zip",
                     artifact_sha256="0" * 64)
    arena.store.save_submission(sub)
    arena.queue.enqueue(Run(id=new_id(), submission_id=sub.id, competition_id=competition.id,
                            team_id=team.id, kind=RunKind.PUBLIC))
    return sub


# ── ลบได้เมื่อยังไม่มีใครส่งงาน ────────────────────────────────────


def test_the_instructor_can_delete_a_competition_they_created_by_mistake(arena, client):
    """**เหตุผลทั้งหมดของฟีเจอร์นี้** — กรอกผิดแล้วต้องแก้ได้"""
    aj = sign_in(arena, AJ)
    reply = client.post(f"/api/competitions/{SLUG}/delete", headers=auth(aj))

    assert reply.status_code == 200, reply.text
    assert reply.json()["deleted"]["slug"] == SLUG
    assert arena.store.competition_by_slug(SLUG) is None


def test_it_stays_deleted_after_a_restart(arena, tmp_path):
    """ลบแล้วต้องหายจริง — โจทย์ที่กลับมาตอนรีสตาร์ทแย่กว่าไม่มีปุ่มลบ"""
    client = TestClient(create_app(arena))
    aj = sign_in(arena, AJ)
    client.post(f"/api/competitions/{SLUG}/delete", headers=auth(aj))
    arena.store.db.close()

    again = build_arena(tmp_path / "artifacts", db_path=tmp_path / "arena.db",
                        course_staff={COURSE: frozenset({AJ})})
    assert again.store.competition_by_slug(SLUG) is None


def test_what_was_deleted_is_written_to_the_audit_log(arena, client):
    """README §7 — และ config ที่เก็บไว้ทำให้สร้างกลับได้ถ้าลบผิดตัว"""
    aj = sign_in(arena, AJ)
    client.post(f"/api/competitions/{SLUG}/delete", headers=auth(aj))

    event = next(e for e in arena.store.audit if e.action == "competition.deleted")
    assert event.actor_id == aj.id
    assert event.payload["slug"] == SLUG
    assert event.payload["config"] == "kind: classification\n", "ต้องเก็บ config ไว้ให้สร้างกลับได้"


# ── 🔒 ลบไม่ได้เมื่อมีงานของนิสิตอยู่ ──────────────────────────────


def test_a_competition_with_submissions_cannot_be_deleted(arena, client):
    """**ข้อสำคัญที่สุด** — ปุ่มบนหน้าเว็บต้องไม่มีทางทำลายงานของนิสิต"""
    add_submission(arena, SLUG)
    aj = sign_in(arena, AJ)

    reply = client.post(f"/api/competitions/{SLUG}/delete", headers=auth(aj))
    assert reply.status_code == 422
    detail = reply.json()["detail"]
    assert "ทำลายงานของนิสิต" in detail
    assert "เลื่อนวันปิดรับ" in detail, "ต้องบอกว่าทำอะไรแทนได้"
    assert arena.store.competition_by_slug(SLUG) is not None


def test_the_rule_lives_in_arena_not_in_the_endpoint(arena):
    """เรียกผ่าน CLI หรือสคริปต์ก็ต้องโดนกติกาเดียวกัน"""
    add_submission(arena, SLUG)
    aj = sign_in(arena, AJ)
    with pytest.raises(CompetitionInUse, match="ทำลายงานของนิสิต"):
        arena.delete_competition(slug=SLUG, actor=aj)


# ── ใครลบได้ ──────────────────────────────────────────────────────


def test_a_student_cannot_delete_a_competition(arena, client):
    student = sign_in(arena, STUDENT)
    reply = client.post(f"/api/competitions/{SLUG}/delete", headers=auth(student))
    assert reply.status_code == 422
    assert arena.store.competition_by_slug(SLUG) is not None


def test_an_instructor_of_another_course_cannot_delete_it(arena, client):
    other = sign_in(arena, "aj-cp463@g.swu.ac.th")
    assert client.post(f"/api/competitions/{SLUG}/delete",
                       headers=auth(other)).status_code == 422
    assert arena.store.competition_by_slug(SLUG) is not None


def test_deleting_something_that_is_not_there_is_a_404(arena, client):
    aj = sign_in(arena, AJ)
    assert client.post("/api/competitions/ไม่มีอันนี้/delete",
                       headers=auth(aj)).status_code == 404


def test_deleting_one_leaves_the_others_alone(arena, client):
    """ไม่มี CASCADE — การลบต้องแตะแถวเดียว"""
    now = datetime.now(timezone.utc)
    arena.store.save_competition(Competition(
        id=new_id(), course_id=COURSE, slug="cp462-keep-1-2026", title="เก็บไว้",
        task_type="prediction", env_plugin="x:PLUGIN", config_path="",
        opens_at=now, closes_at=now + timedelta(days=30),
    ))
    aj = sign_in(arena, AJ)
    client.post(f"/api/competitions/{SLUG}/delete", headers=auth(aj))

    assert arena.store.competition_by_slug("cp462-keep-1-2026") is not None
