"""สิทธิ์ "ผู้สอนของวิชานี้" และการเลื่อนปฏิทินจากหน้าเว็บ

สองอย่างที่ผิดแล้วเจ็บ

  · **นิสิตเลื่อน deadline ของตัวเองได้** — endpoint ยิงตรงได้ด้วย curl การซ่อนปุ่ม
    บนหน้าเว็บไม่ใช่การป้องกัน
  · **ผู้สอนวิชาหนึ่งไปแก้ปฏิทินของอีกวิชา** — ตอนนี้มีผู้สอนคนเดียวจึงยังไม่ต่างกัน
    แต่พอมี TA หรืออาจารย์คนที่สอง มันจะต่างกันทันที และไม่มีอะไรฟ้อง

รายชื่อผู้สอนอยู่ใน environment ไม่ใช่ฐานข้อมูล (`ARENA_COURSE_STAFF_<COURSE_ID>`)
ด้วยเหตุผลเดียวกับ `ARENA_STAFF_EMAILS` — ถ้าแก้ผ่านหน้าเว็บได้ คนที่ยึดสิทธิ์
ได้ครั้งเดียวจะแต่งตั้งตัวเองถาวร
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from core.api import create_app
from core.calendar import ICT, PHASES, CalendarInvalid, as_days, build_phases, day_range
from core.domain import Competition, Course, Phase, new_id
from core.service import build_arena
from core.wiring import course_staff_from_env, env_key_for_course

AJ = "aj@g.swu.ac.th"
TA = "ta@g.swu.ac.th"
STUDENT = "student@g.swu.ac.th"
OTHER = "other-aj@g.swu.ac.th"

SLUG = "cp462-churn-1-2026"
COURSE = "cp462-1-2026"


@pytest.fixture
def arena(tmp_path):
    a = build_arena(
        tmp_path / "artifacts",
        staff_emails=frozenset({AJ}),
        course_staff={COURSE: frozenset({TA})},
    )
    a.store.save_course(Course(id=COURSE, name="CP462", join_code="AAAAAA"))
    a.store.save_course(Course(id="cp463-1-2026", name="CP463", join_code="BBBBBB"))

    now = datetime.now(timezone.utc)
    a.store.save_competition(
        Competition(
            id=new_id(), course_id=COURSE, slug=SLUG, title="churn",
            task_type="prediction", env_plugin="x:PLUGIN", config_path="/dev/null",
            opens_at=now - timedelta(days=1), closes_at=now + timedelta(days=90),
            phases=build_phases({
                "warmup": day_range("2026-09-15", "2026-09-30"),
                "main": day_range("2026-10-01", "2026-10-31"),
                "final": day_range("2026-11-01", "2026-11-30"),
            }),
        )
    )
    return a


def sign_in(arena, email: str):
    return arena.sign_in(email=email, name=email.split("@")[0], google_sub=email)


def auth(user) -> dict:
    return {"Authorization": f"Bearer {user.token}"}


def payload(warmup=("2026-09-08", "2026-09-30"),
            main=("2026-10-01", "2026-10-31"),
            final=("2026-11-01", "2026-11-30")) -> dict:
    return {"phases": json.dumps({"warmup": list(warmup), "main": list(main),
                                  "final": list(final)})}


# ── ใครแก้ได้บ้าง ──────────────────────────────────────────────────


def test_the_course_instructor_can_move_the_calendar(arena):
    client = TestClient(create_app(arena))
    ta = sign_in(arena, TA)
    r = client.post(f"/api/competitions/{SLUG}/calendar", headers=auth(ta), data=payload())
    assert r.status_code == 200, r.text
    assert r.json()["phases"][0]["first_day"] == "2026-09-08"


def test_a_system_wide_instructor_can_manage_any_course(arena):
    """ผู้ดูแลระบบต้องไม่ล็อกตัวเองออกจากวิชาที่ตัวเองไม่ได้สอนแต่ต้องเข้าไปแก้ตอนมีปัญหา"""
    client = TestClient(create_app(arena))
    aj = sign_in(arena, AJ)
    assert client.post(
        f"/api/competitions/{SLUG}/calendar", headers=auth(aj), data=payload()
    ).status_code == 200


def test_a_student_cannot_move_the_deadline(arena):
    """**ด่านสำคัญ** — ซ่อนปุ่มบนหน้าเว็บไม่ใช่การป้องกัน endpoint ยิงตรงได้"""
    client = TestClient(create_app(arena))
    student = sign_in(arena, STUDENT)
    r = client.post(f"/api/competitions/{SLUG}/calendar", headers=auth(student), data=payload())
    assert r.status_code == 403
    assert arena.store.competition_by_slug(SLUG).phases[0].starts_at.astimezone(ICT).day == 15


def test_an_instructor_of_another_course_cannot(arena):
    """ผู้สอนของวิชาอื่นต้องแก้ไม่ได้ — นี่คือสิ่งที่ 'ของวิชานั้นๆ' หมายถึง"""
    arena.course_staff = {**arena.course_staff, "cp463-1-2026": frozenset({OTHER})}
    client = TestClient(create_app(arena))
    other = sign_in(arena, OTHER)
    r = client.post(f"/api/competitions/{SLUG}/calendar", headers=auth(other), data=payload())
    assert r.status_code == 403
    assert COURSE in r.json()["detail"]


def test_course_settings_is_scoped_the_same_way(arena):
    """ตั้งขนาดทีมก็ต้องผูกกับวิชาเดียวกัน — ไม่งั้นสิทธิ์สองแบบในระบบเดียว"""
    arena.course_staff = {**arena.course_staff, "cp463-1-2026": frozenset({OTHER})}
    client = TestClient(create_app(arena))
    other = sign_in(arena, OTHER)
    assert client.post(
        f"/api/courses/{COURSE}/settings", headers=auth(other), data={"size": 3}
    ).status_code == 403
    ta = sign_in(arena, TA)
    assert client.post(
        f"/api/courses/{COURSE}/settings", headers=auth(ta), data={"size": 3}
    ).status_code == 200


def test_me_reports_which_courses_you_can_manage(arena):
    client = TestClient(create_app(arena))
    assert client.get("/api/me", headers=auth(sign_in(arena, TA))).json()["managed_courses"] == [
        COURSE
    ]
    assert client.get("/api/me", headers=auth(sign_in(arena, AJ))).json()["managed_courses"] == [
        COURSE, "cp463-1-2026"
    ]
    assert client.get("/api/me", headers=auth(sign_in(arena, STUDENT))).json()[
        "managed_courses"
    ] == []


# ── กติกาเรื่องวัน ─────────────────────────────────────────────────


def test_the_last_day_is_included(arena):
    client = TestClient(create_app(arena))
    body = client.post(
        f"/api/competitions/{SLUG}/calendar",
        headers=auth(sign_in(arena, TA)),
        data=payload(warmup=("2026-09-15", "2026-09-30")),
    ).json()
    warmup = next(p for p in body["phases"] if p["name"] == "warmup")
    assert warmup["last_day"] == "2026-09-30"

    # เทียบที่ **ขณะเวลา** ไม่ใช่ที่ข้อความ — `+07:00` กับ `Z` เป็นเวลาเดียวกันคนละรูป
    # และ `Phase.contains` ใช้ `start <= when < end` วินาทีสุดท้ายของวันจึงต้องยังอยู่ในช่วง
    ends_at = datetime.fromisoformat(warmup["ends_at"])
    assert ends_at == datetime(2026, 10, 1, 0, 0, tzinfo=ICT)
    assert datetime(2026, 9, 30, 23, 59, 59, tzinfo=ICT) < ends_at, "วันสุดท้ายถูกตัดออกไป"


def test_reading_then_saving_unchanged_does_not_shift_the_calendar(arena):
    """ผู้สอนเปิดฟอร์มแล้วกดบันทึกโดยไม่แก้อะไร ปฏิทินต้องเท่าเดิมเป๊ะ

    เป็นบั๊กคลาสสิกของฟอร์มวันที่ — แปลงไป-กลับไม่ตรงแล้วเลื่อนวันละหนึ่งทุกครั้ง
    """
    client = TestClient(create_app(arena))
    ta = sign_in(arena, TA)
    before = client.get(f"/api/competitions/{SLUG}").json()["phases"]
    same = {p["name"]: [p["first_day"], p["last_day"]] for p in before}

    after = client.post(
        f"/api/competitions/{SLUG}/calendar", headers=auth(ta),
        data={"phases": json.dumps(same)},
    ).json()["phases"]
    assert [(p["starts_at"], p["ends_at"]) for p in after] == [
        (p["starts_at"], p["ends_at"]) for p in before
    ]


@pytest.mark.parametrize(
    "bad,reason",
    [
        ({"warmup": ["2026-09-15", "2026-10-15"]}, "ทับกัน"),
        ({"warmup": ["2026-09-30", "2026-09-15"]}, "มาก่อน"),
        ({"warmup": ["15/09/2026", "2026-09-30"]}, "YYYY-MM-DD"),
    ],
)
def test_bad_calendars_are_rejected_without_changing_anything(arena, bad, reason):
    client = TestClient(create_app(arena))
    ta = sign_in(arena, TA)
    data = json.loads(payload()["phases"])
    data.update(bad)

    r = client.post(
        f"/api/competitions/{SLUG}/calendar", headers=auth(ta),
        data={"phases": json.dumps(data)},
    )
    assert r.status_code == 400, r.text
    assert reason in r.json()["detail"]
    assert as_days(arena.store.competition_by_slug(SLUG).phases[0]) == ("2026-09-15", "2026-09-30")


def test_missing_a_phase_is_rejected(arena):
    client = TestClient(create_app(arena))
    r = client.post(
        f"/api/competitions/{SLUG}/calendar", headers=auth(sign_in(arena, TA)),
        data={"phases": json.dumps({"warmup": ["2026-09-15", "2026-09-30"]})},
    )
    assert r.status_code == 400
    assert "main" in r.json()["detail"]


def test_moving_the_calendar_keeps_each_phase_config_override(arena):
    """ปฏิทินกับ config เป็นคนละเรื่อง — ผู้สอนที่แค่เลื่อนวันต้องไม่เผลอล้างค่าที่
    ทำให้แต่ละ phase ยากต่างกัน (เรื่องของ CP463 แต่กติกาเดียวกันทุกโจทย์)"""
    competition = arena.store.competition_by_slug(SLUG)
    competition.phases[2] = Phase(
        id=competition.phases[2].id, name="final",
        starts_at=competition.phases[2].starts_at, ends_at=competition.phases[2].ends_at,
        config_override={"room.width": 30},
    )
    arena.store.save_competition(competition)

    client = TestClient(create_app(arena))
    client.post(f"/api/competitions/{SLUG}/calendar", headers=auth(sign_in(arena, TA)),
                data=payload())
    after = arena.store.competition_by_slug(SLUG)
    assert after.phases[2].config_override == {"room.width": 30}


def test_the_competition_id_survives_so_old_runs_are_not_orphaned(arena):
    before = arena.store.competition_by_slug(SLUG).id
    client = TestClient(create_app(arena))
    client.post(f"/api/competitions/{SLUG}/calendar", headers=auth(sign_in(arena, TA)),
                data=payload())
    assert arena.store.competition_by_slug(SLUG).id == before


def test_the_change_is_recorded_with_who_did_it(arena):
    client = TestClient(create_app(arena))
    ta = sign_in(arena, TA)
    client.post(f"/api/competitions/{SLUG}/calendar", headers=auth(ta), data=payload())
    events = [e for e in arena.store.audit if e.action == "competition.calendar_changed"]
    assert events and events[-1].actor_id == ta.id


# ── การอ่านรายชื่อจาก environment ──────────────────────────────────


def test_env_key_matches_what_the_reader_expects(monkeypatch):
    """ชื่อตัวแปรที่เอกสารบอกกับที่โค้ดอ่าน ต้องเป็นตัวเดียวกัน

    ผูกสองฝั่งไว้ด้วยกัน — ถ้าใครแก้ข้างเดียว ผู้สอนที่ตั้งค่าตามเอกสารจะไม่ได้สิทธิ์
    โดยไม่มีอะไรฟ้องเลย
    """
    monkeypatch.setenv(env_key_for_course(COURSE), f" {TA.upper()} , x@y.z ")
    assert course_staff_from_env() == {COURSE: frozenset({TA, "x@y.z"})}


def test_an_empty_value_grants_nobody(monkeypatch):
    """ตั้งไว้ว่างต้องไม่ให้สิทธิ์ใคร — ค่าเริ่มต้นที่ปลอดภัย"""
    monkeypatch.setenv(env_key_for_course(COURSE), "   ")
    assert course_staff_from_env() == {}


def test_calendar_module_rejects_a_gap_free_but_reversed_order():
    with pytest.raises(CalendarInvalid, match="ทับกัน"):
        build_phases({
            "warmup": day_range("2026-10-01", "2026-10-31"),
            "main": day_range("2026-09-15", "2026-09-30"),
            "final": day_range("2026-11-01", "2026-11-30"),
        })


def test_a_gap_between_phases_is_allowed():
    """เว้นช่วงสอบกลางภาคได้ — ห้ามแค่ทับกัน"""
    phases = build_phases({
        "warmup": day_range("2026-09-15", "2026-09-30"),
        "main": day_range("2026-10-10", "2026-10-31"),
        "final": day_range("2026-11-01", "2026-11-30"),
    })
    assert [p.name for p in phases] == list(PHASES)


# ── ข้อความตอนเริ่มบริการ ──────────────────────────────────────────


def startup_lines(monkeypatch, tmp_path, env: dict[str, str]) -> list[str]:
    """รัน `arena serve` เท่าที่จำเป็นเพื่ออ่านบรรทัดสรุป — ไม่เปิดพอร์ตจริง"""
    import io
    import contextlib

    from core import cli
    from core.wiring import course_staff_from_env, staff_emails_from_env

    for key in [k for k in os.environ if k.startswith("ARENA_")]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    a = build_arena(tmp_path / "artifacts")
    a.store.save_course(Course(id=COURSE, name="CP462", join_code="AAAAAA"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        a.staff_emails = staff_emails_from_env()
        a.course_staff = course_staff_from_env()
        cli._print_staff(a)
    return buf.getvalue().splitlines()


def test_a_configured_deployment_does_not_warn_that_nobody_is_staff(monkeypatch, tmp_path):
    """**บั๊กที่เคยเกิดจริง** — แทรก `for` ไว้ก่อน `else` แล้วมันกลายเป็น `for...else`

    ผลคือเครื่องที่ตั้ง ARENA_STAFF_EMAILS ถูกต้องแล้วยังขึ้นคำเตือนว่ายังไม่ได้ตั้ง
    ซึ่งชวนให้ไปไล่หาปัญหาผิดที่ · เจอตอนอ่าน journal หลัง restart จริง
    """
    lines = startup_lines(monkeypatch, tmp_path, {"ARENA_STAFF_EMAILS": AJ})
    assert any(AJ in ln for ln in lines), lines
    assert not any("ยังไม่ได้ตั้ง" in ln for ln in lines), lines


def test_an_unconfigured_deployment_still_warns(monkeypatch, tmp_path):
    lines = startup_lines(monkeypatch, tmp_path, {})
    assert any("ยังไม่ได้ตั้ง" in ln for ln in lines), lines


def test_per_course_staff_is_listed(monkeypatch, tmp_path):
    lines = startup_lines(
        monkeypatch, tmp_path,
        {"ARENA_STAFF_EMAILS": AJ, env_key_for_course(COURSE): TA},
    )
    assert any(COURSE in ln and TA in ln for ln in lines), lines


def test_a_typo_in_the_course_id_is_called_out(monkeypatch, tmp_path):
    """ตัวแปรที่สะกดชื่อวิชาผิดจะไม่มีผลกับใครเลย — ต้องบอก ไม่ใช่เงียบ"""
    lines = startup_lines(
        monkeypatch, tmp_path,
        {"ARENA_STAFF_EMAILS": AJ, env_key_for_course("cp999-9-9999"): TA},
    )
    assert any("ไม่มีวิชานี้" in ln for ln in lines), lines


# ── ทะเบียน environment ────────────────────────────────────────────


def test_only_instructors_see_the_environment_catalogue(arena):
    """ไม่ใช่ความลับ แต่เป็นข้อมูลสำหรับคนที่จะสร้างโจทย์ และเปิดเผยชื่อโมดูลบนเครื่อง"""
    client = TestClient(create_app(arena, environments=lambda: [{"env_plugin": "x:P"}]))
    assert client.get("/api/environments", headers=auth(sign_in(arena, TA))).status_code == 200
    assert client.get("/api/environments", headers=auth(sign_in(arena, AJ))).status_code == 200
    assert (
        client.get("/api/environments", headers=auth(sign_in(arena, STUDENT))).status_code == 403
    )


def test_the_catalogue_is_empty_when_nothing_is_wired(arena):
    """ไม่ฉีดเข้ามา = รายการว่าง ไม่ใช่ 500 — `core/api.py` ต้องไม่ import runners เอง"""
    client = TestClient(create_app(arena))
    body = client.get("/api/environments", headers=auth(sign_in(arena, AJ))).json()
    assert body == {"environments": []}


# ── สร้างวิชาและโจทย์จากหน้าเว็บ ───────────────────────────────────


CHURN_YAML = (
    "slug: newtask\ntask: churn\ntitle: โจทย์ใหม่\nkind: classification\n"
    "primary: macro_f1\nlabels: [0, 1]\nn_rows: 12000\n"
    "data_seed: 1\nsplit_seed: 2\nbootstrap_seed: 3\n"
    "ratios: [0.6, 0.15, 0.25]\ngrading_rows: 3000\ngrading_public_ratio: 0.4\n"
)


def fake_prepare(env_plugin, config_text):
    """แทน `core.wiring.prepare_config` — เทสต์นี้ตรวจ endpoint ไม่ใช่ตัวโหลดของ env"""
    if "พัง" in config_text:
        raise ValueError("room.width ต้อง >= 2")
    return {
        "task_type": "prediction",
        "config_hash": "sha256:aa",
        "title": "โจทย์ใหม่",
        "paradigm": "supervised-learning",
        "whitelist": frozenset({"numpy", "pandas"}),
    }


CALENDAR = json.dumps({
    "warmup": ["2026-09-15", "2026-09-30"],
    "main": ["2026-10-01", "2026-10-31"],
    "final": ["2026-11-01", "2026-11-30"],
})


def authoring(arena):
    return TestClient(create_app(arena, prepare_config=fake_prepare))


def new_competition(**kw) -> dict:
    body = {
        "course_id": COURSE, "slug": "cp462-new-1-2026",
        "env_plugin": "tabular.arena:PLUGIN", "config": CHURN_YAML, "phases": CALENDAR,
    }
    body.update(kw)
    return body


def test_only_system_staff_can_create_a_course(arena):
    """ไม่ใช่ `can_manage_course` เพราะยังไม่มีวิชาให้ผูกสิทธิ์ — และถ้าใครก็สร้างได้
    นิสิตจะสร้างวิชาของตัวเองแล้วสั่งงานเข้าคิว"""
    client = authoring(arena)
    form = {"course_id": "cp999-1-2026", "name": "วิชาใหม่", "size": 4}

    assert client.post("/api/courses", headers=auth(sign_in(arena, STUDENT)),
                       data=form).status_code == 403
    assert client.post("/api/courses", headers=auth(sign_in(arena, TA)),
                       data=form).status_code == 403, "ผู้สอนของวิชาอื่นไม่ใช่ผู้ดูแลระบบ"

    got = client.post("/api/courses", headers=auth(sign_in(arena, AJ)), data=form)
    assert got.status_code == 200, got.text
    course = got.json()["course"]
    assert course["id"] == "cp999-1-2026" and course["max_team_size"] == 4
    assert len(course["join_code"]) == 6, "ต้องได้รหัสเข้าวิชาไปแจกในคาบทันที"


@pytest.mark.parametrize("bad", ["cp999_1_2026", "CP 999", "", "วิชา"])
def test_a_bad_course_id_is_rejected_the_same_way_as_the_cli(arena, bad):
    client = authoring(arena)
    r = client.post("/api/courses", headers=auth(sign_in(arena, AJ)),
                    data={"course_id": bad, "name": "x", "size": 4})
    assert r.status_code == 400


def test_creating_a_course_does_not_make_you_its_instructor(arena):
    """สิทธิ์มาจาก environment เท่านั้น — ถ้าคนสร้างกลายเป็นผู้สอนอัตโนมัติ
    ใครที่สร้างวิชาได้จะแต่งตั้งตัวเองได้ ซึ่งคือช่องที่ทั้งการออกแบบพยายามปิด"""
    client = authoring(arena)
    client.post("/api/courses", headers=auth(sign_in(arena, AJ)),
                data={"course_id": "cp999-1-2026", "name": "วิชาใหม่", "size": 4})
    assert "cp999-1-2026" not in arena.course_staff


def test_the_course_instructor_can_create_a_competition(arena):
    client = authoring(arena)
    got = client.post("/api/competitions", headers=auth(sign_in(arena, TA)),
                      data=new_competition())
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["config_hash"] == "sha256:aa"
    assert body["competition"]["task_type"] == "prediction"

    made = arena.store.competition_by_slug("cp462-new-1-2026")
    assert made.config_text == CHURN_YAML, "config ต้องถูกเก็บเป็นเนื้อหา"
    assert made.config_path == "", "ไม่มีไฟล์บนเครื่อง — นี่คือจุดที่ schema v5 ปลดล็อก"
    assert made.config_source() == ("text", CHURN_YAML)
    assert "tabular" not in made.effective_whitelist()


def test_a_student_cannot_create_a_competition(arena):
    client = authoring(arena)
    r = client.post("/api/competitions", headers=auth(sign_in(arena, STUDENT)),
                    data=new_competition())
    assert r.status_code == 403
    assert arena.store.competition_by_slug("cp462-new-1-2026") is None


def test_an_instructor_of_another_course_cannot_create_in_this_one(arena):
    arena.course_staff = {**arena.course_staff, "cp463-1-2026": frozenset({OTHER})}
    client = authoring(arena)
    r = client.post("/api/competitions", headers=auth(sign_in(arena, OTHER)),
                    data=new_competition())
    assert r.status_code == 403


def test_a_config_the_environment_rejects_is_not_stored(arena):
    """**ตรวจด้วยตัวโหลดจริงของ env ก่อนบันทึกเสมอ** — ข้อความที่ผู้สอนเห็นต้องเป็น
    ของ env ไม่ใช่ข้อความกลางๆ ที่ไม่บอกว่าฟิลด์ไหนผิด"""
    client = authoring(arena)
    r = client.post("/api/competitions", headers=auth(sign_in(arena, TA)),
                    data=new_competition(config="พัง"))
    assert r.status_code == 400
    assert "room.width" in r.json()["detail"], "ต้องส่งข้อความของ env ต่อให้ผู้สอน"
    assert arena.store.competition_by_slug("cp462-new-1-2026") is None


def test_a_duplicate_slug_is_refused(arena):
    client = authoring(arena)
    r = client.post("/api/competitions", headers=auth(sign_in(arena, TA)),
                    data=new_competition(slug=SLUG))
    assert r.status_code == 400
    assert SLUG in r.json()["detail"]


def test_a_bad_calendar_is_refused_before_anything_is_written(arena):
    client = authoring(arena)
    overlap = json.dumps({"warmup": ["2026-09-15", "2026-10-15"],
                          "main": ["2026-10-01", "2026-10-31"],
                          "final": ["2026-11-01", "2026-11-30"]})
    r = client.post("/api/competitions", headers=auth(sign_in(arena, TA)),
                    data=new_competition(phases=overlap))
    assert r.status_code == 400 and "ทับกัน" in r.json()["detail"]
    assert arena.store.competition_by_slug("cp462-new-1-2026") is None


def test_creating_without_the_registry_wired_says_so(arena):
    """`core/api.py` ไม่ import runners เอง — ถ้าไม่ได้ฉีดเข้ามาต้องบอกตรงๆ ไม่ใช่ 500"""
    client = TestClient(create_app(arena))
    r = client.post("/api/competitions", headers=auth(sign_in(arena, TA)),
                    data=new_competition())
    assert r.status_code == 503


def test_who_created_it_is_recorded(arena):
    client = authoring(arena)
    ta = sign_in(arena, TA)
    client.post("/api/competitions", headers=auth(ta), data=new_competition())
    events = [e for e in arena.store.audit if e.action == "competition.created"]
    assert events and events[-1].actor_id == ta.id


# ── สิ่งที่เห็นได้ ต้องผูกกับวิชา ไม่ใช่กับ "เป็นผู้สอนคนใดคนหนึ่ง" ────


def with_a_scored_run(arena):
    """ทีมที่ตั้งชื่อบนกระดานไว้ พร้อม run ที่มีคะแนนแล้ว"""
    from core.domain import Run, RunKind, RunStatus, Team

    competition = arena.store.competition_by_slug(SLUG)
    team = Team(id=new_id(), course_id=COURSE, name="ชื่อจริงของทีม",
                alias="นิรนาม", member_ids=[])
    arena.store.save_team(team)
    run = Run(
        id=new_id(), submission_id="s", team_id=team.id, competition_id=competition.id,
        kind=RunKind.PUBLIC, status=RunStatus.DONE, score=0.5,
    )
    arena.queue.runs[run.id] = run
    return team


def names_on_board(arena, email) -> list[str]:
    client = TestClient(create_app(arena))
    body = client.get(f"/api/competitions/{SLUG}/leaderboard",
                      headers=auth(sign_in(arena, email))).json()
    return [r["name"] for r in body["rows"]]


def test_the_course_instructor_sees_real_names_on_their_own_board(arena):
    """ผู้สอนต้องเห็นชื่อจริงของนิสิต**ตัวเอง**เพื่อตัดเกรด — alias มีไว้ลดแรงกดดัน
    ระหว่างนิสิตด้วยกัน ไม่ใช่ซ่อนตัวจากการตรวจ (README §6.1)"""
    with_a_scored_run(arena)
    assert "ชื่อจริงของทีม" in names_on_board(arena, TA)


def test_an_instructor_of_another_course_does_not_see_real_names(arena):
    """**นี่คือสิ่งที่เปลี่ยนจาก `is_staff` มาเป็น `can_manage_course`** — ไม่มีเหตุผล
    ให้ผู้สอนวิชาหนึ่งเห็นชื่อจริงของนิสิตในวิชาที่ตัวเองไม่ได้สอน"""
    arena.course_staff = {**arena.course_staff, "cp463-1-2026": frozenset({OTHER})}
    with_a_scored_run(arena)
    names = names_on_board(arena, OTHER)
    assert "นิรนาม" in names and "ชื่อจริงของทีม" not in names


def test_students_see_only_the_board_names(arena):
    with_a_scored_run(arena)
    names = names_on_board(arena, STUDENT)
    assert "นิรนาม" in names and "ชื่อจริงของทีม" not in names


def test_a_system_wide_instructor_still_sees_real_names(arena):
    with_a_scored_run(arena)
    assert "ชื่อจริงของทีม" in names_on_board(arena, AJ)


def join_code_seen_by(arena, email):
    client = TestClient(create_app(arena))
    user = sign_in(arena, email)
    arena.enroll(user=user, join_code="AAAAAA")
    body = client.get("/api/me", headers=auth(user)).json()
    return next(e["course"]["join_code"] for e in body["enrollments"]
                if e["course"]["id"] == COURSE)


def test_the_course_instructor_gets_the_join_code_to_read_out_in_class(arena):
    assert join_code_seen_by(arena, TA) == "AAAAAA"


def test_a_student_never_gets_the_join_code_back(arena):
    """นิสิตใส่รหัสเข้ามาได้ แต่ต้องไม่ได้รหัสคืนไปแจกต่อ"""
    assert join_code_seen_by(arena, STUDENT) is None


def test_an_instructor_of_another_course_does_not_get_this_courses_join_code(arena):
    arena.course_staff = {**arena.course_staff, "cp463-1-2026": frozenset({OTHER})}
    assert join_code_seen_by(arena, OTHER) is None
