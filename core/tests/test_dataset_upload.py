"""ไฟล์ข้อมูลเดินทางเข้าและออกจากระบบ — **ทางออกมีทางเดียวและมันแคบ**

    POST /api/datasets                    ผู้สอนอัปโหลด CSV → รายชื่อคอลัมน์
    POST /api/competitions/preview        จะแบ่งออกมาเป็นกี่แถวต่อกอง
    GET  /api/competitions/{slug}/data    นิสิตดาวน์โหลด **กองที่แจกเท่านั้น**

สองอย่างที่ผิดแล้วการแข่งจบ

  · **endpoint แจกแถวของกองที่ใช้ตัดสิน** — ทีมที่โหลดไฟล์จะได้เฉลยของสิ่งที่
    ใช้ตัดเกรดไปเลย · เทสต์ข้อสุดท้ายเทียบแถวจริงทีละแถว ไม่ใช่แค่ดูจำนวน
  · **ใครก็อัปโหลดทับได้** — ไฟล์ที่อัปโหลดกลายเป็นเฉลยของโจทย์ทั้งวิชา

และอีกอย่างที่ผิดแล้วน่ารำคาญมากกว่าอันตราย: คลังที่เต็มไปด้วยไฟล์ของโจทย์ที่
ไม่เคยถูกสร้าง เพราะหน้าเว็บเรียกอัปโหลดทุกครั้งที่ผู้สอนเลือกไฟล์
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timedelta, timezone

import pytest
import yaml
from fastapi.testclient import TestClient

from core.api import create_app
from core.calendar import build_phases, day_range
from core.domain import Competition, Course, new_id, proper_title, slugify
from core.service import build_arena

pytest.importorskip("tabular", reason="ต้องติดตั้ง envs/cp462-tabular ก่อน")

from core.wiring import preview_config, student_dataset, upload_dataset  # noqa: E402

AJ = "aj@g.swu.ac.th"
STUDENT = "student@g.swu.ac.th"
OUTSIDER = "outsider@g.swu.ac.th"

COURSE = "cp462-1-2026"
SLUG = "cp462-churn-1-2026"
ENV = "tabular.arena:PLUGIN"


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    from tabular import store

    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setenv(store.DATASETS_ENV, str(root))
    return root


@pytest.fixture
def csv_bytes():
    from tabular.generator import sample_csv

    return sample_csv("churn", seed=20260101, n=2000)


def config_for(digest: str) -> str:
    return yaml.safe_dump({
        "title": "Churn", "kind": "classification", "primary": "macro_f1",
        "dataset": digest, "target": "churned", "drop": ["account_id"],
        "labels": [0, 1], "split_seed": 7, "bootstrap_seed": 11,
    }, allow_unicode=True, sort_keys=True)


@pytest.fixture
def arena(tmp_path):
    a = build_arena(tmp_path / "artifacts", course_staff={COURSE: frozenset({AJ})})
    a.store.save_course(Course(id=COURSE, name="CP462", join_code="AAAAAA"))
    return a


@pytest.fixture
def client(arena):
    return TestClient(create_app(
        arena,
        upload_dataset=upload_dataset,
        preview_config=preview_config,
        student_dataset=student_dataset,
    ))


def sign_in(arena, email: str):
    return arena.sign_in(email=email, name=email.split("@")[0], google_sub=email)


def auth(user) -> dict:
    return {"Authorization": f"Bearer {user.token}"}


def upload(client, user, blob: bytes, *, course: str = COURSE):
    return client.post(
        "/api/datasets",
        headers=auth(user),
        files={"file": ("data.csv", io.BytesIO(blob), "text/csv")},
        data={"course_id": course, "env_plugin": ENV},
    )


def add_competition(arena, config_text: str):
    now = datetime.now(timezone.utc)
    arena.store.save_competition(Competition(
        id=new_id(), course_id=COURSE, slug=SLUG, title="Churn",
        task_type="prediction", env_plugin=ENV,
        config_text=config_text, config_path="",
        opens_at=now - timedelta(days=1), closes_at=now + timedelta(days=90),
        phases=build_phases({
            "warmup": day_range("2026-09-15", "2026-09-30"),
            "main": day_range("2026-10-01", "2026-10-31"),
            "final": day_range("2026-11-01", "2026-11-30"),
        }),
    ))


# ── ชื่อที่ผู้สอนตั้ง กับชื่อที่นิสิตเห็น ──────────────────────────


@pytest.mark.parametrize("raw, expected", [
    ("diabetes-screening", "Diabetes Screening"),
    ("prediction_of_house_prices", "Prediction of House Prices"),
    ("churn", "Churn"),
    ("risk of stroke", "Risk of Stroke"),
    ("  spaced   out  ", "Spaced Out"),
    # ตัวย่อกับคำที่มีตัวใหญ่อยู่กลาง — **ห้ามแก้ให้** เพราะเป็นเจตนาของคนตั้ง
    ("ML model for HR data", "ML Model for HR Data"),
    ("eCommerce sales", "eCommerce Sales"),
    # บุพบทเป็นตัวเล็ก ยกเว้นเมื่อเป็นคำแรกหรือคำสุดท้าย
    ("the end of the road", "The End of the Road"),
    ("to be or not to be", "To Be or Not to Be"),
    # ภาษาไทยไม่มีตัวใหญ่ตัวเล็ก — ต้องไม่ถูกแตะเลย
    ("ทำนายการเลิกใช้บริการ", "ทำนายการเลิกใช้บริการ"),
    ("", ""),
])
def test_proper_title(raw, expected):
    assert proper_title(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Diabetes Screening", "diabetes-screening"),
    ("ราคาบ้าน", ""),
    ("A/B test!", "a-b-test"),
    ("--already--slugged--", "already-slugged"),
])
def test_slugify(raw, expected):
    assert slugify(raw) == expected


def test_the_web_page_uses_the_same_small_words_as_the_server():
    """หน้าเว็บแสดงชื่อที่จะได้ก่อนกดบันทึก — **สองที่ต้องตัดสินเหมือนกัน**

    ถ้าต่างกัน ผู้สอนจะเห็นชื่อหนึ่งบนฟอร์มแล้วได้อีกชื่อหนึ่งบนหน้าโจทย์ ซึ่ง
    เป็นความต่างที่ไม่มีอะไรฟ้องนอกจากคนสังเกตเอง
    """
    import re
    from pathlib import Path

    from core.domain import TITLE_SMALL_WORDS

    index = (Path(__file__).resolve().parents[2] / "web" / "index.html").read_text("utf-8")
    match = re.search(r"SMALL_WORDS\s*=\s*new Set\(\s*'([^']*)'\.split\(' '\)\s*\)", index)
    assert match, "หน้าเว็บต้องประกาศ SMALL_WORDS ในรูปที่เทสต์นี้อ่านได้"
    assert set(match.group(1).split()) == set(TITLE_SMALL_WORDS)


# ── ใครอัปโหลดได้ ─────────────────────────────────────────────────


def test_only_the_course_instructor_can_upload(arena, client, csv_bytes, store_root):
    """ไฟล์ที่อัปโหลดกลายเป็นเฉลยของโจทย์ — นิสิตอัปโหลดทับได้แปลว่าตั้งเฉลยเองได้"""
    student = sign_in(arena, STUDENT)
    assert upload(client, student, csv_bytes).status_code == 403

    aj = sign_in(arena, AJ)
    assert upload(client, aj, csv_bytes).status_code == 200


def test_an_instructor_of_another_course_cannot_upload(arena, client, csv_bytes, store_root):
    other = sign_in(arena, "aj-cp463@g.swu.ac.th")
    assert upload(client, other, csv_bytes).status_code == 403


def test_upload_without_the_env_wired_says_so(arena, csv_bytes):
    """deployment ที่ไม่มี env รับไฟล์ต้องตอบ 503 ไม่ใช่ 500"""
    bare = TestClient(create_app(arena))
    aj = sign_in(arena, AJ)
    assert upload(bare, aj, csv_bytes).status_code == 503


# ── สิ่งที่ผู้สอนได้เห็นหลังอัปโหลด ────────────────────────────────


def test_the_upload_answers_with_the_columns_so_nobody_types_them(arena, client, csv_bytes,
                                                                  store_root):
    """ผู้สอนเลือกคอลัมน์เฉลยจากรายการ ไม่ใช่พิมพ์จากความจำ

    ฟอร์มที่ให้พิมพ์ชื่อคอลัมน์เองคือฟอร์มที่พิมพ์ผิดได้ แล้วความผิดจะไปโผล่
    ตอนนิสิตส่งงานเข้ามาแล้ว
    """
    aj = sign_in(arena, AJ)
    body = upload(client, aj, csv_bytes).json()

    assert body["rows"] == 2000
    names = [c["name"] for c in body["columns"]]
    assert names == ["account_id", "tenure_months", "monthly_spend",
                     "support_tickets", "plan", "region", "churned"]

    churned = next(c for c in body["columns"] if c["name"] == "churned")
    assert churned["values"] == [0, 1], "หน้าเว็บใช้ค่านี้เติม labels ให้"
    account_id = next(c for c in body["columns"] if c["name"] == "account_id")
    assert "values" not in account_id, "คอลัมน์ที่ไม่ซ้ำเลยต้องไม่ถูกเสนอเป็นคลาส"


def test_a_file_that_cannot_be_used_never_reaches_the_store(arena, client, store_root):
    """**ตรวจก่อนเก็บ** — คลังที่มีไฟล์ซึ่งอ่านไม่ได้ แปลว่ามีโจทย์ที่ให้คะแนนไม่ได้

    ซึ่งจะไปโผล่ตอนนิสิตส่งงานเข้ามาแล้ว ไม่ใช่ตอนที่ผู้สอนยังแก้ได้
    """
    aj = sign_in(arena, AJ)
    assert upload(client, aj, b"a,a,y\n" + b"1,2,0\n" * 400).status_code == 400
    assert list(store_root.glob("*.csv")) == [], "ไฟล์ที่ถูกปฏิเสธไม่ควรอยู่ในคลัง"


def test_uploading_the_same_file_twice_does_not_duplicate_it(arena, client, csv_bytes,
                                                             store_root):
    """อ้างด้วยลายนิ้วมือของเนื้อไฟล์ — ผู้สอนที่เลือกไฟล์เดิมซ้ำไม่ทำให้คลังโต"""
    aj = sign_in(arena, AJ)
    first = upload(client, aj, csv_bytes).json()["digest"]
    second = upload(client, aj, csv_bytes).json()["digest"]

    assert first == second and first.startswith("sha256:")
    assert len(list(store_root.glob("*.csv"))) == 1


def test_a_file_the_system_cannot_use_is_refused_with_a_reason(arena, client, store_root):
    aj = sign_in(arena, AJ)
    reply = upload(client, aj, b"a,b,y\n" + b"1,2,0\n" * 20)
    assert reply.status_code == 400
    assert "น้อยกว่าขั้นต่ำ" in reply.json()["detail"]


# ── ตัวอย่างการแบ่ง — ข้อ 7 ของผู้สอน ──────────────────────────────


def test_the_preview_reports_rows_and_class_balance_per_part(arena, client, csv_bytes,
                                                             store_root):
    """**สัดส่วน 0.15 ไม่ได้บอกอะไร — จำนวนแถวกับการกระจายคลาสบอก**"""
    aj = sign_in(arena, AJ)
    digest = upload(client, aj, csv_bytes).json()["digest"]

    reply = client.post("/api/competitions/preview", headers=auth(aj),
                        data={"course_id": COURSE, "env_plugin": ENV,
                              "config": config_for(digest)})
    assert reply.status_code == 200, reply.text
    body = reply.json()

    assert body["sizes"] == {"student": 1400, "test_public": 300, "test_private": 300}
    # สัดส่วนคลาสต้องใกล้เคียงกันทุกกอง — นั่นคือเหตุผลที่แบ่งแบบ stratified
    shares = {
        part: dist["1"] / sum(dist.values()) for part, dist in body["classes"].items()
    }
    assert max(shares.values()) - min(shares.values()) < 0.01, shares


def test_a_student_cannot_ask_how_the_grading_set_looks(arena, client, csv_bytes, store_root):
    """การกระจายคลาสของกองลับเป็นข้อมูลที่ใช้เดาเฉลยได้เมื่อโจทย์ไม่สมดุล"""
    aj = sign_in(arena, AJ)
    digest = upload(client, aj, csv_bytes).json()["digest"]
    student = sign_in(arena, STUDENT)

    reply = client.post("/api/competitions/preview", headers=auth(student),
                        data={"course_id": COURSE, "env_plugin": ENV,
                              "config": config_for(digest)})
    assert reply.status_code == 403


# ── 🔒 ทางออกของข้อมูล ────────────────────────────────────────────


def test_a_member_of_the_course_can_download_the_handed_out_part(arena, client, csv_bytes,
                                                                 store_root):
    aj = sign_in(arena, AJ)
    digest = upload(client, aj, csv_bytes).json()["digest"]
    add_competition(arena, config_for(digest))

    student = sign_in(arena, STUDENT)
    arena.enroll(user=student, join_code="AAAAAA")

    reply = client.get(f"/api/competitions/{SLUG}/data", headers=auth(student))
    assert reply.status_code == 200, reply.text
    assert reply.headers["content-type"].startswith("text/csv")
    assert SLUG in reply.headers["content-disposition"]

    lines = reply.text.splitlines()
    assert len(lines) - 1 == 1400
    assert lines[0].split(",") == ["tenure_months", "monthly_spend", "support_tickets",
                                   "plan", "region", "churned"]


def test_someone_outside_the_course_cannot_download(arena, client, csv_bytes, store_root):
    aj = sign_in(arena, AJ)
    digest = upload(client, aj, csv_bytes).json()["digest"]
    add_competition(arena, config_for(digest))

    outsider = sign_in(arena, OUTSIDER)
    reply = client.get(f"/api/competitions/{SLUG}/data", headers=auth(outsider))
    assert reply.status_code == 403


def test_the_instructor_can_download_without_joining_as_a_student(arena, client, csv_bytes,
                                                                  store_root):
    """ผู้สอนต้องดูไฟล์ที่นิสิตได้รับได้ โดยไม่ต้องเข้าวิชาในฐานะนิสิต"""
    aj = sign_in(arena, AJ)
    digest = upload(client, aj, csv_bytes).json()["digest"]
    add_competition(arena, config_for(digest))

    assert client.get(f"/api/competitions/{SLUG}/data", headers=auth(aj)).status_code == 200


def test_the_download_contains_no_row_from_the_grading_set(arena, client, csv_bytes,
                                                           store_root):
    """**ข้อที่พังแล้วการแข่งจบ** — เทียบทีละแถว ไม่ใช่แค่ดูจำนวน

    เคยพังจริงในรูปแบบก่อนหน้า: ทั้งสองชุดสร้างจากเมล็ดที่อยู่ในไฟล์ที่แจก
    ทำให้คำนวณเฉลยได้ครบทุกแถว · ข้อนี้ยืนหน้า endpoint จริง ไม่ใช่หน้าไลบรารี
    เพราะเส้นทางที่นิสิตใช้จริงคือเส้นนี้
    """
    import pandas as pd
    from tabular.arena import PLUGIN
    from tabular.config import from_mapping

    aj = sign_in(arena, AJ)
    digest = upload(client, aj, csv_bytes).json()["digest"]
    config_text = config_for(digest)
    add_competition(arena, config_text)

    student = sign_in(arena, STUDENT)
    arena.enroll(user=student, join_code="AAAAAA")
    handed_out = pd.read_csv(io.StringIO(
        client.get(f"/api/competitions/{SLUG}/data", headers=auth(student)).text
    ))

    spec = from_mapping(yaml.safe_load(config_text))
    features = [c for c in handed_out.columns if c != spec.target]
    seen = set(map(tuple, handed_out[features].astype(str).itertuples(index=False)))

    for kind in ("public", "private"):
        graded = PLUGIN.grading_data(spec, kind)
        secret = set(map(tuple, graded.X[features].astype(str).itertuples(index=False)))
        assert not (seen & secret), f"{kind}: {len(seen & secret)} แถวหลุดไปกับไฟล์ที่แจก"


def test_downloading_a_competition_that_does_not_exist_is_a_404(arena, client):
    aj = sign_in(arena, AJ)
    assert client.get("/api/competitions/ไม่มีอันนี้/data", headers=auth(aj)).status_code == 404
