"""config ของโจทย์อยู่ในฐานข้อมูล ไม่ใช่ path บนเครื่อง (schema v5)

เดิม `Competition` เก็บแค่ path ซึ่งแปลว่ามันไม่ใช่ข้อมูลที่สมบูรณ์ในตัวเอง
แต่เป็น**ตัวชี้ไปยังไฟล์บนเครื่องหนึ่งเครื่อง** ผลตามมาสามข้อ

  · สร้าง competition จากหน้าเว็บไม่ได้ ต้อง ssh ไปวางไฟล์ก่อน
  · ย้ายเครื่องหรือกู้จากสำรอง แล้ว competition พังถ้าลืมยกไฟล์ไปด้วย
  · ไฟล์ถูกแก้ทีหลังโดยไม่มีร่องรอย แล้ว run เก่ากับใหม่ใช้ config คนละอัน

⚠️ **สิ่งที่อันตรายที่สุดของการย้ายนี้** คือการมีที่มาสองทาง — ถ้าโค้ดคนละที่
ตัดสินคนละแบบว่าจะใช้ `config_text` หรือ `config_path` จะมี run ที่ให้คะแนนด้วย
config คนละอันโดยไม่มีใครรู้ · จึงต้องมีจุดตัดสินจุดเดียว (`config_source()`)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.db import SCHEMA_VERSION, Database
from core.domain import Competition, new_id

YAML = "task: demo\nversion: 1.0.0\n"


def competition(**kw) -> Competition:
    now = datetime.now(timezone.utc)
    args = dict(
        id=new_id(), course_id="c", slug="s", title="t", task_type="prediction",
        env_plugin="x:PLUGIN", config_path="/ไม่มีจริง/config.yaml",
        opens_at=now, closes_at=now + timedelta(days=1),
    )
    args.update(kw)
    return Competition(**args)


# ── จุดตัดสินที่มาของ config ────────────────────────────────────────


def test_content_wins_when_present():
    assert competition(config_text=YAML).config_source() == ("text", YAML)


def test_path_is_the_fallback_for_records_made_before_v5():
    assert competition().config_source() == ("path", "/ไม่มีจริง/config.yaml")


@pytest.mark.parametrize("blank", ["", "   ", "\n\n"])
def test_blank_content_is_not_content(blank):
    """ค่าว่างที่มีแต่ช่องว่างต้องไม่ถูกนับว่าเป็น config — YAML ว่างเปล่าจะ parse
    ได้เป็น `None` แล้วไปพังลึกกว่านั้นในที่ที่หาต้นเหตุยาก"""
    assert competition(config_text=blank).config_source()[0] == "path"


def test_nobody_reads_the_two_fields_directly():
    """**ต้องมีจุดตัดสินจุดเดียว** — สองที่ตัดสินคนละแบบเมื่อไร จะมี run ที่ให้คะแนน
    ด้วย config คนละอันโดยไม่มีใครรู้

    ตรวจแบบ static: นอกจาก `config_source()` เองแล้ว ห้ามมีใครเขียน
    `config_text or config_path` หรือเช็ค `.config_text` เพื่อเลือกทาง
    """
    repo = Path(__file__).resolve().parent.parent.parent
    offenders = []
    for path in list((repo / "core").rglob("*.py")) + list((repo / "runners").rglob("*.py")):
        if path.name in {"domain.py", "db.py"} or "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "config_text or" in text or ".config_text and" in text:
            offenders.append(str(path.relative_to(repo)))
    assert not offenders, f"เลือกที่มาของ config เองแทนที่จะใช้ config_source(): {offenders}"


# ── migration ──────────────────────────────────────────────────────


def make_v4_db(path: Path, config_path: str) -> None:
    """สร้างฐานข้อมูล schema v4 ด้วยมือ — ไม่พึ่งโค้ดปัจจุบันซึ่ง migrate ให้อัตโนมัติ"""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta VALUES('schema_version', '4');
        CREATE TABLE competitions(
            id TEXT PRIMARY KEY, course_id TEXT NOT NULL, slug TEXT NOT NULL,
            title TEXT NOT NULL, task_type TEXT NOT NULL, env_plugin TEXT NOT NULL,
            config_path TEXT NOT NULL, opens_at TEXT NOT NULL, closes_at TEXT NOT NULL,
            quota_per_day INTEGER NOT NULL DEFAULT 5,
            max_final_submissions INTEGER NOT NULL DEFAULT 2,
            phases TEXT NOT NULL DEFAULT '[]',
            import_whitelist TEXT NOT NULL DEFAULT '[]',
            paradigm TEXT NOT NULL DEFAULT 'reinforcement-learning');
        """
    )
    conn.execute(
        "INSERT INTO competitions(id, course_id, slug, title, task_type, env_plugin,"
        " config_path, opens_at, closes_at) VALUES(?,?,?,?,?,?,?,?,?)",
        ("cid", "c", "s", "t", "prediction", "x:PLUGIN", config_path,
         "2026-01-01T00:00:00+00:00", "2026-12-31T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()


def test_migration_pulls_the_file_content_in(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text(YAML, encoding="utf-8")
    db_path = tmp_path / "arena.db"
    make_v4_db(db_path, str(config))

    db = Database(db_path)
    try:
        loaded = db.load_competitions()["cid"]
        assert loaded.config_text == YAML
        assert loaded.config_source() == ("text", YAML)
    finally:
        db.close()


def test_migration_survives_a_config_file_that_is_not_on_this_machine(tmp_path):
    """**ต้องไม่ล้ม** — ฐานข้อมูลที่กู้มาจากเครื่องอื่นจะชี้ไปยัง path ที่ไม่มีอยู่

    migration ที่ล้มแปลว่าบริการเริ่มไม่ได้เลย ซึ่งแย่กว่าการปล่อยให้ competition
    นั้นทำงานเหมือนเดิม (คือถอยไปใช้ path แล้วพังตอนรัน เท่าที่มันพังอยู่แล้ว)
    """
    db_path = tmp_path / "arena.db"
    make_v4_db(db_path, "/ไม่มีอยู่บนเครื่องนี้/main.yaml")

    db = Database(db_path)
    try:
        loaded = db.load_competitions()["cid"]
        assert loaded.config_text == ""
        assert loaded.config_source() == ("path", "/ไม่มีอยู่บนเครื่องนี้/main.yaml")
    finally:
        db.close()


def test_migration_reaches_the_current_version(tmp_path):
    config = tmp_path / "main.yaml"
    config.write_text(YAML, encoding="utf-8")
    db_path = tmp_path / "arena.db"
    make_v4_db(db_path, str(config))

    db = Database(db_path)
    try:
        conn = sqlite3.connect(db_path)
        version = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        conn.close()
        assert int(version) == SCHEMA_VERSION
    finally:
        db.close()


def test_content_survives_a_round_trip_through_the_database(tmp_path):
    db = Database(tmp_path / "arena.db")
    try:
        text = "task: demo\n# ความคิดเห็นภาษาไทย\nroom:\n  width: 20\n"
        db.save_competition(competition(config_text=text))
        assert next(iter(db.load_competitions().values())).config_text == text
    finally:
        db.close()


# ── worker เขียนเนื้อหาลงไฟล์ให้ runner ────────────────────────────


def test_the_worker_writes_the_content_to_a_file_for_the_runner(tmp_path):
    """สัญญาของ plugin รับ *path* — worker จึงต้องแปลงเนื้อหาเป็นไฟล์ให้

    เปลี่ยนสัญญาให้รับเนื้อหาแทนจะกระทบทุก environment ที่มีอยู่และที่จะมี
    โดยไม่ได้อะไรเพิ่ม
    """
    from runners.worker import Worker

    worker = Worker(runner_id="t", store=None, queue=None, artifacts=None, workdir=tmp_path)
    path = Path(worker._config_file(competition(config_text=YAML), tmp_path / "run"))
    assert path.read_text(encoding="utf-8") == YAML
    assert path.suffix == ".yaml", "loader บางตัวอ่านนามสกุลเพื่อเลือกวิธี parse"


def test_the_worker_still_honours_a_legacy_path(tmp_path):
    from runners.worker import Worker

    worker = Worker(runner_id="t", store=None, queue=None, artifacts=None, workdir=tmp_path)
    got = worker._config_file(competition(config_path="/เดิม/main.yaml"), tmp_path / "run")
    assert got == "/เดิม/main.yaml"


def test_the_temp_config_lands_inside_the_run_workdir(tmp_path):
    """ต้องอยู่ใน workdir ของ run เพราะมันถูกลบพร้อมกันตอนจบ — ไม่ทิ้งขยะไว้"""
    from runners.worker import Worker

    worker = Worker(runner_id="t", store=None, queue=None, artifacts=None, workdir=tmp_path)
    run_dir = tmp_path / "run-123"
    path = Path(worker._config_file(competition(config_text=YAML), run_dir))
    assert run_dir in path.parents
