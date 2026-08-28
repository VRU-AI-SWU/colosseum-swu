"""ที่เก็บถาวรด้วย SQLite — README §11

**รูปแบบ: write-through ไม่ใช่ query layer**

    เขียน  →  แก้ object ในหน่วยความจำ แล้ว `db.save_*(obj)` ทันที
    อ่าน   →  จาก dict ในหน่วยความจำเหมือนเดิม ไม่แตะ SQLite เลย
    เริ่ม  →  `build_arena(db_path=...)` ดึงทุกแถวกลับเข้า dict ด้วย `load_*()`

เลือกแบบนี้เพราะตรรกะที่ยากที่สุดของระบบ — fair-share, lease, การนับโควตา — อยู่ใน
`core/queue.py` และมันถูกทดสอบไว้แล้วในรูปของการทำงานกับ dict การเปลี่ยนไปเป็น
query layer แปลว่าต้องเขียนตรรกะนั้นใหม่เป็น SQL แล้วพิสูจน์ใหม่ทั้งหมด
ซึ่งเป็นความเสี่ยงที่ไม่คุ้มกับปัญหาที่กำลังแก้ ("รีสตาร์ทแล้วคะแนนหาย")

**ข้อจำกัดที่ต้องรู้**: working set อยู่ในหน่วยความจำ จึงรองรับ **process เดียว**
worker ที่รันคนละ process จะไม่เห็นการแก้ของอีกฝั่ง — ตรงกับสถาปัตยกรรมที่ออกแบบไว้
([README §10.1](../README.md#101-ภาพรวม-hybrid-web-บน-cloud--runner-ในมหาวิทยาลัย))
ที่ runner ต่อเข้ามาทาง WebSocket ไม่ใช่แชร์ฐานข้อมูลกัน วันที่ต้องมีหลาย process
จริงๆ คือวันที่ย้ายไป Postgres ซึ่งตอนนั้นค่อยเขียน query layer

⚠️ **ไฟล์นี้ไม่เคยเห็นค่า seed** — เหมือน `core/api.py` seed อยู่ฝั่ง runner เท่านั้น
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from core.domain import (
    DEFAULT_MAX_TEAM_SIZE,
    AuditEvent,
    Competition,
    Course,
    EpisodeResult,
    Phase,
    Run,
    RunKind,
    RunStatus,
    Submission,
    Team,
    User,
    new_invite_code,
)

SCHEMA_VERSION = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS courses (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    max_team_size INTEGER NOT NULL,
    join_code     TEXT NOT NULL DEFAULT '',
    archived_at   TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    id           TEXT PRIMARY KEY,
    course_id    TEXT NOT NULL,
    name         TEXT NOT NULL,
    alias        TEXT,
    member_ids   TEXT NOT NULL,
    token        TEXT NOT NULL UNIQUE,
    invite_code  TEXT NOT NULL,
    dissolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_teams_token ON teams(token);
CREATE INDEX IF NOT EXISTS idx_teams_invite ON teams(invite_code);

CREATE TABLE IF NOT EXISTS users (
    id         TEXT PRIMARY KEY,
    email      TEXT NOT NULL,
    name       TEXT NOT NULL,
    -- รหัสถาวรจาก Google · ใช้จับคู่แทนอีเมลเพราะอีเมลเปลี่ยนได้ sub ไม่เปลี่ยน
    google_sub TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    token      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_users_sub ON users(google_sub);

CREATE TABLE IF NOT EXISTS competitions (
    id                     TEXT PRIMARY KEY,
    course_id              TEXT NOT NULL,
    slug                   TEXT NOT NULL UNIQUE,
    title                  TEXT NOT NULL,
    task_type              TEXT NOT NULL,
    env_plugin             TEXT NOT NULL,
    config_path            TEXT NOT NULL,
    opens_at               TEXT NOT NULL,
    closes_at              TEXT NOT NULL,
    quota_per_day          INTEGER NOT NULL,
    max_final_submissions  INTEGER NOT NULL,
    phases                 TEXT NOT NULL,
    import_whitelist       TEXT NOT NULL,
    paradigm               TEXT NOT NULL DEFAULT 'reinforcement-learning'
);

CREATE TABLE IF NOT EXISTS submissions (
    id              TEXT PRIMARY KEY,
    competition_id  TEXT NOT NULL,
    team_id         TEXT NOT NULL,
    submitted_by    TEXT NOT NULL,
    artifact_url    TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    note            TEXT NOT NULL,
    is_final_pick   INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_submissions_team
    ON submissions(team_id, competition_id);

CREATE TABLE IF NOT EXISTS runs (
    id               TEXT PRIMARY KEY,
    submission_id    TEXT NOT NULL,
    competition_id   TEXT NOT NULL,
    team_id          TEXT NOT NULL,
    kind             TEXT NOT NULL,
    status           TEXT NOT NULL,
    lane             TEXT NOT NULL,
    config_hash      TEXT,
    env_version      TEXT,
    score            REAL,
    tiebreak         TEXT NOT NULL,
    metrics          TEXT NOT NULL,
    episodes         TEXT NOT NULL,
    error_message    TEXT,
    attempts         INTEGER NOT NULL,
    runner_id        TEXT,
    lease_expires_at TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_competition ON runs(competition_id, status);
CREATE INDEX IF NOT EXISTS idx_runs_team ON runs(team_id, competition_id);

CREATE TABLE IF NOT EXISTS audit (
    id          TEXT PRIMARY KEY,
    actor_id    TEXT,
    action      TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_target ON audit(target_id);

-- ตัวนับ fair-share ของคิว · ต้องอยู่รอดข้ามการรีสตาร์ท ไม่งั้นทีมที่เพิ่งถูกเสิร์ฟ
-- ไป 5 งานจะกลับมาเท่ากับทีมที่ยังไม่เคยได้คิวเลย
CREATE TABLE IF NOT EXISTS queue_served (
    team_id TEXT PRIMARY KEY,
    count   INTEGER NOT NULL
);
"""


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SchemaMismatch(RuntimeError):
    """ไฟล์ฐานข้อมูลมาจากคนละเวอร์ชันของ schema และไม่มี migration ให้ — ห้ามเดา"""


def _migrate_1_to_2(conn: sqlite3.Connection) -> None:
    """v1 → v2 · เพิ่มตาราง users และแยก token ของทีมออกจาก id

    **โทเคนเดิมถูกเก็บไว้เป็น `token = id`** ไม่ได้สุ่มใหม่ — ทีมที่มีอยู่ตั้งค่า
    `ARENA_TOKEN` ไว้แล้ว การสุ่มใหม่เงียบๆ จะทำให้ `arena submit` ของเขาพังโดย
    ไม่มีคำอธิบาย · ทีมที่สร้างใหม่หลังจากนี้ได้โทเคนสุ่มเสมอ

    ⚠️ แปลว่าทีมเดิมยังมีโทเคนที่เดาได้อยู่ — ต้องลบทีม demo ทิ้งก่อนเปิดให้นิสิตใช้
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(teams)")}
    if "token" not in cols:
        conn.execute("ALTER TABLE teams ADD COLUMN token TEXT NOT NULL DEFAULT ''")
        conn.execute("ALTER TABLE teams ADD COLUMN invite_code TEXT NOT NULL DEFAULT ''")
        conn.execute("ALTER TABLE teams ADD COLUMN dissolved_at TEXT")
    for row in conn.execute("SELECT id, token, invite_code FROM teams").fetchall():
        conn.execute(
            "UPDATE teams SET token = ?, invite_code = ? WHERE id = ?",
            (row["token"] or row["id"], row["invite_code"] or new_invite_code(), row["id"]),
        )


def _migrate_2_to_3(conn: sqlite3.Connection) -> None:
    """v2 → v3 · เพิ่มตาราง courses เพื่อให้ขนาดทีมเป็นข้อมูล ไม่ใช่ค่าคงที่ในโค้ด

    **สร้างแถวให้ทุก course_id ที่ competition อ้างถึงอยู่แล้ว** ด้วยขนาดทีมเท่าเดิม
    (`DEFAULT_MAX_TEAM_SIZE`) — ถ้าไม่ทำ ระบบที่เคยรันอยู่จะตื่นมาแล้วหาวิชาไม่เจอ
    แล้วปฏิเสธการเข้าทีมทั้งหมด ซึ่งเป็นการเปลี่ยนพฤติกรรมที่ไม่มีใครขอ

    ชื่อวิชาตั้งจาก `course_id` ไปก่อน เพราะ v2 ไม่มีที่ให้เก็บชื่อ · ผู้สอนแก้ทีหลังได้
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS courses (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            max_team_size INTEGER NOT NULL
        )
        """
    )
    # **ห้ามสมมติว่าตารางอื่นมีอยู่** — ไฟล์ v1 มีแค่บางตาราง และ migration ที่ล้ม
    # กลางทางจะทิ้งฐานข้อมูลไว้ครึ่งๆ กลางๆ ซึ่งแย่กว่าการไม่ migrate เลย
    present = {
        r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    known = {r["id"] for r in conn.execute("SELECT id FROM courses")}
    seen: set[str] = set()
    for table in ("competitions", "teams"):
        if table in present:
            seen |= {
                r["course_id"]
                for r in conn.execute(f"SELECT DISTINCT course_id FROM {table}")
            }
    for course_id in sorted(seen - known):
        conn.execute(
            "INSERT INTO courses(id, name, max_team_size) VALUES(?, ?, ?)",
            (course_id, course_id, DEFAULT_MAX_TEAM_SIZE),
        )


def _migrate_3_to_4(conn: sqlite3.Connection) -> None:
    """v3 → v4 · รองรับหลายวิชา — โทเคนย้ายจากทีมมาที่คน + รหัสเข้าวิชา + paradigm

    **โทเคนของคนสุ่มใหม่ทั้งหมด ไม่ได้ย้ายค่าเดิมมา** ต่างจากตอน v1→v2 ที่ตั้งใจ
    รักษาโทเคนเดิมไว้ · ที่นี่ทำไม่ได้เพราะความหมายเปลี่ยน: หนึ่งทีมมีได้หลายคน
    การหยิบโทเคนของทีมมาให้คนใดคนหนึ่งจะทำให้อีกคนได้โทเคนที่เคยเป็นของทีม
    ซึ่งอ่านเหมือนยังใช้ร่วมกันอยู่ · ทุกคนต้องเข้าหน้าเว็บไปเอาโทเคนใหม่

    ยอมรับได้เพราะตอน migrate ยังไม่มีนิสิตคนไหนถือโทเคน — ต้นทุนเป็นศูนย์วันนี้
    และจะไม่ถูกกว่านี้อีก

    `teams.token` ยังอยู่ในตารางเพื่อไม่ให้ข้อมูลเก่าหาย แต่ไม่มีใครอ่านมันแล้ว
    """
    from core.domain import new_invite_code, new_token

    # **ห้ามสมมติว่าตารางอื่นมีอยู่** — ไฟล์ v1 มีแค่ `teams` · พลาดข้อนี้มาแล้ว
    # ตอน v2→v3 แล้วพลาดซ้ำที่นี่ ซึ่งแปลว่ามันเป็นกับดักของ migration ทุกตัว
    # ไม่ใช่ความเผลอครั้งเดียว
    present = {
        r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "users" not in present:
        conn.execute(
            "CREATE TABLE users ("
            " id TEXT PRIMARY KEY, email TEXT NOT NULL, name TEXT NOT NULL,"
            " google_sub TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,"
            " token TEXT NOT NULL DEFAULT '')"
        )
        present.add("users")

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)")}
    if "token" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN token TEXT NOT NULL DEFAULT ''")
    for row in conn.execute("SELECT id, token FROM users").fetchall():
        if not row["token"]:
            conn.execute("UPDATE users SET token = ? WHERE id = ?", (new_token(), row["id"]))

    if "courses" not in present:
        return  # v2→v3 สร้างให้อยู่แล้ว — ถ้าไม่มีแปลว่าลำดับ migration ผิด ไม่ใช่เรื่องปกติ
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(courses)")}
    if "join_code" not in cols:
        conn.execute("ALTER TABLE courses ADD COLUMN join_code TEXT NOT NULL DEFAULT ''")
        conn.execute("ALTER TABLE courses ADD COLUMN archived_at TEXT")
    for row in conn.execute("SELECT id, join_code FROM courses").fetchall():
        if not row["join_code"]:
            conn.execute(
                "UPDATE courses SET join_code = ? WHERE id = ?", (new_invite_code(), row["id"])
            )

    if "competitions" not in present:
        return
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(competitions)")}
    if "paradigm" not in cols:
        # competition ที่มีอยู่ก่อน v4 เป็นโจทย์ RL ทั้งหมด — ไม่ได้เดา แต่เป็นข้อเท็จจริง
        # ของ deployment นี้ (มี cp463-vacuum อันเดียว) · ตัวที่สร้างใหม่ต้องระบุเอง
        conn.execute(
            "ALTER TABLE competitions ADD COLUMN paradigm TEXT NOT NULL "
            "DEFAULT 'reinforcement-learning'"
        )


#: เวอร์ชันปลายทาง → ฟังก์ชันที่พาจากเวอร์ชันก่อนหน้ามาถึงมัน
MIGRATIONS = {2: _migrate_1_to_2, 3: _migrate_2_to_3, 4: _migrate_3_to_4}


class Database:
    """SQLite ที่เปิดค้างไว้ตลอดอายุ process

    ใช้ WAL เพื่อให้การอ่านไม่บล็อกการเขียน และล็อกด้วย `threading.Lock` เพราะ
    `arena serve` รัน worker เป็น thread ในกระบวนการเดียวกัน
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        # ⚠️ ลำดับสำคัญ: migrate ก่อน executescript
        #
        # SCHEMA มี index ที่อ้างคอลัมน์ใหม่ (เช่น teams.token) ส่วน
        # `CREATE TABLE IF NOT EXISTS` ไม่แตะตารางที่มีอยู่แล้ว ถ้ารัน SCHEMA ก่อน
        # กับไฟล์เวอร์ชันเก่า มันจะล้มที่ `no such column: token` ก่อนที่ migration
        # จะได้ทำงานเลยสักครั้ง
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._check_version()
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def _check_version(self) -> None:
        """ตั้งเวอร์ชันของไฟล์ใหม่ หรือพาไฟล์เก่าขึ้นมาให้ตรงกับโค้ด"""
        row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None and self._has_tables():
            # ไฟล์มีตารางแล้วแต่ไม่มี meta — เก่ากว่าตอนที่เริ่มบันทึกเวอร์ชัน
            row = {"value": "1"}
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', '1')"
            )
        if row is None:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()
            return
        found = int(row["value"])
        if found == SCHEMA_VERSION:
            return
        if found > SCHEMA_VERSION:
            raise SchemaMismatch(
                f"{self.path} ใช้ schema เวอร์ชัน {found} ซึ่ง**ใหม่กว่า**โค้ดนี้ ({SCHEMA_VERSION})\n"
                "น่าจะเปิดด้วยโค้ดเวอร์ชันเก่า — อัพเดตโค้ดก่อน อย่ารันต่อ"
            )

        missing = [v for v in range(found + 1, SCHEMA_VERSION + 1) if v not in MIGRATIONS]
        if missing:
            raise SchemaMismatch(
                f"{self.path} ใช้ schema เวอร์ชัน {found} แต่โค้ดนี้คาดหวัง {SCHEMA_VERSION}\n"
                f"ไม่มี migration สำหรับเวอร์ชัน {missing} — ต้องเขียนก่อน\n"
                "การเปิดไฟล์เก่าด้วย schema ใหม่โดยไม่ migrate จะทำให้ข้อมูลบางส่วน\n"
                "หายไปเงียบๆ ซึ่งแย่กว่าการล้มทันที"
            )

        # ทำทั้งชุดใน transaction เดียว — migration ที่ล้มกลางทางแล้วทิ้งไฟล์ไว้ครึ่งๆ
        # แย่กว่าไม่ได้เริ่มเลย เพราะครั้งถัดไปจะไม่รู้ว่าค้างอยู่ตรงไหน
        for version in range(found + 1, SCHEMA_VERSION + 1):
            MIGRATIONS[version](self._conn)
        self._conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),)
        )
        self._conn.commit()
        print(f"migrate {self.path.name}: schema {found} → {SCHEMA_VERSION}", file=sys.stderr)

    def _has_tables(self) -> bool:
        return bool(
            self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='teams'"
            ).fetchone()
        )

    def close(self) -> None:
        """ปิดแบบเรียบร้อย — ยุบ WAL กลับเข้าไฟล์หลักก่อน

        WAL ที่ค้างอยู่ไม่ได้ทำให้ข้อมูลหาย (SQLite กู้เองตอนเปิดครั้งถัดไป) แต่การ
        คัดลอกเฉพาะ `arena.db` ไปสำรองโดยลืม `-wal` จะได้ไฟล์ที่ข้อมูลไม่ครบ
        """
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass  # ปิดให้ได้ไว้ก่อน — checkpoint ล้มไม่ได้ทำให้ข้อมูลหาย
            self._conn.close()

    def _write(self, sql: str, params: tuple) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    # ── เขียน ───────────────────────────────────────────────────────

    def save_team(self, team: Team) -> None:
        self._write(
            "INSERT OR REPLACE INTO teams(id, course_id, name, alias, member_ids,"
            " token, invite_code, dissolved_at) VALUES(?,?,?,?,?,?,?,?)",
            (
                team.id, team.course_id, team.name, team.alias,
                json.dumps(team.member_ids), team.token, team.invite_code,
                _dt(team.dissolved_at),
            ),
        )

    def save_course(self, c: Course) -> None:
        self._write(
            "INSERT OR REPLACE INTO courses(id, name, max_team_size, join_code, archived_at)"
            " VALUES(?,?,?,?,?)",
            (c.id, c.name, c.max_team_size, c.join_code, _dt(c.archived_at)),
        )

    def save_user(self, user: User) -> None:
        self._write(
            "INSERT OR REPLACE INTO users(id, email, name, google_sub, token, created_at)"
            " VALUES(?,?,?,?,?,?)",
            (user.id, user.email, user.name, user.google_sub, user.token,
             _dt(user.created_at)),
        )

    def save_competition(self, c: Competition) -> None:
        phases = [
            {
                "id": p.id,
                "name": p.name,
                "starts_at": _dt(p.starts_at),
                "ends_at": _dt(p.ends_at),
                "config_override": p.config_override,
            }
            for p in c.phases
        ]
        self._write(
            "INSERT OR REPLACE INTO competitions(id, course_id, slug, title, task_type,"
            " env_plugin, config_path, opens_at, closes_at, quota_per_day,"
            " max_final_submissions, phases, import_whitelist, paradigm)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                c.id, c.course_id, c.slug, c.title, c.task_type, c.env_plugin, c.config_path,
                _dt(c.opens_at), _dt(c.closes_at), c.quota_per_day, c.max_final_submissions,
                json.dumps(phases, ensure_ascii=False),
                json.dumps(sorted(c.import_whitelist)),
                c.paradigm,
            ),
        )

    def save_submission(self, s: Submission) -> None:
        self._write(
            "INSERT OR REPLACE INTO submissions(id, competition_id, team_id, submitted_by,"
            " artifact_url, artifact_sha256, note, is_final_pick, created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (
                s.id, s.competition_id, s.team_id, s.submitted_by, s.artifact_url,
                s.artifact_sha256, s.note, int(s.is_final_pick), _dt(s.created_at),
            ),
        )

    def save_run(self, r: Run) -> None:
        episodes = [
            {
                "run_id": e.run_id, "seed": e.seed, "score": e.score,
                "status": e.status, "metrics": e.metrics, "replay_url": e.replay_url,
            }
            for e in r.episodes
        ]
        self._write(
            "INSERT OR REPLACE INTO runs(id, submission_id, competition_id, team_id, kind,"
            " status, lane, config_hash, env_version, score, tiebreak, metrics, episodes,"
            " error_message, attempts, runner_id, lease_expires_at, created_at, started_at,"
            " finished_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r.id, r.submission_id, r.competition_id, r.team_id, r.kind.value,
                r.status.value, r.lane, r.config_hash, r.env_version, r.score,
                json.dumps(list(r.tiebreak)),
                json.dumps(r.metrics, ensure_ascii=False, default=str),
                json.dumps(episodes, ensure_ascii=False, default=str),
                r.error_message, r.attempts, r.runner_id, _dt(r.lease_expires_at),
                _dt(r.created_at), _dt(r.started_at), _dt(r.finished_at),
            ),
        )

    def save_audit(self, e: AuditEvent) -> None:
        self._write(
            "INSERT OR REPLACE INTO audit(id, actor_id, action, target_type, target_id,"
            " payload, created_at) VALUES(?,?,?,?,?,?,?)",
            (
                e.id, e.actor_id, e.action, e.target_type, e.target_id,
                json.dumps(e.payload, ensure_ascii=False, default=str), _dt(e.created_at),
            ),
        )

    def save_served(self, team_id: str, count: int) -> None:
        self._write(
            "INSERT OR REPLACE INTO queue_served(team_id, count) VALUES(?,?)", (team_id, count)
        )

    # ── อ่านตอนเริ่ม ─────────────────────────────────────────────────

    def _rows(self, table: str) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(f"SELECT * FROM {table}"))

    def load_teams(self) -> dict[str, Team]:
        return {
            r["id"]: Team(
                id=r["id"], course_id=r["course_id"], name=r["name"], alias=r["alias"],
                member_ids=json.loads(r["member_ids"]),
                token=r["token"], invite_code=r["invite_code"],
                dissolved_at=_parse_dt(r["dissolved_at"]),
            )
            for r in self._rows("teams")
        }

    def load_courses(self) -> dict[str, Course]:
        return {
            r["id"]: Course(
                id=r["id"], name=r["name"], max_team_size=int(r["max_team_size"]),
                join_code=r["join_code"], archived_at=_parse_dt(r["archived_at"]),
            )
            for r in self._rows("courses")
        }

    def load_users(self) -> dict[str, User]:
        return {
            r["id"]: User(
                id=r["id"], email=r["email"], name=r["name"],
                google_sub=r["google_sub"], token=r["token"],
                created_at=_parse_dt(r["created_at"]),
            )
            for r in self._rows("users")
        }

    def load_competitions(self) -> dict[str, Competition]:
        out: dict[str, Competition] = {}
        for r in self._rows("competitions"):
            phases = [
                Phase(
                    id=p["id"], name=p["name"],
                    starts_at=_parse_dt(p["starts_at"]), ends_at=_parse_dt(p["ends_at"]),
                    config_override=p["config_override"],
                )
                for p in json.loads(r["phases"])
            ]
            out[r["id"]] = Competition(
                id=r["id"], course_id=r["course_id"], slug=r["slug"], title=r["title"],
                task_type=r["task_type"], env_plugin=r["env_plugin"],
                config_path=r["config_path"],
                opens_at=_parse_dt(r["opens_at"]), closes_at=_parse_dt(r["closes_at"]),
                quota_per_day=r["quota_per_day"],
                max_final_submissions=r["max_final_submissions"],
                phases=phases,
                import_whitelist=frozenset(json.loads(r["import_whitelist"])),
                paradigm=r["paradigm"],
            )
        return out

    def load_submissions(self) -> dict[str, Submission]:
        return {
            r["id"]: Submission(
                id=r["id"], competition_id=r["competition_id"], team_id=r["team_id"],
                submitted_by=r["submitted_by"], artifact_url=r["artifact_url"],
                artifact_sha256=r["artifact_sha256"], note=r["note"],
                is_final_pick=bool(r["is_final_pick"]), created_at=_parse_dt(r["created_at"]),
            )
            for r in self._rows("submissions")
        }

    def load_runs(self) -> dict[str, Run]:
        out: dict[str, Run] = {}
        for r in self._rows("runs"):
            out[r["id"]] = Run(
                id=r["id"], submission_id=r["submission_id"],
                competition_id=r["competition_id"], team_id=r["team_id"],
                kind=RunKind(r["kind"]), status=RunStatus(r["status"]), lane=r["lane"],
                config_hash=r["config_hash"], env_version=r["env_version"], score=r["score"],
                tiebreak=tuple(json.loads(r["tiebreak"])),
                metrics=json.loads(r["metrics"]),
                error_message=r["error_message"], attempts=r["attempts"],
                runner_id=r["runner_id"], lease_expires_at=_parse_dt(r["lease_expires_at"]),
                created_at=_parse_dt(r["created_at"]),
                started_at=_parse_dt(r["started_at"]),
                finished_at=_parse_dt(r["finished_at"]),
                episodes=[
                    EpisodeResult(
                        run_id=e["run_id"], seed=e["seed"], score=e["score"],
                        status=e["status"], metrics=e["metrics"], replay_url=e["replay_url"],
                    )
                    for e in json.loads(r["episodes"])
                ],
            )
        return out

    def load_audit(self) -> list[AuditEvent]:
        events = [
            AuditEvent(
                id=r["id"], actor_id=r["actor_id"], action=r["action"],
                target_type=r["target_type"], target_id=r["target_id"],
                payload=json.loads(r["payload"]), created_at=_parse_dt(r["created_at"]),
            )
            for r in self._rows("audit")
        ]
        return sorted(events, key=lambda e: e.created_at)

    def load_served(self) -> dict[str, int]:
        return {r["team_id"]: r["count"] for r in self._rows("queue_served")}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                table: self._conn.execute(f"SELECT count(*) c FROM {table}").fetchone()["c"]
                for table in ("teams", "users", "competitions", "submissions", "runs", "audit")
            }
