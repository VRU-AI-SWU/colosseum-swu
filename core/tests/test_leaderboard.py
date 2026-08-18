"""Leaderboard — README §5 และ §6

core ต้องจัดอันดับได้โดย **ไม่รู้สูตรคะแนนของโจทย์เลย** — รับมาแค่สเกลาร์กับคีย์ตัดสินเสมอ
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from core.domain import Run, RunKind, RunStatus, Team, new_id
from core.leaderboard import BaselineMark, build, insert_baselines, next_target

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)

TEAMS = {
    "A": Team(id="A", course_id="c", name="ทีมกวาดบ้าน", alias="ทีมลึกลับ 1"),
    "B": Team(id="B", course_id="c", name="ทีมฝุ่นหาย"),
}

LADDER = [
    BaselineMark("bronze", "🥉 Bronze", 0.244, "sha256:x"),
    BaselineMark("silver", "🥈 Silver", 0.810, "sha256:x"),
    BaselineMark("gold", "🥇 Gold", 1.716, "sha256:x"),
    BaselineMark("diamond", "💎 Diamond", 1.804, "sha256:x"),
]


def done_run(team, score, *, kind=RunKind.PUBLIC, tiebreak=(), at=T0) -> Run:
    return Run(
        id=new_id(), submission_id=new_id(), competition_id="comp", team_id=team,
        kind=kind, status=RunStatus.DONE, score=score, tiebreak=tiebreak, created_at=at,
    )


def test_ranks_by_score_descending():
    rows = build([done_run("A", 1.2), done_run("B", 1.5)], TEAMS)
    assert [r.team_id for r in rows] == ["B", "A"]
    assert [r.rank for r in rows] == [1, 2]


def test_one_row_per_team_using_best_run():
    runs = [done_run("A", 0.5), done_run("A", 1.7), done_run("A", 1.1)]
    rows = build(runs, TEAMS)
    assert len(rows) == 1
    assert rows[0].score == 1.7


def test_public_and_private_never_mix():
    """ปนกันเมื่อไหร่ private leaderboard หมดความหมาย เพราะมันมีไว้กัน overfit ต่อ public"""
    runs = [done_run("A", 1.9, kind=RunKind.PUBLIC), done_run("A", 0.7, kind=RunKind.PRIVATE)]
    assert build(runs, TEAMS, kind=RunKind.PUBLIC)[0].score == 1.9
    assert build(runs, TEAMS, kind=RunKind.PRIVATE)[0].score == 0.7


def test_dryrun_and_rejudge_never_appear():
    runs = [
        done_run("A", 9.9, kind=RunKind.DRYRUN),
        done_run("A", 8.8, kind=RunKind.REJUDGE),
        done_run("A", 1.0),
    ]
    assert [r.score for r in build(runs, TEAMS)] == [1.0]


def test_unfinished_runs_are_ignored():
    pending = done_run("A", 5.0)
    pending.status = RunStatus.RUNNING
    assert build([pending, done_run("B", 1.0)], TEAMS)[0].team_id == "B"


def test_tiebreak_comes_from_the_env_plugin():
    """คะแนนเท่ากัน → ใช้คีย์ที่ plugin ประกาศ · core ไม่รู้ว่าตัวเลขพวกนั้นคืออะไร"""
    runs = [done_run("A", 1.5, tiebreak=(28, 0.4)), done_run("B", 1.5, tiebreak=(30, 0.2))]
    assert [r.team_id for r in build(runs, TEAMS)] == ["B", "A"]


def test_earlier_submission_wins_when_everything_ties():
    runs = [
        done_run("A", 1.5, at=T0 + timedelta(hours=2)),
        done_run("B", 1.5, at=T0),
    ]
    assert [r.team_id for r in build(runs, TEAMS)] == ["B", "A"]


def test_alias_hides_the_real_name_from_students_only():
    rows = build([done_run("A", 1.0)], TEAMS)
    assert rows[0].display_name == "ทีมลึกลับ 1"
    assert build([done_run("A", 1.0)], TEAMS, reveal_names=True)[0].display_name == "ทีมกวาดบ้าน"


def test_team_without_alias_shows_its_name():
    assert build([done_run("B", 1.0)], TEAMS)[0].display_name == "ทีมฝุ่นหาย"


def test_movement_arrows():
    rows = build([done_run("A", 1.0), done_run("B", 2.0)], TEAMS, previous={"A": 1, "B": 5})
    by_team = {r.team_id: r for r in rows}
    assert by_team["B"].movement == "▲4"
    assert by_team["A"].movement == "▼1"
    assert build([done_run("A", 1.0)], TEAMS)[0].movement == "new"


# ── baseline ladder ─────────────────────────────────────────────────


def test_baselines_are_interleaved_by_score():
    rows = build([done_run("A", 1.75), done_run("B", 0.5)], TEAMS)
    items = insert_baselines(rows, LADDER)
    labels = [
        obj.display_name if kind == "team" else obj.label for kind, obj in items
    ]
    assert labels == ["💎 Diamond", "ทีมลึกลับ 1", "🥇 Gold", "🥈 Silver", "ทีมฝุ่นหาย", "🥉 Bronze"]


def test_team_ranks_above_a_baseline_it_ties_with():
    """เสมอกับหมุด = แตะระดับนั้นแล้ว — ควรอ่านว่าอยู่เหนือหมุด"""
    rows = build([done_run("A", 1.716)], TEAMS)
    kinds = [kind for kind, _ in insert_baselines(rows, [LADDER[2]])]
    assert kinds == ["team", "baseline"]


def test_next_target_gives_every_team_something_reachable():
    """หัวใจของ §6.2 — ทีมท้ายตารางต้องมีเป้าหมายถัดไปที่ทำได้ ไม่ใช่แค่ไล่ที่ 1"""
    rows = build([done_run("A", 1.75), done_run("B", 0.3)], TEAMS)
    assert next_target(rows, "B", LADDER).level == "silver"
    assert next_target(rows, "A", LADDER).level == "diamond"


def test_next_target_is_none_after_clearing_the_ladder():
    rows = build([done_run("A", 2.0)], TEAMS)
    assert next_target(rows, "A", LADDER) is None


def test_team_with_no_run_targets_the_lowest_mark():
    assert next_target([], "B", LADDER).level == "bronze"
