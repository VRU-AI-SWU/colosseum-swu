"""จัดอันดับ — README §5

**core ไม่รู้สูตรคะแนนของโจทย์ใดๆ** มันรับมาแค่สเกลาร์ตัวเดียวกับคีย์ตัดสินเสมอ
ที่ env plugin ประกาศไว้ แล้วเรียงให้ ([README §5.1](../README.md#51-metric-เป็นของแต่ละโจทย์-ไม่ใช่ของแพลตฟอร์ม))

สองข้อที่ต้องระวังเป็นพิเศษ

- **public กับ private แยกกันเด็ดขาด** — เอามาปนกันเมื่อไหร่ private leaderboard
  หมดความหมายทันที เพราะมันมีไว้กัน overfit ต่อ public
- **หนึ่งทีมมีได้แถวเดียว** — เอา run ที่ดีที่สุดของทีมนั้น ไม่ใช่ทุก run
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from core.domain import Run, RunKind, Team


@dataclass
class LeaderboardRow:
    rank: int
    team_id: str
    display_name: str
    score: float
    run_id: str
    submission_id: str
    metrics: dict
    previous_rank: int | None = None

    @property
    def movement(self) -> str:
        """ลูกศรบอกการเปลี่ยนอันดับ — README §6.1"""
        if self.previous_rank is None:
            return "new"
        if self.previous_rank > self.rank:
            return f"▲{self.previous_rank - self.rank}"
        if self.previous_rank < self.rank:
            return f"▼{self.rank - self.previous_rank}"
        return "–"


def best_run_per_team(runs: Iterable[Run], kind: RunKind) -> dict[str, Run]:
    """run ที่ดีที่สุดของแต่ละทีมสำหรับ leaderboard ชนิดที่ระบุ

    เรียงด้วย `(score, *tiebreak)` ที่ env plugin ประกาศ — core ไม่ตีความว่าแต่ละตัวคืออะไร
    รู้แค่ว่ามากกว่า = ดีกว่า ตามสัญญาใน README §5.1
    """
    best: dict[str, Run] = {}
    for run in runs:
        if not run.counts_for_leaderboard or run.kind is not kind or run.score is None:
            continue
        current = best.get(run.team_id)
        if current is None or _key(run) > _key(current):
            best[run.team_id] = run
    return best


def _key(run: Run) -> tuple:
    # เวลาที่ส่งก่อนเป็นเกณฑ์สุดท้ายเสมอ — ใส่เป็นค่าติดลบเพราะ "เก่ากว่า = ดีกว่า"
    return (run.score, *run.tiebreak, -run.created_at.timestamp())


def build(
    runs: Iterable[Run],
    teams: dict[str, Team],
    *,
    kind: RunKind = RunKind.PUBLIC,
    reveal_names: bool = False,
    previous: dict[str, int] | None = None,
) -> list[LeaderboardRow]:
    """สร้างตาราง — `reveal_names=True` สำหรับผู้สอน/TA เท่านั้น (README §6.1)"""
    previous = previous or {}
    best = best_run_per_team(runs, kind)
    ordered = sorted(best.values(), key=_key, reverse=True)

    rows = []
    for i, run in enumerate(ordered, start=1):
        team = teams.get(run.team_id)
        rows.append(
            LeaderboardRow(
                rank=i,
                team_id=run.team_id,
                display_name=(
                    team.display_name(reveal=reveal_names) if team else run.team_id
                ),
                score=run.score,
                run_id=run.id,
                submission_id=run.submission_id,
                metrics=run.metrics,
                previous_rank=previous.get(run.team_id),
            )
        )
    return rows


@dataclass
class BaselineMark:
    """หมุดของผู้สอนที่วางไว้บน leaderboard — README §6.2

    คะแนนต้องได้จากการรันจริงบน public seeds ชุดเดียวกับนิสิต แล้ว**ตรึงไว้ทั้งเทอม**
    ถ้าคำนวณใหม่ทุกครั้งที่แสดงผล หมุดจะขยับและนิสิตจะไล่ตามเป้าที่เคลื่อนที่
    """

    level: str  # bronze | silver | gold | diamond
    label: str
    score: float
    config_hash: str  # ผูกกับ config ที่ใช้รัน — เปลี่ยน phase แล้วต้องรันใหม่


def insert_baselines(
    rows: list[LeaderboardRow], marks: Iterable[BaselineMark]
) -> list[tuple[str, LeaderboardRow | BaselineMark]]:
    """แทรกหมุด baseline ลงในตารางตามตำแหน่งคะแนน — คืนรายการที่มีทั้งทีมและหมุดปนกัน

    ทำที่ชั้นการแสดงผล ไม่ใช่ในตาราง เพราะ baseline ไม่ใช่ทีมและไม่ควรกินอันดับ
    """
    items: list[tuple[float, str, LeaderboardRow | BaselineMark]] = [
        (row.score, "team", row) for row in rows
    ]
    items += [(m.score, "baseline", m) for m in marks]
    # เสมอกัน → หมุด baseline อยู่ล่างกว่าทีม เพื่อให้ "ทีมนี้แตะ Gold แล้ว" อ่านได้ตรงไปตรงมา
    items.sort(key=lambda it: (it[0], it[1] == "team"), reverse=True)
    return [(kind, obj) for _score, kind, obj in items]


def next_target(rows: list[LeaderboardRow], team_id: str, marks: Iterable[BaselineMark]) -> BaselineMark | None:
    """หมุดถัดไปที่ทีมนี้ยังไปไม่ถึง — หัวใจของ "ทุกทีมมีเป้าหมายถัดไปที่ทำได้เสมอ" (README §6.2)

    คืน `None` เมื่อผ่านทุกหมุดแล้ว — ตอนนั้นเป้าหมายเปลี่ยนเป็นการไล่ทีมอื่นแทน
    """
    row = next((r for r in rows if r.team_id == team_id), None)
    score = row.score if row else float("-inf")
    ahead = sorted((m for m in marks if m.score > score), key=lambda m: m.score)
    return ahead[0] if ahead else None
