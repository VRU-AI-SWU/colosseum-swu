"""การคิดคะแนน — environment-spec §7

module นี้ต้อง import ได้แยกจาก env: **grader ใช้ตัวนี้ตัวเดียวกับ starter kit**
ถ้าสูตรสองฝั่งต่างกันแม้แต่นิดเดียว นิสิตจะจูนบนสิ่งที่ไม่ตรงกับตอนตัดสิน

การคำนวณใช้ float64 ตลอด และปัดเป็น 6 ตำแหน่งทศนิยม *ตอนบันทึกลง DB เท่านั้น*
(ดู `round_for_storage`) ไม่ใช่ระหว่างคำนวณ
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCORE_DECIMALS = 6


@dataclass(frozen=True)
class EpisodeStats:
    """สิ่งที่ต้องเก็บจาก 1 episode เพื่อคิดคะแนน

    `cleaned_at_t[i]` = จำนวน cell ที่ดูดสำเร็จแล้วหลังผ่านไป i timestep
    (index 0 = ตอนเริ่ม = 0 เสมอ) ความยาว = t_end + 1
    """

    D0: int
    cleaned_at_t: np.ndarray
    collisions: int
    redundant_sucks: int
    sticky_fails: int = 0
    slips: int = 0
    reason: str | None = None

    @property
    def t_end(self) -> int:
        return len(self.cleaned_at_t) - 1

    @property
    def cleaned(self) -> int:
        return int(self.cleaned_at_t[-1])


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    auc: float
    completed: bool
    penalty: float
    coverage: float
    t_end: int
    # ตัวเลขที่แสดงอย่างเดียว ไม่มีผลต่อคะแนน
    collisions: int = 0
    redundant_sucks: int = 0
    sticky_fails: int = 0
    slips: int = 0
    reason: str | None = None


def coverage_curve(stats: EpisodeStats, max_steps: int) -> np.ndarray:
    """c[0..T] โดยที่ episode ที่จบก่อน T ให้ค่าคงที่จนถึง T"""
    T = max_steps
    c = np.zeros(T + 1, dtype=np.float64)
    n = min(stats.t_end, T)
    c[: n + 1] = stats.cleaned_at_t[: n + 1] / float(stats.D0)
    if n < T:
        c[n + 1 :] = c[n]
    return c


def episode_score(
    stats: EpisodeStats,
    max_steps: int,
    *,
    completion_bonus: float = 1.0,
    max_penalty: float = 0.2,
    w_collision: float = 1.0,
    w_redundant: float = 0.2,
) -> ScoreBreakdown:
    """คะแนนของ 1 episode

    AUC ตรงกับ "ดูดครบโดยใช้ timestep น้อยที่สุด" โดยไม่ต้องจูนน้ำหนัก:
    agent ที่ดูดครบที่ timestep t_c จะได้ AUC ราว 1 − t_c/(2T)

    `sticky_fails` และ `slips` ไม่ถูกลงโทษ — เป็นผลของสิ่งที่ agent ควบคุมไม่ได้
    """
    T = float(max_steps)
    c = coverage_curve(stats, max_steps)

    auc = float(np.sum(c[1:]) / T)
    penalty_raw = w_collision * stats.collisions / T + w_redundant * stats.redundant_sucks / T
    penalty = float(min(max_penalty, penalty_raw))
    completed = bool(c[-1] >= 1.0)

    score = auc + completion_bonus * (1.0 if completed else 0.0) - penalty

    return ScoreBreakdown(
        score=score,
        auc=auc,
        completed=completed,
        penalty=penalty,
        coverage=float(c[-1]),
        t_end=stats.t_end,
        collisions=stats.collisions,
        redundant_sucks=stats.redundant_sucks,
        sticky_fails=stats.sticky_fails,
        slips=stats.slips,
        reason=stats.reason,
    )


@dataclass(frozen=True)
class SubmissionScore:
    score: float  # ค่าเฉลี่ยเลขคณิตของ episode_score ทุก seed
    n_completed: int
    worst_episode: float
    mean_t_end_completed: float | None  # None ถ้าไม่มี episode ไหน completed เลย
    mean_coverage: float
    sd_across_seeds: float  # แสดงอย่างเดียว — ไม่ใช้จัดอันดับ
    per_episode: list[ScoreBreakdown] = field(default_factory=list)

    @property
    def tiebreak_key(self) -> tuple[float, int, float, float]:
        """คีย์สำหรับเรียงอันดับ — มากกว่า = อันดับสูงกว่า (§7)

        ลำดับ: คะแนนรวม → จำนวน seed ที่ completed → worst-episode →
        (ติดลบของ) t_end เฉลี่ยเฉพาะ episode ที่ completed
        เกณฑ์สุดท้าย "เวลาที่ส่งก่อน" เป็นเรื่องของแพลตฟอร์ม ไม่ใช่ของ environment
        """
        t_end = self.mean_t_end_completed
        return (
            self.score,
            self.n_completed,
            self.worst_episode,
            -t_end if t_end is not None else float("-inf"),
        )


def submission_score(breakdowns: list[ScoreBreakdown]) -> SubmissionScore:
    if not breakdowns:
        raise ValueError("ต้องมีอย่างน้อย 1 episode")

    scores = np.array([b.score for b in breakdowns], dtype=np.float64)
    completed = [b for b in breakdowns if b.completed]

    return SubmissionScore(
        score=float(scores.mean()),
        n_completed=len(completed),
        worst_episode=float(scores.min()),
        mean_t_end_completed=(
            float(np.mean([b.t_end for b in completed])) if completed else None
        ),
        mean_coverage=float(np.mean([b.coverage for b in breakdowns])),
        sd_across_seeds=float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        per_episode=list(breakdowns),
    )


def round_for_storage(value: float) -> float:
    """ปัดตอนบันทึกลง DB เท่านั้น — ห้ามปัดระหว่างคำนวณ"""
    return round(float(value), SCORE_DECIMALS)
