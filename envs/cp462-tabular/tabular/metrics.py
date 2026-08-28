"""คิดคะแนนและช่วงความเชื่อมั่น — template §6

**อยู่ฝั่ง trusted เสมอ** — ไฟล์นี้เห็นเฉลย จึงห้ามถูก import เข้าไปใน sandbox
ของนิสิตเด็ดขาด (template §5) · โค้ดนิสิตคืนแค่ `y_pred` ออกมาเป็นไฟล์
แล้วการให้คะแนนเกิดข้างนอกทั้งหมด

**คะแนนหลักต้อง "มากกว่าดีกว่า" เสมอ** ไม่ว่าโจทย์ชนิดไหน เพราะ leaderboard
เรียงจากมากไปน้อยทั้งระบบ · regression ที่อยากใช้ RMSE จึงเก็บเป็น `-RMSE`
แล้วแสดงผลกลับด้านตอนวาด — **ห้ามไปกลับทิศการเรียงเฉพาะบางโจทย์** เพราะโค้ดที่
เรียงสองทิศทางตามชนิดโจทย์คือที่ที่บั๊กชอบซ่อน

ช่วงความเชื่อมั่นบอก **ความไม่แน่นอนที่มาจากขนาดของ test set เท่านั้น**
ไม่ได้บอกความไม่แน่นอนจากการเทรน เพราะเราประเมินโมเดลที่ส่งมาแล้วหนึ่งตัว
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn import metrics as skm

#: จำนวนครั้งที่สุ่มซ้ำสำหรับ bootstrap — 1000 พอสำหรับ percentile ที่ 2.5/97.5
#: และเร็วพอที่จะรันในคิวได้โดยไม่ต้องมี stage แยก
BOOTSTRAP_ROUNDS = 1000

#: MAPE ระเบิดเมื่อเป้าหมายใกล้ศูนย์ — ต่ำกว่านี้ถือว่าใช้ไม่ได้ คืน `None` แทน inf
MAPE_MIN_ABS = 1e-6


class MetricError(Exception):
    """คำนวณคะแนนไม่ได้ — ข้อความต้องบอกว่าอะไรไม่ตรงและคาดหวังอะไร"""


@dataclass(frozen=True)
class Score:
    """ผลการให้คะแนนของ submission หนึ่งครั้ง

    `primary` คือตัวเดียวที่ leaderboard ใช้เรียง · `reported` คือทุกอย่างที่
    แสดงให้นิสิตดู ซึ่งมีค่าทางการสอนแม้ไม่ได้ใช้จัดอันดับ
    """

    primary_name: str
    primary: float
    ci_low: float
    ci_high: float
    reported: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_name": self.primary_name,
            "primary": self.primary,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            **self.reported,
        }


def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if len(y_pred) != len(y_true):
        raise MetricError(
            f"จำนวนคำทำนายไม่ตรงกับจำนวนแถว — คาดหวัง {len(y_true)} ได้ {len(y_pred)}\n"
            "  `predict` ต้องคืนค่าให้ครบทุกแถวตามลำดับที่รับเข้ามา"
        )
    if y_pred.ndim != 1:
        raise MetricError(
            f"คำทำนายต้องเป็นอาเรย์มิติเดียว — ได้รูปร่าง {y_pred.shape}\n"
            "  ถ้าโมเดลคืน probability ให้แปลงเป็น label ก่อน (`predict` ไม่ใช่ `predict_proba`)"
        )
    if not np.isfinite(np.asarray(y_pred, dtype="float64")).all():
        raise MetricError("คำทำนายมีค่า NaN หรือ inf — ตรวจ imputer และการแปลงเป้าหมาย")


# ── classification ─────────────────────────────────────────────────


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels) -> dict:
    # `zero_division=0` เพราะคลาสที่โมเดลไม่เคยทายเลยจะให้ precision เป็น 0/0
    # ค่าเริ่มต้นของ sklearn คือเตือนแล้วคืน 0 — เรากำหนดตรงๆ ให้ผลไม่ขึ้นกับเวอร์ชัน
    kw = {"labels": labels, "zero_division": 0}
    return {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "macro_f1": float(skm.f1_score(y_true, y_pred, average="macro", **kw)),
        "weighted_f1": float(skm.f1_score(y_true, y_pred, average="weighted", **kw)),
        "per_class_f1": {
            str(label): float(value)
            for label, value in zip(labels, skm.f1_score(y_true, y_pred, average=None, **kw))
        },
        "confusion_matrix": skm.confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [str(label) for label in labels],
    }


def _classification_primary(name: str, y_true, y_pred, labels) -> float:
    if name == "macro_f1":
        return float(skm.f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    if name == "accuracy":
        return float(skm.accuracy_score(y_true, y_pred))
    raise MetricError(
        f"ไม่รองรับคะแนนหลัก {name!r} สำหรับ classification\n"
        "  ที่ใช้ได้จาก `predict` เปล่าๆ คือ macro_f1 หรือ accuracy\n"
        "  ส่วน roc_auc / pr_auc ต้องบังคับ `predict_proba` ซึ่งยังไม่รองรับ"
    )


# ── regression ─────────────────────────────────────────────────────


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residual = y_true - y_pred
    out = {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(skm.r2_score(y_true, y_pred)),
        "median_ae": float(np.median(np.abs(residual))),
    }
    # MAPE มีความหมายเฉพาะเมื่อเป้าหมายไม่มีค่าใกล้ศูนย์ — ไม่งั้นได้ค่ามหาศาล
    # ที่ดูเหมือนบั๊ก · คืน None ให้ชัดว่า "ใช้ไม่ได้กับโจทย์นี้" ไม่ใช่ "คำนวณพลาด"
    out["mape"] = (
        float(np.mean(np.abs(residual / y_true)))
        if np.all(np.abs(y_true) > MAPE_MIN_ABS)
        else None
    )
    return out


def _regression_primary(name: str, y_true, y_pred) -> float:
    if name == "r2":
        return float(skm.r2_score(y_true, y_pred))
    if name == "neg_rmse":
        return -float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if name == "neg_mae":
        return -float(np.mean(np.abs(y_true - y_pred)))
    raise MetricError(
        f"ไม่รองรับคะแนนหลัก {name!r} สำหรับ regression\n"
        "  ที่ใช้ได้คือ r2, neg_rmse, neg_mae — ทุกตัว 'มากกว่าดีกว่า' ตามที่ leaderboard ต้องการ"
    )


# ── bootstrap ──────────────────────────────────────────────────────


def _resample_indices(
    rng: np.random.Generator, y_true: np.ndarray, stratified: bool
) -> np.ndarray:
    """ดัชนีของการสุ่มซ้ำหนึ่งรอบ

    classification สุ่มแยกตามคลาสเพื่อรักษาสัดส่วน — ถ้าสุ่มรวม รอบที่บังเอิญไม่มี
    คลาสน้อยเลยจะทำให้ macro-F1 กระโดด แล้ว CI กว้างเกินจริง
    """
    n = len(y_true)
    if not stratified:
        return rng.integers(0, n, size=n)
    parts = []
    for label in np.unique(y_true):
        pool = np.flatnonzero(y_true == label)
        parts.append(pool[rng.integers(0, len(pool), size=len(pool))])
    return np.concatenate(parts)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric,
    seed: int,
    stratified: bool,
    rounds: int = BOOTSTRAP_ROUNDS,
) -> tuple[float, float]:
    """ช่วงความเชื่อมั่น 95% ด้วย percentile bootstrap

    **seed ตรึงต่อ competition** — CI ของทุกทีมจึงเทียบกันได้และรันซ้ำได้ค่าเดิม
    ถ้าปล่อยให้สุ่มอิสระ ทีมที่ส่งวันเดียวกันจะได้ CI คนละแบบด้วยเหตุผลที่อธิบายไม่ได้
    """
    rng = np.random.default_rng(seed)
    values = np.empty(rounds, dtype="float64")
    for i in range(rounds):
        idx = _resample_indices(rng, y_true, stratified)
        values[i] = metric(y_true[idx], y_pred[idx])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# ── ทางเข้าหลัก ────────────────────────────────────────────────────


def score(
    y_true,
    y_pred,
    *,
    kind: str,
    primary: str,
    seed: int,
    labels=None,
    rounds: int = BOOTSTRAP_ROUNDS,
) -> Score:
    """ให้คะแนน submission หนึ่งครั้ง — คืนคะแนนหลัก ช่วงความเชื่อมั่น และตัวเลขที่รายงาน

    `kind` เป็น `"classification"` หรือ `"regression"` · `labels` ต้องระบุสำหรับ
    classification เพื่อให้ confusion matrix และ per-class F1 มีลำดับตรึง
    ไม่ขึ้นกับว่าโมเดลบังเอิญทายคลาสไหนบ้าง
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    _check_shapes(y_true, y_pred)

    if kind == "classification":
        if labels is None:
            raise MetricError("classification ต้องระบุ `labels` เพื่อตรึงลำดับของคลาส")
        labels = list(labels)
        unknown = sorted(set(np.unique(y_pred).tolist()) - set(labels))
        if unknown:
            raise MetricError(
                f"คำทำนายมีคลาสที่ไม่มีในโจทย์: {unknown}\n"
                f"  คลาสที่ใช้ได้คือ {labels}"
            )
        reported = _classification_metrics(y_true, y_pred, labels)
        fn = lambda a, b: _classification_primary(primary, a, b, labels)  # noqa: E731
        stratified = True
    elif kind == "regression":
        reported = _regression_metrics(y_true, y_pred)
        fn = lambda a, b: _regression_primary(primary, a, b)  # noqa: E731
        stratified = False
    else:
        raise MetricError(f"ไม่รู้จักชนิดโจทย์ {kind!r} — ที่มีคือ classification, regression")

    point = fn(y_true, y_pred)
    low, high = bootstrap_ci(
        y_true, y_pred, metric=fn, seed=seed, stratified=stratified, rounds=rounds
    )
    return Score(
        primary_name=primary, primary=point, ci_low=low, ci_high=high, reported=reported
    )
