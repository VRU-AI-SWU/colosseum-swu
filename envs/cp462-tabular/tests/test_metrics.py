"""การให้คะแนน — ตัวที่ตัดสินอันดับจริง จึงต้องแม่นและอธิบายได้เมื่อผิด

สามเรื่องที่ผิดแล้วเจ็บและมองไม่เห็นตอนรีวิว

  · **คะแนนหลักต้อง "มากกว่าดีกว่า"** ทุกชนิดโจทย์ ไม่งั้น leaderboard เรียงกลับด้าน
    สำหรับ regression โดยไม่มีใครสังเกตจนกว่าจะตัดเกรด
  · **CI ต้องทำซ้ำได้** — ทีมที่ส่งวันเดียวกันต้องได้ CI ที่เทียบกันได้
  · **ข้อความผิดพลาดต้องบอกว่าอะไรไม่ตรง** — นิสิตเห็นแค่ข้อความนี้ ไม่เห็นโค้ด
"""

from __future__ import annotations

import numpy as np
import pytest

from tabular.metrics import BOOTSTRAP_ROUNDS, MetricError, bootstrap_ci, score

FAST = 120  # bootstrap รอบน้อยลงในเทสต์ — เกณฑ์ที่ตรวจไม่ได้ขึ้นกับจำนวนรอบ


@pytest.fixture
def clf():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    noisy = np.where(rng.random(400) < 0.25, 1 - y, y)   # ทายถูก ~75%
    return y, noisy


@pytest.fixture
def reg():
    rng = np.random.default_rng(0)
    y = rng.normal(1000, 200, size=400)
    return y, y + rng.normal(0, 60, size=400)


# ── คะแนนหลักต้องมากกว่าดีกว่า ─────────────────────────────────────


@pytest.mark.parametrize("primary", ["r2", "neg_rmse", "neg_mae"])
def test_regression_primary_is_higher_is_better(reg, primary):
    """**หัวใจของการเรียง leaderboard** — ทำนายแม่นกว่าต้องได้คะแนนมากกว่าเสมอ"""
    y, good = reg
    bad = good + 400.0                       # เลื่อนออกไปไกล = แย่ลงแน่นอน

    better = score(y, good, kind="regression", primary=primary, seed=1, rounds=FAST)
    worse = score(y, bad, kind="regression", primary=primary, seed=1, rounds=FAST)
    assert better.primary > worse.primary, f"{primary}: ทำนายแม่นกว่าได้คะแนนน้อยกว่า"


@pytest.mark.parametrize("primary", ["macro_f1", "accuracy"])
def test_classification_primary_is_higher_is_better(clf, primary):
    y, good = clf
    bad = 1 - y                              # ทายกลับด้านทุกแถว

    better = score(y, good, kind="classification", primary=primary, seed=1,
                   labels=[0, 1], rounds=FAST)
    worse = score(y, bad, kind="classification", primary=primary, seed=1,
                  labels=[0, 1], rounds=FAST)
    assert better.primary > worse.primary


def test_perfect_prediction_hits_the_ceiling(clf, reg):
    y, _ = clf
    s = score(y, y, kind="classification", primary="macro_f1", seed=1,
              labels=[0, 1], rounds=FAST)
    assert s.primary == pytest.approx(1.0)

    y, _ = reg
    s = score(y, y, kind="regression", primary="r2", seed=1, rounds=FAST)
    assert s.primary == pytest.approx(1.0)
    assert s.reported["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_predicting_the_mean_gives_r2_of_zero(reg):
    """R² เทียบกับการทายค่าเฉลี่ยเสมอ — เป็นเหตุผลที่มันอ่านง่ายกว่า RMSE"""
    y, _ = reg
    s = score(y, np.full_like(y, y.mean()), kind="regression", primary="r2",
              seed=1, rounds=FAST)
    assert s.primary == pytest.approx(0.0, abs=1e-9)


# ── ช่วงความเชื่อมั่น ──────────────────────────────────────────────


def test_ci_is_reproducible_with_the_same_seed(clf):
    y, p = clf
    kw = dict(kind="classification", primary="macro_f1", labels=[0, 1], rounds=FAST)
    a = score(y, p, seed=7, **kw)
    b = score(y, p, seed=7, **kw)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_ci_changes_with_a_different_seed(clf):
    y, p = clf
    kw = dict(kind="classification", primary="macro_f1", labels=[0, 1], rounds=FAST)
    assert score(y, p, seed=7, **kw).ci_low != score(y, p, seed=8, **kw).ci_low


def test_ci_brackets_the_point_estimate(clf, reg):
    y, p = clf
    s = score(y, p, kind="classification", primary="macro_f1", seed=1,
              labels=[0, 1], rounds=400)
    assert s.ci_low <= s.primary <= s.ci_high

    y, p = reg
    s = score(y, p, kind="regression", primary="r2", seed=1, rounds=400)
    assert s.ci_low <= s.primary <= s.ci_high


def test_more_data_gives_a_tighter_ci():
    """CI บอกความไม่แน่นอนที่มาจาก **ขนาดของ test set** — ชุดใหญ่ต้องแคบกว่า"""
    rng = np.random.default_rng(0)

    def width(n):
        y = rng.integers(0, 2, size=n)
        p = np.where(rng.random(n) < 0.25, 1 - y, y)
        s = score(y, p, kind="classification", primary="macro_f1", seed=1,
                  labels=[0, 1], rounds=300)
        return s.ci_high - s.ci_low

    assert width(4000) < width(250)


def test_stratified_resampling_keeps_class_counts_exact():
    """**กลไกที่แท้จริงของการ stratify** — จำนวนของแต่ละคลาสต้องไม่แกว่ง

    การสุ่มรวมทำให้จำนวนคลาสน้อยเปลี่ยนไปทุกรอบ ซึ่งเพิ่มความแปรปรวนให้ macro-F1
    ด้วยเหตุผลที่ไม่เกี่ยวกับคุณภาพของโมเดลเลย
    """
    from tabular.metrics import _resample_indices

    y = np.zeros(300, dtype="int64")
    y[:12] = 1                                   # คลาสน้อย 4%
    rng = np.random.default_rng(0)

    strat_counts = {int(y[_resample_indices(rng, y, True)].sum()) for _ in range(50)}
    assert strat_counts == {12}, "แยกตามคลาสแล้วจำนวนต้องคงที่เป๊ะ"

    plain_counts = {int(y[_resample_indices(rng, y, False)].sum()) for _ in range(50)}
    assert len(plain_counts) > 1, "สุ่มรวมแล้วจำนวนต้องแกว่ง"


def test_plain_bootstrap_is_wider_on_an_imbalanced_set():
    """ผลที่ตามมาจากข้อบน — CI ที่กว้างเกินจริงทำให้สองทีมดูแยกกันไม่ออกทั้งที่ต่างกัน"""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    y = np.zeros(300, dtype="int64")
    y[:12] = 1
    p = y.copy()
    p[rng.choice(np.flatnonzero(y == 1), size=4, replace=False)] = 0
    fn = lambda a, b: f1_score(a, b, average="macro", labels=[0, 1], zero_division=0)  # noqa: E731

    lo_p, hi_p = bootstrap_ci(y, p, metric=fn, seed=1, stratified=False, rounds=800)
    lo_s, hi_s = bootstrap_ci(y, p, metric=fn, seed=1, stratified=True, rounds=800)
    assert (hi_p - lo_p) > (hi_s - lo_s)


# ── MAPE ที่ใช้ไม่ได้ต้องบอกว่าใช้ไม่ได้ ────────────────────────────


def test_mape_is_none_when_the_target_touches_zero():
    """คืน `None` ไม่ใช่ inf — ให้ชัดว่า "ใช้ไม่ได้กับโจทย์นี้" ไม่ใช่ "คำนวณพลาด" """
    y = np.array([0.0, 10.0, 20.0])
    s = score(y, y + 1, kind="regression", primary="r2", seed=1, rounds=FAST)
    assert s.reported["mape"] is None


def test_mape_is_reported_when_the_target_is_safely_away_from_zero(reg):
    y, p = reg
    s = score(y, p, kind="regression", primary="r2", seed=1, rounds=FAST)
    assert s.reported["mape"] is not None and s.reported["mape"] > 0


# ── ข้อความผิดพลาด ─────────────────────────────────────────────────


def test_wrong_number_of_predictions_says_both_numbers(clf):
    y, p = clf
    with pytest.raises(MetricError) as exc:
        score(y, p[:-5], kind="classification", primary="macro_f1", seed=1, labels=[0, 1])
    assert str(len(y)) in str(exc.value) and str(len(y) - 5) in str(exc.value)


def test_probabilities_instead_of_labels_says_so(clf):
    """ความผิดพลาดที่พบบ่อยที่สุด — เรียก `predict_proba` แทน `predict`"""
    y, _ = clf
    proba = np.tile([[0.3, 0.7]], (len(y), 1))
    with pytest.raises(MetricError, match="predict_proba"):
        score(y, proba, kind="classification", primary="macro_f1", seed=1, labels=[0, 1])


def test_nan_predictions_point_at_the_likely_cause(clf):
    y, p = clf
    broken = p.astype("float64")
    broken[3] = np.nan
    with pytest.raises(MetricError, match="imputer"):
        score(y, broken, kind="classification", primary="macro_f1", seed=1, labels=[0, 1])


def test_unknown_class_lists_what_is_allowed(clf):
    y, p = clf
    p = p.copy()
    p[0] = 9
    with pytest.raises(MetricError) as exc:
        score(y, p, kind="classification", primary="macro_f1", seed=1, labels=[0, 1])
    assert "[9]" in str(exc.value) and "[0, 1]" in str(exc.value)


def test_classification_without_labels_is_refused(clf):
    y, p = clf
    with pytest.raises(MetricError, match="labels"):
        score(y, p, kind="classification", primary="macro_f1", seed=1)


def test_unsupported_primary_says_what_is_supported(clf, reg):
    y, p = clf
    with pytest.raises(MetricError, match="predict_proba"):
        score(y, p, kind="classification", primary="roc_auc", seed=1, labels=[0, 1])

    y, p = reg
    with pytest.raises(MetricError, match="r2, neg_rmse, neg_mae"):
        score(y, p, kind="regression", primary="rmse", seed=1)


def test_unknown_kind_lists_what_exists(clf):
    y, p = clf
    with pytest.raises(MetricError, match="classification, regression"):
        score(y, p, kind="clustering", primary="macro_f1", seed=1)


# ── สิ่งที่รายงานให้นิสิต ──────────────────────────────────────────


def test_confusion_matrix_has_a_fixed_label_order_even_if_a_class_is_never_predicted():
    """ลำดับคลาสต้องมาจากโจทย์ ไม่ใช่จากสิ่งที่โมเดลบังเอิญทาย

    ถ้าปล่อยให้ลำดับขึ้นกับผลทำนาย ตารางของสองทีมจะอ่านเทียบกันไม่ได้
    """
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.zeros(6, dtype="int64")            # ทายคลาส 0 อย่างเดียว
    s = score(y, p, kind="classification", primary="macro_f1", seed=1,
              labels=[0, 1, 2], rounds=FAST)
    assert s.reported["labels"] == ["0", "1", "2"]
    assert len(s.reported["confusion_matrix"]) == 3
    assert set(s.reported["per_class_f1"]) == {"0", "1", "2"}


def test_as_dict_flattens_everything_for_storage(clf):
    y, p = clf
    d = score(y, p, kind="classification", primary="macro_f1", seed=1,
              labels=[0, 1], rounds=FAST).as_dict()
    assert d["primary_name"] == "macro_f1"
    assert {"primary", "ci_low", "ci_high", "accuracy", "confusion_matrix"} <= set(d)


def test_default_bootstrap_rounds_is_what_the_spec_says():
    assert BOOTSTRAP_ROUNDS == 1000
