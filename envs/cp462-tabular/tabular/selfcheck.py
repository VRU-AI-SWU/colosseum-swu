"""ตรวจว่าเครื่องนี้ให้ผลตรงกับตัวที่ใช้ตัดสินคะแนน

    python -m tabular.selfcheck

**นี่คือสิ่งที่รับประกันว่าคะแนนที่วัดเองเทียบกับ leaderboard ได้ ไม่ใช่เลขเวอร์ชัน**
บทเรียนตรงกับ `cp463-vacuum` — การตรึงเวอร์ชัน numpy ไม่ได้รับประกันอะไร เพราะ
stream ของตัวสุ่มเปลี่ยนได้ภายใน minor version · ตัวที่จับได้จริงคือการเทียบ
ลายนิ้วมือของข้อมูลและคะแนน baseline กับค่าที่ตรึงไว้

ไม่ต้องใช้ pytest — นิสิตรันคำสั่งเดียวแล้วอ่านผลได้ทันที
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.json"

#: คะแนน baseline ยอมต่างได้เล็กน้อย — จุดทศนิยมของ float ต่างกันได้ระหว่าง BLAS
#: คนละตัว · แต่ **ลายนิ้วมือของข้อมูลต้องตรงเป๊ะ** ไม่มีการยอมให้ต่าง
SCORE_TOLERANCE = 1e-4

GREEN, RED, YELLOW, DIM, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _line(ok: bool | None, title: str, detail: str = "") -> None:
    mark = {True: f"{GREEN}✓{OFF}", False: f"{RED}✗{OFF}", None: f"{YELLOW}!{OFF}"}[ok]
    print(f"  {mark} {title}" + (f" {DIM}{detail}{OFF}" if detail else ""))


def check_versions() -> bool:
    """**รายงานอย่างเดียว ไม่ตัดสิน** — ตัวตัดสินคือคะแนน baseline ข้างล่าง

    เวอร์ชันที่ไม่ตรงกับที่เคยวัดไม่ได้แปลว่าพัง และเวอร์ชันที่ตรงก็ไม่ได้แปลว่าถูก
    """
    import numpy, pandas, sklearn  # noqa: E401

    _line(True, "เวอร์ชันของ dependency",
          f"numpy {numpy.__version__} · pandas {pandas.__version__} · sklearn {sklearn.__version__}")
    return True


def check_data(golden: dict) -> bool:
    """ข้อมูลที่เครื่องนี้สร้างได้ ต้องเหมือนของ grader ทุกบิต"""
    from tabular.config import load
    from tabular.dataset import all_parts
    from tabular.generator import fingerprint

    ok = True
    for slug, want in golden["tasks"].items():
        spec = load(slug)
        if spec.config_hash != want["config_hash"]:
            _line(False, f"config ของ {slug}",
                  f"hash ไม่ตรง — ไฟล์ configs/{slug}.yaml ถูกแก้")
            ok = False
            continue

        parts = all_parts(spec)
        sizes = parts.sizes()
        if sizes != want["sizes"]:
            _line(False, f"ขนาดชุดข้อมูลของ {slug}", f"ได้ {sizes} ควรเป็น {want['sizes']}")
            ok = False
            continue

        got = {"train": fingerprint(parts.train), "val": fingerprint(parts.val)}
        bad = [k for k in ("train", "val") if got[k] != want[k]]
        if bad:
            _line(False, f"ข้อมูลของ {slug}",
                  f"{', '.join(bad)} ไม่ตรงกับ grader — {got} ควรเป็น "
                  f"{{'train': {want['train']!r}, 'val': {want['val']!r}}}")
            ok = False
        else:
            _line(True, f"ข้อมูลของ {slug}",
                  f"train {sizes['train']} · val {sizes['val']} · ตรงกับ grader ทุกบิต")
    return ok


def _reference_pipeline(kind: str):
    """pipeline อ้างอิงสำหรับวัด baseline — **ไม่ใช่เฉลย** แค่ของธรรมดาที่ทำซ้ำได้"""
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                              ("s", StandardScaler())]),
             ["tenure_months", "monthly_spend", "support_tickets"]),
            ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                              ("o", OneHotEncoder(handle_unknown="ignore"))]),
             ["plan", "region"]),
        ],
        remainder="drop",
    )
    if kind == "classification":
        return (Pipeline([("pre", pre), ("m", DummyClassifier(strategy="most_frequent"))]),
                Pipeline([("pre", pre), ("m", HistGradientBoostingClassifier(random_state=0))]))
    return (Pipeline([("pre", pre), ("m", DummyRegressor())]),
            Pipeline([("pre", pre), ("m", HistGradientBoostingRegressor(random_state=0))]))


def check_scores(golden: dict) -> bool:
    """**ตัวตัดสินจริง** — คะแนนที่วัดบนเครื่องนี้ต้องตรงกับที่ grader วัดไว้"""
    from tabular.config import load
    from tabular.dataset import all_parts, grading_data
    from tabular.metrics import score

    ok = True
    for slug, want in golden["tasks"].items():
        spec = load(slug)
        parts = all_parts(spec)
        test = grading_data(spec, "public")
        trivial, strong = _reference_pipeline(spec.kind)

        got = {}
        for name, pipe in (("trivial", trivial), ("strong", strong)):
            pipe.fit(parts.train.X, parts.train.y)
            got[name] = score(
                test.y, pipe.predict(test.X), kind=spec.kind, primary=spec.primary,
                seed=spec.bootstrap_seed, labels=spec.labels or None,
            ).primary

        off = {k: (got[k], v) for k, v in want["baselines"].items()
               if abs(got[k] - v) > SCORE_TOLERANCE}
        if off:
            _line(False, f"คะแนน baseline ของ {slug}",
                  " · ".join(f"{k}: ได้ {a:.6f} ควรเป็น {b:.6f}" for k, (a, b) in off.items()))
            ok = False
        else:
            _line(True, f"คะแนน baseline ของ {slug}",
                  " · ".join(f"{k} {v:+.4f}" for k, v in got.items()) + f" ({spec.primary})")
    return ok


def main() -> int:
    if not GOLDEN_PATH.is_file():
        print(f"{RED}✗ ไม่พบ {GOLDEN_PATH}{OFF} — แพ็กเกจติดตั้งไม่ครบ", file=sys.stderr)
        return 1
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    from tabular import __version__

    print(f"tabular {__version__} — ตรวจว่าเครื่องนี้ให้ผลตรงกับตัวที่ใช้ตัดสิน\n")
    if __version__ != golden["env_version"]:
        _line(None, "env_version",
              f"โค้ด {__version__} · golden {golden['env_version']} — คนละรุ่นกัน")

    results = [check_versions(), check_data(golden), check_scores(golden)]

    if all(results):
        print(f"\n{GREEN}✓ ผ่านครบทุกข้อ{OFF} — คะแนนที่คุณวัดเองเทียบกับ leaderboard ได้ตรงๆ")
        return 0

    print(
        f"\n{RED}✗ ไม่ผ่าน{OFF} — **อย่าเพิ่งเริ่มเขียน** ไม่งั้นคุณจะจูนบนสิ่งที่ไม่ตรงกับตอนวัดจริง\n\n"
        "  สาเหตุที่พบบ่อยที่สุดคือเวอร์ชันของ numpy/pandas/scikit-learn ต่างจากที่ทดสอบไว้\n"
        "  ลองติดตั้งใหม่ด้วย pip install --force-reinstall ตามคำสั่งจากหน้า release\n\n"
        f"  {DIM}ถ้ายังไม่ผ่าน แจ้งผู้สอนพร้อมผลของคำสั่งนี้ทั้งหมด — อย่าแก้ golden.json เอง{OFF}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
