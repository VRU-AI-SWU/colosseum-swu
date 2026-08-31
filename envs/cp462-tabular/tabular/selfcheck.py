"""ตรวจว่าเครื่องนี้ให้ผลตรงกับตัวที่ใช้ตัดสินคะแนน

    python -m tabular.selfcheck
    python -m tabular.selfcheck --data data.csv    # ตรวจไฟล์ที่ดาวน์โหลดมาด้วย

**นี่คือสิ่งที่รับประกันว่าคะแนนที่วัดเองเทียบกับ leaderboard ได้ ไม่ใช่เลขเวอร์ชัน**
บทเรียนตรงกับ `cp463-vacuum` — การตรึงเวอร์ชัน numpy ไม่ได้รับประกันอะไร เพราะ
stream ของตัวสุ่มเปลี่ยนได้ภายใน minor version · ตัวที่จับได้จริงคือการเทียบผล
ที่คำนวณจริงกับค่าที่ตรึงไว้

---

⚠️ **สิ่งที่ตรวจเปลี่ยนไปแล้ว** — เดิมตรวจว่า "ข้อมูลที่เครื่องนี้ *สร้าง* ตรงกับ
ของ grader ไหม" ซึ่งใช้ได้ตอนที่นิสิตสร้างข้อมูลเองจากเมล็ดที่แจก · ตอนนี้ข้อมูล
เป็นไฟล์ที่ดาวน์โหลด ไม่มีอะไรให้สร้าง คำถามจึงเปลี่ยนเป็น

    "เครื่องนี้ *คิดเลข* เหมือน grader ไหม"

ซึ่งตรวจด้วย **ชุดตรวจคงที่** (probe) ที่ฝังมากับแพ็กเกจ ไม่ใช่ข้อมูลของโจทย์ใด
โจทย์หนึ่ง · ดีกว่าเดิมสองอย่าง: ใช้ได้กับทุก competition ไม่ต้องแก้ตามโจทย์ และ
มันตรวจ *เครื่องจักร* (การแบ่ง · การนับ · bootstrap) ซึ่งเป็นสิ่งที่พังจริงเวลา
เวอร์ชันไม่ตรง ไม่ใช่ตรวจตัวเลขชุดเดียวที่บังเอิญตรงกันได้

ไม่ต้องใช้ pytest — นิสิตรันคำสั่งเดียวแล้วอ่านผลได้ทันที
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.json"

#: คะแนน baseline ยอมต่างได้เล็กน้อย — จุดทศนิยมของ float ต่างกันได้ระหว่าง BLAS
#: คนละตัว · แต่ **ลายนิ้วมือของข้อมูลต้องตรงเป๊ะ** ไม่มีการยอมให้ต่าง
SCORE_TOLERANCE = 1e-4

#: ชุดตรวจคงที่ — ตัวเลขพวกนี้เป็นแค่ "ข้อมูลตัวอย่างที่ทำซ้ำได้" ไม่ใช่ข้อมูลของโจทย์
#: **ห้ามเปลี่ยนโดยไม่ pin golden ใหม่** เพราะทุกค่าใน golden.json ผูกกับมัน
PROBE = {"task": "churn", "seed": 20260101, "n": 3000,
         "split_seed": 7, "student_ratio": 0.7, "grading_public_ratio": 0.5}

GREEN, RED, YELLOW, DIM, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _line(ok: bool | None, title: str, detail: str = "") -> None:
    mark = {True: f"{GREEN}✓{OFF}", False: f"{RED}✗{OFF}", None: f"{YELLOW}!{OFF}"}[ok]
    print(f"  {mark} {title}" + (f" {DIM}{detail}{OFF}" if detail else ""))


def probe_dataset():
    """ข้อมูลของชุดตรวจ — สร้างจากตัวสร้างที่ฝังมากับแพ็กเกจ"""
    from tabular.generator import make

    return make(PROBE["task"], seed=PROBE["seed"], n=PROBE["n"])


def probe_split():
    """สามกองของชุดตรวจ — เดินทางเดียวกับที่ grader ใช้แบ่งข้อมูลจริง"""
    from tabular.splits import three_way

    return three_way(
        probe_dataset(),
        kind="classification",
        seed=PROBE["split_seed"],
        student_ratio=PROBE["student_ratio"],
        grading_public_ratio=PROBE["grading_public_ratio"],
    )


def check_versions() -> bool:
    """**รายงานอย่างเดียว ไม่ตัดสิน** — ตัวตัดสินคือคะแนน baseline ข้างล่าง

    เวอร์ชันที่ไม่ตรงกับที่เคยวัดไม่ได้แปลว่าพัง และเวอร์ชันที่ตรงก็ไม่ได้แปลว่าถูก
    """
    import numpy, pandas, sklearn  # noqa: E401

    _line(True, "เวอร์ชันของ dependency",
          f"numpy {numpy.__version__} · pandas {pandas.__version__} · sklearn {sklearn.__version__}")
    return True


def check_pickle_runtime(golden: dict) -> bool:
    """**ข้อที่ตัดสินว่า `pipeline.pkl` ของคุณโหลดได้บน colosseum หรือไม่**

    ต่างจาก `check_versions` ข้างบนที่รายงานเฉยๆ — ข้อนี้ตัดสิน เพราะสิ่งที่คุณส่ง
    ไม่ใช่ซอร์สโค้ดแต่เป็น **โมเดลที่ fit แล้ว** · pickle ที่สร้างด้วย scikit-learn
    คนละ minor กับตัวที่อยู่ใน container จะโหลดแล้วเตือน (`InconsistentVersionWarning`)
    ในกรณีที่ดี และล้มด้วย `AttributeError` ที่หาสาเหตุยากในกรณีที่แย่
    """
    want = golden.get("pickle_runtime") or {}
    if not want:
        return True

    import sklearn

    got = ".".join(sklearn.__version__.split(".")[:2])
    expected = want["scikit-learn"]
    if got != expected:
        _line(False, "scikit-learn ที่ใช้สร้าง pipeline.pkl",
              f"เครื่องนี้ {sklearn.__version__} · colosseum ใช้ {expected}.x — "
              f"pickle ข้าม minor โหลดไม่ได้")
        return False
    _line(True, "scikit-learn ที่ใช้สร้าง pipeline.pkl",
          f"{sklearn.__version__} · ตรงกับที่ colosseum ใช้โหลดโมเดลของคุณ")
    return True


def check_split(golden: dict) -> bool:
    """การแบ่งข้อมูลบนเครื่องนี้ต้องเหมือนของ grader ทุกบิต

    ข้อนี้จับ stream ของ `numpy.random.Generator` ที่เปลี่ยนข้ามเวอร์ชัน — ซึ่งเป็น
    ความต่างที่มองไม่เห็นจากเลขเวอร์ชัน แต่ทำให้ทุกกองเลื่อนไปคนละแถว
    """
    from tabular.splits import PARTS, as_frame
    from tabular.table import fingerprint

    want = golden["probe"]
    split = probe_split()

    sizes = split.sizes()
    if sizes != want["sizes"]:
        _line(False, "ขนาดของสามกองในชุดตรวจ", f"ได้ {sizes} ควรเป็น {want['sizes']}")
        return False

    got = {name: fingerprint(as_frame(getattr(split, name))) for name in PARTS}
    bad = [k for k in got if got[k] != want["fingerprints"][k]]
    if bad:
        _line(False, "การแบ่งข้อมูลของชุดตรวจ",
              f"{', '.join(bad)} ไม่ตรงกับ grader — ตัวสุ่มของ numpy บนเครื่องนี้ให้ผลคนละแบบ")
        return False
    _line(True, "การแบ่งข้อมูลของชุดตรวจ",
          " · ".join(f"{k} {v}" for k, v in sizes.items()) + " · ตรงกับ grader ทุกบิต")
    return True


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
    """**ตัวตัดสินจริง** — คะแนนที่วัดบนเครื่องนี้ต้องตรงกับที่ grader วัดไว้

    ครอบทั้งการนับ (macro-F1 · R²) และ bootstrap ของช่วงความเชื่อมั่น ซึ่งเป็น
    สองส่วนที่ผลต่างกันได้เงียบๆ เมื่อเวอร์ชันของ numpy/sklearn ไม่ตรง
    """
    from tabular.metrics import score

    want = golden["probe"]["baselines"]
    split = probe_split()
    trivial, strong = _reference_pipeline("classification")

    got = {}
    for name, pipe in (("trivial", trivial), ("strong", strong)):
        pipe.fit(split.student.X, split.student.y)
        got[name] = score(
            split.test_public.y, pipe.predict(split.test_public.X),
            kind="classification", primary="macro_f1",
            seed=PROBE["split_seed"], labels=[0, 1],
        ).primary

    off = {k: (got[k], v) for k, v in want.items() if abs(got[k] - v) > SCORE_TOLERANCE}
    if off:
        _line(False, "คะแนน baseline ของชุดตรวจ",
              " · ".join(f"{k}: ได้ {a:.6f} ควรเป็น {b:.6f}" for k, (a, b) in off.items()))
        return False
    _line(True, "คะแนน baseline ของชุดตรวจ",
          " · ".join(f"{k} {v:+.4f}" for k, v in got.items()) + " (macro_f1)")
    return True


def check_data_file(path: Path) -> bool:
    """รายงานลายนิ้วมือของไฟล์ที่ดาวน์โหลดมา — **เทียบกับที่หน้าโจทย์บอกไว้เอง**

    ระบบยืนยันให้ไม่ได้ว่าไฟล์นี้ใช่ของ competition ไหน เพราะแพ็กเกจนี้ไม่รู้จัก
    competition · สิ่งที่ทำได้คือบอกค่าที่ตรวจสอบได้ แล้วให้เทียบกับหน้าโจทย์
    """
    import hashlib

    import pandas as pd

    if not path.is_file():
        _line(False, "ไฟล์ข้อมูล", f"ไม่พบ {path}")
        return False
    blob = path.read_bytes()
    digest = "sha256:" + hashlib.sha256(blob).hexdigest()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        _line(False, "ไฟล์ข้อมูล", f"อ่านเป็น CSV ไม่ได้ — {exc}")
        return False
    _line(True, f"ไฟล์ {path.name}",
          f"{len(frame)} แถว · {len(frame.columns)} คอลัมน์ · {digest[:23]}…")
    print(f"    {DIM}เทียบเลขนี้กับที่หน้าโจทย์บอกไว้ — ถ้าไม่ตรง แปลว่าโหลดผิดไฟล์หรือไฟล์เสีย{OFF}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ตรวจว่าเครื่องนี้ให้ผลตรงกับตัวที่ใช้ตัดสิน")
    ap.add_argument("--data", type=Path, help="ไฟล์ CSV ที่ดาวน์โหลดมาจากหน้าโจทย์")
    args = ap.parse_args(argv)

    if not GOLDEN_PATH.is_file():
        print(f"{RED}✗ ไม่พบ {GOLDEN_PATH}{OFF} — แพ็กเกจติดตั้งไม่ครบ", file=sys.stderr)
        return 1
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    from tabular import __version__

    print(f"tabular {__version__} — ตรวจว่าเครื่องนี้ให้ผลตรงกับตัวที่ใช้ตัดสิน\n")
    if __version__ != golden["env_version"]:
        _line(None, "env_version",
              f"โค้ด {__version__} · golden {golden['env_version']} — คนละรุ่นกัน")

    results = [
        check_versions(),
        check_pickle_runtime(golden),
        check_split(golden),
        check_scores(golden),
    ]
    if args.data is not None:
        results.append(check_data_file(args.data))

    if all(results):
        print(f"\n{GREEN}✓ ผ่านครบทุกข้อ{OFF} — คะแนนที่คุณวัดเองเทียบกับ leaderboard ได้ตรงๆ")
        return 0

    sklearn_pin = (golden.get("pickle_runtime") or {}).get("scikit-learn")
    sklearn_hint = (
        f"  ถ้าข้อที่ไม่ผ่านคือ scikit-learn: pip install 'scikit-learn=={sklearn_pin}.*'\n\n"
        if sklearn_pin
        else ""
    )
    print(
        f"\n{RED}✗ ไม่ผ่าน{OFF} — **อย่าเพิ่งเริ่มเขียน** ไม่งั้นคุณจะจูนบนสิ่งที่ไม่ตรงกับตอนวัดจริง\n\n"
        "  สาเหตุที่พบบ่อยที่สุดคือเวอร์ชันของ numpy/pandas/scikit-learn ต่างจากที่ทดสอบไว้\n"
        "  ลองติดตั้งใหม่ด้วย pip install --force-reinstall ตามคำสั่งจากหน้า release\n\n"
        f"{sklearn_hint}"
        f"  {DIM}ถ้ายังไม่ผ่าน แจ้งผู้สอนพร้อมผลของคำสั่งนี้ทั้งหมด — อย่าแก้ golden.json เอง{OFF}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
