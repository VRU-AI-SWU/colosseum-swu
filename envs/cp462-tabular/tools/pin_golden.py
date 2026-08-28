"""ตรึงค่า golden ที่ `selfcheck` ใช้เทียบ — `tabular/golden.json`

    python tools/pin_golden.py --check     ค่าที่ตรึงไว้ยังตรงกับที่วัดได้ไหม
    python tools/pin_golden.py --write     เขียนทับด้วยค่าที่วัดได้ตอนนี้

**`--write` คือการประกาศว่าโจทย์เปลี่ยนไปแล้ว** ไม่ใช่การซ่อมค่าที่ไม่ตรง
ถ้า `--check` ไม่ผ่านทั้งที่ไม่ได้ตั้งใจแก้อะไร แปลว่ามีอย่างอื่นเปลี่ยน
(เวอร์ชันของ numpy/sklearn) — ต้องหาสาเหตุก่อน ไม่ใช่เขียนทับ

ทุกค่าในนี้วัดจาก **ชุดของนิสิตล้วน** ไม่แตะชุดที่ใช้ตัดสิน เครื่องมือนี้จึงรันได้
โดยไม่ต้องมี `ARENA_SECRETS` — ซึ่งเป็นสิ่งเดียวกับที่ทำให้ `selfcheck` รันบน
เครื่องนิสิตได้
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabular import __version__  # noqa: E402
from tabular.config import CONFIG_DIR, load  # noqa: E402
from tabular.dataset import all_parts  # noqa: E402
from tabular.generator import fingerprint  # noqa: E402
from tabular.metrics import score  # noqa: E402
from tabular.selfcheck import GOLDEN_PATH, SCORE_TOLERANCE, _reference_pipeline  # noqa: E402

SLUGS = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))

COMMENT = (
    "ค่าที่ `python -m tabular.selfcheck` ใช้เทียบ — สร้างด้วย tools/pin_golden.py "
    "อย่าแก้ด้วยมือ · ทุกค่าวัดจากชุดของนิสิตล้วน ไม่มีอะไรจากชุดที่ใช้ตัดสิน"
)


def measure() -> dict:
    tasks = {}
    for slug in SLUGS:
        spec = load(slug)
        parts = all_parts(spec)
        trivial, strong = _reference_pipeline(spec.kind)

        baselines = {}
        for name, pipe in (("trivial", trivial), ("strong", strong)):
            pipe.fit(parts.train.X, parts.train.y)
            baselines[name] = round(
                score(
                    parts.test.y, pipe.predict(parts.test.X),
                    kind=spec.kind, primary=spec.primary,
                    seed=spec.bootstrap_seed, labels=spec.labels or None,
                ).primary,
                6,
            )

        tasks[slug] = {
            "config_hash": spec.config_hash,
            "sizes": parts.sizes(),
            "train": fingerprint(parts.train),
            "val": fingerprint(parts.val),
            "test": fingerprint(parts.test),
            "baselines": baselines,
        }
    return {
        "_comment": COMMENT,
        "env_version": __version__,
        "pickle_runtime": {"scikit-learn": _sklearn_minor()},
        "tasks": tasks,
    }


def _sklearn_minor() -> str:
    import sklearn

    return ".".join(sklearn.__version__.split(".")[:2])


def report(got: dict, want: dict) -> bool:
    ok = True
    for key in ("env_version", "pickle_runtime"):
        if got[key] != want.get(key):
            print(f"  ✗ {key}: ได้ {got[key]} ตรึงไว้ {want.get(key)}")
            ok = False

    for slug, values in got["tasks"].items():
        pinned = want.get("tasks", {}).get(slug)
        if pinned is None:
            print(f"  ✗ {slug}: ไม่มีในค่าที่ตรึงไว้")
            ok = False
            continue
        for field in ("config_hash", "sizes", "train", "val", "test"):
            if values[field] != pinned.get(field):
                print(f"  ✗ {slug}.{field}: ได้ {values[field]} ตรึงไว้ {pinned.get(field)}")
                ok = False
        for name, value in values["baselines"].items():
            pinned_value = pinned.get("baselines", {}).get(name)
            if pinned_value is None or abs(value - pinned_value) > SCORE_TOLERANCE:
                print(f"  ✗ {slug}.baselines.{name}: ได้ {value} ตรึงไว้ {pinned_value}")
                ok = False
        if ok:
            print(f"  ✓ {slug}: {values['sizes']} · "
                  + " · ".join(f"{k} {v:+.4f}" for k, v in values["baselines"].items()))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="เทียบกับค่าที่ตรึงไว้")
    group.add_argument("--write", action="store_true", help="เขียนทับด้วยค่าที่วัดได้ตอนนี้")
    args = parser.parse_args()

    got = measure()

    if args.write:
        GOLDEN_PATH.write_text(
            json.dumps(got, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"เขียน {GOLDEN_PATH} แล้ว")
        report(got, got)
        return 0

    if not GOLDEN_PATH.is_file():
        print(f"✗ ไม่พบ {GOLDEN_PATH}", file=sys.stderr)
        return 1
    want = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if report(got, want):
        print("\n✓ ค่าที่ตรึงไว้ยังตรงกับที่วัดได้")
        return 0
    print("\n✗ ไม่ตรง — หาสาเหตุก่อนจะ --write ทับ", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
