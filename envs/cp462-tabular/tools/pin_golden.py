#!/usr/bin/env python3
"""ตรึงค่าอ้างอิงที่ `tabular.selfcheck` ใช้เทียบ

    python tools/pin_golden.py            # ดูว่าจะเขียนอะไร
    python tools/pin_golden.py --write     # เขียนจริง

⚠️ **รันบนเครื่องที่มีเวอร์ชันเดียวกับ container ของ grader เท่านั้น** — ค่าที่
ตรึงจากเครื่องอื่นจะทำให้ selfcheck ของนิสิตทุกคนไม่ผ่าน ทั้งที่เครื่องเขาถูกต้อง

**สิ่งที่ตรึงเป็นชุดตรวจคงที่ ไม่ใช่ข้อมูลของโจทย์** (`selfcheck.PROBE`) — ข้อมูล
จริงของแต่ละ competition เป็นไฟล์ในคลังบนเซิร์ฟเวอร์ ซึ่งเครื่องนิสิตไม่มีและไม่ควรมี
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tabular import __version__  # noqa: E402
from tabular.metrics import score  # noqa: E402
from tabular.selfcheck import PROBE, GOLDEN_PATH, _reference_pipeline, probe_split  # noqa: E402
from tabular.splits import PARTS, as_frame  # noqa: E402
from tabular.table import fingerprint  # noqa: E402


def build() -> dict:
    import sklearn

    split = probe_split()
    trivial, strong = _reference_pipeline("classification")

    baselines = {}
    for name, pipe in (("trivial", trivial), ("strong", strong)):
        pipe.fit(split.student.X, split.student.y)
        baselines[name] = round(
            score(
                split.test_public.y, pipe.predict(split.test_public.X),
                kind="classification", primary="macro_f1",
                seed=PROBE["split_seed"], labels=[0, 1],
            ).primary,
            8,
        )

    return {
        "env_version": __version__,
        # minor ของ sklearn คือสิ่งที่ตัดสินว่า pickle ของนิสิตโหลดได้ไหม
        "pickle_runtime": {"scikit-learn": ".".join(sklearn.__version__.split(".")[:2])},
        "probe": {
            "spec": dict(PROBE),
            "sizes": split.sizes(),
            "fingerprints": {n: fingerprint(as_frame(getattr(split, n))) for n in PARTS},
            "baselines": baselines,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="เขียนทับ golden.json")
    args = ap.parse_args()

    golden = build()
    text = json.dumps(golden, indent=2, ensure_ascii=False) + "\n"
    if not args.write:
        print(text)
        print(f"— ยังไม่ได้เขียน · ใส่ --write เพื่อเขียนลง {GOLDEN_PATH}", file=sys.stderr)
        return 0

    GOLDEN_PATH.write_text(text, encoding="utf-8")
    print(f"เขียน {GOLDEN_PATH} แล้ว")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
