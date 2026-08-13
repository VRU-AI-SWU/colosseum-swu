"""generate ค่า golden ของ baseline สำหรับ conformance test #11

    python tests/make_golden.py

⚠️ ต้องรันใหม่ **ทุกครั้งที่ config เปลี่ยน** และเมื่อค่าเปลี่ยน = ขึ้น `env_version` + rejudge
ทุก submission (environment-spec §14)

seed ที่ใช้อยู่ในย่านของ conformance test (70001+) ซึ่งเปิดเผยได้ เพราะแยกจาก
train (1–9999) · public (สุ่มจาก 20000–29999) · private (สุ่มจาก 50000–59999) — README §10.4
ค่าที่สุ่มได้จริงของ public/private อยู่ที่ repo colosseum-hypogeum ไม่ใช่ที่นี่
"""

from __future__ import annotations

import json
from pathlib import Path

from vacuum import __version__, load_config
from vacuum.baselines import BASELINES
from vacuum.rollout import evaluate

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE.parent / "configs"
OUT = HERE / "golden_baselines.json"

SEEDS = list(range(70001, 70031))


def main() -> None:
    report = {
        "env_version": __version__,
        "seeds": SEEDS,
        "note": (
            "ค่าชั่วคราวก่อนการ calibrate (§15) — ต้อง generate ใหม่หลังตรึงค่า config"
        ),
        "phases": {},
    }

    for phase in ("warmup", "main", "final"):
        config = load_config(CONFIG_DIR / f"{phase}.yaml")
        scores = {}
        for level, cls in BASELINES.items():
            summary, _ = evaluate(config, cls, SEEDS)
            scores[level] = {
                "score": summary.score,
                "n_completed": summary.n_completed,
                "mean_coverage": summary.mean_coverage,
            }
            print(f"{phase:7s} {level:7s} score={summary.score:.6f} "
                  f"completed={summary.n_completed}/{len(SEEDS)}")
        report["phases"][phase] = {"config_hash": config.config_hash, "scores": scores}

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nเขียน {OUT}")


if __name__ == "__main__":
    main()
