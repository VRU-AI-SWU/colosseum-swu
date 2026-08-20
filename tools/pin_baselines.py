"""ตรึงคะแนน baseline บน **public seeds** — ค่าที่ไปเป็นหมุดบน leaderboard

ทำไมต้องมีสคริปต์นี้แยกจาก `envs/cp463-vacuum/tests/make_golden.py`
    make_golden รันบน**ชุด conformance** (seed สาธารณะ ไม่ลับ) เพื่อจับ regression ของ
    environment — นิสิตรันเองได้ ค่าที่ได้จึงเทียบกับ leaderboard **ไม่ได้**
    สคริปต์นี้รันบน **public seeds ชุดจริง** ซึ่งเป็นชุดเดียวกับที่ใช้ให้คะแนนนิสิตระหว่างเทอม
    ค่าที่ออกมาคือหมุดที่ [README §6.2](../README.md) สั่งให้ **ตรึงไว้ทั้งเทอม**

ผลลัพธ์ปลอดภัยที่จะอยู่ใน repo สาธารณะ — มันมีแต่*คะแนน* ไม่มีค่า seed
[README §10.4](../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries) ระบุชัดว่า golden
value ของ baseline **ไม่ลับ** เพราะเราโชว์บน leaderboard อยู่แล้ว

    export ARENA_SECRETS=/path/to/colosseum-hypogeum
    python tools/pin_baselines.py                  # เขียนไฟล์หมุด
    python tools/pin_baselines.py --check          # ตรวจว่าค่าที่ตรึงไว้ยังตรง (ใช้ใน CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from runners.seeds import expected_config_hash, load_seeds  # noqa: E402
from vacuum import __version__ as env_version  # noqa: E402
from vacuum import load_config  # noqa: E402
from vacuum.baselines import BASELINES, INSTRUCTOR_LEVELS  # noqa: E402
from vacuum.config import CONFIG_DIR  # noqa: E402
from vacuum.rollout import evaluate  # noqa: E402

SLUG = "cp463-vacuum-1-2026"
PHASES = ("warmup", "main", "final")
LADDER = ("bronze", "silver", "gold", "diamond")
OUT = REPO / "core" / "baseline_pins" / f"{SLUG}.json"

#: resample ที่ระดับ seed — timestep ในหนึ่ง episode ไม่เป็นอิสระต่อกัน
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 12345  # ตรึงไว้ → CI ที่รายงานซ้ำได้เป๊ะ


def paired_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """CI 95% ของผลต่าง `b − a` แบบ paired ต่อ seed

    paired เพราะทั้งสอง agent เจอ **ห้องชุดเดียวกัน** — การ resample แยกกันจะทิ้ง
    ข้อมูลนั้นไปแล้วได้ CI ที่กว้างเกินจริง
    """
    diff = b - a
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP_N, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def measure() -> dict:
    missing = [lv for lv in LADDER if lv not in BASELINES]
    if missing:
        raise SystemExit(
            f"ไม่มี baseline: {', '.join(missing)}\n"
            "ตั้ง ARENA_SECRETS ให้ชี้ไปที่ clone ของ colosseum-hypogeum ก่อน\n"
            f"({', '.join(INSTRUCTOR_LEVELS)} มีโค้ดอยู่ที่นั่นเท่านั้น)"
        )

    phases: dict[str, dict] = {}
    for phase in PHASES:
        config = load_config(CONFIG_DIR / f"{phase}.yaml")
        seeds = load_seeds(competition_slug=SLUG, phase=phase, kind="public")

        pinned_hash = expected_config_hash(competition_slug=SLUG, phase=phase)
        if pinned_hash and pinned_hash != config.config_hash:
            raise SystemExit(
                f"{phase}: config_hash ไม่ตรงกับตอนที่ generate seed\n"
                f"  seeds.yaml : {pinned_hash}\n"
                f"  config ตอนนี้: {config.config_hash}\n"
                "คะแนนข้าม hash เอามาเทียบกันไม่ได้ — ต้อง generate seed ใหม่หรือย้อน config"
            )

        print(f"\n{phase}  ({len(seeds)} public seeds · {config.config_hash})")
        levels: dict[str, dict] = {}
        per_seed: dict[str, np.ndarray] = {}
        for level in LADDER:
            summary, results = evaluate(config, BASELINES[level], seeds)
            per_seed[level] = np.array([r.breakdown.score for r in results], dtype=np.float64)
            levels[level] = {
                "score": round(summary.score, 6),
                "n_completed": summary.n_completed,
                "n_seeds": len(seeds),
            }
            print(f"  {level:<8} {summary.score:.6f}  ครบ {summary.n_completed}/{len(seeds)}")

        gaps = {}
        for lo, hi in zip(LADDER, LADDER[1:]):
            ci = paired_ci(per_seed[lo], per_seed[hi])
            gap = float(per_seed[hi].mean() - per_seed[lo].mean())
            separable = ci[0] > 0
            # Warm-up ตั้ง sensor_noise: 0 ไว้ → ตัวกรองของ Diamond เป็น no-op และต้อง
            # ได้เท่ากับ Gold **ทุกหลัก** นี่คือการตรวจในตัวว่าตัวกรองไม่ได้ทำอะไรเกินจำเป็น
            # (overview §6) ไม่ใช่หมุดที่แยกกันไม่ออก จึงไม่นับเป็นปัญหา
            by_design = phase == "warmup" and (lo, hi) == ("gold", "diamond") and gap == 0.0
            gaps[f"{lo}->{hi}"] = {
                "gap": round(gap, 6),
                "ci95": [round(ci[0], 6), round(ci[1], 6)],
                "separable": separable,
                **({"identical_by_design": True} if by_design else {}),
            }
            flag = "✅ เท่ากันตามที่ออกแบบ" if by_design else ("✅" if separable else "⚠️ ทับกัน")
            print(f"    {lo:>7} → {hi:<8} {gap:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]  {flag}")

        phases[phase] = {"config_hash": config.config_hash, "levels": levels, "gaps": gaps}

    return {
        "competition": SLUG,
        "env_version": env_version,
        "seed_set": "public",
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "paired": True},
        "note": (
            "คะแนนบน public seeds ชุดจริง — ตรึงทั้งเทอม (README §6.2) "
            "ไฟล์นี้ไม่มีค่า seed มีแต่คะแนน จึงอยู่ใน repo สาธารณะได้ (README §10.4)"
        ),
        "phases": phases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="เทียบกับไฟล์เดิม ไม่เขียนทับ")
    args = parser.parse_args()

    report = measure()

    unstable = [
        f"{phase}/{name}"
        for phase, entry in report["phases"].items()
        for name, gap in entry["gaps"].items()
        if not gap["separable"] and not gap.get("identical_by_design")
    ]
    if unstable:
        print(f"\n⚠️ ช่องว่างที่แยกกันไม่ได้ทางสถิติ: {', '.join(unstable)}")
        print("   ใช้เป็นเส้นแบ่งเกรดไม่ได้ — ต้องเพิ่มจำนวน seed หรือปรับ config")

    if args.check:
        if not OUT.exists():
            print(f"\n✗ ยังไม่มี {OUT.relative_to(REPO)} — รันโดยไม่ใส่ --check ก่อน")
            return 1
        old = json.loads(OUT.read_text(encoding="utf-8"))
        if old["phases"] == report["phases"]:
            print(f"\n✓ ค่าที่ตรึงไว้ยังตรงกับที่วัดได้")
            return 0
        print(f"\n✗ ค่าที่วัดได้ไม่ตรงกับ {OUT.relative_to(REPO)}")
        for phase in PHASES:
            for level in LADDER:
                a = old["phases"].get(phase, {}).get("levels", {}).get(level, {}).get("score")
                b = report["phases"][phase]["levels"][level]["score"]
                if a != b:
                    print(f"   {phase}/{level}: ตรึงไว้ {a} → วัดได้ {b}")
        print("\n   ถ้าตั้งใจเปลี่ยน environment ต้องขึ้น env_version และ rejudge ทุก submission")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nเขียน {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
