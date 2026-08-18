"""ตรวจว่า environment ในเครื่องคุณตรงกับตัวที่ใช้ตัดสินคะแนน

    python -m vacuum.selfcheck

[README §10.4](../../../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries) สัญญาไว้ว่า
นิสิตต้องยืนยันข้อนี้เองได้ — เพราะถ้า environment ในเครื่องต่างจากตัวที่ใช้ตัดสินแม้แต่นิดเดียว
คุณจะเทรนบนสิ่งที่ไม่ตรงกับตอนวัด แล้ว**พังแบบเงียบๆ หาสาเหตุยากมาก**

ไม่ต้องมี pytest — ตั้งใจให้รันได้ทันทีหลัง `pip install`
"""

from __future__ import annotations

import json
import random
import sys

import numpy as np

from vacuum import __version__, phase_config
from vacuum.baselines import BASELINES
from vacuum.config import CONFIG_DIR, PHASES
from vacuum.env import VacuumEnv
from vacuum.generator import generate_layout
from vacuum.rollout import evaluate

GOLDEN_PATH = CONFIG_DIR.parent / "golden_baselines.json"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


class CheckFailed(Exception):
    pass


def _check(name: str, fn) -> bool:
    try:
        detail = fn()
    except CheckFailed as exc:
        print(f"  {RED}✗{RESET} {name}\n      {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"  {RED}✗{RESET} {name}\n      {type(exc).__name__}: {exc}")
        return False
    print(f"  {GREEN}✓{RESET} {name}" + (f" {DIM}{detail}{RESET}" if detail else ""))
    return True


# ── การตรวจแต่ละข้อ ─────────────────────────────────────────────────


def check_versions() -> str:
    import gymnasium

    problems = []
    if not np.__version__.startswith("2.1."):
        problems.append(f"numpy {np.__version__} (ต้องเป็น 2.1.*)")
    if not gymnasium.__version__.startswith("1.3."):
        problems.append(f"gymnasium {gymnasium.__version__} (ต้องเป็น 1.3.*)")
    if problems:
        raise CheckFailed(
            "เวอร์ชันไม่ตรง: " + " · ".join(problems) + "\n"
            "      numpy เป็น load-bearing — Generator stream ไม่การันตีข้ามเวอร์ชัน\n"
            "      ผังห้องของ seed เดิมจะเปลี่ยนไป ให้ `pip install -U cp463-vacuum` ใหม่"
        )
    return f"numpy {np.__version__} · gymnasium {gymnasium.__version__}"


def check_determinism() -> str:
    config = phase_config("main")
    ref = generate_layout(config, 70001)
    for _ in range(20):
        got = generate_layout(config, 70001)
        if not (
            np.array_equal(got.obstacle, ref.obstacle)
            and np.array_equal(got.dirt0, ref.dirt0)
            and got.start == ref.start
        ):
            raise CheckFailed("seed เดียวกันให้ผังห้องคนละแบบ")
    return f"seed 70001 → D0={ref.D0} free={ref.free_count}"


def check_immune_to_global_rng() -> str:
    """โค้ดของคุณ (หรือ library ที่คุณใช้) เรียก `np.random.seed()` แล้วผังห้องต้องไม่เปลี่ยน"""
    config = phase_config("main")
    ref = generate_layout(config, 70001)
    for seed in (0, 42, 12345):
        np.random.seed(seed)
        random.seed(seed)
        if not np.array_equal(generate_layout(config, 70001).obstacle, ref.obstacle):
            raise CheckFailed(
                "ผังห้องเปลี่ยนหลังจากตั้ง global RNG — environment ในเครื่องคุณมีปัญหา"
            )
    return "np.random.seed / random.seed ไม่มีผล"


def check_reward_is_zero() -> str:
    env = VacuumEnv(phase_config("main"))
    obs, _ = env.reset(seed=70001)
    rng = np.random.Generator(np.random.PCG64(0))
    for _ in range(50):
        _obs, reward, term, trunc, _info = env.step(int(rng.integers(0, 6)))
        if reward != 0.0:
            raise CheckFailed(f"env.step() คืน reward {reward} — ต้องเป็น 0.0 เสมอ")
        if term or trunc:
            break
    return "reward = 0.0 ทุก step (คุณออกแบบ reward เองตอนเทรน)"


def check_observation_shapes() -> str:
    shapes = {}
    for phase in PHASES:
        env = VacuumEnv(phase_config(phase))
        obs, _ = env.reset(seed=70001)
        if obs["grid"].dtype != np.float32:
            raise CheckFailed(f"{phase}: grid dtype = {obs['grid'].dtype} ต้องเป็น float32")
        shapes[phase] = tuple(obs["grid"].shape)
    return " · ".join(f"{k} {v}" for k, v in shapes.items())


def check_golden_scores() -> str:
    """**ข้อที่สำคัญที่สุด** — คะแนนของ baseline ในเครื่องคุณต้องตรงกับที่ grader ใช้เป๊ะ"""
    if not GOLDEN_PATH.exists():
        raise CheckFailed(f"ไม่พบ {GOLDEN_PATH.name} ในแพ็กเกจ")
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    seeds = [int(s) for s in golden["seeds"]]

    mismatches = []
    for phase, entry in golden["phases"].items():
        config = phase_config(phase)
        if config.config_hash != entry["config_hash"]:
            mismatches.append(f"{phase}: config_hash ไม่ตรง")
            continue
        for level, expected in entry["scores"].items():
            summary, _ = evaluate(config, BASELINES[level], seeds)
            if abs(summary.score - expected["score"]) > 1e-9:
                mismatches.append(
                    f"{phase}/{level}: ได้ {summary.score:.6f} ควรเป็น {expected['score']:.6f}"
                )
    if mismatches:
        raise CheckFailed(
            "\n      ".join(mismatches)
            + "\n      → environment ในเครื่องคุณให้ผลไม่ตรงกับตัวที่ใช้ตัดสิน"
        )
    n = sum(len(e["scores"]) for e in golden["phases"].values())
    return f"{n} ค่า × {len(seeds)} seed ตรงทุกตัว"


CHECKS = [
    ("เวอร์ชันของ dependency", check_versions),
    ("ผังห้องเหมือนเดิมทุกครั้งที่ seed เดิม", check_determinism),
    ("ไม่ถูกรบกวนโดย global RNG", check_immune_to_global_rng),
    ("reward เป็น 0 เสมอ", check_reward_is_zero),
    ("รูปร่างของ observation", check_observation_shapes),
    ("คะแนน baseline ตรงกับ grader", check_golden_scores),
]


def main() -> int:
    print(f"\ncp463-vacuum {__version__} — ตรวจว่า environment ตรงกับตัวที่ใช้ตัดสิน\n")
    failed = [name for name, fn in CHECKS if not _check(name, fn)]

    if failed:
        print(
            f"\n{RED}✗ ไม่ผ่าน {len(failed)} ข้อ{RESET} — คะแนนที่คุณวัดในเครื่องอาจไม่ตรงกับบน leaderboard\n"
            f"  ถ้าแก้เองไม่ได้ ให้แจ้งผู้สอนพร้อมผลที่พิมพ์ออกมาข้างบน\n"
        )
        return 1

    print(
        f"\n{GREEN}✓ ผ่านครบทุกข้อ{RESET} — environment ในเครื่องคุณให้ผลตรงกับตัวที่ใช้ตัดสิน\n"
        f"  {DIM}คะแนนที่ได้จาก `arena eval --local` จึงเทียบกับ leaderboard ได้ตรงๆ{RESET}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
