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


#: เวอร์ชันที่ทดสอบแล้วว่าให้ผลตรงกับ grader — ใช้บอกใบ้ตอนคะแนนไม่ตรง ไม่ใช่กฎบังคับ
TESTED_NUMPY = ("2.1", "2.5")


def check_versions() -> str:
    """รายงานเวอร์ชัน — **ไม่ตัดสินจากเลขเวอร์ชัน**

    เดิมข้อนี้บังคับ `numpy 2.1.*` แล้ว raise ถ้าไม่ตรง ซึ่งผิดสองทาง

    1. **เข้มเกินจริง** — วัดแล้วว่า numpy 2.5.2 ให้ผังห้องแฮชเดียวกันและคะแนนตรงกัน
       ถึงทศนิยมที่ 12 การปฏิเสธมันคือการบล็อกนิสิตที่ใช้ Python 3.14
       (ซึ่ง numpy 2.1 ไม่มี wheel ให้) โดยไม่มีเหตุผลที่วัดได้รองรับ
    2. **หลวมเกินจริงพร้อมกัน** — เลขเวอร์ชันไม่ได้รับประกันอะไรเลย numpy อาจเปลี่ยน
       stream ภายใน 2.1.x ก็ได้ การผ่านข้อนี้จึงไม่เคยแปลว่าปลอดภัย

    ตัวที่ตัดสินจริงคือ **`check_golden`** ซึ่งเทียบคะแนน baseline กับค่าที่บันทึกไว้ —
    มันจับการเปลี่ยน stream ได้ทุกกรณีไม่ว่ามาจากเวอร์ชันไหน
    """
    import gymnasium

    note = ""
    if not np.__version__.startswith(TESTED_NUMPY):
        note = f" {YELLOW}(ยังไม่ได้ทดสอบกับรุ่นนี้ — ดูข้อสุดท้าย){RESET}"
    return f"numpy {np.__version__} · gymnasium {gymnasium.__version__}{note}"


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
            + "\n"
            + f"\n      สาเหตุที่พบบ่อยที่สุดคือ numpy (ตอนนี้ {np.__version__})"
            + " เปลี่ยน stream ของตัวสุ่ม"
            + "\n      ลงรุ่นที่ทดสอบแล้วว่าตรงกับ grader:"
            + '\n          pip install "numpy==2.5.*"      # หรือ 2.1.* ถ้า Python ต่ำกว่า 3.14'
            + "\n      แล้วรัน `python -m vacuum.selfcheck` ใหม่"
        )
    n = sum(len(e["scores"]) for e in golden["phases"].values())
    return f"{n} ค่า × {len(seeds)} seed ตรงทุกตัว (bronze/silver ทั้ง 3 phase)"


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
        f"  {DIM}คะแนนที่ได้จาก `arena eval` จึงเทียบกับ leaderboard ได้ตรงๆ{RESET}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
