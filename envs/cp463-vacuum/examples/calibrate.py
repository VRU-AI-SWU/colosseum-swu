"""การทดลอง calibrate ตาม environment-spec §15

⚠️ ต้องทำก่อนเปิดเทอมและก่อนงานอื่นทั้งหมด เพราะผลอาจทำให้ต้องกลับไปแก้ §11

    การทดลองที่ 1 — learned policy ชนะ planner ได้ที่ระดับความยากไหน
    การทดลองที่ 2 — `max_steps` พอให้ดูดครบไหม (เกณฑ์: Gold completed 60–90% ของ seed)
    การทดลองที่ 3 — คะแนน baseline ทั้ง 4 ระดับห่างกันเกินความกว้างของ CI ไหม

**การทดลองที่ 1 sweep `sensor_noise` ไม่ใช่ `action_noise`** — รอบแรก (ส.ค. 2026) วัดแล้วว่า
`action_noise` กด planner ไม่ลงเลย (0 → 0.50 คะแนน Gold ตกแค่ 9% และยังดูดครบทุก seed)
เพราะมันทำให้แค่ *เส้นทาง* ยาวขึ้น ไม่ได้ทำให้ *ความรู้เกี่ยวกับโลก* ผิด
`sensor_noise` ต่างออกไปตรงที่มันทำให้แผนที่สะสมผิดถาวร — จึงเป็นคันโยกจริง
(รายละเอียด: docs/competitions/CP463/1-2026/vacuum-robot/calibration-2026-08.md)

    python examples/calibrate.py --seeds 30
    PPO_MODEL=models/ppo_main.zip python examples/calibrate.py --seeds 30 \
        --policy examples.ppo_agent:PPOAgent
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np

from vacuum import load_config
from vacuum.baselines import BASELINES, INSTRUCTOR_LEVELS
from vacuum.rollout import evaluate

from vacuum.config import CONFIG_DIR

# seed ของการทดลอง — ใช้ช่วงของตัวเอง ไม่ทับกับ train/public/private/conformance
CALIBRATION_SEED_BASE = 80001

# ระดับ sensor_noise ที่ sweep — ค่าที่ใช้จริงใน Main คือ 0.02
# ที่ 0.05 เป็นหน้าผา (Gold ร่วงจาก 1.81 เหลือ 0.75) จึงเป็นขอบบนของช่วงที่มีความหมาย
SENSOR_NOISE_LEVELS = (0.0, 0.01, 0.02, 0.03, 0.05)
BOOTSTRAP_N = 2000


def bootstrap_ci(scores: np.ndarray, n: int = BOOTSTRAP_N) -> tuple[float, float]:
    """CI ของค่าเฉลี่ย โดย resample **ที่ระดับ seed** (template §7.2)

    timestep ในหนึ่ง episode ไม่เป็นอิสระต่อกัน จึง resample ที่ระดับ seed เท่านั้น
    """
    rng = np.random.Generator(np.random.PCG64(12345))  # ตรึงไว้ → CI ของทุกทีมเทียบกันได้
    idx = rng.integers(0, len(scores), size=(n, len(scores)))
    means = scores[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def require_instructor_baselines() -> None:
    """การทดลอง 1–3 เทียบกับ Gold/Diamond ซึ่งโค้ดอยู่ฝั่งผู้สอน

    สคริปต์นี้ถูกแพ็กไปกับ starter kit เพื่อให้นิสิต*อ่านวิธีวัด*ได้ แต่รันเองไม่ครบ
    เพราะหมุดที่ใช้เทียบเป็นของลับ (README §10.4)
    """
    missing = [lv for lv in INSTRUCTOR_LEVELS if lv not in BASELINES]
    if missing:
        raise SystemExit(
            f"ต้องมี baseline ฝั่งผู้สอน ({', '.join(missing)}) ถึงจะรันการทดลองนี้ได้\n"
            "ตั้ง ARENA_SECRETS ให้ชี้ไปที่ clone ของ colosseum-hypogeum ก่อน\n"
            "นิสิตอ่านสคริปต์นี้เพื่อดูวิธีวัดได้ แต่รันไม่ได้ — หมุดที่ใช้เทียบไม่ได้แจก"
        )


def run(config, level: str, seeds: list[int]) -> dict:
    t0 = time.perf_counter()
    summary, results = evaluate(config, BASELINES[level] if level in BASELINES else level, seeds)
    scores = np.array([r.breakdown.score for r in results], dtype=np.float64)
    lo, hi = bootstrap_ci(scores)
    completed = [r.breakdown for r in results if r.breakdown.completed]
    return {
        "score": summary.score,
        "ci": [lo, hi],
        "completion_rate": len(completed) / len(results),
        "mean_coverage": summary.mean_coverage,
        "mean_t_end_completed": summary.mean_t_end_completed,
        "p90_t_end_completed": (
            float(np.percentile([b.t_end for b in completed], 90)) if completed else None
        ),
        "worst": summary.worst_episode,
        "sd": summary.sd_across_seeds,
        "seconds": time.perf_counter() - t0,
    }


def experiment_1(seeds: list[int], policy) -> dict:
    """learned policy ชนะ planner ได้ที่ระดับความยากไหน — sweep `sensor_noise` บน config Main

    สิ่งที่อยากเห็น: มีช่วงของ `sensor_noise` ที่ policy ชนะ Gold **โดยที่คะแนนยังไม่ร่วงทั้งคู่**
    ถ้าไม่มีช่วงนั้นเลย แปลว่าคันโยกนี้ก็ไม่ตอบโจทย์เหมือน `action_noise` และต้องกลับไป
    พิจารณาทางเลือก C+D ของ template §5
    """
    base = load_config(CONFIG_DIR / "main.yaml")
    out: dict[str, dict] = {}
    for noise in SENSOR_NOISE_LEVELS:
        cfg = base.replace(**{"dynamics.sensor_noise": noise})
        entry = {"gold": run(cfg, "gold", seeds), "silver": run(cfg, "silver", seeds)}
        if policy is not None:
            entry["policy"] = run(cfg, policy, seeds)
        out[f"{noise:.3f}"] = entry
    return out


def experiment_2(seeds: list[int]) -> dict:
    """`max_steps` พอให้ดูดครบไหม — เกณฑ์ที่ต้องการ: Gold completed 60–90% ของ seed"""
    out = {}
    for phase in ("warmup", "main", "final"):
        cfg = load_config(CONFIG_DIR / f"{phase}.yaml")
        res = run(cfg, "gold", seeds)
        rate = res["completion_rate"]
        res["verdict"] = (
            "ok"
            if 0.60 <= rate <= 0.90
            else ("max_steps น้อยเกิน — completion_bonus แทบไม่ทำงาน" if rate < 0.60 else "max_steps มากเกิน — แยกทีมไม่ออก")
        )
        res["max_steps"] = cfg.episode.max_steps
        out[phase] = res
    return out


def experiment_3(seeds: list[int]) -> dict:
    """คะแนน baseline ห่างกันเกินความกว้างของ CI ไหม — ถ้าไม่ เส้นแบ่งเกรดจะไม่มีความหมาย"""
    out = {}
    for phase in ("warmup", "main", "final"):
        cfg = load_config(CONFIG_DIR / f"{phase}.yaml")
        levels = {level: run(cfg, level, seeds) for level in BASELINES}
        gaps = {}
        order = [lv for lv in ("bronze", "silver", "gold", "diamond") if lv in BASELINES]
        for lo_name, hi_name in zip(order, order[1:]):
            lo, hi = levels[lo_name], levels[hi_name]
            gaps[f"{lo_name}→{hi_name}"] = {
                "gap": hi["score"] - lo["score"],
                "separated": hi["ci"][0] > lo["ci"][1],
            }
        out[phase] = {"levels": levels, "gaps": gaps}
    return out


def fmt(report: dict) -> str:
    lines = ["# ผลการ calibrate (environment-spec §15)", ""]
    lines.append(f"seeds: {report['n_seeds']} ตัว (ช่วง {CALIBRATION_SEED_BASE}+) · "
                 f"bootstrap {BOOTSTRAP_N} รอบ resample ที่ระดับ seed")
    lines.append("")

    lines += ["## การทดลองที่ 1 — learned policy ชนะ planner ได้ที่ระดับความยากไหน", "",
              "| sensor_noise | Silver | Gold (planner) | PPO (learned) | ใครชนะ |",
              "|---|---|---|---|---|"]
    for noise, entry in report["experiment_1"].items():
        g, s = entry["gold"], entry["silver"]
        p = entry.get("policy")
        if p is None:
            policy_cell, verdict = "ยังไม่มี", "—"
        else:
            policy_cell = f"{p['score']:.4f} [{p['ci'][0]:.3f}, {p['ci'][1]:.3f}] · ครบ {p['completion_rate']*100:.0f}%"
            if p["ci"][0] > g["ci"][1]:
                verdict = "**PPO ชนะชัด**"
            elif g["ci"][0] > p["ci"][1]:
                verdict = "Gold ชนะชัด"
            else:
                verdict = "CI ทับกัน แยกไม่ออก"
        lines.append(
            f"| {noise} | {s['score']:.4f} | "
            f"{g['score']:.4f} [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}] · ครบ {g['completion_rate']*100:.0f}% | "
            f"{policy_cell} | {verdict} |"
        )
    lines.append("")

    lines += ["## การทดลองที่ 2 — max_steps พอให้ดูดครบไหม (เป้า 60–90%)", "",
              "| phase | max_steps | Gold completed | t_end เฉลี่ย | t_end p90 | coverage | สรุป |",
              "|---|---|---|---|---|---|---|"]
    for phase, r in report["experiment_2"].items():
        t_end = r["mean_t_end_completed"]
        p90 = r["p90_t_end_completed"]
        lines.append(
            f"| {phase} | {r['max_steps']} | {r['completion_rate']*100:.0f}% | "
            f"{'—' if t_end is None else f'{t_end:.0f}'} | {'—' if p90 is None else f'{p90:.0f}'} | "
            f"{r['mean_coverage']:.3f} | {r['verdict']} |"
        )
    lines.append("")

    lines += ["## การทดลองที่ 3 — baseline ห่างกันพอไหม", ""]
    for phase, r in report["experiment_3"].items():
        lines += [f"### {phase}", "", "| ระดับ | score | 95% CI | completed |", "|---|---|---|---|"]
        for level, v in r["levels"].items():
            lines.append(
                f"| {level} | {v['score']:.4f} | [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | "
                f"{v['completion_rate']*100:.0f}% |"
            )
        lines.append("")
        for pair, g in r["gaps"].items():
            mark = "✅ แยกได้" if g["separated"] else "⚠️ CI ทับกัน"
            lines.append(f"- {pair}: ห่าง {g['gap']:+.4f} — {mark}")
        lines.append("")
    return "\n".join(lines)


def load_policy(spec: str | None):
    if not spec:
        return None
    module, _, cls = spec.partition(":")
    return getattr(importlib.import_module(module), cls)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--policy", default=None, help="module:Class ของ learned policy (การทดลองที่ 1)")
    parser.add_argument("--out", default="calibration-report.md")
    args = parser.parse_args()

    require_instructor_baselines()

    seeds = list(range(CALIBRATION_SEED_BASE, CALIBRATION_SEED_BASE + args.seeds))
    policy = load_policy(args.policy)

    report = {
        "n_seeds": len(seeds),
        "experiment_1": experiment_1(seeds, policy),
        "experiment_2": experiment_2(seeds),
        "experiment_3": experiment_3(seeds),
    }

    Path(args.out).write_text(fmt(report), encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(fmt(report))


if __name__ == "__main__":
    main()
