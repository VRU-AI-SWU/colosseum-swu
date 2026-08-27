# cp463-vacuum

Environment `vacuum_gridworld` v1.0.0 — โจทย์ [Competition 1 ของ CP463 1/2026](../../docs/competitions/CP463/1-2026/vacuum-robot/overview.md)

implement ตาม [environment-spec.md](../../docs/competitions/CP463/1-2026/vacuum-robot/environment-spec.md)
ซึ่งเป็นเอกสารที่ถือว่าเป็นตัวจริงเมื่อขัดกับเอกสารอื่น

> **environment ตัวนี้คือตัวเดียวกับที่ใช้ตัดสินคะแนน** — ถ้า conformance test (§14) ผ่านทั้งชุด
> แปลว่าเครื่องของคุณให้ผลตรงกับ grader ทุกบิต

## ติดตั้ง

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
```

เวอร์ชันของ `numpy` และ `gymnasium` ถูก pin ไว้และเป็น **load-bearing** — `numpy.random.Generator`
ไม่การันตี stream ข้ามเวอร์ชัน ถ้าอัพ numpy ผังห้องของ seed เดิมจะเปลี่ยน

## ใช้งาน

```python
from vacuum import phase_config, VacuumEnv
from vacuum.baselines import BASELINES
from vacuum.rollout import evaluate

config = phase_config("main")      # config ถูกแพ็กมากับแพ็กเกจ
score, results = evaluate(config, BASELINES["silver"], range(1, 31))
print(score.score, score.n_completed)
```

`env.reset(seed=...)` **ต้องระบุ seed เสมอ** — environment ไม่สุ่ม seed ให้เอง

## โครงสร้าง

| ไฟล์ | spec | หน้าที่ |
|---|---|---|
| `config.py` | §12 | โหลด/validate YAML, `config_hash` |
| `generator.py` | §2, §3 | สายสุ่ม 3 สาย, noise tape, อัลกอริทึมสร้างห้อง |
| `observation.py` | §4 | encoding ทั้ง 3 โหมด |
| `env.py` | §5, §6, §8 | transition, termination, Gymnasium API |
| `scoring.py` | §7 | Coverage AUC + completion bonus — **grader ใช้ไฟล์นี้ตัวเดียวกัน** |
| `replay.py` | §9 | รูปแบบ `.vrp` (header + delta 4 ไบต์/timestep) |
| `viewer.py` | §9 | `.vrp` → หน้าเว็บไฟล์เดียว (`python -m vacuum.viewer replays/1.vrp`) · decode ฝั่ง Python แล้วฝัง event ลง HTML **ไม่มีตัวแตก zstd ฝั่ง JS** |
| `baselines/` | §10 | Bronze / Silver / Gold / Diamond — Gold กับ Diamond ต่างกันแค่ "กรอง noise หรือไม่" |
| `rollout.py` | — | ตัวรันในเครื่อง (**ห้ามใช้เป็นตัวรันของ grader** — ดูหัวไฟล์) |

## starter kit ที่แจกนิสิต

```bash
pip install cp463-vacuum colosseum   # ของจริงใช้ URL จากหน้า release
python -m vacuum.selfcheck           # ยืนยันว่า environment ตรงกับ grader
arena init --dir my-agent            # คัดลอก starter kit ออกมา
cd my-agent                          # ← `arena eval` อ่านจากโฟลเดอร์ปัจจุบัน
arena eval --config main --seeds 1-20
```

ของที่แพ็กไปกับ wheel: `vacuum/configs/` (config ทั้ง 3 phase) · `vacuum/golden_baselines.json` ·
`vacuum/starter/` (agent เปล่า + README ของนิสิต + SOURCES.md) · `vacuum/selfcheck.py`

`selfcheck` คือสิ่งที่ [README §10.4](../../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries)
สัญญาไว้ว่านิสิตต้องยืนยันเองได้ — มันเทียบคะแนน baseline ทั้ง 12 ค่ากับ golden
และตรวจว่าเวอร์ชัน numpy/gymnasium ตรง (numpy เป็น load-bearing) โดยไม่ต้องมี pytest

## เทรน policy เอง

```bash
uv pip install -e ".[train]"
python examples/train_ppo.py --phase main --steps 3000000
```

| ไฟล์ | หน้าที่ |
|---|---|
| `examples/reward_wrappers.py` | ตัวอย่าง reward 2 แบบ — **reward ตอนเทรน ≠ metric ตอนตัดสิน** |
| `examples/map_memory.py` | แผนที่สะสม + feature ให้ policy · **ใช้ไฟล์เดียวกันทั้งตอนเทรนและตอน inference** |
| `examples/train_ppo.py` | PPO (SB3) — เทรนบน training seeds `1–9999` เท่านั้น |
| `examples/ppo_agent.py` | ห่อ policy ให้เป็น `Agent` ตาม interface เดียวกับ baseline |

> observation ที่ environment ให้คือหน้าต่างรอบตัว (POMDP) — policy แบบไม่มีความจำ
> ทำ coverage ไม่ได้โดยหลักการ `map_memory.py` แก้ด้วยการแยก **state estimation** ออกมา
> แล้วให้ policy ตัดสินใจบน belief แทน ซึ่งเป็นสิ่งที่ Silver/Gold ทำอยู่แล้ว
> (อีกทางคือใช้ recurrent policy ให้มันเรียนรู้ที่จะจำเอง — ช้ากว่ามากเมื่อ episode ยาว 1,500 step)

## ทดสอบ

```bash
pytest -q
```

`tests/test_conformance.py` คือ §14 ทั้งชุด ใช้ seed ย่าน 70001+ ซึ่งแยกจาก
train (1–9999) · public (สุ่มจาก 20000–29999) · private (สุ่มจาก 50000–59999)
ค่า seed ของ public/private เป็นความลับ อยู่ที่ repo `colosseum-hypogeum` ไม่ใช่ที่นี่

## สถานะ

✅ ค่าใน `vacuum/configs/*.yaml` **ผ่านการ calibrate 2 รอบแล้ว** (ส.ค. 2026) —
[รายงาน](../../docs/competitions/CP463/1-2026/vacuum-robot/calibration-2026-08.md) ·
รันซ้ำด้วย `python examples/calibrate.py --seeds 30`

การทดลองทั้ง 3 ข้อของ §15 ตอบครบแล้ว รวมถึงฝั่ง learned policy (PPO แพ้ planner ทุกระดับ)

⚠️ **สิ่งที่ยังค้าง** — คะแนน baseline ที่ตรึงไว้ตอนนี้มาจากชุด conformance
ค่าที่จะใช้เป็น **เส้นแบ่งเกรด** ต้องรันบน public seeds อีกครั้งหลังมี runner จริง
