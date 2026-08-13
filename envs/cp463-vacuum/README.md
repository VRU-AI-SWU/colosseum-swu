# cp463-vacuum

Environment `vacuum_gridworld` v1.0.0 — โจทย์ [Competition 1 ของ CP463 1/2026](../../docs/competitions/CP463/1-2026/vacuum-robot/overview.md)

implement ตาม [environment-spec.md](../../docs/competitions/CP463/1-2026/vacuum-robot/environment-spec.md)
ซึ่งเป็นเอกสารที่ถือว่าเป็นตัวจริงเมื่อขัดกับเอกสารอื่น

> **environment ตัวนี้คือตัวเดียวกับที่ใช้ตัดสินคะแนน** — ถ้า conformance test (§14) ผ่านทั้งชุด
> แปลว่าเครื่องของคุณให้ผลตรงกับ grader ทุกบิต

## ติดตั้ง

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
```

เวอร์ชันของ `numpy` และ `gymnasium` ถูก pin ไว้และเป็น **load-bearing** — `numpy.random.Generator`
ไม่การันตี stream ข้ามเวอร์ชัน ถ้าอัพ numpy ผังห้องของ seed เดิมจะเปลี่ยน

## ใช้งาน

```python
from vacuum import load_config, VacuumEnv
from vacuum.baselines import BFSCoverageAgent
from vacuum.rollout import agent_config, evaluate

config = load_config("configs/main.yaml")
score, results = evaluate(config, BFSCoverageAgent, range(1, 31))
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
| `baselines/` | §10 | Bronze / Silver / Gold |
| `rollout.py` | — | ตัวรันในเครื่อง (**ห้ามใช้เป็นตัวรันของ grader** — ดูหัวไฟล์) |

## ทดสอบ

```bash
pytest -q
```

`tests/test_conformance.py` คือ §14 ทั้งชุด ใช้ seed ย่าน 70001+ ซึ่งแยกจาก
train (1–9999) · public (สุ่มจาก 20000–29999) · private (สุ่มจาก 50000–59999)
ค่า seed ของ public/private เป็นความลับ อยู่ที่ repo `colosseum-hypogeum` ไม่ใช่ที่นี่

## สถานะ

✅ ค่าใน `configs/*.yaml` **ผ่านการ calibrate รอบที่ 1 แล้ว** (ส.ค. 2026) —
[รายงาน](../../docs/competitions/CP463/1-2026/vacuum-robot/calibration-2026-08.md) ·
รันซ้ำด้วย `python examples/calibrate.py --seeds 30`

⚠️ **สิ่งที่ยังค้าง** — การทดลองที่ 1 ของ §15 ยังตอบไม่ครบ เพราะยังไม่มี learned policy มาเทียบ
ถ้าเทรน PPO เสร็จแล้วให้รัน `--policy module:Class` เพื่อยืนยันว่าชนะ Gold baseline ได้จริง
ก่อนตรึงค่า `sensor_noise` ถาวร
