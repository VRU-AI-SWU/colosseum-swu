# CP463 · Competition 1 — Vacuum Cleaner Grid World (RL)

> เป็นส่วนหนึ่งของ [term project 1/2026](../term_project.md) — คู่กับ [Competition 2 (Tool-use Agent)](../intelligence-document/overview.md)
> ครอบคลุมเนื้อหา **คาบ 1–5** (MDP → Q-learning → policy gradient → PPO) ใช้สัปดาห์ที่ 1–7
>
> 📐 **รายละเอียดระดับ implement อยู่ที่ [environment-spec.md](environment-spec.md)** — observation encoding, transition ทุกกรณีขอบ,
> อัลกอริทึม generator, replay format, pseudocode ของ baseline, conformance test เอกสารนี้เป็นภาพรวมและเหตุผลเชิงออกแบบ

โจทย์ course project ที่ใช้จริงในวิชา **CP463 (Artificial Intelligence)** ภาคเรียนที่ **1/2026**
รันบน [Arena platform](../../../../../README.md) — เอกสารนี้เป็น **spec ของโจทย์** ส่วน README หลักเป็น spec ของแพลตฟอร์ม
ใช้ template [agent-vs-environment-rl](../../../../task-templates/agent-vs-environment-rl.md) — กติกาที่ใช้ร่วมกับโจทย์ RL อื่นอยู่ที่นั่น เอกสารนี้ระบุเฉพาะส่วนที่เป็นของ CP463

| | |
|---|---|
| **Course** | CP463 — Artificial Intelligence |
| **Term** | 1/2026 |
| **Competition slug** | `cp463-vacuum-1-2026` |
| **ช่วงเวลา** | สัปดาห์ 1–7 |
| **เนื้อหาที่รองรับ** | คาบ 1 (MDP) · คาบ 2 (TD/Q-learning) · คาบ 3 (policy gradient, actor-critic, GAE) · คาบ 4 (on/off-policy, reward shaping) · คาบ 5 (PPO) |
| **Evaluation mode** | Agent vs Environment ([README §3](../../../../../README.md#3-โหมดการแข่งขัน-evaluation-modes)) |
| **Execution model** | Hosted Run บน runner on-prem ([README §4](../../../../../README.md#4-รูปแบบการรัน-execution-models)) |
| **Compute ที่ใช้ตัดสิน** | CPU lane (policy เล็ก ไม่ต้องใช้ GPU) |
| **สถานะ** | ร่าง — ยังต้องทดสอบ config กับ baseline ก่อนเปิดเทอม (ดู §11) |

---

## สารบัญ

1. [สรุปโจทย์](#1-สรุปโจทย์)
2. [Environment Spec](#2-environment-spec)
3. [Configuration](#3-configuration)
4. [Agent Interface และการแพ็กไฟล์ส่ง](#4-agent-interface-และการแพ็กไฟล์ส่ง)
5. [การให้คะแนน](#5-การให้คะแนน)
6. [Baseline Ladder](#6-baseline-ladder)
7. [Seeds และ Public / Private Split](#7-seeds-และ-public--private-split)
8. [แผนตามสัปดาห์ (Phases)](#8-แผนตามสัปดาห์-phases)
9. [Starter Kit](#9-starter-kit)
10. [นโยบายการ Train](#10-นโยบายการ-train)
11. [สิ่งที่ต้องตัดสินใจ/ทดสอบก่อนเปิดเทอม](#11-สิ่งที่ต้องตัดสินใจทดสอบก่อนเปิดเทอม)

---

## 1. สรุปโจทย์

หุ่นยนต์ดูดฝุ่นทำงานในห้องจำลอง 2D grid world หนึ่งห้อง มีวัตถุ (เฟอร์นิเจอร์/สิ่งกีดขวาง) วางอยู่แบบสุ่ม
เป้าหมาย: **ดูดให้ครบพื้นที่ โดยใช้จำนวน decision timestep น้อยที่สุด**

> **นิยาม decision timestep** — 1 timestep = การเรียก `agent.act()` 1 ครั้ง (คือหุ่นตัดสินใจทำ action 1 ครั้ง)
> ตัววัดทั้งหมดของโจทย์นี้นับเป็น timestep ล้วน **ไม่มีการนำเวลาจริง (wall-clock) มาคิดคะแนน** เพราะนิสิตใช้เครื่องสเปคต่างกัน
> agent ที่คิดนานแต่เดินน้อย จึงได้เปรียบ agent ที่คิดเร็วแต่เดินมั่ว — ซึ่งตรงกับสิ่งที่เราอยากสอน (คุณภาพของการวางแผน ไม่ใช่ความแรงของเครื่อง)

---

## 2. Environment Spec

- **Observation** — ผู้สอนเลือกได้ระหว่าง
  - `full` : เห็นทั้ง grid (ง่าย เหมาะกับช่วง warm-up)
  - `local` : เห็นเฉพาะหน้าต่างรอบตัว k×k + ตำแหน่งตัวเอง (ท้าทายกว่า ต้องจำแผนที่เอง)
  - `sensor` : เห็นเฉพาะช่องข้างเคียง 4 ทิศ + สถานะสกปรก (ยากสุด ใกล้เคียงหุ่นจริง)
- **Action space** — `UP / DOWN / LEFT / RIGHT / SUCK / IDLE` (discrete)
- **Termination** — ดูดครบทุกช่อง (จบทันที) หรือครบ `max_steps` หรือแบตหมด
  ทุก action นับเป็น 1 timestep เท่ากันหมด รวมถึง `IDLE` และการเดินชนกำแพง → การอยู่เฉยหรือเดินเสียเปล่าถูกลงโทษโดยอัตโนมัติ
- **Determinism** — ผังห้อง ตำแหน่งเริ่มต้น และการกระจายฝุ่น ถูกสุ่มจาก `seed` ทั้งหมด → seed เดียวกันได้ห้องเดียวกันเสมอ ตรวจซ้ำได้
- **Stochasticity** — ตั้งแต่ phase Main เป็นต้นไป การเคลื่อนที่มีโอกาสพลาดทิศ (`action_noise`) และบางพื้นผิวดูดไม่ติดในครั้งแรก (`sticky_dirt`)
  ความสุ่มนี้ถูกตรึงด้วย seed เช่นกัน → seed เดียวกันได้ผลลัพธ์เดียวกันเสมอ

> **ทำไมต้องมี stochasticity** — ในห้องที่ deterministic และรู้กติกาแน่นอน **BFS/frontier planner ชนะ RL อย่างขาดลอย**
> ถ้าปล่อยไว้แบบนั้น ทีมที่เขียน planner มือจะชนะทีมที่ตั้งใจทำ RL ทั้งเทอม ซึ่งขัดกับวัตถุประสงค์ของวิชา
> การใส่ความไม่แน่นอนทำให้ planner ต้อง replan ตลอดและทำได้แย่ลง ขณะที่วิธีที่หลักสูตรสอน (value estimation ภายใต้ความไม่แน่นอน,
> policy gradient, GAE) กลายเป็นคำตอบที่เป็นธรรมชาติ — รายละเอียดของประเด็นนี้อยู่ที่
> [template §5](../../../../task-templates/agent-vs-environment-rl.md#5-ปัญหาที่ต้องตัดสินใจ-planning-ชนะ-learning)

---

## 3. Configuration

ผู้สอนแก้ค่าเหล่านี้ได้ผ่านหน้าเว็บหรือไฟล์ YAML ต่อ competition

```yaml
task: vacuum_gridworld
version: 1.0.0

room:
  width: 20
  height: 20
  obstacle_density: 0.15        # สัดส่วนช่องที่เป็นสิ่งกีดขวาง
  obstacle_generator: clustered # random | clustered | rooms | fixed
  fixed_layout: null            # ใส่ผังตายตัวได้ถ้าต้องการ
  dirt_distribution: uniform    # uniform | clustered | patchy
  dirt_ratio: 0.6               # สัดส่วนช่องว่างที่สกปรก
  guarantee_connected: true     # การันตีว่าทุกช่องสกปรกไปถึงได้

robot:
  start: random                 # random | corner | center
  observation: local
  observation_window: 5
  battery: 500                  # null = ไม่จำกัด
  move_cost: 1
  suck_cost: 2

dynamics:                       # ตัวที่ทำให้ planning อ่อนลงและ learning คุ้มขึ้น
  action_noise: 0.10            # โอกาสเดินพลาดไปทิศข้างเคียง (0 = deterministic)
  sticky_dirt: 0.15             # สัดส่วนช่องที่ต้อง SUCK สองครั้ง และไม่มีอะไรบอกล่วงหน้า
  sensor_noise: 0.0             # โอกาสที่ observation ผิดพลาด (ใช้ใน phase Final)

episode:
  max_steps: 1000               # จำนวน decision timestep สูงสุดต่อ episode
  stop_on_full_coverage: true   # จบทันทีเมื่อดูดครบ → ยิ่งจบที่ timestep น้อย ยิ่งได้เปรียบ
  step_timeout_ms: 1000         # ตัวกันงานค้างเท่านั้น "ไม่มีผลต่อคะแนน"
                                #   เกินเวลานี้ = ถือว่า agent ล้มเหลว ไม่ใช่ "ได้คะแนนน้อยลง"

penalties:                      # นับเป็นจำนวนครั้ง ไม่เกี่ยวกับเวลา
  collision: 1.0                # ชนสิ่งกีดขวาง
  redundant_suck: 0.2           # ดูดช่องที่สะอาดแล้ว

scoring:
  metric: coverage_auc          # คิดจาก decision timestep ล้วน
  completion_bonus: 1.0         # โบนัสเมื่อดูดครบ 100% (ตั้ง 0 = ใช้ AUC ล้วน)
  max_penalty: 0.2              # เพดานรวมของ penalty กันไม่ให้พลิกลำดับข้ามชั้น

evaluation:
  public_seeds:  [1001, 1002, ..., 1030]   # 30 seed เห็นคะแนนได้ตลอดเทอม
  private_seeds: hidden                      # 100 seed เปิดตอนปิดเทอม
```

---

## 4. Agent Interface และการแพ็กไฟล์ส่ง

```python
class Agent:
    def __init__(self, config: dict): ...
    def reset(self, episode_info: dict) -> None: ...   # เริ่ม episode ใหม่
    def act(self, observation) -> int: ...             # คืน action
```

ส่งเป็น zip/tar ที่มี `agent.py` + `requirements.txt` (จำกัด package ตาม whitelist) + weights
รันบน base image `arena/vacuum:cpu` หรือ `arena/vacuum:cu121`

---

## 5. การให้คะแนน

### 5.1 Metric: Coverage AUC + Completion Bonus

**ตัวแปรที่ใช้คิดคะแนนมีแค่ 2 อย่าง: coverage และจำนวน decision timestep**
ไม่มี wall-clock, ไม่มี FLOPs, ไม่มีขนาดโมเดล เข้ามาเกี่ยวข้อง — เครื่องแรงไม่ได้เปรียบ

ปัญหาของการวัด "coverage สูงสุด" อย่างเดียวคือไม่สนใจว่าใช้กี่ timestep ส่วนการวัด "จำนวน timestep" อย่างเดียวก็ทำให้ agent รีบจบโดยไม่ดูดให้ครบ
วิธีที่รวมทั้งสองอย่างไว้ในตัวเลขเดียวอย่างเป็นธรรมชาติคือ **พื้นที่ใต้กราฟ coverage เทียบกับ timestep**

```
AUC = (1/T) × Σ(t=1..T) coverage_t          # T = max_steps, coverage_t ∈ [0,1]
                                             # ถ้า episode จบก่อน T ให้ถือว่า coverage คงค่าเดิมจนถึง T
```

**ทำไม AUC ถึงตรงกับ "ดูดครบโดยใช้ timestep น้อยที่สุด"** — ถ้า agent ดูดครบที่ timestep `t_c` แล้วจบ ค่า AUC จะออกมาราว `1 − t_c/(2T)`
คือเป็นฟังก์ชันลดตาม `t_c` โดยตรง → **จัดอันดับตามจำนวน timestep ที่ใช้ดูดครบ ให้เองโดยไม่ต้องจูนน้ำหนักใดๆ**
และสำหรับ agent ที่ยังดูดไม่ครบ AUC ก็ยังไล่ระดับได้อย่างต่อเนื่อง (สำคัญมากช่วงต้นเทอมที่ยังไม่มีใครทำครบ — ถ้าใช้ "จำนวน timestep จนดูดครบ" ตรงๆ ทุกทีมจะได้ค่า ∞ เท่ากันหมด ไม่มี gradient ให้ไต่)

**Completion bonus** — AUC ล้วนมีจุดอ่อนคือ agent ที่ดูดได้ 95% เร็วมาก อาจชนะ agent ที่ดูดครบ 100% แต่ช้ากว่า
ซึ่งขัดกับเจตนาของโจทย์ที่ต้องการ "ครบก่อน แล้วค่อยแข่งกันที่ timestep" จึงเพิ่มโบนัสก้อนเดียวเข้าไป

```
episode_score = AUC
              + completion_bonus × 1[coverage สุดท้าย = 100%]
              − min(max_penalty, w_collision × collisions/T + w_redundant × redundant_sucks/T)
```

เนื่องจาก `AUC ≤ 1` และ `completion_bonus = 1.0` โดยมี penalty เพดาน 0.2
→ **ทีมที่ดูดครบจะอยู่เหนือทีมที่ดูดไม่ครบเสมอ** และภายในกลุ่มที่ดูดครบด้วยกัน จะเรียงตามจำนวน timestep ที่ใช้ (น้อยกว่า = สูงกว่า)
เท่ากับได้พฤติกรรมแบบ lexicographic (ครบ → เร็ว) แต่ยังเป็นตัวเลขเดียวที่ต่อเนื่อง ใช้ทำกราฟพัฒนาการได้

### 5.2 การรวมคะแนนและการตัดสินเสมอ

**คะแนนรวมของ submission** = ค่าเฉลี่ยของ `episode_score` ทุก seed
**เกณฑ์ตัดสินเสมอ (ตามลำดับ)**: จำนวน seed ที่ดูดครบ 100% → worst-episode score → จำนวน timestep เฉลี่ยจนดูดครบ → เวลาที่ส่งก่อน

### 5.3 ตัวเลขที่แสดงเพิ่มเติม

ไม่ใช้จัดอันดับ แต่ช่วยให้นิสิตเข้าใจงานตัวเอง

| ตัวเลข | มีผลต่อคะแนน |
|---|---|
| final coverage % เฉลี่ย | ✅ (ผ่าน AUC + bonus) |
| จำนวน timestep เฉลี่ยจนดูดครบ | ✅ (ผ่าน AUC) |
| จำนวน seed ที่ดูดครบ / ทั้งหมด | ✅ (ผ่าน bonus) |
| จำนวนครั้งที่ชน / ดูดซ้ำ | ✅ (ผ่าน penalty) |
| ส่วนเบี่ยงเบนมาตรฐานข้าม seed (ความเสถียร) | ❌ แสดงอย่างเดียว |
| เวลาตัดสินใจเฉลี่ยต่อ timestep (ms) | ❌ **แสดงเพื่อเตือนว่าเสี่ยง timeout เท่านั้น** |

---

## 6. Baseline Ladder

bot ของผู้สอนที่วางไว้บน leaderboard เป็นหมุดหมายให้นิสิตไล่ (กลไกอธิบายไว้ที่ [README §6.2](../../../../../README.md#62-baseline-bot--เป้าหมายระยะสั้นที่จับต้องได้))

| ระดับ | Agent | ความหมาย |
|---|---|---|
| 🥉 Bronze | Random walk + suck | "โค้ดทำงานได้แล้ว" |
| 🥈 Silver | Greedy nearest-dirt | "agent มีกลยุทธ์แล้ว" |
| 🥇 Gold | BFS coverage planner | "ดีกว่าวิธีคลาสสิกที่ไม่ได้เรียนรู้" |
| 💎 Diamond | Solution ของผู้สอน | "ระดับ state-of-the-art ของโจทย์นี้" |

คะแนนของแต่ละระดับต้องได้จากการรันจริงบน public seeds ชุดเดียวกับนิสิต แล้วตรึงค่าไว้ทั้งเทอม
และต้องรันใหม่ทุกครั้งที่เปลี่ยน phase เพราะ config ต่างกันทำให้คะแนนของ baseline ต่างกัน

> **Gold = BFS planner เป็นหมุดหมายที่ตั้งใจ** — ข้อความที่ต้องการสื่อคือ "วิธีดั้งเดิมที่ไม่เรียนรู้ทำได้ถึงแค่นี้เมื่อโลกไม่แน่นอน"
> ทีมที่ส่ง planner มาก็จะไปหยุดอยู่ตรงระดับ Gold พอดี ส่วนการจะแตะ Diamond ต้องใช้ policy ที่เรียนรู้จากประสบการณ์
> ถ้าผลจริงไม่เป็นแบบนี้ แปลว่า `action_noise` ตั้งไว้ต่ำเกินไป (ดู §11 ข้อ 0)

---

## 7. Seeds และ Public / Private Split

| | Public | Private |
|---|---|---|
| จำนวน seed | 30 | 100 |
| เปิดเผย | บอกจำนวน แต่ไม่บอกค่า seed และไม่ให้ผังห้อง | ไม่เปิดเผยจนกว่าจะปิดเทอม |
| ใช้ทำอะไร | leaderboard ระหว่างเทอม + starter kit ให้ทดสอบเองได้ | **ตัดสินอันดับและเกรดจริง** |

- **ห้ามให้ private seeds ทับกับ public เด็ดขาด** และควรสุ่มจากช่วงเลขที่ต่างกันไปเลยเพื่อกันความผิดพลาด
- private seeds เก็บไว้ที่ runner on-prem เท่านั้น ไม่อัพขึ้น cloud
- นโยบายกันการโกงที่เหลือใช้ตามแพลตฟอร์ม ([README §7](../../../../../README.md#7-ความสุจริตทางวิชาการและการกันโกงระบบ))

---

## 8. แผนตามสัปดาห์ (Phases)

**หลักการ: แต่ละ phase ต้องแก้ได้พอดีตอนที่คาบเรียนที่เกี่ยวข้องเพิ่งสอนจบ**
โจทย์จึงกลายเป็นแบบฝึกหัดของคาบนั้นๆ แทนที่จะเป็นก้อนใหญ่ก้อนเดียวที่นิสิตไม่รู้จะเริ่มตรงไหน

| ช่วง | สัปดาห์ | config | คาบที่เพิ่งเรียน | เครื่องมือที่ใช้ได้ |
|---|---|---|---|---|
| **Warm-up** | 1–3 | ห้อง 10×10 · `observation: full` · `action_noise: 0` · ไม่จำกัดแบต | คาบ 1–2 (MDP, TD, Q-learning) | **tabular Q-learning ทำได้จริง** — state space เล็กพอจะเก็บเป็นตารางได้ ตรงกับ worked example ในคาบ 2 เป๊ะ |
| **Main** | 4–6 | ห้อง 20×20 · `observation: local` · `action_noise: 0.10` · `sticky_dirt: 0.15` | คาบ 3–5 (policy gradient, actor-critic, GAE, PPO) | **ตารางเก็บไม่ไหวแล้ว** → ต้อง function approximation → PPO / actor-critic กลายเป็นคำตอบธรรมชาติ |
| **Final** | 7 | ห้อง 30×30 · `sensor_noise: 0.05` · obstacle หนาแน่นขึ้น | — | ทดสอบว่า generalize ได้จริง ไม่ใช่จำ config เดิม |

> **จุดที่ตั้งใจให้เกิด** — ทีมที่ใช้ tabular Q-learning ผ่าน Warm-up มาสบายๆ จะเจอกำแพงทันทีในสัปดาห์ที่ 4
> ซึ่งเป็นสัปดาห์เดียวกับที่คาบเรียนอธิบายว่าทำไมต้องมี function approximation — นิสิตจะเข้าใจเหตุผลจากการเจอปัญหาเอง
> ไม่ใช่จากการฟังบรรยาย

การเปลี่ยน config ระหว่าง phase ยังช่วยไม่ให้ทีมที่นำห่างตั้งแต่ต้นลอยตัว และทีมที่ตามอยู่ยังมีโอกาสไล่

---

## 9. Starter Kit

- โค้ด environment ตัวเดียวกับที่ใช้ตัดสิน (Gymnasium-compatible) ให้ train ที่บ้านได้
- `RandomAgent`, `GreedyAgent`, `BFSCoverageAgent` เป็นตัวอย่างและเป็น baseline
- สคริปต์ประเมินผลในเครื่องตัวเอง (ใช้ public seeds) ที่คำนวณคะแนนด้วยสูตรเดียวกับ §5 เป๊ะ
- ตัวอย่าง reward shaping สำหรับตอน train (**ย้ำ: reward ตอน train ≠ metric ตอนตัดสิน**)
- CLI สำหรับส่งงาน:

```bash
arena init cp463-vacuum-1-2026
arena eval --local --seeds public
arena submit --note "เพิ่ม frontier exploration"
```

---

## 10. นโยบายการ Train

**แพลตฟอร์มทำหน้าที่ประเมินผลเท่านั้น ไม่รับ train ให้** — นิสิต train บนเครื่องตัวเอง / Colab / เครื่องแล็บ

เหตุผล: GPU ที่มี (RTX 3090 หนึ่งใบ) ถ้าเปิดให้ทุกทีมแย่งกัน train คิวจะตันจนใช้ประเมินผลไม่ได้
สำหรับ grid world นั้น inference ของ policy เล็กมาก **CPU ก็เพียงพอ** GPU เลยเหลือไว้ใช้กับโจทย์ prediction-based ของวิชาอื่น

> ถ้าอนาคตอยากเปิดให้ train บนแพลตฟอร์ม → ทำเป็น "training credit" แยกคิว จำกัดโควตาชั่วโมงต่อทีม
> (ประเด็นค้างข้อ 2 ใน [README §16](../../../../../README.md#16-ประเด็นที่ยังต้องตัดสินใจ))

---

## 11. สิ่งที่ต้องตัดสินใจ/ทดสอบก่อนเปิดเทอม

| # | ประเด็น | รายละเอียด |
|---|---|---|
| 0 | **⚠️ `action_noise` และ `sticky_dirt` แรงพอจะกด planner ลงจริงไหม** | ต้องรัน `BFSCoverageAgent` เทียบกับ PPO baseline ที่ noise หลายระดับ (0 / 0.05 / 0.10 / 0.20) ก่อนเปิดเทอม ถ้า planner ยังชนะที่ 0.10 แปลว่า noise น้อยไป ถ้า PPO ก็แพ้ไปด้วยแปลว่าแรงเกินจนโจทย์กลายเป็นเรื่องดวง — **ข้อนี้ต้องเสร็จก่อนข้ออื่นทั้งหมด** |
| 1 | **Observation mode ที่จะใช้จริงในช่วง Main** | `local` น่าจะกำลังดี — `full` ง่ายไป, `sensor` อาจยากเกินสำหรับเทอมเดียว ต้องลองกับ baseline ก่อน |
| 2 | **`max_steps` เพียงพอไหม** | ถ้าตั้งน้อยไปจนไม่มีทีมไหนดูดครบได้เลย `completion_bonus` จะกลายเป็นค่าคงที่ที่ไม่มีผล และคะแนนจะถอยไปเป็น AUC ล้วนโดยปริยาย — ต้องรัน `BFSCoverageAgent` ดูว่าใช้กี่ timestep แล้วตั้ง `max_steps` ให้มีที่เหลือพอสมควร |
| 3 | **น้ำหนัก penalty และ `completion_bonus`** | รันกับ baseline ทั้ง 4 ระดับ ดูว่าคะแนนกระจายตัวห่างกันพอให้แยกชั้นได้จริงไหม |
| 4 | ~~**`battery` จำกัดหรือไม่**~~ | ✅ **ตัดสินใจแล้ว: `battery: null` ทุก phase** ให้ `max_steps` เป็นข้อจำกัดเดียวที่ผูกพัน ([เหตุผล](environment-spec.md#11-config-ของทั้ง-3-phase)) |
| 5 | **จำนวน episode ต่อ run** | 30 public seeds × เวลาที่ใช้รัน 1 episode × จำนวนทีม × โควตาต่อวัน ต้องไม่เกินกำลัง runner ในคืนก่อน deadline |
| 6 | **ผังห้องแบบ `rooms`** | ถ้าจะใช้ generator แบบหลายห้องเชื่อมกัน ต้องเขียนเพิ่มและตรวจ `guarantee_connected` ให้ดี |
