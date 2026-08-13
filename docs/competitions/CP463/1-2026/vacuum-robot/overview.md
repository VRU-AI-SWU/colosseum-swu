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
| **โค้ด** | [`envs/cp463-vacuum`](../../../../../envs/cp463-vacuum/) — v1.0.0 · conformance test §14 ผ่านครบ |
| **สถานะ** | config ผ่านการ calibrate รอบที่ 1 แล้ว ([รายงาน](calibration-2026-08.md)) — เหลือยืนยันว่า learned policy ชนะ Gold ได้ ([§11 ข้อ 0b](#11-สิ่งที่ต้องตัดสินใจทดสอบก่อนเปิดเทอม)) |

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
- **Stochasticity** — ตั้งแต่ phase Main เป็นต้นไป การเคลื่อนที่มีโอกาสพลาดทิศ (`action_noise`)
  บางพื้นผิวดูดไม่ติดในครั้งแรก (`sticky_dirt`) และ **ค่าที่อ่านจากเซนเซอร์ผิดพลาดได้** (`sensor_noise`)
  ความสุ่มทั้งหมดถูกตรึงด้วย seed เช่นกัน → seed เดียวกันได้ผลลัพธ์เดียวกันเสมอ

> **ทำไมต้องมี stochasticity** — ในห้องที่ deterministic และรู้กติกาแน่นอน **BFS/frontier planner ชนะ RL อย่างขาดลอย**
> ถ้าปล่อยไว้แบบนั้น ทีมที่เขียน planner มือจะชนะทีมที่ตั้งใจทำ RL ทั้งเทอม ซึ่งขัดกับวัตถุประสงค์ของวิชา
> รายละเอียดของประเด็นนี้อยู่ที่
> [template §5](../../../../task-templates/agent-vs-environment-rl.md#5-ปัญหาที่ต้องตัดสินใจ-planning-ชนะ-learning)
>
> ⚠️ **แต่ความสุ่มไม่ได้ผลเท่ากันทุกชนิด — ข้อนี้วัดจริงแล้วและผลไม่ตรงกับที่ร่างไว้ตอนแรก**
> ([รายงานการ calibrate](calibration-2026-08.md))
>
> | ชนิดความสุ่ม | ผลต่อ Gold (BFS planner) |
> |---|---|
> | `action_noise` 0 → 0.50 | คะแนนตกแค่ 9% · ยังดูดครบ **ทุก seed** |
> | `sticky_dirt` 0 → 0.80 | คะแนนตกแค่ 3% · ยังดูดครบทุก seed |
> | `sensor_noise` 0 → 0.02 | คะแนนตก **9% และเริ่มดูดไม่ครบ** · ที่ 0.05 พังทั้งชุด |
>
> เหตุผลที่สองตัวแรกไม่ได้ผล: planner ที่ **replan ทุก timestep** เป็น closed-loop controller
> การลื่นหรือการดูดไม่ขึ้นทำให้*เส้นทาง*ยาวขึ้นเท่านั้น ไม่ได้ทำให้*ความรู้เกี่ยวกับโลก*ผิด
> และเพราะหลักการ hardware-independent scoring ไม่คิดเวลา การ replan จึงไม่มีต้นทุนเลย
>
> `sensor_noise` ต่างออกไปตรงที่มันทำให้**แผนที่ที่ agent สะสมไว้ผิดถาวร** สำหรับช่องที่ไม่ได้กลับไปดูซ้ำ
> planner ที่เชื่อเซนเซอร์ตรงๆ จะไล่ตาม "ฝุ่นผี" และหลบ "กำแพงที่ไม่มีอยู่จริง"
> ส่วน agent ที่รวมหลักฐานจากการสังเกตหลายครั้ง (belief state) จะทนได้ — ซึ่งตรงกับ
> value estimation ภายใต้ความไม่แน่นอนที่สอนในคาบ 3–5 พอดี **จึงเป็นคันโยกหลักของโจทย์นี้**

---

## 3. Configuration

ผู้สอนแก้ค่าเหล่านี้ได้ผ่านหน้าเว็บหรือไฟล์ YAML ต่อ competition
ตัวอย่างข้างล่างคือ **config ของ phase Main ที่ใช้จริง** — ไฟล์ตัวจริงอยู่ที่
[`envs/cp463-vacuum/configs/`](../../../../../envs/cp463-vacuum/configs/) และค่าของทั้ง 3 phase
อยู่ที่ [environment-spec §11](environment-spec.md#11-config-ของทั้ง-3-phase)

```yaml
task: vacuum_gridworld
version: 1.0.0
phase: main

room:
  width: 20
  height: 20
  obstacle_density: 0.15        # สัดส่วนช่องที่เป็นสิ่งกีดขวาง
  obstacle_generator: clustered # random | clustered  (rooms/fixed ยังไม่ implement ใน v1.0.0)
  dirt_distribution: uniform    # uniform | clustered  (patchy ยังไม่ implement)
  dirt_ratio: 0.6               # สัดส่วนช่องว่างที่สกปรก
  guarantee_connected: true     # การันตีว่าทุกช่องสกปรกไปถึงได้

robot:
  start: random                 # random | corner | center
  observation: local
  observation_window: 5         # ต้องเป็นเลขคี่
  battery: null                 # ตัดสินใจแล้ว: null ทุก phase — ให้ max_steps เป็นข้อจำกัดเดียว
  move_cost: 1                  # ยังอยู่ใน schema ไว้ใช้ปีถัดไป แต่ไม่มีผลเมื่อ battery = null
  suck_cost: 2

dynamics:
  action_noise: 0.10            # โอกาสเดินพลาดไปทิศข้างเคียง — วัดแล้วว่า *ไม่ใช่* คันโยกความยาก (§2)
  sticky_dirt: 0.15             # สัดส่วนช่องที่ต้อง SUCK สองครั้ง — วัดแล้วว่าไม่ใช่คันโยกเช่นกัน
  sensor_noise: 0.02            # 🔑 คันโยกจริง — โอกาสที่แต่ละค่าใน observation ถูกพลิก

episode:
  max_steps: 1500               # จำนวน decision timestep สูงสุดต่อ episode
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
```

**ค่า seed ไม่ได้อยู่ใน config ของ environment** — เป็นเรื่องของ runner และเก็บที่ on-prem เท่านั้น
ย่านที่ใช้สุ่ม: train `1–9999` · public `20000–29999` · private `50000–59999` ([§7](#7-seeds-และ-public--private-split))

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

**คะแนนที่วัดได้จริง** (30 seed ของชุด conformance · `env_version` 1.0.0 · ค่าที่ตรึงเป็นเส้นเกรดต้องรันบน public seeds อีกครั้ง)

| | Warm-up | Main | Final |
|---|---|---|---|
| 🥉 Bronze | 0.136 | 0.244 | 0.148 |
| 🥈 Silver | 0.631 | 0.810 | 0.563 |
| 🥇 Gold | 1.748 (ครบ 30/30) | 1.716 (ครบ 28/30) | 1.511 (ครบ 22/30) |
| เพดานของสูตร | 2.0 | 2.0 | 2.0 |

> **Gold = BFS planner เป็นหมุดหมายที่ตั้งใจ** — ข้อความที่ต้องการสื่อคือ "วิธีดั้งเดิมที่ไม่เรียนรู้ทำได้ถึงแค่นี้เมื่อโลกไม่แน่นอน"
> ทีมที่ส่ง planner มาก็จะไปหยุดอยู่ตรงระดับ Gold พอดี ส่วนการจะแตะ Diamond ต้องใช้ policy ที่เรียนรู้จากประสบการณ์
>
> ⚠️ **ยังพิสูจน์ไม่ได้ว่าครึ่งหลังของประโยคนั้นจริง** — ช่องว่างระหว่าง Gold กับเพดานเหลือ 0.28 จุดบน Main
> ซึ่งแปลว่า policy ที่จะแตะ Diamond ต้องดูดครบเกือบทุก seed **และเร็วกว่า BFS** โดยเห็นแค่หน้าต่าง 5×5
> ข้อนี้ต้องยืนยันด้วยการเทรน PPO จริงก่อน ([§11 ข้อ 0](#11-สิ่งที่ต้องตัดสินใจทดสอบก่อนเปิดเทอม))
>
> **"Gold" ในที่นี้หมายถึง planner ที่เชื่อเซนเซอร์ตรงๆ** — มันเขียนทับแผนที่ด้วยค่าที่เห็นล่าสุดโดยไม่กรองอะไรเลย
> ถ้าให้ Gold ทำ majority vote จากการสังเกตซ้ำ มันจะทน `sensor_noise` ได้มากขึ้นและหมุดจะขยับสูงขึ้นมาก
> — เป็นการตัดสินใจที่ต้องประกาศให้ชัด เพราะมันคือเส้นแบ่งเกรด

---

## 7. Seeds และ Public / Private Split

| ชุด | ย่านที่สุ่มมา | จำนวนที่ใช้ | นิสิตรู้ | ใช้ทำอะไร |
|---|---|---|---|---|
| **Train** | `1–9999` | ไม่จำกัด | รู้ย่าน + มี generator | เทรนและทดสอบเองในเครื่อง |
| **Public** | `20000–29999` | 30 ต่อ phase | **รู้แค่จำนวน ไม่รู้ค่า** | leaderboard ระหว่างเทอม |
| **Private** | `50000–59999` | 100 (Warm-up) / 150 (Main, Final) | ไม่รู้อะไรเลย | **ตัดสินอันดับและเกรดจริง** |
| **Conformance** | `70001–70030` | 30 | **เปิดเผยทั้งค่าและ golden** | ให้นิสิตยืนยันว่า environment ในเครื่องตรงกับตัวที่ใช้ตัดสิน |

> **ประกาศได้แค่ "ย่าน" ห้ามประกาศ "ช่วงที่ใช้จริง"** — ร่างแรกของ spec เขียนว่า public คือ `20001–20030`
> ซึ่งเป็นช่วงที่มี 30 ค่าพอดีสำหรับ seed 30 ตัว **เท่ากับบอกค่าไปทั้งชุด** และ private leaderboard
> จะไร้ความหมายทันที ย่าน `20000–29999` มี 10,000 ค่าแต่ใช้จริง 30 ค่า รู้ย่านแล้วยังเดาไม่ได้
> ค่าจริงถูกสุ่มแล้วเก็บไว้ที่ [`colosseum-hypogeum`](../../../../../README.md#105-โครงสร้าง-repository)

- **สี่ชุดนี้ต้องไม่ทับกันเลย** — ตรวจให้ตอน generate ไม่ใช่หวังว่าย่านที่แยกกันจะพอ
- private seeds เก็บไว้ที่ runner on-prem เท่านั้น ไม่อัพขึ้น cloud
- ชุด conformance เปิดเผยได้เพราะเป็นคนละช่วงกับที่ใช้ตัดสิน ([README §10.4](../../../../../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries))
- นโยบายกันการโกงที่เหลือใช้ตามแพลตฟอร์ม ([README §7](../../../../../README.md#7-ความสุจริตทางวิชาการและการกันโกงระบบ))

> **นิสิตประเมินตัวเองบน public seeds ในเครื่องไม่ได้** — เพราะไม่รู้ค่า seed ซึ่งเป็นเจตนา
> `arena eval --local` จึงรันบน **training seeds** ที่นิสิตสร้างเองได้ไม่จำกัด ส่วนคะแนน public
> ต้องส่งเข้าระบบเท่านั้น (นี่คือสิ่งที่ทำให้โควตาส่งมีความหมาย — ถ้ารันเองได้ก็ไม่มีอะไรกันการ overfit)
> การยืนยันว่า environment ในเครื่อง "ตรงกับตัวที่ใช้ตัดสิน" ใช้ชุด conformance แทน

---

## 8. แผนตามสัปดาห์ (Phases)

**หลักการ: แต่ละ phase ต้องแก้ได้พอดีตอนที่คาบเรียนที่เกี่ยวข้องเพิ่งสอนจบ**
โจทย์จึงกลายเป็นแบบฝึกหัดของคาบนั้นๆ แทนที่จะเป็นก้อนใหญ่ก้อนเดียวที่นิสิตไม่รู้จะเริ่มตรงไหน

| ช่วง | สัปดาห์ | config | คาบที่เพิ่งเรียน | เครื่องมือที่ใช้ได้ |
|---|---|---|---|---|
| **Warm-up** | 1–3 | ห้อง 10×10 · `observation: full` · ไม่มีความสุ่มเลย · `max_steps: 250` | คาบ 1–2 (MDP, TD, Q-learning) | **tabular Q-learning ทำได้จริง** — state space เล็กพอจะเก็บเป็นตารางได้ ตรงกับ worked example ในคาบ 2 เป๊ะ |
| **Main** | 4–6 | ห้อง 20×20 · `observation: local` (5×5) · `sensor_noise: 0.02` · `max_steps: 1500` | คาบ 3–5 (policy gradient, actor-critic, GAE, PPO) | **ตารางเก็บไม่ไหวแล้ว** → ต้อง function approximation และต้องรับมือ observation ที่เชื่อไม่ได้ 100% → PPO / actor-critic กลายเป็นคำตอบธรรมชาติ |
| **Final** | 7 | ห้อง 30×30 · หน้าต่างแคบลงเหลือ 3×3 · `sensor_noise: 0.01` · obstacle หนาแน่นขึ้น | — | ทดสอบว่า generalize ได้จริง ไม่ใช่จำ config เดิม |

> **จุดที่ตั้งใจให้เกิด** — ทีมที่ใช้ tabular Q-learning ผ่าน Warm-up มาสบายๆ จะเจอกำแพงทันทีในสัปดาห์ที่ 4
> ซึ่งเป็นสัปดาห์เดียวกับที่คาบเรียนอธิบายว่าทำไมต้องมี function approximation — นิสิตจะเข้าใจเหตุผลจากการเจอปัญหาเอง
> ไม่ใช่จากการฟังบรรยาย

การเปลี่ยน config ระหว่าง phase ยังช่วยไม่ให้ทีมที่นำห่างตั้งแต่ต้นลอยตัว และทีมที่ตามอยู่ยังมีโอกาสไล่

---

## 9. Starter Kit

- โค้ด environment ตัวเดียวกับที่ใช้ตัดสิน (Gymnasium-compatible) ให้ train ที่บ้านได้
  → [`envs/cp463-vacuum`](../../../../../envs/cp463-vacuum/) ติดตั้งผ่าน wheel ที่ release ไว้ ไม่ต้อง clone repo
- `RandomAgent`, `GreedyAgent`, `BFSCoverageAgent` เป็นตัวอย่างและเป็น baseline
- **สคริปต์ประเมินผลในเครื่องตัวเอง ที่รันบน training seeds** (`1–9999`) และคำนวณคะแนนด้วย
  `vacuum/scoring.py` ซึ่งเป็น**ไฟล์เดียวกับที่ grader ใช้** ไม่ใช่สูตรที่เขียนซ้ำ
  → ไม่ใช่ public seeds เพราะนิสิตไม่รู้ค่า seed ชุดนั้น ([§7](#7-seeds-และ-public--private-split))
- **ชุด conformance test พร้อม golden value** ให้รันยืนยันว่า environment ในเครื่องตรงกับตัวที่ใช้ตัดสินทุกบิต
- ตัวอย่าง reward shaping สำหรับตอน train (**ย้ำ: reward ตอน train ≠ metric ตอนตัดสิน**)
- CLI สำหรับส่งงาน:

```bash
arena init cp463-vacuum-1-2026
arena eval --local --seeds 1-200      # training seeds — ไม่กินโควตา
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

📊 ผลการวัดทั้งหมดอยู่ที่ [calibration-2026-08.md](calibration-2026-08.md)

| # | ประเด็น | สถานะ |
|---|---|---|
| 0 | ~~**`action_noise` และ `sticky_dirt` แรงพอจะกด planner ลงจริงไหม**~~ | ❌ **วัดแล้ว: ไม่แรงพอ และไม่ใช่แค่ "น้อยไป"** — `action_noise` 0 → 0.50 กด Gold ลงแค่ 9% และยังดูดครบทุก seed · `sticky_dirt` 0.80 กดลง 3% · **เปลี่ยนไปใช้ `sensor_noise` เป็นคันโยกหลักแทน** (Main 0.02 · Final 0.01) เหตุผลอยู่ที่ [§2](#2-environment-spec) |
| 0b | **⚠️ learned policy ชนะ Gold ได้จริงไหม** | ยังไม่ตอบ — ต้องเทรน PPO แล้วรัน `examples/calibrate.py --policy` เทียบที่ `sensor_noise` 0 / 0.01 / 0.02 / 0.05 · **ข้อนี้เหลืออยู่ข้อเดียวที่ยังกั้นการเปิดเทอม** ถ้า PPO ยังแพ้ Gold ที่ 0.02 ต้องกลับไปพิจารณาทางเลือก C+D ของ [template §5](../../../../task-templates/agent-vs-environment-rl.md#5-ปัญหาที่ต้องตัดสินใจ-planning-ชนะ-learning) |
| 0c | **"Gold" คือ planner ที่ไร้เดียงสาเรื่อง noise หรือไม่** | ต้องประกาศให้ชัดเพราะเป็นเส้นแบ่งเกรด — ตอนนี้ `BFSCoverageAgent` เขียนทับแผนที่ด้วยค่าล่าสุดโดยไม่กรอง ถ้าเพิ่ม majority vote หมุด Gold จะขยับขึ้นมาก ([§6](#6-baseline-ladder)) |
| 1 | ~~**Observation mode ที่จะใช้จริงในช่วง Main**~~ | ✅ **`local` window 5** — วัดแล้วว่าขนาดหน้าต่างแทบไม่มีผลต่อ Gold (window 5→3 ตกแค่ 0.005 · โหมด `sensor` ตก 0.013) ตัวที่คุมความยากคือ `sensor_noise` ไม่ใช่ขนาดหน้าต่าง |
| 2 | ~~**`max_steps` เพียงพอไหม**~~ | ✅ **Warm-up 250 · Main 1500 · Final 3000** — ยืนยันแล้วว่า Gold ดูดครบ 30/30 · 28/30 · 22/30 ตามลำดับ `completion_bonus` จึงทำงานทุก phase · Final ยืนยันด้วยว่าตัวที่ผูกพันคือ `sensor_noise` ไม่ใช่ `max_steps` (เพิ่มเป็น 4000 แล้วผลไม่ต่าง) |
| 3 | ~~**น้ำหนัก penalty และ `completion_bonus`**~~ | ✅ **คงค่าเดิม** — ladder แยกกันได้ทุก phase โดย CI ไม่ทับกันเลย ([การทดลองที่ 3](calibration-2026-08.md#3-การทดลองที่-3--baseline-ห่างกันพอไหม)) · ข้อควรระวัง: CI ของ Silver กว้าง ±0.17 เพราะมันติดหลังกำแพงแล้วแกว่งในบาง seed ถ้าใช้เป็นเส้นเกรดต้องตรึงด้วย seed จำนวนมากกว่านี้ |
| 4 | ~~**`battery` จำกัดหรือไม่**~~ | ✅ **`battery: null` ทุก phase** ให้ `max_steps` เป็นข้อจำกัดเดียวที่ผูกพัน ([เหตุผล](environment-spec.md#11-config-ของทั้ง-3-phase)) |
| 5 | **จำนวน episode ต่อ run** | วัดต้นทุนได้แล้ว: Gold ใช้ **~0.5 วินาที/episode** บน Main → 30 seed ≈ 15 วินาทีต่อ run · 10 ทีม × 5 ครั้ง/วัน ≈ 12 นาที/วัน ของเวลา CPU **ไม่เป็นคอขวด** เหลือแค่ยืนยันกับ policy ที่หนักกว่า (PPO + torch) |
| 6 | **ผังห้องแบบ `rooms`** | ยังไม่ implement ใน v1.0.0 — ถ้าจะใช้ต้องขึ้น v1.1.0 พร้อม conformance test ใหม่ |
