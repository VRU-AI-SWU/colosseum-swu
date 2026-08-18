# Task Template — Agent vs Environment (Reinforcement Learning)

Template สำหรับโจทย์ที่นิสิตส่ง **agent** เข้ามาทำงานในสภาพแวดล้อมจำลอง แล้ววัดผลจากพฤติกรรมที่เกิดขึ้น
เอกสารนี้เป็น spec ระดับแพลตฟอร์ม — โจทย์จริงของแต่ละวิชาอยู่ใน [`docs/competitions/`](../competitions/)
ตัวอย่างที่ใช้จริงตัวแรก: [CP463 1/2026 — Vacuum Cleaner Grid World](../competitions/CP463/1-2026/vacuum-robot/overview.md)

---

## สารบัญ

1. [สิ่งที่ template นี้กำหนด และไม่กำหนด](#1-สิ่งที่-template-นี้กำหนด-และไม่กำหนด)
2. [มิติของ environment ที่ต้องประกาศ](#2-มิติของ-environment-ที่ต้องประกาศ)
3. [Reward เป็นส่วนหนึ่งของคำตอบ ไม่ใช่ของโจทย์](#3-reward-เป็นส่วนหนึ่งของคำตอบ-ไม่ใช่ของโจทย์)
4. [โจทย์คือ Contextual MDP ไม่ใช่ MDP เดียว](#4-โจทย์คือ-contextual-mdp-ไม่ใช่-mdp-เดียว)
5. [ปัญหาที่ต้องตัดสินใจ: planning ชนะ learning](#5-ปัญหาที่ต้องตัดสินใจ-planning-ชนะ-learning)
6. [Contract ของสิ่งที่นิสิตส่ง](#6-contract-ของสิ่งที่นิสิตส่ง)
7. [Evaluation protocol และความแปรปรวน](#7-evaluation-protocol-และความแปรปรวน)
8. [Public / Private seeds](#8-public--private-seeds)
9. [Sandbox และข้อจำกัดทรัพยากร](#9-sandbox-และข้อจำกัดทรัพยากร)
10. [ประเด็นที่ต้องตัดสินใจต่อ competition](#10-ประเด็นที่ต้องตัดสินใจต่อ-competition)

---

## 1. สิ่งที่ template นี้กำหนด และไม่กำหนด

### ❌ ไม่กำหนด — ตระกูลอัลกอริทึมที่นิสิตใช้

**model-free / model-based · value-based / policy-based / actor-critic · on-policy / off-policy**
ทั้งหมดนี้เป็นคุณสมบัติของ**วิธีที่นิสิตใช้เทรน** ไม่ใช่คุณสมบัติของโจทย์ แพลตฟอร์มมองไม่เห็นและไม่ควรมองเห็น

เพราะการเทรนเกิดขึ้นบนเครื่องนิสิต สิ่งที่ส่งเข้ามาคือ **policy ที่เทรนเสร็จแล้ว** ซึ่งไม่ว่าจะได้มาจาก DQN, PPO, SAC, MCTS
หรือแม้แต่ heuristic ที่เขียนมือ ก็ล้วน implement interface เดียวกัน:

```python
def act(self, observation) -> action
```

> นี่เป็นเรื่องดีในเชิงการสอน — **การเลือกตระกูลอัลกอริทึมให้เหมาะกับโจทย์คือสิ่งที่เราอยากประเมิน**
> ถ้าเราบังคับว่าต้องใช้ DQN ก็เท่ากับเฉลยส่วนที่ยากที่สุดให้ไปแล้ว
> (ถ้าผู้สอนต้องการจำกัดจริงๆ เช่น "ต้องเป็น learned policy ห้าม plan ตอน inference" ดู §5)

### ✅ กำหนด — สัญญาระหว่างแพลตฟอร์มกับ agent

รูปแบบ observation/action · จำนวน episode และ seed ที่ใช้วัด · วิธีรวมคะแนน · ข้อจำกัดทรัพยากร · สิ่งที่ต้องอัพโหลด

---

## 2. มิติของ environment ที่ต้องประกาศ

มิติเหล่านี้เป็นของโจทย์จริงๆ และมีผลต่อการออกแบบระบบ ต้องประกาศใน `TaskSpec` ทุกข้อ

| มิติ | ตัวเลือก | ทำไมแพลตฟอร์มต้องรู้ |
|---|---|---|
| **Observability** | fully observable (MDP) / partially observable (POMDP) | กำหนดว่า runner ส่งอะไรให้ agent และ agent ต้องมี state ภายในหรือไม่ (มีผลต่อ `reset()`) |
| **Transition** | deterministic / stochastic | stochastic ต้องใช้ episode มากกว่ามากเพื่อให้คะแนนนิ่ง (§7) |
| **Action space** | discrete / continuous / hybrid | กำหนด schema ของ action และการ validate |
| **Horizon** | episodic (finite) / continuing | continuing ต้องนิยามจุดตัดเอง |
| **Agent count** | single / multi | multi-agent ต้องใช้โหมด Agent vs Agent ซึ่งยังไม่ทำ |
| **Episode variation** | fixed environment / procedurally generated ต่อ seed | ถ้า generate ต่อ seed จะกลายเป็นปัญหา generalization (§4) |
| **Reward ตอนประเมิน** | เปิดเผย / ไม่เปิดเผย | **แนะนำ: ไม่เปิดเผย** — agent ได้แค่ observation ไม่ได้ reward ตอนวัดผล |

### โจทย์ vacuum อยู่ตรงไหน

| มิติ | ค่าของโจทย์ CP463 |
|---|---|
| Observability | **POMDP** ทั้ง Main และ Final — `observation: local` และมี `sensor_noise` ทำให้ observation เชื่อไม่ได้ 100% · MDP เฉพาะ phase Warm-up (`full`, ไม่มีความสุ่มเลย) |
| Transition | **stochastic** ตั้งแต่ phase Main (`action_noise: 0.10`, `sticky_dirt: 0.15`) · deterministic เฉพาะ Warm-up |
| Action space | **discrete** 6 actions |
| Horizon | **episodic**, finite (`max_steps`) |
| Agent count | **single** |
| Episode variation | **procedurally generated ต่อ seed** — ผังห้องและฝุ่นสุ่มใหม่ทุก seed |
| Reward ตอนประเมิน | **ไม่เปิดเผย** — วัดจาก coverage AUC ภายนอก |

**ตระกูลอัลกอริทึมที่เข้ากับโจทย์นี้ได้ทั้งหมด** (นิสิตเลือกเอง — นี่คือส่วนที่เป็นการบ้าน)

| แนวทาง | ใช้ได้ไหม | หมายเหตุ |
|---|---|---|
| Off-policy value-based (DQN, Rainbow) | ✅ | ต้องมี memory/frame-stack ถ้าเป็น POMDP |
| On-policy actor-critic (PPO, A2C) | ✅ | เป็นตัวเลือกมาตรฐานสำหรับ environment ที่ generate ต่อ seed |
| Model-based | ✅ แต่ไม่คุ้ม | dynamics ของ grid world ง่ายจนเรียนรู้ model แล้วก็เท่ากับรู้กติกาอยู่แล้ว → กลายเป็น planning |
| Planning ล้วน (BFS + frontier) | ✅ **และแรงมาก** | ⚠️ ดู §5 |
| Imitation learning จาก planner | ✅ | เป็นทางสายกลางที่น่าสนใจ ให้ RL เรียนจาก planner แล้วเร็วกว่าตอน inference |

---

## 3. Reward เป็นส่วนหนึ่งของคำตอบ ไม่ใช่ของโจทย์

จุดที่ template นี้ต่างจากงาน RL benchmark ทั่วไป (Atari, MuJoCo) ที่ reward มาพร้อม environment

- **โจทย์นิยามด้วย score function** (เช่น coverage AUC) ที่ใช้ตัดสินเท่านั้น agent ไม่เห็นค่านี้ระหว่างรัน
- **reward function เป็นสิ่งที่นิสิตออกแบบเอง** สำหรับใช้ตอนเทรนบนเครื่องตัวเอง
- starter kit แจกตัวอย่าง reward shaping ให้ 1–2 แบบ แต่ต้องเขียนกำกับชัดว่า **reward ตอนเทรน ≠ metric ตอนตัดสิน**

> นี่ทำให้ **reward design กลายเป็นทักษะที่ถูกประเมิน** ซึ่งเป็นหัวใจของ RL ในทางปฏิบัติ
> นิสิตจะได้เจอกับตัวเองว่า reward ที่ดูสมเหตุสมผล (เช่น +1 ต่อช่องที่ดูดได้) ทำให้ agent เรียนรู้พฤติกรรมที่ไม่ตรงกับสิ่งที่เราต้องการจริง
> — เป็นบทเรียนเรื่อง reward hacking / specification gaming ที่สอนด้วยการบรรยายได้ยากกว่ามาก

---

## 4. โจทย์คือ Contextual MDP ไม่ใช่ MDP เดียว

เมื่อผังห้องถูก generate ใหม่ทุก seed สิ่งที่นิสิตต้องแก้ไม่ใช่ "หา optimal policy ของ MDP หนึ่งตัว"
แต่คือ **"หา policy ที่ทำงานได้ดีบน distribution ของ MDP"** — ตระกูลเดียวกับ ProcGen / MiniGrid

ผลที่ตามมาซึ่งต้องบอกนิสิตให้ชัดตั้งแต่แรก:

- การจำผังห้องใช้ไม่ได้ — และเป็นเหตุผลว่าทำไม private seeds ถึงจำเป็น (§8)
- ต้องแยกให้ออกระหว่าง **training seeds** (นิสิตใช้เทรนได้ไม่จำกัด) กับ **evaluation seeds** (แพลตฟอร์มถือไว้)
- overfitting ใน RL หน้าตาเหมือน "ได้ 95% บน seed ที่เทรน แต่ 40% บน seed ใหม่" ซึ่งนิสิตจะเจอเองถ้าเราให้ public seeds ที่ไม่ซ้ำกับที่เขาเทรน
- ควรแจก **training seed generator** ให้ (ฟังก์ชันเดียวกับที่ระบบใช้ แต่คนละช่วงเลข) เพื่อให้เขาสร้าง training distribution เองได้ไม่จำกัด

---

## 5. ปัญหาที่ต้องตัดสินใจ: planning ชนะ learning

⚠️ **เรื่องนี้กระทบวัตถุประสงค์ของวิชาโดยตรง ต้องตัดสินใจก่อนเปิดเทอม**

ในโจทย์ coverage ที่ dynamics เป็น deterministic และรู้กติกาแน่นอน **อัลกอริทึมค้นหาแบบดั้งเดิมเหนือกว่า RL อย่างชัดเจน**
frontier exploration + BFS/A* ทำ coverage ได้เกือบ optimal โดยไม่ต้องเทรนอะไรเลย — Gold baseline ที่เราตั้งไว้เองก็คือ BFS coverage planner

ถ้าจัดอันดับด้วยคะแนนล้วน ผลที่น่าจะเกิดคือ **ทีมที่เขียน planner มือ ชนะทีมที่ตั้งใจทำ RL ทั้งเทอม** ซึ่งเป็นบทเรียนที่จริง แต่ไม่ใช่บทเรียนที่วิชานี้ตั้งใจสอน

### ทางเลือกในการรับมือ

| ทางเลือก | วิธีทำ | ข้อดี / ข้อเสีย |
|---|---|---|
| **A. ทำให้ environment ยากสำหรับ planner** (แนะนำ) | เพิ่ม stochasticity และ hidden dynamics ลงใน config | ตรงประเด็นที่สุด ไม่ต้องบังคับอะไรนิสิต แต่ต้องแก้ environment และทดสอบใหม่ |
| **B. บังคับว่าห้าม plan ตอน inference** | ไม่ใส่โค้ด environment ลงใน container ตอนประเมิน + จำกัดเวลาต่อ step ให้แคบ (เช่น 20 ms) จน rollout ไม่ทัน | บังคับได้จริงระดับหนึ่ง แต่นิสิตเขียน simulator เองใหม่ได้ (grid world ง่ายมาก) → เป็นการแข่งกันหลบกติกา |
| **C. ยอมรับ แล้วใช้ baseline เป็นตัวคุม** | ตั้ง Gold = BFS planner แล้วให้เกรดแบบ threshold: "ชนะ planner = เต็ม" | ง่ายที่สุด ไม่ต้องแก้อะไร และการต้องเอาชนะ planner เป็นเป้าที่ชัดเจน แต่ทีมที่ส่ง planner ก็ได้เต็มเหมือนกัน |
| **D. แยกคะแนนเป็นสองส่วน** | คะแนน leaderboard + คะแนนรายงานที่ประเมินวิธีการ (algorithm choice, reward design, ablation) | ประเมินสิ่งที่อยากสอนได้ตรง แต่เพิ่มภาระตรวจของ TA |

### ถ้าเลือก A — คันโยกที่ทำให้ planning อ่อนลงและ learning คุ้มขึ้น

> ⚠️ **วัดจริงแล้วในโจทย์ CP463 และคันโยกส่วนใหญ่ในตารางนี้ไม่ทำงาน**
> — ตัวเลขอยู่ที่ [calibration-2026-08.md](../competitions/CP463/1-2026/vacuum-robot/calibration-2026-08.md)

| คันโยก | ที่คิดไว้ | **ผลจริง** |
|---|---|---|
| `action_noise: 0.1` (เดินพลาดทิศ 10%) | planner ต้อง replan ตลอด → กลายเป็นการแก้ MDP ซึ่ง RL ถนัดกว่า | ❌ **แทบไม่มีผล** — Gold เสียไป 1% ที่ 0.10 และแม้ 0.50 ก็ยังดูดครบทุก seed |
| dynamics ที่ไม่บอก (เช่น บางพื้นผิวดูดไม่ติดในครั้งแรก) | ต้อง**เรียนรู้จากประสบการณ์** planner ที่ไม่รู้กติกาทำไม่ได้ | ❌ **แทบไม่มีผล** ถ้า agent *ตรวจสอบผลได้ทันทีจาก observation* — planner ที่ replan เห็นเองว่าดูดไม่ขึ้นแล้วดูดซ้ำ |
| `sensor_noise` | mapping ยากขึ้น planner ที่พึ่งแผนที่แม่นๆ พัง | ✅ **ได้ผลจริงและได้ผลตัวเดียว** — แต่ชันมาก ต้อง calibrate ละเอียด |
| การกระจายฝุ่นมีรูปแบบซ่อนอยู่ต่อประเภทห้อง | policy ที่เรียนรู้ prior จาก training seeds ได้เปรียบชัดเจน | ยังไม่ได้ทดสอบ — เป็นคันโยกที่เหลืออยู่และน่าจะได้ผลด้วยเหตุผลเดียวกับ `sensor_noise` |
| จุดชาร์จ + แบตจำกัด | เพิ่ม long-horizon credit assignment ซึ่งเป็นจุดแข็งของ RL | ยังไม่ได้ทดสอบ (`battery: null` ทุก phase) |

### บทเรียนที่ใช้กับโจทย์ RL อื่นได้

**เกณฑ์แยกคันโยกที่ได้ผลออกจากที่ไม่ได้ผล**

> ความสุ่มที่ทำให้ **เส้นทางยาวขึ้น** → planner รับมือได้ฟรี
> ความสุ่มที่ทำให้ **ความรู้เกี่ยวกับโลกผิด** → planner พัง

`action_noise` และ `sticky_dirt` อยู่กลุ่มแรก · `sensor_noise` อยู่กลุ่มที่สอง

และมีเงื่อนไขซ้อนอีกชั้น: **hidden dynamics จะซ่อนจริงก็ต่อเมื่อ agent ตรวจสอบผลไม่ได้ทันที**
ถ้า observation บอกผลของ action นั้นในเฟรมถัดไป มันไม่ใช่ hidden dynamics แต่เป็นแค่ค่าคงที่ที่เพิ่ม timestep ให้ทุกคนเท่ากัน

**ทำไม planner ถึงแข็งกว่าที่คาด** — หลักการ hardware-independent scoring ([README §5.1](../../README.md#51-metric-เป็นของแต่ละโจทย์-ไม่ใช่ของแพลตฟอร์ม))
ทำให้การ replan ทุก timestep **ไม่มีต้นทุนเลย** planner จึงกลายเป็น closed-loop controller ที่แทบ optimal
นี่เป็นผลข้างเคียงของหลักการที่ถูกต้องในตัวมันเอง และเป็นสิ่งที่ต้องคิดล่วงหน้าตอนออกแบบโจทย์ RL ทุกตัวบนแพลตฟอร์มนี้

### ⚠️ ผลจริงจาก CP463: ทางเลือก A ไม่พอ แม้จะใช้คันโยกที่ถูกตัวแล้ว

เทรน PPO 4.6M timestep (curriculum + reward shaping + fine-tune) เทียบกับ BFS planner บนโจทย์ vacuum

| `sensor_noise` | Gold (planner) | PPO (learned) |
|---|---|---|
| 0.00 | 1.810 · ดูดครบ 100% | 0.457 · ดูดครบ **0%** |
| 0.02 (ที่ใช้จริง) | 1.649 · 87% | 0.445 · 0% |
| 0.05 | 0.753 · 10% | 0.398 · 0% |

**ทิศทางถูกแต่ระยะห่างเกินเอื้อม** — noise กด planner ลง 58% แต่กด policy ลงแค่ 13%
(คือ learned policy ทนความไม่แน่นอนได้ดีกว่าจริงตามที่ทฤษฎีบอก) แต่ฐานความสามารถของ policy
ต่ำเกินไปจนจุดตัดอยู่เลยโซนที่โจทย์ยังไม่กลายเป็นเรื่องดวง

**เหตุผลเชิงโครงสร้างที่ใช้ได้กับโจทย์ coverage ทุกตัว** — ถ้า score function มีโบนัสก้อนใหญ่
ตอน "ทำภารกิจสำเร็จครบ" (ซึ่งควรมี ดู [README §5.1](../../README.md#51-metric-เป็นของแต่ละโจทย์-ไม่ใช่ของแพลตฟอร์ม))
agent ที่จบไม่ครบจะมีเพดานคะแนนต่ำกว่ามาก **ต่อให้เก่งขึ้นเท่าตัวก็ยังแพ้**
และการ "ทำให้ครบ" ในปัญหาแบบ coverage คือสิ่งที่อัลกอริทึมค้นหาได้มาฟรีจากนิยามของมัน
ส่วน RL ต้องเรียนรู้เอง ซึ่งเป็นปัญหา exploration ระยะยาวที่ยากที่สุดปัญหาหนึ่ง

### สิ่งที่ CP463 ตัดสินใจในที่สุด — และคำถามที่ควรถามก่อน

**เลือก C + D และเลิกไล่ตาม A** เพราะพอย้อนกลับไปถามว่า *"อยากให้นิสิตเรียนรู้อะไร"*
คำตอบคือ (1) ได้นำทฤษฎีมาใช้จริง (2) ได้ฝึกสร้าง agent จริง — **"RL ต้องชนะ planning" ไม่ได้อยู่ในนั้น**
มันเป็นข้อสันนิษฐานที่แอบติดมากับร่างแรกและกลืนเวลาไปมากกว่าที่ควร

> ⚠️ **คำถามที่ต้องถามก่อนออกแบบ environment ไม่ใช่หลังจากนั้น**
>
> "โจทย์นี้ต้องใช้ learning ไหม" เป็นคำถามที่ตอบได้ก่อนเขียนโค้ดสักบรรทัด — ถ้า **dynamics รู้ครบ
> และ state space ค้นหาได้** คำตอบคือไม่ ไม่ว่าจะใส่ noise เท่าไร เพราะ noise ที่ประกาศค่าแล้ว
> เป็นส่วนหนึ่งของแบบจำลองที่วางแผนได้ (และวิธีรับมือ noise ที่ดีที่สุดมักเป็น **parameter-free** —
> โหวตเสียงข้างมาก, replan ทุก step — จึงไม่ต้องรู้ค่าด้วยซ้ำ)
>
> คันโยกเดียวที่เปลี่ยนคำตอบได้จริงคือทำให้ **prior ของโลก sample ง่ายแต่เขียนเป็นสูตรยาก**
> (หลักการเดียวกับ ProcGen) ซึ่งเป็นการตัดสินใจตอนออกแบบ generator ไม่ใช่ตอนจูน config

**ถ้าคำตอบคือ "planning เหมาะกับโจทย์นี้" ก็ไม่ต้องฝืน** — ให้ยอมรับ แล้วย้ายสิ่งที่วัด

| | วัดอะไร |
|---|---|
| leaderboard | คุณภาพของ agent · **ไม่บังคับตระกูลอัลกอริทึม** |
| คะแนนรายงาน | ความเข้าใจว่า *ทำไม* วิธีนั้นเหมาะ — บังคับส่งทั้ง planner และ learned policy แล้ววิเคราะห์เปรียบเทียบ |

การให้นิสิตวัดเองว่า "learning ชนะ planning ที่ระดับความยากไหน" เป็นโจทย์ที่ดีกว่า
"จูน PPO ให้ได้คะแนนสูงสุด" เพราะไม่มีคำตอบในตำรา และเป็นระเบียบวิธีวิจัยจริง

> **ต้องประกาศกับนิสิตตั้งแต่วันแรกว่า planner ก็ได้คะแนนเต็มส่วน leaderboard**
> ไม่งั้นทีมที่ตั้งใจทำ RL ทั้งเทอมจะได้บทเรียนว่า "ความพยายามไม่มีความหมาย"
>
> **และต้องเทรน baseline จริงมาเทียบก่อนเปิดเทอมเสมอ** — ที่ CP463 สมมติฐานเรื่องคันโยกผิดไป
> 2 ใน 3 ตัว และสมมติฐานว่า learning จะชนะ planning ก็ผิดด้วย ถ้าไม่วัดก็จะไปรู้เอาตอนกลางเทอม
> ตอนที่แก้อะไรไม่ได้แล้ว

---

## 6. Contract ของสิ่งที่นิสิตส่ง

```
submission.zip
├── agent.py            # คลาส Agent (ดูล่าง)
├── weights.*           # optional — policy weights (.pt / .npz / .pkl)
├── config.json         # hyperparameter ที่ต้องใช้สร้าง agent
├── requirements.txt
└── SOURCES.md
```

```python
class Agent:
    def __init__(self, config: dict): ...

    def reset(self, episode_info: dict) -> None:
        """เริ่ม episode ใหม่ — ล้าง state ภายใน (สำคัญมากถ้าเป็น POMDP ที่ต้องจำแผนที่)
        episode_info มีแค่ข้อมูลที่ประกาศไว้ใน TaskSpec เช่น ขนาด grid — ไม่มีผังห้องและไม่มี seed"""

    def act(self, observation) -> int | np.ndarray:
        """คืน action ที่อยู่ใน action space"""
```

**ข้อกำหนด**

- `reset()` ต้องล้าง state ให้หมดจริง — ระบบสุ่มลำดับ episode เพื่อตรวจว่าคะแนนไม่เปลี่ยนตามลำดับ (ถ้าเปลี่ยน แปลว่ามี state รั่วข้าม episode)
- ต้อง **deterministic**: ถ้า policy เป็น stochastic ต้องเลือก argmax ตอนประเมิน หรือประกาศ `stochastic: true` แล้วระบบจะตรึง RNG seed และรันซ้ำหลายรอบ (§7)
- ห้ามเขียนไฟล์ถาวรข้ามระหว่าง run — ทุก run เริ่มจาก container สะอาด
- ไม่มี network · ไม่มี ground truth ของ episode · ไม่เห็น seed
- ถ้าไม่มี weights (agent ที่เขียนกฎมือล้วน) ก็ส่งได้ ระบบไม่บังคับว่าต้องมีการเรียนรู้ (เว้นแต่ผู้สอนเลือกทางเลือก B ใน §5)

---

## 7. Evaluation protocol และความแปรปรวน

คะแนน RL แกว่งกว่างาน supervised มาก เพราะมาจากทั้งความยากที่ต่างกันของแต่ละ episode และความสุ่มใน policy/environment

### 7.1 การรัน

```yaml
evaluation:
  seeds: [...]              # ทุกทีมรัน seed ชุดเดียวกันเป๊ะ
  rollouts_per_seed: 1      # ตั้ง > 1 ถ้า environment หรือ policy เป็น stochastic
  deterministic_policy: true
  aggregate: mean           # mean ของ episode_score ทุก (seed × rollout)
```

**ทุกทีมต้องเจอ episode ชุดเดียวกัน** — นี่ไม่ใช่แค่เรื่องความยุติธรรม แต่ทำให้เปรียบเทียบได้แม่นขึ้นมาก
เพราะความแปรปรวนที่มาจาก "ห้องนี้ยากกว่าห้องนั้น" ถูกหักล้างไปในการเทียบแบบจับคู่ (paired comparison)

### 7.2 การรายงานความไม่แน่นอน

- **CI ของคะแนนทีมเดียว**: bootstrap โดย **resample ที่ระดับ seed** ไม่ใช่ระดับ timestep (timestep ในหนึ่ง episode ไม่เป็นอิสระต่อกัน)
- **การเทียบสองทีม**: ใช้ผลต่างรายคู่ต่อ seed แล้วหา CI ของค่าเฉลี่ยผลต่าง — วิธีนี้แคบกว่าการเอา CI ของสองทีมมาดูว่าทับกันไหมมาก
- แสดง **worst-seed score** ควบคู่กับค่าเฉลี่ยเสมอ — agent ที่ดีเฉลี่ยแต่พังสนิทในบางห้อง ไม่ควรถูกมองว่าเทียบเท่า agent ที่เสถียร
- ถ้า sd ข้าม seed สูงกว่าช่องว่างระหว่างทีม ให้เตือนบนหน้า leaderboard ว่า **จำนวน seed ไม่พอจะแยกอันดับ** และเพิ่มจำนวน seed

### 7.3 การจัดการ agent ที่พัง

| กรณี | การจัดการ |
|---|---|
| `act()` โยน exception | episode นั้นได้คะแนน 0 · run ดำเนินต่อ · แสดง traceback ที่กรองแล้วให้นิสิต |
| `act()` เกิน `step_timeout_ms` | episode นั้นถือว่าล้มเหลว **ไม่ใช่ได้คะแนนน้อยลง** (wall-clock ไม่มีผลต่อคะแนน) |
| คืน action นอก action space | ปฏิเสธ submission ตั้งแต่ dry run |
| เกินเวลารวมของ run | run ล้มเหลวทั้งก้อน ไม่ขึ้น leaderboard |

---

## 8. Public / Private seeds

ใช้ตรรกะเดียวกับ prediction-based ทุกประการ — เหตุผลเต็มอยู่ที่
[prediction-based §1.1](prediction-based-supervised.md#11-ทำไม-test-ต้องแยกเป็น-public--private)

| ชุด | นิสิตรู้ | ใช้ทำอะไร |
|---|---|---|
| **training seeds** | รู้ช่วงเลข + มี generator | เทรนได้ไม่จำกัด |
| **public seeds** | ไม่รู้ค่า รู้แค่จำนวน | leaderboard ระหว่างเทอม |
| **private seeds** | ไม่รู้อะไรเลย | ตัดสินอันดับและเกรด |

- สามชุดต้อง **ไม่ทับกันเลย** และควรมาจากช่วงเลขที่แยกกันชัดเพื่อกันความผิดพลาด
- private seeds ควรมีจำนวนมากกว่า public หลายเท่า เพราะความแปรปรวนของ RL สูง
- ต้องมีกติกา **final pick สูงสุด 2 submission ที่ทีมเลือกเองล่วงหน้า** ([เหตุผล](prediction-based-supervised.md#12-กติกา-final-pick--ส่วนที่ขาดไม่ได้)) — ขาดข้อนี้ private ไม่ทำงาน

---

## 9. Sandbox และข้อจำกัดทรัพยากร

| | ค่าเริ่มต้น |
|---|---|
| เลนคิว | CPU (policy ขนาดเล็ก) — ใช้ GPU เฉพาะโจทย์ที่ observation เป็นภาพขนาดใหญ่ |
| เวลารวมต่อ run | 20 นาที |
| `step_timeout_ms` | 1000 (ตัวกันงานค้าง ไม่มีผลต่อคะแนน) |
| RAM | 8 GB |
| ขนาด submission | 200 MB |
| network | ไม่มี |

⚠️ **environment ต้องรันคนละ process กับ agent** — ถ้าอยู่ด้วยกัน นิสิตเอื้อมไปอ่าน state ภายในของ environment
(ผังห้องทั้งใบ, ตำแหน่งเป้าหมาย, ค่า seed) ได้ตรงๆ ด้วยโค้ดไม่กี่บรรทัด แล้วเล่นได้สมบูรณ์แบบโดยไม่ต้องเรียนรู้อะไรเลย
runner ถือ environment ไว้ · agent อยู่ใน sandbox · คุยกันผ่าน pipe ทีละ step
(รายละเอียดและต้นทุนที่ [README §10.4](../../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries))

### 9.1 Wire protocol ระหว่าง runner กับ agent

```
runner (trusted)                          sandbox container (untrusted)
  Env · seed · เฉลย   ──stdin──►   arena-agent-host  ──เรียก──►  Agent.act()
                      ◄─stdout──
                      ◄─stderr──   log ของนิสิต (print / traceback)
```

**ใช้ stdin/stdout ของ container เป็นช่องโปรโตคอล และ agent host ต้องย้าย fd 1 ไปที่ stderr
ก่อนโหลดโค้ดนิสิต**

```python
protocol_out = os.dup(1)   # เก็บ stdout จริงไว้เป็น fd สูงๆ สำหรับโปรโตคอล
os.dup2(2, 1)              # ทำให้ fd 1 ชี้ไปที่ stderr
# ...ตรงนี้ค่อย import agent.py ของนิสิต — print() ของเขาจะไปออก stderr
```

**ทำไมไม่ใช้ fd 3/4 (ซึ่งเป็นสิ่งที่ร่างแรกของ spec นี้เขียนไว้)** — สองเหตุผล

1. **`docker run` ส่ง fd เพิ่มเข้า container ไม่ได้** มันให้แค่ stdin/stdout/stderr
   ถ้าจะใช้ fd 3/4 ต้องไปทำ unix socket bind-mount เข้าไปแทน ซึ่งเพิ่มชิ้นส่วนที่พังได้อีกชิ้น
   โดยไม่ได้อะไรกลับมา
2. **การย้าย fd 1 ไปที่ stderr ปลอดภัยกว่า fd 3/4** — หลัง `dup2` แล้ว โค้ดนิสิตที่เขียนลง fd 1
   ตรงๆ (`os.write(1, ...)`) ก็ยังไปออก stderr ไม่ทำ stream พัง ต่างจาก fd 3/4 ที่ถ้านิสิต
   เผลอเขียนลง fd 3 จะทำโปรโตคอลเสียทันที

ผลลัพธ์ที่ต้องการยังเหมือนเดิมทุกข้อ: **นิสิต `print()` ได้ตามปกติและเห็น log ของตัวเอง**
โดยที่โปรโตคอลไม่พัง

**Framing** — length-prefixed msgpack: `uint32 LE` ของความยาว แล้วตามด้วย payload
ndarray ส่งเป็น `{"__nd__": {"dtype": str, "shape": [int], "data": bytes}}` (ไม่ใช้ pickle — pickle = รันโค้ดได้)

| ทิศทาง | ข้อความ | payload |
|---|---|---|
| → agent | `hello` | `{"protocol": 1, "agent_config": {...}}` — ข้อมูลที่ประกาศใน `TaskSpec` เท่านั้น **ห้ามมี seed หรือผังห้อง** |
| ← agent | `ready` | `{"agent_version": str}` |
| → agent | `reset` | `{"episode_info": {...}}` |
| ← agent | `ok` | `{}` |
| → agent | `act` | `{"obs": <ndarray หรือ dict ของ ndarray>}` |
| ← agent | `action` | `{"action": int \| ndarray}` |
| → agent | `close` | `{}` |
| ← agent | `error` | `{"type": str, "traceback": str}` — runner กรอง path ของระบบก่อนแสดงให้นิสิต |

**กติกา**

- runner เป็นฝ่ายเริ่มทุกข้อความ agent ตอบอย่างเดียว — agent ส่งเองไม่ได้นอกจาก `error`
- ทุกข้อความ `act` ต้องได้คำตอบภายใน `step_timeout_ms` เกินแล้ว **episode นั้นล้มเหลว ไม่ใช่ได้คะแนนน้อยลง** (§7.3)
- runner ตรวจ `action` ว่าอยู่ใน action space **ก่อน**ส่งเข้า environment เสมอ
- **ไม่มี seed และไม่มี ground truth เดินทางผ่านช่องนี้** — ข้อนี้คือทั้งหมดของ trust boundary
  ต่อให้ agent หนีออกจาก sandbox ได้ก็ยังไม่มีเฉลยให้อ่าน (หลักการเดียวกับการแยก stage ของ
  [prediction-based §5](prediction-based-supervised.md#5-pipeline-การประเมินผล))
- ไม่มี state ค้างข้าม episode — `reset` ต้องล้างจริง ระบบสลับลำดับ episode เพื่อตรวจข้อนี้

**ต้นทุน** — ราว 200,000 round trip ต่อ run (1,500 step × 130 episode) ที่ ~0.1 ms ≈ 20 วินาที
ยอมรับได้ และเป็นราคาที่ถูกมากเมื่อเทียบกับการที่โจทย์ทั้งเทอมถูกโกงได้ด้วย `import gc`

**การเทรนไม่ได้เกิดบนแพลตฟอร์ม** — นิสิตเทรนบนเครื่องตัวเอง/Colab แพลตฟอร์มประเมินผลอย่างเดียว
เหตุผลและทางเลือกในอนาคตดู [CP463 §10](../competitions/CP463/1-2026/vacuum-robot/overview.md#10-นโยบายการ-train)

**Artifact ขนาดเล็ก** (policy ทั่วไปไม่เกิน 200 MB) → เก็บถาวรได้ทุก submission ไม่ต้องมีนโยบายลบแบบ deep-learning template

**Replay ไม่ใช้ GPU และแทบไม่มีต้นทุน** — environment ต้องบันทึก trajectory เป็น **delta ต่อ timestep**
(ตำแหน่ง/สถานะที่เปลี่ยน + event flags) ไม่ใช่ snapshot ของ state เต็ม การวาดภาพเกิดบนเบราว์เซอร์ของนิสิตทั้งหมด
สำหรับ grid world ขนาด 20×20 คิดเป็นราว 1–2 KB ต่อ episode หลังบีบอัด → เก็บครบทุก submission ได้
รายละเอียดการคำนวณอยู่ที่ [README §10.3](../../README.md#103-การเก็บและแสดง-replay)

> **ข้อกำหนดของ environment**: ต้อง emit trajectory ที่ **เล่นซ้ำแล้วได้ state เดิมทุกเฟรม** จาก header + delta เท่านั้น
> ถ้า environment มีความสุ่ม (`action_noise`) ต้องบันทึกผลของการสุ่มลง delta ด้วย ไม่ใช่หวังว่า client จะสุ่มได้เหมือนกัน

---

## 10. ประเด็นที่ต้องตัดสินใจต่อ competition

| # | ประเด็น | หมายเหตุ |
|---|---|---|
| 1 | **จะรับมือกับ "planning ชนะ learning" อย่างไร** | §5 — ตัดสินใจก่อนออกแบบ environment เพราะกระทบ config โดยตรง |
| 2 | Observability | fully / partially — เปลี่ยนความยากทั้งโจทย์ |
| 3 | Stochasticity ของ transition | มีผลต่อจำนวน seed และ `rollouts_per_seed` ที่ต้องใช้ |
| 4 | score function | ต้องยุบเป็นสเกลาร์เดียว + ประกาศเกณฑ์ตัดสินเสมอ ([README §5.1](../../README.md#51-metric-เป็นของแต่ละโจทย์-ไม่ใช่ของแพลตฟอร์ม)) |
| 5 | จำนวน public / private seeds | ต้องมากพอให้ sd/√n เล็กกว่าช่องว่างระหว่างทีมที่อยากแยกออก |
| 6 | อนุญาต stochastic policy ไหม | ถ้าอนุญาตต้องเพิ่ม `rollouts_per_seed` → เวลารันเพิ่มเป็นเท่าตัว |
| 7 | แจก training seed generator ไหม | แนะนำให้แจก — ไม่งั้นนิสิตเทรนบน distribution ที่ไม่ตรงกับตอนวัด |
| 8 | มีคะแนนส่วนที่ประเมินวิธีการไหม | ทางเลือก D ใน §5 — ถ้ามี ต้องกำหนดรูปแบบรายงานและ rubric |
