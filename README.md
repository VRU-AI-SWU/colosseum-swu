# Arena — Gamified AI Course Competition Platform

แพลตฟอร์มสำหรับใช้ในการเรียนการสอนวิชา AI ที่เปลี่ยน course project ให้เป็นการแข่งขันตลอดทั้งเทอม
นิสิตพัฒนา model / AI agent ของตัวเอง อัพโหลดขึ้นมาทดสอบซ้ำได้เรื่อยๆ และเห็นผลของตัวเองเทียบกับทีมอื่นบน leaderboard แบบสด

> เอกสารนี้เป็น **design document ของแพลตฟอร์ม** (ยังไม่เริ่มเขียนโค้ด) ใช้เพื่อตกลงขอบเขตและ feature ก่อนลงมือสร้างจริง
> spec ของโจทย์และ task template ไม่ได้อยู่ในเอกสารนี้ — ดู [`docs/`](#โจทย์ที่ใช้จริง)

---

## สารบัญ

1. [เป้าหมายและหลักการออกแบบ](#1-เป้าหมายและหลักการออกแบบ)
2. [แนวคิดหลัก (Domain Model)](#2-แนวคิดหลัก-domain-model)
3. [โหมดการแข่งขัน (Evaluation Modes)](#3-โหมดการแข่งขัน-evaluation-modes)
4. [รูปแบบการรัน (Execution Models)](#4-รูปแบบการรัน-execution-models)
5. [การให้คะแนนและ Leaderboard](#5-การให้คะแนนและ-leaderboard)
6. [Gamification](#6-gamification)
7. [ความสุจริตทางวิชาการและการกันโกงระบบ](#7-ความสุจริตทางวิชาการและการกันโกงระบบ)
8. [Feature ฝั่งผู้สอน / TA](#8-feature-ฝั่งผู้สอน--ta)
9. [Feature ฝั่งนิสิต](#9-feature-ฝั่งนิสิต)
10. [สถาปัตยกรรมระบบ](#10-สถาปัตยกรรมระบบ)
11. [Tech Stack ที่เสนอ](#11-tech-stack-ที่เสนอ)
12. [Data Model (ร่าง)](#12-data-model-ร่าง)
13. [API / CLI (ร่าง)](#13-api--cli-ร่าง)
14. [ขอบเขต MVP และ Roadmap](#14-ขอบเขต-mvp-และ-roadmap)
15. [Non-Goals](#15-non-goals)
16. [ประเด็นที่ยังต้องตัดสินใจ](#16-ประเด็นที่ยังต้องตัดสินใจ)

**เอกสารโจทย์**: [โจทย์ที่ใช้จริง](#โจทย์ที่ใช้จริง)

---

## 1. เป้าหมายและหลักการออกแบบ

### เป้าหมาย

- ใช้กลไก **gamification** กระตุ้นให้นิสิตปรับปรุงงานตัวเองอย่างต่อเนื่องตลอดเทอม แทนที่จะทำทีเดียวตอนใกล้ส่ง
- ให้ **feedback วัดผลได้ทันที** หลังส่งงาน เพื่อให้เกิดวงจร ทดลอง → เห็นผล → ปรับปรุง
- ลดภาระ TA ในการตรวจงาน ด้วยการประเมินผลอัตโนมัติที่ **ยุติธรรม เท่าเทียม และทำซ้ำได้**
- ใช้ซ้ำได้กับวิชาอื่น โดยผู้สอนตั้งโจทย์ใหม่ได้เองโดยไม่ต้องแก้โค้ดแพลตฟอร์ม

### หลักการออกแบบ

| หลักการ | ความหมายในทางปฏิบัติ |
|---|---|
| **Task-agnostic core** | แกนกลาง (ทีม/ส่งงาน/คิว/leaderboard/เกรด) ไม่รู้จักโจทย์ ตัวโจทย์เป็น plugin |
| **Pluggable evaluator** | โจทย์ใหม่ = เขียน environment/scorer ใส่ container ตาม interface ที่กำหนด แล้ว register เข้าระบบ |
| **Reproducible by default** | ทุก run ผูกกับ seed + version ของ environment + hash ของ submission ต้องรันซ้ำได้ผลเดิม |
| **Hardware-independent scoring** | คะแนนต้องไม่ขึ้นกับสเปคเครื่องหรือความเร็ว CPU/GPU ใดๆ — วัดจากปริมาณที่นับได้ในเชิงตรรกะเท่านั้น (เช่น decision timestep) wall-clock ใช้ได้แค่เป็นตัวกันงานค้าง |
| **Fail-soft** | runner ล่ม/ไฟดับ/เน็ตหลุด ต้องไม่ทำให้ submission หาย ระบบต้อง retry ได้เอง |
| **แรงจูงใจ ไม่ใช่แรงกดดัน** | ออกแบบให้ทีมท้ายตารางยังมีเป้าหมายระยะสั้นที่ทำได้ ไม่ใช่แค่แข่งกับที่ 1 |

---

## 2. แนวคิดหลัก (Domain Model)

```
Organization (มหาวิทยาลัย/คณะ)
└── Course (วิชา + ภาคการศึกษา)            เช่น "CP463 1/2026"
    ├── Enrollment (นิสิต ↔ วิชา, role)
    ├── Team (กลุ่มโปรเจกต์)
    └── Competition (โจทย์)                 เช่น "Vacuum Robot Challenge"
        ├── TaskSpec (environment + config + metric)
        ├── Phase (ช่วงเวลา เช่น Warm-up / Main / Final)
        ├── Submission (สิ่งที่ทีมส่ง 1 ครั้ง)
        │   └── Run (การประเมิน 1 ครั้ง: public / private / rejudge)
        │       └── EpisodeResult (ผลรายตอน + replay)
        └── Leaderboard (public / private)
```

**ข้อกำหนดสำคัญ**

- **Submission ≠ Run** — submission 1 ครั้งถูกรันได้หลายครั้ง (public ตอนส่ง, private ตอนปิดเทอม, rejudge เมื่อพบบั๊ก) ประวัติทั้งหมดเก็บไว้ ไม่ทับกัน
- **Team ไม่ใช่ User** — คะแนนผูกกับทีม แต่ log ว่าใครเป็นคนกดส่ง (ใช้ดูการกระจายงานในกลุ่มได้)
- **Course-scoped ทุกอย่าง** — รองรับ 1–3 วิชาพร้อมกัน ผู้สอนแต่ละวิชาเห็นเฉพาะวิชาตัวเอง

---

## 3. โหมดการแข่งขัน (Evaluation Modes)

แพลตฟอร์มออกแบบให้รองรับหลายโหมด แต่ **ปีนี้ implement 2 โหมด** ตามลำดับความสำคัญ

### 3.1 Agent vs Environment ⭐ (priority สูงสุด — ใช้ในวิชา CP463 เทอม 1/2026)

นิสิตส่ง **policy/agent** ระบบรันหลาย episode ในสภาพแวดล้อมจำลองที่ผู้สอนกำหนด แล้ววัดผลจากพฤติกรรมของ agent

- เหมาะกับ: Reinforcement Learning, search/planning, game AI
- ผลลัพธ์ที่ได้: คะแนนรวม + สถิติรายตอน + **replay ให้ดูย้อนหลังได้**
- จุดเด่นทางการสอน: นิสิตเห็นภาพว่า agent ตัวเองทำอะไรผิด ไม่ใช่แค่เห็นตัวเลข
- **ไม่จำกัดตระกูลอัลกอริทึม** — model-free/model-based, on/off-policy, value/policy-based ส่งเข้ามาด้วย interface เดียวกันหมด

โหมดนี้มี 2 template ที่ใช้กลไกร่วมกัน (seed, replay, public/private split) แต่ต่างกันที่สิ่งที่ส่งและทรัพยากรที่นับ

| Template | นิสิตส่ง | ทรัพยากรที่นับ |
|---|---|---|
| [agent-vs-environment-rl](docs/task-templates/agent-vs-environment-rl.md) | policy ที่เทรนเอง | decision timestep |
| [llm-agent-tool-use](docs/task-templates/llm-agent-tool-use.md) | agent harness (ไม่ต้องเทรน) ทำงานกับ LLM ที่แพลตฟอร์มโฮสต์ให้ | token + tool call |

> ประเด็นสำคัญที่อยู่ในเอกสารเหล่านั้น — reward design เป็นส่วนหนึ่งของคำตอบ · การจัดการความแปรปรวนของคะแนน RL ·
> ปัญหา "planning ชนะ learning" · การโฮสต์ LLM กลางเพื่อให้ทุกทีมแข่งกันที่การออกแบบ agent ไม่ใช่ที่งบ API

### 3.2 Prediction-based (แบบ Kaggle — สำหรับวิชาอื่น)

นิสิตส่ง **โมเดลที่เทรนแล้ว + โค้ดเตรียมข้อมูล** ระบบรันบน hidden test set ที่นิสิตไม่มีสิทธิ์เข้าถึงเลย แล้วเทียบกับเฉลย

แบ่งเป็น 2 variant ที่ใช้กลไกเดียวกัน ต่างกันที่ artifact และเลนคิว

| Variant | นิสิตส่ง | เลน |
|---|---|---|
| `classical-ml` | scikit-learn model (pickle) + `preprocess.py` ที่มี `fit`/`transform` | CPU |
| `deep-learning` | PyTorch `state_dict` (.pt) + `model.py` + `dataset.py` | GPU |

ทั้งคู่รายงาน precision / recall / accuracy / F1 พร้อม **95% confidence interval (bootstrap)** และ ROC-AUC / PR curve
ตอนนี้โฟกัสที่ **supervised learning** ก่อน

> รายละเอียดทั้งหมด — contract ของโค้ด, การกัน data leakage, การแยก stage ให้เฉลยไม่อยู่ใน process เดียวกับโค้ดนิสิต,
> นโยบายเก็บ/ลบไฟล์: [`docs/task-templates/prediction-based-supervised.md`](docs/task-templates/prediction-based-supervised.md)

### 3.3 โหมดที่เผื่อโครงสร้างไว้ แต่ยังไม่ทำปีนี้

| โหมด | หมายเหตุ |
|---|---|
| **Agent vs Agent (tournament)** | agent ประกบกันแข่ง จัดอันดับด้วย Elo / Swiss / round-robin — ต้องมี match scheduler เพิ่ม |
| **Judge-based** | ให้ LLM หรือมนุษย์เป็นกรรมการตัดสินคำตอบ (แบบ Chatbot Arena) |

> โครงสร้าง `TaskSpec` + `Evaluator` interface ออกแบบเผื่อไว้ตั้งแต่แรก ให้เพิ่มโหมดใหม่ได้โดยไม่ต้องรื้อแกนกลาง

---

## 4. รูปแบบการรัน (Execution Models)

จุดต่างสำคัญคือ **โค้ดของนิสิตรันที่ไหน**

> **สถานะปัจจุบัน: โจทย์ทุกประเภทที่วางแผนไว้ใช้ Hosted Run ทั้งหมด**
> ตอนแรกออกแบบ Remote Worker ไว้สำหรับโจทย์ prediction-based แต่พอสรุปว่านิสิตส่ง sklearn pickle (เล็กมาก) กับ PyTorch state_dict
> ที่รันบน GPU server ของเราได้ (§3.2) โหมด Remote Worker ก็ไม่มีโจทย์ไหนใช้แล้ว
> **จึงเลื่อนออกจาก scope เวอร์ชันแรก** เก็บไว้เป็นทางออกสำรองถ้าเจอโจทย์ที่โมเดลใหญ่เกินหรือต้องเรียก API ภายนอก (§4.2 เขียนไว้เผื่ออนาคต)

### 4.1 Hosted Run (ค่าเริ่มต้น — ใช้กับโจทย์ RL ของ CP463)

นิสิตอัพโหลดโค้ด + weights → ระบบรันใน container บน **runner on-prem ในมหาวิทยาลัย**

```
นิสิต ──upload──> Cloud API ──job──> On-prem Runner ──docker run──> ผลคะแนน + replay
```

**ข้อดี**

- นิสิต**ไม่เห็น test set** เลย → ตัดปัญหาการโกงและการ overfit ที่ต้นทาง
- ฮาร์ดแวร์เท่ากันทุกทีม → ผลรันเสถียรและทำซ้ำได้ (หมายเหตุ: ถึงอย่างนั้นก็ยัง**ไม่ใช้ wall-clock คิดคะแนน** ตามหลักการข้อ hardware-independent scoring)
- ไม่ต้องพึ่งพาว่าเครื่องนิสิตเปิดอยู่หรือเปล่า
- ทำซ้ำได้ ตรวจสอบย้อนหลังได้

**ข้อจำกัด** — กินทรัพยากรเซิร์ฟเวอร์ และต้องทำ sandbox ให้แน่นหนา

**ข้อกำหนด sandbox**

- **แยก process ระหว่าง agent กับ environment เสมอ** — เฉลย, seed และ state ของโลกอยู่ฝั่ง runner ที่เชื่อถือได้เท่านั้น (ดู [§10.4](#104-ขอบเขตความไว้วางใจ-trust-boundaries))
- ไม่มี network (`--network none`) — ข้อยกเว้นเดียวคือโจทย์ LLM agent ที่เปิด egress ไปยัง **LLM gateway ภายในเส้นเดียว** แบบ default-deny ทุกอย่างอื่น
- จำกัด CPU / RAM / GPU / เวลาต่อ episode / เวลารวมต่อ run / ขนาดไฟล์ที่เขียนได้
- อ่านได้เฉพาะ input ของตัวเอง ไม่มีสิทธิ์แตะ ground truth
- รันด้วย non-root user, read-only rootfs, drop capabilities, ตั้ง pid limit

### 4.2 Remote Worker (โมเดลของนิสิตรันบนเครื่องนิสิตเอง) — เลื่อนออกจาก scope แรก

เก็บไว้เผื่อโจทย์ที่โมเดลใหญ่เกินกว่าจะอัพโหลด หรือต้องเรียกบริการภายนอก — นิสิตห่อ model ของตัวเองเป็นบริการที่ตอบ request ได้

> **การตัดสินใจสำคัญ: ใช้ pull ไม่ใช่ push**
>
> ไอเดียแรกคือให้เซิร์ฟเวอร์ของเรา "ยิง API ไปหาเครื่องนิสิต" แต่ในทางปฏิบัติเครื่องนิสิตอยู่หลัง NAT / WiFi มหาวิทยาลัย ไม่มี public IP และเปิด inbound port ไม่ได้
> ระบบจึงใช้ทิศทางกลับกัน: นิสิตรัน **worker client** ที่เปิดการเชื่อมต่อ **ออกมาหาเซิร์ฟเวอร์เรา** (WebSocket / long-poll) แล้วดึงงานไปทำ
>
> ```
> เครื่องนิสิต ──(outbound WS)──> Cloud API  ──แจก test batch──> เครื่องนิสิต ──ส่งผลกลับ──> Cloud API
> ```
>
> ผลลัพธ์เหมือนกันทุกประการ แต่ใช้งานได้จริงทุกเครือข่าย ไม่ต้องตั้งค่า port forwarding / ngrok / firewall

**สิ่งที่นิสิตต้องทำ** — implement interface เดียว:

```python
class Predictor:
    def load(self, ctx): ...                      # โหลด model ตอนเริ่ม worker
    def predict(self, instances: list) -> list:   # รับ batch ตอบ batch
        ...
```

แล้วรัน `arena worker --competition <slug> --token <TEAM_TOKEN>`

**ข้อควรระวังของโหมดนี้ (และวิธีรับมือ)**

| ความเสี่ยง | มาตรการ |
|---|---|
| นิสิตเห็น test input → เก็บสะสม / ตอบด้วยมือ | leaderboard ที่ตัดสินเกรดจริงต้องรันแบบ hosted เท่านั้น + จำกัดเวลาตอบต่อ instance + สอด honeypot instance ที่รู้คำตอบ |
| ฮาร์ดแวร์แต่ละทีมไม่เท่ากัน | โหมดนี้ **ห้ามใช้ metric ที่วัดความเร็ว** — ให้คะแนนจากความถูกต้องอย่างเดียว |
| เครื่องนิสิตดับกลางคัน | มีระบบ retry + timeout + สถานะ worker ให้เห็นบนหน้าเว็บ + ให้ resume batch ที่ค้าง |
| ตอบช้าเกินจนบล็อกคิว | จำกัดเวลาต่อ batch ถ้าเกินถือว่าตอบผิดในส่วนที่ยังไม่ตอบ |

**ทางเลือกสำรอง (โหมดง่ายสุด):** อัพโหลดไฟล์ผลทำนาย (CSV/JSONL) ตรงๆ ระบบเทียบกับเฉลย — ทำไว้เป็น fallback เสมอ เพราะพังยากที่สุด

---

## 5. การให้คะแนนและ Leaderboard

### 5.1 Metric เป็นของแต่ละโจทย์ ไม่ใช่ของแพลตฟอร์ม

แพลตฟอร์มไม่รู้จักสูตรคะแนนของโจทย์ใดๆ — `TaskSpec` ของแต่ละ competition เป็นผู้ประกาศ metric ของตัวเอง
สิ่งที่แพลตฟอร์ม**บังคับ**กับทุกโจทย์มีเท่านี้

| ข้อกำหนด | เหตุผล |
|---|---|
| ต้องยุบเหลือ **สเกลาร์ตัวเดียว** ต่อ submission (ยิ่งมากยิ่งดี) | ใช้จัดอันดับและพล็อตกราฟพัฒนาการ |
| ต้องประกาศ **เกณฑ์ตัดสินเสมอ** เป็นลำดับที่ชัดเจน | กันอันดับกำกวมตอนคะแนนเท่ากัน |
| **ห้ามใช้ wall-clock เป็นส่วนหนึ่งของคะแนน** | หลักการ hardware-independent scoring (§1) — ให้ใช้ปริมาณเชิงตรรกะ เช่น decision timestep, จำนวน query, จำนวน sample แทน |
| ต้อง **deterministic** เมื่อรันซ้ำด้วย seed เดิม | ตรวจสอบย้อนหลังและ rejudge ได้ |
| ต้องคืน **metric ประกอบ** แยกจากคะแนนหลักด้วย | ให้นิสิตเข้าใจว่าตัวเองแพ้เพราะอะไร |

> ตัวอย่างที่ implement จริงเป็นตัวแรก: [Coverage AUC + Completion Bonus ของ CP463](docs/competitions/CP463/1-2026/vacuum-robot/overview.md#5-การให้คะแนน)

### 5.2 Public / Private Leaderboard

หัวใจของการกันการ overfit leaderboard ซึ่งเป็นปัญหาใหญ่ที่สุดของการใช้ Kaggle ในห้องเรียน

| | Public Leaderboard | Private Leaderboard |
|---|---|---|
| ใช้ seed / test split ไหน | ชุด public (เปิดเผยจำนวน แต่ไม่เปิดผัง) | ชุด private ที่ซ่อนไว้ |
| เห็นเมื่อไร | ทันทีหลังส่ง ตลอดเทอม | เปิดพร้อมกันหลังปิดรับ submission |
| ใช้ทำอะไร | feedback + gamification | **ตัดสินอันดับและเกรดจริง** |
| นิสิตทำอะไรได้ | ส่งกี่ครั้งก็ได้ (ในโควตา) | เลือก submission ไว้ตัดสินได้สูงสุด 2 ชุด |

> ต้องประกาศให้ชัดตั้งแต่วันแรกว่าอันดับที่เห็นระหว่างเทอมไม่ใช่อันดับจริง — บทเรียนเรื่อง generalization ตรงนี้มีคุณค่าทางการสอนในตัวมันเอง

### 5.3 การแปลงคะแนนเป็นเกรด

หลีกเลี่ยงการให้เกรดตามอันดับตรงๆ (ทีมที่ 1 ได้ A, ทีมที่ 20 ได้ D) เพราะลงโทษคนที่อยู่ในห้องที่เก่ง
ระบบรองรับ **หลายวิธี ผู้สอนเลือก/ผสมได้**

1. **Threshold-based** (แนะนำเป็นหลัก) — ผ่าน baseline Bronze/Silver/Gold ได้คะแนนตามขั้น ทุกทีมได้ A ได้ถ้าเก่งพอ
2. **Percentile-based** — คิดคะแนนจาก percentile ในห้อง
3. **Rank bonus** — คะแนนพิเศษเล็กน้อยสำหรับ top 3 (เป็นของหวาน ไม่ใช่ของคาว)
4. **Participation/progress** — คะแนนจากการส่งสม่ำเสมอและพัฒนาการของตัวเอง

ผู้สอนกำหนดสูตรผสมได้ แล้ว **export เป็น CSV** เข้าระบบเกรดของมหาวิทยาลัย

---

## 6. Gamification

### 6.1 Leaderboard สด

- อัพเดตทันทีที่มี run เสร็จ พร้อมลูกศรบอกการเปลี่ยนอันดับ (▲2 / ▼1)
- กราฟ **คะแนนของทีมตัวเองตามเวลา** เทียบกับ median และ top-1 ของห้อง
- โหมด **alias นิรนาม** (ทีมเลือกเองได้) — ลดแรงกดดันของทีมท้ายตาราง ผู้สอนยังเห็นชื่อจริงเสมอ

### 6.2 Baseline Bot — เป้าหมายระยะสั้นที่จับต้องได้

แทนที่จะให้นิสิตแข่งกับที่ 1 อย่างเดียว ระบบวาง bot ของผู้สอนไว้บน leaderboard เป็นหมุดหมาย

| ระดับ | ความหมาย |
|---|---|
| 🥉 Bronze | "โค้ดทำงานได้แล้ว" — วิธีสุ่ม/วิธีตรงไปตรงมาที่สุด |
| 🥈 Silver | "agent มีกลยุทธ์แล้ว" — heuristic แบบ greedy |
| 🥇 Gold | "ดีกว่าวิธีคลาสสิกที่ไม่ได้เรียนรู้" — อัลกอริทึมดั้งเดิมที่จูนมาดี |
| 💎 Diamond | "ระดับ state-of-the-art ของโจทย์นี้" — solution ของผู้สอน |

การข้ามแต่ละขั้นปลดล็อก badge + คะแนนเกรด → **ทุกทีมมีเป้าหมายถัดไปที่ทำได้เสมอ**
แต่ละ competition กำหนด bot จริงของแต่ละระดับเอง (เช่น [ladder ของ CP463](docs/competitions/CP463/1-2026/vacuum-robot/overview.md#6-baseline-ladder))
คะแนนของ bot ต้องได้จากการรันจริงบน public split ชุดเดียวกับนิสิต แล้วตรึงไว้ทั้งเทอม

### 6.3 Replay Viewer (จุดขายของโหมด RL)

เล่นภาพย้อนหลังพฤติกรรมของ agent ในแต่ละ episode ได้ — เล่น/หยุด/เลื่อนเฟรม/ปรับความเร็ว/ดู heatmap ของ state ที่วนซ้ำ
(ตัว renderer เป็นของแต่ละ task — grid world ใช้ Canvas 2D ธรรมดา)

- นิสิตเห็นเลยว่า agent ตัวเองติดอยู่ตรงไหน วนซ้ำ หรือทำ action เสียเปล่า → debug ได้จริง ไม่ใช่เดาจากตัวเลข
- ผู้สอนหยิบ replay ขึ้นมาอธิบายในคาบเรียนได้
- **Replay of the week** — เลือก replay ที่น่าสนใจมาโชว์หน้าแรก (ทีมท้ายตารางก็มีโอกาสได้ขึ้นหน้าแรกจากการแก้ปัญหาที่สร้างสรรค์)

### 6.4 Badges / Achievements

`First Blood` (ส่งคนแรกของห้อง) · `Baseline Breaker` (ชนะ baseline แต่ละระดับ) · `Consistent` (ส่งติดต่อกัน N สัปดาห์) ·
`Comeback` (ขยับขึ้นเกิน 5 อันดับใน 1 สัปดาห์) · `Perfectionist` (ทำคะแนนเต็มได้ทุก seed) ·
`Robust` (ส่วนเบี่ยงเบนข้าม seed ต่ำสุด) · `Early Bird` (ส่งงานสุดท้ายก่อน deadline เกิน 24 ชม.)

> ตั้งใจให้ badge ครอบคลุมมิติอื่นนอกจาก "คะแนนสูง" — ความเสถียร ความสม่ำเสมอ พัฒนาการ ก็ได้รางวัล
> แต่ละ competition เพิ่ม badge เฉพาะโจทย์ของตัวเองได้ (เช่น CP463 มี `Minimalist` สำหรับทีมที่ใช้ timestep น้อยที่สุดโดยยังดูดครบ)

### 6.5 Season / Phase ตามปฏิทินเทอม

แบ่งเทอมเป็นช่วง แต่ละช่วงใช้ config ที่ต่างกัน (`Phase.config_override`) โดยรูปแบบที่แนะนำคือ

| ช่วง | จุดประสงค์ |
|---|---|
| **Warm-up** | ตั้งโจทย์ให้ง่ายกว่าจริง — เป้าหมายคือ "ทุกทีมมีชื่อบน leaderboard" ไม่ใช่การแข่ง |
| **Main** | config จริง แข่งกันยาว คะแนนส่วนใหญ่มาจากช่วงนี้ |
| **Final** | config ที่ยากขึ้น/ต่างออกไป — ทดสอบว่า generalize ได้จริง ไม่ใช่จำ config เดิม |

การเปลี่ยน config ระหว่างช่วงช่วยไม่ให้ทีมที่นำห่างตั้งแต่ต้นลอยตัว และทีมที่ตามอยู่ยังมีโอกาสไล่
แผนรายสัปดาห์จริงกำหนดที่เอกสารโจทย์ (เช่น [แผนของ CP463](docs/competitions/CP463/1-2026/vacuum-robot/overview.md#8-แผนตามสัปดาห์-phases))

### 6.6 การแจ้งเตือน

แจ้งเมื่อ: run เสร็จ · โดนแซง · แซงคนอื่นสำเร็จ · ผ่าน baseline ระดับใหม่ · ใกล้ deadline · run ล้มเหลว
ช่องทาง: ในเว็บ + อีเมล (+ Discord/LINE webhook ถ้าวิชาใช้)

---

## 7. ความสุจริตทางวิชาการและการกันโกงระบบ

| ความเสี่ยง | มาตรการ |
|---|---|
| **Overfit leaderboard** (ยิงมั่วจนบังเอิญได้คะแนนดี) | private leaderboard + โควตาส่ง (เช่น 5 ครั้ง/วัน/ทีม) + จำกัดจำนวน submission ที่เลือกไปตัดสิน |
| **Hardcode คำตอบ / จำ seed** | private seeds ไม่ซ้ำกับ public เลย + สุ่มผังห้องใหม่จาก seed ตอนรัน + ตรวจ agent ที่คะแนน public สูงผิดปกติแต่ private ตก |
| **แอบดู ground truth / หนีออกจาก sandbox** | container ไม่มีเน็ต, non-root, read-only fs, ไม่ mount เฉลย, จำกัด syscall |
| **ลอกโค้ดกันระหว่างทีม** | ตรวจความคล้ายของโค้ดอัตโนมัติทุก submission (แบบ MOSS) + รายงานคู่ที่คล้ายผิดปกติให้ผู้สอน |
| **ใช้ model จากภายนอกโดยไม่อ้างอิง** | บังคับแนบ `SOURCES.md` ตอนส่ง final + ผู้สอนตรวจทีมท็อป |
| **สมาชิกในทีมไม่ทำงาน** | log ว่าใครส่ง + ให้แต่ละทีมกรอก contribution statement ตอนจบ |
| **ทำผลซ้ำไม่ได้** | ทุก run เก็บ seed + env version + hash ของ submission; ทีมท็อป N ทีมถูก rejudge อัตโนมัติก่อนประกาศผล |
| **ยิงงานถล่มระบบ** | rate limit ต่อทีม + จำกัดงานพร้อมกัน 1 งาน/ทีม + คิวแบบยุติธรรม (fair-share ไม่ใช่ FIFO ล้วน) |

**Audit log** — เก็บทุกเหตุการณ์ (ส่งงาน, ผลรัน, การแก้คะแนนโดยผู้สอน, การเปลี่ยน config) แบบ append-only ย้อนดูได้เสมอ

---

## 8. Feature ฝั่งผู้สอน / TA

### จัดการวิชาและทีม
- สร้างวิชา, สร้าง enrollment code, import รายชื่อนิสิตจาก CSV
- สร้าง/แก้ไขทีม, กำหนดขนาดทีมสูงสุด, ย้ายสมาชิก, ดูว่าใครยังไม่มีทีม

### จัดการโจทย์
- สร้าง competition, เลือก task template, แก้ config ผ่านฟอร์มหรือ YAML
- **Preview environment** — สุ่มดูผังห้องที่จะเกิดจาก config ก่อนเปิดใช้จริง
- ตั้ง public/private seeds, โควตาส่ง, deadline, phase
- อัพโหลด baseline agent เพื่อวางหมุดหมาย Bronze/Silver/Gold
- เปิด/ปิดการรับ submission, ประกาศ private leaderboard

### ควบคุมการรัน
- ดูคิวและสถานะ runner แบบสด, ยกเลิก/จัดลำดับงานใหม่
- **Rejudge** — รันซ้ำทั้ง competition เมื่อแก้บั๊กใน environment (พร้อมเก็บผลเดิมไว้เทียบ)
- ดู log ของ run ที่ล้มเหลว, ปรับคะแนนด้วยมือพร้อมบันทึกเหตุผล, disqualify

### ติดตามและให้เกรด
- Dashboard: ทีมไหนยังไม่ส่ง, ทีมไหนคะแนนนิ่งมานาน (= ติดปัญหา ควรเข้าไปช่วย), การกระจายคะแนนของห้อง
- กราฟพัฒนาการรวมของห้องตลอดเทอม (ใช้ประเมินว่าโจทย์ยาก/ง่ายเกินไปสำหรับปีถัดไป)
- ตั้งสูตรแปลงคะแนน → เกรด แล้ว export CSV
- รายงานความคล้ายของโค้ดระหว่างทีม

---

## 9. Feature ฝั่งนิสิต

- **เข้าร่วมวิชา** ด้วย enrollment code, สร้าง/เข้าทีม
- **หน้าโจทย์** — คำอธิบาย, กติกา, spec ของ environment, ลิงก์ starter kit, deadline นับถอยหลัง
- **ส่งงาน** ผ่านหน้าเว็บ หรือ CLI (`arena submit`) — เหมาะกับ workflow ของนิสิต AI ที่อยู่กับ terminal
- **สถานะแบบสด** ระหว่างรอผล: queued → running (x/30 episodes) → done
- **ประวัติ submission** ของทีม พร้อมกราฟคะแนนเทียบระหว่างเวอร์ชันของตัวเอง และช่องให้ใส่โน้ตว่าครั้งนี้แก้อะไร
- **หน้าผลรายละเอียด** — คะแนนรายตอน, seed ไหนทำได้แย่ที่สุด, สถิติเสริม, **replay ดูย้อนหลัง**
- **Log ที่ debug ได้** — stdout/stderr ของ agent ตัวเอง (กรอง path/ข้อมูลระบบออก) และข้อความ error ที่อ่านรู้เรื่องเมื่อ crash หรือ timeout
- **Dry run** — ทดสอบกับ sample environment 1–2 ตอน โดยไม่กินโควตา ใช้เช็คว่าแพ็กไฟล์ถูกต้อง
- **หน้าโปรไฟล์ทีม** — badge ที่ได้, อันดับสูงสุดที่เคยทำได้, จำนวน submission
- **เลือก final submission** ก่อนปิดรับ (สูงสุด 2 ชุด)
- **แนบรายงาน + SOURCES.md** ตอนส่งงานสุดท้าย

---

## 10. สถาปัตยกรรมระบบ

### 10.1 ภาพรวม (Hybrid: web บน cloud + runner ในมหาวิทยาลัย)

```
┌──────────────────────── CLOUD ────────────────────────┐
│  Web App (นิสิต/ผู้สอน)                                │
│  API Server ── Postgres ── Redis (queue)              │
│                └── Object Storage (submissions,       │
│                     replays, datasets, logs)          │
└──────────────────┬────────────────────────────────────┘
                   │  outbound WebSocket เท่านั้น
                   │  (runner เป็นฝ่ายต่อออกมา — ไม่ต้องเปิด port เข้ามหาลัย)
┌──────────────────┴──────── ON-PREM (มหาวิทยาลัย) ─────┐
│  Runner Agent (daemon)                                │
│   ├── CPU worker pool   ── docker run (sandbox)       │
│   └── GPU worker (RTX 3090) ── docker run --gpus      │
│  Local cache: base images, datasets, ground truth     │
└───────────────────────────────────────────────────────┘
```

**ทำไม runner ต้องเป็นฝ่ายต่อออก** — เครื่องในมหาวิทยาลัยมักอยู่หลัง firewall ที่ขอเปิด inbound port ยากหรือช้า
ให้ runner เปิด WebSocket ออกไปหา cloud แล้วรอรับงาน จะติดตั้งได้ทันทีโดยไม่ต้องยุ่งกับฝ่าย IT และปลอดภัยกว่า (ไม่มี port เปิดรับจากอินเทอร์เน็ต)

**ground truth และ private seeds อยู่ที่ on-prem เท่านั้น** ไม่ต้องอัพขึ้น cloud → ลดความเสี่ยงข้อมูลรั่ว

### 10.2 Job Lifecycle

```
submit → validate (ขนาด/โครงสร้างไฟล์/manifest)
       → enqueue (แยกเลน cpu / gpu, fair-share ต่อทีม)
       → runner claim (lease + heartbeat)
       → pull image + materialize submission
       → run N episodes ใน sandbox (แต่ละตอน seed คงที่)
       → รวมคะแนน + บีบอัด replay
       → upload artifacts + report ผลกลับ cloud
       → อัพเดต leaderboard + ยิง notification
```

- **Lease + heartbeat** — ถ้า runner หายไป งานกลับเข้าคิวอัตโนมัติ ไม่ค้าง
- **Idempotent** — รันซ้ำด้วย `run_id` เดิมไม่ทำให้คะแนนซ้ำซ้อน
- **แยกเลน CPU/GPU** — งาน RL grid world เข้าเลน CPU ได้ ไม่ไปแย่ง 3090 กับงาน prediction ของวิชาอื่น

### 10.3 การเก็บและแสดง Replay

> **หลักการ: เซิร์ฟเวอร์เก็บแค่ "log ของสิ่งที่เกิดขึ้น" การวาดภาพเกิดบนเบราว์เซอร์ของนิสิตทั้งหมด**
> ผลคือ replay viewer **ไม่ใช้ GPU เลย** และแทบไม่เพิ่มภาระเซิร์ฟเวอร์ — ไฟล์ replay ถูกเสิร์ฟเป็น static file เหมือนรูปภาพ

**สิ่งที่ห้ามทำ** — อย่าเรนเดอร์วิดีโอ/GIF ฝั่งเซิร์ฟเวอร์ นั่นคือทางเดียวที่ replay จะกลายเป็นภาระจริง
(ffmpeg ต่อ episode × ทุก submission) ถ้านิสิตอยากได้ GIF ไว้แปะรายงาน ให้ export จาก canvas ในเบราว์เซอร์ตอนกดเอง

#### รูปแบบข้อมูล — delta ไม่ใช่ snapshot

```
header : seed, config version, ขนาด grid, ผังเริ่มต้น (bitpack ~100 B)
ต่อ timestep (~4 B) : ตำแหน่งหุ่น (uint16) · action (uint8) · event flags (uint8: cleaned/collision/slip/sticky-fail)
```

state เต็มของแต่ละเฟรม **สร้างใหม่ได้จากการเล่น delta ตั้งแต่ต้น** ฝั่ง client จึงไม่ต้องเก็บ 400 ช่อง × 1000 เฟรม
ส่วน heatmap ช่องที่เดินซ้ำก็คำนวณจาก trajectory เดียวกันในเบราว์เซอร์

#### ต้นทุนจริง (โจทย์ grid world 20×20, 1000 timestep, 30 episodes/run)

| รายการ | ปริมาณ |
|---|---|
| ต่อ episode | ~4 KB ก่อนบีบอัด → **~1–2 KB** หลัง zstd (ลำดับ action ซ้ำเยอะ บีบได้ดีมาก) |
| ต่อ run (30 episodes) | **~40–60 KB** |
|  ทั้งเทอม (10 ทีม × 5 ครั้ง/วัน × 5 วัน × 7 สัปดาห์ ≈ 1,750 run) | **~100 MB** |
| GPU | **0** |
| CPU overhead ตอนรัน | ~1–3% (แค่ append 4 ไบต์ต่อ step) |
| ฝั่งนิสิต | ดาวน์โหลด ~60 KB แล้ววาด Canvas 2D — เบากว่าเว็บทั่วไป |

**สรุป: สำหรับโจทย์ grid world เก็บ replay ครบทุก submission ได้สบาย** ไม่ต้องมีนโยบายลบ
(นโยบายเดิมที่ว่า "เก็บเฉพาะ run ล่าสุด + ดีที่สุด" ตั้งไว้ตอนยังไม่ได้คำนวณ — 200 MB ต่อเทอมไม่คุ้มที่จะไปตัดทิ้ง
และการมี replay ครบทำให้นิสิตย้อนดูพัฒนาการของตัวเองได้ ซึ่งเป็นหนึ่งใน feature ที่ตั้งใจไว้ตั้งแต่ต้น)

#### โจทย์ LLM agent ต่างออกไปมาก

trace ของ Competition 2 คือข้อความ ไม่ใช่ตัวเลข — ใหญ่กว่า grid replay ประมาณ **20–100 เท่า**

| รายการ | ปริมาณ |
|---|---|
| ต่อ task (~10 LLM call, budget 30k token) | ~120 KB ดิบ |
| ต่อ run (40 instance) | ~5 MB ดิบ → **~1 MB** หลัง dedup + บีบอัด |
| ทั้งเทอม | **~3–4 GB** |

**dedup สำคัญกว่าการบีบอัด** — system prompt กับ tool schema ถูกส่งซ้ำทุก call ในหนึ่ง task
เก็บครั้งเดียวแล้วอ้างอิงด้วย hash ตัดขนาดลงได้เกินครึ่งก่อนจะบีบอัดด้วยซ้ำ

นโยบายสำหรับ trace: เก็บครบเฉพาะ **run ล่าสุด + run ที่ดีที่สุดของทีม + final pick**
ส่วน run อื่นเก็บเฉพาะ instance ที่ **ล้มเหลว** (ซึ่งเป็นอันที่นิสิตอยากดูอยู่แล้ว) และเก็บสรุปของ instance ที่ผ่าน

### 10.4 ขอบเขตความไว้วางใจ (Trust Boundaries)

หลักการเดียวที่ครอบคลุมทุกอย่างในหัวข้อนี้: **ป้องกันเชิงโครงสร้าง ไม่ใช่เชิงการตั้งค่า**
— อย่าออกแบบให้ความปลอดภัยขึ้นกับ "โค้ดนิสิตจะไม่ทำสิ่งนั้น" แต่ให้ออกแบบจนสิ่งที่ต้องปกป้อง**ไม่ได้อยู่ในกล่องเดียวกับโค้ดนิสิตตั้งแต่แรก**

#### แยก process ระหว่าง agent กับ environment

⚠️ **ห้ามรัน agent ของนิสิตใน process เดียวกับ environment**

```python
# ❌ แบบนี้ agent เอื้อมไปหยิบเฉลยได้ตรงๆ
env = VacuumEnv(config); env.reset(seed=50042)
while not done:
    action = agent.act(obs)     # agent อยู่ใน process เดียวกับ env
```

นิสิตเขียนแค่ `import gc` แล้วไล่หา object ของ env ก็อ่านผังห้องทั้งใบและค่า seed ได้ทันที → เล่นได้สมบูรณ์แบบทุก episode โดยไม่ต้องเทรนอะไรเลย

```
runner process (trusted)              sandbox container (untrusted)
  VacuumEnv · seed · เฉลย  ──obs──►        Agent.act()
                          ◄─action─
```

ต้นทุน: round trip ราว 200,000 ครั้งต่อ run (1,500 step × 130 episode) ที่ ~0.1 ms ≈ 20 วินาที — รับได้สบาย
เป็นหลักการเดียวกับการแยก stage ของโจทย์ prediction-based ([template §5](docs/task-templates/prediction-based-supervised.md#5-pipeline-การประเมินผล))

#### สิ่งที่เป็นความลับจริง (สั้นกว่าที่คิด)

| | ลับไหม | เหตุผล |
|---|---|---|
| ค่า **private seeds** / private instance | 🔒 สูงสุด | รู้แล้วเทรนเจาะได้ → private leaderboard ไร้ความหมาย |
| ค่า public seeds | 🔒 รองลงมา | รู้แล้ว overfit ได้ ทำให้ feedback ระหว่างเทอมเสียคุณค่ากับทุกคน |
| **ย่าน**ที่ seed ถูกสุ่มมา (เช่น `20000–29999`) | 🔓 ไม่ลับ | ย่านมี 10,000 ค่า ใช้จริง 30 ค่า — รู้ย่านแล้วยังเดาไม่ได้ · ⚠️ **แต่ต้องประกาศเป็นย่านที่กว้างกว่าจำนวนที่ใช้มากๆ** ถ้าประกาศว่า "public คือ 20001–20030" สำหรับ 30 seed ก็เท่ากับเปิดค่าไปทั้งชุด |
| เฉลยของ Competition 2 · คลัง prompt injection ชุด private | 🔒 | — |
| baseline agent ระดับ Diamond | 🔒 | ลอกได้ |
| **golden value ของ baseline** | 🔓 **ไม่ลับ** | เราโชว์บน leaderboard อยู่แล้ว — รู้ว่าต้องทำเท่าไรถึงผ่าน Gold ไม่ได้ช่วยให้ทำได้ |
| โค้ด environment · scorer · แพลตฟอร์ม | 🔓 ไม่ลับ | ความปลอดภัยต้องไม่ขึ้นกับการปิดโค้ด |

**Conformance test ใช้ seed ของตัวเองแยกต่างหาก** (คนละช่วงกับ train/public/private) → ทั้งชุดเทสต์และ golden value เปิดเผยได้
ผลพลอยได้: **นิสิตรัน conformance test เองได้เพื่อยืนยันว่า environment ในเครื่องตัวเองตรงกับตัวที่ใช้ตัดสิน** ตัดภาระ support ไปมาก

#### ของลับอยู่ที่ไหน

```
เครื่อง GPU ในมหาวิทยาลัย
├── /srv/arena/app/        ← โค้ดแพลตฟอร์ม (repo สาธารณะ)
├── /srv/arena/envs/       ← environment package (สาธารณะ)
└── /srv/arena/secrets/    ← 🔒 clone จาก private repo · mount read-only เข้าเฉพาะ runner process
    └── cp463-1-2026.yaml     ไม่เข้า container ของนิสิตเด็ดขาด
```

เก็บประวัติและสำรองใน **private repo แยก** ไม่ใช่ไฟล์ลอยบนเครื่อง — เพื่อให้ย้อนดูได้ว่าเคยใช้ seed ชุดไหนถ้าต้องเปลี่ยนกลางเทอม

### 10.5 โครงสร้าง Repository

**repo แบ่งตามขอบเขตความไว้วางใจ ไม่ใช่ตาม competition**

```
colosseum-swu/                    (public · MIT)
├── core/                         # ไม่รู้จักโจทย์ — ทีม, submission, คิว, leaderboard, เกรด
├── runners/                      # หนึ่งตัวต่อ "ประเภทโจทย์" = task template
│   ├── agent_env/                #   RL: episode loop, replay, แยก process (§10.4)
│   ├── prediction/               #   supervised: 2-stage predict → score, bootstrap CI
│   └── llm_agent/                #   LLM gateway, tool layer, trace
├── envs/                         # หนึ่งโฟลเดอร์ต่อ "โจทย์" — แต่ละตัวมี pyproject.toml ติดตั้งแยกได้
│   ├── cp463-vacuum/
│   └── cp463-intdoc/
├── web/
└── docs/

colosseum-hypogeum/               (private · all rights reserved · clone เฉพาะเครื่อง GPU)
├── tools/make_seeds.py           # สร้างชุด seed + ตรวจว่า public/private ยากพอๆ กัน
└── cp463-1-2026/
    ├── vacuum/seeds.yaml         # ✅ มีแล้ว — ค่า seed ของ 3 phase + Diamond agent (ภายหลัง)
    └── intelligence-document/    # seeds + คลัง prompt injection ชุด private (ก่อนสัปดาห์ 8)
```

**แบ่งตาม competition ไม่ใช่ตามชนิดของไฟล์** — เพราะของลับเกือบทุกอย่างมีอายุเท่ากับ competition
และเวลาปิดเทอมแล้วอยากเปิดของเก่าเป็น archive จะได้ย้ายทั้งโฟลเดอร์

**เกณฑ์ตรวจว่าแบ่งถูก**

> การเพิ่ม competition ของวิชาใหม่ ควรแตะแค่ `envs/` กับไฟล์ config — **ถ้าต้องแก้ `core/` แปลว่าออกแบบผิด**

ใช้ข้อนี้เป็นแบบทดสอบระหว่างเขียนโค้ดได้เลย: ทุกครั้งที่อยากใส่ `if competition == "vacuum"` ลงใน `core/`
แปลว่ามีบางอย่างที่ควรย้ายไปอยู่ใน runner หรือ config แทน

**เหตุผลที่ไม่แยก repo ต่อ competition** — ทีมพัฒนามีคนเดียวถึงสองคน การแก้ protocol ระหว่าง runner กับ core ควรจบใน commit เดียว
ไม่ใช่ต้องไล่ปล่อยเวอร์ชันข้าม repo · แยกเมื่อไหร่ค่อยว่ากันตอนที่ (ก) มีคนอื่นมาดูแลและต้องคุมสิทธิ์แยก
(ข) dependency หนักจนทำ CI พังข้ามกัน หรือ (ค) อยากเปิด source บางโจทย์แต่ไม่เปิดตัวอื่น

**นิสิตไม่เคย clone repo** — ได้ environment ผ่าน `pip install` จาก wheel ที่ release ไว้ จึงไม่ต้องโหลดทั้งแพลตฟอร์มและ dependency ไม่ปนกัน

---

## 11. Tech Stack ที่เสนอ

| ส่วน | เทคโนโลยี | เหตุผล |
|---|---|---|
| Frontend | Next.js (React) + TypeScript + Tailwind + shadcn/ui | ทำ dashboard/leaderboard เร็ว, replay viewer ใช้ Canvas ธรรมดาพอ |
| Backend API | **Python + FastAPI** | ผู้สอน/TA เขียน scorer และ environment ด้วย Python อยู่แล้ว — ไม่ต้องข้ามภาษา |
| Database | PostgreSQL | ต้องใช้ transaction กับการอัพเดตคะแนน/อันดับ |
| Queue | Redis + arq (หรือ Celery) | เบา ติดตั้งง่าย พอกับสเกล 1–3 วิชา |
| Object Storage | S3-compatible (Cloudflare R2 / MinIO) | เก็บ submission, replay, log |
| Runner | Python daemon + Docker SDK | รัน container, คุมทรัพยากร, รายงานผล |
| Sandbox | Docker (+ พิจารณา gVisor ถ้าต้องการแน่นกว่า) | มาตรฐาน ใช้ GPU ผ่าน nvidia-container-toolkit ได้ |
| Auth | Google OAuth (บัญชีมหาวิทยาลัย) + enrollment code | ไม่ต้องจัดการรหัสผ่านเอง |
| Deployment | Docker Compose ทั้งฝั่ง cloud และ on-prem | ทีมเล็ก ดูแลง่าย ไม่ต้องใช้ Kubernetes |
| Observability | structured log + Prometheus metrics (คิว, runner, อัตราล้มเหลว) | ต้องรู้ทันทีเมื่อคิวตันคืนก่อน deadline |

---

## 12. Data Model (ร่าง)

```
users(id, email, name, role, created_at)
courses(id, code, name, term, owner_id, settings)
enrollments(id, course_id, user_id, role)              # student | ta | instructor
teams(id, course_id, name, alias, join_code)
team_members(team_id, user_id, joined_at)

competitions(id, course_id, slug, title, description,
             task_type, execution_mode, config_yaml, env_version,
             opens_at, closes_at, quota_per_day, max_final_submissions, status)
phases(id, competition_id, name, starts_at, ends_at, config_override)

submissions(id, competition_id, team_id, submitted_by,
            artifact_url, artifact_sha256, note, is_final_pick, created_at)
runs(id, submission_id, kind, status, runner_id,
     score, metrics_json, error_message, started_at, finished_at)
     # kind: public | private | dryrun | rejudge
episode_results(id, run_id, seed, score, metrics_json, replay_url)

leaderboard_entries(competition_id, team_id, kind, best_run_id,
                    score, rank, previous_rank, updated_at)
badges(id, code, name, description, icon)
team_badges(team_id, badge_id, competition_id, awarded_at)

runners(id, name, capabilities, status, last_heartbeat)
audit_logs(id, actor_id, action, target_type, target_id, payload, created_at)
```

---

## 13. API / CLI (ร่าง)

### REST API (ตัวอย่าง)

```
POST   /api/competitions/{slug}/submissions      อัพโหลด submission
GET    /api/competitions/{slug}/leaderboard      ?kind=public|private
GET    /api/submissions/{id}                     สถานะ + คะแนน
GET    /api/runs/{id}/episodes                   ผลรายตอน + ลิงก์ replay
POST   /api/submissions/{id}/final-pick          เลือกไปตัดสิน

# Runner protocol (WebSocket, runner เป็นฝ่ายต่อออก)
WS     /api/runner/connect                       register + heartbeat + รับงาน + ส่งผล

# Remote worker ของนิสิต (โหมด prediction-based)
WS     /api/worker/connect                       รับ test batch + ส่งผลทำนายกลับ
```

### CLI

```bash
arena login                                   # เข้าสู่ระบบด้วย Google
arena init cp463-vacuum-1-2026                # ดาวน์โหลด starter kit ของ competition
arena eval --local --seeds public             # ทดสอบในเครื่องตัวเองก่อนส่ง
arena submit --note "เพิ่ม frontier exploration"
arena status                                  # ดูสถานะ run ล่าสุด
arena leaderboard                             # ดูอันดับใน terminal
arena replay <run-id> --episode 3             # เปิด replay
arena worker --competition <slug>             # โหมด remote worker (prediction-based)
```

---

## 14. ขอบเขต MVP และ Roadmap

### M0 — โครงกระดูก
auth (Google OAuth) · course/enrollment/team · competition CRUD · โครง DB · Docker Compose ทั้งชุด

### M1 — แข่งได้จริง (ต้องเสร็จก่อนนิสิตเริ่ม project) 🎯
[CP463 Competition 1](docs/competitions/CP463/1-2026/vacuum-robot/overview.md):
✅ environment + config 3 phase + scorer + baseline agents + conformance test ([`envs/cp463-vacuum`](envs/cp463-vacuum/)) ·
✅ runner + sandbox ([`runners/agent_env`](runners/agent_env/)) · ✅ ตรวจสอบ submission ·
✅ คิว + โควตา + leaderboard ระดับตรรกะ ([`core`](core/)) ·
⬜ ผูก `core` กับ Postgres · ⬜ API + CLI อัพโหลด · ⬜ runner daemon ที่ต่อ WebSocket ·
⬜ หน้าเว็บ leaderboard · ⬜ starter kit ที่แจกนิสิต

> เกณฑ์ว่า M1 พร้อมใช้: ทีมทดสอบส่ง agent แล้วเห็นคะแนนขึ้น leaderboard ได้ครบวงจร โดยไม่มีใครต้องเข้า SSH

### M2 — Gamification (ต้องเสร็จภายในสัปดาห์ที่ 3)
replay viewer · baseline bot Bronze/Silver/Gold · badges · กราฟพัฒนาการ · การแจ้งเตือน · phase/season · alias นิรนาม

### M3 — ปิด competition ได้ (ต้องเสร็จภายในสัปดาห์ที่ 6) ⏰
private leaderboard + การเลือก final submission · rejudge · ตรวจความคล้ายของโค้ด · สูตรเกรด + export CSV · dashboard ผู้สอน

> **กำหนดนี้เร็วกว่าที่คิด** — CP463 Competition 1 ปิดที่สัปดาห์ที่ 7 และต้องเปิด private leaderboard ตอนนั้นเลย
> ไม่ใช่รอปลายเทอม ดังนั้น M3 ต้องพร้อมตั้งแต่กลางเทอม

### M4 — LLM Agent (ต้องเสร็จภายในสัปดาห์ที่ 8) ⏰
[template `llm-agent-tool-use`](docs/task-templates/llm-agent-tool-use.md): **LLM gateway** (vLLM + token metering + budget enforcement) ·
egress policy ให้ container ต่อ gateway ได้เส้นเดียว · tool layer + environment state · trace viewer ·
[CP463 Competition 2](docs/competitions/CP463/1-2026/intelligence-document/overview.md): registration environment + task generator + baselines

> ส่วนที่แพลตฟอร์มยังไม่มีเลยคือ LLM gateway — ควรเริ่มพัฒนาคู่ขนานระหว่างที่ Competition 1 กำลังรัน (สัปดาห์ 1–6)

### M5 — วิชาอื่น (prediction-based)
[template `classical-ml`](docs/task-templates/prediction-based-supervised.md): แจกข้อมูล train/val · รับ pickle + `preprocess.py` · ตรวจ leakage · metrics + bootstrap CI + curves ·
จากนั้น [template `deep-learning`](docs/task-templates/prediction-based-supervised.md): GPU lane · state_dict + `model.py`/`dataset.py` · นโยบายลบไฟล์ · ขัดเกลา multi-course

### หลังจากนั้น
agent vs agent tournament (Elo/Swiss) · training credit queue บน GPU · public archive ของโจทย์ปีก่อน · REST API สาธารณะ

---

## 15. Non-Goals

สิ่งที่ **ไม่ทำ** ในเวอร์ชันแรก เพื่อกันขอบเขตบาน

- ไม่เป็น LMS — ไม่มีคลิปเรียน ควิซ หรือระบบเช็คชื่อ (ใช้ระบบของมหาวิทยาลัยต่อไป)
- ไม่เป็นแพลตฟอร์ม train model ให้นิสิต — ประเมินผลอย่างเดียว นิสิต train บนเครื่องตัวเอง/Colab (ดูประเด็นค้างข้อ 2)
- ไม่มี IDE / notebook ในเว็บ
- ไม่มีระบบ forum/discussion (ใช้ Discord หรือ LINE ของวิชาไปก่อน)
- ไม่รองรับผู้ใช้นอกมหาวิทยาลัย
- ไม่ทำ mobile app (หน้าเว็บ responsive พอ)

---

## 16. ประเด็นที่ยังต้องตัดสินใจ

| # | ประเด็น | ตัวเลือก / ข้อพิจารณา |
|---|---|---|
| 1 | ขนาดทีมและจำนวนทีม | มีผลต่อโควตาคิวและ throughput ที่ต้องรองรับก่อน deadline |
| 2 | ให้ train บนแพลตฟอร์มไหม | ตอนนี้ตอบว่าไม่ ถ้าจะเปิดต้องทำ training credit + คิวแยก + โควตาชั่วโมง/ทีม |
| 3 | โควตาส่งต่อวัน | 5 ครั้ง/วัน/ทีม เป็นจุดตั้งต้น ปรับตามความแรงของ runner จริง |
| 4 | Cloud provider และงบ | มีผลต่อการเลือก object storage และวิธี deploy |
| 5 | บังคับ Google OAuth ของมหาวิทยาลัยหรือรับ Gmail ทั่วไป | กระทบการยืนยันตัวตนและการ import รายชื่อ |
| 6 | ภาษาของ UI | ไทย / อังกฤษ / สองภาษา — กระทบงาน i18n ตั้งแต่ M0 |
| 7 | นโยบายเก็บข้อมูลหลังจบเทอม | เก็บ submission ไว้กี่ปี, เปิดเป็น archive สาธารณะไหม |
| 8 | แผนสำรองเมื่อ runner ล่มช่วงใกล้ deadline | ขยาย deadline อัตโนมัติ? มีเครื่องสำรอง? |

> ประเด็นค้างที่เป็นเรื่องของโจทย์แต่ละวิชา (เช่น observation mode, `max_steps`, น้ำหนัก penalty) อยู่ในเอกสารโจทย์นั้นๆ
> — ของ CP463 ดู [§11](docs/competitions/CP463/1-2026/vacuum-robot/overview.md#11-สิ่งที่ต้องตัดสินใจทดสอบก่อนเปิดเทอม)

---

## โจทย์ที่ใช้จริง

เอกสารโจทย์แยกออกจาก README เพื่อให้แกนกลางยัง task-agnostic

### Task Template (ใช้ซ้ำได้ทุกวิชา) — `docs/task-templates/`

| Template | โหมด | ใช้กับ |
|---|---|---|
| [prediction-based-supervised](docs/task-templates/prediction-based-supervised.md) | Prediction-based | supervised learning — variant `classical-ml` (sklearn) และ `deep-learning` (PyTorch) |
| [agent-vs-environment-rl](docs/task-templates/agent-vs-environment-rl.md) | Agent vs Environment | reinforcement learning / search / planning |
| [llm-agent-tool-use](docs/task-templates/llm-agent-tool-use.md) | Agent vs Environment | LLM agent ที่ใช้ tool — แพลตฟอร์มโฮสต์โมเดลกลางให้ |

### โจทย์รายวิชา — `docs/competitions/<COURSE>/<TERM>/<competition>/`

```
docs/competitions/CP463/1-2026/
├── term_project.md              ← ภาพรวมทั้งเทอม: แมปกับตารางสอน, ปฏิทิน, สัดส่วนคะแนน
├── list-of-topics.html          ← โครงสร้างรายวิชา (อ้างอิง)
├── vacuum-robot/
│   ├── overview.md              ← ภาพรวมโจทย์ + เหตุผลเชิงออกแบบ
│   ├── environment-spec.md      ← build spec ระดับ implement
│   └── calibration-2026-08.md   ← ผลการวัดจริงที่ใช้ตรึงค่า config
└── intelligence-document/
    └── overview.md
```

| วิชา | ช่วง | โจทย์ | Template | เอกสาร |
|---|---|---|---|---|
| CP463 | 1/2026 | ภาพรวม term project | — | [term_project.md](docs/competitions/CP463/1-2026/term_project.md) |
| CP463 | สัปดาห์ 1–7 | Vacuum Robot | [agent-vs-environment-rl](docs/task-templates/agent-vs-environment-rl.md) | [overview](docs/competitions/CP463/1-2026/vacuum-robot/overview.md) · [environment-spec](docs/competitions/CP463/1-2026/vacuum-robot/environment-spec.md) 📐 · [calibration](docs/competitions/CP463/1-2026/vacuum-robot/calibration-2026-08.md) 📊 |
| CP463 | สัปดาห์ 8–14 | Intelligence Document | [llm-agent-tool-use](docs/task-templates/llm-agent-tool-use.md) | [overview](docs/competitions/CP463/1-2026/intelligence-document/overview.md) |

**การเพิ่มโจทย์ใหม่** = สร้างโฟลเดอร์ `<competition>/` ใต้เทอมนั้น แล้วเขียน

- `overview.md` — สรุปโจทย์, template ที่ใช้, การให้คะแนน + เกณฑ์ตัดสินเสมอ, baseline ladder, public/private split, แผนตามสัปดาห์, สิ่งที่ต้องเตรียม
- `environment-spec.md` — รายละเอียดระดับที่ implement ได้โดยไม่ต้องตัดสินใจเพิ่ม (เขียนเมื่อพร้อมลงมือสร้าง)

ถ้าใช้ template ที่มีอยู่แล้ว `overview.md` ระบุแค่ค่าที่ template ต้องการก็พอ
(prediction-based ดู [§13](docs/task-templates/prediction-based-supervised.md#13-ประเด็นที่ต้องตัดสินใจต่อ-competition) · RL ดู [§10](docs/task-templates/agent-vs-environment-rl.md#10-ประเด็นที่ต้องตัดสินใจต่อ-competition) · LLM agent ดู [§11](docs/task-templates/llm-agent-tool-use.md#11-ประเด็นที่ต้องตัดสินใจต่อ-competition))

---

## เริ่มพัฒนา

```bash
git config core.hooksPath tools/hooks   # ⚠️ ทำครั้งเดียวต่อ clone — ดูข้างล่าง
tools/hooks/pre-commit --selftest        # ยืนยันว่า hook ทำงาน

# แพลตฟอร์ม (core + runners)
uv venv --python 3.11
uv pip install -e ".[dev]" -e ./envs/cp463-vacuum
pytest core/tests runners/tests -q                    # 93 ข้อ
docker build -f runners/agent_env/images/Dockerfile.cpu -t arena/vacuum:cpu .
pytest runners/tests/test_docker_sandbox.py -q        # ต้องมี Docker

# ลองใช้จริง — API + worker ในกระบวนการเดียว (dev เท่านั้น)
python -m core.cli serve --port 8000
ARENA_URL=http://127.0.0.1:8000 ARENA_TOKEN=team-1 \
    python -m core.cli submit cp463-vacuum-1-2026 --dir path/to/agent
ARENA_URL=http://127.0.0.1:8000 ARENA_TOKEN=team-1 \
    python -m core.cli leaderboard cp463-vacuum-1-2026

# environment ของโจทย์ (นิสิตติดตั้งแค่ก้อนนี้)
cd envs/cp463-vacuum && uv venv --python 3.11 && uv pip install -e ".[dev]"
pytest -q                        # conformance test §14 — 31 ข้อ
python examples/calibrate.py     # การทดลอง §15
```

| ก้อน | สถานะ |
|---|---|
| [`envs/cp463-vacuum`](envs/cp463-vacuum/) | ✅ v1.0.0 · conformance test 31 ข้อผ่าน · calibrate แล้ว ([รายงาน](docs/competitions/CP463/1-2026/vacuum-robot/calibration-2026-08.md)) |
| [`runners/`](runners/) | ✅ โปรโตคอลแยก process · Docker sandbox · submission validation · worker daemon |
| [`core/`](core/) | 🟡 domain · คิว fair-share · leaderboard · REST API · CLI — **ยังเก็บข้อมูลในหน่วยความจำ** |
| `web/` | ⬜ ยังไม่เริ่ม |

**วงจรที่ปิดครบแล้ว** — อัพโหลด zip → ตรวจแบบ static → เข้าคิว → worker หยิบไปรันใน
Docker sandbox → คะแนนขึ้น leaderboard พร้อม replay ทุก episode
ซึ่งเป็นเกณฑ์ที่ [§14 M1](#14-ขอบเขต-mvp-และ-roadmap) ใช้วัดว่าพร้อมหรือยัง:
*"ทีมทดสอบส่ง agent แล้วเห็นคะแนนขึ้น leaderboard ได้ครบวงจร โดยไม่มีใครต้องเข้า SSH"*

สองอย่างที่พิสูจน์ด้วยเทสต์ ไม่ใช่แค่เขียนไว้ในเอกสาร

- **คะแนนใน sandbox ตรงกับที่นิสิตรันในเครื่องตัวเองถึง 1e-12** ทั้ง 4 baseline × 2 phase
  — เงื่อนไขที่ทำให้ `arena eval --local` มีความหมาย
- **trust boundary ปิดจริง** ([§10.4](#104-ขอบเขตความไว้วางใจ-trust-boundaries)) — เทสต์ให้ agent
  พยายามเอื้อมไปหา environment ผ่าน `gc` · ต่อเน็ต · เขียน rootfs **จริงๆ** แล้วยืนยันว่าทำไม่ได้
  ไม่ใช่อ่านธง `docker run` แล้วเชื่อว่ามันทำงาน

**ที่เหลือก่อนเปิดใช้จริง** — ผูก `core/` เข้ากับ Postgres · Google OAuth แทน team token ·
runner daemon ที่ต่อ WebSocket ออกมาหา cloud · หน้าเว็บ leaderboard + replay viewer ·
ตรึงคะแนน baseline บน public seeds

### pre-commit hook — กันค่า seed หลุด

repo นี้เป็น **public** แต่ค่า seed ของ public/private eval เป็นความลับและอยู่ที่
[`colosseum-hypogeum`](#105-โครงสร้าง-repository) กฎ "ห้ามเอาออกจาก repo นั้น" เป็นวินัยล้วนๆ
[`tools/hooks/pre-commit`](tools/hooks/pre-commit) ทำให้มันเป็นการป้องกันเชิงโครงสร้างแทน — ปฏิเสธ commit ที่มี

- ตัวเลขตรงกับค่า seed ลับตัวใดตัวหนึ่ง (รายงานเป็น `file:line` + ค่าที่ mask ไว้ ไม่พิมพ์ค่าเต็ม)
- ชื่อไฟล์ที่ไม่ควรอยู่ที่นี่ (`seeds*.yaml`, path ที่มี `hypogeum` หรือ `/diamond/`)

หา hypogeum จาก `../colosseum-hypogeum` หรือตัวแปร `ARENA_HYPOGEUM`

**ข้อจำกัดที่ต้องรู้** — hook คุ้มครองได้เท่าที่มันถูกเปิดใช้จริง

| | |
|---|---|
| เครื่องที่ไม่ได้ตั้ง `core.hooksPath` | ไม่มีการตรวจเลย |
| เครื่องที่ไม่ได้ clone hypogeum | ตรวจชื่อไฟล์ได้ แต่ตรวจค่า seed ไม่ได้ (เตือนแล้วปล่อยผ่าน) |
| ประวัติที่ commit ไปแล้ว | ไม่ตรวจย้อนหลัง — hook ดูเฉพาะสิ่งที่ staged |
| `git commit --no-verify` | ข้ามได้ตามปกติของ git |

**ถ้าโดนฟ้องเพราะเลขบังเอิญตรงกัน ให้เปลี่ยนเลขนั้น ไม่ใช่ทำ allowlist** —
การใส่เลขลงไฟล์ allowlist ใน repo สาธารณะเท่ากับประกาศว่าเลขนั้นเป็น seed ซึ่งคือสิ่งที่กำลังกันอยู่พอดี
(โอกาสชนกันเองราว 0.5% ต่อเลข 5 หลักหนึ่งตัว)

