# CP463 · Term Project 1/2026

Course project ของวิชา **CP463 (Artificial Intelligence)** ภาคเรียนที่ **1/2026** รันบน [Arena platform](../../../../README.md)

โครงสร้างรายวิชาแบ่งเป็นสองครึ่ง ([รายละเอียดหัวข้อ](list-of-topics.html)) — course project จึงแบ่งเป็น **2 competition
ที่เดินขนานไปกับตารางสอน** แทนที่จะเป็นโจทย์ก้อนเดียวตลอดเทอม

| | Competition 1 | Competition 2 |
|---|---|---|
| โจทย์ | [Vacuum Robot](vacuum-robot/overview.md) | [Intelligence Document](intelligence-document/overview.md) |
| ครึ่งวิชา | Part 1 — Reinforcement Learning | Part 2 — Agentic AI |
| คาบ | 1–5 | 8–14 |
| สัปดาห์ | 1–7 | 8–14 |
| นิสิตส่งอะไร | policy ที่เทรนเอง | agent harness (โปรแกรม ไม่ต้องเทรน) |
| Template | [agent-vs-environment-rl](../../../task-templates/agent-vs-environment-rl.md) | [llm-agent-tool-use](../../../task-templates/llm-agent-tool-use.md) |
| ทรัพยากรที่นับ | decision timestep | token + tool call |
| Compute | CPU lane | GPU lane (LLM gateway) |

---

## 1. การครอบคลุมเนื้อหา

| คาบ | หัวข้อ | ครอบคลุมโดย |
|---|---|---|
| 1 | MDP, value function, Bellman | Comp 1 · Warm-up |
| 2 | TD learning, Q-learning, ε-greedy | Comp 1 · Warm-up (tabular Q-learning ทำได้จริงใน phase นี้) |
| 3 | REINFORCE, actor-critic, GAE | Comp 1 · Main |
| 4 | on/off-policy, model-based/free, **reward shaping** | Comp 1 · Main (นิสิตออกแบบ reward เอง) |
| 5 | PPO | Comp 1 · Main |
| 6 | DPO, preference optimization | ⚠️ ไม่มี project รองรับ — ดู §3 |
| 7 | GRPO, reward model, reasoning models | ⚠️ ไม่มี project รองรับ — ดู §3 |
| 8 | Anatomy of an agent | Comp 2 |
| 9 | RAG | Comp 2 — **แกนหลัก** (คลังระเบียบ 40k token ยัดเข้า context ไม่ได้ ต้องค้นจริง) |
| 10 | Agentic memory | Comp 2 (context management ภายใน task + `knowledge/`) |
| 11 | Agent harness & design patterns | Comp 2 — **แกนหลัก** |
| 12 | Tool use, MCP, security model | Comp 2 — **แกนหลัก** — tool layer เสิร์ฟผ่าน MCP จริง + หมวด prompt injection |
| 13 | Multi-agent, A2A | Comp 2 — extractor ต่อเอกสาร + reconciler เป็นรูปแบบที่เกิดตามธรรมชาติ (ไม่บังคับ) |
| 14 | Evaluation, environments, benchmarks | Comp 2 + **ตัว arena เองเป็นกรณีศึกษา** |

รวมแล้วครอบคลุม **12 จาก 14 คาบแบบเต็ม** เหลือเฉพาะคาบ 6–7 (DPO/GRPO) ที่ไม่มี project รองรับ

---

## 2. ปฏิทินรวม

| สัปดาห์ | Competition 1 | Competition 2 |
|---|---|---|
| 1–3 | **Warm-up** — 10×10, full obs, deterministic | — |
| 4–6 | **Main** — 20×20, local obs, มี noise | (ผู้สอน/TA เตรียม environment) |
| 7 | **Final** + ปิดรับ + เปิด private leaderboard | — |
| 8–10 | — | เปิดโจทย์ · **Warm-up** เคสหมวด straightforward + derived |
| 11–13 | — | **Main** — ครบทุกหมวด รวม conflicting / missing / injection |
| 14 | — | ปิดรับ + เปิด private leaderboard |
| 15 | ส่งรายงาน + contribution statement | |

การปิด Competition 1 ที่สัปดาห์ 7 ทำให้นิสิตไม่ต้องแบกสองโจทย์พร้อมกัน และได้เห็นผล private ของครึ่งแรก
ก่อนเริ่มครึ่งหลัง — ผลการพลิกอันดับ (shake-up) ของ Comp 1 กลายเป็นบทเรียนที่ใช้สอนต่อได้ทันที

---

## 3. ช่องว่างที่ยอมรับ: คาบ 6–7 (DPO / GRPO)

เนื้อหา preference optimization และ reward modeling **ไม่มี competition รองรับ** และนี่เป็นการตัดสินใจ ไม่ใช่การมองข้าม

เหตุผล: การเทรน LLM ด้วย DPO/GRPO ต้องใช้ GPU ที่นิสิตส่วนใหญ่ไม่มี และ RTX 3090 ใบเดียวรองรับ 10 ทีมไม่ไหว
ถ้าฝืนใส่เข้า arena จะกลายเป็นการแข่งกันว่าใครมีเครื่องแรงกว่า ซึ่งขัดกับหลักการ hardware-independent scoring ของแพลตฟอร์ม

**ทางเลือกในการรองรับเนื้อหาส่วนนี้** (นอก arena)

1. **การบ้านเดี่ยวแบบไม่แข่งขัน** — LoRA + DPO บนโมเดล 0.5B กับ preference dataset เล็กๆ บน Colab ฟรี ส่ง notebook + กราฟ
2. **งานวิเคราะห์** — ให้ preference data + โมเดลสองตัวที่ align ต่างวิธี แล้ววิเคราะห์ว่าต่างกันอย่างไร
3. ปล่อยไว้ที่ quiz/สอบตามที่หลักสูตรออกแบบไว้อยู่แล้ว

> ถ้าอนาคตมี GPU เพิ่ม การเพิ่ม competition ที่สามด้วย template ใหม่ (preference optimization) เป็นทางที่เปิดไว้

---

## 4. สัดส่วนคะแนน (ร่าง — ผู้สอนกำหนด)

| ส่วน | สัดส่วน | วิธีให้คะแนน |
|---|---|---|
| Competition 1 | 30% | threshold ตาม baseline ladder ([เหตุผลที่ไม่ให้ตามอันดับ](../../../../README.md#53-การแปลงคะแนนเป็นเกรด)) |
| Competition 2 | 35% | threshold ตาม baseline ladder |
| รายงาน + การอธิบายวิธีการ | 25% | ประเมินการเลือกอัลกอริทึม, reward design, ablation, การวิเคราะห์ความล้มเหลว |
| การมีส่วนร่วม (ส่งสม่ำเสมอ, contribution statement) | 10% | จากข้อมูลบน arena โดยตรง |

**เหตุผลที่แยกคะแนนรายงานออกมา 25%** — leaderboard วัดผลลัพธ์ แต่วัดไม่ได้ว่าทีมเข้าใจว่าทำไมวิธีของตัวเองถึงได้ผล
ทีมที่ลอกโค้ดมาแล้วได้คะแนนดีจะตอบส่วนนี้ไม่ได้ และทีมที่ทดลองอย่างเป็นระบบแต่ผลไม่ดีที่สุดก็ยังได้คะแนนตามสมควร

---

## 5. สิ่งที่ต้องเสร็จก่อน

| ก่อนเปิดเทอม | ก่อนสัปดาห์ที่ 8 |
|---|---|
| Arena M1 (แข่งได้ครบวงจร) | LLM gateway + token metering |
| Vacuum environment + config 3 phase | Document generator + policy corpus + normalizer |
| Baseline ทั้ง 4 ระดับของ Comp 1 | Baseline ทั้ง 4 ระดับของ Comp 2 |
| **ยืนยันว่า `action_noise` กด planner ลงได้จริง** ([§11 ข้อ 0](vacuum-robot/overview.md#11-สิ่งที่ต้องตัดสินใจทดสอบก่อนเปิดเทอม)) | ตรวจ throughput ของ GPU |
