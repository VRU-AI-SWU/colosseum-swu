# cp462-tabular

ชุดข้อมูลและตัวให้คะแนนของ **CP462 · Introduction to Data Science** —
โจทย์ทำนายผลบนข้อมูลตาราง ตาม
[template prediction-based-supervised](../../docs/task-templates/prediction-based-supervised.md)

> **ชุดข้อมูลตอนนี้เป็นข้อมูลสังเคราะห์** — วิชายังไม่มีชุดจริง · สร้างขึ้นเพื่อให้
> runner, metric และ starter kit เขียนและทดสอบได้ครบก่อน แล้วค่อยสลับเป็นของจริง
> โดยแก้แค่ตัวโหลด ไม่ต้องแตะส่วนอื่น

## สองโจทย์

| slug | ชนิด | ทำนายอะไร | คะแนนหลัก |
|---|---|---|---|
| `churn` | classification | ลูกค้าจะเลิกใช้บริการไหม (ไม่สมดุล ~26%) | macro-F1 |
| `housing` | regression | มูลค่าต่อเดือน (เบ้ขวา) | R² |

**clustering ไม่รองรับ** — เหตุผลอยู่ใน
[template §6](../../docs/task-templates/prediction-based-supervised.md#6-metrics-และ-95-confidence-interval)

## ข้อมูลถูกออกแบบให้บังคับทักษะ ไม่ใช่สุ่มมั่ว

| สิ่งที่ใส่เข้าไป | บังคับให้ทำอะไร |
|---|---|
| ค่าว่างในคอลัมน์ตัวเลขและหมวดหมู่ | ต้อง impute — และต้องอยู่ใน `Pipeline` |
| หมวด `legacy` ที่พบ ~2% | ต้องจัดการหมวดที่ไม่เคยเห็น (`handle_unknown`) |
| ความสัมพันธ์แบบขั้นบันได + interaction | เส้นตรงบนฟีเจอร์ดิบจับไม่ได้ |
| `account_id` ที่ไม่เกี่ยวอะไรเลย | การโยนทุกคอลัมน์เข้าโมเดลไม่ใช่คำตอบ |

**calibrate แล้ว** — มีบันไดพอให้การเลือกโมเดลมีผลจริง

| | ทายเดาไม่ใช้ข้อมูล | เชิงเส้น | gradient boosting |
|---|---|---|---|
| churn (macro-F1) | 0.43 | 0.51 | **0.64** |
| housing (R²) | 0.00 | 0.71 | **0.79** |

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `generator.py` | สร้างข้อมูล — ทำซ้ำได้ทุกบิต ไม่พึ่งการสุ่มของ pandas/sklearn |
| `splits.py` | แบ่ง train / val / test_public / test_private ด้วยเมล็ดที่ตรึงไว้ |

## ทดสอบ

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
pytest -q
```

## สถานะ

| | |
|---|---|
| ตัวสร้างข้อมูล + การแบ่งชุด | ✅ พร้อม (29 เทสต์) |
| metric + bootstrap CI | ⬜ ยังไม่ได้เขียน |
| TaskSpec / config | ⬜ |
| `selfcheck` | ⬜ |
| starter kit | ⬜ |
| runner ฝั่งแพลตฟอร์ม (`task_type="prediction"`) | ⬜ |
