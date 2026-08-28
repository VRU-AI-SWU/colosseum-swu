"""ชื่อข้อความเฉพาะของโจทย์ทำนาย — ส่วนที่เหลืออยู่ใน `runners.sandbox.protocol`

ค่าตรงนี้เป็น **ส่วนหนึ่งของ wire format** — เปลี่ยนเมื่อไรคือ image เก่ากับ
runner ใหม่คุยกันไม่รู้เรื่องทันที
"""

from __future__ import annotations

# runner → predictor
PREDICT = "predict"
# predictor → runner
PREDICTION = "prediction"
