"""ชื่อข้อความเฉพาะของโจทย์ RL — ส่วนที่เหลืออยู่ใน `runners.sandbox.protocol`

ค่าตรงนี้เป็น **ส่วนหนึ่งของ wire format** — เปลี่ยนเมื่อไรคือ image เก่ากับ
runner ใหม่คุยกันไม่รู้เรื่องทันที และอาการที่เห็นคือ agent ทุกตัวล้มพร้อมกัน
โดยไม่มีใครแก้โค้ดของตัวเอง
"""

from __future__ import annotations

# runner → agent
RESET = "reset"
ACT = "act"
# agent → runner
ACTION = "action"
