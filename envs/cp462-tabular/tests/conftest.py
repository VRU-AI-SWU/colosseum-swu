"""ตั้งค่าที่เทสต์ทั้งโฟลเดอร์นี้ใช้ร่วมกัน"""

from __future__ import annotations

import os


# ── เมล็ดของชุดที่ใช้ตัดสินของ CP462 ────────────────────────────────
# เครื่องที่ไม่มี `ARENA_SECRETS` ต้องรันเทสต์ได้ — เปิดเมล็ดสำรองให้เฉพาะตอนนั้น
# ถ้ามีของจริงอยู่ก็ใช้ของจริง ซึ่งเป็นการตรวจที่แข็งแรงกว่า
if not os.environ.get("ARENA_SECRETS"):
    os.environ.setdefault("ARENA_CP462_ALLOW_SEED_FALLBACK", "1")
