"""🔒 เมล็ดของชุดที่ใช้ตัดสิน — อ่านจาก `ARENA_SECRETS` เท่านั้น

    $ARENA_SECRETS/cp462-1-2026/tabular/<slug>.yaml
      grading_seed: <ตัวเลข>

โครงตรงกับของ CP463 (`cp463-1-2026/vacuum/seeds.yaml`) — โฟลเดอร์ต่อวิชา-เทอม
แล้วโฟลเดอร์ต่อ environment

**ไฟล์นี้อยู่ในแพ็กเกจที่แจกนิสิต แต่ค่าที่มันอ่านไม่ได้อยู่ด้วย** — เหมือน
`runners/seeds.py` ของ CP463 · โค้ดเป็นสาธารณะ ของลับอยู่ใน repo ส่วนตัวที่
clone ไว้เฉพาะเครื่อง runner

⚠️ **ห้าม import ไฟล์นี้จากอะไรที่รันในกล่อง** ต่อให้มันอ่านไม่เจอก็ตาม —
การมีทางเรียกอยู่ในกล่องแปลว่ามีทางที่จะได้ค่ามาถ้าวันหนึ่ง mount ผิด
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

#: เมล็ดสำรองสำหรับ dev และเทสต์ — **ไม่ใช่ของจริง**
#:
#: มีไว้ให้พัฒนาและรันเทสต์ได้บนเครื่องที่ไม่มี `ARENA_SECRETS` (แบบเดียวกับชุด
#: conformance ของ CP463) · คะแนนที่วัดด้วยเมล็ดนี้ไม่ใช่คะแนนจริง และ worker
#: ของจริงจะปฏิเสธถ้าไม่ได้เปิด `allow_seed_fallback`
FALLBACK_SEED = 424242

#: ที่อยู่ของไฟล์ลับใต้ `ARENA_SECRETS`
SECRETS_SUBDIR = "cp462-1-2026/tabular"


class GradingSeedUnavailable(RuntimeError):
    """ไม่มีเมล็ดของชุดที่ใช้ตัดสิน — ให้คะแนนจริงไม่ได้"""


def secrets_dir() -> Path | None:
    raw = os.environ.get("ARENA_SECRETS", "").strip()
    return Path(raw) if raw else None


def load_grading_seed(slug: str, *, allow_fallback: bool = False) -> int:
    """เมล็ดของชุดที่ใช้ตัดสินของโจทย์นี้

    `allow_fallback=True` ยอมใช้เมล็ดสำรองพร้อมเตือนดังๆ — dev เท่านั้น
    """
    root = secrets_dir()
    path = (root / SECRETS_SUBDIR / f"{slug}.yaml") if root else None
    if path is not None and path.is_file():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        seed = data.get("grading_seed")
        if not isinstance(seed, int):
            raise GradingSeedUnavailable(f"{path}: ต้องมีฟิลด์ `grading_seed` ที่เป็นจำนวนเต็ม")
        return seed

    if allow_fallback:
        import warnings

        warnings.warn(
            f"⚠️ ใช้เมล็ดสำรองสำหรับชุดที่ใช้ตัดสินของ {slug} — คะแนนที่ได้ไม่ใช่คะแนนจริง",
            stacklevel=2,
        )
        return FALLBACK_SEED

    raise GradingSeedUnavailable(
        f"ไม่พบเมล็ดของชุดที่ใช้ตัดสินสำหรับ {slug!r}\n"
        f"  ตั้ง ARENA_SECRETS ให้ชี้ไปที่ clone ของ colosseum-hypogeum "
        f"ที่มีไฟล์ {SECRETS_SUBDIR}/{slug}.yaml\n"
        "  (ถ้าคุณเป็นนิสิต: ค่านี้ไม่ได้อยู่ในแพ็กเกจโดยตั้งใจ — ชุดที่ใช้ตัดสิน\n"
        "   ต้องเป็นข้อมูลที่คุณไม่เคยเห็น ไม่งั้นคะแนนไม่มีความหมาย)"
    )
