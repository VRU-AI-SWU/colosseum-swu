"""โหลดค่า seed จากที่เก็บของลับ — **อยู่ฝั่ง runner เท่านั้น**

⚠️ **API process ต้องไม่เคยเห็นไฟล์นี้** ([README §10.4](../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries))
API รู้แค่ว่า competition ชื่ออะไร ส่วนค่า seed ถูกอ่านที่ runner ในมหาวิทยาลัย
ซึ่ง mount `colosseum-hypogeum` แบบ read-only ไว้

    /srv/arena/secrets/cp463-1-2026/vacuum/seeds.yaml     ← ตัวจริง
    ARENA_SECRETS=/srv/arena/secrets                       ← บอกตำแหน่ง

การแยกนี้ทำให้ **ต่อให้ cloud API ถูกเจาะ ค่า seed ก็ไม่ได้อยู่ที่นั่นตั้งแต่แรก**
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import yaml

#: ใช้เมื่อไม่มีที่เก็บของลับ — ชุด conformance ที่เปิดเผยได้อยู่แล้ว
FALLBACK_SEEDS = list(range(70001, 70011))


class SecretsUnavailable(RuntimeError):
    pass


def secrets_root() -> Path | None:
    root = os.environ.get("ARENA_SECRETS")
    return Path(root) if root and Path(root).is_dir() else None


def load_seeds(
    *, competition_slug: str, phase: str, kind: str, allow_fallback: bool = False
) -> list[int]:
    """คืนค่า seed ของ phase หนึ่งสำหรับ `kind` ∈ {public, private}

    `allow_fallback=True` ใช้ได้เฉพาะตอน dev/ทดสอบ — บนเครื่องที่ตัดสินคะแนนจริง
    ต้องปล่อยให้ล้มดังๆ ดีกว่ารันด้วย seed ที่ไม่ใช่ของจริงแล้วประกาศคะแนนออกไป
    """
    root = secrets_root()
    if root is None:
        if not allow_fallback:
            raise SecretsUnavailable(
                "ไม่พบที่เก็บของลับ — ตั้ง ARENA_SECRETS ให้ชี้ไปที่ clone ของ colosseum-hypogeum\n"
                "บนเครื่องที่ตัดสินคะแนนจริงห้ามใช้ seed สำรองเด็ดขาด"
            )
        warnings.warn(
            f"⚠️ ใช้ seed สำรอง (ชุด conformance) สำหรับ {competition_slug}/{phase}/{kind} "
            f"— คะแนนที่ได้ไม่ใช่คะแนนจริง",
            stacklevel=2,
        )
        return list(FALLBACK_SEEDS)

    candidates = list(root.rglob("seeds*.yaml"))
    for path in sorted(candidates):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("competition") != competition_slug:
            continue
        phases = data.get("phases") or {}
        if phase not in phases:
            raise SecretsUnavailable(f"{path.name} ไม่มี phase {phase!r}")
        seeds = phases[phase].get(kind)
        if not seeds:
            raise SecretsUnavailable(f"{path.name} ไม่มี seed ชนิด {kind!r} ของ phase {phase!r}")
        return [int(s) for s in seeds]

    raise SecretsUnavailable(
        f"ไม่พบไฟล์ seed ของ competition {competition_slug!r} ใน {root} "
        f"(ค้นจาก {len(candidates)} ไฟล์)"
    )


def expected_config_hash(*, competition_slug: str, phase: str) -> str | None:
    """`config_hash` ที่ผูกไว้ตอน generate seed

    runner ต้องเทียบกับ hash ของ config ที่กำลังจะรัน — ถ้าไม่ตรงแปลว่า config
    เปลี่ยนไปหลังจากที่ seed ถูกสร้าง และคะแนนข้าม hash ห้ามเอามาเทียบกัน
    """
    root = secrets_root()
    if root is None:
        return None
    for path in sorted(root.rglob("seeds*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("competition") == competition_slug:
            return ((data.get("phases") or {}).get(phase) or {}).get("config_hash")
    return None
