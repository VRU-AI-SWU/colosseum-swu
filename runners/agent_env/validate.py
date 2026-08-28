"""ตรวจ submission — environment-spec §13

⚠️ **แยกสองชั้นตามที่มันรันอยู่คนละเครื่อง**

| ชั้น | รันที่ไหน | รันโค้ดนิสิตไหม | เมื่อไร |
|---|---|---|---|
| `static_checks()` | **cloud API** | ❌ **ห้ามเด็ดขาด** | ตอนอัพโหลด ต้องเสร็จใน < 5 วินาที |
| `smoke_test()` | **runner on-prem ใน sandbox** | ✅ | ก่อนเข้าคิวจริง |

การแยกนี้ไม่ใช่เรื่องประสิทธิภาพ — cloud API เป็นเครื่องที่รับไฟล์จากอินเทอร์เน็ต
ถ้ามันรันโค้ดที่อัพโหลดมาเพื่อ "ตรวจ" ก็เท่ากับเปิดให้รันโค้ดอะไรก็ได้บนนั้น
ชั้นแรกจึงใช้ `ast` อ่านโครงสร้างเท่านั้น ไม่มี `import` ไม่มี `exec`

> **การตรวจ import เป็นแค่ตัวช่วยบอกนิสิต ไม่ใช่มาตรการความปลอดภัย**
> `__import__(base64.b64decode(...))` หลบได้ง่ายๆ ตัวที่บังคับจริงคือ sandbox ที่ไม่มีเน็ต
> และไม่มี package นอก whitelist ติดตั้งอยู่ตั้งแต่แรก ([README §4.1](../../README.md#41-hosted-run-ค่าเริ่มต้น--ใช้กับโจทย์-rl-ของ-cp463))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from runners.sandbox import archive as archive_mod
from runners.sandbox.archive import (  # noqa: F401 — เดิมอยู่ไฟล์นี้ ผู้เรียกเดิมยัง import จากที่นี่ได้
    MAX_ARCHIVE_BYTES,
    MAX_FILE_BYTES,
    Problem,
    ValidationReport,
)

ENTRY = "agent.py"
AGENT_CLASS = "Agent"
REQUIRED_METHODS = ("__init__", "reset", "act")

# package ที่ผู้สอนประกาศตอนเปิดเทอม — ค่าเริ่มต้นตาม §13
DEFAULT_WHITELIST = frozenset({"numpy", "torch", "vacuum", "gymnasium"})

# ตัดให้สั้นเพื่อให้เร็ว — shape ของ observation ไม่ขึ้นกับ max_steps
# ใช้ **เลขคี่** โดยตั้งใจ: ถ้า agent มีตัวนับที่รั่วข้าม episode และพฤติกรรมขึ้นกับ
# parity ของตัวนับนั้น ความยาว episode ที่เป็นเลขคู่จะรักษา parity เดิมไว้พอดี
# ทำให้การรั่วมองไม่เห็น — เลขคี่พลิก parity ทุกครั้งจึงไวกว่า
SMOKE_MAX_STEPS = 61
SMOKE_SEEDS = (70001, 70002)


# ── ชั้นที่ 1: static (ปลอดภัยพอจะรันบน cloud API) ─────────────────


def check_import_whitelist(
    report: ValidationReport, whitelist: Iterable[str] = DEFAULT_WHITELIST
) -> ValidationReport:
    """whitelist ของโจทย์นี้ — ตัวตรวจจริงอยู่ที่ `runners/sandbox/archive.py`

    ค่าเริ่มต้นอยู่ที่นี่ไม่ใช่ที่โมดูลกลาง เพราะ `torch`/`gymnasium` เป็นเรื่องของ CP463
    ไม่ใช่ของทุกโจทย์
    """
    return archive_mod.check_import_whitelist(report, whitelist)


def _check_agent_class(source: str, report: ValidationReport) -> None:
    archive_mod.check_class(
        source, report,
        entry=ENTRY, class_name=AGENT_CLASS, methods=REQUIRED_METHODS,
        signature_hint=(
            "ต้องมี __init__(self, config) · reset(self, episode_info) · act(self, observation)"
        ),
    )


def inspect_archive(archive: str | Path) -> ValidationReport:
    """ตรวจ zip ของโจทย์ RL — กติกาที่ใช้ร่วมกันอยู่ที่ `runners/sandbox/archive.py`"""
    return archive_mod.inspect_archive(archive, entry=ENTRY, check_source=_check_agent_class)


# ── ชั้นที่ 2: dynamic (ต้องรันใน sandbox บน runner) ────────────────


@dataclass
class SmokeResult:
    ok: bool
    problems: list[Problem] = field(default_factory=list)
    detail: str = ""

    def __str__(self) -> str:
        return "ผ่าน" if self.ok else "\n".join(str(p) for p in self.problems)


def smoke_test(
    *,
    env_plugin: str,
    config_path: str | Path,
    submission_dir: str | Path,
    launcher: Any | None = None,
) -> SmokeResult:
    """สร้าง agent · รัน 2 episode สั้นๆ · ตรวจว่า `reset()` ล้าง state จริง

    การตรวจ reset เทียบ **การรันต่อกันในกระบวนการเดียว** กับ **การรันแต่ละ seed แยกกระบวนการ**
    ถ้าคะแนนต่างกัน แปลว่ามี state รั่วข้าม episode → คะแนนขึ้นกับลำดับที่ระบบสุ่มมาให้ ไม่ใช่ฝีมือ

    ⚠️ **การสลับลำดับอย่างเดียวไม่พอ** — ถ้าทุก episode ยาวเท่ากัน (ซึ่งเป็นกรณีปกติเมื่อ
    ยังไม่มีใครดูดครบ) ตัวนับที่รั่วจะมีค่าเดิมทุกครั้งที่เริ่ม episode ที่สอง การสลับลำดับ
    จึงมองไม่เห็นอะไรเลย · การเทียบกับการรันแยกกระบวนการจับได้ทุกกรณีเพราะกระบวนการใหม่
    ไม่มีประวัติอะไรติดมาเลย
    """
    from runners.agent_env.runner import run_submission

    common = dict(
        env_plugin=env_plugin,
        config_path=config_path,
        submission_dir=submission_dir,
        config_overrides={"episode.max_steps": SMOKE_MAX_STEPS},
        launcher=launcher,
        run_timeout_s=120,
    )
    forward = run_submission(seeds=list(SMOKE_SEEDS), **common)
    if not forward.ok:
        return SmokeResult(
            ok=False,
            problems=[
                Problem(
                    forward.status,
                    f"รัน smoke test ไม่ผ่าน ({forward.status})",
                    "ดู traceback ข้างล่างแล้วแก้ในเครื่องตัวเองก่อนส่งใหม่",
                )
            ],
            detail=f"{forward.detail or ''}\n{forward.log}".strip(),
        )

    problems = [
        Problem(
            e.status,
            f"seed {e.seed}: episode ล้มเหลว ({e.status})",
            "ดู traceback ข้างล่าง — ตอนรันจริง episode นี้จะได้ 0 คะแนน",
        )
        for e in forward.episodes
        if e.status != "ok"
    ]
    if problems:
        details = "\n\n".join(e.detail or "" for e in forward.episodes if e.status != "ok")
        return SmokeResult(ok=False, problems=problems, detail=details)

    sequential = {e.seed: e.breakdown.score for e in forward.episodes}

    for seed in SMOKE_SEEDS:
        alone = run_submission(seeds=[seed], **common)  # กระบวนการใหม่ ไม่มีประวัติติดมา
        if not alone.ok or alone.episodes[0].status != "ok":
            return SmokeResult(
                ok=False,
                problems=[
                    Problem(
                        alone.status,
                        f"seed {seed} รันเดี่ยวไม่ผ่านทั้งที่รันต่อกันผ่าน",
                        "แปลว่า agent พึ่งพา state จาก episode ก่อนหน้า",
                    )
                ],
                detail=(alone.detail or "") + "\n" + (alone.episodes[0].detail or ""),
            )
        solo = alone.episodes[0].breakdown.score
        if abs(sequential[seed] - solo) > 1e-9:
            return SmokeResult(
                ok=False,
                problems=[
                    Problem(
                        "state_leak",
                        f"seed {seed} ได้ {sequential[seed]:.6f} เมื่อรันต่อจาก episode อื่น "
                        f"แต่ได้ {solo:.6f} เมื่อรันเดี่ยว",
                        "`reset()` ต้องล้าง state ภายในให้หมดจริง — แผนที่ที่สะสมไว้ "
                        "ตัวนับ และ RNG ของ agent ต้องกลับไปเป็นค่าเริ่มต้นทุกครั้ง",
                    )
                ],
            )

    return SmokeResult(ok=True)
