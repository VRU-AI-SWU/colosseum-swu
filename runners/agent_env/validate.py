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

import ast
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_FILE_BYTES = 100 * 1024 * 1024
REQUIRED_METHODS = ("__init__", "reset", "act")

# package ที่ผู้สอนประกาศตอนเปิดเทอม — ค่าเริ่มต้นตาม §13
DEFAULT_WHITELIST = frozenset({"numpy", "torch", "vacuum", "gymnasium"})

# ตัดให้สั้นเพื่อให้เร็ว — shape ของ observation ไม่ขึ้นกับ max_steps
# ใช้ **เลขคี่** โดยตั้งใจ: ถ้า agent มีตัวนับที่รั่วข้าม episode และพฤติกรรมขึ้นกับ
# parity ของตัวนับนั้น ความยาว episode ที่เป็นเลขคู่จะรักษา parity เดิมไว้พอดี
# ทำให้การรั่วมองไม่เห็น — เลขคี่พลิก parity ทุกครั้งจึงไวกว่า
SMOKE_MAX_STEPS = 61
SMOKE_SEEDS = (70001, 70002)


@dataclass
class Problem:
    code: str
    message: str
    fix: str

    def __str__(self) -> str:
        return f"❌ {self.message}\n   วิธีแก้: {self.fix}"


@dataclass
class ValidationReport:
    problems: list[Problem] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not self.problems

    def add(self, code: str, message: str, fix: str) -> None:
        self.problems.append(Problem(code, message, fix))

    def __str__(self) -> str:
        return "ผ่านทุกข้อ" if self.ok else "\n".join(str(p) for p in self.problems)


# ── ชั้นที่ 1: static (ปลอดภัยพอจะรันบน cloud API) ─────────────────


def _resolve_agent_py(names: set[str], report: ValidationReport) -> str | None:
    """หา `agent.py` **ด้วยกติกาเดียวกับที่ `ArtifactStore.extract` ใช้ตอนรัน**

    ต้องตรงกันเป๊ะ ถ้าที่นี่ยอมกว้างกว่า submission จะผ่านการตรวจ **กินโควตาไปหนึ่งครั้ง**
    แล้วค่อยไปตายตอนรันด้วย `agent_init_failed` ซึ่งคือการเสียโควตาให้ความผิดพลาด
    ที่บอกได้ตั้งแต่ตอนรับไฟล์ · กติกาคือ *ราก zip* หรือ *ซ้อนหนึ่งชั้นและมีโฟลเดอร์เดียว*
    ([`core/store.py`](../../core/store.py) `extract`)

    เคสที่พบบ่อยคือรัน `arena submit` จากโฟลเดอร์แม่ที่มีโฟลเดอร์งานหลายอัน
    (`my-agent/` กับ `my-agent-v2/`) — เดิมผ่านการตรวจแล้วไปพังทีหลัง
    """
    if "agent.py" in names:
        return "agent.py"

    nested = sorted({n.split("/", 1)[0] for n in names if n.count("/") == 1 and n.endswith("/agent.py")})
    if len(nested) == 1:
        return f"{nested[0]}/agent.py"

    if len(nested) > 1:
        report.add(
            "ambiguous_agent_py",
            f"มี agent.py อยู่หลายโฟลเดอร์: {', '.join(nested)} — ไม่รู้ว่าจะรันอันไหน",
            "ส่งทีละโฟลเดอร์งาน — `cd <โฟลเดอร์งาน>` แล้ว `arena submit` หรือระบุ `--dir <โฟลเดอร์งาน>`",
        )
    else:
        report.add(
            "missing_agent_py",
            "ไม่พบ agent.py ใน zip",
            "วาง agent.py ไว้ที่ระดับบนสุดของ zip (ซ้อนได้ไม่เกินหนึ่งชั้น)",
        )
    return None


def inspect_archive(archive: str | Path) -> ValidationReport:
    """ตรวจ zip โดยไม่แตกไฟล์ลงดิสก์และไม่รันอะไรเลย"""
    report = ValidationReport()
    path = Path(archive)

    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        report.add(
            "archive_too_big",
            f"ไฟล์ zip ขนาด {path.stat().st_size / 1e6:.0f} MB เกินเพดาน {MAX_ARCHIVE_BYTES / 1e6:.0f} MB",
            "ลบ checkpoint ที่ไม่ใช้ออก หรือส่งเฉพาะ weights ตัวที่ใช้จริง",
        )

    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            total = sum(i.file_size for i in infos)
            if total > MAX_ARCHIVE_BYTES:
                report.add(
                    "expands_too_big",
                    f"ไฟล์ในนี้รวมกันแตกออกได้ {total / 1e6:.0f} MB",
                    "ตรวจว่าไม่ได้ใส่ไฟล์ซ้ำหรือไฟล์ที่บีบอัดได้มากผิดปกติ (zip bomb)",
                )
            for info in infos:
                if info.file_size > MAX_FILE_BYTES:
                    report.add(
                        "file_too_big",
                        f"{info.filename} ขนาด {info.file_size / 1e6:.0f} MB เกิน {MAX_FILE_BYTES / 1e6:.0f} MB",
                        "แยกไฟล์ให้เล็กลง หรือใช้ weights ที่ quantize แล้ว",
                    )
                name = Path(info.filename)
                if name.is_absolute() or ".." in name.parts:
                    report.add(
                        "unsafe_path",
                        f"path ในไฟล์ zip ออกนอกโฟลเดอร์: {info.filename}",
                        "zip ใหม่จากในโฟลเดอร์ submission ตรงๆ อย่าใช้ path แบบ absolute",
                    )

            names = {i.filename for i in infos}
            agent_name = _resolve_agent_py(names, report)
            if agent_name is None:
                return report

            source = zf.read(agent_name).decode("utf-8", errors="replace")
            py_sources = {
                n: zf.read(n).decode("utf-8", errors="replace")
                for n in names
                if n.endswith(".py")
            }
    except zipfile.BadZipFile:
        report.add("bad_zip", "อ่านไฟล์ zip ไม่ได้", "zip ใหม่อีกครั้ง (ไฟล์อาจเสียตอนอัพโหลด)")
        return report

    _check_agent_class(source, report)
    for name, src in py_sources.items():
        report.imports |= _top_level_imports(src, name, report)
    return report


def check_import_whitelist(
    report: ValidationReport, whitelist: Iterable[str] = DEFAULT_WHITELIST
) -> ValidationReport:
    """เตือนเรื่อง import ที่ไม่อยู่ใน whitelist — แยกจาก `inspect_archive` เพราะ
    whitelist เป็นของแต่ละ competition ไม่ใช่ของ runner"""
    import sys

    allowed = set(whitelist) | set(sys.stdlib_module_names)
    for module in sorted(report.imports - allowed):
        report.add(
            "import_not_allowed",
            f"import `{module}` ซึ่งไม่อยู่ใน whitelist ของ competition นี้",
            f"ใช้เฉพาะ stdlib กับ {sorted(whitelist)} — package อื่นไม่ได้ติดตั้งใน sandbox "
            f"และไม่มีเน็ตให้ติดตั้งตอนรัน ถ้าจำเป็นจริงต้องขออนุมัติล่วงหน้า",
        )
    return report


def _check_agent_class(source: str, report: ValidationReport) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        report.add(
            "syntax_error",
            f"agent.py มี syntax error ที่บรรทัด {exc.lineno}: {exc.msg}",
            "รัน `python -c \"import agent\"` ในเครื่องตัวเองก่อนส่ง",
        )
        return

    agent = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Agent"), None
    )
    if agent is None:
        report.add(
            "missing_agent_class",
            "agent.py ไม่มี `class Agent` ที่ระดับบนสุด",
            "ตั้งชื่อคลาสว่า `Agent` เป๊ะๆ (ตัวพิมพ์ใหญ่ A) และอย่าซ่อนไว้ในฟังก์ชันหรือ if",
        )
        return

    methods = {n.name for n in agent.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = [m for m in REQUIRED_METHODS if m not in methods]
    if missing:
        report.add(
            "missing_methods",
            f"`class Agent` ขาดเมธอด {missing}",
            "ต้องมี __init__(self, config) · reset(self, episode_info) · act(self, observation)",
        )


def _top_level_imports(source: str, filename: str, report: ValidationReport) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()  # syntax error ถูกรายงานแยกไปแล้วสำหรับ agent.py
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


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
