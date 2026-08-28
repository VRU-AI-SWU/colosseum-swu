"""ตรวจ zip ของ submission — ส่วนที่ทุกชนิดโจทย์ใช้เหมือนกัน

⚠️ **ไฟล์นี้รันบน cloud API** จึง**ห้าม import หรือ exec โค้ดนิสิตเด็ดขาด**
มันใช้ `ast` อ่านโครงสร้างอย่างเดียว · ถ้าเครื่องที่รับไฟล์จากอินเทอร์เน็ตรันโค้ด
ที่อัพโหลดมาเพื่อ "ตรวจ" ก็เท่ากับเปิดให้รันอะไรก็ได้บนนั้น

สิ่งที่ต่างกันระหว่างโจทย์มีแค่สามอย่าง — ชื่อไฟล์ทางเข้า · ชื่อคลาส · เมธอดที่ต้องมี
ทั้งสามเป็นพารามิเตอร์ของ `inspect_archive` · ที่เหลือ (เพดานขนาด · zip bomb ·
path ที่ออกนอกโฟลเดอร์ · กติกาการหาไฟล์ทางเข้า) ต้องเหมือนกันทุกโจทย์ และ
**ต้องมีที่มาที่เดียว** เพราะกติกาเดียวกันที่เขียนสองที่จะเพี้ยนกันเสมอ

> **การตรวจ import เป็นแค่ตัวช่วยบอกนิสิต ไม่ใช่มาตรการความปลอดภัย**
> `__import__(base64.b64decode(...))` หลบได้ง่ายๆ ตัวที่บังคับจริงคือ sandbox
> ที่ไม่มีเน็ตและไม่มี package นอก whitelist ติดตั้งอยู่ตั้งแต่แรก
"""

from __future__ import annotations

import ast
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_FILE_BYTES = 100 * 1024 * 1024

#: ฟังก์ชันที่รับชื่อไฟล์เป็นอาร์กิวเมนต์แรก — ใช้ตรวจว่าไฟล์ที่โค้ดจะเปิดอยู่ใน zip จริง
_FILE_OPENERS = frozenset({"load", "open", "read_csv", "read_parquet", "read_pickle", "loadtxt"})


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


# ── หาไฟล์ทางเข้า ──────────────────────────────────────────────────


def resolve_entry(names: set[str], entry: str, report: ValidationReport) -> str | None:
    """หาไฟล์ทางเข้า **ด้วยกติกาเดียวกับที่ `ArtifactStore.extract` ใช้ตอนรัน**

    ต้องตรงกันเป๊ะ ถ้าที่นี่ยอมกว้างกว่า submission จะผ่านการตรวจ **กินโควตาไปหนึ่งครั้ง**
    แล้วค่อยไปตายตอนรัน ซึ่งคือการเสียโควตาให้ความผิดพลาดที่บอกได้ตั้งแต่ตอนรับไฟล์
    กติกาคือ *ราก zip* หรือ *ซ้อนหนึ่งชั้นและมีโฟลเดอร์เดียว*
    ([`core/store.py`](../../core/store.py) `extract`)

    เคสที่พบบ่อยคือรัน `arena submit` จากโฟลเดอร์แม่ที่มีโฟลเดอร์งานหลายอัน
    (`my-agent/` กับ `my-agent-v2/`) — เดิมผ่านการตรวจแล้วไปพังทีหลัง
    """
    if entry in names:
        return entry

    suffix = f"/{entry}"
    nested = sorted(
        {n.split("/", 1)[0] for n in names if n.count("/") == 1 and n.endswith(suffix)}
    )
    if len(nested) == 1:
        return f"{nested[0]}{suffix}"

    stem = entry.removesuffix(".py")
    if len(nested) > 1:
        report.add(
            f"ambiguous_{stem}_py",
            f"มี {entry} อยู่หลายโฟลเดอร์: {', '.join(nested)} — ไม่รู้ว่าจะรันอันไหน",
            "ส่งทีละโฟลเดอร์งาน — `cd <โฟลเดอร์งาน>` แล้ว `arena submit` "
            "หรือระบุ `--dir <โฟลเดอร์งาน>`",
        )
    else:
        report.add(
            f"missing_{stem}_py",
            f"ไม่พบ {entry} ใน zip",
            f"วาง {entry} ไว้ที่ระดับบนสุดของ zip (ซ้อนได้ไม่เกินหนึ่งชั้น)",
        )
    return None


# ── ตรวจทั้ง archive ───────────────────────────────────────────────


def inspect_archive(
    archive: str | Path,
    *,
    entry: str,
    check_source: Callable[[str, ValidationReport], None],
) -> ValidationReport:
    """ตรวจ zip โดยไม่แตกไฟล์ลงดิสก์และไม่รันอะไรเลย

    `check_source(source, report)` เป็นการตรวจเฉพาะของโจทย์ — ได้ซอร์สของไฟล์
    ทางเข้ามาแล้วเติม problem เข้า report เอง
    """
    report = ValidationReport()
    path = Path(archive)

    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        report.add(
            "archive_too_big",
            f"ไฟล์ zip ขนาด {path.stat().st_size / 1e6:.0f} MB "
            f"เกินเพดาน {MAX_ARCHIVE_BYTES / 1e6:.0f} MB",
            "ลบ checkpoint ที่ไม่ใช้ออก หรือส่งเฉพาะโมเดลตัวที่ใช้จริง",
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
                        f"{info.filename} ขนาด {info.file_size / 1e6:.0f} MB "
                        f"เกิน {MAX_FILE_BYTES / 1e6:.0f} MB",
                        "แยกไฟล์ให้เล็กลง หรือใช้โมเดลที่เล็กลง",
                    )
                name = Path(info.filename)
                if name.is_absolute() or ".." in name.parts:
                    report.add(
                        "unsafe_path",
                        f"path ในไฟล์ zip ออกนอกโฟลเดอร์: {info.filename}",
                        "zip ใหม่จากในโฟลเดอร์ submission ตรงๆ อย่าใช้ path แบบ absolute",
                    )

            names = {i.filename for i in infos}
            entry_name = resolve_entry(names, entry, report)
            if entry_name is None:
                return report

            source = zf.read(entry_name).decode("utf-8", errors="replace")
            py_sources = {
                n: zf.read(n).decode("utf-8", errors="replace")
                for n in names
                if n.endswith(".py")
            }
    except zipfile.BadZipFile:
        report.add("bad_zip", "อ่านไฟล์ zip ไม่ได้", "zip ใหม่อีกครั้ง (ไฟล์อาจเสียตอนอัพโหลด)")
        return report

    check_source(source, report)
    check_referenced_files(source, entry_name, names, report)
    for src in py_sources.values():
        report.imports |= top_level_imports(src)
    return report


def check_referenced_files(
    source: str, entry_name: str, names: set[str], report: ValidationReport
) -> None:
    """ไฟล์ที่โค้ดจะเปิดด้วยชื่อตรงๆ ต้องอยู่ใน zip ด้วย

    บทเรียนเดียวกับ `resolve_entry` — `joblib.load("pipeline.pkl")` ที่ไม่มี
    `pipeline.pkl` ใน zip จะผ่านการตรวจ กินโควตา แล้วไปตายตอนรัน ทั้งที่บอกได้
    ตั้งแต่ตอนรับไฟล์ · เป็นความผิดพลาดที่พบบ่อยที่สุดของโจทย์ที่ต้องส่งโมเดลมาด้วย

    **ดูเฉพาะชื่อไฟล์ที่เป็นข้อความตรงๆ** ถ้าเป็นตัวแปรหรือ path ที่ประกอบขึ้นมา
    ก็ปล่อยผ่าน — เดาผิดแล้วปฏิเสธ submission ที่ถูกต้อง แย่กว่าปล่อยให้ไปตายตอนรัน
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return  # syntax error ถูกรายงานแยกไปแล้ว

    folder = entry_name.rsplit("/", 1)[0] + "/" if "/" in entry_name else ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else None
        )
        if name not in _FILE_OPENERS:
            continue
        arg = node.args[0]
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
            continue
        wanted = arg.value
        if wanted.startswith("/") or ".." in wanted or "*" in wanted:
            continue
        if folder + wanted in names or wanted in names:
            continue
        report.add(
            "missing_data_file",
            f"โค้ดจะเปิดไฟล์ {wanted!r} แต่ไม่มีไฟล์นั้นใน zip",
            f"ใส่ {wanted} ลงในโฟลเดอร์เดียวกับโค้ดก่อน zip — "
            "ตอนรันจะไม่มีอะไรนอกจากไฟล์ที่ส่งมา",
        )


def check_import_whitelist(
    report: ValidationReport, whitelist: Iterable[str]
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


def check_class(
    source: str,
    report: ValidationReport,
    *,
    entry: str,
    class_name: str,
    methods: Iterable[str],
    signature_hint: str,
) -> None:
    """ไฟล์ทางเข้าต้องมีคลาสชื่อที่กำหนด พร้อมเมธอดครบ — อ่านด้วย `ast` ไม่ import"""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        report.add(
            "syntax_error",
            f"{entry} มี syntax error ที่บรรทัด {exc.lineno}: {exc.msg}",
            f'รัน `python -c "import {entry.removesuffix(".py")}"` ในเครื่องตัวเองก่อนส่ง',
        )
        return

    found = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None
    )
    if found is None:
        report.add(
            f"missing_{class_name.lower()}_class",
            f"{entry} ไม่มี `class {class_name}` ที่ระดับบนสุด",
            f"ตั้งชื่อคลาสว่า `{class_name}` เป๊ะๆ และอย่าซ่อนไว้ในฟังก์ชันหรือ if",
        )
        return

    have = {n.name for n in found.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = [m for m in methods if m not in have]
    if missing:
        report.add(
            "missing_methods",
            f"`class {class_name}` ขาดเมธอด {missing}",
            signature_hint,
        )


def top_level_imports(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()  # syntax error ถูกรายงานแยกไปแล้วสำหรับไฟล์ทางเข้า
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found
