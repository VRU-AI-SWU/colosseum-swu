"""สำรองข้อมูลไปดิสก์อีกลูก — README §11

ตอนนี้ระบบไม่มีสำเนาอะไรเลย ถ้าดิสก์ที่รันอยู่พังกลางเทอม คะแนนทั้งเทอม submission
ทุกชิ้น และ audit trail หายหมด กู้ไม่ได้ — ความเสี่ยงนี้ใหญ่กว่า "พื้นที่จะเต็ม"
หลายเท่า และเป็นเหตุผลที่ดิสก์ลูกที่สองควรเป็นที่สำรอง ไม่ใช่ที่เก็บของร้อน

**ห้ามสำรอง `arena.db` ด้วย `cp`** — ไฟล์นั้นถูกเปิดค้างไว้โดยบริการที่กำลังรัน และ
transaction ล่าสุดอยู่ใน `-wal` การคัดลอกไฟล์เดียวตอนที่มีคนเขียนอยู่ได้สำเนาที่
ข้อมูลไม่ครบหรือใช้ไม่ได้เลย · สคริปต์นี้ใช้ SQLite backup API ซึ่งออกแบบมาให้
ทำงานได้ขณะมีคนเขียนอยู่พอดี

    python tools/backup.py --db /path/data/arena.db \\
                           --artifacts /path/artifacts \\
                           --dest /media/user/hdd/colosseum/backup

ทุกครั้งที่รันจะ **ตรวจสำเนาที่เพิ่งทำ** ด้วยการเปิดมันขึ้นมานับแถว — สำเนาที่ไม่เคย
ถูกเปิดอ่านคือสำเนาที่ยังไม่รู้ว่าใช้ได้จริงหรือเปล่า
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TABLES = ("teams", "competitions", "submissions", "runs", "audit")


def snapshot_db(db: Path, dest: Path, stamp: str) -> Path:
    """คัดลอกฐานข้อมูลแบบที่ปลอดภัยขณะบริการยังเขียนอยู่"""
    out = dest / "db" / f"arena-{stamp}.db"
    out.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(out))
        try:
            src.backup(dst)  # อ่านแบบ page-by-page พร้อมกันกับที่มีคนเขียนได้
        finally:
            dst.close()
    finally:
        src.close()
    return out


def verify(path: Path) -> dict[str, int]:
    """เปิดสำเนาขึ้นมานับแถว — ถ้าเปิดไม่ได้หรือ integrity ไม่ผ่าน ให้ล้มดังๆ"""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        status = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if status != "ok":
            raise RuntimeError(f"สำเนาเสียหาย: {status}")
        return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}
    finally:
        conn.close()


def mirror_artifacts(src: Path, dest: Path) -> str:
    """artifacts เขียนครั้งเดียวแล้วไม่แก้อีก (zip ตั้งชื่อด้วย sha256 · replay เขียนจบแล้วนิ่ง)
    จึง mirror ตรงๆ ได้ ไม่ต้องกลัวคัดลอกไฟล์ที่กำลังถูกเขียน
    """
    out = dest / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        run = subprocess.run(
            ["rsync", "-a", "--delete", f"{src}/", f"{out}/"], capture_output=True, text=True
        )
        if run.returncode != 0:
            raise RuntimeError(f"rsync ล้มเหลว: {run.stderr.strip()}")
        return "rsync"
    shutil.copytree(src, out, dirs_exist_ok=True)  # ทางถอย — ช้ากว่าและไม่ลบของที่หายไป
    return "copytree (ไม่มี rsync — ของที่ถูกลบต้นทางจะค้างอยู่)"


def prune(dest: Path, keep: int) -> list[Path]:
    snaps = sorted((dest / "db").glob("arena-*.db"))
    dead = snaps[:-keep] if keep > 0 and len(snaps) > keep else []
    for p in dead:
        p.unlink()
    return dead


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def tree_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--artifacts", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path, help="โฟลเดอร์บนดิสก์สำรอง")
    ap.add_argument("--keep", type=int, default=14, help="เก็บสำเนาฐานข้อมูลกี่ชุด (0 = ไม่ลบ)")
    ap.add_argument(
        "--allow-same-device",
        action="store_true",
        help="ยอมให้สำรองลงดิสก์ลูกเดียวกัน — สำหรับทดสอบเท่านั้น ไม่ใช่ backup จริง",
    )
    args = ap.parse_args()

    if not args.db.exists():
        print(f"✗ ไม่พบ {args.db}", file=sys.stderr)
        return 1
    if not args.artifacts.is_dir():
        print(f"✗ ไม่พบ {args.artifacts}", file=sys.stderr)
        return 1
    # ✋ ต้องอยู่คนละดิสก์จริงๆ — ไม่ใช่แค่ "โฟลเดอร์ปลายทางมีอยู่"
    #
    # ความล้มเหลวที่เจ็บที่สุดของ backup คือ HDD ยังไม่ mount แล้วสำเนาไปกองอยู่บน
    # mountpoint ว่างซึ่งอยู่บนดิสก์ตัวเดิม — ทุกอย่างดูสำเร็จ ไฟล์มีจริง จนวันที่
    # ดิสก์นั้นพังแล้วถึงรู้ว่าสำเนาก็อยู่บนดิสก์เดียวกันมาตลอด
    # การเทียบ st_dev ตอบคำถามที่เราสนใจจริงๆ ตรงๆ: มันคนละอุปกรณ์ไหม
    probe = args.dest.parent if args.dest.parent.exists() else args.dest.parent.parent
    if not probe.exists():
        print(f"✗ ไม่พบ {probe} — ดิสก์สำรอง mount อยู่หรือเปล่า", file=sys.stderr)
        return 1
    if probe.stat().st_dev == args.db.stat().st_dev and not args.allow_same_device:
        print(
            f"✗ {args.dest} อยู่บนดิสก์ลูกเดียวกับ {args.db}\n"
            "  ถ้าดิสก์นั้นพัง ทั้งของจริงและสำเนาหายพร้อมกัน — เท่ากับไม่มี backup\n"
            "  สาเหตุที่พบบ่อยที่สุดคือดิสก์สำรองยังไม่ได้ mount\n"
            "  (ถ้าตั้งใจจริง เช่นกำลังทดสอบ ใส่ --allow-same-device)",
            file=sys.stderr,
        )
        return 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    args.dest.mkdir(parents=True, exist_ok=True)

    snap = snapshot_db(args.db, args.dest, stamp)
    counts = verify(snap)
    how = mirror_artifacts(args.artifacts, args.dest)
    removed = prune(args.dest, args.keep)

    print(f"✓ สำรองเสร็จ {stamp}")
    print(f"  ฐานข้อมูล  {snap}  ({human(snap.stat().st_size)})")
    print("             " + " · ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"  artifacts  {human(tree_size(args.dest / 'artifacts'))}  [{how}]")
    if removed:
        print(f"  ลบสำเนาเก่า {len(removed)} ชุด (เก็บไว้ {args.keep})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
