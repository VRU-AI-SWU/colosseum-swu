"""สร้างวิชาใหม่ — วิชาแรกของ deployment มาจาก `demo_arena` ซึ่งเป็นของ dev

    python tools/setup_course.py --db /path/arena.db \
        --id cp462-1-2026 --name "CP462 · Introduction to Data Science 1/2026"

**ทำไมไม่ทำเป็นหน้าเว็บ** — การสร้างวิชาเกิดปีละไม่กี่ครั้ง และเป็นจังหวะที่ต้อง
ตัดสินใจเรื่องที่ผิดแล้วแก้ยาก (`id` ของวิชาไปโผล่ใน `Team.course_id` ของทุกทีม
และเปลี่ยนทีหลังแปลว่าต้องไล่แก้ทุกแถว) · ฟอร์มบนหน้าเว็บชวนให้กดเร็วโดยไม่คิด
ส่วนเครื่องมือที่ต้อง `ssh` เข้าไปรันบังคับให้ตั้งใจ

`join_code` สุ่มให้อัตโนมัติ — เป็นรหัสที่ผู้สอนแจกในคาบ ดูซ้ำได้จากแผงผู้สอน
บนหน้าเว็บ หรือรันเครื่องมือนี้ด้วย `--show` เฉยๆ

เปิดฐานข้อมูลผ่าน `core.db.Database` เหมือนเครื่องมือตัวอื่น เพื่อให้ migration
ทำงานและเขียนผ่านเส้นทางเดียวกับที่บริการใช้
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.db import Database  # noqa: E402
from core.domain import (  # noqa: E402
    DEFAULT_MAX_TEAM_SIZE,
    MAX_COURSE_NAME_LENGTH,
    MAX_TEAM_SIZE_CEILING,
    Course,
    new_invite_code,
)

#: `id` ของวิชาไปโผล่ใน `Team.course_id` ของทุกทีมและใน audit trail
#: เปลี่ยนทีหลังแปลว่าต้องไล่แก้ทุกแถว จึงบังคับรูปแบบที่อ่านออกและพิมพ์ซ้ำได้
ID_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789-")


def valid_id(text: str) -> str:
    text = text.strip().lower()
    if not text:
        raise argparse.ArgumentTypeError("ต้องมี id ของวิชา")
    bad = sorted(set(text) - ID_CHARS)
    if bad:
        raise argparse.ArgumentTypeError(
            f"id ใช้ได้เฉพาะ a-z 0-9 และ - เท่านั้น (เจอ {''.join(bad)!r})\n"
            "  แนะนำรูปแบบ <รหัสวิชา>-<ภาค>-<ปี> เช่น cp462-1-2026"
        )
    return text


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--id", type=valid_id, help="เช่น cp462-1-2026")
    ap.add_argument("--name", help="ชื่อที่นิสิตเห็น เช่น 'CP462 · Introduction to Data Science 1/2026'")
    ap.add_argument("--max-team-size", type=int, default=DEFAULT_MAX_TEAM_SIZE)
    ap.add_argument("--show", action="store_true", help="แสดงวิชาที่มีอยู่แล้วออกไป ไม่สร้างอะไร")
    ap.add_argument("--yes", action="store_true", help="ไม่ต้องถามยืนยัน")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"✗ ไม่พบ {args.db}", file=sys.stderr)
        return 1

    db = Database(args.db)
    try:
        courses = db.load_courses()

        if args.show or not args.id:
            print("วิชาที่มีอยู่")
            if not courses:
                print("  (ยังไม่มี)")
            for c in sorted(courses.values(), key=lambda c: c.id):
                state = "" if c.is_open else "  [ปิดรับคนเข้าแล้ว]"
                print(f"  {c.id:<20} {c.name}")
                print(f"  {'':<20} รหัสเข้าวิชา {c.join_code} · ทีมละไม่เกิน {c.max_team_size}{state}")
            if not args.id:
                if not args.show:
                    print("\nสร้างวิชาใหม่ด้วย --id และ --name")
                return 0
            print()

        if args.id in courses:
            print(f"✗ มีวิชา {args.id!r} อยู่แล้ว — {courses[args.id].name}", file=sys.stderr)
            print("  แก้ชื่อกับขนาดทีมได้จากแผงผู้สอนบนหน้าเว็บ", file=sys.stderr)
            return 1

        name = (args.name or "").strip()
        if not name:
            print("✗ ต้องมี --name — ชื่อที่นิสิตเห็น", file=sys.stderr)
            print("  ถ้าไม่ใส่ วิชาจะชื่อเดียวกับ id ซึ่งอ่านไม่รู้เรื่องสำหรับนิสิต", file=sys.stderr)
            return 1
        if len(name) > MAX_COURSE_NAME_LENGTH:
            print(f"✗ ชื่อยาวเกิน {MAX_COURSE_NAME_LENGTH} ตัวอักษร (ตอนนี้ {len(name)})", file=sys.stderr)
            return 1
        if not 1 <= args.max_team_size <= MAX_TEAM_SIZE_CEILING:
            print(f"✗ ขนาดทีมต้องอยู่ระหว่าง 1 ถึง {MAX_TEAM_SIZE_CEILING}", file=sys.stderr)
            return 1

        course = Course(
            id=args.id,
            name=name,
            max_team_size=args.max_team_size,
            join_code=new_invite_code(),
        )
        print("จะสร้างวิชานี้")
        print(f"  id            {course.id}")
        print(f"  ชื่อ           {course.name}")
        print(f"  ทีมละไม่เกิน   {course.max_team_size} คน")
        print(f"  รหัสเข้าวิชา   {course.join_code}   ← แจกนิสิตในคาบ")
        print("\n  วิชานี้ยังไม่มีโจทย์ — นิสิตเข้าได้แต่จะเห็นว่า \"ยังไม่มีโจทย์ในวิชานี้\"")
        print("  เพิ่มโจทย์ทีหลังด้วย tools/setup_competition.py")

        if not args.yes and input("\nยืนยัน? (พิมพ์ yes) ").strip() != "yes":
            print("ยกเลิก")
            return 1

        db.save_course(course)
        print(f"\n✓ สร้างวิชา {course.id} แล้ว · รหัสเข้าวิชา {course.join_code}")
        print("⚠️ ต้อง restart arena-api ให้โหลดสถานะใหม่ —")
        print("   บริการเก็บ working set ไว้ในหน่วยความจำ จึงยังไม่เห็นการแก้ในไฟล์")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
