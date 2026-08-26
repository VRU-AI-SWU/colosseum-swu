"""ปลดระวางทีม — ยุบ ไม่ใช่ลบ

ใช้กับทีม demo ที่โทเคนเดาได้ (`team-1` ฯลฯ) ก่อนเปิดให้นิสิตใช้จริง

**ทำไมยุบไม่ลบ** — งานที่เคยส่งและ audit trail ยังต้องตรวจย้อนหลังได้
([README §7](../README.md)) การลบแถวทีมทิ้งจะทำให้ run ที่อ้างถึงมันกลายเป็น
ข้อมูลกำพร้า ซึ่งแย่กว่าการเก็บไว้แล้วทำเครื่องหมายว่าเลิกใช้

ผลของการยุบ — ทั้งสองข้อมีเทสต์คุมอยู่
  · หายจาก leaderboard   `core/leaderboard.py` กรอง `is_active`
  · โทเคนใช้ไม่ได้ทันที   `core/store.py` `team_by_token` กรอง `is_active`
  · ข้อมูลเดิมยังอยู่ครบในฐานข้อมูล

เปิดไฟล์ผ่าน `core.db.Database` ไม่ใช่ `sqlite3` ตรงๆ เพื่อให้ migration ทำงาน
และเขียนผ่านเส้นทางเดียวกับที่บริการใช้ — เครื่องมือที่เขียน SQL เองมีโอกาส
ไม่ตรงกับ schema ที่โค้ดคาดหวัง แล้วพังเงียบๆ ตอนบริการอ่านกลับ

    python tools/retire_team.py --db /path/arena.db --team team-1 team-2 team-3
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.db import Database  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--team", nargs="+", required=True, help="id ของทีมที่จะยุบ")
    ap.add_argument("--yes", action="store_true", help="ไม่ต้องถามยืนยัน")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"✗ ไม่พบ {args.db}", file=sys.stderr)
        return 1

    db = Database(args.db)
    try:
        teams = db.load_teams()
        runs = db.load_runs()

        missing = [t for t in args.team if t not in teams]
        if missing:
            print(f"✗ ไม่พบทีม: {missing}", file=sys.stderr)
            print(f"  ที่มีอยู่: {sorted(teams)}", file=sys.stderr)
            return 1

        targets = [teams[t] for t in args.team]
        print("จะยุบทีมเหล่านี้")
        for team in targets:
            n = sum(1 for r in runs.values() if r.team_id == team.id)
            state = "ยุบไปแล้ว" if not team.is_active else "ใช้งานอยู่"
            print(f"  {team.id:<10} {team.name:<14} {state} · มี {n} run")

        already = [t for t in targets if not t.is_active]
        todo = [t for t in targets if t.is_active]
        if not todo:
            print("\nทุกทีมถูกยุบไปแล้ว ไม่มีอะไรต้องทำ")
            return 0

        if not args.yes and input("\nยืนยัน? (พิมพ์ yes) ").strip() != "yes":
            print("ยกเลิก")
            return 1

        stamp = datetime.now(timezone.utc)
        for team in todo:
            team.dissolved_at = stamp
            db.save_team(team)

        print(f"\n✓ ยุบแล้ว {len(todo)} ทีม" + (f" (ข้าม {len(already)} ที่ยุบอยู่แล้ว)" if already else ""))
        print("⚠️ ต้อง restart arena-api ให้โหลดสถานะใหม่ —")
        print("   บริการเก็บ working set ไว้ในหน่วยความจำ จึงยังไม่เห็นการแก้ในไฟล์")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
