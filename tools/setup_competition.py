"""ตั้งปฏิทินจริงของ competition — แทน placeholder ที่ `core/wiring.py` สร้างไว้ตอน dev

    python tools/setup_competition.py --db /path/arena.db \
        --warmup 2026-09-15..2026-09-30 \
        --main   2026-10-01..2026-10-31 \
        --final  2026-11-01..2026-11-30

**ทำไมต้องมีเครื่องมือนี้** — record ที่รันอยู่บนเครื่องจริงมาจาก `demo_arena()`
ซึ่งตั้ง `opens_at = เมื่อวาน`, `closes_at = อีก 30 วัน` และมี phase เดียวชื่อ `main`
ค่าพวกนั้นเป็นของสำหรับ dev ไม่ใช่ปฏิทินของวิชา ถ้าเปิดให้นิสิตทั้งอย่างนั้น
พวกเขาจะเห็น deadline ปลอมและไม่มีช่วง Warm-up ให้ลองก่อน

**สิ่งที่เครื่องมือนี้ระวังเป็นพิเศษ**

  · `config_override` ของแต่ละ phase **คำนวณจากไฟล์ YAML จริง** ไม่ได้เขียนมือ
    เพราะค่าที่เขียนมือจะค่อยๆ ไม่ตรงกับ config เมื่อมีคนแก้ YAML

  · ตรวจ `config_hash` ของทุก phase **ก่อน**เขียนลงฐานข้อมูล — hash ที่ไม่ตรงกับ
    ตอน generate seed แปลว่า worker จะโยน `ConfigDrift` ตอนให้คะแนนจริง ซึ่งเป็น
    จังหวะที่แย่ที่สุดที่จะรู้ · ถ้าไม่ตรงเครื่องมือนี้จะไม่เขียนอะไรเลย

  · เขตของวันเป็น **เวลาไทย** ไม่ใช่ UTC · `2026-09-30` หมายถึงถึงสิ้นวันนั้น
    ตามเวลาไทย ไม่ใช่ 07:00 ของวันนั้นซึ่งเป็นสิ่งที่จะได้ถ้าใช้ UTC ตรงๆ

  · แก้ record เดิมโดยคง `id` ไว้ — สร้างใหม่จะทำให้ run ที่ส่งไปแล้วกำพร้า

เปิดฐานข้อมูลผ่าน `core.db.Database` เหมือน `retire_team.py` เพื่อให้ migration
ทำงานและเขียนผ่านเส้นทางเดียวกับที่บริการใช้
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "envs" / "cp463-vacuum"))

from core.db import Database  # noqa: E402
from core.domain import Phase, new_id  # noqa: E402
from runners.seeds import expected_config_hash  # noqa: E402

#: เวลาไทย — นิสิตอ่าน deadline เป็นวันตามปฏิทินของตัวเอง ไม่ใช่ UTC
ICT = timezone(timedelta(hours=7))

PHASES = ("warmup", "main", "final")


def parse_range(text: str) -> tuple[datetime, datetime]:
    """`2026-09-15..2026-09-30` → ครึ่งเปิดตามเวลาไทย

    วันจบ **รวมทั้งวัน** — คืนเที่ยงคืนของวันถัดไป เพราะ `Phase.contains` ใช้
    `starts_at <= when < ends_at` ถ้าคืนเที่ยงคืนของวันจบเอง นิสิตจะเสียวันสุดท้ายไปทั้งวัน
    """
    try:
        lo, hi = text.split("..")
        start = datetime.strptime(lo.strip(), "%Y-%m-%d").replace(tzinfo=ICT)
        end = datetime.strptime(hi.strip(), "%Y-%m-%d").replace(tzinfo=ICT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"รูปแบบต้องเป็น YYYY-MM-DD..YYYY-MM-DD (ได้ {text!r})"
        ) from exc
    if end < start:
        raise argparse.ArgumentTypeError(f"วันจบมาก่อนวันเริ่ม: {text!r}")
    return start, end + timedelta(days=1)


def config_override_for(phase: str, base_path: Path) -> dict:
    """diff จาก config ที่ competition ชี้อยู่ ไปเป็น config ของ phase นี้

    คำนวณจากไฟล์จริงทั้งสองฝั่ง ไม่ได้เขียนค่าไว้ในโค้ด — ค่าที่เขียนมือจะไม่ตรง
    กับ YAML ทันทีที่มีคนแก้ YAML แล้วลืมแก้ที่นี่
    """
    from vacuum import load_config
    from vacuum.config import CONFIG_DIR

    base = asdict(load_config(base_path))
    want = asdict(load_config(CONFIG_DIR / f"{phase}.yaml"))

    override: dict = {}
    for section, values in want.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if base.get(section, {}).get(key) != value:
                    override[f"{section}.{key}"] = value
        elif base.get(section) != values:
            override[section] = values
    return override


def verify(base_path: Path, overrides: dict[str, dict], slug: str) -> list[str]:
    """ตรวจว่า config ที่ประกอบได้ของทุก phase ให้ hash ตรงกับตอน generate seed

    นี่คือด่านที่ทำให้เครื่องมือนี้ปลอดภัยพอจะรันกับเครื่องจริง — ถ้า hash ไม่ตรง
    worker จะโยน `ConfigDrift` ตอนให้คะแนน ซึ่งรู้ตอนนั้นสายเกินไป
    """
    from vacuum import load_config

    base = load_config(base_path)
    problems = []
    for phase in PHASES:
        got = base.replace(**overrides[phase]).config_hash
        want = expected_config_hash(competition_slug=slug, phase=phase)
        if want is None:
            problems.append(f"{phase}: ไม่มี config_hash ที่ตรึงไว้ใน seeds.yaml")
        elif got != want:
            problems.append(
                f"{phase}: config_hash ไม่ตรง\n"
                f"      ประกอบได้ : {got}\n"
                f"      seeds.yaml: {want}"
            )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--slug", default="cp463-vacuum-1-2026")
    for phase in PHASES:
        ap.add_argument(f"--{phase}", required=True, type=parse_range, metavar="YYYY-MM-DD..YYYY-MM-DD")
    ap.add_argument(
        "--opens-now",
        action="store_true",
        help="ให้ opens_at เป็นตอนนี้แทนวันเริ่ม Warm-up — ใช้ตอนที่ผู้สอนยังต้องทดสอบเองก่อนถึงวันจริง",
    )
    ap.add_argument("--quota-per-day", type=int)
    ap.add_argument("--yes", action="store_true", help="ไม่ต้องถามยืนยัน")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"✗ ไม่พบ {args.db}", file=sys.stderr)
        return 1

    ranges = {phase: getattr(args, phase) for phase in PHASES}
    for a, b in zip(PHASES, PHASES[1:]):
        if ranges[a][1] > ranges[b][0]:
            print(f"✗ ช่วง {a} กับ {b} ทับกัน", file=sys.stderr)
            return 1

    db = Database(args.db)
    try:
        competitions = db.load_competitions()
        found = [c for c in competitions.values() if c.slug == args.slug]
        if not found:
            print(f"✗ ไม่พบ competition slug {args.slug!r}", file=sys.stderr)
            print(f"  ที่มีอยู่: {sorted(c.slug for c in competitions.values())}", file=sys.stderr)
            return 1
        competition = found[0]

        base_path = Path(competition.config_path)
        if not base_path.is_file():
            print(f"✗ ไม่พบ config ที่ competition ชี้อยู่: {base_path}", file=sys.stderr)
            return 1

        overrides = {phase: config_override_for(phase, base_path) for phase in PHASES}

        problems = verify(base_path, overrides, args.slug)
        if problems:
            print("✗ config_hash ไม่ตรงกับตอน generate seed — ไม่เขียนอะไรลงฐานข้อมูล\n", file=sys.stderr)
            for p in problems:
                print(f"  · {p}", file=sys.stderr)
            print(
                "\n  ถ้า YAML ถูกแก้หลัง generate seed ต้อง generate seed ใหม่"
                " หรือย้อน config กลับ\n"
                "  (ต้องตั้ง ARENA_SECRETS ให้ชี้ไป clone ของ colosseum-hypogeum ด้วย)",
                file=sys.stderr,
            )
            return 1

        opens_at = datetime.now(timezone.utc) if args.opens_now else ranges[PHASES[0]][0]
        closes_at = ranges[PHASES[-1]][1]

        print(f"competition {competition.slug}  ({competition.title})\n")
        print("  ปฏิทินใหม่ (เวลาไทย)")
        for phase in PHASES:
            start, end = ranges[phase]
            n = len(overrides[phase])
            print(
                f"    {phase:<7} {start.astimezone(ICT):%d %b %Y} – "
                f"{(end - timedelta(days=1)).astimezone(ICT):%d %b %Y}"
                f"   config ต่างจาก base {n} ค่า  ✓ hash ตรง"
            )
        print(f"\n    รับ submission {opens_at.astimezone(ICT):%d %b %Y %H:%M} – "
              f"{(closes_at - timedelta(seconds=1)).astimezone(ICT):%d %b %Y %H:%M}")
        if args.opens_now:
            print("    ⚠️ --opens-now: เปิดรับตั้งแต่ตอนนี้ ทั้งที่ Warm-up ยังไม่เริ่ม")
            print("       งานที่ส่งก่อน Warm-up จะถูกให้คะแนนด้วย config ของ main")

        print("\n  ของเดิมที่จะถูกแทน")
        print(f"    รับ submission {competition.opens_at.astimezone(ICT):%d %b %Y} – "
              f"{competition.closes_at.astimezone(ICT):%d %b %Y}")
        print(f"    phase: {', '.join(p.name for p in competition.phases) or '(ไม่มี)'}")

        runs = db.load_runs()
        stale = [r for r in runs.values() if r.competition_id == competition.id
                 and not (opens_at <= r.created_at < closes_at)]
        if stale:
            print(f"\n  ℹ️ มี {len(stale)} run ที่ส่งไว้นอกปฏิทินใหม่ — ยังอยู่ในฐานข้อมูลครบ")
            print("     แต่ `phase_at` จะหาช่วงไม่เจอ แล้วถอยไปใช้ชื่อ 'main'")

        if not args.yes and input("\nยืนยัน? (พิมพ์ yes) ").strip() != "yes":
            print("ยกเลิก")
            return 1

        competition.opens_at = opens_at
        competition.closes_at = closes_at
        competition.phases = [
            Phase(
                id=new_id(),
                name=phase,
                starts_at=ranges[phase][0],
                ends_at=ranges[phase][1],
                config_override=overrides[phase],
            )
            for phase in PHASES
        ]
        if args.quota_per_day:
            competition.quota_per_day = args.quota_per_day
        db.save_competition(competition)

        print("\n✓ ตั้งปฏิทินแล้ว")
        print("⚠️ ต้อง restart arena-api ให้โหลดสถานะใหม่ —")
        print("   บริการเก็บ working set ไว้ในหน่วยความจำ จึงยังไม่เห็นการแก้ในไฟล์")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
