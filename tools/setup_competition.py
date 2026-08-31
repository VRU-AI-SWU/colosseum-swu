"""สร้าง competition และตั้งปฏิทินจริง — แทน placeholder ที่ `core/wiring.py` สร้างไว้ตอน dev

ตั้งปฏิทินของ competition ที่มีอยู่แล้ว

    python tools/setup_competition.py --db /path/arena.db \
        --warmup 2026-09-15..2026-09-30 \
        --main   2026-10-01..2026-10-31 \
        --final  2026-11-01..2026-11-30

สร้างใหม่ (ใช้ครั้งเดียวต่อ competition)

    python tools/setup_competition.py --db /path/arena.db \
        --slug cp462-churn-1-2026 --create \
        --course cp462-1-2026 --task-type prediction \
        --env-plugin tabular.arena:PLUGIN \
        --config envs/cp462-tabular/tabular/configs/churn.yaml \
        --warmup ... --main ... --final ...

**ทำไมต้องมีเครื่องมือนี้** — record ที่รันอยู่บนเครื่องจริงมาจาก `demo_arena()`
ซึ่งตั้ง `opens_at = เมื่อวาน`, `closes_at = อีก 30 วัน` และมี phase เดียวชื่อ `main`
ค่าพวกนั้นเป็นของสำหรับ dev ไม่ใช่ปฏิทินของวิชา ถ้าเปิดให้นิสิตทั้งอย่างนั้น
พวกเขาจะเห็น deadline ปลอมและไม่มีช่วง Warm-up ให้ลองก่อน

**สิ่งที่เครื่องมือนี้ระวังเป็นพิเศษ**

  · `config_override` ของแต่ละ phase **คำนวณจากไฟล์ YAML จริง** ไม่ได้เขียนมือ
    เพราะค่าที่เขียนมือจะค่อยๆ ไม่ตรงกับ config เมื่อมีคนแก้ YAML

  · **ตรวจของฝั่ง trusted ให้ครบก่อนเขียนลงฐานข้อมูล** — สิ่งที่ตรวจต่างกันตาม
    ชนิดโจทย์ แต่เจตนาเดียวกัน: อะไรที่จะพังตอนให้คะแนนจริง ต้องรู้ตอนนี้
    ไม่ใช่ตอนนิสิตส่งงานแล้ว · ถ้าไม่ผ่านจะไม่เขียนอะไรเลย

  · เขตของวันเป็น **เวลาไทย** ไม่ใช่ UTC · `2026-09-30` หมายถึงถึงสิ้นวันนั้น
    ตามเวลาไทย ไม่ใช่ 07:00 ของวันนั้นซึ่งเป็นสิ่งที่จะได้ถ้าใช้ UTC ตรงๆ

  · แก้ record เดิมโดยคง `id` ไว้ — สร้างใหม่จะทำให้ run ที่ส่งไปแล้วกำพร้า

**การจับกลุ่มไม่ได้อยู่ที่นี่** — ขนาดทีมกับรหัสเข้าวิชาเป็นของ `Course`
(`tools/setup_course.py`) เพราะทุก competition ในวิชาเดียวกันใช้ทีมชุดเดียวกัน
ถ้าแยกไปอยู่ที่ competition นิสิตจะต้องจับกลุ่มใหม่ทุกครั้งที่มีโจทย์ใหม่

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

from core.calendar import ICT, PHASES, CalendarInvalid, build_phases  # noqa: E402
from core.calendar import parse_range as _parse_range  # noqa: E402
from core.db import Database  # noqa: E402
from core.domain import Competition, new_id  # noqa: E402


# ── สิ่งที่ต่างกันตามชนิดโจทย์ ──────────────────────────────────────
#
# ที่เหลือของเครื่องมือนี้ไม่รู้จักชนิดโจทย์เลย — การเพิ่มชนิดที่สามแตะแค่ตรงนี้


class AgentEnvTask:
    """โจทย์ RL (CP463) — config ต่างกันทุก phase และผูกกับ seed ที่ generate ไว้"""

    task_type = "agent_env"
    paradigm = "reinforcement-learning"
    env_root = REPO / "envs" / "cp463-vacuum"
    #: ว่าง = ใช้ค่าปริยายของ `Competition.effective_whitelist()`
    whitelist: frozenset[str] = frozenset()

    def title(self, base_path: Path) -> str | None:
        return None  # config ของ CP463 ไม่มีชื่อโจทย์อยู่ในนั้น

    def overrides(self, phase: str, base_path: Path) -> dict:
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

    def verify(self, base_path: Path, overrides: dict[str, dict], slug: str) -> list[str]:
        """config ที่ประกอบได้ของทุก phase ต้องให้ hash ตรงกับตอน generate seed

        ถ้า hash ไม่ตรง worker จะโยน `ConfigDrift` ตอนให้คะแนน ซึ่งรู้ตอนนั้นสายเกินไป
        """
        from runners.seeds import expected_config_hash
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


class PredictionTask:
    """โจทย์ทำนาย (CP462) — โจทย์เดียวตรึงทั้งเทอม เปลี่ยนแค่ชุดที่ใช้ตัดสิน

    **ไม่มี `config_override` ต่อ phase โดยตั้งใจ** — การเปลี่ยนสเปคกลางเทอมแปลว่า
    `config_hash` เปลี่ยน แล้วคะแนนก่อนกับหลังเทียบกันไม่ได้ · สิ่งที่ต่างกันระหว่าง
    phase คือ *ปฏิทิน* กับ *ชุดที่ใช้ตัดสิน* (public ระหว่างเทอม · private ตอนปิดรับ)
    ซึ่งทั้งคู่ไม่ได้อยู่ใน config
    """

    task_type = "prediction"
    paradigm = "supervised-learning"
    env_root = REPO / "envs" / "cp462-tabular"
    #: ต้องตรงกับที่ติดตั้งอยู่ใน `arena/tabular:cpu` — **ไม่มี `tabular`** เพราะมัน
    #: เห็นเฉลยและจงใจไม่อยู่ใน image · ถ้าใส่เข้าไป นิสิตที่ `import tabular` จะผ่าน
    #: การตรวจ กินโควตา แล้วไปตายในกล่องด้วย ImportError
    whitelist = frozenset({"numpy", "pandas", "sklearn", "scipy", "joblib"})

    def title(self, base_path: Path) -> str | None:
        from tabular.config import load_config

        return load_config(base_path).title

    def overrides(self, phase: str, base_path: Path) -> dict:
        return {}

    def verify(self, base_path: Path, overrides: dict[str, dict], slug: str) -> list[str]:
        """**ไฟล์ข้อมูลต้องแบ่งได้จริง** ก่อนเปิดรับ submission

        เจตนาเดียวกับการตรวจ `config_hash` ของ CP463 — อะไรที่จะทำให้การให้คะแนน
        ล้ม ต้องรู้ตอนตั้งค่า ไม่ใช่ตอนนิสิตส่งงานแล้ว · ถ้าไฟล์ไม่อยู่ในคลังของ
        เครื่องนี้ worker จะล้มทุก run โดยที่นิสิตไม่ได้ทำอะไรผิด

        เดิมข้อนี้ตรวจว่ามีเมล็ดลับใน `ARENA_SECRETS` ไหม · ตอนนี้ไม่มีเมล็ดแล้ว
        และสิ่งที่ตรวจแทนแข็งแรงกว่า เพราะมัน **แบ่งข้อมูลจริง** ไม่ใช่แค่ดูว่ามี
        ไฟล์ลับอยู่ — ปัญหาอย่างคลาสที่บางเกินไปจึงโผล่ตอนนี้ ไม่ใช่ตอนตัดสินเกรด
        """
        from tabular.arena import PLUGIN
        from tabular.config import load_config
        from tabular.store import DatasetError

        try:
            preview = PLUGIN.preview(load_config(base_path))
        except DatasetError as exc:
            return [f"ชุดข้อมูล: {exc}"]

        problems = []
        for label, counts in (preview.get("thin") or {}).items():
            problems.append(
                f"คลาส {label!r} บางเกินไป — จะเหลือ {counts['test_private']} แถว"
                f"ในกองที่ตัดสินรอบสุดท้าย · อันดับจะขึ้นกับโชคมากกว่าโมเดล"
            )
        return problems

    def summary(self, base_path: Path) -> list[str]:
        from tabular.arena import PLUGIN
        from tabular.config import load_config
        from tabular.store import DatasetError

        spec = load_config(base_path)
        lines = [
            f"โจทย์      {spec.title} · {spec.kind} · คะแนนหลัก {spec.primary}",
            f"ชุดข้อมูล   {spec.dataset[:23]}… · เฉลยคือคอลัมน์ {spec.target!r}"
            + (f" · ตัดทิ้ง {spec.drop}" if spec.drop else ""),
        ]
        try:
            sizes = PLUGIN.preview(spec)["sizes"]
        except DatasetError as exc:
            lines.append(f"การแบ่ง     อ่านไฟล์ไม่ได้ — {exc}")
        else:
            lines.append(
                f"การแบ่ง     แจกนิสิต {sizes['student']} แถว · "
                f"กระดาน {sizes['test_public']} · ตัดสิน {sizes['test_private']}"
            )
        lines.append(f"config_hash {spec.config_hash}")
        return lines


TASK_TYPES = {t.task_type: t for t in (AgentEnvTask(), PredictionTask())}


def summary_lines(task, base_path: Path) -> list[str]:
    fn = getattr(task, "summary", None)
    return fn(base_path) if fn else []


# ── ปฏิทิน ─────────────────────────────────────────────────────────


def parse_range(text: str) -> tuple[datetime, datetime]:
    """`2026-09-15..2026-09-30` → ครึ่งเปิดตามเวลาไทย

    กติกาเรื่องวันอยู่ที่ `core/calendar.py` ที่เดียว — ใช้ร่วมกับ endpoint ที่ผู้สอน
    กดจากหน้าเว็บ · ถ้าเขียนแยกกัน deadline ที่หน้าเว็บบอกกับที่ระบบใช้จะเพี้ยนกัน
    ห่อเป็น `ArgumentTypeError` เพื่อให้ argparse พิมพ์ข้อความสวยๆ ให้
    """
    try:
        return _parse_range(text)
    except CalendarInvalid as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _confirmed() -> bool:
    """ถามยืนยัน — **ไม่มี tty ถือว่าไม่ยืนยัน**

    เดิมโยน `EOFError` พร้อม traceback ตอนรันแบบไม่มี stdin (ผ่าน ssh หรือใน
    สคริปต์) ซึ่งอ่านแล้วเหมือนเครื่องมือพัง ทั้งที่พฤติกรรมถูกอยู่แล้วคือไม่เขียน
    อะไรลงไป · คนที่ตั้งใจรันแบบอัตโนมัติใช้ `--yes`
    """
    try:
        return input("\nยืนยัน? (พิมพ์ yes) ").strip() == "yes"
    except EOFError:
        print("\n(ไม่มี stdin ให้ตอบ — ถ้าตั้งใจรันแบบอัตโนมัติให้ใส่ --yes)")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--slug", default="cp463-vacuum-1-2026")
    for phase in PHASES:
        ap.add_argument(
            f"--{phase}", required=True, type=parse_range, metavar="YYYY-MM-DD..YYYY-MM-DD"
        )
    ap.add_argument(
        "--opens-now",
        action="store_true",
        help="ให้ opens_at เป็นตอนนี้แทนวันเริ่ม Warm-up — ใช้ตอนที่ผู้สอนยังต้องทดสอบเองก่อนถึงวันจริง",
    )
    ap.add_argument("--quota-per-day", type=int)
    ap.add_argument("--yes", action="store_true", help="ไม่ต้องถามยืนยัน")

    new = ap.add_argument_group("สร้างใหม่ (ใช้ครั้งเดียวต่อ competition)")
    new.add_argument("--create", action="store_true", help="สร้าง competition ถ้ายังไม่มี")
    new.add_argument("--course", help="id ของวิชา — ทีมกับขนาดทีมมาจากวิชานี้")
    new.add_argument("--task-type", choices=sorted(TASK_TYPES))
    new.add_argument("--env-plugin", help='เช่น "tabular.arena:PLUGIN"')
    new.add_argument("--config", type=Path, help="ไฟล์ config ของโจทย์")
    new.add_argument("--title", help="ค่าปริยายมาจากไฟล์ config ถ้าโจทย์นั้นมีชื่ออยู่ในนั้น")
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

        if found:
            competition, creating = found[0], False
            if args.create:
                print(f"ℹ️ มี {args.slug} อยู่แล้ว — จะตั้งปฏิทินให้ ไม่ได้สร้างใหม่")
        else:
            if not args.create:
                print(f"✗ ไม่พบ competition slug {args.slug!r}", file=sys.stderr)
                print(
                    f"  ที่มีอยู่: {sorted(c.slug for c in competitions.values())}", file=sys.stderr
                )
                print("  ถ้าตั้งใจจะสร้างใหม่ ใส่ --create พร้อม --course/--task-type/"
                      "--env-plugin/--config", file=sys.stderr)
                return 1
            missing = [
                f"--{name.replace('_', '-')}"
                for name in ("course", "task_type", "env_plugin", "config")
                if not getattr(args, name)
            ]
            if missing:
                print(f"✗ --create ต้องมี {', '.join(missing)} ด้วย", file=sys.stderr)
                return 1
            if args.course not in db.load_courses():
                print(f"✗ ไม่พบวิชา {args.course!r} — สร้างด้วย tools/setup_course.py ก่อน",
                      file=sys.stderr)
                return 1
            competition, creating = None, True

        task_type = args.task_type if creating else competition.task_type
        task = TASK_TYPES.get(task_type)
        if task is None:
            print(f"✗ ไม่รู้จัก task_type {task_type!r} — ที่มีคือ {sorted(TASK_TYPES)}",
                  file=sys.stderr)
            return 1
        sys.path.insert(0, str(task.env_root))

        base_path = (args.config if creating else Path(competition.config_path)).resolve()
        if not base_path.is_file():
            print(f"✗ ไม่พบ config: {base_path}", file=sys.stderr)
            return 1

        overrides = {phase: task.overrides(phase, base_path) for phase in PHASES}

        problems = task.verify(base_path, overrides, args.slug)
        if problems:
            print("✗ ตรวจฝั่ง trusted ไม่ผ่าน — ไม่เขียนอะไรลงฐานข้อมูล\n", file=sys.stderr)
            for p in problems:
                print(f"  · {p}", file=sys.stderr)
            print(
                "\n  (ต้องตั้ง ARENA_SECRETS ให้ชี้ไป clone ของ colosseum-hypogeum ด้วย)",
                file=sys.stderr,
            )
            return 1

        opens_at = datetime.now(timezone.utc) if args.opens_now else ranges[PHASES[0]][0]
        closes_at = ranges[PHASES[-1]][1]

        if creating:
            title = args.title or task.title(base_path) or args.slug
            competition = Competition(
                id=new_id(),
                course_id=args.course,
                slug=args.slug,
                title=title,
                task_type=task.task_type,
                env_plugin=args.env_plugin,
                config_path=str(base_path),
                # เก็บ **เนื้อหา** ไว้ด้วยเสมอ — competition ต้องเป็นข้อมูลที่สมบูรณ์
                # ในตัวเอง ไม่ใช่ตัวชี้ไปยังไฟล์บนเครื่องหนึ่งเครื่อง (schema v5)
                config_text=base_path.read_text(encoding="utf-8"),
                opens_at=opens_at,
                closes_at=closes_at,
                quota_per_day=args.quota_per_day or 5,
                import_whitelist=task.whitelist,
                paradigm=task.paradigm,
            )
            print(f"สร้าง competition ใหม่: {competition.slug}  ({competition.title})\n")
            print(f"  วิชา        {competition.course_id}")
            print(f"  ชนิดโจทย์    {competition.task_type} · {competition.paradigm}")
            print(f"  plugin      {competition.env_plugin}")
            print(f"  whitelist   {sorted(competition.effective_whitelist())}")
        else:
            print(f"competition {competition.slug}  ({competition.title})\n")

        for line in summary_lines(task, base_path):
            print(f"  {line}")
        if summary_lines(task, base_path):
            print()

        print("  ปฏิทินใหม่ (เวลาไทย)")
        for phase in PHASES:
            start, end = ranges[phase]
            n = len(overrides[phase])
            note = f"config ต่างจาก base {n} ค่า" if n else "config เดียวกันทุก phase"
            print(
                f"    {phase:<7} {start.astimezone(ICT):%d %b %Y} – "
                f"{(end - timedelta(days=1)).astimezone(ICT):%d %b %Y}   {note}"
            )
        print(f"\n    รับ submission {opens_at.astimezone(ICT):%d %b %Y %H:%M} – "
              f"{(closes_at - timedelta(seconds=1)).astimezone(ICT):%d %b %Y %H:%M}")
        print(f"    โควตา {competition.quota_per_day if not creating else (args.quota_per_day or 5)}"
              " ครั้ง/วัน/ทีม")
        if args.opens_now:
            print("    ⚠️ --opens-now: เปิดรับตั้งแต่ตอนนี้ ทั้งที่ Warm-up ยังไม่เริ่ม")
            print("       งานที่ส่งก่อน Warm-up จะถูกให้คะแนนด้วย config ของ main")

        if not creating:
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

        if not args.yes and not _confirmed():
            print("ยกเลิก — ไม่ได้เขียนอะไรลงฐานข้อมูล")
            return 1

        competition.opens_at = opens_at
        competition.closes_at = closes_at
        competition.phases = build_phases(ranges, overrides=overrides)
        # record ที่สร้างก่อน v5 ยังเก็บแค่ path — ตั้งปฏิทินทีก็ย้ายเนื้อหาเข้ามาที
        if not competition.config_text.strip():
            competition.config_text = base_path.read_text(encoding="utf-8")
        if args.quota_per_day:
            competition.quota_per_day = args.quota_per_day
        db.save_competition(competition)

        print("\n✓ สร้างแล้ว" if creating else "\n✓ ตั้งปฏิทินแล้ว")
        print("⚠️ ต้อง restart arena-api ให้โหลดสถานะใหม่ —")
        print("   บริการเก็บ working set ไว้ในหน่วยความจำ จึงยังไม่เห็นการแก้ในไฟล์")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
