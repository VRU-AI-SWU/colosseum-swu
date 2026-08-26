"""CLI ของนิสิต — README §13

    arena init --dir my-agent             สร้าง starter kit
    arena eval --seeds 1-30               รันในเครื่องตัวเอง ไม่กินโควตา ไม่ต้องต่อเน็ต
    arena eval --check-reset              ตรวจว่า reset() ล้าง state จริง (ทำก่อนส่งเสมอ)
    arena submit --note "เพิ่ม frontier" ส่งงาน
    arena status                          สถานะ run ล่าสุด
    arena leaderboard                     อันดับใน terminal
    arena serve                           รัน API + worker สำหรับ dev

**`eval` ใช้ตัวคิดคะแนนตัวเดียวกับ grader** — ตัวเลขที่เห็นในเครื่องจึงเทียบกับ
บน leaderboard ได้ตรงๆ ต่างกันแค่ seed (นิสิตใช้ training seeds ส่วนที่ตัดสินใช้ public/private
ซึ่งไม่เปิดเผยค่า — [overview §7](../docs/competitions/CP463/1-2026/vacuum-robot/overview.md))

ตั้งค่าผ่านตัวแปรแวดล้อม

    ARENA_URL=http://localhost:8000
    ARENA_TOKEN=team-1
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

DEFAULT_URL = os.environ.get("ARENA_URL", "http://localhost:8000")

#: เพดาน request body ของ Cloudflare แผน Free — **วัดจริงแล้ว** บน
#: colosseum-api.vru-ai.com เมื่อ ส.ค. 2026: 99 MiB ผ่าน · 100 MiB ได้ 413 จาก edge
#: ตัดไว้ต่ำกว่านั้นเพื่อเผื่อ multipart framing ซึ่งบวกเพิ่มอีกไม่กี่ร้อยไบต์
#:
#: ตรวจฝั่ง CLI เพราะถ้าปล่อยไป Cloudflare จะตอบ 413 เปล่าๆ ที่ edge —
#: request ไม่เคยถึง API ของเรา จึงไม่มีทางส่งข้อความที่บอกวิธีแก้กลับไปได้เลย
MAX_UPLOAD_BYTES = 95 * 1024 * 1024
IGNORED = {".git", "__pycache__", ".venv", ".pytest_cache", "models", "tb", ".DS_Store"}


def _client():
    import httpx

    token = os.environ.get("ARENA_TOKEN")
    if not token:
        raise SystemExit("ต้องตั้ง ARENA_TOKEN ก่อน (โทเคนของทีม)")
    return httpx.Client(
        base_url=DEFAULT_URL, headers={"Authorization": f"Bearer {token}"}, timeout=60.0
    )


def pack(directory: Path) -> bytes:
    """zip โฟลเดอร์ปัจจุบัน โดยข้ามของที่ไม่ควรส่ง

    ข้าม `models/` และ `.venv/` ให้อัตโนมัติ — สองอย่างนี้เป็นสาเหตุที่พบบ่อยที่สุด
    ที่ทำให้ไฟล์เกินเพดาน 95 MB โดยที่นิสิตไม่รู้ตัว
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if any(part in IGNORED for part in path.parts):
                continue
            if path.is_file():
                zf.write(path, path.relative_to(directory))
    return buf.getvalue()


def parse_seeds(spec: str) -> list[int]:
    """`"1-30"` หรือ `"1,5,9"` → รายการ seed"""
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif chunk:
            out.append(int(chunk))
    return out


# ── คำสั่ง ──────────────────────────────────────────────────────────


def resolve_config(spec: str, env_plugin: str) -> str:
    """รับได้ทั้งชื่อ phase (`main`) และ path ของไฟล์

    นิสิตที่ `pip install` มาไม่มีโฟลเดอร์ `configs/` ให้ชี้ — config ถูกแพ็กไปกับ
    แพ็กเกจของ environment แล้ว การพิมพ์ชื่อ phase จึงต้องใช้ได้
    """
    if Path(spec).exists():
        return spec
    module = env_plugin.split(":", 1)[0].split(".", 1)[0]
    try:
        config_path = __import__(module, fromlist=["config_path"]).config_path
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"หาไฟล์ config {spec!r} ไม่เจอ และ {module} ไม่มี config_path()") from exc
    try:
        return str(config_path(spec))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{exc}") from exc


def _require_agent_dir(raw: str) -> Path:
    """ยืนยันว่ามี `agent.py` อยู่จริงก่อนไปไกลกว่านี้

    ค่าเริ่มต้นของ `--dir` คือโฟลเดอร์ปัจจุบัน ส่วน `arena init` สร้าง *โฟลเดอร์ใหม่*
    คนที่รัน init แล้วรัน eval ต่อทันทีโดยไม่ `cd` จึงเจอ traceback ของ `agent_host`
    ซึ่งอ่านแล้วเหมือน agent ตัวเองพัง ทั้งที่แค่ยืนผิดที่ · ดักตรงนี้ให้บอกตรงๆ
    """
    target = Path(raw).resolve()
    if (target / "agent.py").is_file():
        return target

    hint = ""
    if target.is_dir():
        nested = sorted(p.parent.name for p in target.glob("*/agent.py"))
        if nested:
            hint = f"\n  โฟลเดอร์งานของคุณน่าจะเป็น {nested[0]}/ — `cd {nested[0]}` ก่อน"
    raise SystemExit(
        f"✗ ไม่พบ agent.py ใน {target}{hint}\n"
        "  ถ้ายังไม่ได้สร้างโฟลเดอร์งาน: `arena init --dir my-agent` แล้ว `cd my-agent`\n"
        "  ถ้าโฟลเดอร์งานอยู่ที่อื่น: ระบุด้วย --dir"
    )


def cmd_eval(args) -> int:
    """รันในเครื่องตัวเอง — ไม่ต่อเน็ต ไม่กินโควตา"""
    from runners.agent_env.runner import run_submission

    _require_agent_dir(args.dir)
    seeds = parse_seeds(args.seeds)
    result = run_submission(
        env_plugin=args.env_plugin,
        config_path=resolve_config(args.config, args.env_plugin),
        submission_dir=Path(args.dir).resolve(),
        seeds=seeds,
        replay_dir=args.replay_dir,
    )
    if not result.ok:
        print(f"❌ {result.status}: {result.detail}", file=sys.stderr)
        if result.log:
            print(result.log[-4000:], file=sys.stderr)
        return 1

    s = result.summary
    worst = min(result.episodes, key=lambda e: e.breakdown.score)
    print(f"seed ที่ใช้     {len(seeds)} ตัว ({args.seeds})")
    print(f"คะแนนรวม       {s.score:+.4f}")
    print(f"ดูดครบ         {s.n_completed}/{len(seeds)}")
    print(f"coverage เฉลี่ย {s.mean_coverage:.4f}")
    print(f"episode แย่สุด  {s.worst_episode:+.4f}  (seed {worst.seed})")
    print(f"sd ข้าม seed    {s.sd_across_seeds:.4f}  (แสดงอย่างเดียว ไม่ใช้จัดอันดับ)")

    if args.per_episode:
        print("\nรายตอน (เรียงจากแย่ไปดี)")
        print(f"  {'seed':>7} {'คะแนน':>9} {'ครบ':>4} {'t_end':>6} {'ชน':>4} {'ดูดซ้ำ':>7}")
        for e in sorted(result.episodes, key=lambda e: e.breakdown.score):
            b = e.breakdown
            print(
                f"  {e.seed:>7} {b.score:>+9.4f} {'✓' if b.completed else '·':>4} "
                f"{b.t_end:>6} {b.collisions:>4} {b.redundant_sucks:>7}"
            )

    if args.verbose and result.log:
        print("\n── log ของ agent (print / stderr) ──")
        print(result.log.rstrip())

    failed = [e for e in result.episodes if e.status != "ok"]
    if failed:
        print(f"\n⚠️ {len(failed)}/{len(seeds)} episode ล้มเหลว — แต่ละอันได้ 0 คะแนน\n")
        for e in failed[:3]:
            print(f"── seed {e.seed}: {e.status} " + "─" * 40)
            print((e.detail or "(ไม่มีรายละเอียด)").rstrip())
            print()
        if len(failed) > 3:
            print(f"(อีก {len(failed) - 3} อัน ไม่แสดง)")
        return 1

    if args.check_reset:
        return _check_reset(args, seeds)
    return 0


def _check_reset(args, seeds: list[int]) -> int:
    """ตรวจว่า `reset()` ล้าง state จริง — **ตรวจเองก่อนส่งได้**

    ระบบตรวจข้อนี้ตอนรับ submission และ**ปฏิเสธ**ถ้าไม่ผ่าน แต่ก่อนหน้านี้นิสิตไม่มี
    ทางรู้ล่วงหน้าเลย เห็นแค่คะแนนต่ำลงเฉยๆ แล้วไล่ debug อัลกอริทึมผิดทาง

    วิธีตรวจ: รัน seed เดียวกันสองแบบ — ต่อกันในกระบวนการเดียว vs แยกกระบวนการ
    ถ้าคะแนนไม่ตรง แปลว่ามีอะไรค้างข้าม episode
    """
    from runners.agent_env.runner import run_submission

    probe = seeds[:3]
    common = dict(
        env_plugin=args.env_plugin,
        config_path=resolve_config(args.config, args.env_plugin),
        submission_dir=Path(args.dir).resolve(),
        # ใช้เลขคี่: ความยาว episode ที่เป็นเลขคู่รักษา parity ของตัวนับที่รั่วไว้พอดี
        # ทำให้การรั่วมองไม่เห็น
        config_overrides={"episode.max_steps": 61},
    )
    print(f"\n── ตรวจว่า reset() ล้าง state จริง (seed {probe}) ──")
    sys.stdout.flush()  # กัน stderr แซงขึ้นไปอยู่เหนือหัวข้อเวลา redirect

    together = run_submission(seeds=probe, **common)
    if not together.ok:
        print(f"❌ รันไม่ผ่าน: {together.status}", file=sys.stderr)
        return 1
    sequential = {e.seed: e.breakdown.score for e in together.episodes}

    bad = []
    for seed in probe:
        alone = run_submission(seeds=[seed], **common)
        if not alone.ok:
            print(f"❌ seed {seed} รันเดี่ยวไม่ผ่าน: {alone.status}", file=sys.stderr)
            return 1
        solo = alone.episodes[0].breakdown.score
        if abs(sequential[seed] - solo) > 1e-9:
            bad.append((seed, sequential[seed], solo))

    if not bad:
        print("✅ ผ่าน — คะแนนตรงกันทั้งสองแบบ submission จะไม่ถูกปฏิเสธด้วยเหตุนี้")
        return 0

    sys.stdout.flush()
    print("❌ **ไม่ผ่าน** — มี state ค้างข้าม episode\n", file=sys.stderr)
    for seed, seq, solo in bad:
        print(
            f"   seed {seed}: ได้ {seq:+.6f} เมื่อรันต่อจาก episode อื่น "
            f"แต่ได้ {solo:+.6f} เมื่อรันเดี่ยว",
            file=sys.stderr,
        )
    print(
        "\n   `reset()` ต้องล้าง state ภายในให้หมด — แผนที่ที่สะสมไว้ ตัวนับ RNG\n"
        "   ทุกอย่างต้องกลับไปเป็นค่าเริ่มต้น · **submission แบบนี้จะถูกปฏิเสธตอนส่ง**",
        file=sys.stderr,
    )
    return 1


def cmd_init(args) -> int:
    """คัดลอก starter kit ออกมาจากแพ็กเกจของ environment (README §13)"""
    import shutil

    module = args.env_plugin.split(":", 1)[0].split(".", 1)[0]
    try:
        package = __import__(module)
    except ImportError:
        raise SystemExit(
            f"ยังไม่ได้ติดตั้ง {module} — `pip install cp463-vacuum` ก่อน"
        ) from None

    source = Path(package.__file__).resolve().parent / "starter"
    if not source.is_dir():
        raise SystemExit(f"{module} ไม่มี starter kit แพ็กมาด้วย")

    target = Path(args.dir).resolve()
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{target} ไม่ว่าง — ระบุโฟลเดอร์ใหม่ด้วย --dir")

    shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"สร้าง starter kit ที่ {target}\n")
    for path in sorted(target.rglob("*")):
        print(f"  {path.relative_to(target)}")
    print(
        "\nขั้นถัดไป\n"
        f"  cd {target}\n"
        "  python -m vacuum.selfcheck              # ตรวจว่าเครื่องคุณตรงกับ grader\n"
        "  arena eval --config main --seeds 1-20   # รัน agent ตัวอย่าง\n"
    )
    return 0


def cmd_submit(args) -> int:
    data = pack(_require_agent_dir(args.dir))
    size_mb = len(data) / 1024 / 1024
    print(f"แพ็กไฟล์ได้ {size_mb:.1f} MB")

    if len(data) > MAX_UPLOAD_BYTES:
        sys.stdout.flush()  # กัน stderr แซงขึ้นไปอยู่เหนือบรรทัด "แพ็กไฟล์ได้"
        big = sorted(
            ((f.stat().st_size, f) for f in Path(args.dir).resolve().rglob("*")
             if f.is_file() and not any(part in IGNORED for part in f.parts)),
            reverse=True,
        )[:3]
        print(
            f"\n❌ ไฟล์ใหญ่เกินไป — {size_mb:.1f} MB เกินเพดาน "
            f"{MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB\n\n"
            "ไฟล์ที่ใหญ่ที่สุดในโฟลเดอร์",
            file=sys.stderr,
        )
        for n, f in big:
            print(f"   {n / 1024 / 1024:>7.1f} MB  {f.relative_to(Path(args.dir).resolve())}",
                  file=sys.stderr)
        print(
            "\nวิธีแก้\n"
            "   · ถ้าเป็น checkpoint ระหว่างเทรน — ส่งเฉพาะตัวสุดท้ายที่ใช้จริง\n"
            "   · ถ้าเป็น policy ที่เทรนมา — บันทึกเฉพาะ weights ไม่ใช่ทั้ง optimizer state\n"
            "     (torch.save(model.state_dict()) ไม่ใช่ torch.save(model))\n"
            "   · โฟลเดอร์ models/ กับ .venv/ ถูกข้ามให้อัตโนมัติอยู่แล้ว",
            file=sys.stderr,
        )
        return 1
    with _client() as client:
        response = client.post(
            f"/api/competitions/{args.competition}/submissions",
            files={"file": ("submission.zip", data, "application/zip")},
            data={"note": args.note, "dry_run": str(args.dry_run).lower()},
        )
    if response.status_code == 422:
        print("❌ ไฟล์ไม่ผ่านการตรวจ:", file=sys.stderr)
        for problem in response.json()["detail"]:
            print(f"   {problem['message']}\n     วิธีแก้: {problem['fix']}", file=sys.stderr)
        return 1
    if response.status_code >= 400:
        print(f"❌ {response.status_code}: {response.json().get('detail')}", file=sys.stderr)
        return 1

    body = response.json()
    print(f"✅ ส่งแล้ว · run {body['run_id'][:8]} · คิวอันดับ {body['queue_position']}")
    print(f"   โควตาเหลือวันนี้ {body['quota_left']} ครั้ง")
    print(f"   ดูสถานะ: arena status {body['submission_id']}")
    return 0


def cmd_status(args) -> int:
    with _client() as client:
        response = client.get(f"/api/submissions/{args.submission_id}")
    if response.status_code >= 400:
        print(f"❌ {response.json().get('detail')}", file=sys.stderr)
        return 1

    body = response.json()
    print(f"submission {body['id'][:8]}  {body['note']}")
    if body["queue_position"] is not None:
        print(f"  รออยู่ในคิวอันดับ {body['queue_position']}")
    for run in body["runs"]:
        line = f"  {run['kind']:8s} {run['status']:8s}"
        if run["score"] is not None:
            line += f" คะแนน {run['score']:+.4f}"
        if run["error"]:
            line += f"  ❌ {run['error'].splitlines()[0][:80]}"
        print(line)
        for key, value in (run.get("metrics") or {}).items():
            if key != "log":
                print(f"      {key}: {value}")
    return 0


def cmd_leaderboard(args) -> int:
    with _client() as client:
        response = client.get(
            f"/api/competitions/{args.competition}/leaderboard", params={"kind": args.kind}
        )
    if response.status_code >= 400:
        print(f"❌ {response.json().get('detail')}", file=sys.stderr)
        return 1

    body = response.json()
    print(f"{body['competition']} · {body['kind']} leaderboard\n")
    for row in body["rows"]:
        if row["type"] == "baseline":
            print(f"      {row['name']:<24s} {row['score']:+.4f}")
        else:
            mark = "→" if row["is_you"] else " "
            print(
                f"{mark} {row['rank']:>3d}. {row['name']:<22s} {row['score']:+.4f}  {row['movement']}"
            )
    target = body.get("next_target")
    if target:
        print(f"\nเป้าหมายถัดไป: {target['name']} ที่ {target['score']:+.4f}")
    return 0


def _pick_launcher(args, DockerLauncher, SubprocessLauncher):
    """เลือกตัวรัน agent — คืน `(launcher, คำอธิบาย)` หรือ `(None, _)` ถ้าต้องหยุด

    `--sandbox auto` (ค่าเริ่มต้น) ใช้ Docker ถ้ามี image พร้อม ไม่งั้นถอยไปใช้ subprocess
    พร้อมเตือนดังๆ — เพราะการพัฒนาบนเครื่องที่ยังไม่ได้ build image ต้องทำได้

    ⚠️ **`--real-seeds` บังคับ Docker** — ถ้ากำลังให้คะแนนจริงด้วย seed จริง แปลว่า
    โค้ดที่รันคือของนิสิต และมันต้องอยู่ใน container ที่ไม่มีเน็ต ไม่ใช่ root
    เขียน rootfs ไม่ได้ ([README §4.1](../README.md)) การถอยไปใช้ subprocess เงียบๆ
    ในโหมดนั้นคือการรันโค้ดที่ไม่ไว้ใจบนเครื่องที่มีเฉลยอยู่
    """
    want = args.sandbox
    ready = DockerLauncher.available()

    if want == "subprocess":
        if args.real_seeds:
            print(
                "✗ --sandbox subprocess ใช้กับ --real-seeds ไม่ได้\n"
                "  โหมดนั้นรันโค้ดของนิสิตด้วย seed จริงบนเครื่องที่มีเฉลยอยู่\n"
                "  โค้ดนั้นต้องอยู่ใน container — build image ก่อนด้วย\n"
                "  docker build -t arena/vacuum:cpu -f runners/agent_env/images/Dockerfile.cpu .",
                file=sys.stderr,
            )
            return None, ""
        return SubprocessLauncher(), "subprocess ⚠️ ไม่มี container ห่อ — dev เท่านั้น"

    if want == "docker" or args.real_seeds:
        if not ready:
            print(
                "✗ ต้องใช้ Docker sandbox แต่ยังไม่พร้อม\n"
                "  ตรวจว่า docker daemon รันอยู่ และมี image arena/vacuum:cpu\n"
                "  docker build -t arena/vacuum:cpu -f runners/agent_env/images/Dockerfile.cpu .",
                file=sys.stderr,
            )
            return None, ""
        return DockerLauncher(), "docker · network none · non-root · read-only rootfs"

    if ready:
        return DockerLauncher(), "docker (auto) · network none · non-root · read-only rootfs"
    print(
        "⚠️ ไม่พบ image arena/vacuum:cpu — ถอยไปใช้ subprocess\n"
        "   โค้ดของนิสิตจะรันโดยไม่มี container ห่อ ห้ามใช้แบบนี้กับของจริง",
        file=sys.stderr,
    )
    return SubprocessLauncher(), "subprocess (auto fallback) ⚠️ ไม่มี container ห่อ"


def cmd_serve(args) -> int:
    """รัน API + worker ในกระบวนการเดียว — **สำหรับ dev เท่านั้น**"""
    import threading

    import uvicorn

    from core.api import create_app
    from core.wiring import CP463_VACUUM_LADDER, demo_arena, google_auth_from_env
    from runners.agent_env.launcher import DockerLauncher, SubprocessLauncher
    from runners.worker import Worker

    root = Path(args.data).resolve()

    launcher, sandbox_note = _pick_launcher(args, DockerLauncher, SubprocessLauncher)
    if launcher is None:
        return 1
    db_path = None if args.ephemeral else root / "arena.db"
    # แยก metadata ออกจาก blob ได้ — สองอย่างนี้โตคนละอัตราและอยากอยู่คนละสื่อ
    #   arena.db   เล็ก · เขียนบ่อย · อยากอยู่บน SSD
    #   artifacts/ ใหญ่ · เขียนครั้งเดียวอ่านนานๆ ครั้ง · อยู่บน HDD ได้สบาย
    # ตรงกับที่ README §11 วางไว้ว่าปลายทางคือ Postgres + object storage คนละที่
    artifacts = Path(args.artifacts).resolve() if args.artifacts else root / "artifacts"
    arena, teams = demo_arena(artifacts, teams=args.teams, db_path=db_path)
    worker = Worker(
        runner_id="dev-worker",
        store=arena.store,
        queue=arena.queue,
        artifacts=arena.artifacts,
        workdir=artifacts.parent / "work",  # แตก zip ข้างๆ ที่เก็บ ไม่ข้ามสื่อ
        launcher=launcher,
        allow_seed_fallback=not args.real_seeds,
    )
    threading.Thread(target=worker.serve_forever, daemon=True).start()

    print(f"โทเคนของทีม: {', '.join(t.id for t in teams)}")
    print(f"sandbox    : {sandbox_note}")
    google = google_auth_from_env()
    if google:
        print(f"ล็อกอิน    : Google Workspace @{google.allowed_domain}")
        print(f"             redirect  {google.redirect_uri}")
        print(f"             ส่งกลับไป {google.web_origin}")
    else:
        print("ล็อกอิน    : ⚠️ ยังไม่ได้ตั้งค่า Google — ใช้โทเคนที่แจกมือเท่านั้น")
    if db_path is None:
        print("⚠️ --ephemeral — ข้อมูลหายเมื่อปิด process")
    else:
        counts = arena.store.db.stats()
        print(f"ฐานข้อมูล  : {db_path}")
        print(f"artifacts  : {artifacts}")
        print(
            "ของเดิมที่โหลดมา: "
            + " · ".join(f"{k} {v}" for k, v in counts.items() if v)
            + (" (ว่าง — เริ่มใหม่)" if not any(counts.values()) else "")
        )
    if args.real_seeds:
        from runners.seeds import secrets_root

        if secrets_root() is None:
            print("✗ --real-seeds ต้องตั้ง ARENA_SECRETS ให้ชี้ไปที่ clone ของ colosseum-hypogeum")
            return 1
        print("⚠️ โหมด dev (ข้อมูลอยู่ในหน่วยความจำ) แต่ใช้ **seed ชุดจริง** — คะแนนเทียบ leaderboard ได้")
    else:
        print(f"⚠️ โหมด dev — ข้อมูลอยู่ในหน่วยความจำ และใช้ seed สำรองที่ไม่ใช่ของจริง")
    uvicorn.run(
        create_app(
            arena,
            baselines={"cp463-vacuum-1-2026": CP463_VACUUM_LADDER},
            allow_origins=args.allow_origin or None,
            google=google,
        ),
        host=args.host,
        port=args.port,
        log_level="warning",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arena", description="Arena CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="สร้าง starter kit ในโฟลเดอร์ใหม่")
    p.add_argument("--dir", default="my-agent")
    p.add_argument("--env-plugin", default="vacuum.arena:PLUGIN")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("eval", help="รันในเครื่องตัวเอง (ไม่กินโควตา)")
    p.add_argument("--dir", default=".")
    p.add_argument("--seeds", default="1-20", help="training seeds เช่น 1-30 หรือ 1,5,9")
    p.add_argument("--config", default="main", help="ชื่อ phase (warmup/main/final) หรือ path")
    p.add_argument("--env-plugin", default="vacuum.arena:PLUGIN")
    p.add_argument("--replay-dir", default=None)
    p.add_argument("--per-episode", action="store_true", help="แสดงคะแนนรายตอนพร้อม seed")
    p.add_argument("--verbose", "-v", action="store_true", help="แสดง print() ของ agent")
    p.add_argument(
        "--check-reset",
        action="store_true",
        help="ตรวจว่า reset() ล้าง state จริง — **ทำก่อนส่งเสมอ** ระบบปฏิเสธ submission ที่ไม่ผ่าน",
    )
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("submit", help="ส่งงาน")
    p.add_argument("competition")
    p.add_argument("--dir", default=".")
    p.add_argument("--note", default="")
    p.add_argument("--dry-run", action="store_true", help="ทดสอบว่าแพ็กไฟล์ถูก ไม่กินโควตา")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="สถานะของ submission")
    p.add_argument("submission_id")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("leaderboard", help="อันดับ")
    p.add_argument("competition")
    p.add_argument("--kind", default="public", choices=["public", "private"])
    p.set_defaults(func=cmd_leaderboard)

    p = sub.add_parser("serve", help="รัน API + worker สำหรับ dev")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data", default=".arena-dev")
    p.add_argument("--teams", type=int, default=3)
    p.add_argument(
        "--artifacts",
        metavar="DIR",
        help="ที่เก็บ zip ของนิสิตกับไฟล์ replay (ค่าเริ่มต้น <data>/artifacts) "
             "— แยกไปไว้บนดิสก์อื่นได้ เพราะเป็นส่วนที่โตเร็วที่สุด",
    )
    p.add_argument(
        "--sandbox",
        choices=("auto", "docker", "subprocess"),
        default="auto",
        help="ตัวรัน agent · auto = ใช้ docker ถ้ามี image พร้อม · --real-seeds บังคับ docker",
    )
    p.add_argument(
        "--ephemeral",
        action="store_true",
        help="ไม่บันทึกลงดิสก์ — ข้อมูลหายเมื่อปิด (ค่าเริ่มต้นคือบันทึกลง <data>/arena.db)",
    )
    p.add_argument(
        "--allow-origin",
        action="append",
        metavar="URL",
        help="โดเมนของหน้าเว็บที่เรียก API นี้ได้ (ใส่ซ้ำได้) เช่น https://colosseum.vru-ai.com",
    )
    p.add_argument(
        "--real-seeds",
        action="store_true",
        help="ใช้ seed ชุดจริงจาก ARENA_SECRETS แทน seed สำรอง — คะแนนที่ได้เทียบ leaderboard ได้",
    )
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
