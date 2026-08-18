"""CLI ของนิสิต — README §13

    arena eval --local --seeds 1-30      รันในเครื่องตัวเอง ไม่กินโควตา ไม่ต้องต่อเน็ต
    arena submit --note "เพิ่ม frontier" ส่งงาน
    arena status                          สถานะ run ล่าสุด
    arena leaderboard                     อันดับใน terminal
    arena serve                           รัน API + worker สำหรับ dev

**`eval --local` ใช้ตัวคิดคะแนนตัวเดียวกับ grader** — ตัวเลขที่เห็นในเครื่องจึงเทียบกับ
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
    ที่ทำให้ไฟล์เกินเพดาน 200 MB โดยที่นิสิตไม่รู้ตัว
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


def cmd_eval(args) -> int:
    """รันในเครื่องตัวเอง — ไม่ต่อเน็ต ไม่กินโควตา"""
    from runners.agent_env.runner import run_submission

    seeds = parse_seeds(args.seeds)
    result = run_submission(
        env_plugin=args.env_plugin,
        config_path=args.config,
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
    print(f"seed ที่ใช้     {len(seeds)} ตัว ({args.seeds})")
    print(f"คะแนนรวม       {s.score:+.4f}")
    print(f"ดูดครบ         {s.n_completed}/{len(seeds)}")
    print(f"coverage เฉลี่ย {s.mean_coverage:.3f}")
    print(f"episode แย่สุด  {s.worst_episode:+.4f}")
    print(f"sd ข้าม seed    {s.sd_across_seeds:.4f}  (แสดงอย่างเดียว ไม่ใช้จัดอันดับ)")

    failed = [e for e in result.episodes if e.status != "ok"]
    if failed:
        print(f"\n⚠️ {len(failed)} episode ล้มเหลว:")
        for e in failed[:5]:
            print(f"   seed {e.seed}: {e.status} — {(e.detail or '').splitlines()[-1:]}")
    return 0


def cmd_submit(args) -> int:
    data = pack(Path(args.dir).resolve())
    print(f"แพ็กไฟล์ได้ {len(data) / 1024:.0f} KB")
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


def cmd_serve(args) -> int:
    """รัน API + worker ในกระบวนการเดียว — **สำหรับ dev เท่านั้น**"""
    import threading

    import uvicorn

    from core.api import create_app
    from core.wiring import CP463_VACUUM_LADDER, demo_arena
    from runners.worker import Worker

    root = Path(args.data).resolve()
    arena, teams = demo_arena(root / "artifacts", teams=args.teams)
    worker = Worker(
        runner_id="dev-worker",
        store=arena.store,
        queue=arena.queue,
        artifacts=arena.artifacts,
        workdir=root / "work",
        allow_seed_fallback=True,
    )
    threading.Thread(target=worker.serve_forever, daemon=True).start()

    print(f"โทเคนของทีม: {', '.join(t.id for t in teams)}")
    print(f"⚠️ โหมด dev — ข้อมูลอยู่ในหน่วยความจำ และใช้ seed สำรองที่ไม่ใช่ของจริง")
    uvicorn.run(
        create_app(arena, baselines={"cp463-vacuum-1-2026": CP463_VACUUM_LADDER}),
        host=args.host,
        port=args.port,
        log_level="warning",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arena", description="Arena CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("eval", help="รันในเครื่องตัวเอง (ไม่กินโควตา)")
    p.add_argument("--dir", default=".")
    p.add_argument("--seeds", default="1-20", help="training seeds เช่น 1-30 หรือ 1,5,9")
    p.add_argument("--config", required=True)
    p.add_argument("--env-plugin", default="vacuum.arena:PLUGIN")
    p.add_argument("--replay-dir", default=None)
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
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
