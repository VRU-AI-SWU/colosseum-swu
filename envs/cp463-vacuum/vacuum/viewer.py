"""Replay viewer — แปลงไฟล์ `.vrp` เป็นหน้าเว็บไฟล์เดียวที่เปิดดูได้ทันที

    python -m vacuum.viewer replays/1.vrp

**ทำไมไม่ใช่หน้าเว็บบนเซิร์ฟเวอร์** — header ของ `.vrp` มี `seed`, ผังสิ่งกีดขวาง,
ตำแหน่งฝุ่นเริ่มต้น และช่องเหนียวครบทั้งห้อง ([replay.py](replay.py)) และไฟล์ก็ตั้งชื่อ
ตาม seed ตรงๆ · ค่า public seed เป็นความลับเพราะรู้แล้ว overfit ได้
([overview.md §7](../../../docs/competitions/CP463/1-2026/vacuum-robot/overview.md))
การเปิดให้ดาวน์โหลด replay ของ run จริงผ่าน API จึงเท่ากับยกผังห้องทั้ง 30 ห้องให้
— มีเทสต์ `test_api_never_reveals_seed_values` กันเรื่องนี้อยู่แล้ว viewer ตัวนี้จึง
ทำงานกับไฟล์ **ในเครื่องนิสิตเอง** ที่ได้จาก `arena eval --replay-dir` (training seed)

**ทำไม decode ฝั่ง Python แล้วฝัง event ลง HTML แทนที่จะให้เบราว์เซอร์อ่าน .vrp เอง**
การอ่าน `.vrp` ในเบราว์เซอร์ต้องมีตัวแตก zstd ฝั่ง JS ซึ่งแปลว่าต้องมีโค้ดชุดที่สอง
ที่เข้าใจรูปแบบไฟล์เดียวกัน · โค้ดสองชุดที่ทำกติกาเดียวกันจะค่อยๆ ไม่ตรงกันเสมอ
(เจอมาแล้วในโปรเจกต์นี้: `inspect_archive` กับ `ArtifactStore.extract`)
ที่นี่จึงใช้ `replay.decode()` ตัวเดียวกับที่ runner ใช้ แล้วฝัง event ดิบ 4 ไบต์/step
ลงไปเป็น base64 — 1500 step = ~8 KB ซึ่งเล็กกว่าตัวแตก zstd เสียอีก

ตรรกะการเล่นซ้ำใน JS เป็นการสะท้อน `replay.frames()` และ**มีเทสต์เทียบทีละเฟรม**
กับฝั่ง Python อยู่ที่ `tests/test_viewer.py` เพื่อไม่ให้ทั้งสองฝั่งเพี้ยนจากกัน
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import webbrowser
from pathlib import Path

from vacuum.replay import BODY_ITEM, ReplayHeader, read_replay

#: ย่านของ seed ที่แจกให้นิสิตใช้เทรน — นอกย่านนี้คือ seed ที่ใช้ตัดสินคะแนน
TRAINING_SEEDS = range(1, 10_000)

ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "SUCK", "IDLE")


def _events_b64(events: list[tuple[int, int, int]]) -> str:
    """event ดิบตามรูปแบบเดียวกับ body ของ `.vrp` — JS แกะด้วย DataView

    ใช้ `BODY_ITEM` ตัวเดียวกับ `replay.py` เพื่อให้ layout ของไบต์มาจากที่เดียว
    """
    return base64.b64encode(b"".join(BODY_ITEM.pack(*e) for e in events)).decode("ascii")


def build_html(header: ReplayHeader, events: list[tuple[int, int, int]], *, label: str) -> str:
    payload = {
        "label": label,
        "seed": header.seed,
        "envVersion": header.env_version,
        "W": header.W,
        "H": header.H,
        "start": list(header.start),
        "D0": header.D0,
        "maxSteps": header.max_steps,
        "obstacle": header.obstacle_b64,
        "dirt0": header.dirt0_b64,
        "sticky": header.sticky_b64,
        "events": _events_b64(events),
        "nEvents": len(events),
    }
    return _TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))


def render(source: Path, out: Path | None = None) -> Path:
    header, events = read_replay(source)
    target = out or source.with_suffix(".html")
    target.write_text(build_html(header, events, label=source.name), encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vacuum.viewer",
        description="เปิดไฟล์ .vrp เป็นหน้าเว็บที่ดูการเดินของ agent ได้ทีละ step",
    )
    parser.add_argument("replay", type=Path, help="ไฟล์ .vrp จาก `arena eval --replay-dir`")
    parser.add_argument("-o", "--out", type=Path, help="ที่เก็บ HTML (ค่าเริ่มต้น: ข้างๆ ไฟล์ .vrp)")
    parser.add_argument("--no-open", action="store_true", help="สร้างไฟล์อย่างเดียว ไม่เปิดเบราว์เซอร์")
    args = parser.parse_args(argv)

    if not args.replay.is_file():
        print(f"✗ ไม่พบไฟล์ {args.replay}", file=sys.stderr)
        print("  สร้างด้วย `arena eval --config main --seeds 1-3 --replay-dir ./replays`", file=sys.stderr)
        return 1

    try:
        header, events = read_replay(args.replay)
    except ValueError as exc:
        print(f"✗ อ่าน {args.replay} ไม่ได้: {exc}", file=sys.stderr)
        return 1

    target = args.out or args.replay.with_suffix(".html")
    target.write_text(build_html(header, events, label=args.replay.name), encoding="utf-8")

    size_kb = target.stat().st_size / 1024
    print(f"✓ สร้าง {target} ({size_kb:.0f} KB)")
    print(f"  ห้อง {header.W}×{header.H} · ฝุ่นเริ่มต้น {header.D0} · {len(events)} step")

    if header.seed not in TRAINING_SEEDS:
        print(
            f"\n⚠️ seed {header.seed} อยู่นอกย่าน training ({TRAINING_SEEDS.start}–"
            f"{TRAINING_SEEDS.stop - 1}) — เป็น seed ที่ใช้ตัดสินคะแนน\n"
            "   ไฟล์ HTML นี้มีผังห้องและตำแหน่งฝุ่นครบ **ห้ามส่งให้นิสิต**\n"
            "   ค่า seed ที่ใช้ตัดสินเป็นความลับเพราะรู้แล้ว overfit ได้",
            file=sys.stderr,
        )

    if not args.no_open:
        webbrowser.open(target.resolve().as_uri())
    return 0


# ── หน้าเว็บ ────────────────────────────────────────────────────────
# ไฟล์เดียวจบ ไม่มี dependency ภายนอก — เปิดด้วย file:// ได้ ไม่ต้องมีเน็ต
# สีอ่านจาก CSS variable เพื่อให้ธีมสว่าง/มืดใช้โค้ดวาดชุดเดียวกัน

_TEMPLATE = r"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vacuum replay</title>
<style>
:root{
  --bg:#faf9f7; --panel:#fff; --ink:#1c1a17; --dim:#6b6560; --line:#e4e0da;
  --floor:#f5f2ed; --wall:#6d6259; --dirt:#c9973f; --clean:#cfe0ca;
  --trail:#cfe0ee; --sticky:#b8749a; --bot:#1f6b48; --hot:#d4643c;
  --accent:#8a5a2b;
}
/* ธีมมืด: กำแพงต้อง *สว่างกว่า* พื้น ไม่ใช่มืดกว่า — บนพื้นหลังดำ ช่องที่มืดกว่า
   อ่านเป็น "หลุม" ไม่ใช่ "ของตัน" และแยกจากพื้นไม่ออกด้วย */
@media (prefers-color-scheme:dark){:root{
  --bg:#14120f; --panel:#1d1a16; --ink:#eceae6; --dim:#9a938b; --line:#332e28;
  --floor:#1b1815; --wall:#544a40; --dirt:#c99a4a; --clean:#33472f;
  --trail:#28405a; --sticky:#a3628a; --bot:#5fd39a; --hot:#e8794c;
  --accent:#d0a05c;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:20px 18px 48px}
h1{font-size:18px;margin:0 0 2px;letter-spacing:.01em}
.sub{color:var(--dim);font-size:13px;margin:0 0 18px}
.sub b{color:var(--ink);font-weight:600}
.stage{display:grid;grid-template-columns:minmax(0,1fr) 232px;gap:18px;align-items:start}
@media(max-width:780px){.stage{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
canvas{display:block;width:100%;height:auto;image-rendering:pixelated;border-radius:6px}
#strip{height:34px;margin-top:10px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px}
button{font:inherit;font-size:13px;padding:6px 11px;border:1px solid var(--line);
  border-radius:7px;background:var(--bg);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
button[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:var(--panel)}
#scrub{flex:1 1 200px;min-width:140px;accent-color:var(--accent)}
.t{font-variant-numeric:tabular-nums;font-size:13px;color:var(--dim);min-width:96px}
dl{display:grid;grid-template-columns:1fr auto;gap:5px 10px;margin:0;font-size:13px}
dt{color:var(--dim)}
dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
dd.bad{color:var(--hot)}
.flags{margin-top:12px;min-height:44px;display:flex;flex-wrap:wrap;gap:5px;align-content:flex-start}
.flag{font-size:11px;padding:2px 7px;border-radius:99px;border:1px solid var(--line);color:var(--dim)}
.flag.on{background:var(--hot);border-color:var(--hot);color:var(--panel)}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--dim);margin-top:12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;
  vertical-align:-1px}
.hint{font-size:12px;color:var(--dim);margin-top:14px}
kbd{font:inherit;font-size:11px;border:1px solid var(--line);border-bottom-width:2px;
  border-radius:4px;padding:0 4px}
</style>
</head>
<body>
<div class="wrap">
  <h1>Vacuum replay — <span id="label"></span></h1>
  <p class="sub">ห้อง <b id="dims"></b> · ฝุ่นเริ่มต้น <b id="d0"></b> ก้อน ·
     seed <b id="seed"></b> · env <b id="env"></b></p>

  <div class="stage">
    <div class="card">
      <canvas id="grid"></canvas>
      <canvas id="strip"></canvas>
      <div class="controls">
        <button id="play">▶ เล่น</button>
        <button id="back" title="ถอย 1 step">◀</button>
        <button id="fwd" title="เดิน 1 step">▶</button>
        <input id="scrub" type="range" min="0" value="0" step="1">
        <span class="t" id="tlabel"></span>
      </div>
      <div class="controls">
        <button data-speed="1">1×</button>
        <button data-speed="4">4×</button>
        <button data-speed="16">16×</button>
        <button data-speed="64">64×</button>
        <button id="heat" aria-pressed="false" title="สีเข้ม = อยู่ตรงนั้นนาน">แผนที่ความหนาแน่น</button>
        <button id="nextstuck" title="ไปยังจุดที่ชนติดกันหลายครั้ง">จุดที่ติด ▸</button>
      </div>
      <div class="legend">
        <span><i style="background:var(--wall)"></i>กำแพง</span>
        <span><i style="background:var(--dirt)"></i>ฝุ่น</span>
        <span><i style="background:var(--clean)"></i>ดูดแล้ว</span>
        <span><i style="background:var(--trail)"></i>เคยผ่าน</span>
        <span><i style="background:var(--sticky)"></i>ช่องเหนียว</span>
        <span><i style="background:var(--bot)"></i>หุ่น</span>
      </div>
    </div>

    <div class="card">
      <dl>
        <dt>ดูดได้</dt><dd id="s-clean"></dd>
        <dt>ครอบคลุม</dt><dd id="s-cov"></dd>
        <dt>ชนกำแพง</dt><dd id="s-col"></dd>
        <dt>ดูดซ้ำ</dt><dd id="s-red"></dd>
        <dt>ดูดไม่ขึ้น</dt><dd id="s-stk"></dd>
        <dt>ลื่น</dt><dd id="s-slip"></dd>
        <dt>ท่าล่าสุด</dt><dd id="s-act"></dd>
      </dl>
      <div class="flags" id="flags"></div>
      <p class="hint">
        <kbd>space</kbd> เล่น/หยุด · <kbd>←</kbd> <kbd>→</kbd> ทีละ step ·
        <kbd>shift</kbd>+ลูกศร ทีละ 10
      </p>
    </div>
  </div>

  <p class="hint">
    ไฟล์นี้ทำงานได้เองทั้งหมด ไม่ต้องต่อเน็ต — ส่งต่อหรือแนบในรายงานได้เลย
  </p>
</div>

<!-- ตรรกะการเล่นซ้ำ — ไม่แตะ DOM เลยโดยตั้งใจ เพื่อให้เทสต์ดึงไปรันใต้ node
     แล้วเทียบทีละเฟรมกับ replay.frames() ของฝั่ง Python ได้ (tests/test_viewer.py) -->
<script id="replay-core">
const DATA = __PAYLOAD__;

// ── ธงของแต่ละ step — ต้องตรงกับ vacuum/replay.py ─────────────────
const F_MOVED=1, F_COLLISION=2, F_SLIPPED=4, F_CLEANED=8, F_STICKY_FAIL=16, F_REDUNDANT=32;
const ACTIONS = ["UP","DOWN","LEFT","RIGHT","SUCK","IDLE"];

const W = DATA.W, H = DATA.H, N = W*H;

function unpackBits(b64){
  const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const out = new Uint8Array(N);
  for (let i = 0; i < N; i++) out[i] = (raw[i >> 3] >> (7 - (i & 7))) & 1;
  return out;
}

const OBSTACLE = unpackBits(DATA.obstacle);
const DIRT0    = unpackBits(DATA.dirt0);
const STICKY   = unpackBits(DATA.sticky);

// event ดิบ 4 ไบต์ต่อ step — layout เดียวกับ body ของ .vrp (`<BBH`)
const EV = (() => {
  const raw = Uint8Array.from(atob(DATA.events), c => c.charCodeAt(0));
  const dv = new DataView(raw.buffer);
  const n = DATA.nEvents;
  const action = new Uint8Array(n), flags = new Uint8Array(n), flat = new Uint16Array(n);
  for (let i = 0; i < n; i++){
    action[i] = dv.getUint8(i*4);
    flags[i]  = dv.getUint8(i*4 + 1);
    flat[i]   = dv.getUint16(i*4 + 2, true);   // little-endian
  }
  return {n, action, flags, flat};
})();

// ── เล่นซ้ำ — สะท้อน replay.frames() ของฝั่ง Python ────────────────
// ไม่แตะ RNG เลย ทุกอย่างอ่านจาก flags กับตำแหน่งที่บันทึกไว้
// มีเทสต์เทียบทีละเฟรมกับฝั่ง Python อยู่ที่ tests/test_viewer.py
function replayTo(t){
  const dirt = DIRT0.slice();
  const visited = new Uint8Array(N);
  const dwell = new Uint32Array(N);
  let [x, y] = DATA.start;
  visited[y*W + x] = 1;
  dwell[y*W + x] += 1;
  let cleaned=0, collisions=0, redundant=0, stickyFails=0, slips=0;
  let action=null, flags=0;

  for (let i = 0; i < t; i++){
    const f = EV.flags[i], flat = EV.flat[i];
    y = Math.floor(flat / W); x = flat % W;
    if (f & F_MOVED)       visited[flat] = 1;
    if (f & F_CLEANED)   { dirt[flat] = 0; cleaned++; }
    if (f & F_COLLISION)   collisions++;
    if (f & F_SLIPPED)     slips++;
    if (f & F_STICKY_FAIL) stickyFails++;
    if (f & F_REDUNDANT)   redundant++;
    dwell[flat] += 1;
    action = EV.action[i]; flags = f;
  }
  return {t, x, y, action, flags, dirt, visited, dwell,
          cleaned, collisions, redundant, stickyFails, slips};
}

// ── จบส่วนที่ไม่แตะ DOM ────────────────────────────────────────────
</script>

<script id="replay-ui">
// ── วาด ───────────────────────────────────────────────────────────
const grid = document.getElementById("grid");
const gctx = grid.getContext("2d");
const strip = document.getElementById("strip");
const sctx = strip.getContext("2d");
let CELL = 22, heat = false, frame = replayTo(0);

function css(name){ return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function resize(){
  // ต้องคิดความสูงด้วย ไม่ใช่แค่ความกว้าง — ห้อง 20×20 ที่พอดีความกว้างจอ
  // จะสูงจนปุ่มควบคุมตกไปอยู่นอกจอ ซึ่งทำให้ viewer ใช้ไม่ได้ทั้งที่วาดถูก
  const availW = grid.parentElement.clientWidth - 28;
  const availH = Math.max(220, innerHeight - 300);
  CELL = Math.max(6, Math.floor(Math.min(availW / W, availH / H)));
  const dpr = window.devicePixelRatio || 1;
  grid.width = W*CELL*dpr; grid.height = H*CELL*dpr;
  grid.style.width = (W*CELL) + "px"; grid.style.height = (H*CELL) + "px";
  gctx.setTransform(dpr,0,0,dpr,0,0);
  strip.width = strip.clientWidth*dpr; strip.height = 34*dpr;
  sctx.setTransform(dpr,0,0,dpr,0,0);
  draw(); drawStrip();
}

function draw(){
  const C = {floor:css("--floor"), wall:css("--wall"), dirt:css("--dirt"),
             clean:css("--clean"), trail:css("--trail"), sticky:css("--sticky"),
             bot:css("--bot"), hot:css("--hot")};
  const maxDwell = heat ? Math.max(1, ...frame.dwell) : 1;

  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++){
    const i = y*W + x, px = x*CELL, py = y*CELL;
    let fill = C.floor;
    if (OBSTACLE[i])            fill = C.wall;
    else if (heat)              fill = C.floor;
    else if (frame.dirt[i])     fill = C.dirt;
    else if (DIRT0[i])          fill = C.clean;      // เคยมีฝุ่นและดูดไปแล้ว
    else if (frame.visited[i])  fill = C.trail;
    gctx.fillStyle = fill;
    gctx.fillRect(px, py, CELL, CELL);

    if (heat && !OBSTACLE[i] && frame.dwell[i]){
      gctx.globalAlpha = 0.15 + 0.85*Math.min(1, frame.dwell[i]/maxDwell);
      gctx.fillStyle = C.hot;
      gctx.fillRect(px, py, CELL, CELL);
      gctx.globalAlpha = 1;
    }
    if (!heat && STICKY[i] && !OBSTACLE[i]){
      gctx.fillStyle = C.sticky;
      gctx.fillRect(px + CELL - Math.max(3, CELL*0.22), py, Math.max(3, CELL*0.22),
                    Math.max(3, CELL*0.22));
    }
    gctx.strokeStyle = css("--line");
    gctx.lineWidth = 0.5;
    gctx.strokeRect(px + 0.25, py + 0.25, CELL - 0.5, CELL - 0.5);
  }

  // หุ่นมีขอบสีพื้นหลังคั่น เพื่อให้เห็นชัดไม่ว่าจะยืนบนฝุ่น พื้น หรือรอยที่เคยผ่าน
  const bx = frame.x*CELL, by = frame.y*CELL, r = Math.max(2.5, CELL*0.33);
  gctx.beginPath();
  gctx.arc(bx + CELL/2, by + CELL/2, r, 0, Math.PI*2);
  gctx.fillStyle = (frame.flags & F_COLLISION) ? C.hot : C.bot;
  gctx.fill();
  gctx.lineWidth = Math.max(1, CELL*0.09);
  gctx.strokeStyle = css("--panel");
  gctx.stroke();
}

function drawStrip(){
  const w = strip.clientWidth, h = 34;
  sctx.clearRect(0,0,w,h);
  sctx.fillStyle = css("--floor"); sctx.fillRect(0,0,w,h);
  const n = EV.n || 1;
  // แถวบน = ดูดสำเร็จ · แถวล่าง = ปัญหา (ชน / ดูดซ้ำ / ดูดไม่ขึ้น)
  for (let i = 0; i < n; i++){
    const f = EV.flags[i], x = (i/n)*w;
    if (f & F_CLEANED){ sctx.fillStyle = css("--bot"); sctx.fillRect(x, 2, Math.max(1, w/n), 13); }
    if (f & (F_COLLISION|F_REDUNDANT|F_STICKY_FAIL)){
      sctx.fillStyle = css("--hot"); sctx.fillRect(x, 19, Math.max(1, w/n), 13);
    }
  }
  const cx = (frame.t/n)*w;
  sctx.fillStyle = css("--ink");
  sctx.fillRect(Math.min(w-2, Math.max(0, cx-1)), 0, 2, h);
}

// ── สถานะบนแผง ────────────────────────────────────────────────────
const FLAG_LABELS = [[F_MOVED,"เดิน"],[F_COLLISION,"ชน"],[F_SLIPPED,"ลื่น"],
                     [F_CLEANED,"ดูดได้"],[F_STICKY_FAIL,"ดูดไม่ขึ้น"],[F_REDUNDANT,"ดูดซ้ำ"]];
const $ = id => document.getElementById(id);

function paint(){
  const free = N - OBSTACLE.reduce((a,b)=>a+b, 0);
  const seen = frame.visited.reduce((a,b)=>a+b, 0);
  $("s-clean").textContent = `${frame.cleaned} / ${DATA.D0}`;
  $("s-cov").textContent   = (seen/free*100).toFixed(1) + "%";
  $("s-col").textContent   = frame.collisions;
  $("s-col").className     = frame.collisions ? "bad" : "";
  $("s-red").textContent   = frame.redundant;
  $("s-red").className     = frame.redundant ? "bad" : "";
  $("s-stk").textContent   = frame.stickyFails;
  $("s-slip").textContent  = frame.slips;
  $("s-act").textContent   = frame.action === null ? "—" : ACTIONS[frame.action];
  $("tlabel").textContent  = `t ${frame.t} / ${EV.n}`;
  $("flags").innerHTML = FLAG_LABELS
    .map(([bit,name]) => `<span class="flag${frame.flags & bit ? " on":""}">${name}</span>`)
    .join("");
  draw(); drawStrip();
}

function seek(t){
  t = Math.max(0, Math.min(EV.n, t|0));
  frame = replayTo(t);
  $("scrub").value = t;
  paint();
}

// ── การควบคุม ─────────────────────────────────────────────────────
let timer = null, speed = 4;

function setPlaying(on){
  if (timer){ clearInterval(timer); timer = null; }
  $("play").textContent = on ? "❚❚ หยุด" : "▶ เล่น";
  if (!on) return;
  timer = setInterval(() => {
    if (frame.t >= EV.n){ setPlaying(false); return; }
    seek(frame.t + 1);
  }, Math.max(8, 120/speed));
}

$("play").onclick = () => setPlaying(!timer);
$("fwd").onclick  = () => { setPlaying(false); seek(frame.t + 1); };
$("back").onclick = () => { setPlaying(false); seek(frame.t - 1); };
$("scrub").oninput = e => { setPlaying(false); seek(+e.target.value); };

document.querySelectorAll("[data-speed]").forEach(b => {
  b.onclick = () => {
    speed = +b.dataset.speed;
    document.querySelectorAll("[data-speed]").forEach(o =>
      o.setAttribute("aria-pressed", String(o === b)));
    if (timer) setPlaying(true);
  };
});

$("heat").onclick = () => {
  heat = !heat;
  $("heat").setAttribute("aria-pressed", String(heat));
  draw();
};

// "จุดที่ติด" = ช่วง 20 step ที่มีปัญหาถี่ที่สุดที่ยังไม่ได้ดู
$("nextstuck").onclick = () => {
  const WIN = 20;
  let best = -1, bestScore = 0;
  for (let i = frame.t + 1; i < EV.n - WIN; i++){
    let s = 0;
    for (let j = i; j < i + WIN; j++)
      if (EV.flags[j] & (F_COLLISION|F_REDUNDANT|F_STICKY_FAIL)) s++;
    if (s > bestScore){ bestScore = s; best = i; }
    if (s >= WIN) break;   // ติดเต็มหน้าต่างแล้ว ไม่ต้องหาต่อ
  }
  setPlaying(false);
  seek(best >= 0 ? best : 0);
};

addEventListener("keydown", e => {
  const step = e.shiftKey ? 10 : 1;
  if (e.key === " "){ e.preventDefault(); setPlaying(!timer); }
  else if (e.key === "ArrowRight"){ e.preventDefault(); setPlaying(false); seek(frame.t + step); }
  else if (e.key === "ArrowLeft"){ e.preventDefault(); setPlaying(false); seek(frame.t - step); }
});

addEventListener("resize", resize);
matchMedia("(prefers-color-scheme:dark)").addEventListener("change", () => { draw(); drawStrip(); });

// ── เริ่ม ──────────────────────────────────────────────────────────
$("label").textContent = DATA.label;
$("dims").textContent  = `${W}×${H}`;
$("d0").textContent    = DATA.D0;
$("seed").textContent  = DATA.seed;
$("env").textContent   = DATA.envVersion;
$("scrub").max = EV.n;
document.querySelector('[data-speed="4"]').setAttribute("aria-pressed", "true");
resize();
seek(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
