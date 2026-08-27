"""Replay viewer — ตรรกะเล่นซ้ำใน JS ต้องให้ผลเท่ากับ `replay.frames()` ของ Python

**นี่คือเหตุผลหลักที่ไฟล์นี้มีอยู่** · viewer ฝัง event ดิบลง HTML แล้วให้ JS
สร้าง state ขึ้นมาใหม่ ซึ่งแปลว่ามีโค้ดสองชุดที่ทำกติกาเดียวกัน — และโค้ดสองชุด
แบบนั้นจะค่อยๆ ไม่ตรงกันเสมอถ้าไม่มีอะไรผูกไว้ (ในโปรเจกต์นี้เคยเกิดกับ
`inspect_archive` กับ `ArtifactStore.extract` มาแล้ว: ฝั่งหนึ่งยอมรับกว้างกว่า
อีกฝั่ง งานเลยผ่านการตรวจ กินโควตา แล้วไปตายตอนรัน)

เทสต์นี้ดึง `<script id="replay-core">` ออกมารันใต้ node จริง แล้วเทียบ
**ทุกเฟรม** ไม่ใช่แค่เฟรมสุดท้าย — ความต่างที่หักล้างกันเองระหว่างทางจะได้ไม่รอด

ข้ามอัตโนมัติถ้าเครื่องไม่มี node (นิสิตไม่จำเป็นต้องมี — viewer รันในเบราว์เซอร์)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap

import numpy as np
import pytest

from vacuum import load_config
from vacuum.config import CONFIG_DIR
from vacuum.env import VacuumEnv
from vacuum.replay import frames, header_from_env
from vacuum.viewer import build_html

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="ต้องมี node ถึงจะรันตรรกะฝั่ง JS ได้")

CORE = re.compile(r'<script id="replay-core">(.*?)</script>', re.S)


def _episode(seed: int, phase: str = "main"):
    """รัน agent ง่ายๆ ให้ครบ episode แล้วคืน (header, events, env)

    ใช้ policy ที่วนท่าไปเรื่อยๆ เพื่อให้เกิดครบทุก flag — เดิน ชน ดูดได้ ดูดซ้ำ
    ดูดไม่ขึ้น และลื่น · ถ้า agent ฉลาดเกินไปจะไม่มีเคสผิดพลาดให้เทียบ
    """
    env = VacuumEnv(load_config(CONFIG_DIR / f"{phase}.yaml"))
    env.reset(seed=seed)
    rng = np.random.default_rng(12345)
    for _ in range(env.config.episode.max_steps):
        _obs, _r, terminated, truncated, _info = env.step(int(rng.integers(0, 6)))
        if terminated or truncated:
            break
    return header_from_env(env), list(env.events), env


def _run_js(html: str, probes: list[int]) -> list[dict]:
    core = CORE.search(html)
    assert core, "ไม่พบ <script id=\"replay-core\"> — โครงของ template เปลี่ยนไปแล้ว"
    harness = textwrap.dedent(
        """
        const PROBES = %s;
        const out = PROBES.map(t => {
          const f = replayTo(t);
          return {
            t: f.t, x: f.x, y: f.y, action: f.action, flags: f.flags,
            cleaned: f.cleaned, collisions: f.collisions, redundant: f.redundant,
            stickyFails: f.stickyFails, slips: f.slips,
            dirt: Array.from(f.dirt).join(""),
            visited: Array.from(f.visited).join(""),
          };
        });
        console.log(JSON.stringify(out));
        """
        % json.dumps(probes)
    )
    proc = subprocess.run(
        [NODE, "--input-type=commonjs", "-e", core.group(1) + harness],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"node ล้มเหลว:\n{proc.stderr}"
    return json.loads(proc.stdout)


@needs_node
def test_js_replay_matches_python_frame_by_frame():
    header, events, _env = _episode(seed=7)
    html = build_html(header, events, label="7.vrp")

    python_frames = list(frames(header, events))
    assert len(python_frames) == len(events) + 1

    # ทุกเฟรมสำหรับ episode สั้นๆ ไม่ไหว (1500 × 400 ช่อง) — สุ่มจุดตรวจแบบกระจาย
    # รวมหัวและท้ายเสมอ เพราะสองจุดนั้นเป็นที่ที่ off-by-one ชอบซ่อน
    last = len(events)
    probes = sorted({0, 1, 2, last - 1, last, *range(0, last, max(1, last // 40))})
    js_frames = _run_js(html, probes)
    assert len(js_frames) == len(probes)

    for probe, js in zip(probes, js_frames):
        py = python_frames[probe]
        where = f"t={probe}"
        assert (js["x"], js["y"]) == py.pos, f"{where}: ตำแหน่งไม่ตรง"
        assert js["action"] == py.action, f"{where}: action ไม่ตรง"
        assert js["flags"] == py.flags, f"{where}: flags ไม่ตรง"
        assert js["cleaned"] == py.cleaned, f"{where}: cleaned ไม่ตรง"
        assert js["collisions"] == py.collisions, f"{where}: collisions ไม่ตรง"
        assert js["redundant"] == py.redundant_sucks, f"{where}: redundant_sucks ไม่ตรง"
        assert js["stickyFails"] == py.sticky_fails, f"{where}: sticky_fails ไม่ตรง"
        assert js["slips"] == py.slips, f"{where}: slips ไม่ตรง"
        assert js["dirt"] == "".join(py.dirt.reshape(-1).astype(int).astype(str)), (
            f"{where}: แผนที่ฝุ่นไม่ตรง"
        )
        assert js["visited"] == "".join(py.visited.reshape(-1).astype(int).astype(str)), (
            f"{where}: ช่องที่เคยผ่านไม่ตรง"
        )


@needs_node
def test_js_unpacks_the_same_layout_python_sees():
    """ผังห้องเดินทางเป็น bitmask base64 — ลำดับบิตผิดแค่นิดเดียวก็ได้ห้องคนละใบ"""
    header, events, _env = _episode(seed=11)
    html = build_html(header, events, label="11.vrp")

    core = CORE.search(html).group(1)
    proc = subprocess.run(
        [NODE, "--input-type=commonjs", "-e", core + """
        console.log(JSON.stringify({
          obstacle: Array.from(OBSTACLE).join(""),
          dirt0: Array.from(DIRT0).join(""),
          sticky: Array.from(STICKY).join(""),
        }));
        """],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)

    def flat(mask: np.ndarray) -> str:
        return "".join(mask.reshape(-1).astype(int).astype(str))

    assert got["obstacle"] == flat(header.obstacle)
    assert got["dirt0"] == flat(header.dirt0)
    assert got["sticky"] == flat(header.sticky)


def test_html_is_self_contained():
    """ต้องเปิดด้วย file:// ได้โดยไม่ต้องต่อเน็ต — นิสิตแนบไฟล์นี้ในรายงานได้

    ถ้าวันหนึ่งมีคนเพิ่ม CDN หรือฟอนต์จากภายนอกเข้ามา เทสต์นี้จะจับได้
    """
    header, events, _env = _episode(seed=3)
    html = build_html(header, events, label="3.vrp")
    for pattern in ("http://", "https://", "src=", "@import", "//cdn"):
        assert pattern not in html, f"หน้าเว็บอ้างของจากภายนอก: {pattern!r}"


def test_payload_carries_every_event():
    """event ทั้งหมดต้องเดินทางไปถึง HTML — ตกหล่นแปลว่า replay สั้นกว่าของจริง"""
    header, events, _env = _episode(seed=5)
    html = build_html(header, events, label="5.vrp")
    payload = json.loads(re.search(r"const DATA = (\{.*?\});", html, re.S).group(1))
    assert payload["nEvents"] == len(events)
    assert payload["seed"] == header.seed
    assert payload["W"] == header.W and payload["H"] == header.H


def test_grading_seed_gets_a_warning(capsys, tmp_path):
    """สร้าง viewer จาก seed ที่ใช้ตัดสิน ต้องเตือนว่าห้ามส่งไฟล์ให้นิสิต

    ไฟล์ HTML มีผังห้องกับตำแหน่งฝุ่นครบ การแจกออกไปคือการยกคำตอบให้
    """
    from vacuum.replay import encode
    from vacuum.viewer import main

    header, events, _env = _episode(seed=7)
    grading = type(header)(**{**header.__dict__, "seed": 20_007})
    path = tmp_path / "20007.vrp"
    path.write_bytes(encode(grading, events))

    assert main([str(path), "--no-open"]) == 0
    assert "ห้ามส่งให้นิสิต" in capsys.readouterr().err


def test_training_seed_gets_no_warning(capsys, tmp_path):
    from vacuum.replay import encode
    from vacuum.viewer import main

    header, events, _env = _episode(seed=7)
    path = tmp_path / "7.vrp"
    path.write_bytes(encode(header, events))

    assert main([str(path), "--no-open"]) == 0
    assert "ห้ามส่ง" not in capsys.readouterr().err


# ── กติกาของสี ─────────────────────────────────────────────────────


def _luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _palette(css: str, block: str) -> dict[str, str]:
    """ดึงตัวแปรสีของธีมหนึ่งออกมาจาก CSS"""
    body = re.search(block, css, re.S).group(1)
    return dict(re.findall(r"--([a-z]+):(#[0-9a-f]{6})", body))


@pytest.mark.parametrize(
    "theme,block",
    [
        ("สว่าง", r"\n:root\{(.*?)\n\}"),
        ("มืด", r"prefers-color-scheme:dark\)\{:root\{(.*?)\n\}\}"),
    ],
)
def test_walls_are_the_darkest_thing_in_the_room(theme, block):
    """กำแพงต้องมืดที่สุดเสมอ ทั้งสองธีม

    เคยทำกลับกันในธีมมืด (กำแพงสว่างกว่าพื้น) ด้วยเหตุผลว่าช่องมืดบนพื้นหลังดำ
    อ่านเป็น "หลุม" · เหตุผลฟังขึ้นแต่ผลจริงคือคนอ่านช่องสีดำ — ซึ่งคือ*พื้น* —
    ว่าเป็นกำแพง แล้วงงว่าทำไมหุ่นเดินทับได้ · คำถามแรกที่ได้จากคนใช้จริงคือข้อนี้
    """
    from vacuum.viewer import _TEMPLATE

    colors = _palette(_TEMPLATE, block)
    for key in ("wall", "floor", "bg", "dirt"):
        assert key in colors, f"ธีม{theme}: ไม่มีตัวแปร --{key}"

    wall, floor = _luminance(colors["wall"]), _luminance(colors["floor"])
    assert wall < floor, (
        f"ธีม{theme}: กำแพง ({colors['wall']}) ต้องมืดกว่าพื้น ({colors['floor']})"
    )
    # พื้นต้องต่างจากพื้นหลังหน้าเว็บพอที่จะเห็นขอบห้อง
    assert abs(floor - _luminance(colors["bg"])) > 0.02, (
        f"ธีม{theme}: พื้นห้องกับพื้นหลังหน้าเว็บใกล้กันเกินไป — มองไม่เห็นขอบห้อง"
    )


def test_legend_says_walls_cannot_be_walked_on():
    """คำอธิบายสีต้องบอกว่ากำแพง*ทำอะไร* ไม่ใช่แค่บอกชื่อสี

    เจาะจงที่ตัวคำอธิบายสีจริงๆ ไม่ใช่แค่หาคำในทั้งหน้า — คำว่า "เดินทับไม่ได้"
    ยังโผล่ในข้อความตอนชี้ดูรายช่องด้วย การหาแบบกว้างจึงผ่านทั้งที่คำอธิบายสีหายไปแล้ว
    """
    from vacuum.viewer import _TEMPLATE

    legend = re.search(r'var\(--wall\)"></i>(.*?)</span>', _TEMPLATE)
    assert legend, "ไม่พบคำอธิบายสีของกำแพง"
    assert "เดินทับไม่ได้" in legend.group(1), (
        f"คำอธิบายสีของกำแพงบอกแค่ {legend.group(1)!r} — ต้องบอกด้วยว่าเดินทับไม่ได้"
    )
