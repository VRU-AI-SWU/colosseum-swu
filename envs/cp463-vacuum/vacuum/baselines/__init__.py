"""Baseline agents — หมุดหมายบน leaderboard (environment-spec §10)

| ระดับ | Agent | ความหมาย | แจกโค้ดไหม |
|---|---|---|---|
| 🥉 Bronze | `RandomAgent` | "โค้ดทำงานได้แล้ว" | ✅ |
| 🥈 Silver | `GreedyAgent` | "agent มีกลยุทธ์แล้ว" — เห็นแค่ในหน้าต่าง | ✅ |
| 🥇 Gold | `BFSCoverageAgent` | "จำแผนที่ได้และวางแผนเป็น" | ❌ |
| 💎 Diamond | `BeliefBFSAgent` | "รู้ว่าเซนเซอร์เชื่อไม่ได้ แล้วทำอะไรกับมัน" | ❌ |

**Gold กับ Diamond ไม่ได้แจกโค้ด** — [README §10.4](../../../../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries)
ระบุไว้ตั้งแต่แรกว่า baseline ระดับ Diamond เป็นของลับเพราะลอกได้ และสองระดับนี้คือ
ก้าวที่โจทย์ตั้งใจให้นิสิตฝึก (Silver→Gold +0.70 · Gold→Diamond ทำให้ดูดครบ 30/30)

ตาราง ladder ใน starter README อธิบายว่าแต่ละระดับ*ทำอะไร*ต่างกัน — เป็นเป้าหมายให้ไล่
ไม่ใช่โค้ดให้ลอก

## ฝั่งผู้สอน

โค้ดของ Gold/Diamond อยู่ที่ `colosseum-hypogeum/agents/cp463-vacuum/`
ตั้ง `ARENA_SECRETS` ให้ชี้ไปที่ clone ของ repo นั้น แล้วมันจะโผล่ใน `BASELINES` เอง

    export ARENA_SECRETS=/srv/arena/secrets
    python -c "from vacuum.baselines import BASELINES; print(sorted(BASELINES))"
    # ['bronze', 'diamond', 'gold', 'silver']   ← ฝั่งผู้สอน
    # ['bronze', 'silver']                       ← สิ่งที่นิสิตเห็น
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from vacuum.baselines.greedy import GreedyAgent
from vacuum.baselines.random_agent import RandomAgent

#: baseline ที่แจกไปกับ wheel — นิสิตอ่านและรันได้
PUBLIC_BASELINES = {
    "bronze": RandomAgent,
    "silver": GreedyAgent,
}

#: ชื่อของ baseline ที่มีโค้ดอยู่ฝั่งผู้สอนเท่านั้น
INSTRUCTOR_LEVELS = ("gold", "diamond")

_AGENT_DIR = Path("agents") / "cp463-vacuum"
_MODULES = {"gold": ("bfs", "BFSCoverageAgent"), "diamond": ("belief_bfs", "BeliefBFSAgent")}


def instructor_agents_path() -> Path | None:
    """หาโฟลเดอร์ agent ฝั่งผู้สอนจาก `ARENA_SECRETS` เท่านั้น

    **ไม่ไล่หาจากโฟลเดอร์ข้างๆ โดยอัตโนมัติ** ถึงแม้จะสะดวกกว่าตอน dev — เพราะมันทำให้
    พฤติกรรมขึ้นกับว่าเครื่องนั้นบังเอิญ clone อะไรไว้ตรงไหน เทสต์ที่ต้องยืนยันว่า
    "นิสิตเห็นแค่ 2 ตัว" จะผ่านบ้างไม่ผ่านบ้างแล้วแต่เครื่อง ซึ่งแย่กว่าการต้องพิมพ์ env var
    """
    root = os.environ.get("ARENA_SECRETS")
    if not root:
        return None
    path = Path(root) / _AGENT_DIR
    return path if path.is_dir() else None


def load_instructor_baselines() -> dict[str, type]:
    """โหลด Gold/Diamond ถ้าเข้าถึงได้ — คืน `{}` เงียบๆ ถ้าไม่มี

    เงียบโดยตั้งใจ: เครื่องของนิสิตจะไม่มีทางเข้าถึงอยู่แล้ว การเตือนทุกครั้งที่ import
    จึงเป็นเสียงรบกวนล้วนๆ ส่วนฝั่งผู้สอนที่ต้องการมันจริงจะรู้เองเมื่อ `BASELINES` ไม่ครบ
    """
    path = instructor_agents_path()
    if path is None:
        return {}

    if str(path) not in sys.path:
        sys.path.insert(0, str(path))  # ให้ bfs.py หา planning.py เจอ

    loaded: dict[str, type] = {}
    for level, (module_name, class_name) in _MODULES.items():
        file = path / f"{module_name}.py"
        if not file.is_file():
            continue
        spec = importlib.util.spec_from_file_location(f"_arena_{module_name}", file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[level] = getattr(module, class_name)
    return loaded


def all_baselines() -> dict[str, type]:
    return {**PUBLIC_BASELINES, **load_instructor_baselines()}


#: baseline ที่ใช้ได้จริงในสภาพแวดล้อมปัจจุบัน — 2 ตัวบนเครื่องนิสิต · 4 ตัวฝั่งผู้สอน
BASELINES = all_baselines()

__all__ = [
    "RandomAgent",
    "GreedyAgent",
    "PUBLIC_BASELINES",
    "INSTRUCTOR_LEVELS",
    "BASELINES",
    "all_baselines",
    "load_instructor_baselines",
    "instructor_agents_path",
]
