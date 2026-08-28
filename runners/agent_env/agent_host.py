"""ฝั่ง **untrusted** — โปรแกรมที่รันอยู่ใน sandbox โหลด `agent.py` ของนิสิตแล้วเสิร์ฟโปรโตคอล

ไฟล์นี้อยู่ใน container ของนิสิต มันจึง**ไม่มีอะไรที่เป็นความลับอยู่ในนั้นเลย**
— ไม่มี seed, ไม่มีผังห้อง, ไม่มีเฉลย ทุกอย่างนั้นอยู่ฝั่ง runner และเดินทางมา
แค่ในรูป observation (รายการไฟล์ที่เข้า container ทั้งหมดอยู่ที่ `images/Dockerfile.cpu`
และถูกตรวจโดย `runners/tests/test_image_contents.py`)

    arena-agent-host --submission /submission

โปรโตคอลวิ่งบน stdin/stdout ของ process นี้ ส่วน stdout ที่โค้ดนิสิตเห็นถูกย้ายไปที่ stderr
ตั้งแต่ก่อนโหลด `agent.py` — นิสิตจึง `print()` ได้ตามปกติโดยไม่ทำ stream พัง
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from runners.agent_env.messages import ACT, ACTION, RESET
from runners.sandbox.host import filtered_traceback, split_protocol_from_stdout
from runners.sandbox.protocol import (
    CLOSE,
    ERROR,
    HELLO,
    OK,
    PROTOCOL_VERSION,
    READY,
    Channel,
    ProtocolError,
)


def _load_agent_class(submission_dir: Path):
    """โหลด `class Agent` จาก `agent.py` ของนิสิต"""
    agent_py = submission_dir / "agent.py"
    if not agent_py.is_file():
        raise FileNotFoundError(f"ไม่พบ agent.py ใน {submission_dir}")

    # ใส่ไว้หน้าสุดเพื่อให้ helper module ที่นิสิตเขียนเอง import ได้
    sys.path.insert(0, str(submission_dir))
    spec = importlib.util.spec_from_file_location("student_agent", agent_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules["student_agent"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "Agent"):
        raise AttributeError("agent.py ต้องนิยาม `class Agent`")
    return module.Agent


def serve(channel: Channel, submission_dir: Path) -> int:
    hello = channel.recv()
    if hello["t"] != HELLO:
        raise ProtocolError(f"ข้อความแรกต้องเป็น {HELLO!r} — ได้ {hello['t']!r}")
    if hello.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"โปรโตคอลไม่ตรงกัน: runner ใช้ {hello.get('protocol')} · host ใช้ {PROTOCOL_VERSION}"
        )

    agent_config: dict[str, Any] = hello.get("agent_config") or {}

    try:
        agent_cls = _load_agent_class(submission_dir)
        agent = agent_cls(dict(agent_config))
    except BaseException:
        # ล้มตอนสร้าง agent = submission ใช้ไม่ได้ทั้งอัน ไม่ใช่แค่ episode เดียว
        channel.send(ERROR, fatal=True, phase="init", traceback=filtered_traceback())
        return 1

    channel.send(READY, protocol=PROTOCOL_VERSION)

    while True:
        try:
            message = channel.recv()
        except EOFError:
            return 0  # runner ปิดไปเฉยๆ ถือว่าจบปกติ

        kind = message["t"]
        if kind == CLOSE:
            return 0

        try:
            if kind == RESET:
                agent.reset(dict(message.get("episode_info") or {}))
                channel.send(OK)
            elif kind == ACT:
                action = agent.act(message["obs"])
                channel.send(ACTION, action=action)
            else:
                raise ProtocolError(f"ไม่รู้จักข้อความ {kind!r}")
        except BaseException:
            # ล้มระหว่าง episode → รายงานแล้วรอคำสั่งถัดไป
            # runner จะเป็นคนตัดสินว่าทิ้ง episode นี้หรือทิ้งทั้ง run (template §7.3)
            channel.send(ERROR, fatal=False, phase=kind, traceback=filtered_traceback())


def main() -> int:
    channel = split_protocol_from_stdout()  # ต้องมาก่อน import อะไรของนิสิต

    parser = argparse.ArgumentParser(prog="arena-agent-host")
    parser.add_argument("--submission", default=os.environ.get("ARENA_SUBMISSION", "/submission"))
    args = parser.parse_args()

    try:
        return serve(channel, Path(args.submission))
    except (ProtocolError, EOFError) as exc:
        print(f"arena-agent-host: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
