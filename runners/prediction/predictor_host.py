"""ฝั่ง **untrusted** — โปรแกรมที่รันอยู่ใน sandbox โหลด `predictor.py` ของนิสิตแล้วเสิร์ฟโปรโตคอล

ไฟล์นี้อยู่ใน container ของนิสิต มันจึง**ไม่มีอะไรที่เป็นความลับอยู่ในนั้นเลย**
— ไม่มีเฉลยของชุดที่ใช้ตัดสิน ไม่มีเมล็ดของการแบ่งข้อมูล มีแค่ฟีเจอร์ที่เดินทาง
มาถึงทีละก้อนผ่าน pipe (template §5)

    arena-predictor-host --submission /submission

**cwd ถูกย้ายไปที่โฟลเดอร์ submission ก่อนสร้าง `Predictor`** เพราะ starter kit
สอนให้เขียน `joblib.load("pipeline.pkl")` แบบ path สัมพัทธ์ ซึ่งเป็นสิ่งที่นิสิต
คาดหวังโดยธรรมชาติว่า "ไฟล์ของฉันอยู่ข้างๆ ฉัน" · โฟลเดอร์นั้น mount แบบอ่านอย่างเดียว
ที่เขียนได้มีแค่ `/tmp`
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from runners.prediction.frame import decode_frame, encode_values
from runners.prediction.messages import PREDICT, PREDICTION
from runners.sandbox.host import filtered_traceback, split_protocol_from_stdout
from runners.sandbox.protocol import (
    CLOSE,
    ERROR,
    HELLO,
    PROTOCOL_VERSION,
    READY,
    Channel,
    ProtocolError,
)


def _load_predictor_class(submission_dir: Path):
    """โหลด `class Predictor` จาก `predictor.py` ของนิสิต"""
    predictor_py = submission_dir / "predictor.py"
    if not predictor_py.is_file():
        raise FileNotFoundError(f"ไม่พบ predictor.py ใน {submission_dir}")

    # ใส่ไว้หน้าสุดเพื่อให้ helper module ที่นิสิตเขียนเอง import ได้
    sys.path.insert(0, str(submission_dir))
    spec = importlib.util.spec_from_file_location("student_predictor", predictor_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules["student_predictor"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "Predictor"):
        raise AttributeError("predictor.py ต้องนิยาม `class Predictor`")
    return module.Predictor


def serve(channel: Channel, submission_dir: Path) -> int:
    hello = channel.recv()
    if hello["t"] != HELLO:
        raise ProtocolError(f"ข้อความแรกต้องเป็น {HELLO!r} — ได้ {hello['t']!r}")
    if hello.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"โปรโตคอลไม่ตรงกัน: runner ใช้ {hello.get('protocol')} · host ใช้ {PROTOCOL_VERSION}"
        )

    predictor_config: dict[str, Any] = hello.get("predictor_config") or {}

    try:
        os.chdir(submission_dir)  # ก่อนสร้าง Predictor — `joblib.load("pipeline.pkl")` ต้องเจอไฟล์
        predictor_cls = _load_predictor_class(submission_dir)
        predictor = predictor_cls(dict(predictor_config))
    except BaseException:
        # ล้มตอนสร้าง = submission ใช้ไม่ได้ทั้งอัน ไม่มีอะไรให้ทำต่อ
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
            if kind == PREDICT:
                y = predictor.predict(decode_frame(message["frame"]))
                channel.send(PREDICTION, y=encode_values(y, where="คำทำนาย"))
            else:
                raise ProtocolError(f"ไม่รู้จักข้อความ {kind!r}")
        except BaseException:
            # ทำนายไม่ได้แม้แต่ก้อนเดียว = ให้คะแนนไม่ได้ แต่ปล่อยให้ runner เป็นคน
            # ตัดสิน ไม่ใช่ตายไปเฉยๆ เพราะ runner ต้องได้ traceback ไปแสดงให้นิสิต
            channel.send(ERROR, fatal=False, phase=kind, traceback=filtered_traceback())


def main() -> int:
    channel = split_protocol_from_stdout()  # ต้องมาก่อน import อะไรของนิสิต

    parser = argparse.ArgumentParser(prog="arena-predictor-host")
    parser.add_argument("--submission", default=os.environ.get("ARENA_SUBMISSION", "/submission"))
    args = parser.parse_args()

    try:
        return serve(channel, Path(args.submission).resolve())
    except (ProtocolError, EOFError) as exc:
        print(f"arena-predictor-host: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
