"""Wire protocol ระหว่าง runner (trusted) กับ agent (untrusted)

spec: [template §9.1](../../docs/task-templates/agent-vs-environment-rl.md)

    [uint32 LE ความยาว][payload msgpack]

**ทำไมไม่ใช้ pickle** — `pickle.loads` บนข้อมูลจากฝั่งที่ไม่น่าเชื่อถือ = รันโค้ดที่เขาส่งมา
บนเครื่องที่เก็บเฉลยและ private seeds ไว้ ซึ่งทำลาย trust boundary ทั้งอันในบรรทัดเดียว
msgpack อ่านได้แค่ข้อมูล ไม่มีทางกลายเป็นโค้ด

**ทำไมต้องมี length prefix** — pipe ไม่มีขอบเขตของข้อความ อ่าน 4096 ไบต์แล้วอาจได้
ข้อความครึ่งเดียวหรือสองข้อความครึ่ง การมีความยาวนำหน้าทำให้ประกอบกลับได้ถูกต้องเสมอ
"""

from __future__ import annotations

import selectors
import struct
import time
from dataclasses import dataclass
from typing import Any, BinaryIO

import msgpack
import numpy as np

PROTOCOL_VERSION = 1
_LEN = struct.Struct("<I")
MAX_FRAME = 32 * 1024 * 1024  # กัน length ที่ถูกแก้ให้ใหญ่มหาศาลจนกิน RAM หมด

# ── ชนิดข้อความ ─────────────────────────────────────────────────────
# runner → agent
HELLO = "hello"
RESET = "reset"
ACT = "act"
CLOSE = "close"
# agent → runner
READY = "ready"
OK = "ok"
ACTION = "action"
ERROR = "error"

_ND = "__nd__"


class ProtocolError(RuntimeError):
    """ฝั่งตรงข้ามส่งอะไรที่ไม่ตรงโปรโตคอล — ถือเป็นความล้มเหลวของ submission"""


# ── ndarray codec ──────────────────────────────────────────────────
# ส่งเป็น dtype + shape + ไบต์ดิบ ไม่ใช่ list ของ float เพราะ
#   (1) list ของ float64 กินขนาดหลายเท่าและช้ากว่า
#   (2) การแปลงไปเป็น list แล้วกลับมาทำให้ dtype หาย → observation ที่ agent เห็น
#       จะเป็น float64 ตอนรันจริงแต่เป็น float32 ตอนเทรน ซึ่งเป็นบั๊กที่หาไม่เจอ


def _pack_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {_ND: (str(obj.dtype), list(obj.shape), np.ascontiguousarray(obj).tobytes())}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"ส่งวัตถุชนิด {type(obj).__name__} ผ่านโปรโตคอลไม่ได้")


def _unpack_hook(obj: dict) -> Any:
    spec = obj.get(_ND)
    if spec is None:
        return obj
    dtype, shape, raw = spec
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(tuple(shape))


def encode(message: dict[str, Any]) -> bytes:
    body = msgpack.packb(message, default=_pack_default, use_bin_type=True)
    return _LEN.pack(len(body)) + body


def decode(body: bytes) -> dict[str, Any]:
    return msgpack.unpackb(body, object_hook=_unpack_hook, raw=False, strict_map_key=False)


# ── channel ────────────────────────────────────────────────────────


@dataclass
class Channel:
    """อ่าน/เขียนข้อความบนคู่ของ stream — ใช้ได้ทั้งฝั่ง runner และฝั่ง agent"""

    reader: BinaryIO
    writer: BinaryIO

    def send(self, kind: str, **payload: Any) -> None:
        self.writer.write(encode({"t": kind, **payload}))
        self.writer.flush()

    def recv(self, timeout: float | None = None) -> dict[str, Any]:
        head = self._read_exactly(_LEN.size, timeout)
        (n,) = _LEN.unpack(head)
        if n > MAX_FRAME:
            raise ProtocolError(f"ข้อความยาว {n} ไบต์ เกินเพดาน {MAX_FRAME}")
        message = decode(self._read_exactly(n, timeout))
        if not isinstance(message, dict) or "t" not in message:
            raise ProtocolError(f"ข้อความไม่มีฟิลด์ 't': {message!r}")
        return message

    def _read_exactly(self, n: int, timeout: float | None = None) -> bytes:
        chunks, remaining = [], n
        deadline = None if timeout is None else time.monotonic() + timeout
        while remaining:
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0 or not self._wait_readable(left):
                    raise TimeoutError(
                        f"ฝั่งตรงข้ามไม่ตอบภายใน {timeout:.3f} วินาที "
                        f"(อ่านได้ {n - remaining}/{n} ไบต์)"
                    )
            chunk = self.reader.read(remaining)
            if not chunk:
                raise EOFError(
                    f"ฝั่งตรงข้ามปิดการเชื่อมต่อกลางข้อความ (ต้องการอีก {remaining} ไบต์)"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _wait_readable(self, timeout: float) -> bool:
        with selectors.DefaultSelector() as sel:
            sel.register(self.reader, selectors.EVENT_READ)
            return bool(sel.select(timeout))

    def close(self) -> None:
        for stream in (self.reader, self.writer):
            try:
                stream.close()
            except OSError:
                pass
