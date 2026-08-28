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
import sys
import time
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable

import msgpack
import numpy as np

PROTOCOL_VERSION = 1
_LEN = struct.Struct("<I")
MAX_FRAME = 32 * 1024 * 1024  # กัน length ที่ถูกแก้ให้ใหญ่มหาศาลจนกิน RAM หมด

# ── ชนิดข้อความที่ทุกโจทย์ใช้ร่วมกัน ────────────────────────────────
# มีแค่การจับมือ การปิด และการรายงานความผิดพลาด — **ชื่อข้อความของงานจริง
# อยู่ในแพ็กเกจของโจทย์** (`agent_env/messages.py` · `prediction/messages.py`)
# เพราะไฟล์นี้ไม่ควรรู้ว่ามี episode หรือมีตารางข้อมูล
# runner → sandbox
HELLO = "hello"
CLOSE = "close"
# sandbox → runner
READY = "ready"
OK = "ok"          # ตอบรับคำสั่งที่ไม่มีผลลัพธ์ให้ส่งกลับ
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


# ── รอให้มีข้อมูลบน pipe ───────────────────────────────────────────
# ต้องแยกตามระบบปฏิบัติการ เพราะ **`select()` บน Windows รับได้เฉพาะ socket**
# การส่ง pipe เข้าไปได้ `OSError [WinError 10038] ... not a socket` ทันที
# (เจอตอนนิสิตรัน `arena eval` บน Windows — บน POSIX ไม่มีทางเจอเพราะ
# epoll/kqueue รับ pipe ได้ปกติ) ฝั่ง Windows จึงถาม pipe ตรงๆ แล้ววนถาม
#
# ทั้งสองทางมองเห็นแค่บัฟเฟอร์ของ **ระบบปฏิบัติการ** ไม่เห็นบัฟเฟอร์ของ Python
# `reader` จึงต้องเป็น stream ที่ไม่มีบัฟเฟอร์ (`bufsize=0` / `buffering=0`)
# ไม่งั้นไบต์ที่ Python ดูดไว้แล้วจะถูกรายงานว่า "ยังไม่มาถึง" แล้วกลายเป็น
# timeout ที่อธิบายไม่ได้ · ข้อกำหนดนี้มีมาแต่เดิม เพราะ selectors ก็มองไม่เห็นเหมือนกัน

#: เริ่มถามถี่ๆ เพื่อไม่ให้ agent ที่ตอบเร็วต้องเสียเวลารอรอบถัดไป
#: แล้วค่อยถ่างออกเพื่อไม่ให้เผา CPU ตอนรอนาน
_POLL_FIRST = 0.0005
_POLL_MAX = 0.005


def _poll_until_readable(
    peek: Callable[[], bool],
    timeout: float,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """วนถาม `peek()` จนกว่าจะมีข้อมูลหรือหมดเวลา

    แยกเป็นฟังก์ชันล้วนๆ ที่รับ `peek`/`clock`/`sleep` เข้ามา เพื่อให้ **เทสต์ได้
    บนทุกแพลตฟอร์ม** ตรรกะนี้เขียนขึ้นเพื่อ Windows แต่ถ้าเทสต์ได้เฉพาะบน Windows
    มันก็จะเป็นโค้ดที่ไม่มีใครรันจนกว่าจะพังใส่นิสิต — ซึ่งเพิ่งเกิดไปแล้วรอบหนึ่ง
    """
    deadline = clock() + timeout
    nap = _POLL_FIRST
    while True:
        if peek():
            return True
        left = deadline - clock()
        if left <= 0:
            return False
        sleep(min(nap, left))
        nap = min(nap * 2, _POLL_MAX)


if sys.platform == "win32":  # pragma: no cover — เลือกเส้นทางตอน import

    def _wait_readable(stream: BinaryIO, timeout: float) -> bool:
        import _winapi
        import msvcrt

        handle = msvcrt.get_osfhandle(stream.fileno())

        def peek() -> bool:
            try:
                return _winapi.PeekNamedPipe(handle)[0] > 0
            except OSError:
                # ปลายทางปิด pipe ไปแล้ว — ตอบว่าอ่านได้ เพื่อให้ `read()` เป็นคน
                # รายงาน EOF ข้อความผิดพลาดจะได้เหมือนกันทุกแพลตฟอร์ม
                return True

        return _poll_until_readable(peek, timeout)

else:

    def _wait_readable(stream: BinaryIO, timeout: float) -> bool:
        with selectors.DefaultSelector() as sel:
            sel.register(stream, selectors.EVENT_READ)
            return bool(sel.select(timeout))


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
        return _wait_readable(self.reader, timeout)

    def close(self) -> None:
        for stream in (self.reader, self.writer):
            try:
                stream.close()
            except OSError:
                pass
