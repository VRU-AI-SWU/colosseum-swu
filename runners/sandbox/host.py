"""ของที่ host ฝั่ง untrusted ทุกชนิดโจทย์ต้องทำเหมือนกัน

    split_protocol_from_stdout()   ย้าย fd 1 ไป stderr แล้วยึด stdout ให้โปรโตคอล
    filtered_traceback()           traceback ที่ตัดความยาวแล้ว

**ไฟล์นี้อยู่ใน container ของนิสิต** — ห้ามมีอะไรที่เป็นความลับ และห้าม import
อะไรนอกจาก stdlib กับ `runners.sandbox.protocol`
"""

from __future__ import annotations

import os
import sys
import traceback

from runners.sandbox.protocol import Channel

#: traceback ยาวกว่านี้ไม่ช่วยให้นิสิตแก้ได้เร็วขึ้น แต่ทำให้หน้าเว็บโหลดช้าลง
MAX_TRACEBACK_CHARS = 8000


def split_protocol_from_stdout() -> Channel:
    """ย้าย fd 1 ไปที่ stderr แล้วเก็บ stdout จริงไว้ให้โปรโตคอล

    **ต้องเรียกก่อน import โค้ดนิสิตทุกกรณี** หลังจากนี้แล้วโค้ดที่เขียนลง fd 1 ตรงๆ
    (`print`, `sys.stdout.write`, `os.write(1, ...)`) จะไปออก stderr ทั้งหมด
    นิสิตจึง `print()` ได้ตามปกติโดยไม่ทำ stream ของโปรโตคอลพัง
    """
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr  # เผื่อกรณีที่มีใครถือ object เดิมไว้แล้ว
    return Channel(
        reader=os.fdopen(0, "rb", buffering=0),
        writer=os.fdopen(protocol_fd, "wb", buffering=0),
    )


def filtered_traceback() -> str:
    text = traceback.format_exc()
    return text[-MAX_TRACEBACK_CHARS:] if len(text) > MAX_TRACEBACK_CHARS else text
