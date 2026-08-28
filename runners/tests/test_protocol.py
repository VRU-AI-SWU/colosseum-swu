"""เทสต์ของ wire protocol — โดยเฉพาะ **การรอบน pipe จริง**

ไฟล์นี้เกิดขึ้นเพราะบั๊กที่หลุดไปถึงเครื่องนิสิต: `Channel._wait_readable` เคยใช้
`selectors.DefaultSelector()` ซึ่งบน Windows คือ `select()` ที่รับได้เฉพาะ socket
`arena eval` จึงพังด้วย `OSError [WinError 10038]` ตั้งแต่ข้อความแรก

บทเรียนคือ **ต้องเทสต์บน pipe จริง ไม่ใช่ BytesIO** — `BytesIO` ไม่มี `fileno()`
จึงข้ามโค้ดส่วนที่พังไปทั้งหมด เทสต์ในไฟล์นี้ใช้ `os.pipe()` ทุกอัน จึงรันแล้วได้ผล
ต่างกันจริงระหว่างแพลตฟอร์ม และจะจับบั๊กแบบเดิมซ้ำได้
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from runners.agent_env.messages import ACT
from runners.sandbox.protocol import (
    _POLL_MAX,
    Channel,
    ProtocolError,
    _poll_until_readable,
    encode,
)

TICK = 0.05  # timeout สั้นๆ ที่ยังยาวพอไม่ให้เครื่องช้าทำเทสต์แกว่ง


@pytest.fixture
def pipe_channel():
    """Channel ที่อ่านจาก pipe จริง คู่กับ fd สำหรับเขียนใส่จากฝั่งเทสต์

    `buffering=0` ทั้งสองฝั่ง — เป็นข้อกำหนดของโปรโตคอล ไม่ใช่รายละเอียดของเทสต์
    ถ้ามีบัฟเฟอร์ของ Python คั่น การรอจะมองไม่เห็นไบต์ที่มาถึงแล้ว
    """
    read_fd, write_fd = os.pipe()
    channel = Channel(
        reader=os.fdopen(read_fd, "rb", buffering=0),
        writer=os.fdopen(os.dup(write_fd), "wb", buffering=0),
    )
    peer = os.fdopen(write_fd, "wb", buffering=0)
    try:
        yield channel, peer
    finally:
        for stream in (peer, channel.reader, channel.writer):
            try:
                stream.close()
            except OSError:
                pass


def test_wait_readable_works_on_a_pipe(pipe_channel):
    """เทสต์ที่จับบั๊ก WinError 10038 — บน Windows อันนี้จะแดงก่อนเพื่อน

    ไม่สนใจว่าคืน True หรือ False สนใจแค่ว่า **ไม่โยน OSError**
    """
    channel, _ = pipe_channel
    assert channel._wait_readable(0.0) is False  # ยังไม่มีใครเขียน


def test_wait_readable_sees_data_that_already_arrived(pipe_channel):
    channel, peer = pipe_channel
    peer.write(b"xyz")
    assert channel._wait_readable(TICK) is True


def test_recv_times_out_when_peer_stays_silent(pipe_channel):
    """agent ที่ค้าง ต้องกลายเป็น TimeoutError ไม่ใช่การรอตลอดกาล"""
    channel, _ = pipe_channel
    started = time.monotonic()
    with pytest.raises(TimeoutError) as exc:
        channel.recv(timeout=TICK)
    assert time.monotonic() - started >= TICK * 0.5
    assert "0/4" in str(exc.value)  # ยังไม่ได้ header สักไบต์


def test_recv_times_out_mid_frame_and_says_how_far_it_got(pipe_channel):
    """ครึ่งข้อความแล้วเงียบ — ข้อความผิดพลาดต้องบอกว่าค้างตรงไหน"""
    channel, peer = pipe_channel
    frame = encode({"t": ACT, "obs": [1, 2, 3]})
    peer.write(frame[: len(frame) // 2])
    with pytest.raises(TimeoutError) as exc:
        channel.recv(timeout=TICK)
    assert "ไม่ตอบภายใน" in str(exc.value)


def test_recv_returns_message_that_arrives_while_waiting(pipe_channel):
    """ไบต์ที่มาถึง *ระหว่าง* รอ ต้องถูกเห็น ไม่ใช่แค่ที่มาก่อนเรียก"""
    channel, peer = pipe_channel

    def write_later():
        time.sleep(TICK / 4)
        peer.write(encode({"t": ACT, "obs": 7}))

    thread = threading.Thread(target=write_later)
    thread.start()
    try:
        assert channel.recv(timeout=TICK * 20) == {"t": ACT, "obs": 7}
    finally:
        thread.join()


def test_recv_reports_eof_when_peer_closes(pipe_channel):
    """ปลายทางตายกลางคัน = EOFError ไม่ใช่ TimeoutError

    ต้องแยกกันให้ออก เพราะสองอันนี้บอกคนละเรื่องกับนิสิต — ตายกับค้าง
    """
    channel, peer = pipe_channel
    channel.writer.close()  # ปิดสำเนาฝั่งเราด้วย ไม่งั้น pipe ยังไม่ EOF
    peer.close()
    with pytest.raises(EOFError):
        channel.recv(timeout=TICK)


def test_frame_longer_than_cap_is_rejected_before_allocating(pipe_channel):
    """ความยาวที่ถูกแก้ให้ใหญ่มหาศาลต้องถูกปัดตกทันที ไม่ใช่ไปจอง RAM ตาม"""
    channel, peer = pipe_channel
    peer.write((2**31).to_bytes(4, "little"))
    with pytest.raises(ProtocolError, match="เกินเพดาน"):
        channel.recv(timeout=TICK)


def test_ndarray_survives_the_round_trip_with_dtype_intact(pipe_channel):
    """dtype ที่เพี้ยนคือบั๊กที่หาไม่เจอ — โมดูลนี้ระบุไว้เองว่าเป็นเหตุผลของ codec"""
    channel, peer = pipe_channel
    obs = np.arange(6, dtype=np.float32).reshape(2, 3)
    peer.write(encode({"t": ACT, "obs": obs}))
    got = channel.recv(timeout=TICK)["obs"]
    assert got.dtype == obs.dtype
    assert got.shape == obs.shape
    assert np.array_equal(got, obs)


# ── ตรรกะการวนถาม (เส้นทางของ Windows) ─────────────────────────────
# `_poll_until_readable` ทำงานจริงเฉพาะบน Windows แต่เทสต์ชุดนี้รันได้ทุกที่
# เพราะฟังก์ชันรับ `peek`/`clock`/`sleep` เข้ามา — ตั้งใจออกแบบให้เป็นแบบนั้น
# เพื่อไม่ให้มีโค้ดที่ไม่มีใครรันจนกว่าจะพังใส่นิสิตอีก


class FakeClock:
    """นาฬิกาที่เดินเฉพาะตอนถูกสั่งให้นอน — เทสต์จึงไม่ต้องรอเวลาจริง"""

    def __init__(self) -> None:
        self.now = 0.0
        self.naps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.naps.append(seconds)
        self.now += seconds


def test_poll_returns_at_once_when_data_is_already_there():
    clock = FakeClock()
    assert _poll_until_readable(lambda: True, 10.0, clock=lambda: clock.now, sleep=clock.sleep)
    assert clock.naps == []  # ต้องไม่หน่วงเลยแม้แต่รอบเดียว


def test_poll_gives_up_at_the_deadline():
    clock = FakeClock()
    assert not _poll_until_readable(
        lambda: False, 0.05, clock=lambda: clock.now, sleep=clock.sleep
    )
    assert clock.now <= 0.05 + 1e-9  # ห้ามนอนเลยเส้นตาย


def test_poll_wakes_up_once_data_arrives():
    clock = FakeClock()
    calls = iter([False, False, False, True])
    assert _poll_until_readable(
        lambda: next(calls), 10.0, clock=lambda: clock.now, sleep=clock.sleep
    )
    assert len(clock.naps) == 3


def test_poll_backs_off_but_never_past_the_cap():
    """ถ้าไม่มีเพดาน การรอนานๆ จะกลายเป็นหน่วงทีละหลายวินาที"""
    clock = FakeClock()
    _poll_until_readable(lambda: False, 1.0, clock=lambda: clock.now, sleep=clock.sleep)
    # ตัดรอบสุดท้ายออก เพราะมันถูกย่อให้พอดีเส้นตายโดยตั้งใจ จึงสั้นกว่ารอบก่อนได้
    body = clock.naps[:-1]
    assert body == sorted(body)  # ถ่างขึ้นเรื่อยๆ
    assert max(clock.naps) <= _POLL_MAX
    assert len(clock.naps) < 500  # ไม่ใช่การ spin รัวๆ


def test_poll_with_zero_timeout_asks_once_and_does_not_sleep():
    """`_wait_readable(0.0)` ต้องเป็นการถามครั้งเดียว ไม่ใช่การรอ"""
    clock = FakeClock()
    asked = []
    assert not _poll_until_readable(
        lambda: asked.append(1) or False, 0.0, clock=lambda: clock.now, sleep=clock.sleep
    )
    assert len(asked) == 1
    assert clock.naps == []


def test_poll_treats_a_broken_pipe_as_readable():
    """เลียนแบบ `peek` ของ Windows ตอน pipe ถูกปิด — ต้องตอบว่าอ่านได้

    เพื่อให้ `read()` เป็นคนรายงาน EOF ข้อความผิดพลาดจะได้เหมือนกันทุกแพลตฟอร์ม
    ถ้าตอบว่าอ่านไม่ได้ นิสิตจะเห็น "agent ไม่ตอบ" ทั้งที่ agent ตายไปแล้ว
    """

    def peek_on_closed_pipe() -> bool:
        try:
            raise OSError(109, "The pipe has been ended")
        except OSError:
            return True

    clock = FakeClock()
    assert _poll_until_readable(
        peek_on_closed_pipe, 10.0, clock=lambda: clock.now, sleep=clock.sleep
    )
    assert clock.naps == []
