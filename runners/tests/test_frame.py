"""ส่งตารางข้ามเส้นแล้วชนิดข้อมูลต้องไม่เพี้ยน

ความผิดพลาดที่ไฟล์นี้ดักไว้ **ไม่ทำให้อะไรพัง** — มันทำให้คะแนนของนิสิตต่ำลง
โดยที่ทุกอย่างดูเหมือนทำงานได้ปกติ

  · `category` กลายเป็น `object` → `OneHotEncoder` เจอค่าที่ไม่รู้จัก แล้วทิ้งเป็น 0 ทั้งแถว
  · ค่าว่างในคอลัมน์ข้อความกลายเป็น `None` → `SimpleImputer` หามันไม่เจอ
    (sklearn หา missing ด้วย `X != X` ซึ่ง `None` ไม่เข้าเงื่อนไข)
  · อาเรย์ที่เขียนไม่ได้ → transformer ที่เขียนทับ input ในที่จะล้มด้วยข้อความที่อ่านไม่รู้เรื่อง

ทั้งสามข้อเทียบ "ก่อนส่ง" กับ "หลังรับ" ตรงๆ ได้ จึงทดสอบได้โดยไม่ต้องมี sandbox
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runners.prediction.frame import (
    FrameError,
    decode_frame,
    decode_values,
    encode_frame,
    encode_values,
)
from runners.sandbox.protocol import decode, encode


def roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    """ผ่าน msgpack จริงด้วย — ไม่ใช่แค่เรียก encode แล้ว decode ต่อกันในหน่วยความจำ

    ตัวเข้ารหัส ndarray อยู่ในชั้น msgpack การข้ามชั้นนั้นจะทำให้เทสต์ผ่านทั้งที่
    ของจริงส่งไม่ได้ (เช่นอาเรย์ dtype object ที่ `tobytes()` คืนที่อยู่ของ pointer)
    """
    body = encode({"t": "predict", "frame": encode_frame(frame)})
    return decode_frame(decode(body[4:])["frame"])


def sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "i": pd.Series([1, 2, 3, 4], dtype="int64"),
            "f": pd.Series([1.5, np.nan, 3.0, -0.25], dtype="float64"),
            "b": pd.Series([True, False, True, False]),
            "cat": pd.Categorical(["a", "b", None, "a"], categories=["b", "a", "z"]),
            # ค่าว่างเขียนเป็น `np.nan` เพราะนั่นคือรูปที่มันกลับมา (ดู
            # `test_missing_text_comes_back_as_nan_not_none` ว่าทำไมถึงต้องเป็นแบบนั้น)
            "text": pd.Series(["หนึ่ง", np.nan, "สาม", "สี่"], dtype=object),
            "when": pd.to_datetime(["2026-01-01", "2026-06-30", "2026-08-28", "2026-12-31"]),
        }
    )


# ── ตารางทั้งใบ ────────────────────────────────────────────────────


def test_every_supported_dtype_survives():
    before = sample()
    after = roundtrip(before)
    assert list(after.columns) == list(before.columns), "ลำดับคอลัมน์ต้องไม่เปลี่ยน"
    pd.testing.assert_frame_equal(after, before, check_dtype=True)


def test_categories_keep_their_order_and_unused_levels():
    """ลำดับของหมวดมีผลต่อผลของ encoder — และหมวดที่ไม่ปรากฏในก้อนนี้ต้องไม่หายไป

    `pipeline` ถูก fit บน train ที่มีครบทุกหมวด ถ้าก้อนที่ส่งมาตัดหมวดที่ไม่ปรากฏทิ้ง
    จำนวนคอลัมน์หลัง one-hot จะไม่ตรงกับตอน fit
    """
    after = roundtrip(sample())
    assert list(after["cat"].cat.categories) == ["b", "a", "z"]
    assert after["cat"].cat.ordered is False


def test_ordered_categories_stay_ordered():
    frame = pd.DataFrame({"size": pd.Categorical(["s", "l"], categories=["s", "m", "l"],
                                                 ordered=True)})
    after = roundtrip(frame)
    assert after["size"].cat.ordered is True
    assert list(after["size"].cat.categories) == ["s", "m", "l"]


def test_missing_values_survive_in_every_dtype():
    after = roundtrip(sample())
    assert after["f"].isna().tolist() == [False, True, False, False]
    assert after["cat"].isna().tolist() == [False, False, True, False]
    assert after["text"].isna().tolist() == [False, True, False, False]


@pytest.mark.parametrize("missing", [None, np.nan])
def test_missing_text_comes_back_as_nan_not_none(missing):
    """**ข้อที่พังเงียบที่สุด** — sklearn หาค่าว่างในคอลัมน์ object ด้วย `X != X`

    `None != None` เป็น False ค่าว่างที่กลับมาเป็น `None` จึงถูกมองว่าเป็นค่าปกติ
    แล้ว `SimpleImputer` จะไม่เติมอะไรเลย จนไปพังลึกกว่านั้นในที่ที่หาต้นเหตุยาก

    ทั้ง `None` และ `np.nan` ที่ส่งไปต้องกลับมาเป็น `np.nan` เหมือนกัน — pandas
    ที่อ่านจาก CSV ให้ `np.nan` ส่วนที่มาจาก dict ของ Python ให้ `None`
    """
    frame = pd.DataFrame({"text": pd.Series(["ก", missing], dtype=object)})
    value = roundtrip(frame)["text"].to_numpy()[1]
    assert value is not None
    assert isinstance(value, float) and np.isnan(value), f"ได้ {value!r}"


def test_index_is_reset_even_when_the_source_was_sliced():
    """index ของแถวต้องไม่เดินทางไปด้วย — ตัวตรวจ row permutation พึ่งเรื่องนี้

    ถ้า index เดิมติดไปด้วย โค้ดที่ `sort_index()` ก่อนทำนายจะผ่านการตรวจนั้น
    ทั้งที่ผลของมันขึ้นกับลำดับจริงๆ
    """
    sliced = sample().iloc[[3, 1]]
    assert list(sliced.index) == [3, 1]
    after = roundtrip(sliced)
    assert list(after.index) == [0, 1]
    assert after["i"].tolist() == [4, 2], "ค่าต้องยังเรียงตามที่ส่งไป"


def test_decoded_arrays_can_be_written_to():
    """`np.frombuffer` คืนอาเรย์อ่านอย่างเดียว — transformer ที่เขียนทับ input จะล้ม

    ตรวจที่ `decode_values` ตรงๆ ไม่ใช่ผ่าน DataFrame เพราะ pandas คัดลอกบล็อกตอน
    สร้างตารางอยู่แล้ว การตรวจผ่านตารางจึงผ่านแม้ตัว decode จะคืนของที่เขียนไม่ได้
    ส่วนคำทำนายที่เดินทางกลับมาไม่ได้ผ่าน pandas เลย
    """
    payload = encode_values(np.array([1.5, 2.5]))
    got = decode_values(decode(encode({"t": "x", "y": payload})[4:])["y"])
    assert got.flags.writeable, "อาเรย์ที่ decode มาต้องเขียนทับได้"
    got[0] = 9.0  # ต้องไม่โยน ValueError: assignment destination is read-only


def test_empty_frame_still_round_trips():
    frame = sample().iloc[:0]
    after = roundtrip(frame)
    assert len(after) == 0
    assert list(after.columns) == list(frame.columns)


# ── สิ่งที่ต้องปฏิเสธ ตั้งแต่ตอนส่ง ────────────────────────────────


def test_pandas_nullable_dtypes_are_refused_with_a_reason():
    """ปฏิเสธดังๆ ดีกว่าแปลงเงียบๆ แล้วค่าว่างเปลี่ยนรูป"""
    frame = pd.DataFrame({"n": pd.Series([1, None, 3], dtype="Int64")})
    with pytest.raises(FrameError, match="Int64|ยังไม่รองรับ"):
        encode_frame(frame)


def test_object_column_with_non_text_is_refused():
    frame = pd.DataFrame({"weird": pd.Series([{"a": 1}, "ok"], dtype=object)})
    with pytest.raises(FrameError, match="weird"):
        encode_frame(frame)


def test_error_message_names_the_column():
    frame = pd.DataFrame({"ok": [1, 2], "พัง": pd.Series([object(), "x"], dtype=object)})
    with pytest.raises(FrameError, match="พัง"):
        encode_frame(frame)


# ── คำทำนายที่เดินทางกลับ ──────────────────────────────────────────


@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 1, 1, 0]),
        np.array([1.5, -2.25, np.nan]),
        np.array(["yes", "no", "yes"], dtype=object),
        np.array([True, False]),
    ],
)
def test_predictions_survive_the_trip(values):
    body = encode({"t": "prediction", "y": encode_values(values)})
    got = decode_values(decode(body[4:])["y"])
    assert len(got) == len(values)
    for a, b in zip(got, values):
        assert (a == b) or (isinstance(b, float) and np.isnan(b) and np.isnan(a))


def test_string_predictions_are_not_sent_as_raw_bytes():
    """อาเรย์ dtype object ที่ `tobytes()` คือที่อยู่ของ pointer — ส่งไปแล้วอีกฝั่งได้ขยะ"""
    labels = np.array(["สแปม", "ปกติ"], dtype=object)
    payload = encode_values(labels)
    assert payload["k"] == "obj", "คอลัมน์ข้อความต้องไม่ถูกส่งผ่านทาง ndarray"
    assert decode_values(decode(encode({"t": "x", "y": payload})[4:])["y"]).tolist() == list(labels)
