"""ส่ง DataFrame ข้ามเส้น trust boundary โดยไม่ให้ชนิดข้อมูลเพี้ยน

**ทำไมต้องมีไฟล์นี้แทนที่จะส่งไฟล์** — ถ้าเขียน CSV/parquet ลงดิสก์แล้ว mount เข้า
container ก็ต้องมีโฟลเดอร์ที่เขียนได้ในกล่อง และมีไฟล์ข้อมูลนอนอยู่บนเครื่อง
การส่งผ่าน pipe ทำให้**ไม่มีอะไรแตะดิสก์เลย** ทั้ง `X` ที่เข้าไปและคำทำนายที่ออกมา

**ทำไมต้องรักษา dtype ให้เป๊ะ** — pipeline ของนิสิตถูก `fit` บน `train.X` ไปแล้ว
ถ้า `X` ตอนตัดสินมาถึงด้วยชนิดที่ต่างออกไป (`category` กลายเป็น `object` ·
`float64` กลายเป็น `object` เพราะมี NaN) `OneHotEncoder` จะเจอค่าที่ไม่เคยเห็น
หรือ `SimpleImputer` จะหา NaN ไม่เจอ ผลคือคะแนนตกโดยที่นิสิตไม่ได้ทำอะไรผิดเลย
และไม่มีใครหาสาเหตุเจอเพราะทุกอย่างดู "ทำงานได้"

**index ถูกรีเซ็ตเป็น 0..n-1 เสมอ — ตั้งใจ ไม่ใช่การมักง่าย**
ตัวตรวจ row permutation สลับลำดับแถวแล้วดูว่าคำทำนายของแต่ละแถวเปลี่ยนไหม
ถ้า index เดิมติดไปด้วย โค้ดที่เรียงตาม index ก่อนทำนายจะผ่านการตรวจนั้นทั้งที่
มันขึ้นกับลำดับจริงๆ · index เดิมยังบอกตำแหน่งของแถวในชุดเต็มด้วย ซึ่งเป็น
ข้อมูลที่ฝั่ง untrusted ไม่ควรได้
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: ชนิดที่ผ่าน numpy ได้ตรงๆ — bool · int · uint · float · datetime64
_NUMPY_KINDS = "biufM"


class FrameError(ValueError):
    """แปลงตารางไม่ได้ — ข้อความต้องบอกชื่อคอลัมน์และชนิดที่เจอเสมอ"""


# ── ค่าหนึ่งชุด (คอลัมน์เดียว หรือคำทำนาย) ─────────────────────────


def encode_values(values: Any, *, where: str = "ค่า") -> dict:
    """อาเรย์ 1 ชุด → dict ที่ msgpack ส่งได้

    `where` เป็นชื่อไว้ใส่ในข้อความผิดพลาด — คอลัมน์ไหนพังต้องรู้ทันทีโดยไม่ต้องเดา
    """
    array = np.asarray(values)
    if array.dtype.kind in _NUMPY_KINDS:
        return {"k": "arr", "v": array}
    if array.dtype.kind in "OUS":
        # object/str ส่งเป็นลิสต์ เพราะ `tobytes()` ของ object dtype คือที่อยู่ของ
        # pointer ไม่ใช่ตัวข้อความ — ส่งไปแล้วอีกฝั่งได้ขยะที่ดูเหมือนข้อมูล
        flat = array.ravel().tolist()
        bad = {type(v).__name__ for v in flat if not _is_sendable(v)}
        if bad:
            raise FrameError(
                f"{where}: ส่งค่าชนิด {sorted(bad)} ผ่านโปรโตคอลไม่ได้\n"
                "  คอลัมน์ข้อความต้องมีแต่ str หรือค่าว่าง"
            )
        return {"k": "obj", "v": [None if _is_missing(v) else str(v) for v in flat],
                "shape": list(array.shape)}
    raise FrameError(f"{where}: ไม่รองรับชนิด {array.dtype!r}")


def decode_values(payload: dict) -> np.ndarray:
    kind = payload.get("k")
    if kind == "arr":
        # `np.frombuffer` คืนอาเรย์ที่เขียนไม่ได้ — transformer หลายตัวของ sklearn
        # เขียนทับ input ในที่ (`copy=False`) แล้วจะได้ ValueError ที่อ่านไม่รู้เรื่อง
        return np.array(payload["v"], copy=True)
    if kind == "obj":
        # ค่าว่างกลับไปเป็น `np.nan` ไม่ใช่ `None` โดยตั้งใจ — sklearn หา missing
        # ในคอลัมน์ object ด้วย `X != X` ซึ่ง `None` ไม่เข้าเงื่อนไข (`None != None`
        # เป็น False) ถ้าคืนเป็น None ตัว imputer จะมองไม่เห็นค่าว่างเลย
        values = [np.nan if v is None else v for v in payload["v"]]
        return np.array(values, dtype=object).reshape(tuple(payload.get("shape") or [len(values)]))
    raise FrameError(f"ไม่รู้จักการเข้ารหัสชนิด {kind!r}")


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _is_sendable(value: Any) -> bool:
    return isinstance(value, str) or _is_missing(value)


# ── ตารางทั้งใบ ────────────────────────────────────────────────────


def encode_frame(frame: pd.DataFrame) -> dict:
    columns = []
    for name in frame.columns:
        columns.append({"name": str(name), **_encode_column(str(name), frame[name])})
    return {"n": int(len(frame)), "columns": columns}


def decode_frame(payload: dict) -> pd.DataFrame:
    data = {c["name"]: _decode_column(c) for c in payload["columns"]}
    return pd.DataFrame(data, index=pd.RangeIndex(payload["n"]))


def _encode_column(name: str, series: pd.Series) -> dict:
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return {
            "k": "cat",
            # `codes` เก็บ -1 แทน NaN อยู่แล้ว จึงไม่ต้องส่ง mask แยก
            "codes": np.asarray(series.cat.codes, dtype="int32"),
            "levels": encode_values(dtype.categories.to_numpy(),
                                    where=f"คอลัมน์ {name!r} (รายการหมวด)"),
            "ordered": bool(dtype.ordered),
        }
    # ⚠️ **ต้องตรวจ ExtensionDtype ก่อนดู `.kind` เสมอ** — `.kind` ของชนิดที่รับค่าว่างได้
    # ของ pandas โกหก: `Int64` รายงาน `'i'` เหมือน int64 ธรรมดา แต่ `to_numpy()` ของมัน
    # แปลงเป็น `float64` เงียบๆ เพื่อให้มีที่เก็บ NaN · คอลัมน์จำนวนเต็มจะเดินทางถึงกล่อง
    # เป็นทศนิยม ซึ่งไม่ทำให้อะไรพัง แค่ทำให้ผลของ pipeline ต่างจากตอน fit
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        raise FrameError(
            f"คอลัมน์ {name!r}: ชนิด {dtype!r} ยังไม่รองรับ\n"
            "  ชนิดที่รับค่าว่างได้ของ pandas (Int64, boolean, string, Float64) แปลงไป-กลับ\n"
            "  แล้วค่าว่างเปลี่ยนรูป และ Int64 กลายเป็น float64 โดยไม่มีอะไรฟ้อง\n"
            "  ใช้ int64/float64/object ธรรมดา หรือ category แทน"
        )
    if dtype.kind in _NUMPY_KINDS or dtype.kind in "OUS":
        return encode_values(series.to_numpy(), where=f"คอลัมน์ {name!r}")
    raise FrameError(
        f"คอลัมน์ {name!r}: ไม่รองรับชนิด {dtype!r}\n"
        "  ที่รองรับคือ int/float/bool/datetime · category · object ที่มีแต่ข้อความ"
    )


def _decode_column(payload: dict):
    if payload["k"] == "cat":
        return pd.Categorical.from_codes(
            np.array(payload["codes"], copy=True),
            categories=pd.Index(decode_values(payload["levels"])),
            ordered=payload["ordered"],
        )
    return decode_values(payload)
