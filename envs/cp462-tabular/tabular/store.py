"""คลังชุดข้อมูล — ไฟล์ CSV ที่ผู้สอนอัปโหลด อ้างถึงด้วยลายนิ้วมือของเนื้อไฟล์

    $ARENA_DATASETS/<sha256>.csv

**นี่คือสิ่งที่มาแทนเมล็ดลับ และมันแข็งแรงกว่า** — เดิมความลับของชุดที่ใช้ตัดสิน
ตั้งอยู่บนสมมติฐานว่า `grading_seed` ไม่รั่ว ถ้ารั่วเมื่อไรนิสิตสร้างเฉลยได้ครบ
ทุกแถวบนเครื่องตัวเอง · ตอนนี้สิ่งที่กันไว้คือ **แถวข้อมูล** ไม่ใช่ตัวเลขที่ใช้
สร้างแถว · ไฟล์เต็มอยู่บนเซิร์ฟเวอร์ที่เดียว และ API เสิร์ฟเฉพาะส่วนที่แจก
ไม่มีตัวเลขไหนที่รู้แล้วย้อนกลับไปได้เนื้อข้อมูลที่ไม่เคยถูกส่งออกไป

**ทำไมอ้างด้วย digest ไม่ใช่ชื่อไฟล์** — `digest` เข้าไปอยู่ใน `config_hash`
ด้วย การสลับไฟล์ใต้ชื่อเดิมกลางเทอมจึงเปลี่ยน hash โดยอัตโนมัติ แล้วแพลตฟอร์ม
ปฏิเสธการเทียบคะแนนเก่ากับใหม่เอง (`ConfigDrift`) · ถ้าอ้างด้วยชื่อ การเปลี่ยน
ข้อมูลใต้ชื่อเดิมจะเงียบสนิท และ leaderboard จะยังดูเหมือนเทียบกันได้ทั้งที่ไม่ใช่

⚠️ **ห้าม import โมดูลนี้จากอะไรที่รันในกล่อง** — เหตุผลเดียวกับ `secrets.py` เดิม
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

#: ที่อยู่ของคลัง — worker กับ API ตั้งค่านี้ · เครื่องนิสิตไม่มี
DATASETS_ENV = "ARENA_DATASETS"

#: ขนาดสูงสุดของไฟล์ที่รับ — กันไฟล์หลุดขนาดที่ทำให้ทั้งเครื่องช้า
#: ตารางเรียนของวิชานี้อยู่ระดับหมื่นแถว ซึ่งไม่ถึงหนึ่งในสิบของเพดานนี้
MAX_BYTES = 64 * 1024 * 1024

#: จำนวนแถวขั้นต่ำที่ยอมรับ — น้อยกว่านี้ช่วงความเชื่อมั่นจะกว้างจนอันดับไม่มีความหมาย
MIN_ROWS = 200

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class DatasetError(Exception):
    """ชุดข้อมูลใช้ไม่ได้ — ต้องล้มตั้งแต่ตอนอัปโหลดหรือตอนโหลด ไม่ใช่ตอนให้คะแนน"""


@dataclass(frozen=True)
class Column:
    """สิ่งที่รู้ได้จากคอลัมน์หนึ่งโดยยังไม่ต้องเลือกว่าอันไหนเป็นเฉลย"""

    name: str
    dtype: str          # numeric | categorical
    missing: int
    unique: int
    #: ค่าที่พบ เรียงตามจำนวน — มีเฉพาะคอลัมน์ที่ค่าไม่เยอะ (เป็นเฉลยของ
    #: classification ได้) · `None` แปลว่าค่าเยอะเกินกว่าจะเป็นคลาส
    values: list[Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "dtype": self.dtype,
            "missing": self.missing,
            "unique": self.unique,
        }
        if self.values is not None:
            out["values"] = self.values
        return out


@dataclass(frozen=True)
class Profile:
    """สรุปไฟล์หนึ่งไฟล์ — สิ่งที่หน้าเว็บต้องใช้สร้างฟอร์มหลังอัปโหลด

    ผู้สอนไม่ควรต้องพิมพ์ชื่อคอลัมน์เฉลยจากความจำ หรือพิมพ์รายการคลาสเอง —
    ทั้งสองอย่างอยู่ในไฟล์แล้ว · ฟอร์มที่ให้พิมพ์เองคือฟอร์มที่พิมพ์ผิดได้
    """

    digest: str
    rows: int
    columns: list[Column]

    def column(self, name: str) -> Column:
        for column in self.columns:
            if column.name == name:
                return column
        raise DatasetError(
            f"ไม่มีคอลัมน์ {name!r} ในชุดข้อมูล — ที่มีคือ {[c.name for c in self.columns]}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "rows": self.rows,
            "columns": [c.as_dict() for c in self.columns],
        }


#: คอลัมน์ที่มีค่าไม่ซ้ำมากกว่านี้ ไม่นับว่าเป็นคลาสของ classification ได้
#: (ค่ามาก = มันเป็น id หรือเป็นตัวเลขต่อเนื่อง ไม่ใช่ป้ายกำกับ)
MAX_CLASSES = 50


def root() -> Path:
    raw = os.environ.get(DATASETS_ENV, "").strip()
    if not raw:
        raise DatasetError(
            f"ยังไม่ได้ตั้ง {DATASETS_ENV} — ชี้ไปที่โฟลเดอร์คลังชุดข้อมูลบนเครื่องเซิร์ฟเวอร์\n"
            "  (ถ้าคุณเป็นนิสิต: คลังนี้ไม่ได้อยู่ในแพ็กเกจโดยตั้งใจ — ชุดที่ใช้ตัดสิน\n"
            "   ต้องเป็นข้อมูลที่คุณไม่เคยเห็น ไม่งั้นคะแนนไม่มีความหมาย)"
        )
    return Path(raw)


def path_of(digest: str) -> Path:
    """ที่อยู่ของไฟล์ — ตรวจรูปแบบ digest ก่อนเสมอ

    `digest` เดินทางมาจาก config ซึ่งแก้ได้ผ่านหน้าเว็บ · ถ้าไม่ตรวจ ค่าที่มี
    `../` จะพาไปอ่านไฟล์นอกคลังได้
    """
    if not DIGEST_RE.match(digest):
        raise DatasetError(f"รหัสชุดข้อมูลผิดรูปแบบ — ต้องเป็น 'sha256:<64 หลัก>' ได้ {digest!r}")
    return root() / f"{digest.split(':', 1)[1]}.csv"


def digest_of(blob: bytes) -> str:
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def put(blob: bytes) -> str:
    """เก็บไฟล์ลงคลัง แล้วคืน digest — อัปโหลดไฟล์เดิมซ้ำได้ ไม่เกิดของซ้ำ"""
    if len(blob) > MAX_BYTES:
        raise DatasetError(
            f"ไฟล์ใหญ่ {len(blob) / 1e6:.1f} MB เกินเพดาน {MAX_BYTES / 1e6:.0f} MB"
        )
    digest = digest_of(blob)
    path = path_of(digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # เขียนลงไฟล์ชั่วคราวก่อนแล้วค่อย rename — ถ้าเขียนตรงๆ แล้วกระบวนการตาย
        # กลางทาง คลังจะมีไฟล์ครึ่งใบที่ digest ไม่ตรงกับเนื้อ และไม่มีใครรู้
        tmp = path.with_suffix(".partial")
        tmp.write_bytes(blob)
        tmp.rename(path)
    return digest


def read(digest: str) -> pd.DataFrame:
    """🔒 ไฟล์เต็มพร้อมเฉลยทุกแถว — ฝั่ง trusted เท่านั้น"""
    path = path_of(digest)
    if not path.is_file():
        raise DatasetError(
            f"ไม่พบชุดข้อมูล {digest} ในคลัง ({path})\n"
            "  ไฟล์อาจถูกลบ หรือ worker ตัวนี้ชี้ ARENA_DATASETS ไปคนละที่กับ API"
        )
    return _parse(path.read_bytes(), source=str(path))


def _parse(blob: bytes, *, source: str) -> pd.DataFrame:
    """อ่าน CSV แล้วบังคับกติกาขั้นต่ำ — ล้มพร้อมบอกว่าต้องแก้อะไร"""
    from io import BytesIO

    try:
        frame = pd.read_csv(BytesIO(blob))
    except Exception as exc:  # pandas โยนได้หลายชนิดมาก
        raise DatasetError(f"{source}: อ่านเป็น CSV ไม่ได้ — {exc}") from exc

    if frame.empty:
        raise DatasetError(f"{source}: ไฟล์ว่าง")

    # **ต้องดูจากหัวไฟล์ดิบ ไม่ใช่จาก `frame.columns`** — pandas เปลี่ยนชื่อคอลัมน์
    # ที่ซ้ำกันให้เงียบๆ (`a`, `a.1`) ผู้สอนจึงได้โจทย์ที่มีคอลัมน์ชื่อประหลาด
    # โดยไม่มีใครบอก แล้วไปงงตอนเลือกคอลัมน์เฉลย
    header = blob.split(b"\n", 1)[0].decode("utf-8", "replace").rstrip("\r")
    names = [name.strip().strip('"') for name in header.split(",")]
    seen, duplicated = set(), []
    for name in names:
        (duplicated.append(name) if name in seen else seen.add(name))
    if duplicated:
        raise DatasetError(
            f"{source}: มีชื่อคอลัมน์ซ้ำ {sorted(set(duplicated))} — ต้องไม่ซ้ำกัน"
        )
    unnamed = [c for c in frame.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        raise DatasetError(
            f"{source}: มีคอลัมน์ที่ไม่มีชื่อ {unnamed}\n"
            "  มักเกิดจากการ export ที่ติด index มาด้วย — ลอง `to_csv(index=False)`"
        )
    return frame


def inspect(blob: bytes, *, source: str = "ไฟล์ที่อัปโหลด") -> tuple[pd.DataFrame, Profile]:
    """อ่านไฟล์ที่เพิ่งอัปโหลดแล้วสรุปให้ — **ยังไม่เก็บลงคลัง**

    คืน DataFrame มาด้วยเพื่อให้ผู้เรียกตรวจต่อได้โดยไม่ต้อง parse ซ้ำ
    """
    frame = _parse(blob, source=source)
    if len(frame) < MIN_ROWS:
        raise DatasetError(
            f"{source}: มี {len(frame)} แถว น้อยกว่าขั้นต่ำ {MIN_ROWS}\n"
            "  ชุดที่ใช้ตัดสินจะเหลือไม่กี่สิบแถว แล้วช่วงความเชื่อมั่นจะกว้าง\n"
            "  จนอันดับบนกระดานสลับกันได้ด้วยความบังเอิญล้วนๆ"
        )
    return frame, Profile(
        digest=digest_of(blob),
        rows=len(frame),
        columns=[_column(frame[name]) for name in frame.columns],
    )


def profile(digest: str) -> Profile:
    """สรุปไฟล์ที่อยู่ในคลังแล้ว"""
    frame = read(digest)
    return Profile(
        digest=digest,
        rows=len(frame),
        columns=[_column(frame[name]) for name in frame.columns],
    )


def _column(series: pd.Series) -> Column:
    unique = int(series.nunique(dropna=True))
    numeric = bool(pd.api.types.is_numeric_dtype(series))
    values: list[Any] | None = None
    if unique <= MAX_CLASSES:
        # เรียงตามค่า ไม่ใช่ตามความถี่ — ลำดับของคลาสต้องไม่ขึ้นกับข้อมูลที่บังเอิญ
        # เจอ เพราะมันตรึงลำดับแกนของ confusion matrix ไว้ทั้งเทอม
        values = [_plain(v) for v in sorted(series.dropna().unique().tolist())]
    return Column(
        name=str(series.name),
        dtype="numeric" if numeric else "categorical",
        missing=int(series.isna().sum()),
        unique=unique,
        values=values,
    )


def _plain(value: Any) -> Any:
    """แปลงค่าของ numpy/pandas ให้เป็นชนิดพื้นฐานที่ JSON และ YAML เขียนได้

    ถ้าไม่แปลง `labels` ที่เก็บลง config จะกลายเป็น `!!python/object/apply:numpy...`
    ใน YAML แล้ว `yaml.safe_load` อ่านกลับไม่ได้ — พังตอนโหลด ไม่ใช่ตอนเขียน
    """
    item = getattr(value, "item", None)
    return item() if callable(item) else value
