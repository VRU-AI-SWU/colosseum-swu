"""TaskSpec — ทุกอย่างที่ต้องประกาศต่อ competition หนึ่งอัน (template §13)

**`config_hash` คือสัญญา** — บันทึกลงทุก run · ถ้าค่านี้เปลี่ยนแปลว่าโจทย์เปลี่ยน
แล้วคะแนนเก่าเทียบกับคะแนนใหม่ไม่ได้ · แพลตฟอร์มตรวจซ้ำตอนให้คะแนนเหมือนที่
`cp463-vacuum` ทำ เพราะการเปลี่ยน config เงียบๆ กลางเทอมคือการเปลี่ยนกติกา
โดยที่ leaderboard ยังดูเหมือนเทียบกันได้

**`dataset` เป็นลายนิ้วมือของเนื้อไฟล์ ไม่ใช่ชื่อไฟล์** จึงอยู่ใน hash ด้วย —
การสลับไฟล์ข้อมูลกลางเทอมเปลี่ยน hash เองโดยอัตโนมัติ ไม่มีทางทำเงียบๆ ได้

---

**ข้อมูลถูกแบ่งสามกอง ไม่ใช่สอง** — และผู้สอนคุมด้วยตัวเลขเดี่ยวสองตัวที่ซ้อนกัน
ไม่ใช่สามตัวที่ต้องรวมกันได้ 1 (ซึ่งกรอกผิดได้และผิดบ่อย)

    ทั้งไฟล์
    ├── student_ratio ────────────► แจกนิสิต — เขาแบ่ง train/val เองตามใจ
    └── ที่เหลือ = ชุดที่ใช้ตัดสิน
        ├── grading_public_ratio ─► leaderboard ระหว่างเทอม
        └── ที่เหลือ ─────────────► ตัดสินรอบสุดท้าย

**ทำไมกองสุดท้ายต้องมี** — ถ้าชุดที่ใช้ตัดสินมีกองเดียว ทีมที่ส่งวันละหลายครั้ง
ตลอดเทอมจะค่อยๆ จูนเข้าหากองนั้นจนคะแนนสูงเกินความสามารถจริง แล้วอันดับสุดท้าย
จะวัด "ความสามารถในการเดา leaderboard" ไม่ใช่ความสามารถของโมเดล (template §1.1)

**นิสิตแบ่ง train/val/test เอง** — ระบบไม่ยุ่ง · หน้าที่ของระบบคือรับ pipeline
กับโมเดลเข้ามาแล้ววัดกับข้อมูลที่นิสิตไม่เคยเห็น · `split_seed` ในนี้คุมการแบ่ง
*สามกองข้างบน* เท่านั้น ไม่เกี่ยวกับการแบ่งฝั่งนิสิตซึ่งเขาเลือกเมล็ดเอง
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

KINDS = ("classification", "regression")

#: คะแนนหลักที่ใช้ได้ต่อชนิดโจทย์ — ทุกตัว "มากกว่าดีกว่า"
PRIMARY_BY_KIND = {
    "classification": ("macro_f1", "accuracy"),
    "regression": ("r2", "neg_rmse", "neg_mae"),
}

#: ขอบเขตของสัดส่วนที่แจกนิสิต — กว้างพอให้เลือกได้ แต่ไม่ถึงขั้นที่กองใดกองหนึ่งว่าง
MIN_STUDENT_RATIO = 0.2
MAX_STUDENT_RATIO = 0.9


class ConfigError(Exception):
    """config ไม่ถูกต้อง — ต้องล้มตั้งแต่โหลด ไม่ใช่ไปพังตอนให้คะแนน"""


@dataclass(frozen=True)
class TaskSpec:
    """โจทย์หนึ่งอัน — อ่านจาก YAML แล้วตรึงไว้ทั้งเทอม"""

    title: str
    kind: str
    primary: str
    #: ลายนิ้วมือของไฟล์ข้อมูลในคลัง (`tabular.store`) — `sha256:...`
    dataset: str
    #: ชื่อคอลัมน์ที่เป็นเฉลย
    target: str
    #: เมล็ดของการแบ่งสามกอง — **ไม่ใช่เมล็ดที่นิสิตใช้แบ่ง train/val ของตัวเอง**
    #:
    #: มีค่าเริ่มต้นเพราะ **ค่าไหนก็ได้ ขอแค่ไม่เปลี่ยนกลางเทอม** — การบังคับให้
    #: ผู้สอนคิดตัวเลขขึ้นมาเองคือการขอให้เขาตัดสินใจเรื่องที่ไม่มีคำตอบที่ดีกว่า
    #: และช่องบังคับที่ไม่มีคำตอบที่ดีกว่าคือช่องที่คนกรอกมั่วแล้วรู้สึกไม่มั่นใจ
    #: · เปลี่ยนได้ถ้าอยากสับข้อมูลใหม่ ซึ่งจะเปลี่ยน `config_hash` ตามไปด้วย
    split_seed: int = 20260101
    bootstrap_seed: int = 20260102
    #: สัดส่วนของทั้งไฟล์ที่แจกนิสิต — ที่เหลือคือชุดที่ใช้ตัดสิน
    student_ratio: float = 0.7
    #: ในชุดที่ใช้ตัดสิน สัดส่วนที่โชว์บน leaderboard ระหว่างเทอม
    #: ที่เหลือซ่อนไว้ตัดสินรอบสุดท้าย · **ไม่ใช่สัดส่วนของทั้งไฟล์**
    grading_public_ratio: float = 0.5
    #: คอลัมน์ที่ตัดทิ้งก่อนถึงมือนิสิต — id ของแถว ชื่อคน วันที่ดึงข้อมูล ฯลฯ
    drop: list[str] = field(default_factory=list)
    #: ลำดับของคลาสที่ตรึงไว้ — classification เท่านั้น · **ต้องไม่เปลี่ยนกลางเทอม**
    #: เพราะ confusion matrix กับ per-class F1 อ้างลำดับนี้
    #: หน้าเว็บเติมให้จากไฟล์ที่อัปโหลด ผู้สอนไม่ต้องพิมพ์เอง
    labels: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ConfigError(f"kind ต้องเป็น {KINDS} — ได้ {self.kind!r}")
        allowed = PRIMARY_BY_KIND[self.kind]
        if self.primary not in allowed:
            raise ConfigError(
                f"{self.kind} ใช้คะแนนหลัก {self.primary!r} ไม่ได้ — ที่ใช้ได้คือ {allowed}\n"
                "  ทุกตัวเป็นแบบ 'มากกว่าดีกว่า' ตามที่ leaderboard ต้องการ"
            )
        if not self.target:
            raise ConfigError("ต้องบอกว่าคอลัมน์ไหนเป็นเฉลย (`target`)")
        if self.target in self.drop:
            raise ConfigError(
                f"คอลัมน์เฉลย {self.target!r} อยู่ในรายการที่ตัดทิ้งด้วย — เลือกอย่างใดอย่างหนึ่ง"
            )
        if self.kind == "classification" and not self.labels:
            raise ConfigError("classification ต้องประกาศ `labels` เพื่อตรึงลำดับของคลาส")
        if self.kind == "regression" and self.labels:
            raise ConfigError("regression ต้องไม่มี `labels`")
        if not MIN_STUDENT_RATIO <= self.student_ratio <= MAX_STUDENT_RATIO:
            raise ConfigError(
                f"สัดส่วนที่แจกนิสิตต้องอยู่ระหว่าง {MIN_STUDENT_RATIO} กับ "
                f"{MAX_STUDENT_RATIO} — ได้ {self.student_ratio}"
            )
        if not 0.0 < self.grading_public_ratio < 1.0:
            raise ConfigError(
                f"grading_public_ratio ต้องอยู่ระหว่าง 0 กับ 1 — ได้ {self.grading_public_ratio}"
            )
        # `dataset` ตรวจแค่รูปแบบตรงนี้ — การมีอยู่จริงของไฟล์ตรวจที่ `store`
        # เพราะโมดูลนี้ต้องโหลดได้บนเครื่องที่ไม่มีคลัง (เช่นตอนคำนวณ hash)
        from tabular.store import DIGEST_RE

        if not DIGEST_RE.match(self.dataset):
            raise ConfigError(
                f"`dataset` ต้องเป็นลายนิ้วมือของไฟล์ในคลัง (sha256:<64 หลัก>) — ได้ {self.dataset!r}"
            )

    @property
    def grading_ratio(self) -> float:
        """สัดส่วนของทั้งไฟล์ที่เป็นชุดที่ใช้ตัดสิน"""
        return 1.0 - self.student_ratio

    def normalized(self) -> dict:
        """รูปแบบมาตรฐานสำหรับคำนวณ hash — เรียงคีย์และแปลง tuple เป็น list"""
        data = asdict(self)
        data["labels"] = [str(label) for label in self.labels]
        data["drop"] = sorted(str(name) for name in self.drop)
        # `title` เป็นข้อความให้คนอ่าน ไม่กระทบการให้คะแนน — แก้ได้โดยไม่ทำให้
        # คะแนนเก่าเทียบไม่ได้ จึงไม่นับเข้า hash
        data.pop("title")
        return data

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def replace(self, **overrides: Any) -> "TaskSpec":
        """สร้าง spec ใหม่โดยแทนค่าบางตัว — ใช้กับ `Phase.config_override`

        รับคีย์แบบตรงๆ ไม่ใช่ dotted เพราะ spec เป็นชั้นเดียว ไม่มี section ซ้อน
        """
        data = asdict(self)
        for key, value in overrides.items():
            if key not in data:
                raise ConfigError(f"ไม่รู้จักฟิลด์ {key!r} ใน TaskSpec")
            data[key] = value
        return TaskSpec(**data)


def load_config(path: str | Path) -> TaskSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return from_mapping(raw, source=str(path))


def from_mapping(raw: Any, *, source: str = "config") -> TaskSpec:
    """สร้าง spec จาก mapping — ใช้กับทั้งไฟล์ YAML และ config ที่เก็บในฐานข้อมูล"""
    if not isinstance(raw, dict):
        raise ConfigError(f"{source} ต้องเป็น mapping ที่ระดับบนสุด")
    unknown = sorted(set(raw) - set(TaskSpec.__dataclass_fields__))
    if unknown:
        raise ConfigError(
            f"{source}: ไม่รู้จักฟิลด์ {unknown}\n"
            "  ถ้ามาจากโจทย์ที่สร้างก่อนเดือน ส.ค. 2026 โครงของ config เปลี่ยนไปแล้ว —\n"
            "  นิสิตแบ่ง train/val/test เอง ระบบจึงไม่มี `ratios` `n_rows` `data_seed` อีก"
        )
    try:
        return TaskSpec(**raw)
    except TypeError as exc:
        raise ConfigError(f"{source}: {exc}") from exc
