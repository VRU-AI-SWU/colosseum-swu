"""TaskSpec — ทุกอย่างที่ต้องประกาศต่อ competition หนึ่งอัน (template §13)

**`config_hash` คือสัญญา** — บันทึกลงทุก run · ถ้าค่านี้เปลี่ยนแปลว่าโจทย์เปลี่ยน
แล้วคะแนนเก่าเทียบกับคะแนนใหม่ไม่ได้ · แพลตฟอร์มตรวจซ้ำตอนให้คะแนนเหมือนที่
`cp463-vacuum` ทำ เพราะการเปลี่ยน config เงียบๆ กลางเทอมคือการเปลี่ยนกติกา
โดยที่ leaderboard ยังดูเหมือนเทียบกันได้

**`dataset` เป็นลายนิ้วมือของเนื้อไฟล์ ไม่ใช่ชื่อไฟล์** จึงอยู่ใน hash ด้วย —
การสลับไฟล์ข้อมูลกลางเทอมเปลี่ยน hash เองโดยอัตโนมัติ ไม่มีทางทำเงียบๆ ได้

---

**ข้อมูลถูกแบ่งสามกอง ไม่ใช่สอง** — ผู้สอนกรอกสองกอง ระบบคำนวณกองที่สามให้

⚠️ **เดิมตัวเลขที่สองเป็นสัดส่วน*ภายในส่วนที่กันไว้* ไม่ใช่ของทั้งไฟล์** ซึ่ง
อธิบายไม่ได้จริง — ผู้สอนถามตรงๆ ว่า "เลข 0.5 มันมาจากไหน" และนั่นคือคำถามที่ถูก
· ตอนนี้ทั้งสองช่องเป็นสัดส่วน**ของทั้งไฟล์**เหมือนกัน เทียบกันได้ตรงๆ และ
กองที่สามเป็นผลลบซึ่งไม่มีอะไรให้กรอกผิด

    ทั้งไฟล์ = 100%
    ├── student_ratio ──► แจกนิสิต — เขาแบ่ง train/val เองตามใจ
    ├── final_ratio ────► 🔒 ตัดสินรอบสุดท้าย
    └── ที่เหลือ ───────► 🔒 leaderboard ระหว่างเทอม (คำนวณให้เอง)

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

#: ขอบเขตของกองที่ใช้ตัดสินรอบสุดท้าย
MIN_FINAL_RATIO = 0.05
MAX_FINAL_RATIO = 0.5


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
    #: สัดส่วน**ของทั้งไฟล์**ที่แจกนิสิต
    student_ratio: float = 0.7
    #: สัดส่วน**ของทั้งไฟล์**ที่เก็บไว้ตัดสินรอบสุดท้าย
    #:
    #: กองที่เหลือ (`1 - student_ratio - final_ratio`) คือ leaderboard ระหว่างเทอม
    #: — ไม่มีช่องให้กรอกเพราะมันเป็นผลลบ และช่องที่กรอกได้แต่ต้องรวมกันพอดี
    #: คือช่องที่กรอกผิดได้
    final_ratio: float = 0.15
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
        if not MIN_FINAL_RATIO <= self.final_ratio <= MAX_FINAL_RATIO:
            raise ConfigError(
                f"สัดส่วนที่ใช้ตัดสินรอบสุดท้ายต้องอยู่ระหว่าง {MIN_FINAL_RATIO} กับ "
                f"{MAX_FINAL_RATIO} — ได้ {self.final_ratio}"
            )
        if self.leaderboard_ratio <= 0.0:
            raise ConfigError(
                f"แจกนิสิต {self.student_ratio:.0%} + ตัดสินรอบสุดท้าย {self.final_ratio:.0%} "
                f"= {self.student_ratio + self.final_ratio:.0%} ไม่เหลือข้อมูลให้ leaderboard เลย\n"
                "  สองค่านี้เป็นสัดส่วนของทั้งไฟล์ รวมกันต้องน้อยกว่า 100%"
            )
        # `dataset` ตรวจแค่รูปแบบตรงนี้ — การมีอยู่จริงของไฟล์ตรวจที่ `store`
        # เพราะโมดูลนี้ต้องโหลดได้บนเครื่องที่ไม่มีคลัง (เช่นตอนคำนวณ hash)
        from tabular.store import DIGEST_RE

        if not DIGEST_RE.match(self.dataset):
            raise ConfigError(
                f"`dataset` ต้องเป็นลายนิ้วมือของไฟล์ในคลัง (sha256:<64 หลัก>) — ได้ {self.dataset!r}"
            )

    @property
    def leaderboard_ratio(self) -> float:
        """สัดส่วนของทั้งไฟล์ที่ใช้ให้คะแนนบนกระดานระหว่างเทอม — **คำนวณ ไม่ได้กรอก**"""
        return 1.0 - self.student_ratio - self.final_ratio

    @property
    def grading_ratio(self) -> float:
        """สัดส่วนของทั้งไฟล์ที่นิสิตไม่เคยเห็น — สองกองรวมกัน"""
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
            "  นิสิตแบ่ง train/val/test เอง ระบบจึงไม่มี `ratios` `n_rows` `data_seed` อีก\n"
            "  และ `grading_public_ratio` ถูกแทนด้วย `final_ratio` ซึ่งเป็นสัดส่วนของ*ทั้งไฟล์*"
        )
    try:
        return TaskSpec(**raw)
    except TypeError as exc:
        raise ConfigError(f"{source}: {exc}") from exc
