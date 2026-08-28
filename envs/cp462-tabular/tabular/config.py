"""TaskSpec — ทุกอย่างที่ต้องประกาศต่อ competition หนึ่งอัน (template §13)

**`config_hash` คือสัญญา** — บันทึกลงทุก run · ถ้าค่านี้เปลี่ยนแปลว่าโจทย์เปลี่ยน
แล้วคะแนนเก่าเทียบกับคะแนนใหม่ไม่ได้ · แพลตฟอร์มตรวจซ้ำตอนให้คะแนนเหมือนที่
`cp463-vacuum` ทำ เพราะการเปลี่ยน config เงียบๆ กลางเทอมคือการเปลี่ยนกติกา
โดยที่ leaderboard ยังดูเหมือนเทียบกันได้

**ทุกเมล็ดแยกกันโดยตั้งใจ** — `data_seed` / `split_seed` / `bootstrap_seed`
ถ้าใช้เมล็ดเดียวกัน การเปลี่ยนอย่างหนึ่งจะลากอย่างอื่นเปลี่ยนตามโดยไม่ตั้งใจ
เช่นแก้จำนวน bootstrap แล้วข้อมูลเปลี่ยนทั้งชุด
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

KINDS = ("classification", "regression")

#: คะแนนหลักที่ใช้ได้ต่อชนิดโจทย์ — ทุกตัว "มากกว่าดีกว่า"
PRIMARY_BY_KIND = {
    "classification": ("macro_f1", "accuracy"),
    "regression": ("r2", "neg_rmse", "neg_mae"),
}


class ConfigError(Exception):
    """config ไม่ถูกต้อง — ต้องล้มตั้งแต่โหลด ไม่ใช่ไปพังตอนให้คะแนน"""


@dataclass(frozen=True)
class TaskSpec:
    """โจทย์หนึ่งอัน — อ่านจาก YAML แล้วตรึงไว้ทั้งเทอม"""

    slug: str
    task: str          # ชื่อใน `generator.TASKS`
    title: str
    kind: str
    primary: str
    #: จำนวนแถวของ **ชุดที่แจกนิสิต** — ชุดที่ใช้ตัดสินใช้ `grading_rows`
    n_rows: int
    data_seed: int
    split_seed: int
    bootstrap_seed: int
    #: train / val / test ของนิสิต — **ทุกส่วนนิสิตมีอยู่แล้ว**
    ratios: tuple[float, float, float]
    #: ขนาดของชุดที่ใช้ตัดสิน · ประกาศต่อสาธารณะได้ — รู้ขนาดไม่ได้ช่วยให้เดาเฉลย
    grading_rows: int = 0
    #: สัดส่วนที่เป็น `test_public` ที่เหลือเป็น `test_private`
    grading_public_ratio: float = 0.4
    #: ลำดับของคลาสที่ตรึงไว้ — classification เท่านั้น · **ต้องไม่เปลี่ยนกลางเทอม**
    #: เพราะ confusion matrix กับ per-class F1 อ้างลำดับนี้
    labels: list[Any] = field(default_factory=list)

    #: 🔒 **เมล็ดของชุดที่ใช้ตัดสิน — ไม่มีในไฟล์ที่แจก และไม่เข้า `config_hash`**
    #:
    #: ฝั่ง trusted ฉีดค่านี้เข้ามาตอนโหลด (`tabular.arena.PLUGIN.load_spec`)
    #: โดยอ่านจาก `ARENA_SECRETS` · ถ้าเป็น `None` แปลว่ากำลังอยู่ฝั่งนิสิต
    #: แล้ว `grading_data()` จะปฏิเสธพร้อมบอกว่าทำไม
    #:
    #: **ไม่เข้า hash โดยตั้งใจ** — `config_hash` คือสัญญาเรื่อง *ข้อมูลของนิสิต
    #: และวิธีให้คะแนน* ซึ่ง `selfcheck` บนเครื่องนิสิตต้องคำนวณให้ตรงกับของ grader ได้
    #: ถ้าเมล็ดลับเข้า hash ด้วย ค่าสองฝั่งจะไม่มีทางตรงกันเลย · การกันเมล็ดลับ
    #: เปลี่ยนกลางเทอมใช้วิธีเดียวกับ CP463: มันอยู่ใน repo ส่วนตัวที่มีประวัติการแก้
    grading_seed: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ConfigError(f"kind ต้องเป็น {KINDS} — ได้ {self.kind!r}")
        allowed = PRIMARY_BY_KIND[self.kind]
        if self.primary not in allowed:
            raise ConfigError(
                f"{self.kind} ใช้คะแนนหลัก {self.primary!r} ไม่ได้ — ที่ใช้ได้คือ {allowed}\n"
                "  ทุกตัวเป็นแบบ 'มากกว่าดีกว่า' ตามที่ leaderboard ต้องการ"
            )
        if self.kind == "classification" and not self.labels:
            raise ConfigError("classification ต้องประกาศ `labels` เพื่อตรึงลำดับของคลาส")
        if self.kind == "regression" and self.labels:
            raise ConfigError("regression ต้องไม่มี `labels`")
        if len(self.ratios) != 3:
            raise ConfigError(f"ratios ต้องมี 3 ค่า (train/val/test ของนิสิต) — ได้ {len(self.ratios)}")
        if abs(sum(self.ratios) - 1.0) > 1e-9:
            raise ConfigError(f"ratios ต้องรวมกันได้ 1.0 — ได้ {sum(self.ratios)}")
        if self.n_rows < 100:
            raise ConfigError(f"ข้อมูล {self.n_rows} แถวน้อยเกินไปสำหรับการแบ่งสามส่วน")
        if self.grading_rows < 100:
            raise ConfigError(
                f"ชุดที่ใช้ตัดสิน {self.grading_rows} แถวน้อยเกินไป — "
                "ช่วงความเชื่อมั่นจะกว้างจนอันดับไม่มีความหมาย"
            )
        if not 0.0 < self.grading_public_ratio < 1.0:
            raise ConfigError(
                f"grading_public_ratio ต้องอยู่ระหว่าง 0 กับ 1 — ได้ {self.grading_public_ratio}"
            )

    def normalized(self) -> dict:
        """รูปแบบมาตรฐานสำหรับคำนวณ hash — เรียงคีย์และแปลง tuple เป็น list"""
        data = asdict(self)
        data["ratios"] = list(self.ratios)
        data["labels"] = [str(label) for label in self.labels]
        # `title` เป็นข้อความให้คนอ่าน ไม่กระทบการให้คะแนน — แก้ได้โดยไม่ทำให้
        # คะแนนเก่าเทียบไม่ได้ จึงไม่นับเข้า hash
        data.pop("title")
        # 🔒 เมล็ดลับต้องไม่เข้า hash — เหตุผลอยู่ที่ฟิลด์นั้น
        data.pop("grading_seed")
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
        data["ratios"] = tuple(data["ratios"])
        return TaskSpec(**data)


def load_config(path: str | Path) -> TaskSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} ต้องเป็น mapping ที่ระดับบนสุด")
    unknown = sorted(set(raw) - set(TaskSpec.__dataclass_fields__))
    if unknown:
        raise ConfigError(f"{path}: ไม่รู้จักฟิลด์ {unknown}")
    raw["ratios"] = tuple(raw.get("ratios", ()))
    try:
        return TaskSpec(**raw)
    except TypeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def spec_path(slug: str) -> Path:
    path = CONFIG_DIR / f"{slug}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
        raise ConfigError(f"ไม่รู้จักโจทย์ {slug!r} — ที่มีคือ {available}")
    return path


def load(slug: str) -> TaskSpec:
    return load_config(spec_path(slug))
