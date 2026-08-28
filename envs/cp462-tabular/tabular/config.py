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
    n_rows: int
    data_seed: int
    split_seed: int
    bootstrap_seed: int
    ratios: tuple[float, float, float, float]
    #: ลำดับของคลาสที่ตรึงไว้ — classification เท่านั้น · **ต้องไม่เปลี่ยนกลางเทอม**
    #: เพราะ confusion matrix กับ per-class F1 อ้างลำดับนี้
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
        if self.kind == "classification" and not self.labels:
            raise ConfigError("classification ต้องประกาศ `labels` เพื่อตรึงลำดับของคลาส")
        if self.kind == "regression" and self.labels:
            raise ConfigError("regression ต้องไม่มี `labels`")
        if len(self.ratios) != 4:
            raise ConfigError(f"ratios ต้องมี 4 ค่า (train/val/test_public/test_private) — ได้ {len(self.ratios)}")
        if abs(sum(self.ratios) - 1.0) > 1e-9:
            raise ConfigError(f"ratios ต้องรวมกันได้ 1.0 — ได้ {sum(self.ratios)}")
        if self.n_rows < 100:
            raise ConfigError(f"ข้อมูล {self.n_rows} แถวน้อยเกินไปสำหรับการแบ่งสี่ส่วน")

    def normalized(self) -> dict:
        """รูปแบบมาตรฐานสำหรับคำนวณ hash — เรียงคีย์และแปลง tuple เป็น list"""
        data = asdict(self)
        data["ratios"] = list(self.ratios)
        data["labels"] = [str(label) for label in self.labels]
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
