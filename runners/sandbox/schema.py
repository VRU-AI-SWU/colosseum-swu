"""อธิบายหน้าตาของ config ให้หน้าเว็บสร้างฟอร์มได้ — ใช้ร่วมกันทุกชนิดโจทย์

**โครงของฟอร์มถูกอนุมานจาก dataclass ของ config เอง ไม่ได้เขียนมือ** — รายการ
ฟิลด์ที่เขียนมือจะเพี้ยนจากโค้ดจริงในวันที่มีคนเพิ่มฟิลด์ใหม่แล้วลืมแก้ ผลคือฟอร์ม
สร้าง config ที่ขาดฟิลด์ แล้ว loader ปฏิเสธ — ผู้สอนกรอกครบทุกช่องแล้วกดบันทึก
ไม่ได้โดยไม่มีใครบอกว่าเพราะอะไร

สิ่งที่อนุมานไม่ได้คือ **ขอบเขตของค่า** (ช่วงตัวเลข · ตัวเลือกที่ implement แล้ว)
เพราะมันอยู่ในโค้ด `validate()` แบบคำสั่ง ไม่ใช่แบบประกาศ · ส่วนนั้นแต่ละ env
ประกาศเพิ่มเองผ่าน `Limit` และ **มีเทสต์บังคับว่าสิ่งที่ประกาศต้องตรงกับสิ่งที่
`validate()` บังคับจริง** ไม่งั้นฟอร์มจะรับค่าที่ loader ไม่รับ
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any


@dataclass(frozen=True)
class Limit:
    """ขอบเขตของฟิลด์หนึ่ง — ส่วนที่อนุมานจาก dataclass ไม่ได้"""

    #: ตัวเลือกทั้งหมด (enum) — `None` = ไม่ใช่ enum
    choices: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    #: คำอธิบายสั้นๆ ที่ขึ้นใต้ช่องกรอก
    help: str = ""
    #: แก้ไม่ได้ผ่านฟอร์ม — ค่าที่เปลี่ยนแล้วต้อง generate seed ใหม่ทั้งชุด
    #: หรือค่าที่ไม่ใช่การตั้งค่าโจทย์ (เช่น `version`)
    fixed: bool = False


@dataclass(frozen=True)
class Field:
    """ช่องกรอกหนึ่งช่องบนฟอร์ม"""

    key: str            # dotted เช่น "room.width" — ตรงกับที่ `Config.replace` รับ
    type: str           # int | float | bool | str | enum
    default: Any
    #: ไม่มีค่าเริ่มต้นใน dataclass = **ช่องที่ต้องกรอก** ไม่ใช่ข้อผิดพลาด
    #: (`TaskSpec` ของ CP462 บังคับให้ประกาศ slug/kind/primary เอง ส่วน `Config`
    #: ของ CP463 มีค่าเริ่มต้นครบทุกช่อง — ทั้งสองแบบต้องอธิบายเป็นฟอร์มได้)
    required: bool = False
    section: str = ""   # กลุ่มบนฟอร์ม เช่น "room"
    choices: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    help: str = ""
    fixed: bool = False

    def as_dict(self) -> dict[str, Any]:
        out = {
            "key": self.key,
            "type": self.type,
            "default": self.default,
            "required": self.required,
            "section": self.section,
            "fixed": self.fixed,
        }
        if self.choices is not None:
            out["choices"] = list(self.choices)
        for name in ("minimum", "maximum"):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        if self.help:
            out["help"] = self.help
        return out


_PYTHON_TO_FORM = {int: "int", float: "float", bool: "bool", str: "str"}


def _form_type(value: Any, annotation: Any) -> str:
    """ชนิดของช่องกรอก — ดูจากค่าเริ่มต้นก่อน แล้วค่อยถอยไปดู annotation

    ดูจากค่าจริงก่อนเพราะ annotation ของ config มักเป็น `int | None` ซึ่งบอก
    ชนิดที่ฟอร์มต้องใช้ไม่ได้ตรงๆ · `bool` ต้องเช็คก่อน `int` เพราะใน Python
    `isinstance(True, int)` เป็นจริง แล้วช่องกาถูกจะกลายเป็นช่องตัวเลข
    """
    for python_type, form_type in _PYTHON_TO_FORM.items():
        if type(value) is python_type:  # noqa: E721 — ต้องเป็นชนิดนั้นเป๊ะ ไม่ใช่ subclass
            return form_type
    text = str(annotation)
    for python_type, form_type in _PYTHON_TO_FORM.items():
        if python_type.__name__ in text:
            return form_type
    return "str"


def derive(config_class: type, limits: dict[str, Limit] | None = None) -> list[Field]:
    """เดินต้นไม้ dataclass ของ config แล้วคืนรายการช่องกรอก

    รองรับซ้อนหนึ่งชั้น (`Config.room.width` → `"room.width"`) ซึ่งเป็นโครงที่
    config ของทุก env ใช้อยู่ · ซ้อนลึกกว่านั้นยังไม่รองรับ และจะล้มดังๆ แทนที่
    จะเงียบ เพราะฟอร์มที่ตกฟิลด์ไปเงียบๆ แย่กว่าฟอร์มที่สร้างไม่ได้
    """
    limits = limits or {}
    out: list[Field] = []

    for parent in fields(config_class):
        default, required = _default_of(parent)

        if is_dataclass(default):
            for child in fields(type(default)):
                key = f"{parent.name}.{child.name}"
                value = getattr(default, child.name)
                if is_dataclass(value):
                    raise TypeError(f"{key}: config ซ้อนเกินหนึ่งชั้น — schema ยังไม่รองรับ")
                out.append(_field(key, value, child.type, False, parent.name, limits.get(key)))
            continue

        out.append(
            _field(parent.name, default, parent.type, required, "", limits.get(parent.name))
        )

    return out


def _default_of(spec) -> tuple[Any, bool]:
    """คืน `(ค่าเริ่มต้น, ต้องกรอกไหม)` — ไม่มีค่าเริ่มต้นแปลว่าต้องกรอก"""
    if spec.default is not MISSING:
        return spec.default, False
    if spec.default_factory is not MISSING:  # type: ignore[misc]
        return spec.default_factory(), False  # type: ignore[misc]
    return None, True


def _field(
    key: str, default: Any, annotation: Any, required: bool, section: str, limit: Limit | None
) -> Field:
    limit = limit or Limit()
    return Field(
        key=key,
        type="enum" if limit.choices else _form_type(default, annotation),
        default=default,
        required=required,
        section=section,
        choices=limit.choices,
        minimum=limit.minimum,
        maximum=limit.maximum,
        help=limit.help,
        fixed=limit.fixed,
    )


def as_dicts(schema: list[Field]) -> list[dict[str, Any]]:
    return [f.as_dict() for f in schema]
