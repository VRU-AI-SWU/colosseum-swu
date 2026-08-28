"""ตรวจ submission ของโจทย์ทำนาย — กติกาที่ใช้ร่วมกันอยู่ที่ `runners/sandbox/archive.py`

⚠️ **แยกสองชั้นตามที่มันรันอยู่คนละเครื่อง** เหมือนโจทย์ RL

| ชั้น | รันที่ไหน | รันโค้ดนิสิตไหม | เมื่อไร |
|---|---|---|---|
| `inspect_archive()` | **cloud API** | ❌ **ห้ามเด็ดขาด** | ตอนอัพโหลด |
| `smoke_test()` | **runner on-prem ใน sandbox** | ✅ | ก่อนเข้าคิวจริง |

**สิ่งที่ต่างจากโจทย์ RL อย่างสำคัญ** — ที่นี่นิสิตส่งไฟล์โมเดลมาด้วย ไม่ใช่แค่ซอร์ส
ความผิดพลาดที่พบบ่อยที่สุดจึงเป็น "ลืมใส่ `pipeline.pkl` ลงใน zip" ซึ่งเดิมจะผ่าน
การตรวจ กินโควตา แล้วไปตายตอนรัน · `check_referenced_files` ของโมดูลกลางจับให้แล้ว
โดยอ่านชื่อไฟล์ที่โค้ดจะเปิดตรงๆ จาก `ast`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from runners.prediction.sandbox import SANDBOX
from runners.sandbox import archive as archive_mod
from runners.sandbox.archive import Problem, ValidationReport  # noqa: F401

ENTRY = "predictor.py"
PREDICTOR_CLASS = "Predictor"
REQUIRED_METHODS = ("__init__", "predict")

#: package ที่ติดตั้งอยู่ใน `arena/tabular:cpu` — ต้องตรงกับ Dockerfile ของ image นั้น
#: การเพิ่มชื่อที่นี่โดยไม่เพิ่มใน image ทำให้ submission ผ่านการตรวจแล้วไปตายตอนรัน
DEFAULT_WHITELIST = frozenset({"numpy", "pandas", "sklearn", "scipy", "joblib"})


def _check_predictor_class(source: str, report: ValidationReport) -> None:
    archive_mod.check_class(
        source, report,
        entry=ENTRY, class_name=PREDICTOR_CLASS, methods=REQUIRED_METHODS,
        signature_hint="ต้องมี __init__(self, config) · predict(self, X)",
    )


def inspect_archive(archive: str | Path) -> ValidationReport:
    return archive_mod.inspect_archive(archive, entry=ENTRY, check_source=_check_predictor_class)


def check_import_whitelist(
    report: ValidationReport, whitelist: Iterable[str] = DEFAULT_WHITELIST
) -> ValidationReport:
    return archive_mod.check_import_whitelist(report, whitelist)


# ── ชั้นที่ 2: dynamic (ต้องรันใน sandbox บน runner) ────────────────


@dataclass
class SmokeResult:
    problems: list[Problem] = field(default_factory=list)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        return "ผ่าน" if self.ok else "\n".join(str(p) for p in self.problems)


#: ข้อความผิดพลาดของ runner → คำอธิบายที่นิสิตเอาไปแก้ได้
_ADVICE = {
    "predictor_init_failed": (
        "สร้าง `Predictor` ไม่สำเร็จ",
        "ที่พบบ่อยคือลืมใส่ไฟล์โมเดลลงใน zip หรือ scikit-learn คนละรุ่นกับที่ colosseum ใช้ "
        "— รัน `python -m tabular.selfcheck` ในเครื่องตัวเองก่อน",
    ),
    "predict_failed": (
        "`predict` โยน exception",
        "ลองเรียก `Predictor({}).predict(X)` บนชุด val ในเครื่องตัวเองก่อนส่ง",
    ),
    "predict_timeout": (
        "`predict` ใช้เวลานานเกินเพดาน",
        "ตัดงานที่ทำซ้ำทุกครั้งที่ทำนายออกไปไว้ตอน fit — โมเดลที่ส่งมาต้อง fit เสร็จแล้ว",
    ),
    "bad_prediction": (
        "คำทำนายไม่ตรงสัญญา",
        "ต้องคืนค่าให้ครบทุกแถวตามลำดับที่รับเข้ามา เป็นอาเรย์มิติเดียว ไม่มี NaN",
    ),
    "nondeterministic": (
        "ทำนายชุดเดิมสองครั้งได้ผลไม่เท่ากัน",
        "ตรึง `random_state` ของทุกตัวใน pipeline",
    ),
    "row_order_dependent": (
        "สลับลำดับแถวแล้วคำทำนายเปลี่ยน",
        "โมเดลใช้ข้อมูลจากแถวอื่นในก้อนเดียวกัน ซึ่งตอนใช้งานจริงไม่มี",
    ),
    "batch_dependent": (
        "ทำนายเฉพาะบางแถวได้ผลต่างจากทำนายทั้งก้อน",
        "โมเดลคำนวณสถิติจากก้อนที่รับเข้ามาแทนค่าที่จำไว้ตอน `fit`",
    ),
}


def smoke_test(
    *, env_plugin: str, config_path: str | Path, submission_dir: str | Path, launcher=None
) -> SmokeResult:
    """เปิดกล่องแล้วทำนายจริงหนึ่งรอบ **พร้อมตรวจครบสามชั้น**

    ⚠️ ต่างจาก smoke test ของโจทย์ RL ตรงที่**ไม่ตัดการตรวจออก** — สามชั้นนั้นคือ
    สิ่งที่กัน leakage และมันเป็นเหตุผลที่พบบ่อยที่สุดที่ submission จะถูกปฏิเสธ
    การให้นิสิตรู้ตั้งแต่ตอน dry run ว่าตกข้อไหน มีค่ามากกว่าเวลาที่ประหยัดได้
    """
    from runners.prediction.runner import run_submission

    result = run_submission(
        env_plugin=env_plugin,
        config_path=config_path,
        submission_dir=submission_dir,
        launcher=launcher or SANDBOX.local(),
    )
    if result.ok:
        return SmokeResult()

    message, fix = _ADVICE.get(
        result.status, (f"รันไม่ผ่าน ({result.status})", "ดูรายละเอียดข้างล่าง")
    )
    return SmokeResult(
        problems=[Problem(result.status, message, fix)],
        detail=(result.detail or "") + ("\n\n--- log ---\n" + result.log if result.log else ""),
    )
