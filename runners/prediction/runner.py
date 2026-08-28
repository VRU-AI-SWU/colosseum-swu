"""ฝั่ง **trusted** — ถือเฉลย · ตรวจสามชั้น · ให้คะแนน (template §4, §5)

ลำดับการทำงานของหนึ่ง submission

    เปิดกล่อง → handshake → predict บนชุดเต็ม (นี่คือคำทำนายที่ใช้ให้คะแนน)
                          → predict ซ้ำชุดเดิม          ต้องได้เท่าเดิม
                          → predict บนชุดที่สลับแถว     ต้องได้เท่าเดิมทีละแถว
                          → predict บน subset 30%       ต้องได้เท่าเดิมทีละแถว
                          → คิดคะแนนจากคำทำนายชุดแรก

**เฉลยไม่เคยเข้ากล่อง** สิ่งที่ส่งเข้าไปมีแค่ `spec.X` ที่ผ่าน `frame.encode_frame`
การให้คะแนนเกิดในไฟล์นี้ทั้งหมด หลังคำทำนายเดินทางกลับออกมาแล้ว

**ทำไมตรวจสามชั้นทั้งที่นิสิตส่ง pipeline ที่ fit แล้ว** — pipeline ที่ fit แล้ว
ยังทำตัวผิดได้ถ้ามี transformer ที่คำนวณสถิติจาก batch ที่รับเข้ามาแทนที่จะใช้ค่า
ที่จำไว้ตอน fit (เจอบ่อยใน transformer ที่นิสิตเขียนเอง) การตรวจจึงย้ายจาก
`transform` มาที่ `predict` ไม่ได้หายไป

การตรวจทั้งสามชั้นรัน **ในรอบเดียวกับตอนให้คะแนนจริง** ไม่ใช่ stage แยก — เรียก
`predict` เพิ่มอีกสามครั้งบนก้อนเล็กๆ ถูกกว่าการเปิดกล่องใหม่มาก
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from runners.prediction import plugin as plugin_mod
from runners.prediction.frame import decode_values, encode_frame
from runners.prediction.messages import PREDICT, PREDICTION
from runners.prediction.sandbox import SANDBOX
from runners.sandbox.launcher import Launcher, SandboxProcess
from runners.sandbox.protocol import (
    CLOSE,
    ERROR,
    HELLO,
    PROTOCOL_VERSION,
    READY,
    ProtocolError,
)

HANDSHAKE_TIMEOUT_S = 120.0  # `joblib.load` ของ pipeline ใหญ่ๆ กินเวลาได้หลายสิบวินาที

#: สัดส่วนของ subset ที่ใช้ตรวจ batch dependence (template §4)
SUBSET_FRACTION = 0.3

#: **ก้อนจิ๋วที่ตรวจด้วยอีกก้อนหนึ่ง** — subset 30% อย่างเดียวไม่พอ
#:
#: เจอตอนเขียนเทสต์: predictor ที่ทำนายด้วย `x > X["num"].mean()` ซึ่งเป็น leakage
#: ชัดๆ **รอดการตรวจ 30% ไปได้** เพราะค่าเฉลี่ยของ 18 แถวบังเอิญใกล้ค่าเฉลี่ยของ
#: 60 แถวมากพอที่ไม่มีแถวไหนตกคนละฝั่ง · ก้อนยิ่งเล็ก สถิติของก้อนยิ่งเหวี่ยงห่าง
#: จากของชุดเต็ม การตรวจจึงไวขึ้นมาก และราคาของมันคือการเรียก `predict` บนไม่กี่แถว
#:
#: pipeline ที่ถูกต้องทำนายทีละแถวเป็นอิสระอยู่แล้ว ก้อนเล็กจึงไม่ทำให้มันตกด้วย
TINY_SUBSET_ROWS = 5

#: จำนวนแถวที่ยกมาแสดงเวลาการตรวจไม่ผ่าน — พอให้เห็นรูปแบบ ไม่ท่วมหน้าจอ
EXAMPLES_SHOWN = 5


@dataclass
class PredictionResult:
    """ผลของ submission หนึ่งครั้ง — โครงเดียวกับ `RunResult` ของโจทย์ RL"""

    status: str
    env_plugin: str
    env_version: str = ""
    config_hash: str = ""
    #: คะแนนของ plugin (`Score`) — `None` เมื่อไปไม่ถึงขั้นให้คะแนน
    score: Any = None
    #: ผลการตรวจสามชั้น — ชื่อ → ผ่านหรือไม่
    checks: dict[str, bool] = field(default_factory=dict)
    n_rows: int = 0
    log: str = ""
    detail: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class _PredictorFailure(Exception):
    """โค้ดนิสิตล้มในระดับที่ให้คะแนนต่อไม่ได้"""

    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def run_submission(
    *,
    env_plugin: str,
    config_path: str | Path,
    submission_dir: str | Path,
    kind: str = "public",
    config_overrides: dict[str, Any] | None = None,
    launcher: Launcher | None = None,
    run_checks: bool = True,
) -> PredictionResult:
    """รัน submission หนึ่งอันแล้วคืนคะแนน — ไม่โยน exception ออกไปนอกฟังก์ชัน

    `run_checks=False` มีไว้สำหรับ smoke test ที่แค่อยากรู้ว่ากล่องเปิดติดไหม
    **ห้ามใช้ตอนให้คะแนนจริง** เพราะการตรวจสามชั้นคือสิ่งที่กัน leakage
    """
    plugin = plugin_mod.resolve(env_plugin)
    spec = plugin.apply_overrides(plugin.load_spec(str(config_path)), config_overrides or {})
    launcher = launcher or SANDBOX.local()
    timeout = plugin.predict_timeout_s(spec)

    result = PredictionResult(
        status="ok",
        env_plugin=env_plugin,
        env_version=plugin.env_version(spec),
        config_hash=plugin.config_hash(spec),
    )

    data = plugin.grading_data(spec, kind)  # 🔒 มีเฉลย — ห้ามส่งอะไรจากตรงนี้เข้ากล่อง
    features = data.X
    result.n_rows = len(features)

    started = time.monotonic()
    box = launcher.start(Path(submission_dir))
    try:
        _handshake(box, plugin.predictor_config(spec))
        y_pred = _predict(box, features, timeout, what="ชุดที่ใช้ตัดสิน")

        if len(y_pred) != len(features):
            raise _PredictorFailure(
                "bad_prediction",
                f"`predict` คืนคำทำนาย {len(y_pred)} ค่า แต่รับ {len(features)} แถว\n"
                "  ต้องคืนให้ครบทุกแถวตามลำดับที่รับเข้ามา",
            )

        if run_checks:
            result.checks = _run_checks(box, features, y_pred, timeout, spec, plugin)

        try:
            result.score = plugin.score(spec, data.y, y_pred)
        except Exception as exc:  # noqa: BLE001 — metric ที่คิดไม่ได้คือ submission ที่ผิดสัญญา
            raise _PredictorFailure("bad_prediction", str(exc)) from exc

        try:
            box.channel.send(CLOSE)
        except OSError:
            pass

    except _PredictorFailure as exc:
        result.status, result.detail = exc.status, exc.detail
    except (ProtocolError, EOFError) as exc:
        result.status, result.detail = "protocol_error", str(exc)
    finally:
        result.log = box.log
        box.close()
        result.seconds = time.monotonic() - started

    return result


# ── คุยกับกล่อง ────────────────────────────────────────────────────


def _handshake(box: SandboxProcess, predictor_config: dict[str, Any]) -> None:
    box.channel.send(HELLO, protocol=PROTOCOL_VERSION, predictor_config=predictor_config)
    try:
        reply = box.channel.recv(timeout=HANDSHAKE_TIMEOUT_S)
    except TimeoutError as exc:
        raise _PredictorFailure(
            "predictor_init_failed", f"predictor ไม่ตอบ handshake: {exc}"
        ) from exc
    except EOFError as exc:
        raise _PredictorFailure(
            "predictor_died",
            f"predictor จบไปก่อนจะ handshake เสร็จ — ดู log\n{box.log[-2000:]}",
        ) from exc

    if reply["t"] == ERROR:
        raise _PredictorFailure(
            "predictor_init_failed",
            "สร้าง Predictor ไม่สำเร็จ:\n" + reply.get("traceback", "(ไม่มี traceback)"),
        )
    if reply["t"] != READY:
        raise ProtocolError(f"คาดว่าจะได้ {READY!r} — ได้ {reply['t']!r}")


def _predict(box: SandboxProcess, features, timeout: float, *, what: str) -> np.ndarray:
    """เรียก `predict` หนึ่งครั้ง — คืนคำทำนายเป็นอาเรย์

    `what` เป็นคำอธิบายว่ากำลังทำนายก้อนไหน เพื่อให้ข้อความผิดพลาดบอกได้ว่า
    พังตอนให้คะแนนหรือพังตอนตรวจ ซึ่งเป็นคนละเรื่องกันสำหรับคนที่มาไล่ปัญหา
    """
    box.channel.send(PREDICT, frame=encode_frame(features))
    try:
        reply = box.channel.recv(timeout=timeout)
    except TimeoutError as exc:
        raise _PredictorFailure(
            "predict_timeout",
            f"`predict` บน{what} ({len(features)} แถว) ไม่เสร็จใน {timeout:.0f} วินาที",
        ) from exc
    except EOFError as exc:
        raise _PredictorFailure(
            "predictor_died", f"predictor จบไปกลางคัน ตอนทำนาย{what} — ดู log\n{box.log[-2000:]}"
        ) from exc

    if reply["t"] == ERROR:
        raise _PredictorFailure(
            "predict_failed",
            f"`predict` บน{what} โยน exception:\n{reply.get('traceback', '(ไม่มี traceback)')}",
        )
    if reply["t"] != PREDICTION:
        raise ProtocolError(f"คาดว่าจะได้ {PREDICTION!r} — ได้ {reply['t']!r}")
    return decode_values(reply["y"])


# ── การตรวจสามชั้น ─────────────────────────────────────────────────


def _check_rng(config_hash: str) -> np.random.Generator:
    """ตัวสุ่มของการตรวจ — ผูกกับ `config_hash` ไม่ใช่กับเวลา

    ตรึงไว้เพื่อให้ **การตรวจที่ไม่ผ่านเกิดซ้ำได้** ผู้สอนจะได้ไล่ปัญหากับนิสิตบน
    ชุดเดียวกัน · ผูกกับ config_hash แทนค่าคงที่เพื่อให้แต่ละโจทย์ใช้คนละชุด
    """
    return np.random.default_rng(int(config_hash[-8:], 16))


def _run_checks(box, features, y_pred, timeout, spec, plugin) -> dict[str, bool]:
    n = len(features)
    rng = _check_rng(plugin.config_hash(spec))

    # 1) ทำนายซ้ำบน input เดิมเป๊ะๆ ต้องได้ผลเดิม
    again = _predict(box, features, timeout, what="ชุดเดิมรอบสอง (ตรวจความคงที่)")
    _require_same(
        y_pred, again, np.arange(n),
        status="nondeterministic",
        headline="ทำนายชุดเดิมสองครั้งแล้วได้ผลไม่เท่ากัน",
        advice="ตรึง `random_state` ของทุกตัวใน pipeline · เลี่ยงอะไรที่สุ่มตอน `predict`\n"
               "  โมเดลที่ให้ผลไม่ซ้ำเดิมทำให้อันดับไม่มีความหมายและ rejudge ไม่ได้",
    )

    # 2) สลับลำดับแถว — คำทำนายของแต่ละแถวต้องไม่เปลี่ยน
    order = rng.permutation(n)
    shuffled = _predict(
        box, features.iloc[order].reset_index(drop=True), timeout,
        what="ชุดที่สลับลำดับแถว (ตรวจ leakage)",
    )
    _require_same(
        y_pred[order], shuffled, order,
        status="row_order_dependent",
        headline="สลับลำดับแถวแล้วคำทำนายของแถวเดิมเปลี่ยนไป",
        advice="แปลว่าโมเดลใช้ข้อมูลจากแถวอื่นในก้อนเดียวกัน ซึ่งตอนใช้งานจริงไม่มี\n"
               "  มักเกิดจาก transformer ที่ `fit` ใหม่ตอน `predict` แทนที่จะใช้ค่าที่จำไว้",
    )

    # 3) ทำนายเฉพาะ subset — ผลของแถวที่อยู่ในทั้งสองก้อนต้องตรงกัน
    #    สองขนาด: 30% ตามที่ template ประกาศไว้ และก้อนจิ๋วที่ไวกว่ามาก (ดู TINY_SUBSET_ROWS)
    sizes = [max(1, int(n * SUBSET_FRACTION))]
    if n > TINY_SUBSET_ROWS:
        sizes.append(TINY_SUBSET_ROWS)

    for size in sizes:
        take = np.sort(rng.choice(n, size=size, replace=False))
        subset = _predict(
            box, features.iloc[take].reset_index(drop=True), timeout,
            what=f"subset {len(take)} แถว (ตรวจ leakage)",
        )
        _require_same(
            y_pred[take], subset, take,
            status="batch_dependent",
            headline=f"ทำนายเฉพาะ {len(take)} แถวแล้วได้ผลต่างจากตอนทำนายทั้ง {n} แถว",
            advice="แปลว่าโมเดลคำนวณสถิติจากก้อนที่รับเข้ามา (ค่าเฉลี่ย · ส่วนเบี่ยงเบน ·\n"
                   "  รายการหมวด) แทนที่จะใช้ค่าที่จำไว้ตอน `fit` — ตอนใช้งานจริงที่ทำนาย\n"
                   "  ทีละแถว โมเดลแบบนี้จะให้ผลคนละอย่างกับที่วัดไว้",
        )

    return {"determinism": True, "row_permutation": True, "subset_consistency": True}


def _require_same(expected, got, row_ids, *, status: str, headline: str, advice: str) -> None:
    if len(expected) != len(got):
        raise _PredictorFailure(
            status,
            f"{headline}\n  ได้คำทำนาย {len(got)} ค่า แต่ส่งไป {len(expected)} แถว\n  {advice}",
        )

    # เทียบแบบ element-wise ที่ทนทั้งตัวเลขและข้อความ — `==` บนอาเรย์ object ใช้ได้
    # แต่ NaN ไม่เท่ากับตัวเอง จึงต้องนับ NaN ทั้งคู่ว่า "เหมือนกัน" แยกต่างหาก
    same = expected == got
    if expected.dtype.kind == "f" and got.dtype.kind == "f":
        same = same | (np.isnan(expected) & np.isnan(got))
    bad = np.flatnonzero(~np.asarray(same, dtype=bool))
    if bad.size == 0:
        return

    shown = bad[:EXAMPLES_SHOWN]
    lines = "\n".join(
        f"    แถวที่ {int(row_ids[i])}: ได้ {got[i]!r} ควรเป็น {expected[i]!r}" for i in shown
    )
    more = f"\n    (และอีก {bad.size - shown.size} แถว)" if bad.size > shown.size else ""
    raise _PredictorFailure(
        status, f"{headline} — ไม่ตรง {bad.size} จาก {len(expected)} แถว\n{lines}{more}\n  {advice}"
    )
