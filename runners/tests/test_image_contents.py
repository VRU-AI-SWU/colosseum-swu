"""รายการไฟล์ที่เข้า container — ต้อง**พอ**และต้อง**ไม่เกิน** ทุก image

เทสต์นี้เกิดตอนย้าย `protocol.py` กับ `launcher.py` ออกจาก `runners/agent_env/`
ไปเป็นของกลางที่ `runners/sandbox/` การย้ายแบบนั้นทำให้ `COPY` ใน Dockerfile ผิดได้
สองทาง และทั้งสองทางเงียบมากบนเครื่องที่ยังไม่มี docker

  · **ขาด** — image build ผ่าน แต่ host ตายตอน `import` ทันทีที่รัน อาการที่นิสิตเห็น
    คือ submission ทุกอันล้มพร้อมกันโดยไม่มีใครแก้โค้ดตัวเอง
  · **เกิน** — ไฟล์ฝั่ง trusted หลุดเข้ากล่อง ซึ่งไม่มีอะไรฟ้องเลย

ทั้งสองข้อตรวจได้โดยไม่ต้องมี docker: อ่าน `COPY` จาก Dockerfile จริง แล้ว
ประกอบต้นไม้ไฟล์ตามนั้นในโฟลเดอร์ชั่วคราว แล้วสั่ง import ด้วย path ที่มีแค่นั้น
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent

#: ไฟล์ฝั่ง trusted — **ห้ามอยู่ใน image ไหนก็ตาม**
#: `runner.py`/`plugin.py` ถือ env · เฉลย · seed · `launcher.py` รู้วิธีสั่ง docker
#: `seeds.py` อ่านที่เก็บของลับ · `worker.py` คุยกับคิวและฐานข้อมูล
FORBIDDEN = (
    "runners/agent_env/runner.py",
    "runners/agent_env/plugin.py",
    "runners/agent_env/validate.py",
    "runners/agent_env/sandbox.py",
    "runners/prediction/runner.py",
    "runners/prediction/plugin.py",
    "runners/prediction/sandbox.py",
    "runners/sandbox/launcher.py",
    "runners/seeds.py",
    "runners/worker.py",
)

#: image ที่มีอยู่ · (ชื่อ, Dockerfile, โมดูล host ที่ต้อง import ได้)
IMAGES = [
    ("arena/vacuum:cpu", "runners/agent_env/images/Dockerfile.cpu",
     "runners.agent_env.agent_host"),
    ("arena/tabular:cpu", "runners/prediction/images/Dockerfile.cpu",
     "runners.prediction.predictor_host"),
]
IDS = [name for name, _, _ in IMAGES]


def copied_runner_files(dockerfile: str) -> list[str]:
    """path (เทียบจาก root ของ repo) ของไฟล์ใต้ `runners/` ที่ Dockerfile คัดลอกเข้า image"""
    text = (REPO / dockerfile).read_text(encoding="utf-8")
    out = [
        m.group(1)
        for line in text.splitlines()
        if (m := re.match(r"^COPY\s+(runners/\S+)\s+\S+\s*$", line.strip()))
    ]
    assert out, f"อ่าน COPY จาก {dockerfile} ไม่เจอเลย — รูปแบบไฟล์เปลี่ยนไปหรือเปล่า"
    return out


def build_tree(root: Path, files) -> None:
    for rel in files:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)


def import_host(root: Path, module: str) -> subprocess.CompletedProcess:
    """สั่ง import host module โดยเห็นเฉพาะไฟล์ใน `root`

    ต้องถอด root ของ repo ออกจาก `sys.path` เองเพราะ `.venv` ติดตั้งแบบ editable
    ไว้ (`_editable_impl_colosseum.pth`) — ถ้าไม่ถอด การ import จะไปเจอไฟล์จริง
    แล้วเทสต์นี้จะผ่านตลอดกาลแม้ Dockerfile จะขาดไฟล์
    """
    script = (
        "import sys\n"
        f"sys.path[:] = [p for p in sys.path if p not in {{{str(REPO)!r}, ''}}]\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        f"import {module} as m\n"
        f"assert m.__file__.startswith({str(root)!r}), m.__file__\n"
        "print('OK')\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=120
    )


# ── พอไหม ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,dockerfile,module", IMAGES, ids=IDS)
def test_copy_list_is_enough_to_import_the_host(tmp_path, name, dockerfile, module):
    """**ข้อสำคัญที่สุด** — ไฟล์เท่าที่ Dockerfile คัดลอก ต้องพอให้ host เริ่มทำงานได้"""
    copied = copied_runner_files(dockerfile)
    build_tree(tmp_path, copied)
    run = import_host(tmp_path, module)
    assert run.returncode == 0, (
        f"{name}: {module} import ไม่ผ่านด้วยไฟล์ที่ {dockerfile} คัดลอก\n"
        f"ไฟล์ที่มี: {copied}\n{run.stderr}"
    )


def _droppable() -> list[tuple[str, str, str]]:
    out = []
    for _name, dockerfile, module in IMAGES:
        for path in copied_runner_files(dockerfile):
            if not path.endswith("__init__.py"):
                out.append((dockerfile, module, path))
    return out


@pytest.mark.parametrize(
    "dockerfile,module,dropped", _droppable(),
    ids=[f"{d.split('/')[1]}:{p.split('/')[-1]}" for d, _m, p in _droppable()],
)
def test_every_copied_module_is_actually_needed(tmp_path, dockerfile, module, dropped):
    """ตัดโมดูลไหนออกก็ต้องพัง — พิสูจน์ว่าเทสต์ข้างบน**ล้มได้จริง** ไม่ใช่ผ่านเพราะบังเอิญ

    และพิสูจน์ว่ารายการ `COPY` ไม่มีของเกิน — ของเกินใน image ที่รันโค้ดนิสิต
    คือพื้นที่โจมตีที่ไม่มีใครได้อะไรตอบแทน

    `__init__.py` ไม่เข้าข้อนี้เพราะ Python ยัง import ได้โดยไม่มีมัน (namespace
    package) กฎของมันเป็นอีกข้อหนึ่ง อยู่ที่เทสต์ถัดไป
    """
    build_tree(tmp_path, [f for f in copied_runner_files(dockerfile) if f != dropped])
    run = import_host(tmp_path, module)
    assert run.returncode != 0, f"ตัด {dropped} ออกแล้วยัง import ผ่าน — บรรทัด COPY นั้นไม่จำเป็น"


# ── เกินไหม ────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,dockerfile,module", IMAGES, ids=IDS)
def test_trusted_files_never_enter_the_image(name, dockerfile, module):
    copied = set(copied_runner_files(dockerfile))
    leaked = sorted(copied & set(FORBIDDEN))
    assert not leaked, f"{name}: ไฟล์ฝั่ง trusted ถูกคัดลอกเข้า image ของนิสิต: {leaked} — README §10.4"


@pytest.mark.parametrize("name,dockerfile,module", IMAGES, ids=IDS)
def test_images_do_not_share_the_other_tasks_files(name, dockerfile, module):
    """image ของโจทย์หนึ่งต้องไม่มีไฟล์ของโจทย์อื่นติดไปด้วย

    ไม่ใช่เรื่องความลับ แต่เป็นเรื่องพื้นที่โจมตีและความชัดเจน: ถ้า image ของ CP462
    มี `agent_host` ติดไปด้วย จะมีทางรันโค้ดที่ไม่มีใครตั้งใจให้รันอยู่ในกล่อง
    """
    other = "prediction" if "agent_env" in dockerfile else "agent_env"
    strays = [f for f in copied_runner_files(dockerfile) if f.startswith(f"runners/{other}/")]
    assert not strays, f"{name}: มีไฟล์ของ {other} ติดมาด้วย: {strays}"


@pytest.mark.parametrize("name,dockerfile,module", IMAGES, ids=IDS)
def test_every_package_in_the_image_has_its_init(name, dockerfile, module):
    """ทุกโฟลเดอร์ที่มีไฟล์เข้า image ต้องมี `__init__.py` เข้าไปด้วย

    ขาดแล้ว import ยังผ่าน — Python จะทำให้มันเป็น **namespace package** แทน
    ซึ่งเป็นของที่ผสมกับโฟลเดอร์ชื่อเดียวกันจากที่อื่นบน `sys.path` ได้
    ในกล่องที่รันโค้ดนิสิตและมี `/submission` อยู่บน path ด้วย นั่นคือช่องที่
    นิสิตวางโฟลเดอร์ชื่อ `runners/` ของตัวเองเข้ามาแทนของจริงได้
    """
    copied = copied_runner_files(dockerfile)
    missing = sorted({str(Path(f).parent / "__init__.py") for f in copied} - set(copied))
    assert not missing, f"{name}: โฟลเดอร์ใน image ขาด __init__.py: {missing}"


@pytest.mark.parametrize("name,dockerfile,module", IMAGES, ids=IDS)
def test_package_init_pulls_nothing_in(name, dockerfile, module):
    """`__init__.py` ของแพ็กเกจโจทย์ต้องไม่ import อะไร

    แพ็กเกจนั้นมีทั้งไฟล์ที่อยู่ในกล่องและไฟล์ฝั่ง trusted การ import ที่ระดับแพ็กเกจ
    จะลากของฝั่ง trusted เข้ากล่องทุกครั้งที่มีใคร import อะไรก็ตามในแพ็กเกจนี้
    """
    package = module.rsplit(".", 1)[0].replace(".", "/")
    source = (REPO / package / "__init__.py").read_text(encoding="utf-8")
    code = [ln for ln in source.splitlines()
            if ln.strip().startswith(("import ", "from ")) and "__future__" not in ln]
    assert not code, f"{package}/__init__.py มี import: {code} — ย้ายไปไฟล์แยก (ดู sandbox.py)"


# ── เวอร์ชันในกล่องต้องตรงกับที่นิสิตติดตั้ง ────────────────────────


def test_the_prediction_image_pins_the_same_sklearn_as_the_env_package():
    """**ข้อนี้เฉพาะโจทย์ทำนาย** — นิสิตส่ง pickle ที่ fit แล้ว ไม่ใช่ซอร์ส

    pickle ที่สร้างด้วย scikit-learn คนละ minor โหลดแล้วได้ `InconsistentVersionWarning`
    ในกรณีที่ดี และ `AttributeError` ที่อ่านไม่รู้เรื่องในกรณีที่แย่ · เวอร์ชันใน image
    กับที่นิสิตติดตั้งจึงต้องเป็นตัวเดียวกัน ซึ่งต่างจากโจทย์ RL ที่หลวมกว่านี้ได้
    เพราะที่นั่นเดินทางข้ามเส้นด้วยซอร์สโค้ด
    """
    dockerfile = (REPO / "runners/prediction/images/Dockerfile.cpu").read_text(encoding="utf-8")
    env_toml = (REPO / "envs/cp462-tabular/pyproject.toml").read_text(encoding="utf-8")

    in_image = re.search(r'"scikit-learn==([\d.]+)\.\*"', dockerfile)
    for_students = re.search(r'"scikit-learn==([\d.]+)\.\*"', env_toml)
    assert in_image, "Dockerfile ต้องตรึง scikit-learn ถึงระดับ minor"
    assert for_students, (
        "envs/cp462-tabular/pyproject.toml ต้องตรึง scikit-learn ถึงระดับ minor เหมือน image"
    )
    assert in_image.group(1) == for_students.group(1), (
        f"image ใช้ sklearn {in_image.group(1)} · นิสิตติดตั้ง {for_students.group(1)}\n"
        "  pickle ข้าม minor โหลดไม่ได้ — สองที่นี้ต้องตรงกันเสมอ"
    )
