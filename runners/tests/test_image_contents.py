"""รายการไฟล์ที่เข้า container — ต้อง**พอ**และต้อง**ไม่เกิน**

เทสต์นี้เกิดตอนย้าย `protocol.py` กับ `launcher.py` ออกจาก `runners/agent_env/`
ไปเป็นของกลางที่ `runners/sandbox/` การย้ายแบบนั้นทำให้ `COPY` ใน Dockerfile ผิดได้
สองทาง และทั้งสองทางเงียบมากบนเครื่องที่ยังไม่มี docker

  · **ขาด** — image build ผ่าน แต่ `arena-agent-host` ตายตอน `import` ทันทีที่รัน
    อาการที่นิสิตเห็นคือ submission ทุกอันล้มพร้อมกันโดยไม่มีใครแก้โค้ดตัวเอง
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
DOCKERFILE = REPO / "runners" / "agent_env" / "images" / "Dockerfile.cpu"

#: ไฟล์ฝั่ง trusted — **ห้ามอยู่ใน image** ไม่ว่าจะด้วยเหตุผลอะไร
#: `runner.py`/`plugin.py` ถือ env กับ seed · `launcher.py` รู้วิธีสั่ง docker
#: `seeds.py` อ่านที่เก็บของลับ · `worker.py` คุยกับคิวและฐานข้อมูล
FORBIDDEN = (
    "runners/agent_env/runner.py",
    "runners/agent_env/plugin.py",
    "runners/agent_env/validate.py",
    "runners/agent_env/sandbox.py",
    "runners/sandbox/launcher.py",
    "runners/seeds.py",
    "runners/worker.py",
)


def copied_runner_files() -> list[str]:
    """path (เทียบจาก root ของ repo) ของไฟล์ใต้ `runners/` ที่ Dockerfile คัดลอกเข้า image"""
    out = []
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^COPY\s+(runners/\S+)\s+\S+\s*$", line.strip())
        if match:
            out.append(match.group(1))
    return out


@pytest.fixture(scope="module")
def copied() -> list[str]:
    files = copied_runner_files()
    assert files, f"อ่าน COPY จาก {DOCKERFILE} ไม่เจอเลย — รูปแบบไฟล์เปลี่ยนไปหรือเปล่า"
    return files


def build_tree(root: Path, files) -> None:
    for rel in files:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)


def import_agent_host(root: Path) -> subprocess.CompletedProcess:
    """สั่ง import `agent_host` โดยเห็นเฉพาะไฟล์ใน `root`

    ต้องถอด root ของ repo ออกจาก `sys.path` เองเพราะ `.venv` ติดตั้งแบบ editable
    ไว้ (`_editable_impl_colosseum.pth`) — ถ้าไม่ถอด การ import จะไปเจอไฟล์จริง
    แล้วเทสต์นี้จะผ่านตลอดกาลแม้ Dockerfile จะขาดไฟล์
    """
    script = (
        "import sys\n"
        f"sys.path[:] = [p for p in sys.path if p not in {{{str(REPO)!r}, ''}}]\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "import runners.agent_env.agent_host as m\n"
        f"assert m.__file__.startswith({str(root)!r}), m.__file__\n"
        "print('OK')\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=120
    )


def test_copy_list_is_enough_to_import_the_host(tmp_path, copied):
    """**ข้อสำคัญที่สุด** — ไฟล์เท่าที่ Dockerfile คัดลอก ต้องพอให้ host เริ่มทำงานได้"""
    build_tree(tmp_path, copied)
    run = import_agent_host(tmp_path)
    assert run.returncode == 0, (
        f"agent_host import ไม่ผ่านด้วยไฟล์ที่ Dockerfile คัดลอก\n"
        f"ไฟล์ที่มี: {copied}\n{run.stderr}"
    )


@pytest.mark.parametrize(
    "dropped", [f for f in copied_runner_files() if not f.endswith("__init__.py")]
)
def test_every_copied_module_is_actually_needed(tmp_path, copied, dropped):
    """ตัดโมดูลไหนออกก็ต้องพัง — พิสูจน์ว่าเทสต์ข้างบน**ล้มได้จริง** ไม่ใช่ผ่านเพราะบังเอิญ

    และพิสูจน์ว่ารายการ `COPY` ไม่มีของเกิน — ของเกินใน image ที่รันโค้ดนิสิต
    คือพื้นที่โจมตีที่ไม่มีใครได้อะไรตอบแทน

    `__init__.py` ไม่เข้าข้อนี้เพราะ Python ยัง import ได้โดยไม่มีมัน (namespace
    package) กฎของมันเป็นอีกข้อหนึ่ง อยู่ที่เทสต์ถัดไป
    """
    build_tree(tmp_path, [f for f in copied if f != dropped])
    run = import_agent_host(tmp_path)
    assert run.returncode != 0, f"ตัด {dropped} ออกแล้วยัง import ผ่าน — บรรทัด COPY นั้นไม่จำเป็น"


def test_every_package_in_the_image_has_its_init(copied):
    """ทุกโฟลเดอร์ที่มีไฟล์เข้า image ต้องมี `__init__.py` เข้าไปด้วย

    ขาดแล้ว import ยังผ่าน — Python จะทำให้มันเป็น **namespace package** แทน
    ซึ่งเป็นของที่ผสมกับโฟลเดอร์ชื่อเดียวกันจากที่อื่นบน `sys.path` ได้
    ในกล่องที่รันโค้ดนิสิตและมี `/submission` อยู่บน path ด้วย นั่นคือช่องที่
    นิสิตวางโฟลเดอร์ชื่อ `runners/` ของตัวเองเข้ามาแทนของจริงได้
    """
    missing = sorted(
        {str(Path(f).parent / "__init__.py") for f in copied} - set(copied)
    )
    assert not missing, f"โฟลเดอร์ใน image ขาด __init__.py: {missing}"


@pytest.mark.parametrize("path", FORBIDDEN)
def test_trusted_files_never_enter_the_image(copied, path):
    assert path not in copied, (
        f"{path} เป็นไฟล์ฝั่ง trusted แต่ถูกคัดลอกเข้า image ของนิสิต — README §10.4"
    )


def test_agent_env_package_init_pulls_nothing_in(copied):
    """`runners/agent_env/__init__.py` ต้องไม่ import อะไร

    แพ็กเกจนี้มีทั้งไฟล์ที่อยู่ในกล่องและไฟล์ฝั่ง trusted การ import ที่ระดับแพ็กเกจ
    จะลากของฝั่ง trusted เข้ากล่องทุกครั้งที่มีใคร import อะไรก็ตามในแพ็กเกจนี้
    """
    source = (REPO / "runners" / "agent_env" / "__init__.py").read_text(encoding="utf-8")
    code = [ln for ln in source.splitlines()
            if ln.strip().startswith(("import ", "from ")) and "__future__" not in ln]
    assert not code, f"__init__.py มี import: {code} — ย้ายไปไฟล์แยก (ดู sandbox.py)"
