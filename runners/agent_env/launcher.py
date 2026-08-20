"""วิธีเปิด process ของ agent — แยกออกจากตัว runner เพื่อให้สลับ sandbox ได้

| launcher | ใช้เมื่อไร | แยก process | sandbox |
|---|---|---|---|
| `SubprocessLauncher` | ทดสอบ · dev · `arena eval` ในเครื่องนิสิต | ✅ | ❌ |
| `DockerLauncher` | **ตัวที่ใช้ตัดสินคะแนนจริง** | ✅ | ✅ |

⚠️ `SubprocessLauncher` แยก process แล้วจึงกัน "agent เอื้อมไปอ่าน state ของ env" ได้จริง
แต่**ไม่ได้กันการเข้าถึงระบบไฟล์หรือเครือข่าย** — ห้ามใช้ตัดสินคะแนนบนเครื่องที่มีของลับอยู่
"""

from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from runners.agent_env.protocol import Channel

MAX_LOG_BYTES = 1024 * 1024  # ตัด log ที่ 1 MB เหมือนกับ template ของ prediction-based


class _Drain(threading.Thread):
    """ดูด stderr ของ agent ออกมาเรื่อยๆ

    **ต้องมี** ไม่ใช่ของประดับ — ถ้าปล่อยให้ pipe ของ stderr เต็ม (64 KB บน Linux)
    process ของนิสิตจะบล็อกที่ `print()` แล้ว runner จะรอ action ที่ไม่มีวันมา
    กลายเป็น timeout ที่อธิบายไม่ได้
    """

    def __init__(self, stream):
        super().__init__(daemon=True)
        self.stream = stream
        self.chunks: list[bytes] = []
        self.size = 0
        self.truncated = False

    def run(self) -> None:
        for chunk in iter(lambda: self.stream.read(4096), b""):
            if self.size < MAX_LOG_BYTES:
                self.chunks.append(chunk[: MAX_LOG_BYTES - self.size])
                self.size += len(chunk)
            else:
                self.truncated = True

    def text(self) -> str:
        out = b"".join(self.chunks).decode("utf-8", errors="replace")
        return out + "\n[... log ถูกตัดที่ 1 MB ...]" if self.truncated else out


@dataclass
class AgentProcess:
    """process ของ agent ที่เปิดอยู่ พร้อมช่องคุยและ log"""

    channel: Channel
    _proc: subprocess.Popen
    _drain: _Drain
    _extra_cleanup: list = field(default_factory=list)

    @property
    def log(self) -> str:
        return self._drain.text()

    def alive(self) -> bool:
        return self._proc.poll() is None

    def close(self, grace: float = 2.0) -> int | None:
        self.channel.close()
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=grace)
        self._drain.join(timeout=grace)
        for fn in self._extra_cleanup:
            fn()
        return self._proc.returncode


class Launcher(Protocol):
    def start(self, submission_dir: Path) -> AgentProcess: ...


def _wrap(proc: subprocess.Popen, cleanup: list | None = None) -> AgentProcess:
    drain = _Drain(proc.stderr)
    drain.start()
    return AgentProcess(
        channel=Channel(reader=proc.stdout, writer=proc.stdin),
        _proc=proc,
        _drain=drain,
        _extra_cleanup=cleanup or [],
    )


@dataclass
class SubprocessLauncher:
    """เปิด agent host เป็น process ลูกด้วย interpreter เดียวกัน — ไม่มี sandbox"""

    python: str = sys.executable

    def start(self, submission_dir: Path) -> AgentProcess:
        proc = subprocess.Popen(
            [self.python, "-m", "runners.agent_env.agent_host", "--submission", str(submission_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        return _wrap(proc)


@dataclass
class DockerLauncher:
    """**ตัวที่ใช้ตัดสินคะแนนจริง** — รัน agent ใน container ที่ถูกตัดสิทธิ์ทุกทาง

    ข้อกำหนดทั้งหมดมาจาก [README §4.1](../../README.md#41-hosted-run-ค่าเริ่มต้น--ใช้กับโจทย์-rl-ของ-cp463)

    | มาตรการ | ธง | กันอะไร |
    |---|---|---|
    | ไม่มีเครือข่าย | `--network none` | ดึงเฉลยออก · โหลด weights เพิ่ม · โจมตีเครื่องอื่น |
    | ไม่ใช่ root | `--user 10001:10001` | หนีออกจาก container ผ่านช่องที่ต้องใช้สิทธิ์ |
    | rootfs เขียนไม่ได้ | `--read-only` | ฝังของไว้ข้าม run |
    | ตัด capability ทั้งหมด | `--cap-drop ALL` | mount · ptrace · ยุ่งกับเครือข่าย |
    | ห้ามยกระดับสิทธิ์ | `--security-opt no-new-privileges` | setuid binary |
    | จำกัด RAM / CPU / process | `--memory` `--cpus` `--pids-limit` | กินทรัพยากรจนเครื่องล่ม (fork bomb) |
    | submission อ่านอย่างเดียว | `-v ...:ro` | แก้ไฟล์ตัวเองระหว่างรัน |
    | `/tmp` เป็น tmpfs ขนาดจำกัด | `--tmpfs` | ต้องมีที่เขียนบ้าง (numpy/torch ใช้) แต่ไม่ให้เขียนลงดิสก์จริง |

    **สิ่งที่ไม่ได้ mount เข้าไปคือสิ่งที่สำคัญที่สุด** — ไม่มี seed, ไม่มีผังห้อง, ไม่มีเฉลย
    ต่อให้หลุดออกจาก container ได้ก็ยังไม่มีอะไรให้อ่าน (README §10.4)
    """

    image: str = "arena/vacuum:cpu"
    memory: str = "8g"
    cpus: str = "1.0"
    pids_limit: int = 256
    tmpfs_size: str = "256m"
    docker: str = "docker"
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def available(cls, image: str = "arena/vacuum:cpu", docker: str = "docker") -> bool:
        """มี docker และมี image อยู่จริงไหม

        ตรวจ image ด้วยไม่ใช่แค่ตัว docker เพราะ daemon ที่รันอยู่แต่ไม่มี image
        จะพังตอน `docker run` ซึ่งเป็นตอนที่ agent ของนิสิตกำลังรอผลอยู่แล้ว
        """
        import shutil

        if shutil.which(docker) is None:
            return False
        try:
            subprocess.run([docker, "image", "inspect", image], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, OSError):
            return False

    def start(self, submission_dir: Path) -> AgentProcess:
        submission = Path(submission_dir).resolve()
        cmd = [
            self.docker, "run", "--rm", "-i",
            "--network", "none",
            "--user", "10001:10001",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", self.memory,
            "--memory-swap", self.memory,  # ไม่ให้ swap ออกไปเกินเพดาน RAM
            "--cpus", self.cpus,
            "--pids-limit", str(self.pids_limit),
            "--tmpfs", f"/tmp:rw,noexec,nosuid,size={self.tmpfs_size}",
            "-v", f"{submission}:/submission:ro",
            *self.extra_args,
            self.image,
            "--submission", "/submission",
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        return _wrap(proc)
