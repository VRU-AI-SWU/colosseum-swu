"""กล่องของโจทย์ RL — **ฝั่ง trusted เท่านั้น ห้ามคัดลอกเข้า image**

แยกจาก `__init__.py` เพราะแพ็กเกจนี้มีไฟล์ที่อยู่ใน container ของนิสิตด้วย
(`agent_host.py` · `messages.py`) ถ้า `SANDBOX` อยู่ที่ `__init__.py` การ import
`runners.agent_env.messages` ในกล่องจะลาก `runners/sandbox/launcher.py` — ไฟล์ที่
รู้วิธีสั่ง docker — เข้าไปด้วย
"""

from __future__ import annotations

from runners.sandbox.launcher import Sandbox

SANDBOX = Sandbox(
    image="arena/vacuum:cpu",
    host_module="runners.agent_env.agent_host",
    dockerfile="runners/agent_env/images/Dockerfile.cpu",
)
