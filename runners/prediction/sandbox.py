"""กล่องของโจทย์ทำนาย — **ฝั่ง trusted เท่านั้น ห้ามคัดลอกเข้า image**

แยกจาก `__init__.py` ด้วยเหตุผลเดียวกับ `runners/agent_env/sandbox.py`
"""

from __future__ import annotations

from runners.sandbox.launcher import Sandbox

SANDBOX = Sandbox(
    image="arena/tabular:cpu",
    host_module="runners.prediction.predictor_host",
    dockerfile="runners/prediction/images/Dockerfile.cpu",
)
