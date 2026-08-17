"""Agent ที่ห่อ policy ที่เทรนแล้ว — implement interface เดียวกับ baseline ทุกตัว

แพลตฟอร์มเห็นแค่ `__init__` / `reset` / `act` เท่านั้น จะเก็บ state อะไรไว้ข้างในก็ได้
ที่นี่เก็บ `MapMemory` ซึ่งเป็นตัวเดียวกับที่ใช้ตอนเทรน — **นี่คือจุดที่ห้ามผิด**
ถ้า feature ตอน inference ต่างจากตอนเทรนแม้แต่นิดเดียว policy จะทำงานผิดแบบเงียบๆ

ใช้กับการทดลองที่ 1 ของ §15

    PPO_MODEL=models/ppo_main.zip python examples/calibrate.py --policy examples.ppo_agent:PPOAgent
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from examples.map_memory import MapMemory

DEFAULT_MODEL = "models/ppo_main.zip"


class PPOAgent:
    def __init__(self, config: dict, model=None, model_path: str | None = None):
        self.memory = MapMemory(config)
        if model is not None:
            self.model = model  # ใช้ตอนวัดระหว่างเทรน (ไม่ต้องเซฟ/โหลด)
        else:
            path = Path(model_path or os.environ.get("PPO_MODEL", DEFAULT_MODEL))
            if not path.exists():
                raise FileNotFoundError(
                    f"หาไฟล์ policy ไม่เจอ: {path}\n"
                    f"เทรนก่อนด้วย `python examples/train_ppo.py --phase main` "
                    f"หรือชี้ที่ไฟล์อื่นผ่านตัวแปร PPO_MODEL"
                )
            from stable_baselines3 import PPO

            self.model = PPO.load(str(path), device="cpu")

    def reset(self, episode_info: dict) -> None:
        self.memory.reset()

    def act(self, observation) -> int:
        self.memory.update(observation)
        features = self.memory.features()
        # deterministic=True เพราะการประเมินต้องทำซ้ำได้ (template §6)
        action, _ = self.model.predict(
            {k: v[None] for k, v in features.items()}, deterministic=True
        )
        return int(np.asarray(action).reshape(-1)[0])
