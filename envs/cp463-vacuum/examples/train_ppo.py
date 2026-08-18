"""เทรน PPO บนโจทย์ vacuum — ตัวอย่างสำหรับ starter kit และตัวที่ใช้ปิดการทดลองที่ 1 ของ §15

    python examples/train_ppo.py --phase main --steps 3000000
    python examples/train_ppo.py --phase warmup --steps 500000 --reward sparse

โครงสร้าง

    VacuumEnv  ──► RewardWrapper ──► FeatureWrapper ──► PPO
    (reward=0)     (นิสิตออกแบบ)     (MapMemory §)      (SB3)

**เทรนบน training seeds เท่านั้น** (1–9999) ซึ่งไม่ทับกับ public/private ที่ใช้ตัดสิน
สุ่ม seed ใหม่ทุก episode → policy ต้องแก้ *distribution ของห้อง* ไม่ใช่ห้องเดียว
(contextual MDP — ดู template §4)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.map_memory import MapMemory, feature_space  # noqa: E402
from examples.reward_wrappers import REWARDS  # noqa: E402
from vacuum import load_config  # noqa: E402
from vacuum.env import VacuumEnv  # noqa: E402
from vacuum.rollout import agent_config  # noqa: E402

from vacuum.config import CONFIG_DIR  # noqa: E402
TRAIN_SEEDS = (1, 9999)  # ย่านที่แจกให้นิสิต — ห้ามแตะ public/private


class FeatureWrapper(gym.Wrapper):
    """แปลง observation ดิบ → feature จากแผนที่สะสม และสุ่ม training seed ให้ทุก episode"""

    def __init__(self, env: VacuumEnv, cfg: dict, seed_rng: np.random.Generator):
        super().__init__(env)
        self.memory = MapMemory(cfg)
        self.seed_rng = seed_rng
        self.observation_space = feature_space()

    def reset(self, *, seed: int | None = None, options=None):
        if seed is None:
            seed = int(self.seed_rng.integers(TRAIN_SEEDS[0], TRAIN_SEEDS[1] + 1))
        obs, info = self.env.reset(seed=seed, options=options)
        self.memory.reset()
        self.memory.update(obs)
        return self.memory.features(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        self.memory.update(obs)
        return self.memory.features(), reward, terminated, truncated, info


def make_env(phase: str, reward: str, rank: int, seed: int, overrides: dict | None = None):
    def _init():
        config = load_config(CONFIG_DIR / f"{phase}.yaml")
        if overrides:
            config = config.replace(**overrides)
        env = VacuumEnv(config)
        env = REWARDS[reward](env)
        rng = np.random.Generator(np.random.PCG64([seed, rank]))
        return FeatureWrapper(env, agent_config(config), rng)

    return _init


# ── curriculum ──────────────────────────────────────────────────────
# ห้องเล็กลงแต่ **โครงสร้างของปัญหาเหมือนเดิมทุกอย่าง** — ยัง local 5×5 · ยังมี
# sensor_noise · ยังต้องสำรวจก่อนถึงจะรู้ว่าฝุ่นอยู่ไหน
#
# ที่ไม่ใช้ config warmup.yaml เป็นด่านแรกเพราะมันเป็น `observation: full`
# → ช่อง unknown เป็น 0 ตลอด agent จะไม่เคยเรียนรู้ที่จะสำรวจเลย แล้ว transfer มา main ไม่ได้
# บทเรียนเดียวกับที่ §11 ตั้งใจให้เกิดกับนิสิต แต่กลับด้าน: ด่านที่ง่ายกว่าต้อง "ง่ายกว่า
# ในเชิงขนาด" ไม่ใช่ "ง่ายกว่าในเชิงชนิดของปัญหา"
CURRICULUM = [
    ({"room.width": 10, "room.height": 10, "episode.max_steps": 400}, 0.25),
    ({}, 0.75),
]


class MapExtractor(BaseFeaturesExtractor):
    """CNN เล็กๆ บนแผนที่ ego-centric + MLP บน scalar

    ใช้ CNN เพราะ **ตำแหน่งสัมพัทธ์สำคัญกว่าตำแหน่งสัมบูรณ์** — "มีฝุ่นอยู่ทางซ้ายสองช่อง"
    เป็นรูปแบบเดียวกันไม่ว่าหุ่นจะยืนที่มุมไหนของห้อง การใช้ MLP บน map ที่แบนแล้ว
    ทำให้ policy ต้องเรียนรูปแบบเดียวกันซ้ำใหม่ทุกตำแหน่ง
    """

    def __init__(self, observation_space, features_dim: int = 288):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space["map"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.map_head = nn.Sequential(nn.Linear(n_flat, 256), nn.ReLU())
        self.scalar_head = nn.Sequential(
            nn.Linear(observation_space["scalars"].shape[0], 32), nn.ReLU()
        )

    def forward(self, obs) -> torch.Tensor:
        return torch.cat([self.map_head(self.cnn(obs["map"])), self.scalar_head(obs["scalars"])], dim=1)


class ScoreCallback(BaseCallback):
    """วัดด้วย **คะแนนจริงของ competition** ระหว่างเทรน ไม่ใช่ reward ที่เราออกแบบเอง

    สำคัญเพราะ reward ที่ขึ้นไม่ได้แปลว่าคะแนนขึ้น — ถ้าสองเส้นนี้แยกทางกันเมื่อไร
    แปลว่า reward design กำลังพา agent ไปผิดทาง (ดู reward_wrappers.py)
    """

    def __init__(self, phase: str, every: int, n_seeds: int = 8, verbose: int = 1):
        super().__init__(verbose)
        self.phase, self.every, self.n_seeds = phase, every, n_seeds
        self._next = every

    def _on_training_start(self) -> None:
        # ตอน resume `num_timesteps` เริ่มที่ค่าของ checkpoint (เช่น 2.2M) ไม่ใช่ 0
        # ถ้าตั้ง _next เป็นค่าคงที่ตอนสร้าง เงื่อนไขจะเป็นจริงทันทีและยิงทุก step
        self._next = self.num_timesteps + self.every

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next = self.num_timesteps + self.every

        from examples.ppo_agent import PPOAgent
        from vacuum.rollout import evaluate

        config = load_config(CONFIG_DIR / f"{self.phase}.yaml")
        agent = PPOAgent(agent_config(config), model=self.model)
        summary, _ = evaluate(config, lambda _cfg: agent, range(70001, 70001 + self.n_seeds))
        self.logger.record("competition/score", summary.score)
        self.logger.record("competition/completion_rate", summary.n_completed / self.n_seeds)
        self.logger.record("competition/coverage", summary.mean_coverage)
        if self.verbose:
            print(
                f"  [{self.num_timesteps:>9,}] คะแนนจริง {summary.score:+.4f} · "
                f"ดูดครบ {summary.n_completed}/{self.n_seeds} · coverage {summary.mean_coverage:.3f}",
                flush=True,
            )
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="main", choices=["warmup", "main", "final"])
    ap.add_argument("--reward", default="shaped", choices=list(REWARDS))
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None, help="ไฟล์ .zip ที่จะเทรนต่อ")
    ap.add_argument("--curriculum", action="store_true",
                    help="เริ่มจากห้องเล็กก่อนแล้วค่อยขยายเป็นขนาดจริง")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01,
                    help="ลดลงตอน fine-tune เพื่อให้ argmax คมขึ้น (การประเมินใช้ argmax)")
    args = ap.parse_args()

    out = Path(args.out or f"models/ppo_{args.phase}")
    out.parent.mkdir(parents=True, exist_ok=True)

    stages = CURRICULUM if args.curriculum else [({}, 1.0)]

    def build(overrides: dict):
        return VecMonitor(
            SubprocVecEnv(
                [make_env(args.phase, args.reward, i, args.seed, overrides) for i in range(args.n_envs)]
            )
        )

    venv = build(stages[0][0])

    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu")
        model.learning_rate = args.lr
        model.ent_coef = args.ent_coef
        model._setup_lr_schedule()
        print(f"เทรนต่อจาก {args.resume} ({model.num_timesteps:,} steps) "
              f"· lr={args.lr} ent_coef={args.ent_coef}")
    else:
        model = PPO(
            "MultiInputPolicy",
            venv,
            policy_kwargs=dict(
                features_extractor_class=MapExtractor,
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
            ),
            n_steps=256,
            batch_size=512,
            n_epochs=4,
            gamma=0.995,  # episode ยาว 1,500 step → ต้องมองไกล
            gae_lambda=0.95,
            ent_coef=0.01,
            learning_rate=3e-4,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=args.seed,
            device="cpu",  # net เล็ก · overhead ของการย้ายข้อมูลไป GPU ไม่คุ้ม
            verbose=1,
            tensorboard_log=str(out.parent / "tb"),
        )

    callbacks = [
        CheckpointCallback(save_freq=max(100_000 // args.n_envs, 1), save_path=str(out.parent),
                           name_prefix=out.name),
        ScoreCallback(args.phase, every=250_000),
    ]

    for i, (overrides, share) in enumerate(stages):
        steps = int(args.steps * share)
        if i > 0:
            venv.close()
            venv = build(overrides)
            model.set_env(venv)
        label = "ขนาดจริง" if not overrides else f"ด่านที่ {i + 1} {overrides}"
        print(f"\n=== {label} · {steps:,} steps ===", flush=True)
        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=(i == 0 and not args.resume),
            callback=callbacks,
            progress_bar=False,
        )
        model.save(str(out))

    print(f"\nบันทึก {out}.zip")


if __name__ == "__main__":
    main()
