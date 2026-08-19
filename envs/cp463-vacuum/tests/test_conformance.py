"""Conformance tests — environment-spec §14

**นี่คือสัญญาที่ทำให้ starter kit กับ grader เป็นสิ่งเดียวกัน** ทั้งสองฝั่งรัน test ชุดนี้ใน CI
ชุดทดสอบและ golden value ของ test #11 เปิดเผยได้ เพราะใช้ seed คนละช่วงกับ public/private
(README §10.4) — นิสิตรันเองได้เพื่อยืนยันว่า environment ในเครื่องตรงกับตัวที่ใช้ตัดสิน
"""

from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from vacuum import load_config
from vacuum.baselines import BASELINES
from vacuum.config import Config, ConfigError, from_dict
from vacuum.env import DOWN, IDLE, LEFT, RIGHT, SUCK, UP, VacuumEnv
from vacuum.generator import DX, DY, generate_layout
from vacuum.observation import SENSOR_ORDER
from vacuum.replay import decode, encode, frames, header_from_env
from vacuum.rollout import agent_config, evaluate, run_episode
from vacuum.scoring import EpisodeStats, episode_score

# config กับ golden ถูกแพ็กไปกับตัวแพ็กเกจ — เทสต์จึงอ้างจากที่เดียวกับที่นิสิตได้ไป
from vacuum.config import CONFIG_DIR

GOLDEN_PATH = CONFIG_DIR.parent / "golden_baselines.json"

# ⚠️ seed ของ conformance test ใช้ย่านของตัวเองแยกต่างหาก — ห้ามทับกับ
# train (1–9999) · public (สุ่มจาก 20000–29999) · private (สุ่มจาก 50000–59999)
# ค่าที่สุ่มได้จริงของ public/private เป็นความลับ อยู่ที่ repo colosseum-hypogeum
TEST_SEEDS = [70001, 70002, 70003, 70004, 70005]


@pytest.fixture(scope="module")
def warmup() -> Config:
    return load_config(CONFIG_DIR / "warmup.yaml")


@pytest.fixture(scope="module")
def main() -> Config:
    return load_config(CONFIG_DIR / "main.yaml")


@pytest.fixture(scope="module")
def final() -> Config:
    return load_config(CONFIG_DIR / "final.yaml")


# ── #1 ──────────────────────────────────────────────────────────────


def test_layout_determinism(main):
    """seed เดียวกัน 100 ครั้ง → obstacle/dirt/sticky/start เหมือนกันทุกบิต"""
    ref = generate_layout(main, 70001)
    for _ in range(100):
        got = generate_layout(main, 70001)
        assert np.array_equal(got.obstacle, ref.obstacle)
        assert np.array_equal(got.dirt0, ref.dirt0)
        assert np.array_equal(got.sticky, ref.sticky)
        assert got.start == ref.start
        assert got.D0 == ref.D0


# ── #2 ──────────────────────────────────────────────────────────────


def test_layout_independent_of_max_steps(main):
    """เปลี่ยน max_steps แล้วผังห้องของ seed เดิมต้องไม่เปลี่ยน (จับการใช้ RNG สายเดียว)"""
    for seed in TEST_SEEDS:
        a = generate_layout(main, seed)
        b = generate_layout(main.replace(**{"episode.max_steps": 137}), seed)
        assert np.array_equal(a.obstacle, b.obstacle)
        assert np.array_equal(a.dirt0, b.dirt0)
        assert np.array_equal(a.sticky, b.sticky)
        assert a.start == b.start


# ── #3 ──────────────────────────────────────────────────────────────


def _reachable(obstacle: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    H, W = obstacle.shape
    seen = np.zeros((H, W), dtype=bool)
    x, y = start
    seen[y, x] = True
    queue = deque([(x, y)])
    while queue:
        cx, cy = queue.popleft()
        for d in range(4):
            nx, ny = cx + DX[d], cy + DY[d]
            if 0 <= nx < W and 0 <= ny < H and not obstacle[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                queue.append((nx, ny))
    return seen


@pytest.mark.parametrize("phase", ["warmup", "main", "final"])
def test_connectivity(phase, request):
    """ทุก dirty cell เดินถึงได้จาก start ด้วย 4-connectivity

    ถ้าข้อนี้พัง coverage จะแตะ 100% ไม่ได้ในบาง seed ไม่ว่า agent จะเก่งแค่ไหน
    → completion_bonus กลายเป็นของที่แจกไม่ได้ และ seed นั้นได้คะแนนต่ำโดยไม่เกี่ยวกับฝีมือ
    """
    config = request.getfixturevalue(phase)
    for seed in TEST_SEEDS:
        layout = generate_layout(config, seed)
        seen = _reachable(layout.obstacle, layout.start)
        assert not (layout.dirt0 & ~seen).any(), f"{phase} seed={seed}: มี dirty cell ที่เดินไปไม่ถึง"
        # ถมช่องที่เข้าไม่ถึงแล้ว → ช่องว่างทั้งหมดต้องเดินถึงได้หมด
        assert int(seen.sum()) == layout.free_count


# ── #4 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("phase", ["warmup", "main", "final"])
def test_dirt_count_exact(phase, request):
    """D0 == max(1, round(dirt_ratio * free_count)) ทุก seed — ตัวหารของ coverage ห้ามแกว่ง"""
    config = request.getfixturevalue(phase)
    for seed in TEST_SEEDS:
        layout = generate_layout(config, seed)
        expected = max(1, int(round(config.room.dirt_ratio * layout.free_count)))
        assert layout.D0 == expected
        assert int(layout.dirt0.sum()) == expected
        # sticky ต้องเป็น subset ของ dirt ตอนเริ่ม และมีจำนวนตายตัวเช่นกัน
        assert not (layout.sticky & ~layout.dirt0).any()
        assert int(layout.sticky.sum()) == int(round(config.dynamics.sticky_dirt * layout.D0))


# ── #5 ──────────────────────────────────────────────────────────────


def test_observation_shapes(warmup, main, final):
    """shape/dtype ของทั้ง 3 โหมดตรงตาม §4 · นอกขอบใน local ต้องเป็น obstacle=1.0"""
    for config, shape in ((warmup, (4, 10, 10)), (main, (3, 5, 5)), (final, (3, 3, 3))):
        env = VacuumEnv(config)
        obs, _ = env.reset(seed=70001)
        assert obs["grid"].shape == shape
        assert obs["grid"].dtype == np.float32
        assert obs["pos"].shape == (2,) and obs["pos"].dtype == np.float32
        assert obs["scalars"].shape == (2,) and obs["scalars"].dtype == np.float32
        assert config.robot.observation == "full" or obs in env.observation_space

    # โหมด sensor: (5, 2) เรียงตาม [current, UP, DOWN, LEFT, RIGHT]
    sensor_cfg = main.replace(**{"robot.observation": "sensor", "robot.observation_window": None})
    env = VacuumEnv(sensor_cfg)
    obs, _ = env.reset(seed=70001)
    assert obs["grid"].shape == (len(SENSOR_ORDER), 2)

    # นอกขอบใน local = กำแพง: บังคับให้หุ่นอยู่มุมซ้ายบน
    local_corner = main.replace(**{"robot.start": "corner", "dynamics.action_noise": 0.0})
    env = VacuumEnv(local_corner)
    obs, _ = env.reset(seed=70001)
    x, y = env.x, env.y
    r = local_corner.robot.observation_window // 2
    for j in range(local_corner.robot.observation_window):
        for i in range(local_corner.robot.observation_window):
            gx, gy = x - r + i, y - r + j
            if not (0 <= gx < local_corner.room.width and 0 <= gy < local_corner.room.height):
                assert obs["grid"][0][j][i] == 1.0, "cell นอกขอบต้องรายงานเป็น obstacle"
                assert obs["grid"][1][j][i] == 0.0


def test_observation_window_must_be_odd(main):
    with pytest.raises(ConfigError, match="เลขคี่"):
        main.replace(**{"robot.observation_window": 4})


# ── #6 ──────────────────────────────────────────────────────────────


def test_slip_tape_alignment(main):
    """สอง policy ที่ต่างกันบน seed เดียวกัน → ค่าสุ่มที่ใช้ที่ timestep t ต้องเป็นค่าเดียวกัน

    นี่คือ common random numbers: ใครเลือกเดินที่ timestep ไหนก็เจอ "ดวง" ก้อนเดียวกัน
    """
    seed = 70002
    env_a, env_b = VacuumEnv(main), VacuumEnv(main)
    env_a.reset(seed=seed)
    env_b.reset(seed=seed)
    assert np.array_equal(env_a.tape.slip, env_b.tape.slip)
    assert np.array_equal(env_a.tape.slip_dir, env_b.tape.slip_dir)

    # policy A เดินตลอด · policy B สลับ SUCK/เดิน → action ต่างกันตั้งแต่ timestep 0
    noise = main.dynamics.action_noise
    for env, policy in ((env_a, lambda t: RIGHT), (env_b, lambda t: SUCK if t % 2 == 0 else RIGHT)):
        env.reset(seed=seed)
        for t in range(40):
            a = policy(t)
            env.step(a)
            slipped = bool(env.events[t][1] & (1 << 2))
            expected = a in (UP, DOWN, LEFT, RIGHT) and env.tape.slip[t] < noise
            assert slipped == expected, f"timestep {t}: การลื่นต้องผูกกับ slip_tape[t] เท่านั้น"


# ── #7 ──────────────────────────────────────────────────────────────


def test_collision_semantics(main):
    """เดินชนกำแพง → ตำแหน่งไม่เปลี่ยน · collisions+1 · t+1"""
    config = main.replace(**{"robot.start": "corner", "dynamics.action_noise": 0.0})
    env = VacuumEnv(config)
    env.reset(seed=70001)
    before = (env.x, env.y)
    _, _, _, _, info = env.step(UP)  # มุมซ้ายบน → เดินขึ้นคือออกนอกขอบ
    assert (env.x, env.y) == before
    assert info["collisions"] == 1
    assert info["t"] == 1


def test_idle_costs_a_timestep_only(main):
    config = main.replace(**{"dynamics.action_noise": 0.0})
    env = VacuumEnv(config)
    env.reset(seed=70001)
    _, _, _, _, info = env.step(IDLE)
    assert info["t"] == 1
    assert info["collisions"] == 0 and info["redundant_sucks"] == 0


# ── #8 ──────────────────────────────────────────────────────────────


def test_sticky_semantics(main):
    """SUCK ครั้งแรกบน sticky → ฝุ่นยังอยู่ · ไม่เพิ่ม redundant_sucks
    ครั้งที่สองสำเร็จแม้เดินออกไปแล้วกลับมา
    """
    config = main.replace(**{"dynamics.action_noise": 0.0, "robot.start": "random"})
    env = VacuumEnv(config)

    # หา seed ที่หุ่นเริ่มบนช่อง sticky พอดี ไม่งั้นวางหุ่นเองไม่ได้ (state เป็นของ env)
    seed = None
    for candidate in range(70000, 70400):
        layout = generate_layout(config, candidate)
        if layout.sticky[layout.start[1], layout.start[0]]:
            seed = candidate
            break
    assert seed is not None, "หา seed ที่หุ่นเริ่มบนช่อง sticky ไม่เจอ"

    env.reset(seed=seed)
    _, _, _, _, info = env.step(SUCK)
    assert info["cleaned"] == 0, "SUCK ครั้งแรกบน sticky ต้องไม่สำเร็จ"
    assert info["sticky_fails"] == 1
    assert info["redundant_sucks"] == 0, "sticky ครั้งแรกห้ามนับเป็น redundant (agent ไม่มีทางรู้)"

    # เดินออกไปแล้วกลับมา — ต้องยังดูดขึ้นในครั้งที่สอง
    x0, y0 = env.x, env.y
    moved = None
    for d, back in ((RIGHT, LEFT), (LEFT, RIGHT), (DOWN, UP), (UP, DOWN)):
        env.step(d)
        if (env.x, env.y) != (x0, y0):
            moved = back
            break
    assert moved is not None
    env.step(moved)
    assert (env.x, env.y) == (x0, y0)
    _, _, _, _, info = env.step(SUCK)
    assert info["cleaned"] == 1, "SUCK ครั้งที่สองบน sticky ต้องสำเร็จเสมอ"


def test_redundant_suck_on_clean_cell(main):
    config = main.replace(**{"dynamics.action_noise": 0.0, "dynamics.sticky_dirt": 0.0})
    env = VacuumEnv(config)
    env.reset(seed=70001)
    env.dirt[env.y, env.x] = False  # บังคับให้ช่องปัจจุบันสะอาด
    _, _, _, _, info = env.step(SUCK)
    assert info["redundant_sucks"] == 1


# ── #9 ──────────────────────────────────────────────────────────────

# trajectory ที่เขียนมือ + คะแนนที่คำนวณด้วยมือ — ครอบคลุมกรณีจบก่อน T และ penalty ชนเพดาน
SCORE_CASES = [
    # (ชื่อ, D0, cleaned_at_t, T, collisions, redundant, คะแนนที่คาด)
    ("จบเร็ว ดูดครบ 2 ช่องใน 2 step", 2, [0, 1, 2], 10, 0, 0, 1.95),
    ("ดูดไม่ครบ จบที่ max_steps", 2, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], 10, 0, 0, 0.30),
    ("penalty ชนเพดาน 0.2", 2, [0] * 11, 10, 8, 10, -0.20),
    ("penalty ไม่ชนเพดาน", 4, [0] * 50 + [4], 100, 2, 5, 1.48),
    ("ดูดครบพอดีที่ timestep สุดท้าย", 1, [0, 0, 0, 0, 0, 1], 5, 0, 0, 1.20),
]


@pytest.mark.parametrize("name,D0,cleaned,T,collisions,redundant,expected", SCORE_CASES)
def test_score_reference(name, D0, cleaned, T, collisions, redundant, expected):
    stats = EpisodeStats(
        D0=D0,
        cleaned_at_t=np.array(cleaned, dtype=np.int64),
        collisions=collisions,
        redundant_sucks=redundant,
    )
    got = episode_score(stats, T)
    assert got.score == pytest.approx(expected, abs=1e-12), name


def test_score_range(main):
    """episode_score อยู่ในช่วง [-0.2, 2.0] เมื่อใช้ค่า default"""
    for name, cls in BASELINES.items():
        result = run_episode(main, cls(agent_config(main)), 70001)
        assert -0.2 - 1e-9 <= result.breakdown.score <= 2.0 + 1e-9, name


# ── #10 ─────────────────────────────────────────────────────────────


def test_replay_roundtrip(main):
    """เล่น replay แล้วสร้าง state ทุกเฟรมได้ตรงกับตอนรันจริง 100%"""
    config = main
    env = VacuumEnv(config)
    obs, _ = env.reset(seed=70003)
    agent = BASELINES["silver"](agent_config(config))
    agent.reset({})

    live = [(env.x, env.y, env.dirt.copy(), env.visited.copy(), 0, 0, 0, 0, 0)]
    for _ in range(300):
        obs, _, term, trunc, info = env.step(agent.act(obs))
        live.append(
            (
                env.x, env.y, env.dirt.copy(), env.visited.copy(),
                info["cleaned"], info["collisions"], info["redundant_sucks"],
                info["sticky_fails"], info["slips"],
            )
        )
        if term or trunc:
            break

    header, events = decode(encode(header_from_env(env), env.events))
    replayed = list(frames(header, events))
    assert len(replayed) == len(live)

    for t, (frame, truth) in enumerate(zip(replayed, live)):
        x, y, dirt, visited, cleaned, collisions, redundant, sticky_fails, slips = truth
        assert frame.pos == (x, y), f"เฟรม {t}: ตำแหน่งไม่ตรง"
        assert np.array_equal(frame.dirt, dirt), f"เฟรม {t}: ผังฝุ่นไม่ตรง"
        assert np.array_equal(frame.visited, visited), f"เฟรม {t}: visited ไม่ตรง"
        assert (frame.cleaned, frame.collisions, frame.redundant_sucks) == (
            cleaned, collisions, redundant,
        ), f"เฟรม {t}: ตัวนับไม่ตรง"
        assert (frame.sticky_fails, frame.slips) == (sticky_fails, slips)


def test_replay_size_budget(main):
    """ขนาด replay ต้องอยู่ในงบที่ README §10.3 คำนวณไว้ (~1–2 KB ต่อ episode)"""
    config = main
    env = VacuumEnv(config)
    obs, _ = env.reset(seed=70004)
    agent = BASELINES["silver"](agent_config(config))
    agent.reset({})
    while True:
        obs, _, term, trunc, _ = env.step(agent.act(obs))
        if term or trunc:
            break
    blob = encode(header_from_env(env), env.events)
    assert len(blob) < 4096, f"replay ใหญ่เกินงบ: {len(blob)} ไบต์"


# ── #11 ─────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_golden_baselines():
    """คะแนน baseline บน seed ชุดคงที่ → ตรงค่าที่บันทึกไว้ (จับ regression ทุกชนิด)

    golden ถูกแยกสองไฟล์ตามว่าโค้ดของ baseline นั้นแจกให้นิสิตหรือไม่

        vacuum/golden_baselines.json                          bronze · silver
        $ARENA_SECRETS/agents/cp463-vacuum/golden_instructor.json   🔒 gold · diamond

    เครื่องที่ไม่มีของลับจะตรวจได้แค่ชุดแรก ซึ่งเป็นสิ่งที่นิสิตตรวจได้เหมือนกัน
    """
    from vacuum.baselines import instructor_agents_path

    checked = 0
    sources = [GOLDEN_PATH]
    if (secret_dir := instructor_agents_path()) is not None:
        sources.append(secret_dir / "golden_instructor.json")

    for path in sources:
        if not path.exists():
            continue
        golden = json.loads(path.read_text(encoding="utf-8"))
        for phase, entry in golden["phases"].items():
            config = load_config(CONFIG_DIR / f"{phase}.yaml")
            assert config.config_hash == entry["config_hash"], (
                f"{phase}: config เปลี่ยนไปจากตอน generate golden — "
                f"ต้องขึ้น env_version และ rejudge ทุก submission"
            )
            for level, expected in entry["scores"].items():
                assert level in BASELINES, f"{level} ไม่มีให้เรียกใช้ในสภาพแวดล้อมนี้"
                score, _ = evaluate(
                    config, BASELINES[level], [int(s) for s in golden["seeds"]]
                )
                assert score.score == pytest.approx(expected["score"], abs=1e-9), f"{phase}/{level}"
                assert score.n_completed == expected["n_completed"], f"{phase}/{level}"
                checked += 1

    assert checked >= 6, "อย่างน้อยต้องตรวจ bronze/silver ครบ 3 phase"


def test_students_cannot_see_instructor_baselines(monkeypatch):
    """โค้ดของ Gold/Diamond ต้องไม่อยู่ในแพ็กเกจที่นิสิตติดตั้ง (README §10.4)

    ถ้าวันหนึ่งมีคนย้ายกลับเข้ามา เทสต์นี้จะฟ้องก่อนที่มันจะหลุดไปกับ wheel
    """
    from vacuum.baselines import PUBLIC_BASELINES, all_baselines

    monkeypatch.delenv("ARENA_SECRETS", raising=False)
    assert sorted(all_baselines()) == ["bronze", "silver"]
    assert sorted(PUBLIC_BASELINES) == ["bronze", "silver"]

    import vacuum.baselines.common as common

        assert not hasattr(common, banned), f"{banned} ไม่ควรอยู่ในแพ็กเกจที่แจกให้นิสิต"


# ── #12 ─────────────────────────────────────────────────────────────


def test_reward_is_always_zero(main):
    """env.step() คืน reward == 0.0 เสมอ — reward เป็นสิ่งที่นิสิตออกแบบเอง"""
    env = VacuumEnv(main)
    obs, _ = env.reset(seed=70001)
    rng = np.random.Generator(np.random.PCG64(0))
    for _ in range(200):
        obs, reward, term, trunc, _ = env.step(int(rng.integers(0, 6)))
        assert reward == 0.0
        if term or trunc:
            break


# ── #13 ─────────────────────────────────────────────────────────────


def test_immune_to_global_rng(main):
    """เรียก np.random.seed()/random.seed() ก่อน reset() → ผังห้องและ noise tape ต้องไม่เปลี่ยน

    ถ้าข้อนี้พัง: agent.py ของนิสิตที่เผลอเรียก np.random.seed(42) (หรือ import
    library ที่ทำแบบนั้นเอง) จะทำให้ทีมนั้นเจอห้องคนละแบบกับทีมอื่น → leaderboard ไร้ความหมาย
    และอาการที่เห็นคือ "คะแนนไม่ตรงกันเฉยๆ" ไม่ใช่ error ทำให้ debug แทบไม่ได้
    """
    env = VacuumEnv(main)
    env.reset(seed=70005)
    ref_layout, ref_tape = env.layout, env.tape.slip.copy()

    for global_seed in (0, 1, 42, 12345):
        np.random.seed(global_seed)
        random.seed(global_seed)
        env2 = VacuumEnv(main)
        env2.reset(seed=70005)
        assert np.array_equal(env2.layout.obstacle, ref_layout.obstacle)
        assert np.array_equal(env2.layout.dirt0, ref_layout.dirt0)
        assert np.array_equal(env2.layout.sticky, ref_layout.sticky)
        assert env2.layout.start == ref_layout.start
        assert np.array_equal(env2.tape.slip, ref_tape)


# ── การตรวจเพิ่มเติมที่ไม่ได้อยู่ใน §14 แต่ป้องกันบั๊กที่แพงมาก ──────────


def test_reset_requires_seed(main):
    with pytest.raises(ValueError, match="seed"):
        VacuumEnv(main).reset()


def test_config_rejects_unknown_field():
    """พิมพ์ชื่อฟิลด์ผิดแล้วระบบเงียบๆ ใช้ค่า default = บั๊กที่หาไม่เจอจนเปิดเทอม"""
    with pytest.raises(ConfigError, match="action_noize"):
        from_dict({"task": "vacuum_gridworld", "dynamics": {"action_noize": 0.1}})


def test_config_hash_changes_with_any_value(main):
    assert main.config_hash != main.replace(**{"dynamics.action_noise": 0.11}).config_hash
    assert main.config_hash != main.replace(**{"episode.max_steps": 1501}).config_hash
    # `phase` เป็นชื่อเรียกของมนุษย์ ไม่มีผลต่อพฤติกรรม → ต้องไม่เปลี่ยน hash
    assert main.config_hash == main.replace(phase="เปลี่ยนชื่อ").config_hash


def test_agent_never_sees_seed_or_layout(main):
    """agent_config และ episode_info ต้องไม่มี seed หรือผังห้อง (template §6)"""
    cfg = agent_config(main)
    assert "seed" not in cfg
    for value in cfg.values():
        assert not isinstance(value, np.ndarray)


def test_episode_order_does_not_affect_scores(main):
    """สลับลำดับ episode แล้วคะแนนต้องไม่เปลี่ยน — จับ state ที่รั่วข้าม episode"""
    seeds = TEST_SEEDS
    forward, _ = evaluate(main, BASELINES["silver"], seeds)
    backward, _ = evaluate(main, BASELINES["silver"], list(reversed(seeds)))
    assert sorted(b.score for b in forward.per_episode) == pytest.approx(
        sorted(b.score for b in backward.per_episode)
    )
