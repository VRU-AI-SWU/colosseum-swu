"""ฝั่ง **trusted** — ถือ environment, seed และเฉลยไว้ แล้วขับ agent ผ่านโปรโตคอลทีละ step

นี่คือชิ้นที่ทำให้ [README §10.4](../../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries) เป็นจริง

    runner process (trusted)              sandbox (untrusted)
      Env · seed · เฉลย  ──obs──►             Agent.act()
                         ◄─action─

ต่อให้ agent หนีออกจาก sandbox ได้ ก็ยังไม่มีเฉลยให้อ่าน เพราะมันไม่เคยเข้าไปอยู่ในนั้น

**สิ่งเดียวที่เดินทางไปฝั่ง agent คือ observation** — ไม่มี seed, ไม่มีผังห้อง, ไม่มีคะแนน
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from runners.agent_env import plugin as plugin_mod
from runners.agent_env.messages import ACT, ACTION, RESET
from runners.agent_env.sandbox import SANDBOX
from runners.sandbox.launcher import Launcher, SandboxProcess
from runners.sandbox.protocol import (
    CLOSE,
    ERROR,
    HELLO,
    OK,
    PROTOCOL_VERSION,
    READY,
    ProtocolError,
)

DEFAULT_RUN_TIMEOUT_S = 20 * 60  # template §9
HANDSHAKE_TIMEOUT_S = 60.0  # โหลด torch + weights อาจกินเวลาหลายสิบวินาที


@dataclass
class EpisodeOutcome:
    seed: int
    status: str  # ok | agent_error | agent_timeout | invalid_action
    breakdown: Any
    detail: str | None = None
    replay_bytes: int = 0


@dataclass
class RunResult:
    status: str  # ok | agent_init_failed | agent_died | run_timeout | protocol_error
    env_plugin: str
    env_version: str
    config_hash: str
    episodes: list[EpisodeOutcome] = field(default_factory=list)
    summary: Any = None
    log: str = ""
    detail: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class _AgentFailure(Exception):
    """agent ล้มในระดับที่ทำให้ทั้ง run ไปต่อไม่ได้"""

    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def run_submission(
    *,
    env_plugin: str,
    config_path: str | Path,
    submission_dir: str | Path,
    seeds: Iterable[int],
    config_overrides: dict[str, Any] | None = None,
    replay_dir: str | Path | None = None,
    launcher: Launcher | None = None,
    run_timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
) -> RunResult:
    plugin = plugin_mod.resolve(env_plugin)
    config = plugin.apply_overrides(plugin.load_config(str(config_path)), config_overrides or {})
    seeds = list(seeds)
    launcher = launcher or SANDBOX.local()
    step_timeout = plugin.step_timeout_ms(config) / 1000.0

    result = RunResult(
        status="ok",
        env_plugin=env_plugin,
        env_version=plugin.version,
        config_hash=plugin.config_hash(config),
    )

    started = time.monotonic()
    agent = launcher.start(Path(submission_dir))
    try:
        _handshake(agent, plugin.agent_config(config))
        env = plugin.make_env(config)

        for seed in seeds:
            if time.monotonic() - started > run_timeout_s:
                raise _AgentFailure(
                    "run_timeout",
                    f"เกินเวลารวมของ run ({run_timeout_s:.0f} วินาที) ที่ seed {seed} "
                    f"— ทำได้ {len(result.episodes)}/{len(seeds)} episode",
                )
            result.episodes.append(
                _run_episode(agent, plugin, config, env, seed, step_timeout, replay_dir)
            )

        try:
            agent.channel.send(CLOSE)
        except OSError:
            pass
        result.summary = plugin.aggregate([e.breakdown for e in result.episodes])

    except _AgentFailure as exc:
        result.status, result.detail = exc.status, exc.detail
    except (ProtocolError, EOFError) as exc:
        result.status, result.detail = "protocol_error", str(exc)
    finally:
        result.log = agent.log
        agent.close()
        result.seconds = time.monotonic() - started

    return result


def _handshake(agent: SandboxProcess, agent_config: dict[str, Any]) -> None:
    agent.channel.send(HELLO, protocol=PROTOCOL_VERSION, agent_config=agent_config)
    try:
        reply = agent.channel.recv(timeout=HANDSHAKE_TIMEOUT_S)
    except TimeoutError as exc:
        raise _AgentFailure("agent_init_failed", f"agent ไม่ตอบ handshake: {exc}") from exc
    except EOFError as exc:
        raise _AgentFailure(
            "agent_died", f"agent จบไปก่อนจะ handshake เสร็จ — ดู log\n{agent.log[-2000:]}"
        ) from exc

    if reply["t"] == ERROR:
        raise _AgentFailure(
            "agent_init_failed",
            f"สร้าง Agent ไม่สำเร็จ:\n{reply.get('traceback', '(ไม่มี traceback)')}",
        )
    if reply["t"] != READY:
        raise ProtocolError(f"คาดว่าจะได้ {READY!r} — ได้ {reply['t']!r}")


def _run_episode(
    agent: SandboxProcess,
    plugin: plugin_mod.EnvPlugin,
    config: Any,
    env: Any,
    seed: int,
    step_timeout: float,
    replay_dir: str | Path | None,
) -> EpisodeOutcome:
    obs, _info = env.reset(seed=seed)

    # episode_info มีแค่สิ่งที่ประกาศไว้ใน TaskSpec — **ไม่มี seed ไม่มีผังห้อง** (template §6)
    reply = _exchange(agent, RESET, {"episode_info": plugin.agent_config(config)}, OK, step_timeout)
    if isinstance(reply, _Failure):
        return _failed(plugin, config, seed, reply)

    done = False
    while not done:
        reply = _exchange(agent, ACT, {"obs": obs}, ACTION, step_timeout)
        if isinstance(reply, _Failure):
            return _failed(plugin, config, seed, reply)
        try:
            obs, _reward, terminated, truncated, _info = env.step(reply["action"])
        except ValueError as exc:
            # action นอกช่วง/ผิดชนิด — §13 ควรจับได้ตอน validate แล้ว ถ้ามาถึงตรงนี้คือ episode พัง
            return _failed(plugin, config, seed, _Failure("invalid_action", str(exc)))
        done = terminated or truncated

    breakdown = plugin.episode_score(env, config)
    replay_bytes = 0
    if replay_dir is not None:
        path = Path(replay_dir)
        path.mkdir(parents=True, exist_ok=True)
        replay_bytes = plugin.write_replay(str(path / f"{seed}.vrp"), env)
    return EpisodeOutcome(seed=seed, status="ok", breakdown=breakdown, replay_bytes=replay_bytes)


@dataclass
class _Failure:
    """episode นี้พัง แต่ run ไปต่อได้ — ใช้ชนิดของตัวเองไม่ใช่ tuple เพราะ dict ที่สำเร็จ
    ก็เป็น truthy การเช็คด้วย `if reply:` จึงเข้า branch ผิดเสมอ (บั๊กที่เจอตอนเขียนครั้งแรก)"""

    status: str
    detail: str


def _exchange(
    agent: SandboxProcess, kind: str, payload: dict, expect: str, timeout: float
) -> dict | _Failure:
    """ส่งข้อความหนึ่งแล้วรอคำตอบ — คืน dict ถ้าสำเร็จ หรือ `_Failure` ถ้า episode พัง"""
    try:
        agent.channel.send(kind, **payload)
        reply = agent.channel.recv(timeout=timeout)
    except TimeoutError as exc:
        # ⚠️ timeout = **episode ล้มเหลว ไม่ใช่ได้คะแนนน้อยลง** (template §7.3)
        # wall-clock ไม่มีผลต่อคะแนนตามหลัก hardware-independent scoring
        return _Failure("agent_timeout", str(exc))
    except EOFError as exc:
        raise _AgentFailure("agent_died", f"process ของ agent จบไปกลาง run: {exc}") from exc

    if reply["t"] == ERROR:
        if reply.get("fatal"):
            raise _AgentFailure("agent_init_failed", reply.get("traceback", ""))
        return _Failure("agent_error", reply.get("traceback", "(ไม่มี traceback)"))
    if reply["t"] != expect:
        raise ProtocolError(f"คาดว่าจะได้ {expect!r} — ได้ {reply['t']!r}")
    return reply


def _failed(plugin, config, seed: int, failure: _Failure) -> EpisodeOutcome:
    return EpisodeOutcome(
        seed=seed,
        status=failure.status,
        breakdown=plugin.zero_score(config),
        detail=failure.detail,
    )
