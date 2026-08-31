"""Runner daemon — หยิบงานจากคิวแล้วรันใน sandbox

    claim → โหลด seed จากที่เก็บของลับ → แตก submission → รันใน sandbox → รายงานผล

บนของจริง worker ตัวนี้อยู่บนเครื่อง GPU ในมหาวิทยาลัยและต่อ WebSocket **ออกไปหา** cloud
([README §10.1](../README.md#101-ภาพรวม-hybrid-web-บน-cloud--runner-ในมหาวิทยาลัย))
เวอร์ชันนี้คุยกับคิวในหน่วยความจำโดยตรง — ตรรกะการทำงานเหมือนกันทุกประการ
ต่างแค่ช่องทางที่หยิบงาน

**seed ถูกอ่านที่นี่ ไม่ใช่ที่ API** — ถ้าวันหนึ่งมีคนย้ายการโหลด seed ไปฝั่ง API
ของลับจะเดินทางขึ้น cloud ทันทีโดยไม่มีใครสังเกต
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.domain import Competition, Run, RunKind, RunStatus
from core.queue import JobQueue, LeaseExpired
from core.store import ArtifactStore, Store
from runners.agent_env.runner import RunResult, run_submission
from runners.agent_env.sandbox import SANDBOX as AGENT_ENV_SANDBOX
from runners.agent_env.validate import ENTRY as AGENT_ENV_ENTRY, smoke_test
from runners.prediction.sandbox import SANDBOX as PREDICTION_SANDBOX
from runners.prediction.validate import ENTRY as PREDICTION_ENTRY
from runners.sandbox.launcher import Launcher
from runners.seeds import SecretsUnavailable, expected_config_hash, load_seeds

HEARTBEAT_EVERY = 20.0

#: กล่องของแต่ละชนิดโจทย์ — worker เครื่องเดียวรับงานได้ทุกชนิดที่อยู่ในนี้
#:
#: **แยกตาม task_type ไม่ใช่ตัวเดียวใช้ทุกงาน** เพราะแต่ละชนิดใช้ image คนละใบ
#: ที่มีไลบรารีคนละชุด · การส่งงาน CP462 เข้า image ของ CP463 จะล้มด้วย
#: ImportError ที่ไม่ได้ชี้ไปที่ต้นเหตุเลย
SANDBOXES = {
    "agent_env": AGENT_ENV_SANDBOX,
    "prediction": PREDICTION_SANDBOX,
}

#: ชื่อไฟล์ทางเข้าของแต่ละชนิดโจทย์ — **ต้องตรงกับ `ENTRY` ในตัวตรวจ zip ของชนิดนั้น**
#: `ArtifactStore.extract` ใช้ค่านี้หาโฟลเดอร์ที่จะ mount · ถ้าไม่ตรงกับตัวตรวจ
#: submission ที่ห่อด้วยโฟลเดอร์ชั้นเดียวจะผ่านการตรวจแล้วไปตายตอนรัน
ENTRY_FILES = {
    "agent_env": AGENT_ENV_ENTRY,
    "prediction": PREDICTION_ENTRY,
}

#: run ที่ต้องผ่าน smoke test ก่อน — **ไม่รวม private กับ rejudge** โดยตั้งใจ
#:
#: submission ที่มาถึงรอบ private ผ่าน smoke test ตอนรัน public มาแล้ว การตรวจซ้ำ
#: มีแต่ทางเสีย: ถ้ามันล้มด้วยเหตุบังเอิญ (docker สะดุด) final pick ของทีมนั้นจะถูก
#: ปฏิเสธในรอบตัดเกรด ซึ่งเป็นจังหวะที่แก้ตัวไม่ได้แล้ว
SMOKE_TESTED_KINDS = frozenset({RunKind.PUBLIC, RunKind.DRYRUN})


class ConfigDrift(RuntimeError):
    """config เปลี่ยนไปหลังจากที่ seed ถูก generate — ห้ามรันต่อ"""


class UnknownTaskType(RuntimeError):
    """competition ประกาศ task_type ที่ worker ไม่มี runner ให้"""


@dataclass
class Worker:
    """หนึ่งตัวต่อหนึ่งเลน — เครื่องหนึ่งรันหลายตัวพร้อมกันได้"""

    runner_id: str
    store: Store
    queue: JobQueue
    artifacts: ArtifactStore
    workdir: Path
    #: task_type → launcher · ว่างไว้ = ใช้ `SubprocessLauncher` ของแต่ละชนิด
    #: ซึ่ง ⚠️ **ไม่มี container ห่อ** — dev กับเทสต์เท่านั้น
    launchers: dict[str, Launcher] = field(default_factory=dict)
    lanes: tuple[str, ...] = ("cpu",)
    allow_seed_fallback: bool = False

    def run_once(self) -> Run | None:
        """หยิบงานหนึ่งชิ้นแล้วทำจนจบ — คืน `None` ถ้าคิวว่าง"""
        run = self.queue.claim(self.runner_id, lanes=self.lanes)
        if run is None:
            return None

        stop = threading.Event()
        beat = threading.Thread(target=self._heartbeat, args=(run.id, stop), daemon=True)
        beat.start()
        try:
            self._execute(run)
        except Exception as exc:  # noqa: BLE001 — งานพังต้องไม่ทำให้ worker ตาย
            self._report_failure(run, f"{type(exc).__name__}: {exc}")
        finally:
            stop.set()
            beat.join(timeout=2)
        return run

    def drain(self, limit: int | None = None) -> int:
        """ทำงานจนคิวว่าง — ใช้ในเทสต์และโหมด dev"""
        done = 0
        while (limit is None or done < limit) and self.run_once() is not None:
            done += 1
        return done

    def serve_forever(self, poll: float = 2.0) -> None:
        """โหมด daemon — คิวว่างก็รอแล้วถามใหม่"""
        while True:
            if self.run_once() is None:
                time.sleep(poll)

    # ── ภายใน ───────────────────────────────────────────────────────

    def _heartbeat(self, run_id: str, stop: threading.Event) -> None:
        """ต่ออายุ lease ระหว่างที่งานยังรันอยู่

        ถ้า worker ตายไปเฉยๆ heartbeat หยุดเอง แล้ว lease จะหมดอายุ
        คิวจึงเอางานกลับไปแจกใหม่โดยไม่ต้องมีใครมาเก็บกวาด
        """
        while not stop.wait(HEARTBEAT_EVERY):
            try:
                self.queue.heartbeat(run_id, self.runner_id)
            except (LeaseExpired, KeyError):
                return

    def _launcher_for(self, competition: Competition) -> Launcher:
        sandbox = SANDBOXES.get(competition.task_type)
        if sandbox is None:
            raise UnknownTaskType(
                f"ไม่รู้จัก task_type {competition.task_type!r} — "
                f"ที่ลงทะเบียนไว้คือ {sorted(SANDBOXES)}"
            )
        return self.launchers.get(competition.task_type) or sandbox.local()

    def _execute(self, run: Run) -> None:
        submission = self.store.submissions[run.submission_id]
        competition = self.store.competitions[run.competition_id]

        if competition.task_type == "prediction":
            self._execute_prediction(run, competition, submission)
        else:
            self._execute_agent_env(run, competition, submission)

    # ── โจทย์ทำนาย (CP462) ──────────────────────────────────────────

    def _execute_prediction(self, run: Run, competition: Competition, submission) -> None:
        """ไม่มี seed และไม่มี episode — ของลับคือ **เฉลยของชุดที่ใช้ตัดสิน**

        ซึ่งไม่ได้เดินทางผ่าน worker เลย · env plugin เป็นคนโหลดเมล็ดลับเองตอน
        `load_spec` แล้วเฉลยอยู่ในกระบวนการของ runner จนจบ ไม่เคยเข้ากล่อง
        """
        from runners.prediction.runner import run_submission as run_prediction
        from runners.prediction.validate import smoke_test as prediction_smoke

        launcher = self._launcher_for(competition)
        kind = "private" if run.kind is RunKind.PRIVATE else "public"
        workdir = self.workdir / run.id
        try:
            submission_dir = self.artifacts.extract(
                submission.artifact_url, workdir, entry=ENTRY_FILES["prediction"]
            )

            if run.kind in SMOKE_TESTED_KINDS:
                smoke = prediction_smoke(
                    env_plugin=competition.env_plugin,
                    config_path=competition.config_path,
                    submission_dir=submission_dir,
                    launcher=launcher,
                )
                if not smoke.ok:
                    self._report_failure(run, str(smoke), log=smoke.detail)
                    return

            result = run_prediction(
                env_plugin=competition.env_plugin,
                config_path=competition.config_path,
                submission_dir=submission_dir,
                kind=kind,
                config_overrides=self._overrides(competition, run),
                launcher=launcher,
            )
            self._report_prediction(run, result)
        finally:
            self.artifacts.clear_workdir(workdir)

    def _report_prediction(self, run: Run, result) -> None:
        if not result.ok:
            self._report_failure(run, f"{result.status}: {result.detail}", log=result.log)
            return

        score = result.score
        self.queue.report(
            run.id,
            self.runner_id,
            status=RunStatus.DONE,
            score=score.primary,
            # เสมอกันแล้วตัดสินด้วยขอบล่างของช่วงความเชื่อมั่น — ทีมที่คะแนนนิ่งกว่า
            # ชนะทีมที่บังเอิญได้เท่ากันแต่ช่วงกว้าง
            tiebreak=(score.ci_low,),
            metrics={
                **score.as_dict(),
                "n_rows": result.n_rows,
                "checks": result.checks,
                "seconds": result.seconds,
            },
            config_hash=result.config_hash,
            env_version=result.env_version,
        )
        self.store.record(
            "run.completed", "run", run.id, actor_id=self.runner_id, score=score.primary,
        )

    # ── โจทย์ RL (CP463) ────────────────────────────────────────────

    def _execute_agent_env(self, run: Run, competition: Competition, submission) -> None:
        launcher = self._launcher_for(competition)
        phase = self._phase_name(competition, run)

        try:
            seeds = load_seeds(
                competition_slug=competition.slug,
                phase=phase,
                kind="public" if run.kind is not RunKind.PRIVATE else "private",
                allow_fallback=self.allow_seed_fallback,
            )
        except SecretsUnavailable as exc:
            self._report_failure(run, str(exc))
            return

        if run.kind is RunKind.DRYRUN:
            seeds = seeds[:2]  # dry run ใช้ชุดเล็กเพื่อไม่ให้กินคิว (template §10)

        workdir = self.workdir / run.id
        try:
            submission_dir = self.artifacts.extract(
                submission.artifact_url, workdir, entry=ENTRY_FILES["agent_env"]
            )

            if run.kind in SMOKE_TESTED_KINDS:
                smoke = smoke_test(
                    env_plugin=competition.env_plugin,
                    config_path=competition.config_path,
                    submission_dir=submission_dir,
                    launcher=launcher,
                )
                if not smoke.ok:
                    self._report_failure(run, str(smoke), log=smoke.detail)
                    return

            result = run_submission(
                env_plugin=competition.env_plugin,
                config_path=competition.config_path,
                submission_dir=submission_dir,
                seeds=seeds,
                config_overrides=self._overrides(competition, run),
                replay_dir=self.artifacts.replay_path(run.id),
                launcher=launcher,
            )
            self._check_config_hash(competition, phase, result)
            self._report(run, result)
        finally:
            self.artifacts.clear_workdir(workdir)

    def _phase_name(self, competition: Competition, run: Run) -> str:
        phase = competition.phase_at(run.created_at)
        return phase.name if phase else "main"

    def _overrides(self, competition: Competition, run: Run) -> dict:
        phase = competition.phase_at(run.created_at)
        return dict(phase.config_override) if phase else {}

    def _check_config_hash(self, competition: Competition, phase: str, result: RunResult) -> None:
        expected = expected_config_hash(competition_slug=competition.slug, phase=phase)
        if expected and result.config_hash and expected != result.config_hash:
            raise ConfigDrift(
                f"config ของ phase {phase!r} เปลี่ยนไปหลังจากที่ seed ถูก generate\n"
                f"  seed ผูกกับ {expected}\n"
                f"  ตอนนี้เป็น   {result.config_hash}\n"
                f"คะแนนข้าม config hash ห้ามเอามาเทียบกัน — ต้อง generate seed ใหม่หรือย้อน config"
            )

    def _report(self, run: Run, result: RunResult) -> None:
        if not result.ok:
            self._report_failure(run, f"{result.status}: {result.detail}", log=result.log)
            return

        summary = result.summary
        # ⚠️ **ห้ามส่งค่า seed ข้ามเส้นนี้** — สิ่งที่ report ไปจะไปโผล่ที่ API แล้วถึงมือนิสิต
        # README §10.4 จัดค่า public seed เป็นความลับรองจาก private: รู้แล้ว overfit ได้
        # ทำให้ feedback ระหว่างเทอมเสียคุณค่ากับทุกคน · นิสิตต้องการแค่ "ตอนไหนพัง"
        # ซึ่งเลขลำดับตอบได้เท่ากัน และลำดับคงที่เพราะ seed ถูกรันตามลำดับใน seeds.yaml
        episodes = [
            {
                "episode": i,
                "score": e.breakdown.score,
                "status": e.status,
                "coverage": getattr(e.breakdown, "coverage", None),
                "t_end": getattr(e.breakdown, "t_end", None),
                "replay_bytes": e.replay_bytes,
            }
            for i, e in enumerate(result.episodes, 1)
        ]
        self.queue.report(
            run.id,
            self.runner_id,
            status=RunStatus.DONE,
            score=summary.score,
            tiebreak=summary.tiebreak_key[1:],  # ตัวแรกคือ score เอง
            metrics={
                "n_completed": summary.n_completed,
                "worst_episode": summary.worst_episode,
                "mean_coverage": summary.mean_coverage,
                "sd_across_seeds": summary.sd_across_seeds,
                "seconds": result.seconds,
                "episodes": episodes,
            },
            config_hash=result.config_hash,
            env_version=result.env_version,
        )
        self.store.record(
            "run.completed", "run", run.id,
            actor_id=self.runner_id, score=summary.score, seeds=len(result.episodes),
        )

    def _report_failure(self, run: Run, message: str, log: str = "") -> None:
        try:
            self.queue.report(
                run.id, self.runner_id, status=RunStatus.FAILED, error_message=message,
                metrics={"log": log[-4000:]} if log else {},
            )
        except LeaseExpired:
            return  # งานถูกแจกให้คนอื่นไปแล้ว — ผลของเราไม่มีความหมาย
        self.store.record("run.failed", "run", run.id, actor_id=self.runner_id, error=message[:500])
