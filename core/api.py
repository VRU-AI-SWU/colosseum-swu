"""REST API — README §13

    POST   /api/competitions/{slug}/submissions   อัพโหลด submission
    GET    /api/competitions/{slug}/leaderboard   ?kind=public|private
    GET    /api/submissions/{id}                  สถานะ + คะแนน
    GET    /api/runs/{id}/episodes                ผลรายตอน + ลิงก์ replay
    POST   /api/submissions/{id}/final-pick       เลือกไปตัดสิน

**auth เป็นของชั่วคราว** — ใช้ team token ผ่าน `Authorization: Bearer <token>`
ของจริงใช้ Google OAuth ของมหาวิทยาลัย (README §11) แต่รูปแบบของ endpoint ไม่เปลี่ยน

⚠️ **process นี้ไม่เคยเห็นค่า seed** — มันรู้แค่ว่า competition ชื่ออะไร
การโหลด seed เกิดที่ worker บนเครื่องในมหาวิทยาลัย (`runners/seeds.py`)
"""

# ⚠️ **ห้ามใส่ `from __future__ import annotations` ในไฟล์นี้**
# มันทำให้ annotation กลายเป็นสตริง แล้ว FastAPI resolve `Depends` ที่ประกาศไว้ใน closure
# ไม่เจอ → ตีเป็น query parameter แทน dependency แล้วทุก endpoint จะตอบ 422
# อาการที่เห็นคือ "Field required: team" ซึ่งไม่ได้ชี้ไปที่สาเหตุเลย

from typing import Annotated, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from core.domain import (
    CompetitionClosed,
    QuotaExceeded,
    RunKind,
    RunStatus,
    Team,
)
from fastapi.middleware.cors import CORSMiddleware

from core.leaderboard import BaselineMark, build, insert_baselines, next_target
from core.service import Arena, SubmissionRejected, TooManyFinalPicks


def create_app(
    arena: Arena,
    baselines: Optional[dict[str, list[BaselineMark]]] = None,
    allow_origins: Optional[list] = None,
) -> FastAPI:
    """`allow_origins` = โดเมนของหน้าเว็บที่เรียก API นี้ได้ (README §10.1)

    ต้องตั้งเพราะหน้าเว็บอยู่บน Cloudflare Pages (`colosseum.vru-ai.com`) ส่วน API
    อยู่บนเครื่องในมหาวิทยาลัยหลัง tunnel (`colosseum-api.vru-ai.com`) — คนละ origin
    เบราว์เซอร์จึงบล็อกทุก request จนกว่าจะมี CORS header

    **ห้ามใส่ `"*"`** — endpoint ทั้งหมดยืนยันตัวตนด้วย `Authorization: Bearer <team token>`
    การเปิดให้ทุกโดเมนเรียกได้ แปลว่าหน้าเว็บใดก็ตามที่นิสิตเปิดอยู่ ยิง request
    ในนามของทีมได้ถ้าดักโทเคนไปได้ · ไม่ส่งมาเลย = ไม่เปิด CORS ให้ใคร ซึ่งถูกต้อง
    สำหรับตอน dev ที่เรียกผ่าน localhost หรือ CLI
    """
    app = FastAPI(title="Arena", version="0.1.0")
    baselines = baselines or {}

    if allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allow_origins),
            allow_credentials=False,  # เราใช้ Bearer header ไม่ใช่ cookie
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    def current_team(authorization: Annotated[Optional[str], Header()] = None) -> Team:
        token = (authorization or "").removeprefix("Bearer ").strip()
        team = arena.store.team_by_token(token)
        if team is None:
            raise HTTPException(401, "ต้องมี Authorization: Bearer <team token>")
        return team

    TeamDep = Annotated[Team, Depends(current_team)]

    # ── ส่งงาน ──────────────────────────────────────────────────────

    @app.post("/api/competitions/{slug}/submissions", status_code=201)
    async def create_submission(
        slug: str,
        team: TeamDep,
        file: Annotated[UploadFile, File()],
        note: Annotated[str, Form()] = "",
        dry_run: Annotated[bool, Form()] = False,
    ):
        try:
            submission, run = arena.submit(
                slug=slug,
                team=team,
                user_id=team.member_ids[0] if team.member_ids else team.id,
                archive=await file.read(),
                note=note,
                dry_run=dry_run,
            )
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except CompetitionClosed as exc:
            raise HTTPException(409, str(exc)) from exc
        except QuotaExceeded as exc:
            raise HTTPException(429, str(exc)) from exc
        except SubmissionRejected as exc:
            # 422 พร้อมรายการปัญหาที่ "บอกวิธีแก้" ตาม §13 ไม่ใช่แค่บอกว่าผิด
            raise HTTPException(
                422,
                detail=[
                    {"code": p.code, "message": p.message, "fix": p.fix} for p in exc.problems
                ],
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

        return {
            "submission_id": submission.id,
            "run_id": run.id,
            "kind": run.kind.value,
            "sha256": submission.artifact_sha256,
            "quota_left": arena.quota_left(slug=slug, team_id=team.id),
            "queue_position": arena.queue.position_of(run.id),
        }

    @app.get("/api/submissions/{submission_id}")
    def get_submission(submission_id: str, team: TeamDep):
        try:
            status = arena.submission_status(submission_id)
        except KeyError as exc:
            raise HTTPException(404, "ไม่พบ submission") from exc
        if status["submission"].team_id != team.id:
            raise HTTPException(403, "ดู submission ของทีมอื่นไม่ได้")

        submission = status["submission"]
        return {
            "id": submission.id,
            "note": submission.note,
            "sha256": submission.artifact_sha256,
            "is_final_pick": submission.is_final_pick,
            "created_at": submission.created_at.isoformat(),
            "queue_position": status["queue_position"],
            "runs": [
                {
                    "id": r.id,
                    "kind": r.kind.value,
                    "status": r.status.value,
                    "score": r.score,
                    "error": r.error_message,
                    "env_version": r.env_version,
                    "config_hash": r.config_hash,
                    "metrics": {k: v for k, v in r.metrics.items() if k != "episodes"},
                }
                for r in status["runs"]
            ],
        }

    @app.get("/api/runs/{run_id}/episodes")
    def get_episodes(run_id: str, team: TeamDep):
        run = arena.queue.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "ไม่พบ run")
        if run.team_id != team.id:
            raise HTTPException(403, "ดู run ของทีมอื่นไม่ได้")
        return {
            "run_id": run.id,
            "status": run.status.value,
            "score": run.score,
            "episodes": run.metrics.get("episodes", []),
            "log": run.metrics.get("log", ""),
        }

    @app.post("/api/submissions/{submission_id}/final-pick")
    def final_pick(submission_id: str, team: TeamDep, picked: bool = True):
        try:
            submission = arena.set_final_pick(
                submission_id=submission_id,
                team=team,
                picked=picked,
                user_id=team.member_ids[0] if team.member_ids else team.id,
            )
        except KeyError as exc:
            raise HTTPException(404, "ไม่พบ submission") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except TooManyFinalPicks as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"submission_id": submission.id, "is_final_pick": submission.is_final_pick}

    # ── leaderboard ─────────────────────────────────────────────────

    @app.get("/api/competitions/{slug}/leaderboard")
    def leaderboard(slug: str, team: TeamDep, kind: str = "public"):
        competition = arena.store.competition_by_slug(slug)
        if competition is None:
            raise HTTPException(404, f"ไม่รู้จัก competition {slug!r}")
        try:
            run_kind = RunKind(kind)
        except ValueError as exc:
            raise HTTPException(400, "kind ต้องเป็น public หรือ private") from exc
        if run_kind not in (RunKind.PUBLIC, RunKind.PRIVATE):
            raise HTTPException(400, "leaderboard มีแค่ public กับ private")

        runs = [r for r in arena.queue.runs.values() if r.competition_id == competition.id]
        marks = baselines.get(slug, [])
        rows = build(runs, arena.store.teams, kind=run_kind)

        return {
            "competition": slug,
            "kind": run_kind.value,
            "rows": [
                (
                    {
                        "type": "baseline",
                        "level": obj.level,
                        "name": obj.label,
                        "score": obj.score,
                    }
                    if item_kind == "baseline"
                    else {
                        "type": "team",
                        "rank": obj.rank,
                        "name": obj.display_name,
                        "score": obj.score,
                        "movement": obj.movement,
                        "is_you": obj.team_id == team.id,
                        "metrics": obj.metrics,
                    }
                )
                for item_kind, obj in insert_baselines(rows, marks)
            ],
            "next_target": (
                lambda m: {"level": m.level, "name": m.label, "score": m.score} if m else None
            )(next_target(rows, team.id, marks)),
        }

    # ── สถานะระบบ ───────────────────────────────────────────────────

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "queue_depth": {lane: arena.queue.depth(lane) for lane in ("cpu", "gpu")},
            "running": sum(
                1 for r in arena.queue.runs.values() if r.status is RunStatus.RUNNING
            ),
        }

    return app
