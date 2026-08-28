"""REST API — README §13

    POST   /api/competitions/{slug}/submissions   อัพโหลด submission
    GET    /api/competitions/{slug}/leaderboard   ?kind=public|private
    GET    /api/competitions/{slug}               ปฏิทิน + phase — **ไม่ต้องล็อกอิน**
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

from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from core.domain import (
    AliasInvalid,
    CompetitionClosed,
    QuotaExceeded,
    RunKind,
    RunStatus,
    Team,
    TeamSizeInvalid,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from core.auth import AuthError, GoogleAuth
from core.leaderboard import BaselineMark, build, insert_baselines, next_target
from core.service import (
    Arena,
    InviteInvalid,
    SubmissionRejected,
    TeamFull,
    TooManyFinalPicks,
)


def _default_course_id(arena: Arena) -> str:
    """วิชาเดียวต่อหนึ่ง deployment ในตอนนี้ — ทีมผูกกับวิชา ไม่ใช่กับ competition

    ถ้าวันหนึ่งมีหลายวิชาบนเครื่องเดียว ตรงนี้ต้องรับ course จาก URL แทนการเดา
    """
    competitions = list(arena.store.competitions.values())
    if not competitions:
        raise HTTPException(503, "ยังไม่มี competition ที่ลงทะเบียนไว้บนเซิร์ฟเวอร์นี้")
    return competitions[0].course_id


def create_app(
    arena: Arena,
    baselines: Optional[dict[str, list[BaselineMark]]] = None,
    allow_origins: Optional[list] = None,
    google: Optional[GoogleAuth] = None,
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

    # ── ล็อกอินด้วย Google ──────────────────────────────────────────
    #
    # ไม่มี cookie ไม่มี session — จบที่การส่งโทเคนของทีมกลับไปให้หน้าเว็บ
    # เพราะ `arena submit` ต้องใช้โทเคนนั้นอยู่แล้ว การมีสองกลไกยืนยันตัวตน
    # แปลว่ามีสองที่ให้พลาด

    def _require_google() -> GoogleAuth:
        if google is None:
            raise HTTPException(
                503,
                "ยังไม่ได้ตั้งค่าการล็อกอินด้วย Google บนเซิร์ฟเวอร์นี้ "
                "(ต้องมี ARENA_GOOGLE_CLIENT_ID / ARENA_GOOGLE_CLIENT_SECRET)",
            )
        return google

    @app.get("/auth/google/login")
    def google_login():
        cfg = _require_google()
        return RedirectResponse(cfg.authorize_url(cfg.make_state()), status_code=302)

    @app.get("/auth/google/callback")
    def google_callback(
        code: Optional[str] = None,
        state: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """ปลายทางที่ Google ส่งกลับมา — จบด้วยการ redirect ไปหน้าเว็บเสมอ

        **ไม่คืน JSON แม้ตอนล้มเหลว** เพราะปลายทางนี้คือเบราว์เซอร์ของนิสิต
        การโชว์ JSON ดิบใส่หน้าคนที่แค่กดปุ่มล็อกอินคือการโยนภาระให้เขาแปลเอง
        """
        cfg = _require_google()
        if error:
            return RedirectResponse(cfg.redirect_error(f"Google ปฏิเสธคำขอ ({error})"), 302)
        if not code or not state:
            return RedirectResponse(cfg.redirect_error("คำขอล็อกอินไม่ครบ"), 302)
        try:
            cfg.check_state(state)
            identity = cfg.exchange(code)
        except AuthError as exc:
            return RedirectResponse(cfg.redirect_error(str(exc)), 302)

        course_id = _default_course_id(arena)
        _user, team = arena.sign_in(
            google_sub=identity.sub,
            email=identity.email,
            name=identity.name,
            course_id=course_id,
        )
        return RedirectResponse(cfg.redirect_back(team.token), 302)

    # ── ตัวตนของผู้เรียกและทีม ──────────────────────────────────────

    @app.get("/api/me")
    def me(team: TeamDep):
        members = [
            {"name": u.name, "email": u.email}
            for uid in team.member_ids
            if (u := arena.store.users.get(uid)) is not None
        ]
        course = arena.store.course(team.course_id)
        return {
            "team": {
                "id": team.id,
                "name": team.name,
                "token": team.token,
                "invite_code": team.invite_code,
                "alias": team.alias,
                "shown_as": team.display_name(reveal=False),
                "members": members,
                "is_solo": len(team.member_ids) <= 1,
            },
            "course": {
                "id": course.id,
                "name": course.name,
                "max_team_size": course.max_team_size,
            },
            # หน้าเว็บใช้ตัวนี้ตัดสินว่าจะโชว์แผงของผู้สอนไหม — **ไม่ใช่ด่านความปลอดภัย**
            # ด่านจริงอยู่ที่ endpoint ซึ่งตรวจซ้ำเสมอ การซ่อนปุ่มเป็นแค่ความสะอาดของ UI
            "is_staff": arena.team_acts_as_staff(team),
        }

    @app.post("/api/teams/alias")
    def set_alias(team: TeamDep, alias: str = Form("")):
        """ตั้งชื่อที่จะขึ้นกระดานของทีมตัวเอง — ส่งค่าว่างมาเพื่อกลับไปใช้ชื่อจริง

        ทีมตั้งของตัวเองเท่านั้น ไม่ต้องมีสิทธิ์พิเศษ — README §6.1 ให้เป็นสิทธิ์ของทีม
        """
        try:
            team = arena.set_alias(
                team=team, raw=alias,
                actor_id=team.member_ids[0] if team.member_ids else None,
            )
        except AliasInvalid as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"alias": team.alias, "shown_as": team.display_name(reveal=False)}

    @app.post("/api/courses/{course_id}/max-team-size")
    def set_max_team_size(course_id: str, team: TeamDep, size: int = Form(...)):
        """ผู้สอนเปลี่ยนขนาดทีมสูงสุดของวิชา

        ตรวจสิทธิ์ที่นี่เสมอ ไม่พึ่งว่าหน้าเว็บซ่อนปุ่มให้แล้ว — endpoint เป็นสิ่งที่
        ยิงตรงได้ด้วย curl และโทเคนของทีมก็อยู่ในมือนิสิตทุกคนอยู่แล้ว
        """
        if not arena.team_acts_as_staff(team):
            raise HTTPException(403, "เฉพาะผู้สอนเท่านั้นที่เปลี่ยนขนาดทีมได้")
        if course_id != team.course_id:
            raise HTTPException(403, "เปลี่ยนค่าของวิชาที่ตัวเองไม่ได้สอนไม่ได้")
        try:
            course = arena.set_max_team_size(
                course_id=course_id,
                size=size,
                actor_id=team.member_ids[0] if team.member_ids else None,
            )
        except TeamSizeInvalid as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"course": {"id": course.id, "name": course.name,
                           "max_team_size": course.max_team_size}}

    @app.post("/api/teams/rotate-token")
    def rotate_token(team: TeamDep):
        """ออกโทเคนใหม่ — ใช้เมื่อโทเคนเดิมหลุด

        ยืนยันด้วยโทเคน*เดิม* ซึ่งฟังดูย้อนแย้งแต่ถูกต้อง: คนที่ยังถือโทเคนอยู่คือ
        เจ้าของโดยนิยาม และถ้ามีคนอื่นถือด้วยก็ยิ่งต้องรีบเปลี่ยน · คนที่กดก่อนได้ก่อน
        ซึ่งฝ่ายที่ถูกตัดออกไปคือฝ่ายที่ต้องไปคุยกับผู้สอน
        """
        before = team.token[:4]
        arena.rotate_token(
            team=team,
            actor_id=team.member_ids[0] if team.member_ids else None,
        )
        return {"token": team.token, "previous_prefix": before}

    @app.post("/api/teams/join")
    def join_team(team: TeamDep, invite_code: str = Form(...)):
        """เข้าทีมเพื่อนด้วยรหัสเชิญ

        ยืนยันตัวตนด้วยโทเคนของทีม*ปัจจุบัน* — ทีมเดี่ยวมีสมาชิกคนเดียวอยู่แล้ว
        จึงรู้ว่าใครเป็นคนกด ส่วนทีมที่มีหลายคนจะเข้าทีมอื่นไม่ได้ (ต้องแยกออกก่อน)
        ซึ่งเป็นข้อจำกัดที่ตั้งใจ — การย้ายทั้งทีมไปรวมกับอีกทีมควรผ่านผู้สอน
        """
        if len(team.member_ids) != 1:
            raise HTTPException(
                409,
                "ทีมที่มีสมาชิกมากกว่าหนึ่งคนย้ายเองไม่ได้ — ติดต่อผู้สอนถ้าต้องการรวมทีม",
            )
        user = arena.store.users.get(team.member_ids[0])
        if user is None:
            raise HTTPException(409, "ทีมนี้ไม่ได้ผูกกับบัญชีที่ล็อกอินด้วย Google")
        try:
            joined = arena.join_team(
                user=user, invite_code=invite_code, course_id=team.course_id
            )
        except InviteInvalid as exc:
            raise HTTPException(404, str(exc)) from exc
        except TeamFull as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"team_id": joined.id, "name": joined.name, "token": joined.token}

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
        # ผู้สอนเห็นชื่อจริงเสมอ (README §6.1) — alias เป็นการลดแรงกดดันระหว่างนิสิต
        # ด้วยกัน ไม่ใช่การซ่อนตัวจากการตรวจ · ก่อนหน้านี้ไม่มีใครเห็นชื่อจริงเลย
        rows = build(runs, arena.store.teams, kind=run_kind,
                     reveal_names=arena.team_acts_as_staff(team))

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

    @app.get("/api/competitions/{slug}")
    def competition_info(slug: str):
        """ปฏิทินของ competition — **ไม่ต้องล็อกอิน**

        กำหนดเวลาเป็นข้อมูลสาธารณะ และนิสิตควรเห็นก่อนล็อกอินด้วย · ที่สำคัญกว่านั้น
        คือ phase **เปลี่ยนกติกาการให้คะแนนระหว่างทาง** — ห้อง 10×10 ไม่มี noise
        ในช่วง Warm-up กลายเป็น 30×30 มี noise และฝุ่นกระจุกตัวในช่วง Final
        คนที่จูน agent บน Warm-up แล้วตื่นมาเจอคะแนนตกฮวบวันที่ 1 ต.ค. จะคิดว่า
        ตัวเองทำพัง ทั้งที่โจทย์เปลี่ยน · ก่อนมี endpoint นี้ วิธีรู้เดียวคือส่งงาน
        แล้วโดน `CompetitionClosed` ซึ่งเป็นการรู้ตอนที่สายแล้ว

        **ไม่ส่ง `config_path` กับ `config_override` ออกไป** — อันแรกเป็น path
        บนเครื่องเซิร์ฟเวอร์ซึ่งไม่ใช่เรื่องของใคร ส่วนอันหลังนิสิตมี YAML อยู่ใน
        wheel อยู่แล้ว ไม่ต้องส่งซ้ำ · ไม่มี seed เกี่ยวข้องเลย

        ส่ง `now` ของเซิร์ฟเวอร์กลับไปด้วย เพื่อให้หน้าเว็บนับถอยหลังโดยไม่ต้อง
        เชื่อนาฬิกาของเครื่องนิสิต ซึ่งตั้งผิดกันบ่อยกว่าที่คิด
        """
        competition = arena.store.competition_by_slug(slug)
        if competition is None:
            raise HTTPException(404, f"ไม่รู้จัก competition {slug!r}")

        now = datetime.now(timezone.utc)
        current = competition.phase_at(now)
        return {
            "slug": competition.slug,
            "title": competition.title,
            "now": now.isoformat(),
            "opens_at": competition.opens_at.isoformat(),
            "closes_at": competition.closes_at.isoformat(),
            "is_open": competition.is_open(now),
            "quota_per_day": competition.quota_per_day,
            "max_final_submissions": competition.max_final_submissions,
            "current_phase": current.name if current else None,
            "phases": [
                {
                    "name": p.name,
                    "starts_at": p.starts_at.isoformat(),
                    "ends_at": p.ends_at.isoformat(),
                }
                for p in competition.phases
            ],
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
