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

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Annotated, Callable, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile

from core.domain import (
    PARADIGMS,
    AliasInvalid,
    Competition,
    CourseIdInvalid,
    CourseNameInvalid,
    CompetitionClosed,
    QuotaExceeded,
    ParadigmUnknown,
    PreferredNameInvalid,
    RunKind,
    RunStatus,
    Team,
    TeamNameInvalid,
    TeamSizeInvalid,
    User,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from core.auth import AuthError, GoogleAuth
from core.calendar import CalendarInvalid, as_days
from core.leaderboard import BaselineMark, build, insert_baselines, next_target
from core.service import (
    Arena,
    InviteInvalid,
    NotEnrolled,
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


def _phase_ranges(raw_json: str) -> dict[str, tuple[str, str]]:
    """`{"warmup": ["2026-09-15", "2026-09-30"], ...}` → รูปที่ `set_calendar` รับ

    ใช้ร่วมกันระหว่าง endpoint ที่สร้าง competition กับที่เลื่อนปฏิทิน — สองทางนี้
    รับรูปเดียวกัน ถ้าแปลงคนละที่จะรับคนละรูปโดยไม่มีใครตั้งใจ
    """
    try:
        raw = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"phases ไม่ใช่ JSON ที่อ่านได้: {exc}") from exc
    if not isinstance(raw, dict):
        raise HTTPException(400, "phases ต้องเป็น object ของ ชื่อ phase → [วันเริ่ม, วันจบ]")

    ranges = {}
    for name, pair in raw.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise HTTPException(400, f"{name}: ต้องเป็น [วันเริ่ม, วันจบ] — ได้ {pair!r}")
        ranges[str(name)] = (str(pair[0]), str(pair[1]))
    return ranges


def create_app(
    arena: Arena,
    baselines: Optional[dict[str, list[BaselineMark]]] = None,
    allow_origins: Optional[list] = None,
    google: Optional[GoogleAuth] = None,
    environments: Optional[Callable[[], list[dict]]] = None,
    prepare_config: Optional[Callable[[str, str], dict]] = None,
    #: สามอย่างที่โจทย์ทำนายต้องมี — รับไฟล์ · บอกว่าจะแบ่งออกมายังไง · แจกให้นิสิต
    #: `None` = deployment นี้ไม่มี env ที่รับไฟล์ข้อมูล แล้ว endpoint ที่ต้องใช้
    #: จะตอบ 503 พร้อมบอกว่าขาดอะไร ซึ่งชัดกว่าการที่บริการไม่ยอมเริ่ม
    upload_dataset: Optional[Callable[[str, bytes], dict]] = None,
    preview_config: Optional[Callable[[str, str], dict]] = None,
    student_dataset: Optional[Callable[[str, str], bytes]] = None,
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

    def current_user(authorization: Annotated[Optional[str], Header()] = None) -> User:
        """โทเคนบอกว่าเป็น **ใคร** ไม่ใช่ว่าอยู่ทีมไหน

        เดิมโทเคนเป็นของทีม ซึ่งพังเมื่อมีหลายวิชา (คนหนึ่งอยู่หลายทีม จึงมีหลายโทเคน)
        และทำให้ตอนล็อกอินครั้งแรกยังไม่มีอะไรส่งกลับไปให้หน้าเว็บใช้เข้าวิชา
        """
        token = (authorization or "").removeprefix("Bearer ").strip()
        user = arena.store.user_by_token(token)
        if user is None:
            raise HTTPException(401, "ต้องมี Authorization: Bearer <โทเคนของคุณ>")
        return user

    UserDep = Annotated[User, Depends(current_user)]

    def team_in(user: User, course_id: str) -> Team:
        """ทีมที่ผู้เรียกใช้ทำงานในวิชานั้น — 409 ถ้ายังไม่ได้เข้าวิชา"""
        try:
            return arena.team_for(user=user, course_id=course_id)
        except NotEnrolled as exc:
            raise HTTPException(409, str(exc)) from exc

    def owns(user: User, team_id: str) -> bool:
        """ผู้เรียกอยู่ในทีมนั้นไหม — ใช้แทนการเทียบ `team.id` ตรงๆ

        เดิมเทียบกับทีมที่โทเคนชี้ถึง ซึ่งใช้ได้เพราะโทเคนเป็นของทีม · ตอนนี้โทเคน
        เป็นของคน จึงต้องถามว่า "คนนี้อยู่ในทีมที่เป็นเจ้าของงานชิ้นนั้นหรือเปล่า"
        """
        team = arena.store.teams.get(team_id)
        return team is not None and user.id in team.member_ids

    def team_for_competition(user: User, slug: str) -> tuple[Competition, Team]:
        competition = arena.store.competition_by_slug(slug)
        if competition is None:
            raise HTTPException(404, f"ไม่รู้จัก competition {slug!r}")
        return competition, team_in(user, competition.course_id)

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

        user = arena.sign_in(
            google_sub=identity.sub, email=identity.email, name=identity.name
        )
        # ส่งโทเคน **ของคน** กลับไป — ตอนนี้อาจยังไม่ได้อยู่วิชาไหนเลย
        # หน้าเว็บจะใช้มันเรียก "เข้าวิชาด้วยรหัส" เป็นขั้นถัดไป
        return RedirectResponse(cfg.redirect_back(user.token), 302)

    # ── ตัวตนของผู้เรียกและทีม ──────────────────────────────────────

    @app.get("/api/me")
    def me(user: UserDep):
        """ตัวตนของผู้เรียก + **ทุกวิชาที่อยู่** ไม่ใช่วิชาเดียว

        นิสิตที่เรียนทั้ง AI และ ML มีทีมคนละทีมในสองวิชา แต่ใช้โทเคนอันเดียว
        หน้าเว็บจึงต้องได้รายการมาทั้งหมดเพื่อทำตัวสลับวิชา
        """

        def team_view(team: Team) -> dict:
            course = arena.store.course(team.course_id)
            return {
                "course": {
                    "id": course.id,
                    "name": course.name,
                    "max_team_size": course.max_team_size,
                    # รหัสเข้าวิชาให้เฉพาะผู้สอน**ของวิชานั้น** — เขาคือคนที่ต้องอ่าน
                    # ให้นิสิตฟังในคาบ · ผู้สอนวิชาอื่นไม่มีเหตุผลที่จะได้รหัสนี้ไป
                    "join_code": (
                        course.join_code
                        if arena.can_manage_course(user.email, course.id)
                        else None
                    ),
                },
                "team": {
                    "id": team.id,
                    "name": team.name,
                    "invite_code": team.invite_code,
                    "alias": team.alias,
                    "shown_as": team.display_name(reveal=False),
                    # ผู้สอน**ของวิชานี้**เห็นชื่อจาก Google เสมอ — เขาต้องรู้ว่า
                    # ใครเป็นใครเพื่อตัดเกรด · เพื่อนร่วมชั้นเห็นชื่อที่เจ้าตัวตั้ง
                    "members": [
                        {
                            "name": u.shown_as(
                                reveal=arena.can_manage_course(user.email, course.id)
                            ),
                            "email": u.email,
                        }
                        for uid in team.member_ids
                        if (u := arena.store.users.get(uid)) is not None
                    ],
                    "is_solo": len(team.member_ids) <= 1,
                },
                "competitions": [
                    {
                        "slug": c.slug,
                        "title": c.title,
                        "paradigm": c.paradigm,
                        "is_open": c.is_open(datetime.now(timezone.utc)),
                    }
                    for c in sorted(
                        (k for k in arena.store.competitions.values()
                         if k.course_id == course.id),
                        key=lambda k: k.opens_at,
                    )
                ],
            }

        return {
            "user": {"name": user.name, "email": user.email, "token": user.token},
            # หน้าเว็บใช้ตัวนี้ตัดสินว่าจะโชว์แผงของผู้สอนไหม — **ไม่ใช่ด่านความปลอดภัย**
            # ด่านจริงอยู่ที่ endpoint ซึ่งตรวจซ้ำเสมอ การซ่อนปุ่มเป็นแค่ความสะอาดของ UI
            "is_staff": arena.is_staff(user.email),
            "preferred_name": user.preferred_name,
            # วิชาที่คนนี้จัดการได้ — หน้าเว็บใช้ตัดสินว่าจะแสดงแผงผู้สอนของวิชาไหน
            # ผู้สอนระดับทั้งระบบได้ทุกวิชา ส่วนคนอื่นได้เฉพาะที่ประกาศไว้ใน
            # ARENA_COURSE_STAFF_<COURSE_ID>
            "managed_courses": arena.managed_courses(user.email),
            "enrollments": [
                team_view(arena.store.team_of(user.id, cid))
                for cid in arena.store.courses_of(user.id)
            ],
            "paradigms": [
                {"id": p.id, "name": p.name, "blurb": p.blurb} for p in PARADIGMS.values()
            ],
        }

    @app.post("/api/courses/join")
    def join_course(user: UserDep, code: str = Form(...)):
        """เข้าวิชาด้วยรหัสที่ผู้สอนแจกในคาบ"""
        try:
            team = arena.enroll(user=user, join_code=code)
        except InviteInvalid as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"course_id": team.course_id, "team_id": team.id}

    @app.post("/api/users/rotate-token")
    def rotate_user_token(user: UserDep):
        """ออกโทเคนใหม่ให้ **คนเดียว** ไม่กระทบเพื่อนร่วมทีม

        เดิมโทเคนเป็นของทีม การเปลี่ยนจึงทำให้ทุกคนในทีมต้องตั้งค่าใหม่พร้อมกัน
        ทั้งที่หลุดคนเดียว
        """
        before = user.token[:4]
        arena.rotate_user_token(user=user)
        return {"token": user.token, "previous_prefix": before}

    @app.post("/api/users/display-name")
    def set_display_name(user: UserDep, name: str = Form("")):
        """ตั้งชื่อที่อยากให้เพื่อนร่วมชั้นเรียก — ส่งค่าว่างเพื่อกลับไปใช้ชื่อจาก Google

        เป็นสิทธิ์ของเจ้าตัว ไม่ต้องมีสิทธิ์พิเศษ · ชื่อจาก Google ยังอยู่ครบและ
        ผู้สอนของวิชานั้นเห็นเสมอ
        """
        try:
            updated = arena.set_preferred_name(user=user, raw=name, actor_id=user.id)
        except PreferredNameInvalid as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "preferred_name": updated.preferred_name,
            "shown_as": updated.shown_as(reveal=False),
        }

    @app.post("/api/teams/name")
    def rename_team(user: UserDep, course_id: str = Form(...), name: str = Form("")):
        """ทีมเปลี่ยนชื่อตัวเอง — สิทธิ์ของทีม ไม่ต้องเป็นผู้สอน

        ชื่อเริ่มต้นเป็นชื่อ-นามสกุลของคนที่เข้าวิชาคนแรก ซึ่งอ่านแปลกทันทีที่มี
        เพื่อนเข้ามาร่วม · `team_in` เป็นตัวบังคับว่าเปลี่ยนได้เฉพาะทีมของตัวเอง
        """
        team = team_in(user, course_id)
        try:
            team = arena.rename_team(team=team, raw=name, actor_id=user.id)
        except TeamNameInvalid as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"name": team.name, "shown_as": team.display_name(reveal=False)}

    @app.post("/api/teams/alias")
    def set_alias(user: UserDep, course_id: str = Form(...), alias: str = Form("")):
        """ตั้งชื่อที่จะขึ้นกระดานของทีมตัวเอง — ส่งค่าว่างมาเพื่อกลับไปใช้ชื่อจริง

        ต้องบอกว่าวิชาไหน เพราะคนหนึ่งมีทีมได้หลายทีม (ทีมละวิชา)
        ทีมตั้งของตัวเองเท่านั้น ไม่ต้องมีสิทธิ์พิเศษ — README §6.1 ให้เป็นสิทธิ์ของทีม
        """
        team = team_in(user, course_id)
        try:
            team = arena.set_alias(team=team, raw=alias, actor_id=user.id)
        except AliasInvalid as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"alias": team.alias, "shown_as": team.display_name(reveal=False)}

    @app.post("/api/courses/{course_id}/settings")
    def course_settings(
        course_id: str,
        user: UserDep,
        size: Optional[int] = Form(None),
        name: Optional[str] = Form(None),
    ):
        """ผู้สอนแก้ค่าของวิชา — ส่งมาเฉพาะฟิลด์ที่อยากเปลี่ยน

        **ชื่อวิชาแก้ได้ด้วย** เพราะวิชาที่ migrate มาจาก schema เก่าได้ชื่อเป็น id
        ของเครื่อง (`cp463-1-2026`) ซึ่งนิสิตต้องอ่าน · การไม่มีทางแก้แปลว่า
        ต้องเข้าไปยุ่งกับฐานข้อมูลตรงๆ ซึ่งไม่ใช่สิ่งที่ผู้สอนควรต้องทำ

        ตรวจสิทธิ์ที่นี่เสมอ ไม่พึ่งว่าหน้าเว็บซ่อนปุ่มให้แล้ว — endpoint ยิงตรงได้ด้วย curl
        """
        if course_id not in arena.store.courses:
            raise HTTPException(404, f"ไม่รู้จักวิชา {course_id!r}")
        if not arena.can_manage_course(user.email, course_id):
            raise HTTPException(403, "เฉพาะผู้สอนของวิชานี้เท่านั้นที่แก้ค่าได้")
        try:
            course = arena.update_course(
                course_id=course_id, size=size, name=name, actor_id=user.id
            )
        except (TeamSizeInvalid, CourseNameInvalid) as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"course": {"id": course.id, "name": course.name,
                           "max_team_size": course.max_team_size,
                           "join_code": course.join_code}}

    @app.post("/api/teams/join")
    def join_team(user: UserDep, course_id: str = Form(...), invite_code: str = Form(...)):
        """เข้าทีมเพื่อนด้วยรหัสเชิญ — ภายในวิชาเดียวกันเท่านั้น

        ทีมที่มีหลายคนจะเข้าทีมอื่นไม่ได้ (ต้องแยกออกก่อน) ซึ่งเป็นข้อจำกัดที่ตั้งใจ —
        การย้ายทั้งทีมไปรวมกับอีกทีมควรผ่านผู้สอน
        """
        team = team_in(user, course_id)
        if len(team.member_ids) != 1:
            raise HTTPException(
                409,
                "ทีมที่มีสมาชิกมากกว่าหนึ่งคนย้ายเองไม่ได้ — ติดต่อผู้สอนถ้าต้องการรวมทีม",
            )
        try:
            joined = arena.join_team(
                user=user, invite_code=invite_code, course_id=course_id
            )
        except InviteInvalid as exc:
            raise HTTPException(404, str(exc)) from exc
        except TeamFull as exc:
            raise HTTPException(409, str(exc)) from exc
        # ไม่มีโทเคนให้คืนอีกแล้ว — โทเคนเป็นของคน ไม่เปลี่ยนเมื่อย้ายทีม
        return {"team_id": joined.id, "name": joined.name}

    # ── ส่งงาน ──────────────────────────────────────────────────────

    @app.post("/api/competitions/{slug}/submissions", status_code=201)
    async def create_submission(
        slug: str,
        user: UserDep,
        file: Annotated[UploadFile, File()],
        note: Annotated[str, Form()] = "",
        dry_run: Annotated[bool, Form()] = False,
    ):
        # **หัวใจของการใช้โทเคนอันเดียวได้ทุกวิชา** — competition บอกว่าวิชาไหน
        # แล้วเราหาทีมของคนนี้ในวิชานั้น · นิสิตจึงไม่ต้องสลับโทเคนตามวิชา
        _competition, team = team_for_competition(user, slug)
        try:
            submission, run = arena.submit(
                slug=slug,
                team=team,
                user_id=user.id,
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
    def get_submission(submission_id: str, user: UserDep):
        try:
            status = arena.submission_status(submission_id)
        except KeyError as exc:
            raise HTTPException(404, "ไม่พบ submission") from exc
        if not owns(user, status["submission"].team_id):
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
    def get_episodes(run_id: str, user: UserDep):
        run = arena.queue.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "ไม่พบ run")
        if not owns(user, run.team_id):
            raise HTTPException(403, "ดู run ของทีมอื่นไม่ได้")
        return {
            "run_id": run.id,
            "status": run.status.value,
            "score": run.score,
            "episodes": run.metrics.get("episodes", []),
            "log": run.metrics.get("log", ""),
        }

    @app.post("/api/submissions/{submission_id}/final-pick")
    def final_pick(submission_id: str, user: UserDep, picked: bool = True):
        try:
            existing = arena.store.submissions.get(submission_id)
            if existing is None:
                raise KeyError(submission_id)
            submission = arena.set_final_pick(
                submission_id=submission_id,
                team=team_in(user, arena.store.competitions[existing.competition_id].course_id),
                picked=picked,
                user_id=user.id,
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
    def leaderboard(slug: str, user: UserDep, kind: str = "public"):
        competition = arena.store.competition_by_slug(slug)
        if competition is None:
            raise HTTPException(404, f"ไม่รู้จัก competition {slug!r}")
        # คนที่ยังไม่ได้เข้าวิชานี้ยังดูกระดานได้ แค่ไม่มีแถวไหนเป็น "ของคุณ"
        # การปิดกระดานของวิชาอื่นไม่ได้กันอะไร ในเมื่อคะแนนเป็นข้อมูลสาธารณะอยู่แล้ว
        mine = arena.store.team_of(user.id, competition.course_id)
        my_team_id = mine.id if mine else ""
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
        #
        # **ผูกกับวิชาของ competition นี้ ไม่ใช่ `is_staff` ทั้งระบบ** — ผู้สอนต้องเห็น
        # ชื่อจริงของนิสิต*ตัวเอง*เพื่อตัดเกรด แต่ไม่มีเหตุผลให้เห็นชื่อจริงของนิสิต
        # ในวิชาที่ตัวเองไม่ได้สอน · เดิมใช้ `is_staff` เพราะตอนนั้นมีผู้สอนคนเดียว
        rows = build(runs, arena.store.teams, kind=run_kind,
                     reveal_names=arena.can_manage_course(user.email, competition.course_id))

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
                        "is_you": obj.team_id == my_team_id,
                        "metrics": obj.metrics,
                    }
                )
                for item_kind, obj in insert_baselines(rows, marks)
            ],
            "next_target": (
                lambda m: {"level": m.level, "name": m.label, "score": m.score} if m else None
            )(next_target(rows, my_team_id, marks)),
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
            # หน้าเว็บใช้เลือกคำอธิบายของแต่ละ phase — โจทย์ RL เปลี่ยนกติกาทุก phase
            # ส่วนโจทย์ทำนายไม่เปลี่ยนเลย การใช้ข้อความชุดเดียวกันจะโกหกฝั่งหนึ่งเสมอ
            "task_type": competition.task_type,
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
                    # วันแบบที่คนกรอก (วันจบรวมทั้งวัน) — ฟอร์มของผู้สอนเติมค่าเดิม
                    # จากตรงนี้ · ถ้าให้หน้าเว็บแปลงเอง มันจะเลื่อนไปหนึ่งวันทุกครั้ง
                    # ที่เปิดฟอร์มแล้วกดบันทึกโดยไม่แก้อะไร
                    "first_day": as_days(p)[0],
                    "last_day": as_days(p)[1],
                }
                for p in competition.phases
            ],
        }

    @app.post("/api/courses")
    def create_course(
        user: UserDep,
        # `Form("")` ไม่ใช่ `Form(...)` โดยตั้งใจ — ให้ค่าว่างตกมาถึงตัวตรวจของเรา
        # แล้วผู้สอนได้ข้อความไทยที่บอกวิธีแก้ แทน 422 ของ FastAPI ที่เป็นภาษาอังกฤษ
        # และพูดถึงโครงสร้าง request ซึ่งคนกรอกฟอร์มไม่ได้สนใจ
        course_id: str = Form(""),
        name: str = Form(""),
        size: int = Form(6),
    ):
        """สร้างวิชาใหม่ — **ผู้สอนระดับทั้งระบบเท่านั้น**

        ไม่ใช่ `can_manage_course` เพราะยังไม่มีวิชาให้ผูกสิทธิ์ · และการปล่อยให้
        ใครก็ได้สร้างวิชาแปลว่านิสิตสร้างวิชาของตัวเองแล้วสั่งงานเข้าคิวได้
        (`ARENA_STAFF_EMAILS` ทำหน้าที่เหมือน sudoers — ดู `Arena.staff_emails`)
        """
        if not arena.is_staff(user.email):
            raise HTTPException(403, "เฉพาะผู้สอนระดับทั้งระบบเท่านั้นที่สร้างวิชาใหม่ได้")
        try:
            course = arena.create_course(
                course_id=course_id, name=name, max_team_size=size, actor_id=user.id
            )
        except (CourseIdInvalid, CourseNameInvalid, TeamSizeInvalid) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"course": {"id": course.id, "name": course.name,
                           "max_team_size": course.max_team_size,
                           "join_code": course.join_code}}

    @app.post("/api/competitions")
    def create_competition(
        user: UserDep,
        course_id: str = Form(""),
        slug: str = Form(""),
        env_plugin: str = Form(""),
        config: str = Form(""),
        phases: str = Form("{}"),
        title: str = Form(""),
        paradigm: str = Form(""),
        quota_per_day: int = Form(5),
    ):
        """สร้าง competition ใหม่จากหน้าเว็บ — config เดินทางมาเป็น **เนื้อหา** ไม่ใช่ path

        **ตรวจ config ด้วยตัวโหลดจริงของ environment นั้น** ก่อนบันทึกเสมอ ไม่ใช่
        ตรวจเองซ้ำ — ตัวตรวจที่เขียนแยกจะเพี้ยนจาก `validate()` ของ env แล้วฟอร์ม
        จะรับ config ที่ตอนรันจริงใช้ไม่ได้ ซึ่งไปโผล่ตอนนิสิตส่งงานแล้ว
        """
        if not arena.can_manage_course(user.email, course_id):
            raise HTTPException(403, f"เฉพาะผู้สอนของวิชา {course_id} เท่านั้นที่สร้างโจทย์ได้")
        if prepare_config is None:
            raise HTTPException(503, "deployment นี้ยังไม่ได้ต่อทะเบียน environment")

        try:
            meta = prepare_config(env_plugin, config)
        except Exception as exc:  # noqa: BLE001 — ข้อความมาจาก env จริง อ่านรู้เรื่องกว่า
            raise HTTPException(400, f"config ใช้ไม่ได้: {exc}") from exc

        try:
            competition = arena.create_competition(
                slug=slug,
                course_id=course_id,
                title=title or meta["title"],
                task_type=meta["task_type"],
                env_plugin=env_plugin,
                config_text=config,
                paradigm=paradigm or meta["paradigm"],
                ranges=_phase_ranges(phases),
                quota_per_day=quota_per_day,
                import_whitelist=meta["whitelist"],
                actor_id=user.id,
            )
        except (CourseIdInvalid, CalendarInvalid, ParadigmUnknown) as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"competition": competition_info(competition.slug), "config_hash": meta["config_hash"]}

    @app.get("/api/environments")
    def list_environments(user: UserDep):
        """โจทย์ชนิดไหนที่ deployment นี้สร้าง competition ได้ พร้อมหน้าตาของ config

        หน้าเว็บสร้างฟอร์มจากตรงนี้ **ไม่ใช่เขียนฟอร์มมือต่อ env** — ฟอร์มที่เขียนมือ
        จะเพี้ยนจาก config จริงในวันที่มีคนเพิ่มฟิลด์ แล้วผู้สอนกรอกครบแต่บันทึกไม่ได้

        **เฉพาะคนที่จัดการวิชาได้อย่างน้อยหนึ่งวิชา** — ไม่ใช่ความลับ แต่เป็นข้อมูล
        สำหรับคนที่จะสร้างโจทย์ และมันเปิดเผยชื่อโมดูลบนเครื่อง ซึ่งนิสิตไม่ต้องรู้
        """
        if not arena.managed_courses(user.email):
            raise HTTPException(403, "เฉพาะผู้สอนเท่านั้น")
        # ⚠️ ชื่อฟังก์ชันนี้ต้อง **ไม่ตรงกับ** พารามิเตอร์ `environments` ของ
        # `create_app` — ตั้งชื่อซ้ำเมื่อไร ชื่อในขอบเขตนี้จะไปชี้ที่ตัว endpoint เอง
        # แล้วบรรทัดล่างจะเรียกตัวเองแทนที่จะเรียกของที่ฉีดเข้ามา
        #
        # ฉีดเข้ามาตอน wiring เหมือน `validators` และ `baselines` — `core/api.py`
        # ต้องไม่ import `runners/` หรือ `envs/` ตรงๆ (README §10.5)
        return {"environments": environments() if environments else []}

    # ── ชุดข้อมูลของโจทย์ทำนาย ──────────────────────────────────────
    #
    # ทั้งสามข้างล่างแตะข้อมูลที่นิสิตไม่ควรเห็นทั้งใบ — สิทธิ์จึงตรวจที่ endpoint
    # เสมอ ไม่พึ่งว่าหน้าเว็บซ่อนปุ่มให้แล้ว

    @app.post("/api/datasets")
    async def upload_dataset_file(
        user: UserDep,
        file: Annotated[UploadFile, File()],
        course_id: Annotated[str, Form()] = "",
        env_plugin: Annotated[str, Form()] = "",
    ):
        """ผู้สอนอัปโหลด CSV — คืนรหัสของไฟล์ พร้อมรายชื่อคอลัมน์และค่าที่พบ

        หน้าเว็บใช้คำตอบนี้เติมช่อง "คอลัมน์เฉลย" กับ "คลาสทั้งหมด" ให้ · ผู้สอน
        จึงไม่ต้องพิมพ์ชื่อคอลัมน์จากความจำ ซึ่งพิมพ์ผิดได้และความผิดจะไปโผล่
        ตอนนิสิตส่งงานเข้ามาแล้ว

        **ตรวจก่อนเก็บเสมอ** — ไฟล์ที่ `load_spec` อ่านไม่ได้ต้องไม่เข้าคลัง

        **ผู้สอนของวิชานั้นเท่านั้น** — ไฟล์ที่อัปโหลดกลายเป็นเฉลยของโจทย์
        """
        if not arena.can_manage_course(user.email, course_id):
            raise HTTPException(403, f"เฉพาะผู้สอนของวิชา {course_id} เท่านั้นที่อัปโหลดข้อมูลได้")
        if upload_dataset is None:
            raise HTTPException(503, "deployment นี้ยังไม่ได้ต่อ env ที่รับไฟล์ข้อมูล")

        blob = await file.read()
        try:
            return upload_dataset(env_plugin, blob)
        except Exception as exc:  # noqa: BLE001 — ข้อความมาจาก env จริง อ่านรู้เรื่องกว่า
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/competitions/preview")
    def preview_split(
        user: UserDep,
        course_id: str = Form(""),
        env_plugin: str = Form(""),
        config: str = Form(""),
    ):
        """config ชุดนี้จะแบ่งข้อมูลออกมาหน้าตายังไง — **ก่อน**กดสร้าง

        ผู้สอนกรอกสัดส่วนเป็นตัวเลข 0–1 แต่สิ่งที่เขาตัดสินใจจริงคือจำนวนแถวและ
        การกระจายคลาสในแต่ละกอง · ฟอร์มที่ไม่บอกตัวเลขนั้นคือฟอร์มที่ให้เดา
        """
        if not arena.can_manage_course(user.email, course_id):
            raise HTTPException(403, f"เฉพาะผู้สอนของวิชา {course_id} เท่านั้น")
        if preview_config is None:
            raise HTTPException(503, "deployment นี้ยังไม่ได้ต่อ env ที่รับไฟล์ข้อมูล")

        try:
            return preview_config(env_plugin, config)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/competitions/{slug}/data")
    def download_data(slug: str, user: UserDep):
        """นิสิตดาวน์โหลดกองที่แจก — **ทางออกทางเดียวของข้อมูลจากเซิร์ฟเวอร์**

        ต้องอยู่ในวิชานั้นจริง · ไม่ใช่แค่มีโทเคนที่ใช้ได้ · ไฟล์นี้คือเฉลยของ
        กองที่แจก การปล่อยให้คนนอกวิชาโหลดได้ไม่ได้ทำให้การแข่งพัง แต่มันคือ
        ข้อมูลของวิชาที่ผู้สอนเป็นคนตัดสินใจว่าใครได้เห็น
        """
        competition = arena.store.competition_by_slug(slug)
        if competition is None:
            raise HTTPException(404, f"ไม่รู้จัก competition {slug!r}")
        if student_dataset is None:
            raise HTTPException(503, "deployment นี้ยังไม่ได้ต่อ env ที่รับไฟล์ข้อมูล")

        # ไม่ใช้ `team_in` เพราะมันตอบ 409 ให้เอง — ที่นี่ผู้สอนที่ยังไม่ได้เข้าวิชา
        # ในฐานะนิสิตต้องโหลดได้ด้วย จึงต้องถามสองคำถามแล้วค่อยตัดสิน
        try:
            arena.team_for(user=user, course_id=competition.course_id)
        except NotEnrolled:
            if not arena.can_manage_course(user.email, competition.course_id):
                raise HTTPException(
                    403, f"ต้องเข้าวิชา {competition.course_id} ก่อนถึงจะโหลดข้อมูลได้"
                ) from None

        kind, source = competition.config_source()
        config_text = source if kind == "text" else Path(source).read_text(encoding="utf-8")
        try:
            blob = student_dataset(competition.env_plugin, config_text)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, str(exc)) from exc

        return Response(
            content=blob,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{slug}.csv"'},
        )

    @app.post("/api/competitions/{slug}/calendar")
    def set_calendar(slug: str, user: UserDep, phases: str = Form(...)):
        """ผู้สอนเลื่อนปฏิทินจากหน้าเว็บ

        `phases` เป็น JSON — `{"warmup": ["2026-09-15", "2026-09-30"], ...}`
        วันจบ **รวมทั้งวัน** เหมือนที่ `tools/setup_competition.py` ทำ

        ตรวจสิทธิ์ที่นี่เสมอ ไม่พึ่งว่าหน้าเว็บซ่อนปุ่มให้แล้ว — endpoint ยิงตรงได้ด้วย curl
        **สิทธิ์ผูกกับวิชาของ competition นี้** ไม่ใช่แค่ "เป็นผู้สอนคนใดคนหนึ่ง"
        """
        competition = arena.store.competition_by_slug(slug)
        if competition is None:
            raise HTTPException(404, f"ไม่รู้จัก competition {slug!r}")
        if not arena.can_manage_course(user.email, competition.course_id):
            raise HTTPException(
                403, f"เฉพาะผู้สอนของวิชา {competition.course_id} เท่านั้นที่แก้ปฏิทินได้"
            )

        ranges = _phase_ranges(phases)

        try:
            competition = arena.set_calendar(slug=slug, ranges=ranges, actor_id=user.id)
        except CalendarInvalid as exc:
            raise HTTPException(400, str(exc)) from exc

        return competition_info(slug)

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
