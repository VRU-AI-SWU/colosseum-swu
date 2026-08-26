"""ยืนยันตัวตนด้วย Google Workspace ของมหาวิทยาลัย — README §11

**สิ่งที่การล็อกอินเปลี่ยน คือ*วิธีที่โทเคนถูกแจก* ไม่ใช่ตัวโทเคน**

`arena submit` รันจาก terminal จึงต้องมี credential ที่อยู่ได้นานเสมอ ไม่ว่าจะ
ล็อกอินด้วยวิธีไหน · ก่อนหน้านี้ผู้สอนต้องส่งโทเคนให้นิสิตทีละคน ต่อไปนิสิตล็อกอิน
แล้วเห็นโทเคนของทีมตัวเองบนหน้าเว็บ

    เบราว์เซอร์ → /auth/google/login → Google → /auth/google/callback
                                                      ↓
                        colosseum.vru-ai.com/#token=... (เก็บลง localStorage)

ใช้ **authorization-code flow ฝั่งเซิร์ฟเวอร์** ไม่ใช่ flow ฝั่งเบราว์เซอร์ เพราะ
เราต้องออกโทเคนของเราเองอยู่แล้ว และ client secret ไม่ควรอยู่ในหน้าเว็บ

ไม่ใช้ cookie/session โดยตั้งใจ — หน้าเว็บกับ API อยู่คนละโดเมน การทำ cookie
ข้าม origin ต้อง `SameSite=None` + `credentials` ใน CORS ซึ่งเพิ่มพื้นที่ผิดพลาด
โดยไม่ได้อะไรเพิ่ม ในเมื่อโทเคนของทีมทำหน้าที่เดียวกันอยู่แล้ว
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
import urllib.parse
from dataclasses import dataclass

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v3/userinfo"

#: state มีอายุสั้น — มันมีไว้กัน CSRF ระหว่างเปลี่ยนหน้า ไม่ใช่ session
STATE_TTL_SECONDS = 600


class AuthError(Exception):
    """ล็อกอินไม่ผ่าน — ข้อความต้องบอกนิสิตได้ว่าต้องทำอะไรต่อ"""


@dataclass(frozen=True)
class GoogleIdentity:
    sub: str
    email: str
    name: str
    hd: str | None  # โดเมนของ Google Workspace ที่บัญชีนี้สังกัด


@dataclass
class GoogleAuth:
    """ตั้งค่าจากตัวแปรแวดล้อม — **client_secret ห้ามอยู่ใน repo**

    `allowed_domain` เป็นการตรวจชั้นที่สอง ชั้นแรกคือการตั้ง OAuth consent screen
    เป็น *Internal* ซึ่งกันคนนอกองค์กรตั้งแต่หน้าล็อกอินของ Google เอง
    ตรวจซ้ำที่นี่เพราะการตั้งค่าใน console เปลี่ยนได้โดยที่โค้ดไม่รู้ตัว และเราไม่อยากให้
    ความปลอดภัยขึ้นกับสิ่งที่มองไม่เห็นจากใน repo
    """

    client_id: str
    client_secret: str
    redirect_uri: str
    #: ที่ที่จะส่งนิสิตกลับไปพร้อมโทเคน
    web_origin: str
    allowed_domain: str = "g.swu.ac.th"
    #: ใช้เซ็น state — สุ่มใหม่ทุกครั้งที่บริการเริ่ม ไม่ต้องเก็บถาวร
    #: ผลคือ state ที่ค้างอยู่ตอน restart ใช้ไม่ได้ ซึ่งถูกต้อง (แค่ล็อกอินใหม่)
    signing_key: bytes = b""

    def __post_init__(self) -> None:
        if not self.signing_key:
            object.__setattr__(self, "signing_key", secrets.token_bytes(32))

    # ── state ────────────────────────────────────────────────────────

    def make_state(self, *, now: float | None = None) -> str:
        """สตริงที่เซ็นไว้ ส่งไป Google แล้วได้กลับมา — กัน CSRF

        เซ็นด้วย HMAC แทนการเก็บใน session เพราะไม่ต้องมี state ฝั่งเซิร์ฟเวอร์
        และใช้ได้แม้มีหลาย process
        """
        payload = json.dumps({"n": secrets.token_urlsafe(8), "t": int(now or time.time())})
        raw = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
        sig = hmac.new(self.signing_key, raw.encode(), hashlib.sha256).hexdigest()[:32]
        return f"{raw}.{sig}"

    def check_state(self, state: str, *, now: float | None = None) -> None:
        raw, _, sig = (state or "").partition(".")
        expected = hmac.new(self.signing_key, raw.encode(), hashlib.sha256).hexdigest()[:32]
        if not raw or not hmac.compare_digest(sig, expected):
            raise AuthError("คำขอล็อกอินไม่ถูกต้อง — กดเข้าสู่ระบบใหม่อีกครั้ง")
        try:
            issued = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))["t"]
        except Exception as exc:  # noqa: BLE001
            raise AuthError("คำขอล็อกอินไม่ถูกต้อง — กดเข้าสู่ระบบใหม่อีกครั้ง") from exc
        if (now or time.time()) - issued > STATE_TTL_SECONDS:
            raise AuthError("คำขอล็อกอินหมดอายุ — กดเข้าสู่ระบบใหม่อีกครั้ง")

    # ── ขั้นตอนของ OAuth ─────────────────────────────────────────────

    def authorize_url(self, state: str) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            # บอก Google ว่ารับเฉพาะโดเมนนี้ — หน้าเลือกบัญชีจะกรองให้เลย
            "hd": self.allowed_domain,
            # ให้เลือกบัญชีทุกครั้ง กันกรณีเครื่องคอมพิวเตอร์ในแล็บที่มีคนล็อกอินค้างไว้
            "prompt": "select_account",
        }
        return f"{AUTH_ENDPOINT}?{urllib.parse.urlencode(params)}"

    def exchange(self, code: str, *, client=None) -> GoogleIdentity:
        """แลก authorization code เป็นตัวตนของผู้ใช้

        อ่านข้อมูลจาก `userinfo` แทนการถอด id_token เอง — ได้ข้อมูลชุดเดียวกัน
        โดยไม่ต้องยุ่งกับการตรวจลายเซ็น JWT และไม่ต้องเพิ่ม dependency
        ทั้งสอง request คุยกับ Google ผ่าน TLS ตรงๆ ไม่ผ่านเบราว์เซอร์
        """
        import httpx

        owns_client = client is None
        client = client or httpx.Client(timeout=15.0)
        try:
            token_res = client.post(
                TOKEN_ENDPOINT,
                data={
                    "code": code,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uri": self.redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            if token_res.status_code != 200:
                raise AuthError(f"แลกโทเคนกับ Google ไม่สำเร็จ ({token_res.status_code})")
            access_token = token_res.json().get("access_token")
            if not access_token:
                raise AuthError("Google ไม่ได้ส่ง access token กลับมา")

            info_res = client.get(
                USERINFO_ENDPOINT, headers={"Authorization": f"Bearer {access_token}"}
            )
            if info_res.status_code != 200:
                raise AuthError(f"อ่านข้อมูลบัญชีจาก Google ไม่สำเร็จ ({info_res.status_code})")
            info = info_res.json()
        finally:
            if owns_client:
                client.close()

        identity = GoogleIdentity(
            sub=info.get("sub", ""),
            email=info.get("email", ""),
            name=info.get("name") or info.get("email", ""),
            hd=info.get("hd"),
        )
        self._require_allowed(identity)
        return identity

    def _require_allowed(self, identity: GoogleIdentity) -> None:
        if not identity.sub or not identity.email:
            raise AuthError("Google ไม่ได้ส่งข้อมูลบัญชีมาครบ")
        domain = identity.hd or identity.email.rpartition("@")[2]
        if domain.lower() != self.allowed_domain.lower():
            raise AuthError(
                f"ต้องใช้บัญชี @{self.allowed_domain} เท่านั้น "
                f"(ที่ใช้อยู่คือ {identity.email}) — ออกจากระบบ Google แล้วเข้าใหม่ด้วยบัญชีมหาวิทยาลัย"
            )

    def redirect_back(self, token: str) -> str:
        """ส่งโทเคนกลับไปหน้าเว็บผ่าน fragment ของ URL

        ใช้ `#` ไม่ใช่ `?` เพราะ fragment **ไม่ถูกส่งไปเซิร์ฟเวอร์** จึงไม่ไปโผล่ใน
        access log ของ Cloudflare หรือของใครก็ตามที่อยู่ระหว่างทาง
        หน้าเว็บอ่านค่าแล้วล้าง fragment ทิ้งทันที
        """
        return f"{self.web_origin}/#token={urllib.parse.quote(token)}"

    def redirect_error(self, message: str) -> str:
        return f"{self.web_origin}/#error={urllib.parse.quote(message)}"
