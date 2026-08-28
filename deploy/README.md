# ติดตั้งเครื่อง runner ตั้งแต่ศูนย์

เครื่องเดียวรันทุกอย่าง — API, worker, sandbox, ฐานข้อมูล และเป็นที่เดียวที่ของลับอยู่
([README §10.4](../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries))

```
nvme0n1 → /                          nvme1n1 → /media/ratchainant/hdd
~/VRU-AI/projects/colosseum/         $HDD/backup/
├── app/       โค้ด + .venv           ├── db/        สำเนา arena.db ย้อนหลัง 14 วัน
├── secrets/   🔒 hypogeum            └── artifacts/ mirror ของ artifacts
├── data/      arena.db (+ -wal)
└── data/artifacts/  zip + replay
```

> ⚠️ **ดิสก์ที่ mount ที่ `/media/ratchainant/hdd` เป็น NVMe SSD 2TB ไม่ใช่จานหมุน**
> ชื่อ mountpoint ทำให้เข้าใจผิด · และมันไม่ใช่ดิสก์เปล่า — ใช้ไปแล้ว 85% (เหลือ 276 GB)
> ซึ่งยังพอสำหรับ backup ของเรา แต่ต้องรู้ไว้ว่ามีของอื่นอยู่

**ระบบทั้งหมดอยู่บน SSD · HDD เป็นที่สำรองอย่างเดียว**

เหตุผลที่ไม่ย้ายของร้อนไป HDD: ความเสี่ยงที่ใหญ่กว่า "พื้นที่จะเต็ม" คือ **ตอนนี้
ไม่มีสำเนาอะไรเลย** ถ้า SSD พังกลางเทอม คะแนนทั้งเทอม submission ทุกชิ้น และ
audit trail หายพร้อมกัน กู้ไม่ได้ · ดิสก์ลูกที่สองมีค่ากับเรามากกว่าในฐานะที่สำรอง

ส่วนพื้นที่: artifacts โตช้ากว่าที่กลัว — replay 120 KB ต่อ run และ zip ถูก dedup
ด้วย sha256 อยู่แล้ว ทั้งเทอมประมาณหลัก GB เดียว ถ้าวันหนึ่งไม่พอจริงค่อยซื้อ SSD
เพิ่ม หรือย้ายเฉพาะ artifacts ด้วย `--artifacts` ซึ่งรองรับไว้แล้ว

### พื้นที่ — วัดของจริงแล้วเป็นแบบนี้

วัดจากเครื่อง dev ที่ใช้งานมาไม่กี่วัน

| | ขนาดจริง |
|---|---|
| **Docker บนเครื่องจริง** | **24.9 GB** — เป็น image ของ*งานอื่น*บนเครื่องนี้ ลบได้ 21.9 GB · build cache แค่ 2.7 GB |
| `.venv` | 64 MB |
| image `arena/vacuum:cpu` | 365 MB |
| replay ต่อหนึ่ง run (30 episode) | 120 KB |
| zip ของ agent ที่ไม่มี weights | 8 KB |
| `arena.db` | 4 KB |

Docker เก็บของไว้ที่ `/var/lib/docker` บน **SSD ของระบบ** ซึ่งอยู่นอกการแบ่งที่วางไว้
และมันโตทุกครั้งที่ `docker build` — ถ้าไม่จัดการ มันจะเต็มก่อน artifacts หลายเท่า

**แต่ 11.3 GB ในนั้นเป็น build cache ซึ่งเป็นขยะสะสม ไม่ใช่สภาพนิ่ง** — เครื่องนี้
build ซ้ำหลายรอบตอนพัฒนา image ส่วนเครื่อง production build ราว 2–3 ครั้งทั้งเทอม
สภาพนิ่งจริงคือ base image + `arena/vacuum:cpu` ≈ 1–2 GB

ทางแก้จึงเป็นบรรทัดเดียว ไม่ต้องย้าย Docker ไปไหน

```bash
docker builder prune -af
```

⛔ **ห้าม `docker system prune -a` บนเครื่องนี้** — เครื่องนี้ใช้ร่วมกับงานอื่น
คำสั่งนั้นจะลบ image ของงานอื่นที่ไม่ได้รันอยู่ตอนนั้นไปด้วย (21.9 GB ที่ "ลบได้"
ส่วนใหญ่คือของคนอื่น) · `arena/vacuum:cpu` เพิ่มแค่ 365 MB

**ตั้งใจไม่ย้าย `data-root`** — ทั้งสองลูกเป็น NVMe เหมือนกัน ย้ายไปก็ไม่ได้อะไรเพิ่ม
แต่ได้จุดพังเพิ่มมาหนึ่งจุด (mount order) และถ้าดิสก์ปลายทางมีปัญหา ระบบให้คะแนน
หยุดทั้งหมด ไม่ใช่แค่ข้อมูลหาย · Arena ใช้พื้นที่รวมทั้งเทอมราว 5–10 GB ซึ่ง 191 GB
ที่ว่างบน `/` รับได้สบาย

> เขียนสำหรับ **Linux Mint** (ฐาน Ubuntu) ซึ่งเป็นเครื่องที่ใช้จริง · เครื่อง dev บน
> macOS ใช้คำสั่งเดียวกันได้ยกเว้นส่วน systemd กับ `apt`
>
> **ห้ามตั้ง tunnel บนเครื่อง dev** — กุญแจของ tunnel จะกระจายไปอยู่หลายเครื่อง
> โดยไม่จำเป็น

---

## 1. ของที่ต้องมีก่อน

```bash
sudo apt update && sudo apt install -y git docker.io
```

### Python — 3.11 ถึง 3.13 ใช้ได้ทั้งหมด

**สิ่งที่เป็น load-bearing คือเวอร์ชัน `numpy` ไม่ใช่เวอร์ชัน Python** —
`numpy.random.Generator` ไม่การันตี stream ข้ามเวอร์ชันของ *numpy* จึงตรึง `==2.1.*` ไว้
แต่ PCG64 เขียนด้วย C อยู่ใน numpy ตัว Python ไม่เกี่ยว

วัดแล้วบน 3.11 / 3.12 / 3.13 ได้ผังห้องแฮชเดียวกัน คะแนนตรงกันถึงทศนิยมที่ 12
และ conformance 32 ข้อผ่านทั้งสามเวอร์ชัน · Mint 22.3 มากับ 3.12 จึงใช้ได้เลย

```bash
sudo apt install -y python3-venv
```

ถ้าอยากตรึงเวอร์ชันให้เหมือนกันทุกเครื่องก็ใช้ `uv` ดึงมาต่างหากได้ ไม่แตะ Python ของระบบ

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12
```

ให้ผู้ใช้ที่จะรันบริการเข้ากลุ่ม docker แล้ว login ใหม่

```bash
sudo usermod -aG docker $USER
```

## 2. โค้ดกับของลับ

ตั้งตัวแปรสองตัวนี้ก่อน แล้วคำสั่งที่เหลือคัดลอกไปวางได้เลย

```bash
export ARENA=~/VRU-AI/projects/colosseum && export HDD=/media/ratchainant/hdd/colosseum
```

```bash
mkdir -p $ARENA && cd $ARENA && mkdir -p $HDD/backup
```

```bash
git clone https://github.com/VRU-AI-SWU/colosseum-swu.git app
```

```bash
git clone git@github.com:VRU-AI-SWU/colosseum-hypogeum.git secrets && chmod 700 secrets
```

⚠️ **`secrets/` ต้องเป็น 700** และห้าม mount เข้า container ของนิสิตเด็ดขาด
`DockerLauncher` mount เฉพาะโฟลเดอร์ submission เข้าไปแบบ read-only เท่านั้น

```bash
cd $ARENA/app && uv venv --python 3.12 && uv pip install -e envs/cp463-vacuum -e ".[api,cli,dev]"
```

`[api]` เป็นสิ่งที่ขาดไม่ได้ — `fastapi` กับ `uvicorn` อยู่ใน extra ตัวนั้น
ลืมแล้ว `arena serve` จะพังตอนสตาร์ทด้วย `ModuleNotFoundError` · `[dev]` ให้ pytest
สำหรับขั้นตรวจข้างล่าง · `[cli]` ให้ httpx ไว้ยิง API จากเครื่องเดียวกัน

ยืนยันว่า **numpy ได้ 2.1.x** — ข้อนี้พลาดแล้วเงียบ ส่วนเวอร์ชัน Python ไม่สำคัญ

```bash
$ARENA/app/.venv/bin/python -VV && $ARENA/app/.venv/bin/python -c "import numpy; print(numpy.__version__)"
```

ตรวจว่า environment ตรงกับตอน calibrate

```bash
cd $ARENA/app && ARENA_SECRETS=$ARENA/secrets .venv/bin/python -m pytest core/tests runners/tests envs/cp463-vacuum/tests -q
```

## 3. Docker image ของ sandbox

**บังคับ** — `--real-seeds` ไม่ยอมเริ่มถ้าไม่มี image นี้

```bash
cd $ARENA/app && docker build -t arena/vacuum:cpu -f runners/agent_env/images/Dockerfile.cpu .
```

> ถ้า build ล้มด้วย TLS error ที่ `registry-1.docker.io` — เครือข่ายมหาวิทยาลัยเคยส่ง
> cert `*.swu.ac.th` มาแทนของจริง แก้ชั่วคราวด้วย `DOCKER_BUILDKIT=0 docker build ...`

## 4. ตรึงหมุด baseline

ค่าที่ commit ไว้ถูกวัดบนเครื่อง dev — ยืนยันว่าเครื่องนี้ให้ผลตรงกัน ถ้าไม่ตรง
แปลว่า environment ต่างกันจริง และคะแนนของนิสิตจะเทียบข้ามเครื่องไม่ได้

```bash
cd $ARENA/app && ARENA_SECRETS=$ARENA/secrets .venv/bin/python tools/pin_baselines.py --check
```

## 5. Cloudflare Tunnel

ติดตั้ง binary โดยไม่ต้อง sudo — เก็บไว้ที่ `~/.local/bin` แล้วให้ systemd เรียกจากที่นั่น

```bash
mkdir -p ~/.local/bin && curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/.local/bin/cloudflared && chmod +x ~/.local/bin/cloudflared
```

แล้วรันสามคำสั่งนี้ — **รันจากโฟลเดอร์ไหนก็ได้** เพราะมันเขียนลง `~/.cloudflared/` เสมอ

ขั้นแรกเปิดเบราว์เซอร์ให้ authorize แล้วเลือกโซน `vru-ai.com`

⚠️ **ต้องล็อกอินด้วยบัญชีที่เป็นเจ้าของโซนนั้น** — หน้า authorize แสดงเฉพาะโซนของบัญชี
ที่ล็อกอินอยู่ ถ้าเข้าผิดบัญชีจะเห็นตารางว่างเปล่าโดยไม่มีคำอธิบายว่าทำไม
(บัญชีที่ถูกต้องของเราคือ `vru0ai@g.swu.ac.th`)

```bash
cloudflared tunnel login
```

```bash
cloudflared tunnel create porta-triumphalis
```

```bash
cloudflared tunnel route dns porta-triumphalis colosseum-api.vru-ai.com
```

จากนั้นเอา `<TUNNEL-ID>` ที่ได้ไปเติมใน [`cloudflared/config.yml`](cloudflared/config.yml)
แล้วคัดลอกไปที่ `~/.cloudflared/config.yml` — รายละเอียดอยู่ที่ [`cloudflared/README.md`](cloudflared/README.md)

🔒 `~/.cloudflared/cert.pem` เป็นสิทธิ์ระดับ**บัญชี** ในโซน `vru-ai.com` ใครได้ไป
สร้าง tunnel และแก้ DNS ของโดเมนได้ · เก็บไว้เฉพาะเครื่องนี้ ห้ามคัดลอกไปที่อื่น

## 5.5 ตั้งค่าล็อกอินด้วย Google

สร้าง OAuth client ที่ [Google Cloud console](https://console.cloud.google.com/apis/credentials)
→ **Create Credentials → OAuth client ID → Web application**

| ช่อง | ค่า |
|---|---|
| Authorized redirect URIs | `https://colosseum-api.vru-ai.com/auth/google/callback` |
| (สำหรับ dev) | `http://localhost:8000/auth/google/callback` |
| Authorized JavaScript origins | เว้นว่าง — เราใช้ flow ฝั่งเซิร์ฟเวอร์ |

แล้วไปที่ **Audience → User type = Internal** เพื่อให้เฉพาะบัญชีใน Workspace ของ
มหาวิทยาลัยผ่าน flow ได้ตั้งแต่หน้าล็อกอินของ Google เอง · โค้ดยังเช็ค `hd` ซ้ำอีกชั้น
เพราะการตั้งค่าใน console เปลี่ยนได้โดยที่โค้ดไม่รู้ตัว

**client secret ไม่อยู่ใน repo** — เขียนลงไฟล์ที่ systemd อ่าน

```bash
sudo install -m 600 -o root -g root /dev/null /etc/arena.env && sudo nano /etc/arena.env
```

```
ARENA_GOOGLE_CLIENT_ID=<client id>
ARENA_GOOGLE_CLIENT_SECRET=<secret>
ARENA_WEB_ORIGIN=https://colosseum.vru-ai.com
ARENA_STAFF_EMAILS=aj@g.swu.ac.th,ta@g.swu.ac.th
```

`ARENA_STAFF_EMAILS` คืออีเมลของคนที่เปลี่ยนกติกาของวิชาได้จากหน้าเว็บ (ตอนนี้คือ
ขนาดทีมสูงสุด) — คั่นด้วยจุลภาค · **ว่างไว้ = ไม่มีใครเป็นผู้สอน** ซึ่งเป็นค่าเริ่มต้น
ที่ถูกต้อง เพราะการเดาว่า "คนแรกที่ล็อกอินคือผู้สอน" จะทำให้ใครที่รู้ URL ก่อนเพื่อน
ยึดสิทธิ์ไปได้

อยู่ที่นี่แทนที่จะอยู่ในฐานข้อมูล **โดยตั้งใจ เหมือน sudoers** — ถ้าแก้ผ่านหน้าเว็บได้
คนที่ยึดสิทธิ์ผู้สอนได้ครั้งเดียวจะแต่งตั้งตัวเองถาวรและถอดคนอื่นออกได้ · การแก้ไฟล์นี้
ต้องมีสิทธิ์ root บนเครื่อง ซึ่งเป็นระดับที่เหมาะกับสิ่งที่มันควบคุม

⚠️ **สิทธิ์ผู้สอนผูกกับทีมที่มีแต่ผู้สอนเท่านั้น** — โทเคนใช้ร่วมกันทั้งทีม ถ้าผู้สอน
ไปอยู่ทีมเดียวกับนิสิต ทีมนั้นจะไม่มีสิทธิ์ผู้สอน เพราะไม่งั้นนิสิตคนนั้นจะถือโทเคน
ที่เปลี่ยนกติกาของทั้งวิชาได้

ตรวจว่าถูกต้อง (ไม่แสดงค่า secret ออกมา — แค่ความยาวกับตัวอักษรแรกๆ)

```bash
sudo python3 ~/VRU-AI/projects/colosseum/app/tools/check_env.py
```

⚠️ ช่องว่างหน้า/หลังค่า และเครื่องหมายคำพูดครอบค่า เป็นสองสาเหตุที่พบบ่อยที่สุดของ
"ใส่ถูกแล้วแต่ใช้ไม่ได้" — systemd เก็บอัญประกาศไปเป็นส่วนหนึ่งของค่าด้วย แล้ว Google
ปฏิเสธโดยไม่บอกสาเหตุ · สคริปต์ตรวจให้ทั้งสองอย่าง

## 6. ให้มันรันเองหลังไฟดับ

เทอมหนึ่งยาว 16 สัปดาห์ — บริการที่ต้องมีคนมา `ssh` เข้าไปสตาร์ทใหม่ทุกครั้งที่ไฟดับ
จะดับจริงในสัปดาห์ที่นิสิตกำลังเร่งส่งงาน

```bash
sudo cp $ARENA/app/deploy/systemd/arena-api.service /etc/systemd/system/ && sudo systemctl enable --now arena-api
```

```bash
sudo cp $ARENA/app/deploy/systemd/cloudflared-porta-triumphalis.service /etc/systemd/system/ && sudo systemctl enable --now cloudflared-porta-triumphalis
```

สำรองข้อมูลทุกคืนตี 3 — **แก้ path ในสองไฟล์ให้ตรงกับเครื่องก่อน**

```bash
sudo cp $ARENA/app/deploy/systemd/arena-backup.{service,timer} /etc/systemd/system/ && sudo systemctl enable --now arena-backup.timer
```

ทดสอบว่ามันทำงานจริงเลยทันที ไม่ต้องรอถึงตี 3

```bash
sudo systemctl start arena-backup && journalctl -u arena-backup -n 20 --no-pager
```

ตรวจว่าทั้งสองขึ้นแล้ว

```bash
systemctl status arena-api cloudflared --no-pager && curl -s https://colosseum-api.vru-ai.com/api/health
```

---

## เพดานที่วัดไว้แล้ว

**เพดานขนาดไฟล์** — ✅ วัดแล้ว: **100 MiB พอดี** (99 MiB ผ่าน · 100 MiB ได้ 413 จาก edge)
request ที่เกินไม่เคยถึง API เรา จึงส่งข้อความบอกวิธีแก้กลับไม่ได้ — `arena submit`
ตรวจให้ก่อนอัพโหลดและตัดที่ 95 MB แล้ว

**timeout ของ request** — Cloudflare ตัดที่ 100 วินาทีด้วย error 524 · ไม่กระทบเรา
เพราะการรับ submission ทำแค่การตรวจแบบ static ที่ไม่รันโค้ด (วัดได้ 1.9 วิ สำหรับไฟล์
25 MB ซึ่งส่วนใหญ่เป็นเวลาส่งข้อมูล) ส่วนการให้คะแนนเกิดแบบ async ที่ worker

## สถานะที่ติดตั้งไว้จริง (20 ส.ค. 2026)

| | |
|---|---|
| เครื่อง | `gpu-linux-server` · Linux Mint 22.3 · `10.1.137.113` (LAN `enp5s0`) |
| เข้าถึงเพื่อดูแล | `ssh gpu-linux-server` ผ่าน Tailscale หรือ mDNS ในวง LAN |
| โค้ด + ของลับ | `~/VRU-AI/projects/colosseum/{app,secrets,data}` |
| tunnel | `porta-triumphalis` → `colosseum-api.vru-ai.com` |
| หน้าเว็บ | `colosseum.vru-ai.com` — Worker + static assets ([`web/`](../web/)) |
| สำรองข้อมูล | ทุกคืน 03:00 → `/media/ratchainant/hdd/colosseum/backup` (NVMe ลูกที่สอง) |
| service | `arena-api` · `cloudflared-porta-triumphalis` · `arena-backup.timer` — enabled ทั้งหมด |

ยืนยันแล้วว่าทุกอย่างขึ้นเองหลัง reboot จริง (เจ้าของเครื่อง restart เอง 20 ส.ค. 14:25 —
service ขึ้นครบภายใน 47 วินาที ข้อมูลไม่หาย HDD mount กลับมาเอง)

## ⚠️ เน็ตของมหาวิทยาลัยมี captive portal ที่หมดอายุทุก 3 ชั่วโมง

`ipass.swu.ac.th` ให้ session ละ 3 ชั่วโมง ต้องกด Refresh เอง — **นี่คือสาเหตุที่พบบ่อยที่สุด
ของอาการ "ระบบล่ม" ที่ไม่ได้เกิดจากโค้ด**

อาการ: HTTPS **ทุกโดเมน**ล้มพร้อมกัน และ cert ที่ได้กลายเป็น `CN=*.swu.ac.th`
ซึ่งไม่ตรงกับ hostname ที่ขอ

```bash
echo | openssl s_client -connect github.com:443 -servername github.com 2>/dev/null | openssl x509 -noout -subject
```

ได้ `*.swu.ac.th` = หลุด session · ได้ `CN=github.com` = เน็ตปกติ

**ตรวจว่ากระทบทุกโดเมนหรือเฉพาะโดเมนเรา ก่อนจะไปไล่หาสาเหตุที่โค้ดเสมอ** —
ถ้า github กับ google ก็เข้าไม่ได้ ปัญหาไม่ได้อยู่ที่ tunnel หรือ deploy

### เกิดขึ้นจริงแล้วหนึ่งครั้ง — ระบบล่ม 12.5 ชั่วโมง

25 ส.ค. 21:54 → 26 ส.ค. 10:27 · tunnel เหลือ 0 connection ตลอดคืน
uptime ของวันนั้น **48%** · หลักฐานว่าเป็นเรื่องเน็ตไม่ใช่ tunnel: `Failed to refresh DNS`
เกิด 12 ครั้ง/ชั่วโมงตลอดคืน และ Tailscale อัพโหลด log ค้างสำเร็จตอน 10:29 พร้อมกัน

**ตัวเลข 48% นี้เป็นของ LAN ที่มี captive portal เท่านั้น ไม่ใช่คุณภาพของระบบ**

### ✅ ตัดสินแล้ว: ตอนใช้จริงจะไม่ใช้ LAN

เครื่องจะต่อ wifi ของมหาวิทยาลัยที่**ไม่มีการตัด connection** (`SCI@SWU` / `Sci@WiFi`)
LAN ใช้เฉพาะช่วงพัฒนาเพราะเร็วกว่า จึงไม่ต้องทำ keepalive อัตโนมัติหรือขอยกเว้นจาก IT

⚠️ **แต่ต้องวัดซ้ำหลังสลับ** — จุดที่เครื่องตั้งอยู่สัญญาณ wifi อ่อน (IT กำลังแก้)
wifi ที่หลุดๆ ติดๆ จะให้อาการเหมือนกันเป๊ะแต่คนละสาเหตุ · ก่อนเปิดให้นิสิตใช้
ต้องปล่อยให้รันบน wifi ครบหนึ่งวันเต็มแล้วเช็คว่า tunnel ไม่หลุด

```bash
journalctl -u cloudflared-porta-triumphalis -b --no-pager | grep -c "Connection terminated"
```

ตัวเลขควรอยู่หลักหน่วยถึงหลักสิบต่อวัน ถ้าได้หลักร้อยขึ้นไปแปลว่าเน็ตยังไม่นิ่งพอ

## เดินเครื่องประจำวัน

| | คำสั่ง |
|---|---|
| ดู log | `journalctl -u arena-api -f` |
| อัพเดตโค้ด | `cd $ARENA/app && git pull && sudo systemctl restart arena-api` |
| สำรองข้อมูล | อัตโนมัติทุกคืน · ดูผลด้วย `journalctl -u arena-backup -n 20` |
| กู้ข้อมูล | หยุดบริการ → คัดลอกสำเนาจาก `$HDD/backup/db/` ทับ `data/arena.db` (ลบ `-wal`/`-shm` เดิมด้วย) → สตาร์ทใหม่ |
| ล้าง docker cache | `docker builder prune -af` หลัง build ทุกครั้ง |
| ดูสถานะ tunnel | Cloudflare dashboard → Zero Trust → Networks → Tunnels |

### เข้าถึงเพื่อดูแลระบบ

API bind ที่ `127.0.0.1` เท่านั้น ทางเข้าจากภายนอกจึงมีทางเดียวคือ tunnel
เวลาผู้สอนต้องการยิง API ตรงๆ (เช่นสั่ง private run ตอนตัดเกรด) ให้ทำ port forward
ผ่าน Tailscale แทนการเปิด bind ให้กว้างขึ้น

```bash
ssh -L 8000:127.0.0.1:8000 gpu-linux-server
```

**อย่าเปลี่ยนไป bind ที่ IP ของ tailnet** ถึงแม้จะดูสะดวกกว่า — ทุกอุปกรณ์ใน tailnet
จะยิง API ได้โดยข้าม Cloudflare ทั้งหมด ทำให้ rate limit และ Access ที่จะใส่ทีหลัง
ไม่มีความหมาย และเป็นทางเข้าที่ไม่มีใครจำได้ว่ามีอยู่

⚠️ **สำรอง `arena.db-wal` ไปด้วยเสมอ** — คัดลอกแค่ `arena.db` ตอนที่บริการกำลังรัน
จะได้ไฟล์ที่ข้อมูลไม่ครบ เพราะ transaction ล่าสุดยังอยู่ใน WAL
