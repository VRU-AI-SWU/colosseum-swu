# สถาปัตยกรรม — สิ่งที่มีจริงวันนี้

**เอกสารนี้อธิบายระบบที่รันได้จริง ณ วันที่ 20 ส.ค. 2026** — ตอนนี้ระบบ**ขึ้นจริงแล้ว**
ที่ `colosseum-api.vru-ai.com` บนเครื่อง GPU ในมหาวิทยาลัย ส่วน
[README §10](../README.md#10-สถาปัตยกรรมระบบ) เป็น**เป้าหมายที่ออกแบบไว้** ซึ่งบางส่วน
ยังไม่ได้สร้าง การอ่าน §10 อย่างเดียวจะเข้าใจผิดว่าระบบทำอะไรได้บ้าง

> ทุกครั้งที่แก้โครงสร้างจริง **ต้องแก้ไฟล์นี้ในคอมมิตเดียวกัน** — เอกสารสถาปัตยกรรม
> ที่ตามหลังโค้ดคือเอกสารที่คนอ่านแล้วตัดสินใจผิด

---

## 1. ภาพรวมของการติดตั้ง

```mermaid
flowchart TB
    subgraph cloud["Cloudflare"]
        page["colosseum.vru-ai.com<br/>หน้าเว็บ leaderboard<br/>Worker + static assets ✅"]
        edge["colosseum-api.vru-ai.com<br/>Cloudflare edge"]
    end

    subgraph uni["เครื่อง GPU ในมหาวิทยาลัย (Linux Mint) ✅ ขึ้นแล้ว"]
        cfd["cloudflared<br/>porta-triumphalis<br/>systemd · Restart=always"]
        subgraph proc["arena serve — process เดียว ✅"]
            api["FastAPI<br/>core/api.py"]
            worker["Worker thread<br/>runners/worker.py"]
        end
        db[("SQLite<br/>arena.db")]
        art[("artifacts/<br/>zip + replay")]
        secrets[["🔒 ~/VRU-AI/projects/colosseum/secrets/<br/>seeds.yaml · โค้ดเฉลย"]]
        sandbox["sandbox ที่รัน agent ของนิสิต"]
    end

    student["นิสิต<br/>arena submit"] -->|"HTTPS"| edge
    page -->|"fetch JSON · CORS ตั้งแล้ว"| edge
    edge <-->|"QUIC · UDP 7844<br/>cloudflared ต่อ *ออกไป*"| cfd
    cfd -->|"127.0.0.1:8000"| api
    api <--> worker
    api --> db
    worker --> db
    api --> art
    worker --> art
    worker -->|"อ่านตอนรับงาน"| secrets
    worker -->|"obs / action<br/>ผ่าน stdin/stdout"| sandbox

    style secrets fill:#fff3cd,stroke:#856404
```

**ทำไม cloudflared ต่อออกไปแทนที่จะเปิด port รับ** — เครื่องในมหาวิทยาลัยอยู่หลัง firewall
ที่ขอเปิด inbound port ยาก การให้ tunnel ต่อออกไปติดตั้งได้ทันทีและไม่มี port เปิดรับ
จากอินเทอร์เน็ตเลย · ผลการวัดบนเครือข่ายจริงอยู่ที่
[README §10.1](../README.md#101-ภาพรวม-hybrid-web-บน-cloud--runner-ในมหาวิทยาลัย)
และวิธีติดตั้งอยู่ที่ [deploy/cloudflared/](../deploy/cloudflared/README.md)

### ต่างจาก README §10 ตรงไหน

| §10 วางไว้ | วันนี้ | ผลที่ตามมา |
|---|---|---|
| runner เป็น daemon แยกเครื่อง ต่อ WebSocket เข้ามา | worker เป็น **thread ใน process เดียวกับ API** | ยังกระจายงานข้ามเครื่องไม่ได้ · WebSocket ที่วาดไว้ยังไม่มีอยู่จริง |
| Postgres + Redis | **SQLite** write-through ([`core/db.py`](../core/db.py)) | รองรับ process เดียว ซึ่งพอดีกับข้อบน |
| Object Storage | โฟลเดอร์บนดิสก์ ([`core/store.py`](../core/store.py)) | ต้องสำรองไฟล์เอง |
| หน้าเว็บ leaderboard | ✅ ขึ้นแล้ว — Worker + static assets | ส่งงานยังต้องใช้ `arena submit` โดยตั้งใจ |

---

## 2. เส้นทางของ submission

```mermaid
sequenceDiagram
    autonumber
    actor S as นิสิต
    participant API as FastAPI
    participant Q as JobQueue
    participant DB as SQLite
    participant W as Worker
    participant SB as sandbox
    participant SEC as 🔒 secrets

    S->>API: POST /submissions (zip)
    API->>API: ตรวจแบบ static — ไม่รันโค้ด<br/>agent.py · class Agent · import whitelist
    Note over API: ไม่ผ่าน → 422 พร้อม **วิธีแก้** ไม่ใช่แค่บอกว่าผิด
    API->>API: ตรวจโควตา · 1 งาน/ทีม · หน้าต่างเวลา
    API->>Q: enqueue(Run)
    Q->>DB: save_run(queued)
    API-->>S: 201 + queue_position

    W->>Q: claim() — fair-share ทีมที่ถูกเสิร์ฟน้อยสุดก่อน
    Q->>DB: save_run(running + lease)
    W->>SEC: load_seeds(phase, public)
    Note over W,SEC: **seed ไม่เคยออกจากฝั่งนี้**
    W->>SB: obs → action ทีละ timestep
    SB-->>W: action
    W->>W: ตรวจ config_hash ตรงกับตอน generate seed
    W->>Q: report(score, metrics)
    Note over W,Q: รายงานเป็น "episode 1..30" ไม่ใช่ค่า seed
    Q->>DB: save_run(done + score)

    S->>API: GET /leaderboard
    API->>DB: อ่านจาก dict ในหน่วยความจำ
    API-->>S: อันดับ + หมุด baseline ที่ตรึงไว้
```

heartbeat ต่ออายุ lease ทุก 20 วินาที ถ้า worker ตายกลางคัน lease หมดอายุแล้วงานกลับเข้าคิวเอง
งานที่ล้มซ้ำเกิน 3 ครั้งถูกทำเครื่องหมายว่าล้มเหลว ไม่วนไม่รู้จบ ([`core/queue.py`](../core/queue.py))

---

## 3. ขอบเขตความไว้วางใจ — เส้นทางของของลับ

```mermaid
flowchart LR
    subgraph untrusted["ไม่ไว้ใจ — โค้ดของนิสิต"]
        agent["Agent.act()"]
    end
    subgraph trusted["ไว้ใจ — ฝั่งผู้สอน"]
        env["VacuumEnv<br/>ผังห้อง · เฉลย"]
        seeds[["seeds.yaml"]]
        gold[["โค้ด Gold/Diamond"]]
    end
    subgraph public["สาธารณะ — ถึงมือนิสิตได้"]
        board["leaderboard · /episodes"]
        wheel["cp463-vacuum wheel"]
    end

    agent <-->|"msgpack ผ่าน stdin/stdout<br/>obs เท่านั้น"| env
    env --> board
    seeds -.->|"❌ ปิดแล้ว: worker เคยส่งค่า seed<br/>ไปกับ metrics"| board
    gold -.->|"❌ ปิดแล้ว: เคยแจกไปกับ wheel<br/>และเคยมี pseudocode ใน env-spec"| wheel

    style untrusted fill:#f8d7da,stroke:#721c24
    style trusted fill:#fff3cd,stroke:#856404
    style public fill:#d4edda,stroke:#155724
```

สามชั้นที่บังคับไว้เชิงโครงสร้าง ไม่ใช่เชิงวินัย

| ชั้น | บังคับด้วย |
|---|---|
| agent อยู่คนละ process กับ environment | [`runners/agent_env/launcher.py`](../runners/agent_env/launcher.py) — `import gc` แล้วไล่หา env ไม่เจอเพราะมันไม่ได้อยู่ใน process เดียวกัน |
| API ไม่เคยเห็นค่า seed | worker เป็นคนอ่านจาก `ARENA_SECRETS` และรายงานเป็นเลขลำดับ · เทสต์ `test_api_never_reveals_seed_values` |
| โค้ดเฉลยไม่อยู่ในแพ็กเกจที่แจก | `ARENA_SECRETS` เท่านั้นที่โหลด Gold/Diamond ได้ · pre-commit hook ตรวจว่าไม่มีวิธี implement หลุดเข้า repo สาธารณะ |

รายละเอียดเต็มอยู่ที่ [README §10.4](../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries)

---

## 4. ⚠️ ช่องว่างที่รู้ตัว

### 4.1 ~~`smoke_test()` ไม่ถูกเรียกใช้จริง~~ ✅ ปิดแล้ว

`smoke_test()` เขียนครบและมีเทสต์ของตัวเอง 6 ข้อมาตลอด **แต่ไม่มีใครเรียกมันนอกจากเทสต์**
ผลคือ starter kit ที่บอกนิสิตว่า "ระบบตรวจข้อนี้ตอนรับ submission และ**ปฏิเสธ**ถ้าไม่ผ่าน"
ไม่เป็นความจริง — agent ที่ `reset()` ไม่สะอาดถูกรับเข้าไปแล้วได้คะแนนต่ำกว่าที่ควร
โดยไม่มีใครบอกว่าเพราะอะไร ซึ่งเป็นอาการที่ debug ยากที่สุด

ตอนนี้ [`runners/worker.py`](../runners/worker.py) เรียกก่อนรันจริงแล้ว และเทสต์ที่เพิ่มมา
ตรวจ**การต่อสาย** ไม่ใช่ตัวฟังก์ชัน (`test_state_leaking_agent_is_rejected_by_the_worker`)

**ไม่ตรวจ run แบบ private กับ rejudge โดยตั้งใจ** — submission พวกนั้นผ่าน smoke test
ตอนรัน public มาแล้ว ถ้าตรวจซ้ำแล้วมันล้มด้วยเหตุบังเอิญ final pick ของทีมนั้นจะถูก
ปฏิเสธในรอบตัดเกรด ซึ่งเป็นจังหวะที่แก้ตัวไม่ได้แล้ว

### 4.2 ~~`arena serve` ใช้ SubprocessLauncher ไม่ใช่ Docker~~ ✅ ปิดแล้ว

`Worker.launcher` เคยเป็น `SubprocessLauncher` เสมอและ CLI ไม่มีธงให้เลือก แปลว่าถ้าเปิด
ให้นิสิตส่งงานจริง โค้ดของนิสิตจะรันบนเครื่องที่มีเฉลยอยู่โดยไม่มี container ห่อ

ตอนนี้มี `arena serve --sandbox auto|docker|subprocess`

| โหมด | พฤติกรรม |
|---|---|
| `auto` (ค่าเริ่มต้น) | ใช้ Docker ถ้ามี image พร้อม · ไม่มีก็ถอยไป subprocess **พร้อมเตือนดังๆ** |
| `docker` | ต้องมี Docker — ไม่พร้อมก็ไม่ยอมเริ่ม |
| `subprocess` | บังคับไม่ใช้ container · **ใช้กับ `--real-seeds` ไม่ได้** |

**`--real-seeds` บังคับ Docker เสมอ** — ถ้ากำลังให้คะแนนด้วย seed จริง แปลว่าโค้ดที่รัน
คือของนิสิตและมันอยู่บนเครื่องเดียวกับเฉลย การถอยไป subprocess เงียบๆ ในโหมดนั้นคือ
การรันโค้ดที่ไม่ไว้ใจไว้ข้างๆ ของลับ · เทสต์อยู่ที่ `core/tests/test_cli_sandbox.py`

### 4.3 🔴 โทเคนของทีมเดาได้ ทั้งที่ API เปิดสาธารณะแล้ว

โทเคนคือ `team-1`, `team-2`, `team-3` ([`core/wiring.py`](../core/wiring.py)) ซึ่งเดาถูก
ตั้งแต่ครั้งแรก · ตอนที่มันรันแค่บน localhost เรื่องนี้ไม่สำคัญ แต่ตอนนี้
`colosseum-api.vru-ai.com` อยู่บนอินเทอร์เน็ตแล้ว ใครที่รู้ URL ส่งงานในนามทีมใดก็ได้
กินโควตาของทีมนั้น และเห็นคะแนนรายตอนของเขา

วันนี้ความเสี่ยงยังต่ำเพราะนิสิตยังไม่ได้รับ URL — **ต้องปิดก่อนแจก**

⚠️ **Cloudflare Access ไม่ใช่ทางแก้** ถึงแม้จะเร็วที่สุด — มันบังคับ browser flow ซึ่ง
`arena submit` ที่รันจาก terminal ทำไม่ได้ ต้องให้นิสิตรัน `cloudflared access login`
ก่อนทุกครั้ง · Access เหมาะกับ**หน้าเว็บ** ไม่ใช่ API

✅ **ตัดสินแล้ว (ส.ค. 2026): ใช้ Google Workspace `@g.swu.ac.th`** ไม่รับ Gmail ทั่วไป
และไม่ทำ OTP ทางอีเมล (ต้องมี SMTP · ต้องจัดการ deliverability, rate limit, code หมดอายุ
— ทำมากกว่าเพื่อความปลอดภัยที่น้อยกว่า)

**OTP หรือ OAuth ก็ไม่ได้แทนที่ token** — `arena submit` รันจาก terminal จึงต้องมี
credential ที่อยู่ได้นานเสมอ สิ่งที่การล็อกอินเปลี่ยนคือ*วิธีที่ token ถูกแจก*
จาก "ผู้สอนส่งให้ทีละคน" เป็น "นิสิตล็อกอินแล้วเห็น token ของตัวเอง"

### 4.4 ~~Windows: `select()` รับได้เฉพาะ socket~~ ✅ ปิดแล้ว (v1.0.2)

ข้อนี้เคยเขียนไว้ในตารางข้างล่างว่า *"ตรวจโค้ดแล้วไม่มี API เฉพาะ POSIX"* พร้อมหมายเหตุ
ว่าพิสูจน์ได้แค่ "ไม่เห็นอะไรที่จะพัง" **หมายเหตุนั้นถูก และการตรวจนั้นพลาดจริง**

`Channel._wait_readable` ใช้ `selectors.DefaultSelector()` ซึ่งบน Windows คือ
`SelectSelector` → `select()` ที่**รับได้เฉพาะ socket** ส่ง pipe เข้าไปได้
`OSError [WinError 10038]` ทันทีตั้งแต่ข้อความแรก · `arena eval` จึงพังทั้งคำสั่ง
ขณะที่ `pip install` และ `vacuum.selfcheck` ผ่านหมด — เพราะสองอันนั้นไม่เปิด process ลูก

ทำไมการตรวจโค้ดถึงพลาด — ไล่หา API ที่**ชื่อบอกว่าเป็น POSIX** (`fcntl`, `setsid`,
`preexec_fn`, `SIGKILL`) แต่ `selectors` เป็น API ที่มีอยู่ทุกแพลตฟอร์มและ*ทำงานคนละแบบ*
ซึ่งอันตรายกว่า เพราะมันไม่ประกาศตัว

**สิ่งที่ทำให้บั๊กนี้อยู่รอดมาได้คือไม่มีเทสต์ไหนแตะ `Channel` เลยสักอัน** —
ตอนนี้มี [`runners/tests/test_protocol.py`](../runners/tests/test_protocol.py) ที่
ใช้ `os.pipe()` จริง (ไม่ใช่ `BytesIO` ซึ่งไม่มี `fileno()` จึงข้ามโค้ดที่พังไปทั้งหมด)

ตรรกะการวนถามของฝั่ง Windows แยกเป็น `_poll_until_readable(peek, timeout, clock, sleep)`
ที่รับตัวช่วยเข้ามาได้ **เพื่อให้เทสต์รันบน macOS/Linux ด้วย** — โค้ดที่รันได้แค่บน
แพลตฟอร์มที่ไม่มีใครรัน CI ให้ ก็คือโค้ดที่จะพังใส่นิสิตอีกครั้ง

### 4.5 ที่ยังเหลือ

| | สถานะ |
|---|---|
| หน้าเว็บ leaderboard | ✅ ขึ้นแล้วที่ `colosseum.vru-ai.com` ([`web/`](../web/)) |
| Google OAuth | ✅ ใช้งานแล้ว — `@g.swu.ac.th` เท่านั้น ([`core/auth.py`](../core/auth.py)) · token ยังเป็น credential ของ `arena submit` ตามเดิม ดู §4.3 |
| runner daemon + WebSocket | ยังไม่มี — worker เป็น thread |
| เพดานไฟล์บน named tunnel | ✅ วัดแล้ว **100 MiB** · `arena submit` ตรวจให้ที่ 95 MB |
| replay viewer | ✅ `python -m vacuum.viewer replays/1.vrp` → หน้าเว็บไฟล์เดียว ([`vacuum/viewer.py`](../envs/cp463-vacuum/vacuum/viewer.py)) · **ทำงานกับไฟล์ในเครื่องเท่านั้น** เพราะ header ของ `.vrp` มี seed + ผังห้องครบ การเสิร์ฟผ่าน API จะยกผังห้อง public ให้ทั้งชุด |
| Windows | ✅ **ทดสอบครบทั้งเส้นทางแล้ว** (27 ส.ค. 2026 · Python 3.14.4) — install · `selfcheck` · `eval` · `--check-reset` · `submit --dry-run` · `submit` · `status` · `arena eval` เคยพัง แก้ใน v1.0.2 ([§4.4](#44-windows-select-รับได้เฉพาะ-socket--ปิดแล้ว-v102)) · **คะแนนตรงกับ macOS ทุกบิต** `0.27318887262207453` (float64 ครบ 17 หลัก) |

### 4.6 🔻 ต้องทำก่อนเปิดให้นิสิตใช้จริง

รายการนี้แยกจาก §4.5 เพราะมันไม่ใช่ "ฟีเจอร์ที่ยังไม่มี" แต่เป็น **ของที่ต้องเก็บกวาด
ก่อนคนนอกเข้ามา** ซึ่งเป็นคนละจังหวะกัน

**ยุบทีมทดสอบ** — ตัดสินใจแล้ว (27 ส.ค. 2026) ว่า*ปล่อยไว้ก่อน* แล้วยุบทีเดียวตอนระบบ
พร้อมเปิดจริง เพราะยังต้องใช้ทดสอบอยู่

| ทีม | run | หมายเหตุ |
|---|---|---|
| `vru ai` | 3 | บัญชี `vru-ai@g.swu.ac.th` — ยุบได้เลย |
| `Ratchainant Thammasudjarit` | 2 | **บัญชีจริงของผู้สอน** · ยุบแล้วล็อกอินใหม่จะได้ทีมเดี่ยวใหม่พร้อมโทเคนใหม่ |

```bash
python tools/retire_team.py --db /path/arena.db --team <team-id> ...
sudo systemctl restart arena-api   # บริการเก็บ working set ในหน่วยความจำ
```

การยุบทำให้โทเคนใช้ไม่ได้ทันทีและคะแนนหายจาก leaderboard **แต่ข้อมูลยังอยู่ครบ** —
เหตุผลอยู่ในหัวไฟล์ [`tools/retire_team.py`](../tools/retire_team.py)

---

## 5. ที่อยู่ของแต่ละอย่าง

```
colosseum-swu/                    (สาธารณะ)
├── core/          แพลตฟอร์ม — ไม่รู้จักโจทย์ใดๆ
│   ├── domain.py     Team · Competition · Submission · Run
│   ├── queue.py      fair-share + lease + heartbeat
│   ├── db.py         SQLite write-through
│   ├── service.py    กติกาทั้งหมดอยู่ที่นี่ที่เดียว
│   └── api.py        REST · ไม่เคยเห็น seed
├── runners/       รันงาน — ไม่รู้จัก core
│   ├── worker.py     หยิบงาน · โหลด seed · รายงานผล
│   └── agent_env/    protocol · launcher · sandbox
├── envs/          โจทย์ · หนึ่งโฟลเดอร์ต่อหนึ่ง competition
├── deploy/        ติดตั้งเครื่อง runner ตั้งแต่ศูนย์ · systemd · cloudflared
└── docs/          เอกสารนี้ · สเปคของโจทย์ · task template

colosseum-hypogeum/               (ส่วนตัว · clone เฉพาะเครื่อง GPU)
├── cp463-1-2026/vacuum/seeds.yaml
├── agents/cp463-vacuum/          โค้ด Gold/Diamond
└── docs/cp463-1-2026/vacuum/     สเปคของ baseline ที่ไม่แจก
```

กฎที่ทำให้โครงสร้างนี้ไม่พังตามเวลา: **`core/` ไม่ import `runners/` และกลับกัน**
การผูกทั้งสองฝั่งเกิดที่ [`core/wiring.py`](../core/wiring.py) ที่เดียว การเพิ่ม competition
ใหม่จึงแตะแค่ `envs/` กับไฟล์นั้น ([README §10.5](../README.md#105-โครงสร้าง-repository))
