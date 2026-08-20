# ติดตั้งเครื่อง runner ตั้งแต่ศูนย์

เครื่องเดียวรันทุกอย่าง — API, worker, sandbox, ฐานข้อมูล และเป็นที่เดียวที่ของลับอยู่
([README §10.4](../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries))

```
SSD ของระบบ                              HDD "colosseum"
~/VRU-AI/projects/colosseum/             $HDD/
├── app/       โค้ดแพลตฟอร์ม               ├── artifacts/   zip ของนิสิต + replay
├── secrets/   🔒 hypogeum                │   ├── submissions/
├── data/      arena.db (+ -wal)         │   └── replays/
└── app/.venv/ ~64 MB                    └── work/        แตก zip ชั่วคราว
```

เกณฑ์ที่ใช้แบ่ง: **เล็ก+เขียนบ่อย อยู่ SSD · ใหญ่+เขียนครั้งเดียว อยู่ HDD**
`arena.db` ถูกเขียนทุกครั้งที่สถานะงานเปลี่ยน แต่ทั้งไฟล์เล็กมาก ส่วน artifacts
เขียนครั้งเดียวแล้วแทบไม่อ่านอีก จนกว่าจะมีคนขอดู replay

### ⚠️ ตัวกินพื้นที่ที่ใหญ่ที่สุดไม่ได้อยู่ใน `--data`

วัดจากเครื่อง dev ที่ใช้งานมาไม่กี่วัน

| | ขนาดจริง |
|---|---|
| **Docker (image + build cache)** | **14.3 GB** — ในนั้นเป็น cache ที่ลบได้ 11.3 GB |
| `.venv` | 64 MB |
| image `arena/vacuum:cpu` | 365 MB |
| replay ต่อหนึ่ง run (30 episode) | 120 KB |
| zip ของ agent ที่ไม่มี weights | 8 KB |
| `arena.db` | 4 KB |

Docker เก็บของไว้ที่ `/var/lib/docker` บน **SSD ของระบบ** ซึ่งอยู่นอกการแบ่งที่วางไว้
และมันโตทุกครั้งที่ `docker build` — ถ้าไม่จัดการ มันจะเต็มก่อน artifacts หลายเท่า

ทางแก้ที่ตรงจุดที่สุดคือย้าย Docker ไปอยู่บน HDD ด้วย

```bash
sudo mkdir -p $HDD/docker && echo '{"data-root": "'$HDD'/docker"}' | sudo tee /etc/docker/daemon.json && sudo systemctl restart docker
```

ถ้าไม่อยากย้าย อย่างน้อยต้องล้าง cache หลัง build ทุกครั้ง

```bash
docker builder prune -af
```

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

### ⚠️ Python 3.11 — `apt install python3.11` ใช้ไม่ได้บน Mint

Mint ไม่มี `python3.11` ใน repo มาตรฐาน ไม่ว่ารุ่นไหน

| Mint | ฐาน | python3 ที่ได้ |
|---|---|---|
| 21.x | Ubuntu 22.04 | 3.10 |
| 22.x | Ubuntu 24.04 | 3.12 |

**เวอร์ชันนี้เป็น load-bearing ห้ามใช้ตัวที่ใกล้เคียง** —
[`pyproject.toml`](../envs/cp463-vacuum/pyproject.toml) ตรึงไว้ที่ `==3.11.*` คู่กับ
`numpy==2.1.*` เพราะ `numpy.random.Generator` ไม่การันตี stream ข้ามเวอร์ชัน
ใช้ 3.10 หรือ 3.12 แล้ว **ผังห้องของ seed เดิมจะเปลี่ยน** คะแนนทุกค่าที่ประกาศไปแล้ว
รวมถึงหมุด baseline จะใช้เทียบไม่ได้ทันที และจะไม่มีอะไรฟ้อง นอกจาก `pin_baselines --check`
ในข้อ 4

ใช้ `uv` ดึง Python 3.11 มาต่างหาก — ไม่ต้องเพิ่ม PPA และไม่แตะ Python ของระบบ
(เป็นตัวเดียวกับที่ repo นี้ใช้ build wheel อยู่แล้ว)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.11
```

ให้ผู้ใช้ที่จะรันบริการเข้ากลุ่ม docker แล้ว login ใหม่

```bash
sudo usermod -aG docker $USER
```

## 2. โค้ดกับของลับ

ตั้งตัวแปรสองตัวนี้ก่อน แล้วคำสั่งที่เหลือคัดลอกไปวางได้เลย

```bash
export ARENA=~/VRU-AI/projects/colosseum && export HDD=/media/$USER/colosseum
```

```bash
mkdir -p $ARENA && cd $ARENA && mkdir -p $HDD/artifacts $HDD/work
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
cd $ARENA/app && uv venv --python 3.11 && uv pip install -e envs/cp463-vacuum -e .
```

ยืนยันว่าได้ 3.11 จริง — ข้อนี้พลาดแล้วเงียบ

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

ติดตั้ง `cloudflared` ตาม [คู่มือของ Cloudflare](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
แล้วรันสามคำสั่งนี้ — **รันจากโฟลเดอร์ไหนก็ได้** เพราะมันเขียนลง `~/.cloudflared/` เสมอ

ขั้นแรกเปิดเบราว์เซอร์ให้ authorize แล้วเลือกโซน `vru-ai.com`

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

## 6. ให้มันรันเองหลังไฟดับ

เทอมหนึ่งยาว 16 สัปดาห์ — บริการที่ต้องมีคนมา `ssh` เข้าไปสตาร์ทใหม่ทุกครั้งที่ไฟดับ
จะดับจริงในสัปดาห์ที่นิสิตกำลังเร่งส่งงาน

```bash
sudo cp $ARENA/app/deploy/systemd/arena-api.service /etc/systemd/system/ && sudo systemctl enable --now arena-api
```

```bash
sudo cloudflared service install && sudo systemctl enable --now cloudflared
```

ตรวจว่าทั้งสองขึ้นแล้ว

```bash
systemctl status arena-api cloudflared --no-pager && curl -s https://colosseum-api.vru-ai.com/api/health
```

---

## หลังติดตั้งเสร็จ ต้องวัดสองอย่าง

ทั้งคู่ยังไม่รู้ค่าจริง เพราะที่วัดไปเป็น quick tunnel ซึ่งไม่บังคับเพดานของ zone

**เพดานขนาดไฟล์** — แผน Free ประกาศไว้ที่ 100 MB · ต้องรู้ค่าจริงก่อนประกาศให้นิสิต
เพราะทีมที่ส่ง policy ที่เทรนมาอาจมี weights หลักสิบ MB

```bash
dd if=/dev/zero of=/tmp/blob.bin bs=1M count=105 && curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" -F "file=@/tmp/blob.bin" https://colosseum-api.vru-ai.com/api/health
```

**timeout ของ request** — Cloudflare ตัดที่ 100 วินาทีด้วย error 524 · การรับ submission
ต้องจบเร็วกว่านั้นเสมอ (การตรวจแบบ static ไม่รันโค้ด จึงเร็ว) แต่ต้องยืนยัน

## เดินเครื่องประจำวัน

| | คำสั่ง |
|---|---|
| ดู log | `journalctl -u arena-api -f` |
| อัพเดตโค้ด | `cd $ARENA/app && git pull && sudo systemctl restart arena-api` |
| สำรองข้อมูล | `$ARENA/data/` (เล็ก · สำคัญที่สุด) **และ** `$HDD/artifacts/` |
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
