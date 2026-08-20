# หน้าเว็บ leaderboard

ไฟล์เดียว ไม่มีขั้นตอน build ไม่มี framework — `index.html` เปิดตรงๆ ก็ทำงาน

## ทำไมไม่มี build step

หน้านี้มีหน้าที่เดียวคือแสดง JSON จาก API ให้อ่านง่าย การใส่ bundler เข้ามาแปลว่า
ทุกคนที่จะแก้สีหรือเพิ่มคอลัมน์ต้องลง toolchain ก่อน ซึ่งไม่คุ้มกับสิ่งที่ได้
ปลายทางของ API เดาจาก `location.hostname` เอาเอง จึงไม่ต้องมีไฟล์ config ต่อ environment

    localhost / 127.0.0.1   →  http://localhost:8000
    ที่อื่น                  →  https://colosseum-api.vru-ai.com

## รันตอนพัฒนา

ต้องรันสองอย่าง — API กับตัวเสิร์ฟไฟล์ static · **`--allow-origin` ต้องตรงกับ origin
ของหน้าเว็บเป๊ะๆ** ไม่งั้นเบราว์เซอร์บล็อกทุก request

```bash
ARENA_SECRETS=/path/to/colosseum-hypogeum python -m core.cli serve --port 8000 --real-seeds --allow-origin http://localhost:4321
```

```bash
python3 -m http.server 4321 --directory web --bind 127.0.0.1
```

แล้วเปิด <http://localhost:4321> ใส่โทเคน `team-1`

## deploy

เป็น **Worker ที่เสิร์ฟ static asset อย่างเดียว** ไม่ใช่ Pages — แบบเดียวกับ `vru-ai-web`
ที่เสิร์ฟเว็บแล็บอยู่แล้ว ตั้งค่าทั้งหมดอยู่ใน [`wrangler.jsonc`](../wrangler.jsonc) ที่ราก repo
รวมถึงการผูกโดเมน จึงเห็นจากในโค้ดได้ว่าอะไรชี้มาที่นี่

```bash
npx wrangler deploy
```

ขึ้น <https://colosseum.vru-ai.com> · ฝั่ง API อนุญาต origin นี้ไว้แล้วใน
[`deploy/systemd/arena-api.service`](../deploy/systemd/arena-api.service)
(`--allow-origin https://colosseum.vru-ai.com`)

ไฟล์ทุกไฟล์ใน `web/` ถูกเสิร์ฟสาธารณะ — ของที่ไม่ควรให้นิสิตเห็นต้องใส่ใน `.assetsignore`
(README นี้อยู่ในนั้นแล้ว)

## สิ่งที่หน้านี้ทำและไม่ทำ

ทำ — leaderboard สาธารณะ · หมุด baseline แทรกตามคะแนน · เป้าหมายถัดไปของทีมคุณ
([README §6.2](../README.md) บอกว่าข้อนี้คือหัวใจ: ทุกทีมต้องมีเป้าที่ทำได้เสมอ) ·
สถานะคิว · รีเฟรชเองทุก 30 วินาที

**ไม่ทำ — การส่งงาน** ยังต้องใช้ `arena submit` จาก terminal โดยตั้งใจ เพราะการส่งงาน
ต้องแพ็กโฟลเดอร์ทั้งชุดพร้อมข้ามไฟล์ที่ไม่ควรส่ง (`models/`, `.venv/`) ซึ่ง CLI ทำให้แล้ว
การทำอัพโหลดผ่านเว็บจะกลายเป็นการสอนให้นิสิต zip เองแล้วเจอปัญหาไฟล์เกินขนาด

## โทเคน

ตอนนี้ยืนยันตัวตนด้วยโทเคนของทีม (`team-1`) ที่นิสิตพิมพ์เอง เก็บใน `localStorage`
ของเบราว์เซอร์ · เป็นของชั่วคราวจนกว่าจะมี Google OAuth ([README §11](../README.md))

⚠️ **โทเคนคือรหัสผ่าน** — ใครมีก็ส่งงานในนามทีมนั้นได้ ตอนเปลี่ยนไปใช้ OAuth
ต้องเอาช่องนี้ออกด้วย ไม่ใช่ปล่อยไว้เป็นทางเข้าสำรอง
