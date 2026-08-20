# เปิด API ออกสู่อินเทอร์เน็ตด้วย Cloudflare Tunnel

เครื่องในมหาวิทยาลัยเป็นที่ที่ **seed กับฐานข้อมูลอยู่** และตาม
[README §10.4](../../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries) มันต้องอยู่ที่นั่นเท่านั้น
ส่วนหน้าเว็บที่นิสิตเปิดอยู่บน Cloudflare — tunnel คือสะพานเดียวที่เชื่อมสองฝั่ง
**โดยไม่ต้องขอเปิด inbound port กับฝ่าย IT**

```
  นิสิต ──► colosseum.vru-ai.com        (Cloudflare Pages · หน้าเว็บ static)
              │  fetch JSON
              ▼
          colosseum-api.vru-ai.com      (Cloudflare edge)
              │  QUIC · UDP 7844 · เป็น cloudflared ที่ต่อ *ออกไป*
              ▼
          เครื่องในมหาวิทยาลัย            127.0.0.1:8000  arena serve
              └── seeds · arena.db · โค้ดเฉลย  ← ไม่ข้ามสะพาน
```

ผลการทดสอบบนเครือข่ายมหาวิทยาลัยอยู่ที่ [README §10.1](../../README.md#101-ภาพรวม-hybrid-web-บน-cloud--runner-ในมหาวิทยาลัย)

## ชื่อสองอย่างที่ต้องไม่สับสน

| | คือ | ค่าที่ใช้ |
|---|---|---|
| ชื่อ tunnel | ชื่อของ**เครื่อง**ใน Cloudflare account · นิสิตไม่เคยเห็น | `porta-triumphalis` |
| hostname | ชื่อที่**นิสิตพิมพ์** | `colosseum-api.vru-ai.com` |

ตั้งชื่อ tunnel ตาม**ทางเข้า**ไม่ใช่ตามบริการที่อยู่ข้างหลัง เพราะทางเข้าเดียวจะเสิร์ฟ
หลาย competition (Vacuum, Tool-use Agent) และปีหน้ายังใช้ชื่อเดิมได้ — ปีการศึกษาอยู่ใน
competition slug (`cp463-vacuum-1-2026`) ไม่ใช่ใน hostname

**`porta-triumphalis` อยู่บนเครื่อง GPU ในมหาวิทยาลัย** ไม่ใช่เครื่องอื่น เพราะ API
ต้องอ่าน seed และของลับถูก clone ไว้ที่ `/srv/arena/secrets/` บนเครื่องนั้นเครื่องเดียว
([README §10.4](../../README.md#104-ขอบเขตความไว้วางใจ-trust-boundaries))

### จะต้องมี tunnel ตัวที่สองเมื่อไร

⚠️ **ไม่ใช่ "เมื่อมีเครื่องเพิ่ม"** — tunnel มีไว้ให้สิ่งที่ต้อง**ถูกเรียกจากข้างนอก**
เครื่องที่รันแต่ worker ไม่ต้องมี tunnel เลย เพราะ [§10.1](../../README.md#101-ภาพรวม-hybrid-web-บน-cloud--runner-ในมหาวิทยาลัย)
ออกแบบให้ runner เป็นฝ่ายต่อ WebSocket **ออกไป** หา cloud ซึ่งเป็นเหตุผลทั้งหมดที่เลือก
แบบนั้น (ไม่ต้องขอเปิด inbound port กับฝ่าย IT) ต่อให้แล็บได้งบซื้อเครื่องเพิ่มอีกสิบเครื่อง
เพื่อมารันงาน ก็ยังใช้ tunnel ตัวเดียว

จะต้องมีตัวที่สองก็ต่อเมื่อมี**บริการที่สองที่ต้องมี hostname สาธารณะของตัวเอง** เช่น
instance สำหรับทดสอบที่แยกจากของจริง หรือเซิร์ฟเวอร์ replay ที่แยกออกไปต่างหาก
ซึ่งอาจไม่เกิดขึ้นเลย

ถ้าถึงวันนั้น ใช้ชื่อประตูบานอื่นของโคลอสเซียมต่อได้ — `porta-libitinaria` (ประตูที่ใช้
หามผู้แพ้ออก) เหมาะกับ instance ทดสอบที่พังบ่อยเป็นปกติ

| ชื่อ | คือ | ที่มา |
|---|---|---|
| `porta-triumphalis` | tunnel ของ API — ทางเข้าเดียวจากอินเทอร์เน็ต | ประตูแห่งชัยชนะ ทางที่ขบวนผู้ชนะเดินเข้า |
| `colosseum-hypogeum` | repo ของลับ (มีอยู่แล้ว) | ห้องใต้ดินใต้พื้นสังเวียน ที่ซ่อนของก่อนขึ้นเวที |

## ติดตั้งครั้งเดียว

ขั้นที่ 1 ต้องเปิดเบราว์เซอร์เพื่อ authorize แล้วเลือกโซน `vru-ai.com`

```bash
cloudflared tunnel login
```

```bash
cloudflared tunnel create porta-triumphalis
```

ผูก DNS — คำสั่งนี้สร้าง CNAME ในโซน `vru-ai.com` ให้เอง

```bash
cloudflared tunnel route dns porta-triumphalis colosseum-api.vru-ai.com
```

จากนั้นแก้ `config.yml` ในโฟลเดอร์นี้ ใส่ `<TUNNEL-ID>` ที่ได้จากขั้นที่ 2
(ดูซ้ำได้ด้วย `cloudflared tunnel list`) แล้วคัดลอกไปที่ `~/.cloudflared/config.yml`

## รันประจำวัน

สองอย่างนี้ต้องรันคู่กัน — tunnel ส่งต่อไปที่ `127.0.0.1:8000` เท่านั้น

```bash
ARENA_SECRETS=/path/to/colosseum-hypogeum python -m core.cli serve --host 127.0.0.1 --port 8000 --real-seeds --data /srv/arena --allow-origin https://colosseum.vru-ai.com
```

```bash
cloudflared tunnel run porta-triumphalis
```

⚠️ **`--host 127.0.0.1` ไม่ใช่ `0.0.0.0`** — ให้เข้าถึงได้ทางเดียวคือผ่าน tunnel
ถ้า bind กว้างกว่านี้ ใครก็ตามที่อยู่บนเครือข่ายมหาวิทยาลัยจะยิง API ได้ตรงๆ
โดยข้าม Cloudflare ทั้งหมด ซึ่งทำให้ rate limit และ Access ที่จะใส่ทีหลังไร้ความหมาย

⚠️ **`--allow-origin` ต้องตรงกับโดเมนของหน้าเว็บเป๊ะๆ** และ**ห้ามใส่ `*`** — ทุก
endpoint ยืนยันตัวตนด้วย `Authorization: Bearer <team token>` การเปิดให้ทุกโดเมน
เรียกได้แปลว่าหน้าเว็บใดก็ตามที่นิสิตเปิดอยู่ ยิง request ในนามของทีมได้ถ้าดักโทเคนไปได้

## ยังไม่ได้ตรวจ — ต้องทำหลังตั้งเสร็จ

ตัวเลขที่วัดไว้ใน §10.1 มาจาก **quick tunnel** (`trycloudflare.com`) ซึ่งไม่บังคับเพดาน
ของ zone · แผน Free ประกาศเพดาน request body ที่ 100 MB และ origin timeout 100 วินาที
(error 524) ต้องวัดซ้ำบน `colosseum-api.vru-ai.com` จริงก่อนประกาศเพดานไฟล์ให้นิสิต

```bash
dd if=/dev/zero of=/tmp/blob.bin bs=1m count=105 && curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" -F "file=@/tmp/blob.bin" https://colosseum-api.vru-ai.com/api/health
```
