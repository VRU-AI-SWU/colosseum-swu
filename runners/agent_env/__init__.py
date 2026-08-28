"""task template `agent_env` — โจทย์แบบ agent เดินในสภาพแวดล้อม (CP463)

    plugin.py      สัญญาที่ environment ต้องทำตามเพื่อให้ runner รันได้
    runner.py      ฝั่ง trusted — ถือ env · ถือ seed · นับคะแนน
    agent_host.py  ฝั่ง untrusted — อยู่ใน container โหลด `agent.py` ของนิสิต
    messages.py    ชื่อข้อความเฉพาะของโจทย์ชนิดนี้ (อยู่ใน container ด้วย)
    sandbox.py     image + host module ของโจทย์นี้ — **ฝั่ง trusted**
    validate.py    ตรวจ zip ก่อนเข้าคิว

ส่วนกล่องกับโปรโตคอลใช้ของกลางที่ `runners/sandbox/`

⚠️ **ไฟล์นี้ต้องไม่ import อะไรเลย** — `agent_host` กับ `messages` อยู่ในแพ็กเกจนี้
และทั้งคู่ถูกคัดลอกเข้า container ของนิสิต การ import ที่ระดับแพ็กเกจจะลากไฟล์นั้น
เข้าไปใน image ด้วย ซึ่งเป็นวิธีที่ไฟล์ฝั่ง trusted จะหลุดเข้ากล่องแบบเงียบๆ
(`SANDBOX` จึงอยู่ที่ `sandbox.py` ไม่ใช่ที่นี่)
"""
