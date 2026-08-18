"""agent ตัวแรกของคุณ — รันได้เลยแต่ยังโง่มาก

ตัวนี้เดินสุ่มแล้วดูดเมื่อเจอฝุ่นใต้ตัวเอง ได้คะแนนราวๆ ระดับ 🥉 Bronze
งานของคุณคือทำให้มันดีกว่านี้

    python -m vacuum.selfcheck                       # ตรวจว่า environment ตรงกับ grader
    arena eval --config main --seeds 1-20            # วัดผลในเครื่อง ไม่กินโควตา
    arena submit cp463-vacuum-1-2026 --note "..."    # ส่งจริง
"""

import numpy as np


class Agent:
    """สามเมธอดนี้คือทั้งหมดที่ระบบรู้จักเกี่ยวกับคุณ

    จะเก็บ state อะไรไว้ข้างในก็ได้ — แผนที่ที่สะสมเอง, ความจำ, โมเดลที่เทรนมา
    ระบบเห็นแค่ว่าคุณคืน action อะไรออกมา
    """

    def __init__(self, config: dict):
        """เรียกครั้งเดียวตอนเริ่ม run

        `config` มีเท่าที่ประกาศไว้ใน spec — **ไม่มีผังห้อง ไม่มี seed**

            width, height              ขนาดห้อง
            observation                "full" | "local" | "sensor"
            observation_window         ขนาดหน้าต่าง (เมื่อ observation = "local")
            max_steps                  จำนวน timestep สูงสุดต่อ episode
            action_noise               โอกาสเดินพลาดทิศ
            sticky_dirt                สัดส่วนช่องที่ต้องดูดสองครั้ง
            sensor_noise               โอกาสที่แต่ละค่าใน observation ถูกพลิก

        💡 ค่าสามตัวสุดท้ายเป็น **แบบจำลองของความไม่แน่นอนที่คุณได้มาฟรี** —
           agent ที่ใช้มันจริงจังทำได้ดีกว่า agent ที่เชื่อสิ่งที่เห็นตรงๆ มาก
        """
        self.width = config["width"]
        self.height = config["height"]
        self.rng = np.random.Generator(np.random.PCG64(0))

    def reset(self, episode_info: dict) -> None:
        """เรียกก่อนเริ่มทุก episode

        ⚠️ **ต้องล้าง state ภายในให้หมดจริง** — แผนที่ที่สะสมไว้ ตัวนับ ความจำ ทุกอย่าง

        ตรวจเองก่อนส่งด้วย **`arena eval --check-reset`** — ระบบตรวจข้อนี้ตอนรับ submission
        และ**ปฏิเสธ**ถ้าไม่ผ่าน ถ้าไม่ตรวจเอง อาการที่เห็นจะเป็นแค่ "คะแนนต่ำกว่าที่ควร"
        แล้วคุณจะไล่ debug อัลกอริทึมผิดทาง

        หมายเหตุ: `episode_info` มีคีย์**ชุดเดียวกับ `config`** ที่ส่งเข้า `__init__`
        — ไม่มีข้อมูลเฉพาะ episode ไม่มี seed ไม่มีเลข episode
        มันมีไว้เผื่อ agent ที่ไม่ได้เก็บ config ไว้ตอน `__init__` เท่านั้น
        """
        self.rng = np.random.Generator(np.random.PCG64(0))

    def act(self, observation) -> int:
        """คืน action หนึ่งตัวต่อหนึ่ง timestep

            0 = UP     1 = DOWN    2 = LEFT
            3 = RIGHT  4 = SUCK    5 = IDLE

        `observation` เป็น dict สามคีย์

            grid     รูปร่างขึ้นกับโหมด — **ทั้งสามโหมดถูกใช้จริงคนละ phase**

                     "local"  (Main, Final): float32 (3, k, k) — หน้าต่างรอบตัว
                              channel 0 = สิ่งกีดขวาง · 1 = ฝุ่น · 2 = ช่องที่เคยเหยียบ
                              **คุณอยู่ตรงกลางหน้าต่างเสมอ** ที่ index (k//2, k//2)
                              ค่านอกขอบห้อง: สิ่งกีดขวาง = 1.0 · ฝุ่น = 0.0

                     "full"   (Warm-up): float32 (4, H, W) — ทั้งห้อง
                              channel 0–2 เหมือน local · **channel 3 = ตำแหน่งคุณแบบ one-hot**

                     "sensor" : float32 (5, 2) — 5 ช่องเรียงตาม
                              [ช่องที่ยืนอยู่, UP, DOWN, LEFT, RIGHT]
                              แต่ละแถวเป็น [สิ่งกีดขวาง, ฝุ่น]

            pos      float32 (2,) — ตำแหน่งของคุณในห้อง normalize ให้อยู่ใน [0, 1]
            scalars  float32 (2,) — [เวลาที่ผ่านไป / max_steps, แบตที่เหลือ]
                     ⚠️ ปีนี้ปิดระบบแบตไว้ทุก phase → `scalars[1]` เป็น 1.0 เสมอ
                        ไม่ต้องวางแผนเรื่องแบต

        ⚠️ **ไม่มี coverage รวมใน observation** โดยตั้งใจ — คุณไม่รู้ว่าเก็บฝุ่นไปกี่ % แล้ว
           ต้องประมาณเอาเองจากสิ่งที่เห็น

        ⚠️ **กับดักที่เจอบ่อยที่สุด: การแปลง `pos` กลับเป็นพิกัด**

            x = int(obs["pos"][0] * self.width)          # ❌ ผิด — ได้ x == width ตอนอยู่ขอบขวา
            x = round(obs["pos"][0] * (self.width - 1))  # ✅ ถูก

        `pos` ถูกหารด้วย `width − 1` ไม่ใช่ `width` แบบผิดจะทำให้ `IndexError`
        เฉพาะตอนหุ่นไปถึงขอบ ซึ่งอาจไม่เกิดเลยใน seed ที่คุณทดสอบ
        มีตัวช่วยให้แล้วที่ `vacuum.baselines.common.decode_pos`
        """
        k = observation["grid"].shape[-1]
        dirty_here = observation["grid"][1][k // 2][k // 2] > 0.5

        if dirty_here:
            return 4  # SUCK

        # ── ตรงนี้คือที่ที่งานของคุณเริ่ม ──────────────────────────────
        # ตัวอย่างนี้เดินสุ่ม ซึ่งแย่มากเพราะมันลืมทุกอย่างที่เพิ่งเห็น
        #
        # ทิศทางที่ควรลอง เรียงจากง่ายไปยาก
        #   1. จำแผนที่ — สะสมสิ่งที่เห็นไว้ใน array ขนาด (height, width)
        #      แล้วเดินไปหาฝุ่นที่รู้ว่าอยู่ตรงไหน แทนที่จะเดินมั่ว
        #   2. สำรวจอย่างมีระบบ — เดินไปหา "ขอบของสิ่งที่ยังไม่เคยเห็น" (frontier)
        #   3. **จัดการความไม่แน่นอน** — เมื่อ sensor_noise > 0 สิ่งที่เห็นผิดได้
        #      อย่าเชื่อค่าล่าสุด ให้รวมหลักฐานจากการเห็นซ้ำหลายครั้ง
        #   4. เรียนรู้ policy จากประสบการณ์ — ดู examples/train_ppo.py
        #
        # `vacuum.baselines` มีตัวอย่างที่รันได้จริงของข้อ 1–3 ให้อ่าน
        return int(self.rng.integers(0, 4))
