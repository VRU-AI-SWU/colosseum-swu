# CP463 · Competition 1 — Environment Implementation Spec

**Build spec** ของ `vacuum_gridworld` v1.0.0 — เอกสารนี้ต้องละเอียดพอที่คนสองคนอ่านแล้ว implement ออกมาได้ผลลัพธ์ตรงกันทุกบิต
ถ้าอ่านแล้วยังต้องตัดสินใจอะไรเอง แปลว่าเอกสารนี้ยังไม่สมบูรณ์ — โปรดแจ้งเพื่อเติม

เอกสารประกอบ: [ภาพรวมโจทย์และการให้คะแนน](overview.md) · [template](../../../../task-templates/agent-vs-environment-rl.md)

> **ทำไมต้องเป๊ะขนาดนี้** — starter kit ที่นิสิตใช้เทรน กับ environment ที่ใช้ตัดสิน ต้องเป็นสิ่งเดียวกันทุกประการ
> ถ้าต่างกันแม้แต่การเรียงแกนของ observation นิสิตทั้งห้องจะเทรนบนสิ่งที่ไม่ตรงกับตอนวัด และจะพังแบบเงียบๆ หาสาเหตุยากมาก
> §14 กำหนด conformance test ที่ทั้งสองฝั่งต้องผ่านเหมือนกัน

---

## สารบัญ

1. [ระบบพิกัดและโครงสร้าง state](#1-ระบบพิกัดและโครงสร้าง-state)
2. [วินัยเรื่อง RNG](#2-วินัยเรื่อง-rng)
3. [อัลกอริทึมสร้างห้อง](#3-อัลกอริทึมสร้างห้อง)
4. [Observation encoding](#4-observation-encoding)
5. [Action และ transition](#5-action-และ-transition)
6. [Termination](#6-termination)
7. [การคิดคะแนน](#7-การคิดคะแนน)
8. [Gymnasium API](#8-gymnasium-api)
9. [Replay format](#9-replay-format)
10. [Baseline agents](#10-baseline-agents)
11. [Config ของทั้ง 3 phase](#11-config-ของทั้ง-3-phase)
12. [โครงสร้าง package และเวอร์ชัน](#12-โครงสร้าง-package-และเวอร์ชัน)
13. [Submission validation](#13-submission-validation)
14. [Conformance tests](#14-conformance-tests)
15. [ค่าที่ยังต้อง calibrate](#15-ค่าที่ยังต้อง-calibrate)

---

## 1. ระบบพิกัดและโครงสร้าง state

- Grid ขนาด `W × H` cell พิกัด `(x, y)` โดย `x ∈ [0, W)` ไปทางขวา `y ∈ [0, H)` ไปทาง**ล่าง** จุดกำเนิดอยู่มุมซ้ายบน
- array ทุกตัวเป็น **row-major** index ด้วย `[y][x]` → shape `(H, W)`
- **ไม่มีขอบกำแพงในกริด** — cell นอกช่วง `[0,W) × [0,H)` ถือเป็นกำแพงโดยปริยาย ค่า `width: 20` หมายถึงมีช่องใช้งานได้ 20 ช่องจริง
- `flat_index(x, y) = y * W + x` ใช้เป็นตัวแทน cell ในรูปตัวเลขเดียว (ใช้ใน replay และการ tie-break)

**State ของ episode**

| ตัวแปร | ชนิด | ความหมาย |
|---|---|---|
| `obstacle` | `bool[H, W]` | คงที่ตลอด episode |
| `dirt` | `bool[H, W]` | `True` = ยังสกปรก จะเป็น `False` เมื่อดูดสำเร็จ · เป็น `True` ได้เฉพาะบน cell ที่ไม่ใช่ obstacle |
| `sticky` | `bool[H, W]` | คงที่ · subset ของ `dirt` ตอนเริ่ม |
| `sticky_hit` | `bool[H, W]` | เคยพยายามดูด cell sticky นั้นไปแล้วกี่ครั้ง (เก็บเป็น bool เพราะต้องการแค่ครั้งเดียว) |
| `visited` | `bool[H, W]` | cell ที่หุ่นเคยยืน (รวมจุดเริ่มต้น) |
| `pos` | `(int, int)` | ตำแหน่งหุ่นปัจจุบัน |
| `t` | `int` | จำนวน timestep ที่ผ่านไปแล้ว เริ่มที่ 0 |
| `battery` | `int \| None` | คงเหลือ · `None` = ไม่จำกัด |
| ตัวนับ | `int` | `cleaned`, `collisions`, `redundant_sucks`, `sticky_fails`, `slips` |

---

## 2. วินัยเรื่อง RNG

ใช้ `numpy.random.Generator(PCG64(...))` เท่านั้น **ห้ามใช้ `random` ของ Python หรือ global numpy RNG**

**เหตุผล 3 ข้อ เรียงตามความสำคัญ**

1. **ห้าม RNG แบบ global** — `np.random.seed()` และ `random.seed()` ตั้งค่าสถานะระดับ process
   ถ้า `agent.py` ของนิสิตเขียน `np.random.seed(42)` (หรือ import library ที่ทำแบบนั้นเอง เช่น stable-baselines3, torch dataloader)
   **ผังห้องที่ทีมนั้นเจอจะเปลี่ยนไป** → ทุกทีมเจอห้องคนละแบบ → leaderboard ไร้ความหมาย
   และอาการที่เห็นคือ "คะแนนไม่ตรงกันเฉยๆ" ไม่ใช่ error ทำให้ debug แทบไม่ได้
   พอ environment ถือ `Generator` เป็น object ของตัวเอง โค้ดข้างนอกแตะไม่ได้เลย — ปัญหาหายไปโดยโครงสร้าง (ดู test #13)
2. **ต้องแตกสายสุ่มอิสระได้** — `SeedSequence` แตกสาย layout / noise / sensor ได้อย่างถูกต้องทางสถิติ
   ถ้าใช้ `RandomState` แบบเดิมคนมักแฮ็กด้วย `seed+1`, `seed+2` ซึ่งให้สายที่สหสัมพันธ์กันในบาง bit generator
3. **⚠️ `Generator` ไม่รับประกัน stream ข้ามเวอร์ชัน numpy** (ต่างจาก `RandomState` ที่ถูกแช่แข็งตาม NEP 19)
   **การ pin `numpy == 2.1.*` ใน §12 จึงเป็นเงื่อนไขที่ระบบพึ่งพาอยู่ ไม่ใช่แค่ความสะอาดของ dependency**
   ถ้าอัพ numpy ผังห้องของ seed เดิมอาจเปลี่ยน → ต้องขึ้น `env_version` และ rejudge ทั้งหมด

```python
LAYOUT_STREAM = 0x5EED
NOISE_STREAM  = 0xA11CE

layout_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, LAYOUT_STREAM])))
noise_rng  = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, NOISE_STREAM])))
```

**แยกสองสายเพราะ** — ถ้าใช้สายเดียว การเปลี่ยน `max_steps` (ซึ่งเปลี่ยนจำนวน draw ของ noise) จะทำให้ผังห้องของ seed เดิมเปลี่ยนไปด้วย
ซึ่งจะทำให้เทียบผลข้าม phase หรือข้ามเวอร์ชันไม่ได้เลย

### ลำดับการ draw จาก `layout_rng` — ห้ามสลับ

1. obstacle layout (§3.1)
2. dirt layout (§3.3) — การบังคับ connectivity (§3.2) ไม่ใช้ RNG
3. sticky selection (§3.4)
4. robot start (§3.5)

### Noise tape — common random numbers

**Noise tape คือการสุ่มค่าไว้ล่วงหน้าทั้งม้วน แล้วเรียกใช้ตามหมายเลข timestep แทนการสุ่มสดตอนต้องใช้**

```python
slip_tape     = noise_rng.random(max_steps)              # float64 ใน [0, 1)
slip_dir_tape = noise_rng.integers(0, 2, max_steps)      # 0 หรือ 1
```

**สร้างล่วงหน้าทั้งเทป และใช้ `tape[t]` ตาม index ของ timestep ไม่ว่า agent จะทำ action อะไร**
— คำสำคัญคือ *อ้างตาม index* ไม่ใช่ *ดึงค่าถัดไป*

#### ทำไมต้องทำแบบนี้

ถ้าดึงค่าสุ่มสดเฉพาะตอนที่ agent เดิน สองทีมที่ทำ action ต่างกันจะหลุดออกจากกันทันที

| timestep | ทีม A | ทีม B |
|---|---|---|
| 0 | เดิน → ใช้ค่าสุ่มลำดับที่ 1 | SUCK → ไม่ดึงค่าสุ่ม |
| 1 | เดิน → ค่าสุ่มลำดับที่ 2 | เดิน → **ค่าสุ่มลำดับที่ 1** |
| 2 | เดิน → ค่าสุ่มลำดับที่ 3 | เดิน → **ค่าสุ่มลำดับที่ 2** |

ทั้งที่อยู่ในห้องเดียวกัน แต่กลับเดินอยู่บน "ดวง" คนละชุดตั้งแต่ timestep แรกที่ action ต่างกัน
พอมีเทป ที่ timestep 2 ทั้งคู่ใช้ `slip_tape[2]` เหมือนกัน — ใครเลือกเดินในจังหวะนั้นก็เจอดวงก้อนเดียวกัน

เทคนิคนี้เรียกว่า **common random numbers** เป็นวิธีมาตรฐานในการลดความแปรปรวนของการเปรียบเทียบในงาน simulation
หลักการเดียวกับการให้นักเรียนสองคนทำ**ข้อสอบชุดเดียวกัน** แทนที่จะสุ่มข้อสอบให้คนละชุด — ความต่างของคะแนนสะท้อนฝีมือมากขึ้นเพราะความยากถูกหักล้างออกไป

**ข้อจำกัดที่ต้องเข้าใจตรงกัน** — มันไม่ได้ทำให้จับคู่กันสมบูรณ์ เพราะที่ timestep 2 ทีม A กับ B ยืนคนละช่อง
ค่าสุ่มก้อนเดียวกันจึงให้ผลต่างกันอยู่ดี สิ่งที่ได้คือ**สหสัมพันธ์บางส่วน** ไม่ใช่การหักล้างทั้งหมด
แต่ราคาที่จ่ายคือศูนย์ (เทป 1,500 ค่า ≈ 24 KB ต่อ episode) จึงไม่มีเหตุผลที่จะไม่ทำ

---

## 3. อัลกอริทึมสร้างห้อง

> **ภาพรวมของทั้ง section** — เป้าหมายคือสร้าง "ห้อง" ที่ **หน้าตาเหมือนห้องจริง** และ **แก้ได้จริง** จากตัวเลข seed ตัวเดียว
> โดยต้องได้ผลเดิมเป๊ะทุกครั้ง
>
> ```
> seed → ① วางสิ่งกีดขวาง → ② ถมช่องที่เข้าไม่ถึง → ③ โปรยฝุ่น → ④ เลือกช่องเหนียว → ⑤ วางหุ่น
> ```
>
> แต่ละขั้นแก้ปัญหาคนละอย่าง และ **ลำดับสลับไม่ได้** เพราะขั้นหลังต้องใช้ผลของขั้นก่อน
> (โปรยฝุ่นก่อนถมช่อง = ฝุ่นอาจไปตกในช่องที่เข้าไม่ถึง → ดูดครบไม่ได้ตลอดกาล)

### 3.1 Obstacle — วางสิ่งกีดขวาง

**แนวคิด** — "สิ่งกีดขวาง" ในห้องจริงคือเฟอร์นิเจอร์ ซึ่ง**เกาะกลุ่มกัน** ไม่ใช่กระจายเป็นจุดๆ ทั่วห้อง
โซฟาหนึ่งตัวกินพื้นที่ติดกันหลายช่อง ตู้ชิดผนังยาวเป็นแนว เราจึงมี generator สองแบบที่ให้ห้องคนละบุคลิก

| generator | วิธีคิด | ห้องที่ได้ | เหมาะกับ |
|---|---|---|---|
| `random` | โรยจุดสุ่มกระจายทั่วห้องอย่างเป็นอิสระ | เหมือนทุ่งเสาหลักกระจัดกระจาย เดินอ้อมง่าย ไม่มีซอกมุม | **Warm-up** — ง่าย ไม่มีกับดักเชิงโครงสร้าง |
| `clustered` | **random walk**: หย่อนจุดเริ่มลงมั่ว แล้วเดินสุ่มต่อไป 3–8 ก้าว ระบายช่องที่เดินผ่านเป็นกำแพง ทำซ้ำจนครบโควตา | ก้อนเฟอร์นิเจอร์รูปร่างบิดเบี้ยว มีซอก มีทางตัน | **Main / Final** — บังคับให้ agent ต้องวางแผนเส้นทาง |

**ทำไม random walk ถึงให้ก้อนที่ดูเป็นธรรมชาติ** — เพราะแต่ละก้าวขยับไปช่องข้างเคียงเท่านั้น ช่องที่ถูกระบายจึงติดกันเสมอ
แต่ทิศทางสุ่ม รูปร่างที่ได้เลยไม่ใช่สี่เหลี่ยมเรียบร้อย — คล้ายกองของที่วางซ้อนกันมากกว่ากล่องเรขาคณิต
ความยาว 3–8 ก้าวเป็นค่าที่ให้ก้อนใหญ่พอจะเป็นอุปสรรค แต่ไม่ใหญ่จนตัดห้องขาดเป็นสองส่วนบ่อยเกินไป

**สองรายละเอียดที่ดูแปลกแต่จำเป็น**

- *ทำไมเดินทับช่องเดิมได้* — random walk ย้อนกลับมาที่เดิมได้ ถ้าช่องนั้นเป็นกำแพงอยู่แล้วเราไม่นับซ้ำ (`placed` ไม่เพิ่ม)
  ผลคือก้อนแน่นขึ้นแทนที่จะยืดยาว ซึ่งเป็นสิ่งที่ต้องการ
- *ทำไมต้องมี `guard < 10_000`* — ในกรณีสุดขั้ว (ห้องเล็กมาก + density สูงมาก) วงอาจวนหาช่องว่างไม่เจอจนไม่รู้จบ
  `guard` เป็นเบรกฉุกเฉิน **ไม่ใช่ตรรกะปกติ** ถ้าชนเบรกบ่อยแปลว่า config ผิด ไม่ใช่โค้ดผิด

```python
DX = [0, 0, -1, 1]   # UP, DOWN, LEFT, RIGHT
DY = [-1, 1, 0, 0]

def generate_obstacles(rng, W, H, density, generator):
    obs = np.zeros((H, W), dtype=bool)
    target = int(round(density * W * H))
    if target == 0:
        return obs

    if generator == "random":
        idx = rng.permutation(W * H)[:target]
        obs.reshape(-1)[idx] = True

    elif generator == "clustered":
        placed, guard = 0, 0
        while placed < target and guard < 10_000:
            guard += 1
            x = int(rng.integers(0, W))
            y = int(rng.integers(0, H))
            run_len = int(rng.integers(3, 9))          # 3..8
            for _ in range(run_len):
                if not obs[y, x]:
                    obs[y, x] = True
                    placed += 1
                    if placed >= target:
                        break
                d = int(rng.integers(0, 4))
                nx, ny = x + DX[d], y + DY[d]
                if 0 <= nx < W and 0 <= ny < H:
                    x, y = nx, ny
    else:
        raise ValueError(generator)
    return obs
```

`obstacle_generator: rooms` **ยังไม่ implement ใน v1.0.0** — ถ้าจะใช้ใน phase Final ต้องเพิ่มเป็น v1.1.0 พร้อม conformance test ใหม่

### 3.2 บังคับ connectivity — ถมช่องที่เข้าไม่ถึง (ไม่ใช้ RNG)

**ปัญหาที่ต้องแก้** — random walk อาจล้อมพื้นที่จนขาดจากส่วนอื่นโดยไม่ตั้งใจ เกิด "ห้องลับ" ที่หุ่นเดินเข้าไม่ได้

```
█ █ █ █ █ █ █          █ = กำแพง   · = พื้น
█ · · █ · · █          ช่อง (4,1) กับ (5,1) ถูกล้อมสนิท
█ · 🤖 █ · · █          หุ่นเดินไปไม่ถึงตลอดกาล
█ · · █ █ █ █
```

ถ้าปล่อยไว้แล้วฝุ่นดันไปตกในนั้น **coverage จะแตะ 100% ไม่ได้ไม่ว่าจะเก่งแค่ไหน** → `completion_bonus` กลายเป็นของที่แจกไม่ได้
และคะแนนของ seed นั้นจะต่ำผิดปกติเทียบกับ seed อื่นโดยไม่เกี่ยวกับฝีมือ — เป็นความไม่ยุติธรรมที่ซ่อนอยู่ในข้อมูล

**วิธีแก้: ถมทิ้ง ไม่ใช่สุ่มใหม่** — หาก้อนพื้นที่เชื่อมต่อกันที่ใหญ่ที่สุด แล้วเปลี่ยนพื้นที่นอกก้อนนั้นให้เป็นกำแพงไปเลย
ห้องที่เหลือจึงการันตีว่า**ทุกช่องเดินถึงกันได้หมด**

| ทางเลือก | ผล |
|---|---|
| สุ่มผังใหม่จนกว่าจะผ่าน | จำนวนรอบไม่แน่นอน · เปลืองค่าสุ่มไม่คงที่ → **ทำลาย determinism ของสายสุ่ม** · มีโอกาสวนไม่จบ |
| **ถมช่องที่เข้าไม่ถึง** (เลือกใช้) | จบใน pass เดียว · ไม่ใช้ค่าสุ่มเลย · ไม่มีทางวนไม่จบ |

**ราคาที่จ่าย** — จำนวนกำแพงจริงมากกว่า `obstacle_density` ที่ตั้งไว้เล็กน้อย ระบบจึงรายงาน `effective_density` กลับมาใน `info`
ให้รู้ว่าห้องจริงหนาแน่นแค่ไหน (ใช้ตอน calibrate ตาม §15)

```python
free = ~obs
components = connected_components(free, connectivity=4)
largest = component ที่มีจำนวน cell มากที่สุด
          ถ้าเท่ากัน → เลือก component ที่มี flat_index ต่ำสุด
obs |= (free & ~largest)          # ช่องว่างที่เข้าไม่ถึงกลายเป็นกำแพง
```

**ทำแบบนี้แทนการสุ่มใหม่จนกว่าจะผ่าน** เพราะ deterministic แน่นอน จบใน pass เดียว และไม่มีโอกาสวนไม่รู้จบ
ผลข้างเคียงคือ density จริงอาจสูงกว่าที่ตั้งไว้เล็กน้อย — ยอมรับได้ และ `info` จะรายงาน `effective_density` ให้

ถ้าหลังจากนี้ `free_count < 4` ให้ถือว่า config ใช้ไม่ได้ และ **raise error ตอนสร้าง competition** ไม่ใช่ตอนรัน

### 3.3 Dirt — โปรยฝุ่น (อัลกอริทึมเดียว ต่างกันที่น้ำหนัก)

**แนวคิด** — แทนที่จะเขียนอัลกอริทึมแยกต่อรูปแบบการกระจาย เราใช้กลไกเดียว: **ให้คะแนนน้ำหนักกับทุกช่องว่าง แล้วจับสลากตามน้ำหนัก**
เปลี่ยนรูปแบบการกระจาย = เปลี่ยนแค่สูตรน้ำหนัก

| `dirt_distribution` | น้ำหนัก | ผลที่ได้ |
|---|---|---|
| `uniform` | เท่ากันหมด (`w = 1`) | ฝุ่นกระจายทั่วห้องอย่างสม่ำเสมอ |
| `clustered` | สุ่มจุดศูนย์กลางมา `k` จุด แล้ว `w = exp(−d/3)` เมื่อ `d` = ระยะ Manhattan ถึงศูนย์กลางที่ใกล้ที่สุด | ฝุ่นเกาะกลุ่มเป็นหย่อม เหมือนคราบใต้โต๊ะอาหารหรือหน้าประตู |

**`exp(−d/3)` ทำอะไร** — เป็นน้ำหนักที่ลดลงตามระยะห่างจากศูนย์กลาง โดยตัวเลข 3 คือ "รัศมีที่รู้สึกได้"
ช่องที่อยู่ห่างศูนย์กลาง 3 ช่องมีโอกาสสกปรกราว 37% ของช่องที่อยู่ตรงศูนย์กลางพอดี · ห่าง 6 ช่องเหลือราว 14%
ค่านี้ยิ่งน้อยหย่อมยิ่งกระชับ ยิ่งมากยิ่งเบลอจนใกล้เคียง `uniform`

**ทำไมต้องจับสลากแบบไม่คืนที่ ด้วยจำนวนตายตัว** — ทางเลือกที่ง่ายกว่าคือทอยเหรียญรายช่อง (Bernoulli ความน่าจะเป็น `dirt_ratio`)
แต่วิธีนั้นทำให้ **`D0` แกว่งไปมาระหว่าง seed** เช่นตั้ง `dirt_ratio: 0.6` บนพื้น 340 ช่อง อาจได้ 196 บ้าง 215 บ้าง

`D0` คือ **ตัวหารของ coverage** (§7) ถ้ามันแกว่ง ความยากของแต่ละ seed จะต่างกันโดยไม่จำเป็น และเพิ่มความแปรปรวนให้คะแนน
การล็อกจำนวนตายตัวจึงเป็นการลด noise ฟรีๆ ด้วยเหตุผลเดียวกับ noise tape ใน §2

```python
free_idx = flat indices ของ cell ที่ไม่ใช่ obstacle (เรียงจากน้อยไปมาก)
n_dirt = max(1, int(round(dirt_ratio * len(free_idx))))

if dirt_distribution == "uniform":
    w = np.ones(len(free_idx))
elif dirt_distribution == "clustered":
    k = max(1, int(round(len(free_idx) * dirt_ratio / 25)))
    centers = rng.choice(free_idx, size=k, replace=False)
    d = manhattan distance ของแต่ละ free cell ไปยัง center ที่ใกล้ที่สุด
    w = np.exp(-d / 3.0)
else:
    raise ValueError

chosen = rng.choice(free_idx, size=n_dirt, replace=False, p=w / w.sum())
dirt.reshape(-1)[chosen] = True
D0 = n_dirt
```

**เลือกแบบไม่คืนที่ด้วยจำนวนตายตัว** ไม่ใช่ Bernoulli รายช่อง → `D0` เท่ากันเป๊ะทุกครั้งสำหรับ config เดียวกัน
ทำให้คะแนนข้าม seed เทียบกันง่ายขึ้นและตัวหารของ coverage ไม่แกว่ง

`dirt_distribution: patchy` ยังไม่ implement ใน v1.0.0

### 3.4 Sticky — เลือกช่องที่ดูดครั้งเดียวไม่ขึ้น

**แนวคิด** — เลียนแบบคราบฝังแน่นที่ต้องดูดซ้ำ และ **มองไม่ออกจากภายนอกว่าช่องไหนเป็น** agent รู้ได้ทางเดียวคือลองดูดแล้วพบว่าฝุ่นยังอยู่

นี่คือ **hidden dynamics** ที่ทำให้ planner บริสุทธิ์เสียเปรียบ (§5 ของ [template](../../../../task-templates/agent-vs-environment-rl.md#5-ปัญหาที่ต้องตัดสินใจ-planning-ชนะ-learning))
เพราะ planner ที่คำนวณเส้นทางล่วงหน้าจะสมมติว่า "ดูดหนึ่งครั้งจบ" เสมอ แล้วเดินจากไปทั้งที่ยังไม่สะอาด
ส่วน policy ที่เรียนรู้จากประสบการณ์จะจับรูปแบบได้ว่าควรตรวจซ้ำหรือดูดสองครั้งในบางสถานการณ์

เลือกจาก**ช่องที่สกปรกอยู่แล้ว**เท่านั้น และใช้จำนวนตายตัวด้วยเหตุผลเดียวกับ §3.3

```python
n_sticky = int(round(sticky_dirt * D0))
if n_sticky > 0:
    chosen = rng.choice(dirty_idx, size=n_sticky, replace=False)   # dirty_idx เรียงจากน้อยไปมาก
    sticky.reshape(-1)[chosen] = True
```

### 3.5 Robot start — วางหุ่น

**แนวคิด** — จุดเริ่มต้นเปลี่ยนความยากของโจทย์มากกว่าที่คิด เพราะมันกำหนดว่า agent ต้อง "สำรวจ" มากแค่ไหนก่อนจะเห็นภาพรวมห้อง

| `start` | กติกา | ผลต่อโจทย์ |
|---|---|---|
| `random` | `rng.choice(free_idx)` | ยากที่สุดและยุติธรรมที่สุด — agent ต้องทำงานได้จากทุกจุด จำ "แผนสำเร็จรูป" ไม่ได้ |
| `corner` | free cell ที่ `x + y` น้อยที่สุด · เสมอ → `flat_index` ต่ำสุด | ง่ายที่สุด — เริ่มจากมุมทำให้กวาดเป็นแนวได้เป็นระบบ เหมาะกับ Warm-up |
| `center` | free cell ที่ระยะ Manhattan ถึง `(W // 2, H // 2)` น้อยที่สุด · เสมอ → `flat_index` ต่ำสุด | กลางทาง — เห็นรอบตัวเร็วแต่ต้องตัดสินใจว่าจะไปทางไหนก่อน |

**ทำไม `corner` และ `center` ต้องมีกฎตัดสินเสมอ** — ช่องที่ `x + y` เท่ากันมีหลายช่อง (เช่น (0,2), (1,1), (2,0))
ถ้าไม่กำหนดว่าจะเลือกอันไหน สอง implementation อาจเลือกคนละช่อง → ผังห้องเดียวกันแต่หุ่นเริ่มคนละที่ → คะแนนไม่ตรงกัน
กฎ "flat_index ต่ำสุด" ทำให้ตัดสินได้แน่นอนเสมอ — เป็นรูปแบบเดียวกับที่ใช้ตัดสินเสมอใน BFS ของ Gold baseline (§10)

หุ่นเริ่มบน cell ที่สกปรกได้ (ไม่ต้องล้างออก) และ `visited[start] = True` ตั้งแต่ก่อน timestep แรก

---

## 4. Observation encoding

ทุกโหมดคืน `dict` โครงสร้างเดียวกัน ต่างกันแค่ `grid`

```python
obs = {
    "grid":    np.ndarray,        # float32 — รายละเอียดตามโหมด
    "pos":     np.float32[2],     # [x / max(W-1, 1), y / max(H-1, 1)]
    "scalars": np.float32[2],     # [t / max_steps, battery_left / battery_init]
}
```

`scalars[1] = 1.0` เสมอเมื่อ `battery: null`

> **`scalars` ไม่มี coverage** โดยตั้งใจ — ในโหมด `local`/`sensor` การบอก coverage รวมเท่ากับแอบให้ข้อมูลทั้งแผนที่
> โหมด `full` คำนวณเองได้จาก channel ฝุ่นอยู่แล้ว จึงไม่ต้องมีในทุกโหมด

### 4.1 `full`

`grid` shape `(4, H, W)` float32 ค่าเป็น 0.0 หรือ 1.0

| channel | ความหมาย |
|---|---|
| 0 | obstacle |
| 1 | dirt (ที่ยังไม่ถูกดูด) |
| 2 | visited |
| 3 | ตำแหน่งหุ่น (one-hot) |

### 4.2 `local`

`grid` shape `(3, k, k)` โดย `k = observation_window` (**ต้องเป็นเลขคี่** ไม่งั้น raise ตอนโหลด config)
หน้าต่างมีหุ่นอยู่ตรงกลางพอดีที่ index `(k//2, k//2)`

| channel | ความหมาย | ค่านอกขอบ grid |
|---|---|---|
| 0 | obstacle | **1.0** (นอกขอบคือกำแพง) |
| 1 | dirt | 0.0 |
| 2 | visited | 0.0 |

ไม่มี channel ตำแหน่งหุ่น เพราะหุ่นอยู่กลางหน้าต่างเสมอ — ตำแหน่งสัมบูรณ์อ่านได้จาก `obs["pos"]`

> **ให้ตำแหน่งสัมบูรณ์เป็นการตัดสินใจโดยตั้งใจ** — ถ้าไม่ให้ นิสิตต้องทำ dead reckoning ภายใต้ `action_noise` ซึ่งยากเกินระดับวิชา
> การเอา `pos` ออกเป็น "คันโยกความยาก" ที่เก็บไว้ใช้ในอนาคตได้ ถ้าปีนี้ง่ายเกินไป

### 4.3 `sensor`

`grid` shape `(5, 2)` float32 — 5 cell เรียงตามลำดับ **`[current, UP, DOWN, LEFT, RIGHT]`** ตายตัว

| column | ความหมาย | ค่านอกขอบ |
|---|---|---|
| 0 | obstacle | 1.0 |
| 1 | dirt | 0.0 |

### 4.4 `sensor_noise` — ใช้ได้กับ **ทุกโหมด** ไม่ใช่เฉพาะโหมด `sensor`

ชื่อ `sensor_noise` สื่อว่าผูกกับโหมด `sensor` แต่ config ที่ใช้จริงตั้ง `sensor_noise > 0`
คู่กับ `observation: local` ทั้ง phase Main และ Final (§11) — ส่วนนี้จึงนิยามให้ครบทุกโหมด

**พลิกเฉพาะ channel ที่เป็นค่าจากเซนเซอร์** คือ `obstacle` กับ `dirt` เท่านั้น

| โหมด | channel ที่ถูกพลิก | channel ที่ไม่ถูกพลิก | จำนวนค่าต่อ 1 observation |
|---|---|---|---|
| `full` | 0 (obstacle), 1 (dirt) | 2 (visited), 3 (ตำแหน่งหุ่น) | `2 × H × W` |
| `local` | 0 (obstacle), 1 (dirt) | 2 (visited) | `2 × k × k` |
| `sensor` | column 0, 1 | — | `2 × 5` |

**เหตุผลที่ไม่พลิก `visited`** — มันคือร่องรอยที่ระบบจำให้ ไม่ใช่ค่าที่อ่านจากเซนเซอร์
และ **ไม่พลิก channel ตำแหน่งหุ่น** เพราะจะทำให้หุ่น "หายไป" หรือโผล่สองที่ในเฟรมเดียว ซึ่งไม่มีความหมายทางกายภาพ

**กติกาการสุ่ม**

- ทุกค่าที่พลิกได้ ถูกพลิกอย่าง**เป็นอิสระต่อกัน**ด้วยความน่าจะเป็น `sensor_noise`
- ใช้สายสุ่มที่สามแยกต่างหาก `SENSOR_STREAM = 0x53E4`
- เทปมี shape `(max_steps + 1, n_sensed)` — มี `+1` เพราะต้องมี observation ที่ `t = 0` จาก `reset()` ด้วย
- **ใช้ `tape[t]` ตาม index ของ timestep เสมอ ไม่ว่า agent จะทำอะไร** ตามหลัก common random numbers เหมือน slip tape
- ลำดับของค่าในแต่ละแถวตรึงเป็น **obstacle ทั้งผืนก่อน แล้วตามด้วย dirt** (row-major ทั้งคู่)
  ลำดับนี้เป็นส่วนหนึ่งของสัญญา — เปลี่ยนเมื่อไหร่ผลของ seed เดิมเปลี่ยน ต้องขึ้น `env_version`
- สร้างเทปเฉพาะเมื่อ `sensor_noise > 0` เท่านั้น (ที่ `full` ห้อง 20×20 เทปจะกิน 2×400×1501 ค่า ≈ 19 MB โดยไม่จำเป็น)

`sensor_noise` ไม่มีผลกับ `pos` และ `scalars`

> **นี่คือคันโยกความยากหลักของโจทย์** — ต่างจาก `action_noise` ที่ทำให้แค่เส้นทางยาวขึ้น
> `sensor_noise` ทำให้**แผนที่ที่ agent สะสมไว้ผิดถาวร** สำหรับช่องที่ไม่ได้กลับไปดูซ้ำ
> ตัวเลขที่วัดได้อยู่ที่ [calibration-2026-08.md](calibration-2026-08.md)

---

## 5. Action และ transition

```
0 = UP     (y − 1)
1 = DOWN   (y + 1)
2 = LEFT   (x − 1)
3 = RIGHT  (x + 1)
4 = SUCK
5 = IDLE
```

การเปลี่ยนสถานะที่ timestep `t` (0-indexed) เมื่อได้รับ action `a` **ทำตามลำดับนี้เท่านั้น**

```python
# ── 1. movement ────────────────────────────────────────────────
if a in (0, 1, 2, 3):
    d = a
    if action_noise > 0 and slip_tape[t] < action_noise:
        perp = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}[a]   # UP/DOWN→LEFT,RIGHT
        d = perp[slip_dir_tape[t]]
        slips += 1
    nx, ny = x + DX[d], y + DY[d]
    if not (0 <= nx < W and 0 <= ny < H) or obstacle[ny, nx]:
        collisions += 1                    # ชน = อยู่ที่เดิม แต่ timestep ยังเดิน
    else:
        x, y = nx, ny
        visited[y, x] = True
    if battery is not None:
        battery -= move_cost

# ── 2. suck ────────────────────────────────────────────────────
elif a == 4:
    if dirt[y, x]:
        if sticky[y, x] and not sticky_hit[y, x]:
            sticky_hit[y, x] = True        # ครั้งแรกไม่ติด
            sticky_fails += 1
        else:
            dirt[y, x] = False
            cleaned += 1
    else:
        redundant_sucks += 1
    if battery is not None:
        battery -= suck_cost

# ── 3. idle ────────────────────────────────────────────────────
elif a == 5:
    pass                                   # ไม่เสียแบต แต่เสีย timestep

t += 1
```

**กติกาที่ต้องยึดตามนี้เป๊ะ**

| กรณี | พฤติกรรม |
|---|---|
| `action_noise` ใช้กับอะไร | **เฉพาะ action 0–3** ไม่ใช้กับ SUCK และ IDLE |
| ลื่นแล้วไปชนกำแพง | นับ **ทั้ง** `slips` และ `collisions` หุ่นอยู่ที่เดิม |
| ชนกำแพง | เสีย timestep · เสีย `move_cost` · `collisions += 1` · ตำแหน่งไม่เปลี่ยน |
| SUCK บน sticky ครั้งแรก | **ไม่นับเป็น `redundant_suck`** (agent ไม่มีทางรู้ล่วงหน้า การลงโทษจะไม่ยุติธรรม) นับเป็น `sticky_fails` ซึ่งไม่มีผลต่อคะแนน |
| SUCK บน sticky ครั้งที่สอง | สำเร็จเสมอ ไม่ว่าจะเว้นไปนานแค่ไหนหรือเดินออกไปแล้วกลับมา |
| SUCK บน cell สะอาด | `redundant_sucks += 1` (มีผลต่อคะแนน) |
| IDLE | ไม่เสียแบต ไม่มี penalty แต่เสีย 1 timestep ซึ่งกดค่า AUC เอง |
| ตำแหน่งเริ่มต้น | `visited[start] = True` ก่อน timestep แรก |

---

## 6. Termination

ตรวจ **หลังจาก** transition ของ timestep เสร็จ ตามลำดับนี้

| ลำดับ | เงื่อนไข | ค่าใน `info` |
|---|---|---|
| 1 | `cleaned == D0` | `terminated=True`, `reason="complete"` |
| 2 | `battery is not None and battery <= 0` | `terminated=True`, `reason="battery"` |
| 3 | `t >= max_steps` | `truncated=True`, `reason="max_steps"` |

`stop_on_full_coverage: false` จะปิดเงื่อนไขข้อ 1 (มีไว้ใช้ debug เท่านั้น **ห้ามใช้ใน config ที่ตัดสินคะแนน** เพราะจะทำให้ AUC เปลี่ยนความหมาย)

---

## 7. การคิดคะแนน

```python
D0 = จำนวน cell สกปรกตอนเริ่ม (≥ 1 เสมอ)
c[t] = cleaned_after_t_timesteps / D0            # c[0] = 0.0

T = max_steps
t_end = timestep ที่ episode จบ
for t in range(t_end + 1, T + 1):
    c[t] = c[t_end]                              # จบก่อนกำหนด → ค่าคงที่จนถึง T

AUC = sum(c[1..T]) / T

penalty_raw = w_collision * collisions / T + w_redundant * redundant_sucks / T
penalty     = min(max_penalty, penalty_raw)

completed = 1.0 if c[T] >= 1.0 else 0.0
episode_score = AUC + completion_bonus * completed - penalty
```

- `sticky_fails` และ `slips` **ไม่ถูกลงโทษ** — เป็นผลของสิ่งที่ agent ควบคุมไม่ได้
- `episode_score` อยู่ในช่วง `[-0.2, 2.0]` เมื่อใช้ค่า default (`completion_bonus=1.0`, `max_penalty=0.2`)
- **คะแนนของ submission** = ค่าเฉลี่ยเลขคณิตของ `episode_score` ทุก seed
- **การตัดสินเสมอ** ตามลำดับ: จำนวน seed ที่ `completed=1` → `min(episode_score)` ข้าม seed → ค่าเฉลี่ยของ `t_end` เฉพาะ episode ที่ completed → เวลาที่ส่งก่อน

การคำนวณต้องใช้ `float64` ตลอด และปัดเป็น 6 ตำแหน่งทศนิยมตอนบันทึกลง DB เท่านั้น

---

## 8. Gymnasium API

> **Gymnasium คืออะไรในบริบทนี้** — มันเป็น **มาตรฐานหน้าตาของ environment** ไม่ใช่ framework ทำ RL
> ให้แค่สัญญา (`reset`/`step`/spaces), ระบบ wrapper, และ vector env สำหรับรันขนาน
> **ไม่มี** อัลกอริทึม · ไม่มี training loop · **ไม่มีโปรโตคอลการประเมินหรือ leaderboard** — ส่วนหลังคือช่องว่างที่ arena เข้ามาเติม
>
> เหตุผลที่ยังใช้: นิสิตเสียบ environment เข้า Stable-Baselines3 / CleanRL ได้ทันทีโดยไม่ต้องเขียน adapter
> เวลาในเทอมจึงไปลงที่ reward design และการเลือกอัลกอริทึม ไม่ใช่การต่อท่อ
>
> **สิ่งที่ Gymnasium ไม่การันตีให้ และเราต้องบังคับเอง**: determinism — `self.np_random` เป็นแค่ของที่ให้มาใช้
> ไม่มีอะไรห้ามเขียน `np.random.rand()` ในโค้ด env และ `check_env` ก็จับไม่ได้ (จึงต้องมี §2 และ test #13)

```python
class VacuumEnv(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array", "ansi"]}

    def __init__(self, config: dict, render_mode: str | None = None): ...
    def reset(self, *, seed: int, options=None) -> tuple[dict, dict]: ...
    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]: ...
```

- **`reset()` ต้องส่ง `seed` เสมอ** — ถ้าไม่ส่งให้ raise ไม่ใช่สุ่มเอง (กันการทดลองที่ทำซ้ำไม่ได้)
- เรียก `super().reset(seed=seed)` ตามธรรมเนียมของ Gymnasium ได้ แต่ **อย่าใช้ `self.np_random` เป็นสายเดียว** —
  เราต้องการสองสายแยกกันตาม §2 จึงต้องสร้าง `SeedSequence` ของเราเองต่างหาก
- `observation_space` ประกาศเป็น `spaces.Dict` ตาม §4 (SB3 รองรับผ่าน `MultiInputPolicy` นิสิตไม่ต้องแบน tensor เอง)
- การแยก `terminated` / `truncated` ใน §6 **ไม่ใช่เรื่องรูปแบบ** — algorithm ต้อง bootstrap ค่า value ต่างกันในสองกรณี
  ถ้าแมปผิด agent จะเรียนรู้ผิดแบบเงียบๆ หาสาเหตุยาก
- **`reward` คืน `0.0` เสมอทุก step** โดยตั้งใจ — score เป็นเรื่องภายนอก และ **reward เป็นสิ่งที่นิสิตออกแบบเอง**
  starter kit มี `RewardWrapper` ตัวอย่างให้ 2 แบบ

`info` ที่คืนทุก step:

```python
{
  "cleaned": int, "collisions": int, "redundant_sucks": int,
  "sticky_fails": int, "slips": int,
  "coverage": float,          # cleaned / D0
  "t": int, "battery_left": int | None,
  "reason": str | None,       # เฉพาะ step สุดท้าย
}
```

`info` ที่คืนจาก `reset()` เพิ่ม: `D0`, `free_count`, `effective_density`, `config_hash`

---

## 9. Replay format

ไฟล์เดียวต่อ episode `.vrp` (vacuum replay) = header + body ต่อกัน แต่ละส่วนบีบด้วย zstd ระดับ 10

**Header** (JSON)

```json
{
  "format": "vrp/1",
  "env_version": "1.0.0",
  "config_hash": "sha256:...",
  "seed": 1001,
  "W": 20, "H": 20,
  "obstacle_b64": "...",     // bitpack row-major, W*H bits
  "dirt0_b64": "...",        // bitpack
  "sticky_b64": "...",       // bitpack
  "start": [x, y],
  "max_steps": 1500,
  "D0": 204
}
```

**Body** — 4 ไบต์ต่อ timestep เรียงตามเวลา

| offset | ชนิด | ความหมาย |
|---|---|---|
| 0 | `uint8` | action ที่ agent เลือก (0–5) |
| 1 | `uint8` | flags — bit0 moved · bit1 collision · bit2 slipped · bit3 cleaned · bit4 sticky_fail · bit5 redundant_suck |
| 2–3 | `uint16 LE` | `flat_index` ของตำแหน่งหุ่น **หลัง** transition |

**ข้อกำหนด**: เล่น header + body ตั้งแต่ต้นต้องสร้าง state ทุกเฟรมขึ้นมาใหม่ได้ครบ โดย **client ไม่ต้องรู้จัก RNG เลย**
(นี่คือเหตุผลที่ต้องบันทึก `slipped` และตำแหน่งจริงลงไป ไม่ใช่หวังว่า client จะสุ่มซ้ำได้เหมือนกัน)

ขนาดโดยประมาณ: 1,500 timestep × 4 B = 6 KB → หลัง zstd ~1.5–2 KB ([การคำนวณเต็ม](../../../../../README.md#103-การเก็บและแสดง-replay))

---

## 10. Baseline agents

ทั้งสี่ตัวอยู่ใน starter kit และเป็นหมุดหมายบน leaderboard — **สเปคของ Gold มีผลต่อเกรดโดยตรง จึงต้องตรึงให้แน่น**

### 🥉 Bronze — `RandomAgent`

```
if observation บอกว่า cell ปัจจุบันสกปรก: return SUCK
else: return uniform random จาก {UP, DOWN, LEFT, RIGHT}
```

(ไม่เลือก IDLE เลย — IDLE ไม่มีประโยชน์ในโจทย์นี้และจะทำให้ baseline อ่อนเกินจนไม่มีความหมาย)

### 🥈 Silver — `GreedyAgent`

```
สร้าง/อัพเดตแผนที่ภายในจาก observation ที่เห็น
if cell ปัจจุบันสกปรก: return SUCK
if มี cell สกปรกอยู่ในหน้าต่างที่มองเห็น:
    เดินหนึ่งก้าวเข้าหา cell สกปรกที่ใกล้ที่สุด (Manhattan, เสมอ→ flat_index ต่ำสุด)
else:
    เดินสุ่มโดยเลี่ยงทิศที่รู้ว่าเป็นกำแพง
```

**มองแค่ในหน้าต่าง ไม่วางแผนระยะไกล** — จุดนี้คือสิ่งที่แยก Silver ออกจาก Gold

> 🔒 ส่วนนี้ถูกลบออกจากประวัติ — สเปคของ Gold/Diamond ย้ายไป `colosseum-hypogeum`

---

## 11. Config ของทั้ง 3 phase

> **ตัดสินใจแล้ว: `battery: null` ทุก phase** — ให้ `max_steps` เป็นข้อจำกัดเดียวที่ผูกพัน
> ถ้ามีทั้งแบตและ `max_steps` จะเกิดข้อจำกัดสองชั้นที่ทับซ้อนกัน ทำให้อธิบายคะแนนยากและ tune ยาก
> ฟิลด์ `battery` / `move_cost` / `suck_cost` ยังคงอยู่ใน schema ไว้ใช้ปีถัดไป

> ✅ **ค่าในตารางนี้ผ่านการ calibrate รอบที่ 1 แล้ว** (ส.ค. 2026) — ไฟล์ตัวจริงคือ
> [`envs/cp463-vacuum/vacuum/configs/*.yaml`](../../../../../envs/cp463-vacuum/vacuum/configs/) ซึ่งมีคอมเมนต์กำกับที่มาของทุกค่าที่เปลี่ยน
> เหตุผลและตัวเลขเต็มอยู่ที่ [calibration-2026-08.md](calibration-2026-08.md)

| | Warm-up (สัปดาห์ 1–3) | Main (4–6) | Final (7) |
|---|---|---|---|
| `width × height` | 10 × 10 | 20 × 20 | 30 × 30 |
| `obstacle_density` | 0.10 | 0.15 | 0.22 |
| `obstacle_generator` | `random` | `clustered` | `clustered` |
| `dirt_distribution` | `uniform` | `uniform` | `clustered` |
| `dirt_ratio` | 0.60 | 0.60 | 0.60 |
| `start` | `corner` | `random` | `random` |
| `observation` | `full` | `local` | `local` |
| `observation_window` | — | 5 | 3 |
| `action_noise` | 0.00 | 0.10 | 0.10 |
| `sticky_dirt` | 0.00 | 0.15 | 0.15 |
| **`sensor_noise`** 🔑 | **0.00** | **0.02** | **0.02** |
| `max_steps` | **250** | 1500 | 3000 |
| `battery` | `null` | `null` | `null` |
| `w_collision` | 1.0 | 1.0 | 1.0 |
| `w_redundant` | 0.2 | 0.2 | 0.2 |
| `completion_bonus` | 1.0 | 1.0 | 1.0 |
| `max_penalty` | 0.2 | 0.2 | 0.2 |
| public seeds | 30 | 30 | 30 |
| private seeds | 100 | 150 | 150 |
| — *Gold ดูดครบ* | *30/30* | *28/30* | *13/30* |
| — *Diamond ดูดครบ* | *30/30* | *30/30* | *27/30* |

**สิ่งที่เปลี่ยนจากร่างแรกและทำไม**

| ค่า | ร่างแรก | ตอนนี้ | เหตุผล |
|---|---|---|---|
| `sensor_noise` (Main) | 0.00 | **0.02** | เป็นคันโยกเดียวที่กด planner ลงได้จริง — `action_noise`/`sticky_dirt` วัดแล้วว่าแทบไม่มีผล |
| `sensor_noise` (Final) | 0.05 | **0.02** | เปลี่ยน 2 รอบ · 0.05 → Gold ครบแค่ 2/30 · 0.01 → ดูดีถ้าใช้ Gold เป็นเพดาน **แต่แยก Diamond จาก Gold ไม่ออก** (CI คร่อมศูนย์) · 0.02 → ช่องว่าง +0.517 แยกได้ และ Diamond ยังครบ 77% |
| `max_steps` (Warm-up) | 400 | **250** | Gold ดูดครบที่ t_end 137 (p90 = 144) → 400 กว้างเกินจนช่องว่างเหนือ Gold เหลือแค่ 0.16 จุด |
| `max_steps` (Main) | 1500 | 1500 | คงเดิม — ลดลงพร้อมกับการเพิ่ม `sensor_noise` จะทบกันจน Gold เหลือ 16/30 |
| `max_steps` (Final) | 3000 | 3000 | คงเดิม — ยืนยันแล้วว่าไม่ใช่ตัวที่ผูกพัน (เพิ่มเป็น 4000 แล้วผลไม่ต่าง) |

**ที่มาของ `max_steps`** — ตั้งจาก **p90 ของ `t_end` ของ Gold คูณ ~1.2–2** (วัดจริง ไม่ใช่ประมาณ)
เช่น Main: Gold ดูดครบที่ t_end เฉลี่ย 606 · p90 = 631 → 1500 ให้ที่เหลือพอสมควรแม้ในกรณีแย่
เกณฑ์ที่ใช้ตัดสินว่าพอหรือไม่คือ **Gold ต้องดูดครบได้ในสัดส่วนที่มีนัย** ไม่ใช่ตัวเลข `max_steps` เอง

**ช่วง seed** — ห้ามทับกันเด็ดขาด และ private ไม่เปิดเผยจนกว่าจะปิด competition

| ชุด | ย่านที่สุ่มมา | จำนวนที่ใช้ | เปิดเผย | ใช้ที่ไหน |
|---|---|---|---|---|
| `train` | `1–9999` | ไม่จำกัด | ✅ แจก generator ให้นิสิต | นิสิตเทรนและรัน `arena eval --local` |
| `public` | `20000–29999` | 30 ต่อ phase | ❌ บอกแค่ย่านและจำนวน | leaderboard ระหว่างเทอม |
| `private` | `50000–59999` | 100 / 150 / 150 | ❌ | ตัดสินอันดับและเกรด |
| `conformance` | `70001–70030` | 30 | ✅ **เปิดเผยพร้อม golden value** | test #11 — ให้นิสิตยืนยันว่า environment ในเครื่องตรงกับ grader |
| `calibration` | `80001–80030` | 30 | ✅ | การทดลองของผู้สอน (`examples/calibrate.py`) — ไม่แตะชุดที่ใช้ตัดสิน |

⚠️ **ย่านเปิดเผยได้ แต่ค่าที่สุ่มได้จริงห้ามเปิด** — ถ้าจำนวนที่ใช้เท่ากับขนาดของย่าน (เช่น `20001–20030` สำหรับ 30 seed)
ก็เท่ากับเปิดค่าไปแล้วทั้งชุด ค่าจริงอยู่ที่ `colosseum-hypogeum/cp463-1-2026/vacuum/seeds.yaml`
ซึ่งผูก `config_hash` ของแต่ละ phase ไว้ด้วย → ถ้า config เปลี่ยน runner ต้องปฏิเสธการรันแทนที่จะรันด้วย seed ที่ไม่ตรง

---

## 12. โครงสร้าง package และเวอร์ชัน

```
cp463-vacuum/
├── pyproject.toml
├── vacuum/
│   ├── __init__.py          # __version__ = "1.0.0"
│   ├── env.py               # VacuumEnv
│   ├── generator.py         # §3
│   ├── observation.py       # §4
│   ├── scoring.py           # §7  (ต้อง import ได้แยก — grader ใช้ตัวนี้ตัวเดียวกัน)
│   ├── replay.py            # §9
│   ├── config.py            # โหลด+validate YAML, คำนวณ config_hash
│   └── baselines/{random,greedy,bfs}.py
├── configs/{warmup,main,final}.yaml
├── tests/                   # §14
└── examples/train_ppo.py, examples/reward_wrappers.py
```

**เวอร์ชันที่ตรึง** (ต้องเหมือนกันทั้ง starter kit และ runner image)

```
python == 3.11.*
numpy == 2.1.*        # ← load-bearing: Generator stream ไม่การันตีข้ามเวอร์ชัน (§2 ข้อ 3)
gymnasium == 1.3.*    # เวอร์ชันล่าสุด ณ เม.ย. 2026 · ต้องการ python ≥ 3.10
zstandard == 0.23.*
pyyaml == 6.*
```

**เวอร์ชันของ `numpy` กับ `gymnasium` ต้องเหมือนกันเป๊ะทั้ง starter kit และ runner image**
ถ้าเปลี่ยนตัวใดตัวหนึ่งต้องรัน conformance test ใหม่ทั้งชุด และถ้า golden value เปลี่ยน = ขึ้น `env_version` + rejudge

`config_hash = sha256` ของ config ที่ normalize แล้ว (เรียง key, ตัด comment) — บันทึกลงทุก run และทุก replay
**การเปลี่ยนค่าใดๆ ใน config ต้องทำให้ hash เปลี่ยน** และคะแนนข้าม hash ห้ามเอามาเทียบกันบน leaderboard เดียวกัน

---

## 13. Submission validation

ตรวจตอนอัพโหลด ต้องเสร็จใน < 5 วินาที และ error ต้องบอกวิธีแก้

| ตรวจ | เกณฑ์ |
|---|---|
| โครงสร้าง | มี `agent.py` ที่นิยาม `class Agent` และมี `__init__`, `reset`, `act` |
| ขนาด | zip ≤ 200 MB · ไฟล์เดี่ยว ≤ 100 MB |
| import | เฉพาะ stdlib + numpy + torch + package ใน whitelist (ประกาศตอนเปิดเทอม) |
| smoke test | สร้าง agent แล้วรัน 1 episode บน seed ตัวอย่างให้จบได้ |
| action ที่คืน | ต้องเป็น `int` ใน `[0, 5]` — ถ้าคืนชนิดอื่นหรือนอกช่วง = ปฏิเสธ |
| `reset()` ล้าง state จริง | รัน 2 episode สลับลำดับ ผลต้องเหมือนรันแยก (จับ state รั่วข้าม episode) |

---

## 14. Conformance tests

**นี่คือสัญญาที่ทำให้ starter kit กับ grader เป็นสิ่งเดียวกัน** — ทั้งสองฝั่งรัน test ชุดนี้ใน CI

| # | Test | สิ่งที่ยืนยัน |
|---|---|---|
| 1 | `test_layout_determinism` | seed เดียวกัน 100 ครั้ง → `obstacle/dirt/sticky/start` เหมือนกันทุกบิต |
| 2 | `test_layout_independent_of_max_steps` | เปลี่ยน `max_steps` แล้วผังห้องของ seed เดิมต้องไม่เปลี่ยน (จับการใช้ RNG สายเดียว) |
| 3 | `test_connectivity` | ทุก dirty cell เดินถึงได้จาก start ด้วย 4-connectivity |
| 4 | `test_dirt_count_exact` | `D0 == max(1, round(dirt_ratio * free_count))` ทุก seed |
| 5 | `test_observation_shapes` | shape/dtype ของทั้ง 3 โหมดตรงตาม §4 · นอกขอบใน `local` ต้องเป็น obstacle=1.0 |
| 6 | `test_slip_tape_alignment` | รัน 2 policy ที่ต่างกันบน seed เดียวกัน → `slip_tape[t]` ที่ถูกใช้ที่ timestep t ต้องเป็นค่าเดียวกัน |
| 7 | `test_collision_semantics` | เดินชนกำแพง → ตำแหน่งไม่เปลี่ยน · `collisions+1` · `t+1` |
| 8 | `test_sticky_semantics` | SUCK ครั้งแรกบน sticky → ฝุ่นยังอยู่ · ไม่เพิ่ม `redundant_sucks` · ครั้งที่สองสำเร็จแม้เดินออกไปแล้วกลับมา |
| 9 | `test_score_reference` | 5 trajectory ที่เขียนมือ (รวมกรณีจบก่อน T และกรณี penalty ชนเพดาน) → `episode_score` ตรงค่าที่ hardcode ไว้ |
| 10 | `test_replay_roundtrip` | เล่น replay แล้วสร้าง state ทุกเฟรมได้ตรงกับตอนรันจริง 100% |
| 11 | `test_golden_baselines` | Gold baseline บน 30 public seeds → คะแนนตรงค่าที่บันทึกไว้ (จับ regression ทุกชนิด) |
| 12 | `test_reward_is_always_zero` | `env.step()` คืน `reward == 0.0` เสมอ |
| 13 | `test_immune_to_global_rng` | เรียก `np.random.seed(0)` และ `random.seed(0)` ก่อน `reset()` → ผังห้องและ noise tape ต้องไม่เปลี่ยน (จับการเผลอใช้ global RNG ตาม §2 ข้อ 1) |

Test #11 คือ **regression suite ตัวจริง** — ค่า golden ต้อง generate หลัง implement เสร็จ แล้ว commit ลง repo
ถ้าค่าเปลี่ยนเมื่อไหร่ต้องขึ้น `env_version` และ rejudge ทุก submission

**สถานะ: implement ครบทั้ง 13 ข้อแล้ว** ที่ [`envs/cp463-vacuum/tests/test_conformance.py`](../../../../../envs/cp463-vacuum/tests/test_conformance.py)
(พร้อมการตรวจเพิ่มอีก 18 ข้อที่ไม่ได้อยู่ในตารางนี้ เช่น config ที่พิมพ์ชื่อฟิลด์ผิดต้อง raise
และ `config_hash` ต้องเปลี่ยนเมื่อค่าใดๆ เปลี่ยน)

- ค่า golden อยู่ที่ `tests/golden_baselines.json` · generate ใหม่ด้วย `python tests/make_golden.py`
- ไฟล์ golden เก็บ `config_hash` ของทุก phase ไว้ด้วย — ถ้า config เปลี่ยนแล้วลืม generate ใหม่ test จะ**ฟ้องที่ hash ก่อน**
  แทนที่จะฟ้องเป็นตัวเลขคะแนนที่ไม่ต่างกัน ซึ่งอ่านไม่ออกว่าเกิดอะไรขึ้น
- test #11 ติด marker `slow` (ใช้เวลา ~17 วินาที) — ตอนพัฒนาใช้ `pytest -m "not slow"` ได้ แต่ CI ต้องรันเต็ม

---

## 15. การ calibrate

✅ **รอบที่ 1 เสร็จแล้ว (ส.ค. 2026)** — ผลเต็มอยู่ที่ [calibration-2026-08.md](calibration-2026-08.md)
รันซ้ำได้ด้วย `python examples/calibrate.py --seeds 30`

### การทดลองที่ 1 — `action_noise` กด planner ลงได้จริงไหม

รัน `BFSCoverageAgent` เทียบกับ PPO ที่เทรนแล้ว บน 30 seed × `action_noise ∈ {0, 0.05, 0.10, 0.20}` ที่ config Main

❌ **ผล: `action_noise` ไม่ใช่คันโยก** — ที่ 0 → 0.50 Gold ตกจาก 1.828 เหลือ 1.655 และ**ยังดูดครบทุก seed**
ที่ค่าที่ตั้งไว้จริง (0.10) planner เสียไปแค่ 0.018 จุด · `sticky_dirt` แม้ตั้ง 0.80 ก็ตกแค่ 3%

**สาเหตุ** — planner ที่ replan ทุก timestep เป็น closed-loop controller และหลักการ
hardware-independent scoring ทำให้การคิดใหม่ทุก step ไม่มีต้นทุน ความสุ่มสองแบบนี้ทำให้*เส้นทาง*ยาวขึ้น
แต่ไม่ได้ทำให้*ความรู้เกี่ยวกับโลก*ผิด ซึ่งอย่างหลังต่างหากที่ทำลาย planner

✅ **คันโยกที่ใช้แทน: `sensor_noise`** — Main 0 → 0.02 กด Gold จาก 1.810 เหลือ 1.649 และเริ่มดูดไม่ครบ
(ที่ 0.05 พังทั้งชุด เหลือ 0.753 จึงห้ามตั้งเกิน 0.03)

❌ **ฝั่ง learned policy ก็ตอบแล้ว: PPO แพ้ Gold ทุกระดับ** (0.445 vs 1.649 ที่ `sensor_noise: 0.02`)
และไม่เคยดูดครบเลยสักครั้ง — แพ้แม้แต่ Silver
เทรน 4.6M timestep (curriculum + fine-tune) ด้วย `examples/train_ppo.py`

ทิศทางของคันโยกถูก — `sensor_noise` 0 → 0.05 กด Gold ลง 58% แต่กด PPO ลงแค่ 13%
**แต่ฐานความสามารถของ policy ต่ำเกินไปจนจุดตัดอยู่เลยโซนที่ใช้ได้จริง**
รายละเอียดและทางเลือกที่เหลือ: [calibration-2026-08.md §1.4 และ §5.1](calibration-2026-08.md#14-learned-policy-ชนะ-planner-ได้ไหม)

### การทดลองที่ 2 — `max_steps` พอให้ดูดครบไหม

รัน Gold บนแต่ละ phase ดู `t_end` ของ episode ที่ completed
ถ้าไม่มี episode ไหน completed เลย → `completion_bonus` กลายเป็นค่าคงที่ที่ไม่มีผล ต้องเพิ่ม `max_steps`
เกณฑ์ที่ต้องการ: **Gold ควร completed ประมาณ 60–90% ของ seed** — ต่ำกว่านั้นโจทย์ยากเกิน สูงกว่านั้นแยกทีมไม่ออก

✅ **ผล: ผ่านทุก phase หลังปรับ** — แต่ **ต้องวัดกับ baseline ที่ดีที่สุด ไม่ใช่กับ Gold**
(Diamond ดูดครบ 30/30 · 30/30 · 27/30 · ส่วน Gold ได้ 30/30 · 28/30 · 13/30)

รอบแรกผมใช้ Gold เป็นตัวอ้างอิงและตั้ง `sensor_noise` ของ Final เป็น 0.01 เพราะ Gold ครบ 73%
ซึ่ง**ผิด** — Gold ไม่ใช่เพดานของโจทย์ พอมี Diamond จึงเห็นว่าที่ 0.01 โจทย์ง่ายเกินไปจนแยก
สองระดับบนไม่ออก เกณฑ์นี้ต้องอ่านว่า *"agent ที่ดีที่สุดที่เรามี ควรดูดครบ 60–90%"*

ข้อสังเกตสำคัญอีก 2 ข้อ

1. **`max_steps` ปรับ Gold ให้ตกมาอยู่ในช่วง 60–90% ไม่ได้** — `t_end` ของ Gold เกาะกันแน่นมาก
   (Main: เฉลี่ย 606 · p90 631) ตั้งเหนือค่านั้นก็ครบ ~100% ตั้งต่ำกว่าก็ร่วงเป็น ~0% ไม่มีช่วงกลาง
   → **ตัวที่คุมสัดส่วนนี้จริงคือ `sensor_noise`** ส่วน `max_steps` ควรตั้ง "สูงพอ" แล้วจบ
2. Final ยืนยันชัด: เพิ่ม `max_steps` จาก 3000 → 4000 ทำให้ completed เปลี่ยนจาก 21/30 เป็น 20/30 (ไม่ต่าง)
   แต่ปิด `sensor_noise` ทำให้เป็น 30/30 ทันที

### การทดลองที่ 3 — คะแนน baseline ทั้ง 4 ระดับห่างกันพอไหม

รันทั้ง 4 ตัวบน 30 public seed ต้องได้คะแนนที่ **ห่างกันเกินความกว้างของ CI** ไม่งั้นเส้นแบ่งเกรดจะไม่มีความหมาย

✅ **ผล: ผ่านทุก phase** CI ไม่ทับกันเลยสักคู่ (ช่องว่างเล็กสุด +0.297 ที่ Final)

⚠️ **CI ของ Silver กว้างผิดปกติ (±0.17)** เพราะ `GreedyAgent` เดินเข้าหาเป้าทีละก้าวแบบ Manhattan
โดยไม่มี BFS → ติดหลังกำแพงแล้วแกว่งไปมาจนจบ episode ในบาง seed
ความแปรปรวนนี้มาจาก**บุคลิกของ baseline** ไม่ใช่ความยากของ seed จึงไม่กระทบความยุติธรรม
แต่ถ้าจะใช้หมุด Silver เป็นเส้นแบ่งเกรด ต้องตรึงด้วย seed จำนวนมากกว่า 30

### สถานะของค่าที่ต้องตรึง

| ค่า | สถานะ |
|---|---|
| `sensor_noise` ทั้ง 3 phase | ✅ **ตรึงแล้ว (0.00 / 0.02 / 0.02)** — เลือกจากเกณฑ์ว่าแยก Gold ออกจาก Diamond ได้ทางสถิติ |
| `max_steps` ทั้ง 3 phase | ✅ ตรึงแล้ว (250 / 1500 / 3000) |
| `action_noise` · `sticky_dirt` | ✅ ตรึงแล้ว (0 / 0.10 / 0.15) — เก็บไว้เพราะไม่เสียหาย ไม่ใช่เพราะมันทำหน้าที่ตามที่ออกแบบ |
| `obstacle_density` ของ Final | ✅ คงเดิม 0.22 — ไม่พบว่าเป็นตัวคุมความยาก |
| คะแนน golden ของ baseline | ⚠️ generate แล้วบน **ชุด conformance (70001+)** สำหรับ test #11 · ยัง**ไม่ได้**รันบน public seeds ซึ่งเป็นค่าที่ไปเป็น threshold เกรด |

### รอบต่อไปต้องทำอะไร

1. **ตัดสินว่าจะรับมือกับ "planner ชนะ" อย่างไร** — 3 ทางเลือกอยู่ที่
   [calibration §5.1](calibration-2026-08.md#51-ข้อที่ต้องตัดสินก่อนเปิด-competition) ข้อเสนอคือยอมรับ
   (ทางเลือก C+D ของ [template §5](../../../../task-templates/agent-vs-environment-rl.md#5-ปัญหาที่ต้องตัดสินใจ-planning-ชนะ-learning))
   แล้วประกาศให้ชัดตั้งแต่วันแรกว่า planner ก็ได้คะแนนเต็มส่วน leaderboard
2. **ตัดสินว่า Gold คือ planner ที่ไร้เดียงสาเรื่อง noise หรือไม่** — กระทบเส้นแบ่งเกรดโดยตรง
3. รัน baseline ทั้งชุดบน **public seeds** แล้วตรึงค่าเป็น threshold เกรด
