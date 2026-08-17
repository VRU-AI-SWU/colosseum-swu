"""ตัวอย่าง reward สำหรับตอนเทรน — environment-spec §8

⚠️ **reward ตอนเทรน ≠ metric ตอนตัดสิน**

`VacuumEnv.step()` คืน `reward = 0.0` เสมอโดยตั้งใจ คะแนนจริงคำนวณจากภายนอกด้วย
`vacuum.scoring` และ agent ไม่เคยเห็นค่านั้นระหว่างรัน **การออกแบบ reward เป็นส่วนหนึ่งของคำตอบ
ไม่ใช่ส่วนหนึ่งของโจทย์** — ไฟล์นี้ให้ตัวอย่างสองแบบเพื่อให้เห็นว่าการเลือกต่างกันแล้วผลต่างกันจริง

ลองเทรนทั้งสองแบบแล้วเทียบด้วย `arena eval --local` ดู จะเห็นเรื่องที่บรรยายให้ฟังยาก
"""

from __future__ import annotations

import gymnasium as gym


class SparseCoverageReward(gym.Wrapper):
    """แบบตรงไปตรงมาที่สุด: ได้ +1 ทุกครั้งที่ดูดสำเร็จ

    **ดูสมเหตุสมผลแต่เรียนรู้ยากมาก** — ระหว่างการดูดสองครั้ง agent เดินเปล่าหลายสิบ step
    โดยไม่ได้ reward อะไรเลย (credit assignment ยาว) และไม่มีอะไรบอกว่า "เดินไปทางไหนดี"
    ที่สำคัญกว่านั้น: มันไม่ลงโทษการใช้ timestep เลย ทั้งที่ metric จริงคิดจาก timestep เป็นหลัก
    → agent ที่ได้ reward สูงสุดในสายตาของ wrapper นี้ อาจได้คะแนนจริงต่ำ
    """

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        cleaned_before = getattr(self, "_prev_cleaned", 0)
        reward = float(info["cleaned"] - cleaned_before)
        self._prev_cleaned = info["cleaned"]
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._prev_cleaned = 0
        return self.env.reset(**kwargs)


class ShapedCoverageReward(gym.Wrapper):
    """แบบที่ออกแบบให้สอดคล้องกับ metric จริง (coverage AUC + completion bonus)

    | ส่วนประกอบ | ทำไมต้องมี |
    |---|---|
    | `clean_bonus` ต่อช่องที่ดูดได้ | สัญญาณหลัก |
    | `step_cost` ทุก timestep | metric จริงคิดจากจำนวน timestep — ถ้าไม่ใส่ agent จะไม่รีบ |
    | `explore_bonus` เมื่อเหยียบช่องใหม่ | แก้ปัญหา reward เบาบางช่วงต้น episode ที่ยังหาฝุ่นไม่เจอ |
    | `collision_cost` / `redundant_cost` | ตรงกับ penalty ในสูตรคะแนนจริง |
    | `completion_bonus` | metric จริงมีโบนัสก้อนใหญ่ตอนดูดครบ ถ้าไม่สะท้อนไว้ agent จะไม่ไล่เก็บช่องสุดท้าย |

    `explore_bonus` เป็นตัวที่ต้องระวังที่สุด: ตั้งสูงไป agent จะเดินสำรวจเพลินโดยไม่ดูด
    (**reward hacking** แบบคลาสสิก — มันทำสิ่งที่เราสั่ง ไม่ใช่สิ่งที่เราตั้งใจ)

    ── บทเรียนจากค่าชุดแรกที่ตั้งไว้ผิด ─────────────────────────────────────
    ค่าชุดแรก (`step_cost=0.01`, `redundant_cost=0.02`, `completion_bonus=20`)
    **ให้น้ำหนักเวลาน้อยเกินไปราว 4 เท่า** เทียบกับ metric จริง

    | | reward ชุดแรก | metric จริง |
    |---|---|---|
    | ดูดครบทั้งห้อง (204 ช่อง) | 204 | AUC ≤ 1.0 |
    | ประหยัด 900 timestep | 9 (4% ของข้างบน) | ~0.3 (30%) |
    | โบนัสดูดครบ | 20 (10%) | 1.0 (50%) |

    ผลคือ "ยืนนิ่งแล้ว SUCK ที่เดิมไปเรื่อยๆ" เสียแค่ 0.03 ต่อ step ทั้งที่ในสูตรจริง
    มันคือการทิ้ง episode ทั้งอัน — พอ policy ยังไม่คมช่วงต้นการเทรน argmax จึงไปเกาะ SUCK
    ได้ทั้งกระดานโดยไม่โดนลงโทษพอ (coverage 0.005 ตอนวัดแบบ deterministic
    ทั้งที่แบบ stochastic ได้ 0.59) — เป็นตัวอย่างที่ดีมากว่า **reward ที่ "ดูสมเหตุสมผล"
    ทำให้ agent เรียนพฤติกรรมที่ไม่ตรงกับสิ่งที่เราต้องการจริงได้อย่างไร**

    ค่า default ด้านล่างคือชุดที่ปรับให้สัดส่วนใกล้เคียง metric จริงแล้ว
    """

    def __init__(
        self,
        env,
        clean_bonus: float = 1.0,
        step_cost: float = 0.04,
        explore_bonus: float = 0.05,
        collision_cost: float = 0.10,
        redundant_cost: float = 0.20,
        completion_bonus: float = 100.0,
    ):
        super().__init__(env)
        self.clean_bonus = clean_bonus
        self.step_cost = step_cost
        self.explore_bonus = explore_bonus
        self.collision_cost = collision_cost
        self.redundant_cost = redundant_cost
        self.completion_bonus = completion_bonus

    def reset(self, **kwargs):
        self._prev = {"cleaned": 0, "collisions": 0, "redundant_sucks": 0}
        self._visited = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        d_clean = info["cleaned"] - self._prev["cleaned"]
        d_coll = info["collisions"] - self._prev["collisions"]
        d_redundant = info["redundant_sucks"] - self._prev["redundant_sucks"]
        self._prev = {k: info[k] for k in self._prev}

        # นับช่องใหม่ที่เหยียบ — อ่านจาก env โดยตรงเพราะเป็นข้อมูลที่ agent เห็นเองอยู่แล้ว
        visited = int(self.env.unwrapped.visited.sum())
        d_visit = max(0, visited - self._visited)
        self._visited = visited

        reward = (
            self.clean_bonus * d_clean
            + self.explore_bonus * d_visit
            - self.step_cost
            - self.collision_cost * d_coll
            - self.redundant_cost * d_redundant
        )
        if terminated and info.get("reason") == "complete":
            reward += self.completion_bonus

        return obs, float(reward), terminated, truncated, info


REWARDS = {"sparse": SparseCoverageReward, "shaped": ShapedCoverageReward}
