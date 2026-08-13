"""Baseline agents — หมุดหมายบน leaderboard (environment-spec §10)

| ระดับ | Agent | ความหมาย |
|---|---|---|
| 🥉 Bronze | `RandomAgent` | "โค้ดทำงานได้แล้ว" |
| 🥈 Silver | `GreedyAgent` | "agent มีกลยุทธ์แล้ว" |
| 🥇 Gold | `BFSCoverageAgent` | "ดีกว่าวิธีคลาสสิกที่ไม่ได้เรียนรู้" |
| 💎 Diamond | solution ของผู้สอน (learned policy) | ยังไม่อยู่ใน repo นี้ |

คะแนนของแต่ละระดับต้องได้จากการรันจริงบน public seeds ชุดเดียวกับนิสิต
แล้วตรึงค่าไว้ทั้งเทอม และต้องรันใหม่ทุกครั้งที่เปลี่ยน phase
"""

from vacuum.baselines.bfs import BFSCoverageAgent
from vacuum.baselines.greedy import GreedyAgent
from vacuum.baselines.random_agent import RandomAgent

BASELINES = {
    "bronze": RandomAgent,
    "silver": GreedyAgent,
    "gold": BFSCoverageAgent,
}

__all__ = ["RandomAgent", "GreedyAgent", "BFSCoverageAgent", "BASELINES"]
