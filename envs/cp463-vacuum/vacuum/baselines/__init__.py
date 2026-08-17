"""Baseline agents — หมุดหมายบน leaderboard (environment-spec §10)

| ระดับ | Agent | ความหมาย |
|---|---|---|
| 🥉 Bronze | `RandomAgent` | "โค้ดทำงานได้แล้ว" |
| 🥈 Silver | `GreedyAgent` | "agent มีกลยุทธ์แล้ว" |
| 🥇 Gold | `BFSCoverageAgent` | "จำแผนที่ได้และวางแผนเป็น" — แต่เชื่อเซนเซอร์ตรงๆ |
| 💎 Diamond | `BeliefBFSAgent` | "วางแผนบน belief" — รู้ว่าเซนเซอร์เชื่อไม่ได้แล้วทำอะไรกับมัน |

Gold กับ Diamond ใช้ตรรกะการวางแผนเดียวกันเป๊ะ ต่างกันแค่แผนที่ที่มันวางแผนอยู่บน
— ความต่างของคะแนนสองตัวนี้จึงเป็นการวัดผลของ "การกรอง noise" ล้วนๆ

**ladder ไม่ได้เรียงตามตระกูลอัลกอริทึม** — วัดแล้วว่า learned policy แพ้ planner
ในโจทย์นี้โดยธรรมชาติ ([overview §12](../../../../docs/competitions/CP463/1-2026/vacuum-robot/overview.md))

คะแนนของแต่ละระดับต้องได้จากการรันจริงบน public seeds ชุดเดียวกับนิสิต
แล้วตรึงค่าไว้ทั้งเทอม และต้องรันใหม่ทุกครั้งที่เปลี่ยน phase
"""

from vacuum.baselines.belief_bfs import BeliefBFSAgent
from vacuum.baselines.bfs import BFSCoverageAgent
from vacuum.baselines.greedy import GreedyAgent
from vacuum.baselines.random_agent import RandomAgent

BASELINES = {
    "bronze": RandomAgent,
    "silver": GreedyAgent,
    "gold": BFSCoverageAgent,
    "diamond": BeliefBFSAgent,
}

__all__ = [
    "RandomAgent", "GreedyAgent", "BFSCoverageAgent", "BeliefBFSAgent", "BASELINES",
]
