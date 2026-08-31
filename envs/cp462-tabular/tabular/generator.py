"""สร้างไฟล์ CSV ตัวอย่างที่ทำซ้ำได้ทุกบิต — **เครื่องมือ ไม่ใช่แหล่งข้อมูลของการให้คะแนน**

⚠️ **บทบาทของไฟล์นี้เปลี่ยนไปแล้ว** — เดิมมันคือแหล่งข้อมูลจริงของโจทย์ นิสิต
เรียก `make()` บนเครื่องตัวเองเพื่อสร้างชุดเดียวกับที่ grader ใช้ · ตอนนี้
การให้คะแนนอ่านจาก **ไฟล์ในคลัง** (`tabular.store`) เสมอ และโมดูลนี้เหลือหน้าที่
เดียวคือปั๊มไฟล์ตัวอย่างให้ผู้สอนอัปโหลด ตอนที่วิชายังไม่มีชุดข้อมูลจริง

เหตุผลที่ต้องเปลี่ยน: ถ้าข้อมูลสร้างจากเมล็ดได้ ใครที่รู้เมล็ดก็สร้างเฉลยของ
ชุดที่ใช้ตัดสินได้ทั้งชุด — เคยเกิดขึ้นจริงและทำคะแนนได้ 1.0000 · ความลับที่
ตั้งอยู่บนไฟล์ที่ไม่เคยถูกส่งออกไปแข็งแรงกว่าความลับที่ตั้งอยู่บนตัวเลข

**ข้อมูลถูกออกแบบให้บังคับทักษะที่วิชาอยากสอน** ไม่ใช่แค่สุ่มตัวเลขมั่วๆ

| สิ่งที่ใส่เข้าไป | บังคับให้นิสิตต้องทำอะไร |
|---|---|
| ค่าว่างในคอลัมน์ตัวเลขและหมวดหมู่ | ต้อง impute — และต้องอยู่ใน `Pipeline` ไม่ใช่ทำมือก่อน |
| หมวดหมู่ที่พบน้อยมาก | ต้องจัดการหมวดที่ไม่เคยเห็นตอนเทรน (`handle_unknown`) |
| ความสัมพันธ์แบบไม่เป็นเส้นตรงและมี interaction | linear model ล้วนจะสู้ tree ไม่ได้ — เห็นผลของการเลือกโมเดล |
| คอลัมน์ที่ไม่เกี่ยวอะไรเลย | การโยนทุกคอลัมน์เข้าโมเดลไม่ใช่คำตอบเสมอไป |
| สเกลต่างกันหลายเท่า | ต้อง scale สำหรับโมเดลที่ไวต่อสเกล |

**ทุกอย่างต้องทำซ้ำได้ทุกบิต** — นิสิตเทรนบนข้อมูลชุดเดียวกับที่ grader ใช้แบ่ง
ถ้าต่างกันแม้แถวเดียว คะแนนที่วัดเองจะเทียบกับ leaderboard ไม่ได้ · ใช้
`numpy.random.Generator(PCG64(seed))` สายเดียวและ**ห้ามพึ่งการสุ่มของ pandas
หรือ sklearn** ซึ่งเปลี่ยน stream ข้ามเวอร์ชันได้โดยไม่บอก
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabular.table import Dataset, fingerprint as _fingerprint

#: หมวดหมู่ของแผน — ตัวสุดท้ายพบน้อยมากโดยตั้งใจ (ดูตารางในหัวไฟล์)
PLANS = ("basic", "standard", "premium", "legacy")
PLAN_WEIGHTS = (0.46, 0.34, 0.18, 0.02)

REGIONS = ("north", "central", "south", "east", "west")

#: สัดส่วนค่าว่างของแต่ละคอลัมน์ที่ยอมให้ว่างได้
MISSING_RATE = {"tenure_months": 0.06, "monthly_spend": 0.04, "plan": 0.03}


def _base_frame(rng: np.random.Generator, n: int, id_offset: int = 0) -> pd.DataFrame:
    """คอลัมน์ตั้งต้นที่ทั้งสองโจทย์ใช้ร่วมกัน

    สร้างเป็น numpy array ทั้งหมดก่อนแล้วค่อยประกอบเป็น DataFrame — การสุ่มผ่าน
    pandas จะผูกผลลัพธ์กับเวอร์ชันของ pandas ซึ่งเป็นสิ่งที่เราคุมไม่ได้
    """
    tenure = rng.gamma(shape=2.0, scale=11.0, size=n)
    spend = 180 + 55 * rng.lognormal(mean=0.0, sigma=0.55, size=n)
    tickets = rng.poisson(lam=1.3, size=n)
    plan = rng.choice(len(PLANS), size=n, p=PLAN_WEIGHTS)
    region = rng.integers(0, len(REGIONS), size=n)
    # คอลัมน์ที่ไม่เกี่ยวกับเป้าหมายเลย — มีไว้ให้เห็นว่าการโยนทุกคอลัมน์เข้าโมเดล
    # ไม่ใช่คำตอบเสมอไป · ชื่อบอกตรงๆ ว่าเป็น id เพื่อไม่ให้เป็นกับดักที่ไม่แฟร์
    account_id = rng.permutation(n) + 100_000 + id_offset

    return pd.DataFrame(
        {
            "account_id": account_id.astype("int64"),
            "tenure_months": np.round(tenure, 2),
            "monthly_spend": np.round(spend, 2),
            "support_tickets": tickets.astype("int64"),
            "plan": pd.Categorical.from_codes(plan, categories=list(PLANS)),
            "region": pd.Categorical.from_codes(region, categories=list(REGIONS)),
        }
    )


def _punch_holes(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """เจาะค่าว่างตามสัดส่วนที่กำหนด — **หลัง**สร้างเป้าหมายแล้วเสมอ

    ถ้าเจาะก่อน เป้าหมายจะขึ้นกับว่าแถวไหนบังเอิญว่าง ซึ่งทำให้ค่าว่างกลายเป็น
    สัญญาณที่ทำนายเป้าหมายได้ — เป็นการรั่วที่นิสิตจะหาเจอแล้วได้คะแนนสูงเกินจริง
    """
    out = frame.copy()
    for column, rate in MISSING_RATE.items():
        mask = rng.random(len(out)) < rate
        out.loc[mask, column] = None
    return out


def make_churn(seed: int, n: int, id_offset: int = 0) -> Dataset:
    """classification — ทำนายว่าลูกค้าจะเลิกใช้บริการไหม (ไม่สมดุล ~26%)

    ความสัมพันธ์เป็นขั้นบันได: ลูกค้าใหม่มากและเก่ามากเลิกน้อย ช่วงเดือนที่ 8–22
    เลิกเยอะ — เส้นตรงบนฟีเจอร์ดิบจับไม่ได้

    วัดแล้ว (macro-F1 บน 6000 train / 2000 test) · ทายคลาสเดียว 0.43 ·
    logistic 0.51 · gradient boosting 0.64 — มีบันไดพอให้การเลือกโมเดลมีผลจริง
    """
    rng = np.random.default_rng(seed)
    frame = _base_frame(rng, n, id_offset)

    tenure = frame["tenure_months"].to_numpy()
    spend = frame["monthly_spend"].to_numpy()
    tickets = frame["support_tickets"].to_numpy()
    plan_code = frame["plan"].cat.codes.to_numpy()

    # **ห้ามสร้างเป้าหมายด้วยสูตร logistic ล้วน** — ถ้าทำแบบนั้น logistic regression
    # จะเกือบดีที่สุดโดยโครงสร้าง แล้วโจทย์จะไม่สอนอะไรเรื่องการเลือกโมเดล
    # (วัดแล้วรอบแรก: logistic 0.534 vs boosting 0.547 — แทบไม่ต่าง)
    #
    # ความเสี่ยงจึงสร้างจาก **ขั้นบันไดและ interaction** ที่โมเดลเชิงเส้นบนฟีเจอร์ดิบ
    # แสดงออกไม่ได้ ต้องสร้างฟีเจอร์เองหรือใช้โมเดลที่แบ่งช่วงได้
    spend_z = (spend - 300.0) / 100.0

    risk = np.full(n, -2.6)
    # ช่วงอันตรายคือเดือนที่ 8–22 — ก่อนหน้ายังใหม่ หลังจากนั้นผูกพันแล้ว
    risk += np.where((tenure >= 8) & (tenure <= 22), 2.3, 0.0)
    risk += np.where(tenure > 40, -1.1, 0.0)
    # ร้องเรียนมีผลแบบขั้น ไม่ใช่เชิงเส้น — ครั้งแรกไม่เท่าไร ตั้งแต่สามครั้งคือสัญญาณ
    risk += np.where(tickets >= 3, 1.9, 0.0) + np.where(tickets == 2, 0.5, 0.0)
    # interaction จริง: แผน legacy อันตรายเฉพาะเมื่อจ่ายแพง
    risk += np.where((plan_code == 3) & (spend_z > 0.5), 2.4, 0.0)
    risk += 0.25 * spend_z

    prob = 1.0 / (1.0 + np.exp(-risk))
    y = (rng.random(n) < prob).astype("int64")

    return Dataset(
        X=_punch_holes(frame, rng),
        y=pd.Series(y, name="churned"),
    )


def make_housing(seed: int, n: int, id_offset: int = 0) -> Dataset:
    """regression — ทำนายมูลค่าต่อเดือน (เบ้ขวา)

    เป้าหมายเบ้เพราะมูลค่าจริงเบ้เสมอ · ผลของ tenure อิ่มตัวแบบเอกซ์โพเนนเชียล
    ซึ่งเส้นตรงจับไม่ได้ และมี interaction ระหว่างแผนกับระยะเวลา

    วัดแล้ว (R² บน 6000 train / 2000 test) · ทายค่าเฉลี่ย 0.00 · linear 0.71 ·
    gradient boosting 0.79
    """
    rng = np.random.default_rng(seed)
    frame = _base_frame(rng, n, id_offset)

    tenure = frame["tenure_months"].to_numpy()
    spend = frame["monthly_spend"].to_numpy()
    tickets = frame["support_tickets"].to_numpy()
    plan_code = frame["plan"].cat.codes.to_numpy()
    region_code = frame["region"].cat.codes.to_numpy()

    # เช่นเดียวกับ churn — ถ้าเป็นเชิงเส้นล้วน `LinearRegression` จะชนะ tree
    # (วัดแล้วรอบแรก: linear 0.339 vs boosting 0.312) ซึ่งตรงข้ามกับที่อยากสอน
    base = 3800 + 5.1 * spend
    # ผลของ tenure อิ่มตัว — เพิ่มเร็วช่วงแรกแล้วนิ่ง เส้นตรงจับไม่ได้
    base += 2600.0 * (1.0 - np.exp(-tenure / 9.0))
    base += np.array([0, 850, 2100, -400])[plan_code]
    base += np.array([120, 640, -230, 90, 310])[region_code]
    # interaction: แผน premium ได้ประโยชน์จากการอยู่นานมากกว่าแผนอื่น
    base += np.where(plan_code == 2, 46.0 * np.minimum(tenure, 36.0), 0.0)
    base -= np.where(tickets >= 3, 620.0, 95.0 * tickets)
    value = base * np.exp(rng.normal(0.0, 0.07, size=n))  # noise แบบคูณ → เบ้ขวา

    return Dataset(
        X=_punch_holes(frame, rng),
        y=pd.Series(np.round(value, 2), name="monthly_value"),
    )


TASKS = {"churn": make_churn, "housing": make_housing}


def make(task: str, seed: int, n: int, id_offset: int = 0) -> Dataset:
    """สร้างข้อมูลหนึ่งชุด

    `id_offset` เลื่อนช่วงของ `account_id` — ชุดที่แจกนิสิตกับชุดที่ใช้ตัดสินเป็น
    คนละ dataset ที่สร้างจากคนละเมล็ด แต่ `account_id` เริ่มที่ 100_000 เท่ากันทั้งคู่
    ถ้าไม่เลื่อน สองชุดจะมี id ชนกันทั้งหมด แล้วเทสต์ที่ตรวจว่า "ชุดตัดสินไม่ทับกับ
    ที่นิสิตได้รับ" จะจับอะไรไม่ได้เลย
    """
    try:
        return TASKS[task](seed, n, id_offset)
    except KeyError:
        raise ValueError(f"ไม่รู้จักโจทย์ {task!r} — ที่มีคือ {sorted(TASKS)}") from None


def fingerprint(dataset: Dataset) -> str:
    """ลายนิ้วมือของทั้งชุด — `selfcheck` เอาไปเทียบกับค่า golden"""
    return _fingerprint(pd.concat([dataset.X, dataset.y], axis=1))


def sample_csv(task: str, *, seed: int, n: int) -> bytes:
    """ไฟล์ตัวอย่างพร้อมอัปโหลด — ฟีเจอร์กับเฉลยรวมเป็นตารางเดียว

    นี่คือทางออกเดียวของโมดูลนี้ที่ระบบจริงใช้ · ผลลัพธ์ต้องเดินทางเดียวกับ
    ไฟล์ที่ผู้สอนอัปโหลดเองทุกประการ — ถ้ามันพิเศษกว่า เส้นทางที่ทดสอบบ่อยที่สุด
    จะไม่ใช่เส้นทางที่ผู้สอนใช้จริง
    """
    data = make(task, seed=seed, n=n)
    return pd.concat([data.X, data.y], axis=1).to_csv(index=False).encode("utf-8")
