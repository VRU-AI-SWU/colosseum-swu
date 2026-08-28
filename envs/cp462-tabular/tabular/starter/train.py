"""เทรนโมเดลแล้วบันทึกเป็น pipeline.pkl — รันไฟล์นี้ก่อนส่งงานทุกครั้ง

    python train.py --task churn

**นี่คือจุดตั้งต้น ไม่ใช่คำตอบ** — pipeline ข้างล่างเป็นของธรรมดาที่สุดที่ทำงานได้
หน้าที่ของคุณคือทำให้มันดีกว่านี้ · ลองเปลี่ยนโมเดล สร้างฟีเจอร์ใหม่ จูนพารามิเตอร์
"""

import argparse

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabular.config import load
from tabular.dataset import open_data
from tabular.metrics import score

NUMERIC = ["tenure_months", "monthly_spend", "support_tickets"]
CATEGORICAL = ["plan", "region"]
# `account_id` ไม่ได้อยู่ในสองรายการข้างบนโดยตั้งใจ — มันไม่เกี่ยวกับเป้าหมายเลย
# `remainder="drop"` จึงทิ้งมันไป · ลองใส่เข้าไปดูแล้วเทียบคะแนนก็ได้


def build_pipeline(kind: str) -> Pipeline:
    """ทุกขั้นตอนอยู่ในนี้ — เพราะระบบส่งข้อมูลดิบเข้ามาตอนทำนาย"""
    pre = ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                # `handle_unknown="ignore"` จำเป็น — ชุดที่ใช้ตัดสินอาจมีหมวดหมู่
                # ที่ไม่เคยโผล่ในชุดที่คุณเทรน ถ้าไม่ตั้ง โมเดลจะพังทั้งชุด
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL),
        ],
        remainder="drop",
    )
    model = (
        HistGradientBoostingClassifier(random_state=0)
        if kind == "classification"
        else HistGradientBoostingRegressor(random_state=0)
    )
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="churn", help="churn หรือ housing")
    ap.add_argument("--out", default="pipeline.pkl")
    args = ap.parse_args()

    spec = load(args.task)
    data = open_data(spec)          # train + val เท่านั้น — ชุดที่ใช้ตัดสินอยู่ในระบบ
    train, val = data["train"], data["val"]

    pipe = build_pipeline(spec.kind).fit(train.X, train.y)

    # วัดบน val ที่โมเดลไม่เคยเห็น — ตัวเลขนี้ประมาณคะแนนบน leaderboard ได้
    # แต่ **ไม่เท่ากันเป๊ะ** เพราะชุดที่ใช้ตัดสินเป็นคนละชุด
    result = score(
        val.y, pipe.predict(val.X),
        kind=spec.kind, primary=spec.primary,
        seed=spec.bootstrap_seed, labels=spec.labels or None,
    )
    print(f"{spec.slug} · {spec.primary} บน val = {result.primary:+.4f} "
          f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}]")

    joblib.dump(pipe, args.out)
    print(f"บันทึก {args.out} แล้ว — ส่งด้วย `arena submit {spec.slug}-1-2026`")


if __name__ == "__main__":
    main()
