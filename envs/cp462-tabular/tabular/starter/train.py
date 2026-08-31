"""เทรนโมเดลแล้วบันทึกเป็น pipeline.pkl — รันไฟล์นี้ก่อนส่งงานทุกครั้ง

    python train.py --data data.csv --target churned --kind classification

**นี่คือจุดตั้งต้น ไม่ใช่คำตอบ** — pipeline ข้างล่างเป็นของธรรมดาที่สุดที่ทำงานได้
หน้าที่ของคุณคือทำให้มันดีกว่านี้ · ลองเปลี่ยนโมเดล สร้างฟีเจอร์ใหม่ จูนพารามิเตอร์

---

**คุณแบ่ง train/val เอง และเลือกเมล็ดเอง** — ระบบไม่ยุ่งกับส่วนนี้เลย · `--seed`
ข้างล่างเป็นของคุณล้วนๆ เปลี่ยนได้ตามใจ ไม่มีผลกับคะแนนบน leaderboard

สิ่งที่ระบบทำคือรับ `pipeline.pkl` ไปรันกับ **ข้อมูลที่คุณไม่เคยเห็น** ซึ่งอยู่บน
เซิร์ฟเวอร์ · ไฟล์ที่คุณดาวน์โหลดมาไม่มีข้อมูลส่วนนั้นอยู่ ไม่ว่าจะพยายามแค่ไหน

**คะแนนบน val ของคุณจะไม่เท่ากับบน leaderboard เป๊ะ** — คนละข้อมูล · ถ้ามันต่าง
กันมากผิดปกติ แปลว่าโมเดล overfit กับส่วนที่คุณมี ซึ่งเป็นสิ่งที่ต้องแก้ ไม่ใช่บั๊ก
"""

import argparse

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabular.metrics import score


def build_pipeline(X: pd.DataFrame, kind: str) -> Pipeline:
    """ทุกขั้นตอนอยู่ในนี้ — เพราะระบบส่งข้อมูลดิบเข้ามาตอนทำนาย

    เลือกคอลัมน์ตาม dtype แทนการเขียนชื่อไว้ตรงๆ เพราะคอลัมน์เป็นอะไรก็ได้แล้วแต่
    โจทย์ · **นี่เป็นจุดเริ่มต้นที่หยาบ** — คอลัมน์ที่เป็นตัวเลขไม่ได้แปลว่าควร
    ปฏิบัติกับมันแบบตัวเลขเสมอไป (รหัสไปรษณีย์ · รหัสสาขา) ลองแยกเองดู
    """
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    pre = ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                # `handle_unknown="ignore"` จำเป็น — ชุดที่ใช้ตัดสินอาจมีหมวดหมู่
                # ที่ไม่เคยโผล่ในชุดที่คุณเทรน ถ้าไม่ตั้ง โมเดลจะพังทั้งชุด
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
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
    ap.add_argument("--data", default="data.csv", help="ไฟล์ที่ดาวน์โหลดจากหน้าโจทย์")
    ap.add_argument("--target", required=True, help="ชื่อคอลัมน์เฉลย — ดูได้ที่หน้าโจทย์")
    ap.add_argument("--kind", default="classification",
                    choices=["classification", "regression"])
    ap.add_argument("--primary", default=None, help="ว่างไว้จะใช้ macro_f1 หรือ r2 ตาม kind")
    ap.add_argument("--val-size", type=float, default=0.2, help="สัดส่วนที่กันไว้วัดเอง")
    ap.add_argument("--seed", type=int, default=0, help="เมล็ดของคุณเอง — เปลี่ยนได้ตามใจ")
    ap.add_argument("--out", default="pipeline.pkl")
    args = ap.parse_args()

    primary = args.primary or ("macro_f1" if args.kind == "classification" else "r2")

    frame = pd.read_csv(args.data)
    if args.target not in frame.columns:
        raise SystemExit(
            f"ไม่มีคอลัมน์ {args.target!r} ใน {args.data} — ที่มีคือ {list(frame.columns)}"
        )
    y = frame[args.target]
    X = frame.drop(columns=[args.target])

    # แบ่งเอง ด้วยเมล็ดของเราเอง — `stratify` ทำให้สัดส่วนคลาสบน val ใกล้เคียง
    # ของจริง ซึ่งสำคัญมากเมื่อคลาสไม่สมดุล
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed,
        stratify=y if args.kind == "classification" else None,
    )

    pipe = build_pipeline(X_train, args.kind).fit(X_train, y_train)

    result = score(
        y_val, pipe.predict(X_val),
        kind=args.kind, primary=primary, seed=args.seed,
        labels=sorted(y.dropna().unique().tolist()) if args.kind == "classification" else None,
    )
    print(f"{primary} บน val ของคุณ = {result.primary:+.4f} "
          f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}]")
    print("  (ตัวเลขนี้ประมาณคะแนนบน leaderboard ได้ แต่ไม่เท่ากันเป๊ะ — คนละข้อมูล)")

    joblib.dump(pipe, args.out)
    print(f"บันทึก {args.out} แล้ว — ส่งด้วย `arena submit <รหัส competition>`")


if __name__ == "__main__":
    main()
