### CODE BY CLAUDE ###
"""טעינת קובץ תאונות הדרכים של הלמ"ס (2024) והכנת הפיצ'רים.

משותף לשתי המחברות, כדי שה־EDA והמודלים ישתמשו בדיוק באותן עמודות.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "h20241331data.csv"
FIGURES = ROOT / "figures"

RANDOM_STATE = 42
TARGET = "HUMRAT_TEUNA"
# קוד הלמ"ס: 1 = קטלנית, 2 = קשה, 3 = קלה (לאמת במילון הקודים)
SEVERITY_NAMES = {1: "קטלנית", 2: "קשה", 3: "קלה"}

# ערך אחד בכל השורות, או מזהה שורה: אין בהם מידע
DROP_NO_INFO = ["pk_teuna_fikt", "sug_tik", "SHNAT_TEUNA"]

# מזהי רחוב / צומת / כביש עם מאות ערכים. One-hot שלהם היה יוצר אלפי עמודות כמעט ריקות,
# ולכן מחליפים אותם בדגלים שבהמשך ובעמודות מיקום גסות יותר (מחוז, נפה, X, Y).
DROP_HIGH_CARDINALITY = [
    "REHOV1", "REHOV2", "BAYIT", "ZOMET_IRONI", "ZOMET_LO_IRONI",
    "KVISH1", "KVISH2", "KM", "SEMEL_YISHUV", "EZOR_TIVI", "MAAMAD_MINIZIPALI",
]

CATEGORICAL = [
    "THUM_GEOGRAFI", "SUG_DEREH", "YEHIDA", "HODESH_TEUNA", "SUG_YOM",
    "YOM_LAYLA", "YOM_BASHAVUA", "SUG_TEUNA", "HAD_MASLUL", "RAV_MASLUL",
    "MEHIRUT_MUTERET", "TKINUT", "ROHAV", "SIMUN_TIMRUR", "TEURA",
    "MEZEG_AVIR", "PNE_KVISH", "SUG_EZEM", "MERHAK_EZEM", "LO_HAZA",
    "OFEN_HAZIYA", "MEKOM_HAZIYA", "KIVUN_HAZIYA", "MAHOZ", "NAFA",
    "ZURAT_ISHUV", "STATUS_IGUN", "HOUR",
]
NUMERIC = ["X", "Y"]
BINARY = ["HAS_STREET", "HAS_ROAD_NUMBER", "AT_JUNCTION_ID"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV, encoding="utf-8-sig")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """עמודות חדשות ששומרות את החלק השימושי מעמודות המזהים שנזרקו."""
    out = df.copy()
    # ל־SHAA יש 96 קודים: רבעי שעה ביממה (1 = 00:00–00:14). לאמת במילון.
    # מקבצים לשעות, כדי לרדת מ־96 קטגוריות ל־24.
    out["HOUR"] = (out["SHAA"] - 1) // 4
    # החסרים כאן מבניים (תאונה עירונית מול בין־עירונית),
    # ולכן עצם העובדה שהשדה מלא היא מידע.
    out["HAS_STREET"] = out["REHOV1"].notna().astype(int)
    out["HAS_ROAD_NUMBER"] = out["KVISH1"].notna().astype(int)
    out["AT_JUNCTION_ID"] = (out["ZOMET_IRONI"].notna() | out["ZOMET_LO_IRONI"].notna()).astype(int)
    return out


def details_missing(df: pd.DataFrame) -> pd.Series:
    """תאונות שבהן כל שדות תנאי הדרך "לא ידוע" (כ־15% מהשורות).

    77% מהן "קשות", לעומת 21% בשאר הדאטה: ארטיפקט של הדיווח, לא תנאי בדרך.
    מודל היה לומד "אין פרטים, אז קשה".
    """
    return ((df["MEZEG_AVIR"] == 9) & (df["PNE_KVISH"] == 9) & (df["TKINUT"] == 0)
            & (df["ROHAV"] == 0) & (df["MEHIRUT_MUTERET"] == 0))


def get_xy(df: pd.DataFrame, drop_missing_details: bool = True):
    """פיצ'רים ומשתנה מטרה, מוכנים ל־train_test_split.

    כברירת מחדל מוציאים את התאונות בלי פרטי דרך (ראו details_missing).
    """
    if drop_missing_details:
        df = df[~details_missing(df)]
    # ל־22 תאונות אין קואורדינטות: מוציאים אותן במקום להשלים ערך
    df = df.dropna(subset=["X", "Y"])
    df = add_features(df)
    X = df[CATEGORICAL + NUMERIC + BINARY]
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """נכנס לתוך כל Pipeline, כך שהוא לומד רק מחלק האימון (שיעור 1: Data Leakage).

    קטגוריות: One-hot. מספריות (X, Y): MinMaxScaler לטווח 0–1 (שיעור 2: Scaling).
    """
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", MinMaxScaler(), NUMERIC),
        ("bin", "passthrough", BINARY),
    ])
####
