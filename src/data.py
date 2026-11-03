### CODE BY CLAUDE ###
"""טעינת קובץ תאונות הדרכים של הלמ"ס (2024) והכנת הפיצ'רים.

משותף לשתי המחברות, כדי שה־EDA והמודלים ישתמשו בדיוק באותן עמודות.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "h20241331data.csv"
FIGURES = ROOT / "figures"

RANDOM_STATE = 42
TARGET = "HUMRAT_TEUNA"
# קוד הלמ"ס: 1 = קטלנית, 2 = קשה, 3 = קלה (לאמת במילון הקודים)
SEVERITY_NAMES = {1: "קטלנית", 2: "קשה", 3: "קלה"}

# ההחלטות על העמודות: פגישת הצוות (3.11). הרשימות כאן הן תיעוד; למודל נכנס רק מה שב־CATEGORICAL, NUMERIC ו־BINARY.

# ערך אחד בכל השורות, או מזהה שורה (מספר תאונה): אין בהם מידע
DROP_NO_INFO = ["pk_teuna_fikt", "sug_tik", "SHNAT_TEUNA"]

# הצוות החליט להוריד
DROP_BY_TEAM = ["SEMEL_YISHUV", "MAAMAD_MINIZIPALI"]

# SUG_DEREH מפוצלת לשני דגלים: URBAN (עירוני) ו־JUNCTION (בצומת)
REPLACED = ["SUG_DEREH"]

# נשארות בדאטה ונבדקות ב־EDA, אבל לא נכנסות למודל: אלה מזהים עם מאות ערכים,
# והקוד של רחוב או צומת עירוני חוזר בכמה יישובים, כך שבלי סמל היישוב אין לו משמעות.
# ZOMET_IRONI ו־ZOMET_LO_IRONI מאוחדות לעמודה אחת: ZOMET.
EDA_ONLY = ["REHOV1", "REHOV2", "BAYIT", "ZOMET_IRONI", "ZOMET_LO_IRONI", "KVISH1", "KVISH2", "KM"]

CATEGORICAL = [
    "THUM_GEOGRAFI", "YEHIDA", "HODESH_TEUNA", "SUG_YOM",
    "YOM_LAYLA", "YOM_BASHAVUA", "SUG_TEUNA", "HAD_MASLUL", "RAV_MASLUL",
    "MEHIRUT_MUTERET", "TKINUT", "ROHAV", "SIMUN_TIMRUR", "TEURA",
    "MEZEG_AVIR", "PNE_KVISH", "SUG_EZEM", "MERHAK_EZEM", "LO_HAZA",
    "OFEN_HAZIYA", "MEKOM_HAZIYA", "KIVUN_HAZIYA", "MAHOZ", "NAFA", "EZOR_TIVI",
    "ZURAT_ISHUV", "STATUS_IGUN", "HOUR",
]
NUMERIC = ["X", "Y"]
BINARY = ["URBAN", "JUNCTION", "HAS_STREET", "HAS_ROAD_NUMBER"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV, encoding="utf-8-sig")


# מילון הקודים של הלמ"ס: לכל עמודה, מספר הטבלה שלה (MS_TAVLA) במילון
CODES_CSV = ROOT / "data" / "raw" / "codes_dictionary_subset.csv"
SETTLEMENTS_CSV = ROOT / "data" / "raw" / "settlements.csv"
CODE_TABLE = {
    "YEHIDA": 1, "SUG_DEREH": 2, "HUMRAT_TEUNA": 4, "SUG_TEUNA": 5, "HAD_MASLUL": 10,
    "RAV_MASLUL": 11, "MEHIRUT_MUTERET": 12, "TKINUT": 13, "ROHAV": 14, "SIMUN_TIMRUR": 15,
    "TEURA": 16, "MEZEG_AVIR": 18, "PNE_KVISH": 19, "SUG_EZEM": 21, "MERHAK_EZEM": 22,
    "LO_HAZA": 23, "OFEN_HAZIYA": 24, "MEKOM_HAZIYA": 25, "KIVUN_HAZIYA": 26, "SUG_YOM": 37,
    "YOM_LAYLA": 38, "YOM_BASHAVUA": 39, "HODESH_TEUNA": 60, "THUM_GEOGRAFI": 68,
    "MAHOZ": 77, "NAFA": 79, "EZOR_TIVI": 80, "ZURAT_ISHUV": 81, "STATUS_IGUN": 205,
}
# שמות לדגלים שיצרנו (אין להם טבלה במילון)
FLAG_LABELS = {
    "URBAN": {1: "עירוני", 0: "לא עירוני"},
    "JUNCTION": {1: "בצומת", 0: "לא בצומת"},
}
# שם בעברית לכל עמודה (מטבלה 0 במילון), לכותרות הגרפים
COLUMN_NAMES = {
    "YEHIDA": "יחידה משטרתית", "SUG_DEREH": "סוג הדרך", "HUMRAT_TEUNA": "חומרת התאונה",
    "SUG_TEUNA": "סוג התאונה", "HAD_MASLUL": "דרך חד מסלולית", "RAV_MASLUL": "דרך דו מסלולית",
    "MEHIRUT_MUTERET": "מהירות מותרת", "TKINUT": "תקינות הדרך", "ROHAV": "רוחב הכביש",
    "SIMUN_TIMRUR": "סימון / תמרור", "TEURA": "תאורה", "MEZEG_AVIR": "מזג אוויר",
    "PNE_KVISH": "מצב פני הכביש", "YOM_LAYLA": "יום/לילה", "YOM_BASHAVUA": "יום בשבוע",
    "MAHOZ": "מחוז", "NAFA": "נפה", "EZOR_TIVI": "אזור טבעי", "SEMEL_YISHUV": "יישוב", "HOUR": "שעה",
    "HAS_ROAD_NUMBER": "יש מספר כביש", "URBAN": "עירוני / לא עירוני", "JUNCTION": "צומת / לא צומת",
}


@lru_cache(maxsize=None)
def code_labels(col: str) -> dict:
    """{קוד: שם} לעמודה, לפי מילון הקודים. עמודה שלא במילון: מילון ריק."""
    if col in FLAG_LABELS:
        return FLAG_LABELS[col]
    if col == "SEMEL_YISHUV":
        # שמות היישובים לא במילון התאונות, אלא ברשימת היישובים של הלמ"ס (data.gov.il).
        # קוד 0 = התאונה לא קרתה בתוך יישוב (דרך לא־עירונית).
        names = pd.read_csv(SETTLEMENTS_CSV, encoding="utf-8")
        return {0: "לא ביישוב", **dict(zip(names["SEMEL_YISHUV"], names["SHEM_YISHUV"]))}
    if col not in CODE_TABLE:
        return {}
    codes = pd.read_csv(CODES_CSV, encoding="utf-8")
    table = codes[codes["MS_TAVLA"] == CODE_TABLE[col]]
    return dict(zip(table["KOD"], table["TEUR"]))


def label(col: str, code) -> str:
    """השם של קוד אחד, ואם אין לו שם במילון: "קוד N"."""
    return code_labels(col).get(code, f"קוד {code}")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """עמודות חדשות ששומרות את החלק השימושי מעמודות המזהים שנזרקו."""
    out = df.copy()
    # ל־SHAA יש 96 קודים: רבעי שעה ביממה (1 = 00:00–00:14). לאמת במילון.
    # מקבצים לשעות, כדי לרדת מ־96 קטגוריות ל־24.
    out["HOUR"] = (out["SHAA"] - 1) // 4
    # SUG_DEREH: ‏1 = עירונית בצומת, 2 = עירונית לא בצומת, 3 = לא עירונית בצומת, 4 = לא עירונית לא בצומת.
    # מפצלים לשני דגלים (החלטת הצוות), כדי שכל אחד מהם יהיה פיצ'ר בפני עצמו.
    out["URBAN"] = out["SUG_DEREH"].isin([1, 2]).astype(int)
    out["JUNCTION"] = out["SUG_DEREH"].isin([1, 3]).astype(int)
    # צומת עירוני וצומת לא עירוני אף פעם לא מלאים יחד, ולכן מאחדים לעמודה אחת (החלטת הצוות). ל־EDA בלבד.
    out["ZOMET"] = out["ZOMET_IRONI"].fillna(out["ZOMET_LO_IRONI"])
    # החסרים כאן מבניים (תאונה עירונית מול בין־עירונית),
    # ולכן עצם העובדה שהשדה מלא היא מידע.
    out["HAS_STREET"] = out["REHOV1"].notna().astype(int)
    out["HAS_ROAD_NUMBER"] = out["KVISH1"].notna().astype(int)
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
    # שורות כפולות (שיעור 3). בקובץ הזה אין כאלה, אבל הבדיקה נשארת בקוד.
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "pk_teuna_fikt"])
    # ל־20 תאונות אין קואורדינטות. לא מוציאים אותן: ה־X וה־Y שלהן מושלמים
    # בחציון בתוך ה־Pipeline (שיעור 3: השלמת ערכים חסרים).
    df = add_features(df)
    X = df[CATEGORICAL + NUMERIC + BINARY]
    y = df[TARGET]
    return X, y


def build_preprocessor(scaler: str = "minmax", dense: bool = False) -> ColumnTransformer:
    """נכנס לתוך כל Pipeline, כך שהוא לומד רק מחלק האימון (שיעור 1: Data Leakage).

    קטגוריות: One-hot.
    מספריות (X, Y): השלמת חסרים בחציון (שיעור 3), ואז נרמול:
      "minmax"   = MinMaxScaler, טווח 0–1 (שיעורים 2–3)
      "standard" = StandardScaler, ממוצע 0 וסטיית תקן 1, z-score (שיעור 3)
    dense=True מחזיר מטריצה רגילה ולא דלילה, כי PCA צריך כזו (שיעור 3).
    """
    scalers = {"minmax": MinMaxScaler(), "standard": StandardScaler()}
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", scalers[scaler]),
    ])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense), CATEGORICAL),
        ("num", numeric, NUMERIC),
        ("bin", "passthrough", BINARY),
    ])
####
