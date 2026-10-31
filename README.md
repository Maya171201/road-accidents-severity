<div dir="rtl">

# חיזוי חומרת תאונות דרכים בישראל (2024)

פרויקט בקורס נושאים מתקדמים בלמידת מכונה (Data Mining), אוניברסיטת אריאל, תשפ"ז.
צוות: _שמות_

> **שימוש ב־GenAI:** הקוד בריפו נכתב בעזרת Claude (Anthropic).
> כל קטע קוד שנכתב ע"י AI מסומן `### CODE BY CLAUDE ###` … `####`, כפי שדורשות הנחיות הקורס.

## הדאטה
תאונות דרכים עם נפגעים 2024, קובץ לשימוש הציבור (PUF) של הלשכה המרכזית לסטטיסטיקה:
https://data.gov.il/dataset/2023-puf (הקובץ `h20241331data.csv`). כל שורה היא תאונה אחת: 8,315 תאונות, 45 עמודות.
מילון הקודים (`h20241331codebook.xlsx`) נמצא באותו עמוד.

**משתנה המטרה:** `HUMRAT_TEUNA` (חומרת התאונה): קטלנית 4.9% · קשה 29.5% · קלה 65.6%.

## ממצאים עד עכשיו
1. **משתנה מטרה לא מאוזן:** רק 4.9% מהתאונות קטלניות, ולכן לא בוחרים מודל לפי Accuracy.
2. **חסרים מבניים:** עמודות הרחוב ריקות בתאונות בין־עירוניות, ועמודות מספר הכביש ריקות בתאונות עירוניות. במקומן נוצרו דגלים.
3. **ארטיפקט בדיווח:** ב־1,235 תאונות (15%) **כל** שדות תנאי הדרך "לא ידוע", ו־77% מהן "קשות" (לעומת 21% בשאר). כשמשאירים אותן, F1-macro ב־CV עולה מ־0.437 ל־0.528 (ב־XGBoost) בלי ללמוד שום דבר על הדרך. לכן הניסויים הראשיים בלעדיהן (7,058 תאונות, אחרי שהוצאו גם 22 תאונות בלי קואורדינטות).

## תוצאות הסיווג (‏5-fold stratified CV על ה־train, ‏7,058 תאונות)
**רק שיטות שנלמדו בהרצאות הקורס.** בראש כל מחברת יש טבלה שמראה מאיזה שיעור כל שיטה.

**ניסוי 1: ‏6 מודלים × 3 שיטות איזון** (הטובים ביותר):
| מודל | טיפול באי־איזון | F1-macro | Recall קטלנית | PR-AUC |
|---|---|---|---|---|
| Random Forest | class_weight | 0.447 | 0.354 | 0.448 |
| XGBoost | class_weight | 0.437 | 0.408 | 0.446 |
| Logistic Regression | class_weight | 0.435 | 0.538 | 0.443 |
| Decision Tree | SMOTE | 0.424 | 0.266 | 0.411 |
| MLP | SMOTE | 0.404 | 0.146 | 0.405 |
| KNN | בלי טיפול | 0.328 | 0.016 | 0.408 |

**ניסוי 2: כל שיטות ה־Resampling משיעור 1 על Random Forest:**
Random Oversampling ‏0.460 · class_weight ‏0.447 · SMOTE ‏0.421 · SMOTEENN ‏0.356 · בלי טיפול 0.320 · Tomek Links ‏0.314 · NearMiss ‏0.261 (F1-macro).

**על ה־test** (Random Forest + Random Oversampling, נבחר לפי ה־CV): F1-macro ‏0.434, ‏Recall קטלנית 0.139. הורדת הסף לקטלנית ל־0.3 מעלה את ה־Recall ל־0.266, ול־0.2 מעלה אותו ל־0.456.
הטבלאות המלאות ב־`figures/`.

## מבנה הריפו
</div>

```
data/raw/        קובץ הלמ"ס המקורי (לא נערך)
data/processed/
notebooks/       01_eda.ipynb, 02_classification.ipynb
src/data.py      טעינה, פיצ'רים חדשים, עיבוד מקדים (משותף למחברות)
figures/         כל הגרפים וטבלאות התוצאות
```

<div dir="rtl">

## איך מריצים
</div>

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook notebooks/
```

<div dir="rtl">

מריצים קודם את `01_eda.ipynb` ואחר כך את `02_classification.ipynb`. בכל שלב אקראי `random_state=42`, ולכן המספרים זהים בכל הרצה.

## ספריות
pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost, jupyter (הגרסאות ב־`requirements.txt`).

</div>
