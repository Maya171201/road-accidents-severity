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
3. **ארטיפקט בדיווח:** ב־1,235 תאונות (15%) **כל** שדות תנאי הדרך "לא ידוע", ו־77% מהן "קשות" (לעומת 21% בשאר). כשמשאירים אותן, F1-macro ב־CV עולה מ־0.447 ל־0.520 בלי ללמוד שום דבר על הדרך. לכן הניסויים הראשיים בלעדיהן (7,080 תאונות).

## תוצאות הסיווג (‏5-fold stratified CV על ה־train, ‏7,080 תאונות)
| מודל | טיפול באי־איזון | F1-macro | Recall קטלנית | PR-AUC |
|---|---|---|---|---|
| Random Forest | class_weight | 0.457 | 0.392 | 0.452 |
| XGBoost | sample_weight | 0.447 | 0.396 | 0.455 |
| Logistic Regression | class_weight | 0.437 | 0.508 | 0.451 |
| Decision Tree | SMOTE | 0.430 | 0.237 | 0.418 |
| MLP | SMOTE | 0.396 | 0.146 | 0.404 |
| KNN | בלי טיפול | 0.334 | 0.025 | 0.407 |

אחרי כוונון (`RandomizedSearchCV`), נבחר XGBoost לפי F1-macro ב־CV (‏0.460). על ה־test הוא מגיע ל־F1-macro של 0.458 ול־Recall של 0.238 בקטלנית. הורדת הסף לקטלנית ל־0.2 מעלה את ה־Recall ל־0.475. הטבלאות המלאות: `figures/experiment1_results.csv` ו־`figures/test_results.csv`.

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
