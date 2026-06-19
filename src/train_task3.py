import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

data_path = Path("data/train.csv")

df = pd.read_csv(data_path)

TEXT_COL = "lyrics"
LABEL_COL = "has_gender_stereotype"

X = df[TEXT_COL].fillna("")
y = df[LABEL_COL].fillna("N")
y = y.apply(lambda value: "Y" if value == "Y" else "N")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=20000
    )),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train, y_train)

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

joblib.dump(model, "models/task3_model.pkl")

pred = model.predict(X_val)

report = classification_report(
    y_val,
    pred,
    zero_division=0
)

macro_precision = precision_score(y_val, pred, average="macro", zero_division=0)
macro_recall = recall_score(y_val, pred, average="macro", zero_division=0)
macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)

print(report)
print("Macro Precision:", macro_precision)
print("Macro Recall:", macro_recall)
print("Macro F1:", macro_f1)

with open("outputs/task3_metrics.txt", "w", encoding="utf-8") as f:
    f.write("Task 3 - Gender Stereotype Detection\n")
    f.write("====================================\n\n")
    f.write(report)
    f.write("\n")
    f.write(f"Macro Precision: {macro_precision:.4f}\n")
    f.write(f"Macro Recall: {macro_recall:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")

print("\nModelo guardado en models/task3_model.pkl")
print("Métricas guardadas en outputs/task3_metrics.txt")
