import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

data_path = Path("data/train.csv")

df = pd.read_csv(data_path)

TEXT_COL = "lyrics"
LABEL_COLS = ["type_sexualization", "type_violence", "type_hate"]

X = df[TEXT_COL].fillna("")
Y = df[LABEL_COLS].fillna(0).astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=20000
    )),
    ("clf", OneVsRestClassifier(
        LogisticRegression(max_iter=1000, class_weight="balanced")
    ))
])

model.fit(X_train, y_train)

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

joblib.dump(model, "models/task2_model.pkl")

pred = model.predict(X_val)

report = classification_report(
    y_val,
    pred,
    target_names=["S", "V", "H"],
    zero_division=0
)

macro_precision = precision_score(y_val, pred, average="macro", zero_division=0)
macro_recall = recall_score(y_val, pred, average="macro", zero_division=0)
macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)

print(report)
print("Macro Precision:", macro_precision)
print("Macro Recall:", macro_recall)
print("Macro F1:", macro_f1)

with open("outputs/task2_metrics.txt", "w", encoding="utf-8") as f:
    f.write("Task 2 - Misogyny Type Classification\n")
    f.write("=====================================\n\n")
    f.write(report)
    f.write("\n")
    f.write(f"Macro Precision: {macro_precision:.4f}\n")
    f.write(f"Macro Recall: {macro_recall:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")

print("\nModelo guardado en models/task2_model.pkl")
print("Métricas guardadas en outputs/task2_metrics.txt")
