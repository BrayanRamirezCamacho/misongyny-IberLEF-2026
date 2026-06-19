import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

data_path = Path("data/train.csv")

if not data_path.exists():
    raise FileNotFoundError(
        "No se encontró data/train.csv"
    )

df = pd.read_csv(data_path)

if df.empty:
    raise ValueError(
        "El archivo train.csv está vacío"
    )

else:
	print("Columnas encontradas:")
	print(df.columns.tolist())

	print("\nDistribución de clases:")
	print(df["is_misogynystic"].value_counts())

TEXT_COL = "lyrics"
LABEL_COL = "is_misogynystic"

X = df[TEXT_COL].fillna("")
y = df[LABEL_COL]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

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

joblib.dump(
    model,
    "models/task1_model.pkl"
)

print("\nModelo guardado en:")
print("models/task1_model.pkl")

pred = model.predict(X_val)

print(classification_report(y_val, pred))
print("Macro F1:", f1_score(y_val, pred, average="macro"))
