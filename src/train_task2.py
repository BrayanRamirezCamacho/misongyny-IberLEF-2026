import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("data/train.csv")

TEXT_COL = "lyrics"
LABELS = ["S", "V", "H", "NR"]

X = df[TEXT_COL].fillna("")
Y = df[LABELS]

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
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
pred = model.predict(X_val)

print(classification_report(y_val, pred, target_names=LABELS))
