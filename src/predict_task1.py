import pandas as pd
import joblib
from pathlib import Path

model_path = Path("models/task1_model.pkl")

if not model_path.exists():
    raise FileNotFoundError(
        "No existe models/task1_model.pkl. Ejecuta train_task1.py primero."
    )

model = joblib.load(model_path)

test_df = pd.read_csv("data/test.csv")

preds = model.predict(
    test_df["lyrics"].fillna("")
)

Path("outputs").mkdir(exist_ok=True)

submission = pd.DataFrame({
    "id": test_df["id"],
    "preds": preds
})

submission.to_csv(
    "outputs/task_1_predictions.csv",
    index=False
)

print("Archivo generado:")
print("outputs/task_1_predictions.csv")
