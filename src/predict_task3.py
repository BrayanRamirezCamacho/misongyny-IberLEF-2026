import joblib
import pandas as pd
from pathlib import Path

model_path = Path("models/task3_model.pkl")

if not model_path.exists():
    raise FileNotFoundError(
        "No existe models/task3_model.pkl. Ejecuta train_task3.py primero."
    )

model = joblib.load(model_path)

test_df = pd.read_csv("data/test.csv")

preds = model.predict(
    test_df["lyrics"].fillna("")
)

submission = pd.DataFrame({
    "id": test_df["id"],
    "preds": preds
})

Path("outputs").mkdir(exist_ok=True)

submission.to_csv(
    "outputs/task_3_predictions.csv",
    index=False
)

print("Archivo generado:")
print("outputs/task_3_predictions.csv")
