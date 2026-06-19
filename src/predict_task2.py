import joblib
import pandas as pd
from pathlib import Path

model_path = Path("models/task2_model.pkl")

if not model_path.exists():
    raise FileNotFoundError(
        "No existe models/task2_model.pkl. Ejecuta train_task2.py primero."
    )

model = joblib.load(model_path)

test_df = pd.read_csv("data/test.csv")

preds = model.predict(
    test_df["lyrics"].fillna("")
)

submission = pd.DataFrame({
    "id": test_df["id"],
    "S": preds[:, 0],
    "V": preds[:, 1],
    "H": preds[:, 2],
})

Path("outputs").mkdir(exist_ok=True)

submission.to_csv(
    "outputs/task_2_predictions.csv",
    index=False
)

print("Archivo generado:")
print("outputs/task_2_predictions.csv")
