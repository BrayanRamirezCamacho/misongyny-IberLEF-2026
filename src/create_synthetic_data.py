import pandas as pd
from pathlib import Path

Path("data").mkdir(exist_ok=True)

data = [
    {
        "id": "TRAIN_0001",
        "lyrics": "Ella camina libre por la ciudad y decide su propio camino",
        "is_misogynystic": "NM",
        "type_sexualization": 0,
        "type_violence": 0,
        "type_hate": 0,
        "has_gender_stereotype": "N",
    },
    {
        "id": "TRAIN_0002",
        "lyrics": "La canción describe a una mujer solo por su cuerpo y apariencia",
        "is_misogynystic": "M",
        "type_sexualization": 1,
        "type_violence": 0,
        "type_hate": 0,
        "has_gender_stereotype": "Y",
    },
    {
        "id": "TRAIN_0003",
        "lyrics": "Dice que las mujeres deben quedarse en casa y obedecer",
        "is_misogynystic": "M",
        "type_sexualization": 0,
        "type_violence": 0,
        "type_hate": 1,
        "has_gender_stereotype": "Y",
    },
    {
        "id": "TRAIN_0004",
        "lyrics": "La letra habla de respeto, igualdad y amor entre personas",
        "is_misogynystic": "NM",
        "type_sexualization": 0,
        "type_violence": 0,
        "type_hate": 0,
        "has_gender_stereotype": "N",
    },
    {
        "id": "TRAIN_0005",
        "lyrics": "El personaje amenaza y controla a su pareja en la canción",
        "is_misogynystic": "M",
        "type_sexualization": 0,
        "type_violence": 1,
        "type_hate": 0,
        "has_gender_stereotype": "N",
    },
    {
        "id": "TRAIN_0006",
        "lyrics": "La letra celebra que ella estudia, trabaja y toma decisiones",
        "is_misogynystic": "NM",
        "type_sexualization": 0,
        "type_violence": 0,
        "type_hate": 0,
        "has_gender_stereotype": "N",
    },
]

df = pd.DataFrame(data)
df.to_csv("data/train.csv", index=False)

test = pd.DataFrame({
    "id": ["TEST_0001", "TEST_0002", "TEST_0003"],
    "lyrics": [
        "La canción reduce a una mujer a su apariencia física",
        "Habla de igualdad y respeto en una relación",
        "Dice que ella debe obedecer siempre a su pareja",
    ]
})

test.to_csv("data/test.csv", index=False)

print("Synthetic train and test datasets created.")
