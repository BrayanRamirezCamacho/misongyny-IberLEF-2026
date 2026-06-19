import random
from pathlib import Path

import pandas as pd


random.seed(42)

Path("data").mkdir(exist_ok=True)


non_misogynistic_templates = [
    "Ella camina libre y decide su propio camino en la ciudad",
    "La canción habla de respeto igualdad y confianza en una relación",
    "La letra celebra que ella estudia trabaja y toma decisiones",
    "Se describe una relación basada en apoyo mutuo y cariño",
    "Ella sueña con viajar aprender y construir su futuro",
    "La historia presenta a una mujer fuerte independiente y creativa",
    "La canción reconoce su talento su voz y sus logros personales",
    "El texto habla de amor sin control ni posesión",
    "La letra muestra admiración por su inteligencia y valentía",
    "La canción defiende la libertad y la dignidad de todas las personas",
]

sexualization_templates = [
    "La letra describe a una mujer únicamente por su cuerpo y apariencia",
    "El cantante habla de ella como si fuera un objeto de deseo",
    "La canción reduce su valor a su belleza física",
    "El texto insiste en mirar su cuerpo sin mencionar su personalidad",
    "La letra presenta a la mujer como un premio o trofeo",
    "La canción usa lenguaje de posesión sobre el cuerpo de una mujer",
]

violence_templates = [
    "El personaje amenaza a su pareja y controla sus decisiones",
    "La letra describe una relación con intimidación y miedo",
    "El cantante dice que no acepta que ella termine la relación",
    "La canción normaliza los celos extremos y el control",
    "El texto presenta gritos amenazas y dominio sobre una mujer",
    "La letra sugiere castigar a una mujer por desobedecer",
]

hate_templates = [
    "La canción afirma que las mujeres son inferiores a los hombres",
    "La letra desprecia a las mujeres y niega su capacidad",
    "El texto usa insultos generalizados contra las mujeres",
    "La canción presenta a las mujeres como incapaces de decidir",
    "La letra dice que las mujeres solo deben obedecer",
    "El texto expresa rechazo y desprecio hacia las mujeres",
]

stereotype_templates = [
    "La canción dice que las mujeres deben quedarse en casa",
    "La letra afirma que una mujer debe cocinar limpiar y obedecer",
    "El texto sugiere que los hombres mandan y las mujeres cuidan",
    "La canción repite que una buena mujer debe ser sumisa",
    "La letra asocia a las mujeres únicamente con tareas domésticas",
    "El texto dice que una mujer no debería tener ambiciones propias",
]


def make_row(i, category):
    if category == "NM":
        lyrics = random.choice(non_misogynistic_templates)
        return {
            "id": f"TRAIN_{i:04d}",
            "lyrics": lyrics,
            "is_misogynystic": "NM",
            "type_sexualization": 0,
            "type_violence": 0,
            "type_hate": 0,
            "has_gender_stereotype": "N",
        }

    if category == "S":
        lyrics = random.choice(sexualization_templates)
        return {
            "id": f"TRAIN_{i:04d}",
            "lyrics": lyrics,
            "is_misogynystic": "M",
            "type_sexualization": 1,
            "type_violence": 0,
            "type_hate": 0,
            "has_gender_stereotype": random.choice(["Y", "N"]),
        }

    if category == "V":
        lyrics = random.choice(violence_templates)
        return {
            "id": f"TRAIN_{i:04d}",
            "lyrics": lyrics,
            "is_misogynystic": "M",
            "type_sexualization": 0,
            "type_violence": 1,
            "type_hate": 0,
            "has_gender_stereotype": random.choice(["Y", "N"]),
        }

    if category == "H":
        lyrics = random.choice(hate_templates)
        return {
            "id": f"TRAIN_{i:04d}",
            "lyrics": lyrics,
            "is_misogynystic": "M",
            "type_sexualization": 0,
            "type_violence": 0,
            "type_hate": 1,
            "has_gender_stereotype": random.choice(["Y", "N"]),
        }

    if category == "ST":
        lyrics = random.choice(stereotype_templates)
        return {
            "id": f"TRAIN_{i:04d}",
            "lyrics": lyrics,
            "is_misogynystic": "M",
            "type_sexualization": 0,
            "type_violence": 0,
            "type_hate": random.choice([0, 1]),
            "has_gender_stereotype": "Y",
        }


rows = []

categories = (
    ["NM"] * 80
    + ["S"] * 40
    + ["V"] * 40
    + ["H"] * 40
    + ["ST"] * 40
)

random.shuffle(categories)

for i, category in enumerate(categories, start=1):
    rows.append(make_row(i, category))

train_df = pd.DataFrame(rows)
train_df.to_csv("data/train.csv", index=False)


test_examples = [
    "La canción habla de respeto igualdad y apoyo mutuo",
    "La letra reduce a una mujer a su cuerpo y apariencia",
    "El personaje amenaza a su pareja si ella decide irse",
    "El texto afirma que las mujeres deben obedecer siempre",
    "La canción celebra que ella estudia trabaja y decide",
    "La letra dice que una mujer debe quedarse en casa",
] * 5

test_df = pd.DataFrame({
    "id": [f"TEST_{i:04d}" for i in range(1, len(test_examples) + 1)],
    "lyrics": test_examples,
})

test_df.to_csv("data/test.csv", index=False)

print("Synthetic datasets created.")
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print("\nTask 1 distribution:")
print(train_df["is_misogynystic"].value_counts())
print("\nTask 2 distribution:")
print(train_df[["type_sexualization", "type_violence", "type_hate"]].sum())
print("\nTask 3 distribution:")
print(train_df["has_gender_stereotype"].value_counts())
