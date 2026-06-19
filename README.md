# MiSonGyny 2026 Baseline

Este repositorio contiene una solución reproducible para el concurso [MiSonGyny 2026: Misogyny in Song Lyrics](https://www.codabench.org/competitions/15010/), parte de IberLEF 2026.
Debido a limitaciones de acceso al conjunto de datos oficial (problemas con la verificación de la cuenta), este proyecto implementa un pipeline completo utilizando un conjunto de datos sintético que respeta la estructura y el esquema de etiquetado descritos por los organizadores del concurso.

## Objetivo general

El objetivo de MiSonGyny es detectar contenido misógino en letras de canciones en español e identificar distintas manifestaciones de misoginia presentes en ellas.

## Objetivos específicos

Construir modelos base para:

1. Detección binaria de misoginia (la canción es misógina o no lo es).
2. Clasificación multilabel del tipo de misoginia (Sexualización, Violencia u Odio).
3. Identificación binaria de estereotipos de género (estereotípica o no).

## Tareas del Concurso

### Task 1 — Detección de Misoginia
Clasificación binaria: M / NM.

### Task 2 — Clasificación del Tipo de Misoginia
Clasificación multietiqueta:
- S → Sexualización
- V → Violencia
- H → Odio

### Task 3 — Detección de Estereotipos de Género
Clasificación binaria: Y / N.

## Metodología

Se utiliza una representación TF-IDF (Term Frequency - Inverse Document Frequency) de las letras de las canciones, seguida de modelos lineales de clasificación:

- TF-IDF (unigramas y bigramas)
- Regresión Logística para tareas binarias.
- One-vs-Rest Logistic Regression para clasificación multilabel (Task 2).
- Evaluación mediante Precision, Recall y Macro F1

## Justificación

El enfoque TF-IDF + Logistic Regression se eligió como baseline por ser:

- rápido de entrenar,
- interpretable,
- reproducible,
- adecuado para clasificación de texto con pocos recursos computacionales.

## Estructura del Repositorio

```
misongyny-2026/
├── data/
├── models/
├── outputs/
├── notebooks/
└── src/
```

## Ejecución

```
python src/create_synthetic_data.py

python src/train_task1.py
python src/predict_task1.py

python src/train_task2.py
python src/predict_task2.py

python src/train_task3.py
python src/predict_task3.py
```

## Limitaciones

Esta solución no usa Grandes Modelos de Lenguaje (LLMs) ni embeddings de contexto. Su propósito es establecer una línea base funcional para comparación futura.
Utiliza datos sintéticos compatibles con el formato oficial del concurso debido a que el conjunto de datos original no estuvo disponible durante el desarrollo.

## Posibles mejoras

- Usar BETO, RoBERTa o multilingual BERT.
- Optimizar hiperparámetros.
- Manejar desbalance de clases.
- Analizar errores por categoría.

## Autor

Brayan Ramírez

Maestría en Ciencia de Datos

Universidad de Sonora

Mentorado por el Dr. Julio Waissman Vilanova
