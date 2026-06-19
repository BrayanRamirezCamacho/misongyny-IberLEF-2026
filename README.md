# MiSonGyny 2026 Baseline

Este repositorio contiene una solución reproducible para el concurso MiSonGyny 2026: Misogyny in Song Lyrics, parte de IberLEF 2026.

## Objetivo

Construir modelos base para:

1. Detección binaria de misoginia (la canción es misógina o no lo es).
2. Clasificación multilabel del tipo de misoginia (Sexualización, Violencia u Odio).
3. Identificación binaria de estereotipos de género (estereotípica o no).

## Metodología

Se utiliza una representación TF-IDF de las letras de las canciones, seguida de modelos lineales de clasificación:

- Logistic Regression para tareas binarias.
- One-vs-Rest Logistic Regression para clasificación multilabel.

## Justificación

El enfoque TF-IDF + Logistic Regression se eligió como baseline por ser:

- rápido de entrenar,
- interpretable,
- reproducible,
- adecuado para clasificación de texto con pocos recursos computacionales.

## Ejecución

```bash
pip install -r requirements.txt
python src/train_task1.py
python src/train_task2.py
python src/train_task3.py
```

## Limitaciones

Esta solución no usa modelos de lenguaje grandes ni embeddings contextuales. Su propósito es establecer una línea base funcional para comparación futura.

## Posibles mejoras

- Usar BETO, RoBERTa o multilingual BERT.
- Ajustar hiperparámetros.
- Manejar desbalance de clases.
Analizar errores por categoría.

## AVISO DE PROCEDENCIA DE LOS DATOS DE ENTRENAMIENTO

Debido a las limitaciones de acceso al conjunto de datos oficial de MiSonGyny 2026 (correo de activación de la cuenta no llega al correo electrónico), este repositorio incluye un conjunto de datos sintético que sigue la misma estructura de columnas y formato de etiquetas descrito por los organizadores. El objetivo es implementar un flujo de trabajo base reproducible compatible con las definiciones oficiales de las tareas.
