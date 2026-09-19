# Miniproyecto 4 - NLP

**Reseñas a la carta: GPT-2 condicionado sobre reseñas turísticas en español**

Maestría · Universidad Icesi · Curso de Procesamiento de Lenguaje Natural

Autores: Juan José Aguado · Juan David Cruz · Juan Diego Ramírez

---

## El problema

Ajustar un GPT-2 en español para **generar reseñas turísticas con la polaridad y el tipo de
establecimiento que se le pidan**, sobre el corpus Rest-Mex de los Miniproyectos
[1](../miniproyecto%201/), [2](../miniproyecto%202/) y [3](../miniproyecto%203/).

> **¿Puede un modelo generativo aprender no solo el estilo de las reseñas sino la polaridad
> que se le pide, y sirve lo que genera para algo más que leerlo?**

- **H1.** El *fine-tuning* baja mucho la perplejidad; el prefijo de control, un poco más.
- **H2.** El control funciona en 1★ y 5★ y falla en 2★–4★, como todos los clasificadores anteriores.
- **H3.** Reseñas sintéticas de 1★–3★ mejoran a TF-IDF + LogReg más que el sobremuestreo.

## La propuesta

| Sección | Qué hace |
|---|---|
| §4.4–4.8 | Tokenizador, formato con prefijo de control, jueces y métricas |
| §5 | Modelo base: perplejidad de tres GPT-2 en español y *prompting* sin ajuste |
| §6 | **Técnica 1**: *fine-tuning* sin condición (como el guía) |
| §7 | **Técnica 2**: *fine-tuning* condicional por estrellas y tipo |
| §8 | Decodificación: `generate` del guía corregido y ocho estrategias comparadas |
| §9 | LoRA y otro checkpoint |
| §10 | El generador como clasificador |
| §11 | ¿Memoriza? |
| §12 | Aumento de datos para el clasificador de MP1 |
| §13–§15 | Demo, análisis cualitativo, conclusiones |

## Qué corregimos del notebook guía

`pad_token = eos_token` (el modelo no aprende a terminar), `generate` sin caché y decodificando
token a token (rompe tildes), split sin semilla y evaluación solo a ojo. Detalle en
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## Resultados

> Pendiente de la corrida de referencia ([`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)).

## Cómo ejecutarlo

**Colab (recomendado):** abrir `notebooks/miniproyecto4_restmex_gpt.ipynb`, GPU T4, ejecutar todo.

**Local:**

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
python -m spacy download es_core_news_lg
jupyter lab notebooks/miniproyecto4_restmex_gpt.ipynb
```

## Estructura

```
.
├── CLAUDE.md · README.md · consigna.txt · rubrica.txt · requirements.txt
├── docs/        SPEC · PLAN · DATASET · DECISIONS · EXPERIMENTS
├── notebooks/   miniproyecto4_restmex_gpt.ipynb
└── results/     figures/ · metrics/
```
