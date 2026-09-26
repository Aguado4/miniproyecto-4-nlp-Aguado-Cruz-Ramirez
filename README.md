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
| §9 | LoRA, otro checkpoint y condición balanceada |
| §10 | El generador como clasificador |
| §11 | ¿Memoriza? |
| §12 | Aumento de datos para el clasificador de MP1 |
| §13–§15 | Demo, análisis cualitativo, conclusiones |

## Qué corregimos del notebook guía

`pad_token = eos_token` (el modelo no aprende a terminar), `generate` sin caché y decodificando
token a token (rompe tildes), split sin semilla y evaluación solo a ojo. Detalle en
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## Resultados

Corrida de referencia: RTX 4060 local, 56 celdas de código, 0 errores, **34 min**. Detalle en
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

| Pregunta | Respuesta | Dónde |
|---|---|---|
| ¿Aprende el dominio? | Perplejidad **105 → 35** por token (218 → 57 por palabra) | §6, §7 |
| ¿Obedece el tipo de establecimiento? | **Sí: 0,900**, con techo medible de 0,951 | §7 |
| ¿Obedece la estrella pedida? | **Solo en parte: 0,333** (azar 0,200), pero MAE 0,967 frente a 1,55 sin ajuste | §7 |
| ¿Cuánto depende de la decodificación? | Mucho: de 0,333 con top-p a **0,533 con beam 4** | §8 |
| ¿Se puede ajustar con el 0,2 % de los pesos? | LoRA queda a un 25 % de perplejidad, con **0,236 %** entrenable | §9 |
| ¿Sirve como clasificador? | No en macro-F1 (0,387 contra 0,500 del TF-IDF de MP1), sí en accuracy y MAE | §10 |
| ¿Memoriza? | No: copia **menos** que las reseñas reales no vistas (0,1 % contra 0,3 %) | §11 |
| ¿Sirven sus reseñas como aumento de datos? | **No: empeoran** el clasificador (0,4997 contra 0,5235) por ruido de etiqueta | §12 |

**Veredicto:** H1 confirmada, H2 parcial (el tipo sí, la polaridad ordinal no), **H3 refutada**.
Las dos hipótesis que cayeron son las que más enseñan, y §10, §12 y §14 miden el mecanismo en
cada caso: el fallo de polaridad es asimétrico (73 % de error grave en 1★ y 0 % en 5★) y el
modelo **subusa la negación** en las reseñas negativas que escribe (2,80 por 100 palabras contra
3,37 en las reales).

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
