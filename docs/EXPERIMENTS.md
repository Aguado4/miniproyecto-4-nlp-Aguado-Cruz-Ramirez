# EXPERIMENTS - Bitácora de resultados

> **Regla:** solo números efectivamente ejecutados. **Estado:** sin ejecutar.

| Campo | Valor |
|---|---|
| Fecha / plataforma / GPU | — |
| Python / torch / transformers | — |
| Semilla | 42 |
| Checkpoint principal | `mrm8488/spanish-gpt2` |
| `MAX_LEN_GPT` / truncamiento | — |

## 1. Baselines heredados (deben reproducir MP1: acc 0.6565, macro-F1 0.1585)

| Baseline | Accuracy | macro-F1 |
|---|---:|---:|
| Clase mayoritaria | | |
| Azar estratificado | | |

## 2. Jueces sobre reseñas reales de test

| Juez | macro-F1 | Accuracy |
|---|---:|---:|
| Estrellas (TF-IDF + LogReg) | | |
| Tipo (TF-IDF + LogReg) | | |

## 3. Perplejidad en test (solo tokens de la reseña)

| Modelo | Perplejidad |
|---|---:|
| `mrm8488/spanish-gpt2` base | |
| `DeepESP/gpt2-spanish` base | |
| `datificate/gpt2-small-spanish` base | |
| FT sin condición | |
| FT condicional (prefijo correcto) | |

## 4. Control (FT condicional, top-p)

| Estrella pedida | Acierto del juez | Acierto del juez en reales |
|---|---:|---:|
| 1★ – 5★ | | |

## 5. Decodificación

| Estrategia | distinct-2 | rep-4 | Fluidez (PPL base) | Control (acc) | Longitud |
|---|---:|---:|---:|---:|---:|

## 6. Estudios `CFG_EST`

| Variante | PPL | Control | % entrenable | Tiempo (s) |
|---|---:|---:|---:|---:|

## 7. Clasificador generativo · 8. Aumento de datos

| Modelo | macro-F1 | MAE | QWK |
|---|---:|---:|---:|
| MP1 · TF-IDF + LogReg | 0.5235 | 0.376 | 0.7261 |
| GPT-2 condicional como clasificador | | | |
| TF-IDF + sobremuestreo | | | |
| TF-IDF + sintéticas | | | |
