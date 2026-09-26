# EXPERIMENTS - Bitácora de resultados

> **Regla:** solo números efectivamente ejecutados. **Estado:** corrida de referencia completa.

| Campo | Valor |
|---|---|
| Fecha | 2026-09-26 |
| Plataforma / GPU | Local (Windows 11) · NVIDIA GeForce RTX 4060, 8 GB |
| Python / torch / transformers | 3.12.2 / 2.11.0+cu128 / 5.17.0 |
| Semilla | 42 (fijada antes de cada entrenamiento y de cada bloque de generación) |
| Checkpoint principal | `mrm8488/spanish-gpt2` (el del notebook guía) |
| `CFG_GPT` | 10.000 reseñas de train, 3 épocas, batch 16, lr 5e-5, fp16 |
| `CFG_EST` | 3.000 reseñas, 3 épocas |
| `MAX_LEN_GPT` | 160 (P95 con prefijo = 195, tope 160) · 14,0 % del train truncado · mediana 74 tokens |
| Ejecución | Restart & Run All, 56/56 celdas de código, 0 errores, **2.048 s (34 min)** |
| Referencias externas | MP1 `EXPERIMENTS.md` §2 (mismo test de Split A) |

> **Sobre el presupuesto.** El reparto entre reseñas y épocas se eligió **midiendo** su efecto
> sobre el control, no por criterio (D-411). Una corrida previa con 2 épocas sobre las 32.000
> reseñas daba perplejidad 32,9 frente a 36,0, a cambio de casi dos horas. Las cifras de abajo
> describen un generador con presupuesto reproducible, no el mejor posible sobre este corpus.

> **Sobre la reproducibilidad exacta.** Con `fp16` los núcleos de cuDNN no son deterministas, así
> que repetir el notebook mueve los resultados en el 3.º-4.º decimal. Además, el control y las
> métricas de decodificación se miden sobre 60 peticiones por fila: **diferencias menores de ~0,05
> no deben interpretarse como reales.**

## 1. Baselines heredados (bloqueante: deben reproducir MP1)

| Baseline | Accuracy | macro-F1 |
|---|---:|---:|
| Clase mayoritaria (5★) | 0.6565 | 0.1585 |
| Azar estratificado | 0.4798 | 0.1931 |

Reproducen exactamente los de MP1, y las 63 celdas heredadas son byte-idénticas a las de aquel
notebook (celdas 3 a 65) → submuestra y Split A idénticos, comparación válida.

## 2. Jueces sobre reseñas reales de test (§4.7)

| Juez | Métrica | Valor |
|---|---|---:|
| Estrellas (TF-IDF + LogReg, modelo 1 de MP1) | macro-F1 | 0.5235 (MP1: 0.5235) |
| Estrellas, acierto por clase | 1★ / 2★ / 3★ / 4★ / 5★ | 0.695 / 0.295 / 0.488 / 0.538 / 0.761 |
| Tipo de establecimiento | accuracy | 0.9505 |

Estas cifras son el **techo** contra el que se lee el control: el juez no distingue mejor que eso.

## 3. Tokenizador (§4.5)

| Métrica | Valor |
|---|---:|
| Vocabulario (con `<\|pad\|>`) | 50.267 |
| Fertilidad en este corpus | 1.26 subpalabras/palabra |
| Fertilidad de `gpt2` inglés, referencia | 2.05 |

## 4. Perplejidad en test (§5 a §7, solo tokens de la reseña, 600 reseñas)

| Modelo | PPL por token | PPL por palabra | tokens/palabra |
|---|---:|---:|---:|
| base · `mrm8488/spanish-gpt2` (web) | 105.17 | 218.16 | 1.16 |
| base · `DeepESP/gpt2-spanish` (libros) | 163.30 | 381.55 | 1.17 |
| base · `datificate/gpt2-small-spanish` (Wikipedia) | 163.19 | 420.14 | 1.19 |
| FT sin condición (§6) | 36.04 | 63.21 | 1.16 |
| **FT condicional (§7)** | **34.89** | **56.94** | 1.14 |

La columna por palabra es la única comparable entre checkpoints, porque cada tokenizador parte
las palabras a su manera. Entrenamiento: 7,2 min cada ajuste. La validación se evalúa **por pasos** y no por época (con
tres épocas, «por época» daría tres puntos de curva): §6 va de 3.741 a 3.577 y §7 de 3.297 a
3.153, las dos aplanadas en los dos últimos puntos, así que el modelo converge sin sobreajustar.

## 5. Control (§7, 60 peticiones por medición)

| Qué se controla | top-p 0,92 | codiciosa + `no_repeat` | Base sin ajuste | Techo (juez en reales) |
|---|---:|---:|---:|---:|
| Tipo de establecimiento | 0.900 | 0.867 | 0.550 | 0.951 |
| Estrella exacta | 0.333 | 0.350 | 0.283 | 0.555 |
| MAE de la estrella | 1.217 | 0.967 | 1.550 | |

Azar = 0.200. Distribución de lo generado **sin** condición (§6), según el juez: 3,3 / 0,0 / 0,0 /
30,0 / 66,7 %, contra 2,6 / 2,6 / 7,5 / 21,6 / 65,6 % en el train real.

## 6. Decodificación (§8, 60 muestras por estrategia)

| Estrategia | distinct-2 | rep-4 | Fluidez (PPL base) | Control ★ | Control tipo | Palabras | s |
|---|---:|---:|---:|---:|---:|---:|---:|
| codiciosa | 0.055 | 0.352 | 7.07 | 0.350 | 0.867 | 41.3 | 0.93 |
| codiciosa + `no_repeat` | 0.128 | 0.000 | 28.77 | 0.350 | 0.867 | 28.5 | 0.93 |
| **beam 4 + `no_repeat`** | 0.135 | 0.002 | 19.34 | **0.533** | **1.000** | 33.5 | 2.36 |
| muestreo T=0,7 | 0.628 | 0.006 | 42.68 | 0.383 | 0.933 | 41.2 | 0.96 |
| muestreo puro T=1 | **0.864** | 0.000 | 130.59 | 0.367 | 0.867 | 58.3 | 0.95 |
| top-k 50 | 0.735 | 0.003 | 55.66 | 0.367 | 0.900 | 55.6 | 0.96 |
| top-p 0,92 (la usada en §10-§14) | 0.830 | 0.000 | 80.02 | 0.333 | 0.900 | 56.0 | 1.18 |
| ε-codiciosa del guía (eps=0,5) | 0.690 | 0.020 | 57.87 | 0.350 | 0.917 | 37.9 | 29.56 |

**Corrección del `generate` del guía verificada:** con `eps = 1` produce token a token la misma
salida que la codiciosa de Hugging Face. Su bucle en Python cuesta **29,56 s** frente a **1,18 s**
de la generación por lotes con caché de claves y valores: **25 veces**.

## 7. Estudios `CFG_EST` (§9, 3.000 reseñas, 3 épocas)

| Variante | PPL palabra | Control ★ | MAE ★ | Control tipo | % entrenable | s |
|---|---:|---:|---:|---:|---:|---:|
| `mrm8488` ajuste completo | **74.19** | 0.217 | 1.617 | 0.850 | 100.000 | 167.1 |
| `mrm8488` LoRA r=8 (`c_attn`) | 92.40 | 0.200 | 1.617 | 0.683 | **0.236** | **126.1** |
| `DeepESP` ajuste completo | 86.09 | **0.300** | 1.483 | **0.933** | 100.000 | 179.8 |
| `mrm8488` condición balanceada | 76.71 | 0.250 | **1.183** | 0.833 | 100.000 | 254.4 |

Balancear las clases **no** mejora el acierto exacto, solo el MAE. Medido también fuera del
notebook con cuatro configuraciones (D-411): la palanca del control son las épocas.

## 8. El generador como clasificador (§10, 600 reseñas de test)

| Modelo | macro-F1 | Accuracy | MAE | QWK |
|---|---:|---:|---:|---:|
| GPT-2 condicional, prior uniforme | 0.3510 | 0.5267 | 0.723 | 0.4595 |
| GPT-2 condicional, prior de train | 0.3866 | **0.7133** | **0.388** | 0.5596 |
| TF-IDF + LogReg de MP1, mismo subconjunto | **0.4996** | 0.6550 | 0.402 | **0.7132** |

## 9. Memorización (§11)

| Medida | Generadas | Reales de test |
|---|---:|---:|
| 8-gramas presentes en train | 0.1 % | 0.3 % |
| Con ≥ 50 % de 8-gramas copiados | 0.0 % | |
| Similitud TF-IDF máxima con train | 0.28 | |

## 10. Aumento de datos (§12, mismo test de MP1, 4.000 reseñas)

| Variante de train | macro-F1 | Accuracy | MAE | QWK | F1 en 1★ |
|---|---:|---:|---:|---:|---:|
| real (el de MP1) | 0.5235 | 0.6785 | 0.376 | **0.7261** | 0.5959 |
| + sobremuestreo de 1★-3★ | **0.5303** | **0.6805** | **0.371** | 0.7250 | 0.5882 |
| + 1.200 sintéticas de 1★-3★ | 0.4997 | 0.6683 | 0.412 | 0.6884 | 0.5338 |
| + sintéticas filtradas por el juez (192) | 0.5217 | 0.6775 | 0.381 | 0.7207 | 0.5911 |

El juez reconoce la estrella pedida en solo el **16,0 %** de las sintéticas (192 de 1.200), porque
§12 genera justo las clases donde el control falla. El filtro recupera el daño **descartando el
84 % de los datos**, y es circular (mismo modelo que filtra y que se entrena).

## 11. Modos de falla (§14)

| Estrella pedida | Repetición | Corta o truncada | Polaridad cruzada (≥2★) |
|---|---:|---:|---:|
| 1★ | 0 % | 56.8 % | **73.0 %** |
| 2★ | 0 % | 49.2 % | 50.0 % |
| 3★ | 0 % | 49.2 % | 34.5 % |
| 4★ | 0 % | 41.7 % | 0.0 % |
| 5★ | 0 % | 58.3 % | 0.0 % |

Negaciones por 100 palabras en las pedidas como 1★: **2.80** generadas contra **3.37** en las
reales de train. El truncamiento es consecuencia de `max_new_tokens` = 90, no del modelo.

## 12. Veredicto de hipótesis

| Hipótesis | Veredicto |
|---|---|
| **H1** · el ajuste baja mucho la perplejidad; el prefijo un poco más | **Confirmada** (105.2 → 36.0 → 34.9) |
| **H2** · el control funciona en los extremos y falla en 2★-4★ | **Parcial.** Tipo 0.900 sobre techo 0.951; estrella 0.333 con MAE 0.967. El fallo no es en 2★-4★ sino **asimétrico**: 73 % de polaridad cruzada en 1★ y 0 % en 4★-5★ |
| **H3** · las sintéticas mejoran el clasificador más que el sobremuestreo | **Refutada** (0.4997 contra 0.5303 del sobremuestreo y 0.5235 del train real), con el mecanismo medido: 16 % de control en las clases generadas |
