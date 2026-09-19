# SPEC - Especificación del entregable

> **Estado:** borrador · **Versión:** 0.1 · **Última actualización:** 2026-09-19
>
> Contrato del entregable. Si la implementación se desvía, se actualiza primero este archivo.

---

## 1. Problema

### 1.1 Enunciado

Generar reseñas turísticas en español **controladas**: dado un número de estrellas (1 a 5) y un
tipo de establecimiento (hotel, restaurante, atracción), producir una reseña verosímil con esa
polaridad. Se ajusta un GPT-2 en español sobre el corpus Rest-Mex de las entregas anteriores.

> **¿Puede un modelo generativo aprender no solo el estilo de las reseñas sino la polaridad que
> se le pide, y sirve lo que genera para algo más que leerlo?**

### 1.2 Por qué esta pregunta

El guía de la Sesión 4 ajusta GPT-2 con chistes y evalúa a ojo («no parece muy gracioso»).
Nuestro corpus tiene algo que los chistes no: **una etiqueta**. Eso permite medir la generación
con números en lugar de impresiones:

- **Perplejidad** sobre reseñas reales no vistas (¿aprendió el dominio?).
- **Control**: un clasificador juez entrenado con reseñas reales (TF-IDF + LogReg, el modelo 1
  de MP1) mide si lo generado «a 1★» se lee como 1★.
- **Utilidad**: las entregas anteriores chocaron con la escasez de 1★ y 2★ (~840 reseñas
  cada una). Si el generador produce 1★ creíbles, deberían servir como **aumento de datos**.

Hipótesis:

> **H1.** El *fine-tuning* baja la perplejidad sobre reseñas reales muy por debajo del modelo
> base, y el prefijo de control la baja un poco más (la estrella informa sobre el texto).
>
> **H2 (control).** El juez reconoce la estrella pedida en las clases extremas (1★, 5★) a un
> nivel parecido a su acierto sobre reseñas reales, y falla en 2★–4★, las mismas fronteras
> difusas que ningún clasificador de MP1–MP3 resolvió.
>
> **H3 (utilidad).** Añadir reseñas sintéticas de 1★–3★ al train de TF-IDF + LogReg sube el
> macro-F1 sobre el mismo test de MP1 más que duplicar reseñas reales (sobremuestreo).

### 1.3 Por qué la propuesta es adecuada

1. **El corpus es grande y homogéneo** (32.000 reseñas de train, registro coloquial estable):
   mucho más favorable que los ~2.400 chistes del guía, cuya distribución el propio guía
   reconoce lejana a la del preentrenamiento.
2. **La etiqueta permite evaluar la generación objetivamente**, y cerrar el círculo con las
   tres entregas anteriores (mismo split, mismo test, misma `evaluar`).
3. **El prefijo en texto plano** (`Reseña de hotel con 1 estrella:`) sirve igual para el
   modelo base (*prompting*), el ajustado y LoRA, así que las comparaciones son justas.

## 2. Datos

`vg055/Rest-Mex2025`, ficha en [`DATASET.md`](DATASET.md). EDA y protocolo heredados de MP1;
generador entrenado sobre el train de Split A.

## 3. Protocolo y métricas

| Métrica | Qué mide | Dónde |
|---|---|---|
| **Perplejidad** (test, solo tokens de la reseña, sin el prefijo) | Ajuste al dominio | §5–§7, §9 |
| **Control** (acierto y macro-F1 del juez sobre la estrella pedida) | ¿Obedece la polaridad? | §7, §8 |
| **distinct-1/2** | Diversidad léxica entre muestras | §8 |
| **rep-4** | Repetición interna (4-gramas repetidos en un mismo texto) | §8 |
| **Fluidez** (perplejidad bajo el modelo base, independiente del ajuste) | Castellano bien formado | §8 |
| **Copia** (8-gramas presentes en train, similitud TF-IDF máxima) | ¿Memoriza? | §11 |
| **macro-F1 de TF-IDF + LogReg** con/sin aumento | Utilidad | §12 |

- `CFG_GPT` para las corridas principales; `CFG_EST` (8.000 reseñas, 1 época) para §9.
- El juez se calibra reportando su propio acierto sobre reseñas reales de test.

## 4. Estructura del notebook

Archivo único: `notebooks/miniproyecto4_restmex_gpt.ipynb`

### Sección 0 — Portada y problema (propias)

### Secciones 1–4.3 — Bloque heredado del Miniproyecto 1

Celdas 3–70 del notebook de MP1 **sin modificar** (entorno, corpus, EDA, protocolo, `evaluar`,
baselines), con celdas puente antes y después, verificación de identidad y assert de baselines.

### Secciones 4.4–4.8 — Protocolo propio

| # | Contenido |
|---|---|
| 4.4 | Dependencias, `CHECKPOINT`, `CFG_GPT`, `CFG_EST` |
| 4.5 | Tokenizador BPE byte-level, fertilidad, `MAX_LEN_GPT` (P95 con prefijo), token `<|pad|>` |
| 4.6 | Formato de entrenamiento con prefijo de control, conjuntos |
| 4.7 | Jueces: TF-IDF + LogReg para estrellas y para tipo, entrenados con train real; su acierto en test real como referencia |
| 4.8 | Funciones de métricas: perplejidad, distinct-n, rep-4, control |

### Sección 5 — El modelo base, sin ajustar

Lo que hace el guía (distribución del siguiente token), sobre nuestro dominio. Perplejidad en
test de **tres GPT-2 en español** preentrenados en corpus distintos (`mrm8488/spanish-gpt2`,
el del guía; `DeepESP/gpt2-spanish`, libros; `datificate/gpt2-small-spanish`, Wikipedia).
Generación con *prompt* de control: ¿obedece sin ajuste?

### Sección 6 — Técnica 1: *fine-tuning* sin condición

GPT-2 ajustado sobre las reseñas tal cual (como el guía con chistes). Curvas, perplejidad,
muestras.

### Sección 7 — Técnica 2: *fine-tuning* condicional

Mismo modelo y datos, con prefijo de control. Perplejidad (comparable con §6: se mide solo
sobre la reseña), control por estrella (matriz de confusión juez × estrella pedida) y por tipo.

### Sección 8 — Estrategias de decodificación

1. **La función `generate` del guía, corregida** (§D-405): con caché de claves/valores, sin
   decodificar token a token, semántica de `eps` explícita. Verificación: con `eps = 1` produce
   exactamente los mismos tokens que la decodificación codiciosa de Hugging Face. Tiempo con y
   sin caché frente a la longitud.
2. Comparación de codiciosa, codiciosa + `no_repeat_ngram`, *beam search*, muestreo con
   temperatura, top-k, top-p y ε-codiciosa del guía, en diversidad, repetición, fluidez y
   control. Gráfica de compromiso diversidad–control.

### Sección 9 — Estudios con `CFG_EST`: LoRA y otro checkpoint

*Fine-tuning* completo vs. LoRA (`c_attn`, r=8) vs. `DeepESP/gpt2-spanish` completo, mismas
8.000 reseñas: perplejidad, control, % de parámetros entrenables, tiempo.

### Sección 10 — El generador como clasificador (aporte propio)

Para cada reseña de test, la perplejidad del texto bajo cada uno de los 5 prefijos; se
predice la estrella que mejor lo explica (clasificación generativa, regla de Bayes con prior
uniforme). Se evalúa con la `evaluar` heredada y se sitúa en la tabla de MP1.

### Sección 11 — ¿Memoriza?

Fracción de 8-gramas generados que aparecen en train y similitud TF-IDF máxima con el train,
frente a la misma medida en reseñas reales de test (que tampoco se vieron).

### Sección 12 — Aumento de datos para la clasificación (aporte propio, H3)

TF-IDF + LogReg (modelo 1 de MP1) entrenado con: (a) train real, (b) + sobremuestreo de
1★–3★, (c) + reseñas sintéticas de 1★–3★ del generador condicional. Mismo test de MP1.

### Sección 13 — Demo

`generar_resena(estrellas, tipo, inicio)`, con pares fijos (misma apertura a 1★ y a 5★; hotel
frente a restaurante) y un widget opcional.

### Sección 14 — Análisis cualitativo y modos de falla

Muestras por estrella con la predicción del juez; conteo de fallas: repetición, texto
truncado o vacío, polaridad cruzada, «alucinaciones» de lugares.

### Sección 15 — Conclusiones y limitaciones

## 5. Criterios de aceptación

- [ ] «Restart & Run All» sin errores en ≤ 45 min con T4.
- [ ] Bloque heredado idéntico a MP1 y baselines reproducidos.
- [ ] Resultados de entrenamiento (curvas, perplejidad) y ejemplos de generación en §6 y §7.
- [ ] Ningún defecto del guía replicado (§D-403, §D-405).
- [ ] Toda gráfica/tabla con su lectura; conclusiones responden H1–H3.

## 6. Mapa rúbrica → notebook

| Criterio | Pts | Dónde |
|---|---:|---|
| Notebook completo | 1 | Narrativa en cada celda, cierre por sección, §15 |
| Reproducibilidad | 2 | Bloque heredado, semilla antes de cada entrenamiento y generación, `CFG_GPT`/`CFG_EST`, degradación en CPU |
| Modelo de generación (entrenamiento + ejemplos + conceptos) | 2 | §5–§8: curvas, perplejidad, muestras, decodificación explicada |
| Innovación | 2 | Control por prefijo + juez (§7), `generate` corregido y medido (§8), LoRA y checkpoints (§9), generador como clasificador (§10), memorización (§11), aumento de datos (§12), demo (§13) |

## 7. Fuera de alcance

- Modelos > 400 M parámetros, LLMs por API, RLHF.
- Evaluación humana formal (§14 es cualitativa, declarada como tal).
