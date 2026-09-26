# CLAUDE.md — Instrucciones de trabajo para este repositorio

Fuente de verdad operativa para cualquier sesión de agente. Léelo completo antes de tocar nada.

---

## 1. Qué es este repositorio

Entregable del **Miniproyecto 4** del curso de NLP (Maestría, Universidad Icesi): **un único
Jupyter Notebook** que ajusta un **GPT-2 en español** para **generar reseñas turísticas
condicionadas a estrellas y tipo de establecimiento**, mide la generación (perplejidad, control
de polaridad con un juez, diversidad, memorización) y la usa como clasificador generativo y como
aumento de datos para el clasificador de MP1.

Guía: `icesi-nlp/Sesion4/1-text-generation.ipynb` (`mrm8488/spanish-gpt2` + chistes). **No se
usa su dataset**: se trabaja sobre `vg055/Rest-Mex2025`.

`consigna.txt` y `rubrica.txt` están en la raíz y **no se modifican** (7 puntos, 4 criterios).

### Relación con las entregas anteriores

- **Secciones 1–4.3 heredadas del Miniproyecto 1 SIN MODIFICAR** (celdas 3–65), con celdas
  puente antes y después. **No editar esas celdas** (tag `heredado-mp1`; una celda verifica la
  identidad). Lo propio empieza en §4.4 y usa `CFG_GPT` / `MAX_LEN_GPT`; nunca reasignar `CFG`,
  `MAX_LEN`, `evaluar`, `df`, `ds`, `conteo` ni los splits heredados (§D-402).
- El test de Split A es el mismo de MP1–MP3: el aumento de datos (§12) y el clasificador
  generativo (§10) entran en la misma tabla.

## 2. Documentos (leer en este orden)

`docs/SPEC.md` (contrato) · `docs/PLAN.md` · `docs/DATASET.md` · `docs/DECISIONS.md` (desde
D-401) · `docs/EXPERIMENTS.md` (solo números ejecutados).

## 3. Reglas duras

### 3.1 Reproducibilidad — 2 de 7 puntos

- «Restart & Run All» limpio. `SEED = 42`; **fijar semilla antes de cada entrenamiento y de
  cada bloque de generación** (el muestreo es aleatorio).
- Sin rutas locales; checkpoints desde HuggingFace Hub. Sin checkpoints en disco.
- GPU/CPU detectados; en CPU se degrada (`n_train_max`, menos muestras), nunca falla.
- Presupuesto ~30 min (medido en RTX 4060 local; ver `PLAN.md` y D-411). El notebook debe
  poder reproducirlo quien lo evalúa, así que el presupuesto manda sobre la métrica.

### 3.2 Narrativa — 1 punto

Todo en español; markdown antes de cada celda de código explicando el porqué; toda gráfica y
tabla con su lectura; **sin guiones largos en la narrativa** (convención de MP1).

### 3.3 Modelo de generación — 2 puntos

Mostrar **resultados de entrenamiento** (curvas, perplejidad) **y ejemplos de generación**,
con los conceptos bien usados (modelado causal, decodificación, perplejidad).

### 3.4 Innovación — 2 puntos

Control por prefijo + juez, `generate` corregido y medido, LoRA y checkpoints, condición
balanceada, generador como clasificador, memorización, aumento de datos con y sin filtro del
juez, demo (SPEC §6).

## 4. Convenciones técnicas

- Prefijo de control en texto plano (§D-404); perplejidad **solo sobre tokens de la reseña**.
- Relleno `<|pad|>` distinto de EOS (§D-403). Generación con `padding_side="left"`.
- Jueces TF-IDF + LogReg entrenados solo con train real (§D-406).
- La `evaluar(...)` heredada se usa tal cual donde aplica (§10, §12).

## 5. Qué NO hacer

- No usar el dataset de chistes ni datos del guía.
- No copiar los defectos del guía: `pad = eos`, `generate` sin caché y con `decode` por token,
  split sin semilla, evaluación solo a ojo.
- No añadir sintéticas a val/test. No inventar resultados. No hacer `git push` sin pedirlo.
- No dejar `TODO` ni `<!-- LEER -->` en el entregable final.
