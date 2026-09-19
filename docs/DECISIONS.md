# DECISIONS - Registro de decisiones de diseño

Formato ADR abreviado. Estados: `aceptada` · `propuesta` · `revisada` · `revertida`

> Numeración desde **D-401**. Las de entregas anteriores se citan como `MP1 §D-00N`,
> `MP2 §D-2NN`, `MP3 §D-3NN`.

---

## D-401 · Generación condicionada de reseñas, no texto libre

**Estado:** aceptada · 2026-09-19

**Contexto.** El guía ajusta GPT-2 sobre chistes y evalúa a ojo. Nuestro corpus trae estrellas
y tipo de establecimiento.

**Decisión.** Generar reseñas condicionadas a (estrellas, tipo). La etiqueta convierte la
evaluación subjetiva en medible (juez, perplejidad condicional) y abre dos usos: clasificación
generativa (§10) y aumento de datos (§12).

---

## D-402 · Reutilizar corpus y Secciones 1–4.3 de MP1

**Estado:** aceptada · 2026-09-19

Igual que MP3 §D-302: celdas 3–70 de MP1 sin modificar. Aquí además el Split A importa por dos
razones: el test de perplejidad no se ve en entrenamiento, y el experimento de aumento (§12) se
mide sobre el mismo test que dio 0,5235 a TF-IDF en MP1.

---

## D-403 · Token de relleno propio, distinto de EOS

**Estado:** aceptada · 2026-09-19

**Contexto.** El guía hace `pad_token = eos_token`. `DataCollatorForLanguageModeling` pone la
etiqueta `-100` en todo lo que sea relleno; si relleno y fin de texto son el mismo id, **el
modelo nunca aprende a emitir el fin de texto** y genera hasta agotar `max_length`.

**Decisión.** Añadir `<|pad|>` como token de relleno y redimensionar los embeddings. Cada
reseña de entrenamiento termina en EOS, que sí entra en la pérdida.

---

## D-404 · Prefijo de control en texto plano, no tokens especiales

**Estado:** aceptada · 2026-09-19

**Alternativas.** Tokens especiales `<|1★|>`… (práctica tipo CTRL, Keskar et al. 2019).

**Decisión.** Prefijo `Reseña de <tipo> con <n> estrella(s):\n`. Razones: (1) el modelo base
lo puede leer sin ajuste, así que §5 mide un *prompting* justo; (2) con LoRA (§9) unos tokens
nuevos quedarían con embeddings aleatorios congelados; (3) no cambia el vocabulario entre
técnicas. El prefijo se excluye del cálculo de perplejidad para que §6 (sin prefijo) y §7 sean
comparables.

---

## D-405 · Corregir la función `generate` del guía

**Estado:** aceptada · 2026-09-19

| Aspecto del guía | Problema | Corrección |
|---|---|---|
| Recalcula el modelo sobre toda la secuencia en cada paso | Costo cuadrático en la longitud | Caché de claves/valores (`past_key_values`) |
| `tokenizer.decode` de cada token y concatenación | Rompe caracteres multibyte (tildes, ñ, emojis) en BPE byte-level | Decodificar la secuencia completa al final |
| `eps` es la probabilidad de **explotar** | En RL, ε suele ser la de **explorar**: se presta a confusión | Se conserva la semántica del guía y se documenta |
| `np.random` sin semilla | Irreproducible | Generador `torch.Generator` con semilla |

Se verifica que con `eps = 1` coincide token a token con `model.generate(do_sample=False)` y se
mide el tiempo con y sin caché.

---

## D-406 · Jueces TF-IDF + LogReg entrenados con reseñas reales

**Estado:** aceptada · 2026-09-19

**Contexto.** Hace falta medir si lo generado tiene la polaridad pedida. Un juez BETO (MP3)
sería más preciso pero costaría otro *fine-tuning* dentro del presupuesto.

**Decisión.** El modelo 1 de MP1 (macro-F1 0,52), entrenado solo con train real, y otro igual
para `Type`. Su acierto sobre reseñas **reales** de test es la referencia: no se le puede pedir
a lo generado más control del que el juez detecta en lo real.

---

## D-407 · Hiperparámetros de ajuste: lr 5e-5, 2 épocas, sin checkpoints en disco

**Estado:** aceptada · 2026-09-19

El guía usa 2e-5 por 10 épocas sobre ~2.400 chistes. Con 32.000 reseñas, 2 épocas a 5e-5 con
*warmup* (la tasa estándar de ajuste de GPT-2) dan más pasos en menos tiempo. Mejor modelo por
pérdida de validación; directorio temporal borrado tras cada corrida.

---

## Plantilla para nuevas entradas

## D-4NN · Título breve

**Estado:** propuesta · AAAA-MM-DD

**Contexto.** …  **Decisión.** …  **Alternativas.** …  **Consecuencias.** …
