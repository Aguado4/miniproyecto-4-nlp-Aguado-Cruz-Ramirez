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

Igual que MP3 §D-302: celdas 3–65 de MP1 sin modificar. Aquí además el Split A importa por dos
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

## D-407 · Hiperparámetros de ajuste: lr 5e-5, sin checkpoints en disco

**Estado:** aceptada · 2026-09-19 · **revisada** 2026-09-26 (número de épocas: ver D-411)

El guía usa 2e-5 por 10 épocas sobre ~2.400 chistes. Aquí se usa 5e-5 con *warmup*, la tasa
estándar de ajuste de GPT-2, que da más avance por paso. El número de épocas se fijó después con
medición, no por criterio: **D-411**. Mejor modelo por pérdida de validación, evaluada por pasos
y no por época (con pocas épocas, «por época» daría un único punto de curva); directorio
temporal borrado tras cada corrida.

---

## D-408 · Todos los checkpoints se cargan en fp32

**Estado:** aceptada · 2026-09-26

**Contexto.** `DeepESP/gpt2-spanish` publica sus pesos en **fp16**, y `transformers` 5.x respeta
el dtype del checkpoint al cargar (antes convertía a fp32 por omisión). Ajustar un modelo cuyos
pesos son fp16 con Adam hace que la pérdida explote y se vaya a `nan` en tres pasos: los
gradientes pequeños se redondean a cero y los grandes desbordan, sin pesos maestros en fp32 que
lo absorban. En la prueba de humo esto salía como pérdida 196 → 1.075 → `nan` y, acto seguido,
un *device-side assert* de CUDA al generar con un modelo lleno de `nan`.

**Decisión.** Una sola función `cargar_modelo(ckpt, tok)` (§4.5) que carga siempre con
`dtype=torch.float32` y redimensiona los embeddings para `<|pad|>`. La media precisión se
aplica donde corresponde, en el `Trainer` con `fp16=True`, que sí mantiene una copia maestra de
los pesos en fp32 y escala la pérdida.

**Alternativas.** (a) Excluir DeepESP de §9: perdería la comparación entre corpus de
preentrenamiento, que es parte del aporte. (b) Bajar la tasa de aprendizaje para ese
checkpoint: no arregla la causa y rompería la comparación, que exige el mismo ajuste.

**Consecuencias.** Las tres variantes de §9 reciben exactamente el mismo tratamiento, así que la
comparación entre checkpoints es válida. Cargar en fp32 sube algo el uso de memoria, sin efecto
en una RTX 4060 de 8 GB con `batch` 16 y `MAX_LEN_GPT` ≤ 160.

---

## D-409 · Resumen de resultados al principio del notebook

**Estado:** aceptada · 2026-09-26

**Contexto.** La retroalimentación del Miniproyecto 1 (7,0/7,0) señaló como único punto de mejora
que «el notebook es bastante extenso; algunas secciones podrían sintetizarse para destacar los
experimentos principales».

**Decisión.** No recortar experimentos, sino añadir en §0 una tabla de **resultados principales
con enlace a la sección** que los produce, para que se pueda leer el hallazgo antes de recorrer
el detalle. El bloque heredado se declara como tal desde el índice, de modo que quede claro qué
es nuevo en esta entrega.

**Alternativas.** Eliminar secciones: se descartó porque la innovación vale 2 de 7 puntos y cada
sección responde a una pregunta distinta de la rúbrica.

**Consecuencias.** El notebook sigue siendo largo, pero es navegable de arriba abajo sin leerlo
completo, y el lector ve primero las cifras que responden H1, H2 y H3.

---

## D-410 · Liberar el optimizador entre secciones

**Estado:** aceptada · 2026-09-26

**Contexto.** En la primera corrida de referencia, §7 (ajuste condicional) tardó **75 minutos**
frente a los **15** de §6 (ajuste sin condición), con el mismo modelo, los mismos datos y los
mismos hiperparámetros. La causa no era el cálculo: al llegar a §7 seguían vivos el modelo de §6
y, sobre todo, su `Trainer`, y un GPT-2 de 124 M arrastra **1 GB de estados de Adam** además de
sus 0,5 GB de pesos. Con 8 GB de VRAM eso deja a §7 en presión de memoria, donde el asignador de
PyTorch recicla bloques continuamente. No falla: simplemente se vuelve varias veces más lento,
que es la forma más difícil de detectar este problema.

**Decisión.** Una función `liberar(...)` en §6 que suelta modelos y, explícitamente,
`optimizer` y `lr_scheduler` de los `Trainer` que ya rindieron sus números. Se llama al terminar
§6 (el modelo sin condición ya dio perplejidad y muestras) y al terminar §7, donde se conserva
`m_cond` porque lo usan §8 a §14 pero no su optimizador. La celda imprime la memoria en uso tras
liberar, para que el efecto sea verificable y no un acto de fe.

**Alternativas.** (a) Bajar el `batch` a 8: haría comparables entre sí las secciones pero no con
MP1–MP3 en tiempo, y trata el síntoma. (b) Reejecutar el notebook por partes: incompatible con
«Restart & Run All» (`CLAUDE.md` §3.1).

**Consecuencias.** §6 y §7 trabajan con la misma memoria disponible, así que sus tiempos son
comparables entre sí, que es justo lo que la tabla de costo computacional afirma. En una T4 de
16 GB el problema no se habría manifestado, lo que explica por qué el presupuesto de `PLAN.md`
lo pasaba por alto.

---

## D-411 · Presupuesto: tres épocas sobre 10.000 reseñas, no una sobre 32.000

**Estado:** aceptada · 2026-09-26

**Contexto.** La primera corrida de referencia tardaba cerca de dos horas, y el entregable tiene
que poder reproducirlo quien lo evalúa. Había que recortar, pero el reparto entre «cuántas
reseñas» y «cuántas pasadas» no es indiferente: cambia qué aprende el modelo, no solo cuánto
tarda.

**Contexto medido.** En vez de elegir a ojo, se midió el control por prefijo en cuatro
configuraciones, con 10.000 reseñas y 60 peticiones por medición (fuera del notebook, para no
gastar presupuesto de la corrida):

| Datos | Decodificación | 1 época | 3 épocas |
|---|---|---:|---:|
| proporción natural | top-p 0,92 | 0,233 | **0,300** |
| proporción natural | codiciosa + `no_repeat` | 0,150 | **0,233** |
| balanceado por estrella | top-p 0,92 | 0,250 | 0,250 |
| balanceado por estrella | codiciosa + `no_repeat` | 0,117 | **0,383** |

El MAE de la estrella cuenta la misma historia con más claridad: de 1,60 a 1,25 en el caso
natural, y de 1,67 a **0,72** en el balanceado con codiciosa. **Las épocas son la palanca.**

**Decisión.** `CFG_GPT`: 3 épocas sobre 10.000 reseñas (6,7 min por ajuste, medido), y
`CFG_EST`: 3 épocas sobre 3.000 para los estudios de §9, de modo que compartan el régimen de
entrenamiento de las corridas principales y sean comparables con ellas. El notebook completo
queda en torno a 30 minutos.

**Alternativas.** (a) Una época sobre las 32.000: el mismo coste, pero **peor control**, que es
justo lo que la entrega quiere mostrar. (b) Dos épocas sobre 32.000, la configuración original:
mejores números, pero dos horas de corrida y nadie la reproduce.

**Consecuencias.** La perplejidad es peor que con el presupuesto grande (la corrida de una época
sobre 16.000 daba 40,7 frente a 32,9 con dos épocas sobre 32.000), y eso se declara en
`EXPERIMENTS.md`: las cifras describen un generador con presupuesto de clase, no el mejor
generador posible sobre este corpus. La comparación entre §6 y §7 sigue siendo válida porque
ambos reciben exactamente el mismo tratamiento.

---

## D-412 · El control se mide con dos decodificaciones, y el balanceo entra como estudio

**Estado:** aceptada · 2026-09-26

**Contexto.** Al ver que el control de estrellas era bajo, la primera hipótesis fue la escasez de
1★ y 2★ (2,6 % del corpus cada una): con 10.000 reseñas el modelo ve unas 220 de 1★. El
experimento de D-411 **la refuta**: balancear las clases no mejora el control con una época
(0,250 frente a 0,233) y solo ayuda combinado con más épocas y decodificación codiciosa.

**Decisión.** Dos consecuencias para el notebook. Primera: §7 mide el control con **dos
decodificaciones** (top-p 0,92 y codiciosa + `no_repeat`), porque el control no es una propiedad
solo del modelo sino de la pareja modelo–decodificación, y presentar una sola cifra elegida a
conveniencia sería engañoso. Segunda: la condición balanceada entra como **cuarta variante de
§9**, con `idx_balanceado`, en lugar de cambiar el entrenamiento de §7; así §6 y §7 conservan la
proporción real del corpus y siguen siendo comparables entre sí, y el lector ve la pregunta
respondida con una fila de tabla.

**Alternativas.** Entrenar §7 con clases balanceadas: mejoraría un poco el titular a cambio de
romper la comparación con §6 y de esconder el hallazgo, que es que el balanceo no es la palanca.

**Consecuencias.** §9 pasa de tres a cuatro variantes (unos 2 min más). El veredicto de H2 es
matizado y no binario: el tipo de establecimiento se controla bien, la polaridad ordinal solo en
parte, y cuánto depende de cómo se decodifique.

---

## Plantilla para nuevas entradas

## D-4NN · Título breve

**Estado:** propuesta · AAAA-MM-DD

**Contexto.** …  **Decisión.** …  **Alternativas.** …  **Consecuencias.** …
