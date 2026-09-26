# PLAN - Ejecución por fases

> Estado global: **corrida de referencia completa** (local, RTX 4060, ~30 min, 0 errores).
> Números en `EXPERIMENTS.md`; decisiones de presupuesto en D-411.

Leyenda: `[ ]` pendiente · `[~]` código listo, falta ejecutar y leer · `[x]` hecho

---

## Fase 0 - Scaffolding

- [x] Revisar consigna, rúbrica y `Sesion4/1-text-generation.ipynb`
- [x] Docs, README, requirements, `.gitignore`, carpetas
- [x] Copiar §1–§4.3 de MP1 sin modificar (celdas 3–65) + verificación
- [x] Repositorio remoto y colaboradores

## Fase 1 - Heredado y protocolo (SPEC §1–§4.8)

- [x] Bloque heredado corre y los baselines reproducen MP1 (**bloqueante**): 63 celdas
  byte-idénticas, juez de estrellas en el 0.5235 exacto de MP1
- [x] 4.4 Config · 4.5 Tokenizador · 4.6 Formato · 4.7 Jueces · 4.8 Métricas

## Fase 2 - Generación (SPEC §5–§8)

- [x] 5 Modelo base y tres checkpoints · 6 FT sin condición · 7 FT condicional · 8 Decodificación

## Fase 3 - Estudios y usos (SPEC §9–§12)

- [x] 9 LoRA, checkpoint y condición balanceada · 10 Clasificador generativo · 11 Memorización
  · 12 Aumento de datos (cuatro variantes, con sintéticas filtradas por el juez)

## Fase 4 - Cierre (SPEC §13–§15)

- [x] 13 Demo · 14 Análisis cualitativo
- [x] 15 Conclusiones; sin `<!-- LEER -->`; Restart & Run All; `EXPERIMENTS.md`
- [ ] Confirmar con el profesor la reutilización del corpus y del bloque heredado (D-402)

## Presupuesto de tiempo (medido en RTX 4060 local)

Reparto elegido con medición, no a ojo: ver D-411. Los tiempos de una T4 serán algo mayores.

| Bloque | min |
|---|---:|
| Heredado (entorno, EDA con spaCy, protocolo) | ~2 |
| §4.4–4.8 (tokenización, jueces) | <1 |
| §5 perplejidad de 3 checkpoints + *prompting* | ~1 |
| §6 FT sin condición (10k × 3 épocas) | ~7 |
| §7 FT condicional (10k × 3 épocas) + control × 2 decodificaciones | ~7 |
| §8 decodificación (8 estrategias × 60 muestras) | ~3 |
| §9 cuatro corridas `CFG_EST` (3k × 3 épocas) | ~9 |
| §10 clasificador generativo (5 prefijos × 600 reseñas) | ~1 |
| §11–§12 (1.200 sintéticas + 4 TF-IDF) | ~3 |
| §13–§14 | <1 |
| **Total** | **~30** |
