# PLAN - Ejecución por fases

> Estado global: **código completo, sin corrida de referencia.** Falta: corrida en T4,
> lecturas `<!-- LEER -->`, conclusiones §15.

Leyenda: `[ ]` pendiente · `[~]` código listo, falta ejecutar y leer · `[x]` hecho

---

## Fase 0 - Scaffolding

- [x] Revisar consigna, rúbrica y `Sesion4/1-text-generation.ipynb`
- [x] Docs, README, requirements, `.gitignore`, carpetas
- [x] Copiar §1–§4.3 de MP1 sin modificar (celdas 3–70) + verificación
- [x] Repositorio remoto y colaboradores

## Fase 1 - Heredado y protocolo (SPEC §1–§4.8)

- [ ] Bloque heredado corre y los baselines reproducen MP1 (**bloqueante**)
- [~] 4.4 Config · 4.5 Tokenizador · 4.6 Formato · 4.7 Jueces · 4.8 Métricas

## Fase 2 - Generación (SPEC §5–§8)

- [~] 5 Modelo base y tres checkpoints · 6 FT sin condición · 7 FT condicional · 8 Decodificación

## Fase 3 - Estudios y usos (SPEC §9–§12)

- [~] 9 LoRA y checkpoint · 10 Clasificador generativo · 11 Memorización · 12 Aumento de datos

## Fase 4 - Cierre (SPEC §13–§15)

- [~] 13 Demo · 14 Análisis cualitativo
- [ ] 15 Conclusiones; quitar `<!-- LEER -->`; Restart & Run All en T4; `EXPERIMENTS.md`

## Presupuesto de tiempo (T4, estimado — verificar)

| Bloque | min |
|---|---:|
| Heredado (entorno, EDA con spaCy, protocolo) | 6–8 |
| §4.4–4.8 (tokenización, jueces) | 1–2 |
| §5 perplejidad de 3 checkpoints + muestras | 2–3 |
| §6 FT sin condición (32k × 2 épocas) | 7–9 |
| §7 FT condicional (32k × 2 épocas) + control | 8–10 |
| §8 decodificación (8 estrategias × 100 muestras) | 3–4 |
| §9 tres corridas `CFG_EST` | 4–5 |
| §10 clasificador generativo (5 × 1.000 reseñas) | 1 |
| §11–§12 (3.000 reseñas generadas + 3 TF-IDF) | 4–5 |
| §13–§14 | 1 |
| **Total** | **~37–48** |
