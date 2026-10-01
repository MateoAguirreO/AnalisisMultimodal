# Inventario de modelos y protocolo de selección del multimodal final

Fecha: 2026-09-28. Fuentes: repo `Psiquiatria` (código, CSV y el documento `Recopilacion_Experimentos_CRISP-DM.md`) y `pipeline_multimodal_xai/resultados/`. Contexto de los errores: `auditoria_psiquiatria/AUDITORIA.md`.

## 1. Inventario

"AUC reportado" es la cifra tal cual aparece en la fuente. Ningún número de esta tabla es comparable directamente con otro, porque cada uno viene de un protocolo distinto.

### Depresión

| Modelo | Modalidad | AUC reportado | Problemas | ¿Desplegable? |
|---|---|---|---|---|
| CNN2D log-mel 4s | voz (espectrograma) | 0.576 | Grid de 16 combinaciones | No (sin pesos) |
| BernoulliNB, features artesanales | voz | 0.634 | Ajustado sobre los mismos datos | No |
| ET sobre wav2vec, por segmento (GroupKFold) | voz (SSL) | **0.580** | El menos sesgado del track SSL | No (pooling aleatorio, E6) |
| ET sobre wav2vec, por paciente | voz (SSL) | 0.679 | Grid de 8640 combinaciones + 6 modelos SSL (E5) | No (E6) |
| BiLSTM sobre wav2vec | voz (SSL) | 0.766 → 0.709 (fix) → 0.669 (v4) | Mejor de 34 configuraciones (E5) | No (E6) |
| Destilación (estudiante eGeMAPS) | voz | 0.648 | Fugas E3 y E4 (limpio ≈ 0.58); mejor de 406 | Sí, pero hereda la fuga |
| ET sobre AU (v4) | rostro | 0.612 | `n_frames_detected` (E7) | Sí |
| Fusión voting (matriz v2, con BiGRU) | voz + rostro | 0.677 | Ramas con E5 | No |
| **Nuestro: eGeMAPS `l1logreg`** | voz | 0.593 | Selección leve: screening ⊂ confirmación | **Sí** (en plataforma) |
| **Nuestro: AU `anova_rf`** | rostro | 0.638 | Ídem | **Sí** |
| **Nuestro: soft vote** | voz + rostro | 0.674 (perm. p = 0.040) | Ídem; la permutación no repite la selección | **Sí** |

### Ansiedad

| Modelo | Modalidad | AUC reportado | Problemas | ¿Desplegable? |
|---|---|---|---|---|
| AST gammatone 3s (fase3) | voz (espectrograma) | 0.564 | Ranking invertido 3 veces | No |
| NearestCentroid, features artesanales | voz | 0.418 | — | No |
| MLP sobre wav2vec2-large-robust + mixup | voz (SSL) | 0.668 | 6 SSL × 6 experimentos (E5); umbral en val (afecta F1, no AUC) | No (E6) |
| CNN1D sobre xlsr-300m | voz (SSL) | 0.614–0.621 | E5 | No (E6) |
| Destilación (`Kd_bilstm_lazypredict`) | voz | 0.61 (máximo de 406) | Profesor 0.502 (E2) | — |
| Destilación (`KD_BiLSTM_XGBoost`) | voz | Profesor 0.284 en test; estudiantes 0.500 | E2 | — |
| XGB eGeMAPS base (holdout de 16) | voz | 0.651 (IC 0.54–0.75) | Un solo holdout de 16 pacientes | Sí |
| **Nuestro: eGeMAPS `anova_xgb`** | voz | 0.572 | Selección leve | **Sí** |
| **Nuestro: AU `anova_logreg`** | rostro | 0.661 | Ídem | **Sí** |
| **Nuestro: soft vote** | voz + rostro | 0.668 (perm. p = 0.045) | Ídem | **Sí** |

### Lectura del inventario

- Donde las cifras son comparables, **la representación SSL (wav2vec) es la mejor señal de voz**, en ambos ejes (~0.67 incluso con sesgos). eGeMAPS rinde 0.57–0.59.
- Nada de voz SSL es desplegable tal como está (E6). Hay que re-extraerlo.
- Espectrogramas, features artesanales y destilación quedan fuera: rinden menos o dependen de un profesor roto o con fuga.

## 2. Hallazgo nuevo: desajuste de audio entre entrenamiento y plataforma

`features_*_egemaps.csv` se calculó sobre `NNN_editado.wav`, audio **editado a mano** en el servidor (`batch_copy_preprocess_audio.py`). Comparado con el `.wav` crudo de cada participante:
- Duración retenida: mediana **67%**, rango 32–99%.
- **29 de 79 participantes conservan menos del 60%**.
- Lo más probable es que la edición haya quitado la voz del entrevistador. No es reproducible automáticamente.

La plataforma procesa el `.wav` **crudo completo**. La rama de voz desplegada se entrenó con un audio distinto del que recibe en inferencia. La validación con el pid 61 no lo detectó: es de los menos editados (85%), y r = 1.0 sobre 88 features de escalas muy distintas es una prueba débil.

Por eso la rama de embeddings se extrae del **audio crudo + Demucs**, igual que lo hará la plataforma (`preparar_audio_crudo.py`).

## 3. Lista corta

| Rama | Representación | Configuraciones candidatas (elegidas en CV interna) |
|---|---|---|
| **V** rostro | 66 AU/pose/emoción (`dataset_au_features.csv`, sin `n_frames_detected`) | Las 5 de `fusion.VIDEO_CONFIGS` |
| **A** voz eGeMAPS | 88 eGeMAPS por segmento (audio editado, tal como está entrenada hoy) | Las 4 de `audio_branch.AUDIO_CONFIGS` |
| **E** voz wav2vec | wav2vec2-large-robust sobre audio crudo + Demucs | `e_logreg`: escalado → {sin PCA, PCA 16, PCA 32} → logreg L2 balanceada, C ∈ {0.001, 0.01, 0.1}. `e_et`: ExtraTrees balanceado (500 árboles, `min_samples_leaf=3`) |

Representación E, **fijada de antemano** para no abrir otro barrido:
- Segmentos de 5 s con 50% de solape, igual que eGeMAPS.
- En cada segmento, mean pooling sobre frames en cada capa.
- Promedio de las 24 capas transformer y luego promedio sobre segmentos → 1024 dimensiones por participante.
- Se guardan todas las capas, pero el análisis por capa es **solo exploratorio**.

## 4. Protocolo de evaluación (declarado antes de ver resultados)

- **CV totalmente anidada**, con una semilla externa nueva que nunca se usó para elegir nada: `RepeatedStratifiedKFold(5, 5, random_state=2026)`.
  - En cada fold externo, cada rama elige su configuración y sus hiperparámetros **solo con el train externo** (CV interna de 5).
  - Luego predice el test externo.
- **Fusiones:** soft vote (promedio de probabilidades, sin parámetros) de `A+V` (el actual), `E+V`, `A+E+V` y `A+E`.
- Cada eje se evalúa **por separado**. No hay modelado conjunto ansiedad–depresión.
- **Regla de decisión, por eje:**
  1. La comparación principal es `E+V` o `A+E+V` contra `A+V`.
  2. Se adopta la rama E solo si la fusión que la incluye supera a `A+V` en AUC medio y en la mayoría de las 5 repeticiones, y pasa un test de permutación que **repite toda la selección** anidada.
  3. Si no, se queda `A+V`.
- Las demás combinaciones y el análisis por capa se reportan como exploratorios.
- **Exploratorio de fidelidad a la plataforma.** La rama A se evalúa en 3 variantes:
  - `A_ed`: entrena y prueba con audio editado. Es la cifra reportada hasta hoy.
  - `A_ed→crudo`: entrena con audio editado y prueba con eGeMAPS del audio crudo. Es lo que hace la plataforma hoy.
  - `A_crudo`: entrena y prueba con audio crudo. Sería la rama reentrenada de forma consistente.

  Las eGeMAPS del audio crudo salen de la misma función de la plataforma (`egemaps_audio_crudo.py`).

## 5. Ronda 2: audio editado (declarada el 2026-09-29, antes de ver resultados)

**Contexto:**
- La ronda 1 mostró que la voz del audio crudo sigue la sesión de grabación y que solo el rostro sobrevive al AUC intra-día (`RESULTADOS.md` §3).
- Mateo trajo del servidor los 79 `NNN_recortado_denoised.wav`: editados a mano (sin entrevistador), recortados y con Demucs.
- **Validación:** las eGeMAPS recalculadas con la función de la plataforma coinciden con `features_depresion_egemaps.csv`: mismo número de segmentos y diferencia relativa p95 = 0.0001 en los pids 1, 3, 16 y 61.

**Rama nueva E_ed:** misma representación wav2vec fijada en §3 (mean pooling, promedio de capas 1–24, media sobre segmentos), sobre el audio editado. Mismas configuraciones candidatas que E.

**Métrica principal: AUC intra-día.** Solo cuenta pares positivo/negativo grabados el mismo día. Los días salen de `sesiones_grabacion.py`: 7 días, en dos campañas. El AUC agrupado se reporta como secundario.

**Referencia:** V, el rostro solo, que es la recomendación vigente tras la ronda 1.

**Regla de decisión, por eje:**
1. La voz vuelve (multimodal) solo si `E_ed+V` o `A_ed+E_ed+V` supera a V en AUC intra-día medio **y** en al menos 3 de las 5 repeticiones.
2. Si ambas lo cumplen, se toma la de mayor AUC intra-día medio. Es una selección entre 2 y se declara como tal.
3. La fusión elegida debe además pasar el test de permutación que repite toda la selección anidada.
4. Si ninguna cumple, se mantiene el rostro solo.

Se reportan como exploratorios: E_ed solo (¿tiene señal la voz del participante por sí misma?), A_ed+E_ed y la comparación E_ed vs E (audio editado vs crudo).

## 6. Ronda 4: búsqueda del mejor modelo (declarada el 2026-09-30, antes de ver resultados)

- **Métrica:** AUC estándar sobre todas las muestras juntas, por decisión de Mateo. Nada de AUC intra-día.
- **Búsqueda honesta** (`buscar_mejor_modelo.py`). Compiten las ramas y todas sus combinaciones por soft vote. El candidato **AUTO** elige en cada fold externo el de mayor AUC OOF dentro del train externo. **La cifra reportable como "mejor modelo" es la de AUTO**; el máximo de la tabla es optimista por construcción.
- **Rama nueva RD, micro-ventanas densas** (`extraer_microventanas_densas.py`). Cubre todo el audio editado con trozos contiguos. La duración del trozo, entre {0.1, 0.19, 0.5, 1.0} s, se elige en la CV interna como hiperparámetro; última capa; media por participante.
- **Ramas en competencia en depresión:** R, RD y V. VT sale: 0.544 en la corrida del 2026-09-30.
- **Ansiedad:** solo rostro (V, VT), n = 80.
