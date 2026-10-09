# Análisis de `shapLimePorMuestra/`: ¿qué señal sostiene el título del artículo?

## 0. ACTUALIZACIÓN (2026-09-30): ambos ejes con el pipeline corregido. Esta sección reemplaza a las de abajo.

Fuente: `shap_matrix_<eje>.csv` y `lime_matrix_<eje>.csv` (80 participantes; entrena con 75 y prueba con 5; 5 repeticiones), resumidos en `shap_resumen_<eje>.csv` (`Psiquiatria/Multimodal_XAI/resumen_shap.py`).

**Regla:** solo se interpretan features de modelos que predicen (AUC > 0.55). Cada feature que se reporta cumple cuatro condiciones:
- dirección SHAP clara: correlación entre el valor de la feature y su SHAP, |ρ| ≥ 0.8;
- LIME coincide en el signo;
- está en el top-10 en las 5 repeticiones;
- el mismo sentido aparece en su AUC univariado.

### Depresión: rostro AU (AUC 0.605). Los demás modelos interpretables no predicen.

| Feature | Qué es | Dirección | AUC univariado | q (FDR) |
|---|---|---|---|---|
| `anger_std` | variabilidad de la expresión de enojo (py-feat) | **más → más riesgo** | 0.746 | **0.029** |
| `anger_mean` | expresión de enojo media | **más → más riesgo** | 0.753 | **0.029** |
| `AU01_mean` | elevador interno de la ceja | **menos → más riesgo** | 0.719 (invertido) | 0.067 |
| `AU05_std` | variabilidad del elevador del párpado superior | menos → más riesgo | 0.648 (invertido) | 0.60 |

Es el mismo patrón que en el análisis original (§3) y que en `pipeline_multimodal_xai`. La señal facial de depresión es estable entre pipelines.

### Ansiedad: rostro AU (AUC 0.659) y voz eGeMAPS (AUC 0.633)

| Feature | Qué es | Dirección | AUC univariado | q (FDR) |
|---|---|---|---|---|
| `AU15_std` | variabilidad del depresor de la comisura labial (comisuras hacia abajo) | **más → más riesgo** | 0.695 | 0.36 |
| `AU20_std` | variabilidad del estirador labial | **más → más riesgo** | 0.651 | 0.39 |
| `AU04_mean` | descenso de cejas (ceño) | **más → más riesgo** | 0.660 | 0.39 |
| `AU01_mean` | elevador interno de la ceja | **menos → más riesgo** | 0.697 (invertido) | 0.36 |
| `logRelF0-H1-H2` (voz) | diferencia entre los dos primeros armónicos; más baja = fonación más "apretada" o tensa | **menos → más riesgo** | 0.689 (invertido) | 0.49 |
| `hammarbergIndexV_stddevNorm` (voz) | variabilidad del balance espectral (índice de Hammarberg) | más → más riesgo | 0.683 | 0.49 |
| `mfcc2_stddevNorm` (voz) | variabilidad del 2.º coeficiente cepstral | menos → más riesgo | 0.693 (invertido) | 0.49 |

En ansiedad ninguna feature sobrevive a la corrección FDR por sí sola. El modelo las combina; el titular debe decir "se asocian", no "predicen".

### Titulares propuestos

**Para un artículo con ambos ejes:**
1. *Rostro y voz explicables ante el riesgo de ansiedad y depresión en jóvenes rurales: tensión perioral y fonación tensa en ansiedad; expresión de enojo y menor elevación de la ceja interna en depresión*
2. *Lo que el rostro y la voz revelan, y lo que no: marcadores explicables y evaluación honesta del riesgo de ansiedad y depresión en jóvenes de Samaná, Colombia*

**Por eje:**
- Depresión: *La expresión facial de enojo y la menor elevación de la ceja interna se asocian con el riesgo de depresión en jóvenes rurales: un análisis multimodal explicable*
- Ansiedad: *Tensión en la boca y en la voz: la variabilidad perioral (AU15, AU20) y una fonación más tensa se asocian con el riesgo de ansiedad*

**Hilo común:** la **menor elevación de la ceja interna (AU1)** aparece en ambos ejes. Es un candidato a marcador compartido del riesgo afectivo.

**Cautelas obligatorias en el texto:**
- la etiqueta es "DX IA", un tamizaje PHQ-9 / GAD-7 con corte en síntomas leves (≥ 5), no un diagnóstico clínico;
- el rendimiento es modesto: AUTO 0.56–0.58; mejor par 0.68–0.70;
- N = 80 con ~20 positivos;
- solo `anger_*` sobrevive a FDR.

---

Fecha: 2026-09-28. Análisis sobre los 4 CSV de la carpeta, sin reentrenar nada.

> **Actualización (mismo día):** estos CSV los generó `Depresion/Multimodal/build_oof_matrix_v4_balanceado.py` del repo `Psiquiatria`. La auditoría de ese código (`auditoria_psiquiatria/AUDITORIA.md`) encontró que la rama de voz explicada aquí, el estudiante de destilación, tiene una fuga medida de +0.078 de AUC. La lectura de la rama facial (§3) se mantiene, con la salvedad de `n_frames_detected`.

## 1. Qué contiene la carpeta

| Archivo | Contenido |
|---|---|
| `oof_matrix_detail_by_repeat.csv` | 790 filas = 79 participantes × 10 repeticiones. Probabilidad *out-of-fold* de 4 modelos por repetición. |
| `oof_matrix_final.csv` | Promedio de las 10 repeticiones por participante (79 filas). |
| `shap_matrix.csv` | SHAP por participante: 67 features faciales (`au__*`) + 88 eGeMAPS (`destilacion_egemaps__*`). |
| `lime_matrix.csv` | LIME por participante, solo las 67 features faciales. |

**Esquema de validación:** 10 repeticiones × 5 folds estratificados. Cada modelo se entrena con ~63 participantes y predice sobre ~16 que no vio (4–5 positivos por fold). En total son 50 modelos por rama, y cada participante recibe 10 predicciones fuera de muestra.

**Eje:** `y_true` coincide 100% con `target_depresion` (58 sin riesgo / 21 con riesgo). Con ansiedad coincide solo el 90%. **Todo lo que sigue es sobre depresión.** No hay datos de ansiedad en esta carpeta.

**Qué explica cada matriz** (verificado por aditividad):
- SHAP `au__*` → `prob_ml_video`. La suma de SHAP más una base de 0.266 reproduce la probabilidad (r = 1.00).
- SHAP `destilacion_egemaps__*` → `prob_destilacion_estudiante`, con base 0.371 (r = 0.99).
- LIME `au__*` → modelo de video. El valor guardado es el coeficiente del sustituto lineal local, con signo casi constante por feature: el signo indica la *dirección* del efecto, no la contribución del caso.
- `bilstm_embeddings` y `extratrees_embeddings` **no tienen explicación** en la carpeta, porque operan sobre embeddings no interpretables.

## 2. Desempeño (AUC fuera de muestra, depresión, n = 79)

| Modelo | AUC promedio 10 rep. | AUC por repetición (media ± DE) |
|---|---|---|
| `bilstm_embeddings` | **0.708** | 0.669 ± 0.042 |
| `destilacion_estudiante` (voz, eGeMAPS) | 0.698 | 0.665 ± 0.035 |
| `extratrees_embeddings` | 0.665 | 0.670 ± 0.031 |
| `ml_video` (rostro, AU) | 0.632 | 0.612 ± 0.054 |
| Promedio de los 4 | 0.703 | — |

El modelo facial, que es el que tiene SHAP y LIME, es el **más débil** de los cuatro.

## 3. Rostro: qué empuja hacia "con riesgo"

Se ordenó por |SHAP| medio. La dirección es la correlación de Spearman entre el valor de la feature y su SHAP, y se contrastó con el signo de LIME. Como control independiente del modelo se aplicó Mann-Whitney positivos vs negativos, con corrección FDR (Benjamini-Hochberg) sobre 67 features.

| # SHAP | Feature | Dirección (↑ riesgo) | LIME concuerda | AUC univariado | q (FDR) |
|---|---|---|---|---|---|
| 1 | `anger_std` (variabilidad de expresión de enojo, py-feat) | más alta | sí | 0.74 | **0.036** |
| 2 | `anger_mean` (expresión de enojo media) | más alta | sí | 0.75 | **0.036** |
| 3 | `AU15_std` (depresor de comisura labial) | más variable | sí | 0.57 | 0.59 |
| 4 | `AU01_mean` (elevador interno de ceja) | **más baja** | sí | 0.72 (invertido) | 0.077 |
| 5 | `Yaw_mean` (giro de cabeza) | más baja | sí | 0.66 (invertido) | 0.54 |
| 6 | `n_frames_detected` | más alta | sí | 0.60 | — |
| 8 | `AU20_mean` (estirador labial) | más alta | sí | 0.60 | — |
| 13 | `AU05_std` (elevador del párpado superior) | menos variable | sí | 0.65 (invertido) | 0.54 |

- La dirección SHAP y el signo LIME coinciden en el **100% del top-20** y en el 85% de las 67 features. El ranking de magnitud también concuerda (ρ = 0.87).
- **Lectura por participante:** en 10 de los 21 positivos, `anger_mean` está entre las 3 features que más empujan hacia riesgo. En los negativos, `anger_*` y `AU01_mean` son las que más empujan hacia "sin riesgo".
- **Qué captura `anger` aquí:** correlaciona con más `AU20` (ρ = 0.28) y con **menos sonrisa**: `AU12` ρ = −0.25, `AU06` ρ = −0.25, `happiness` ρ = −0.25. Es una mezcla de tensión perioral y menor sonrisa, no un "enojo" clínico.

### Hipótesis que los datos NO sostienen

| Hipótesis | Features | Rank SHAP | AUC | p |
|---|---|---|---|---|
| Parpadeo repetido | `AU43_mean` / `AU43_std` (ojos cerrados) | 57 / 24 | 0.60 / 0.56 | 0.19 / 0.42 |
| Sonrisa | `AU12_mean` / `happiness_mean` | 41 / 40 | 0.51 / 0.48 | 0.93 / 0.75 |
| Apertura de boca | `AU25_mean` / `AU26_mean` | 30 / 17 | 0.39 / 0.40 | 0.15 / 0.17 |

Además, las features son medias y desviaciones de toda la entrevista, muestreadas con salto de frames. Un parpadeo (100–400 ms) no queda bien representado en esa agregación, así que tampoco se puede afirmar lo contrario.

### Confusores en el top-6

- `n_frames_detected` (#6) es control de calidad, ligado a la duración del clip. `pipeline_multimodal_xai` la descarta como posible confusor, pero este modelo la usa.
- `Yaw_mean` (#5) es la orientación media de la cabeza y depende de cómo se sostuvo el celular.

Ambas deberían excluirse, o discutirse explícitamente, antes de publicar.

## 4. Voz (modelo destilado, eGeMAPS)

- Pesa menos en su propio modelo: la suma de |SHAP| promedia 0.075, contra 0.31 del facial.
- **Ninguna feature es discriminante por sí sola:** todas tienen q ≥ 0.57.
- El top está dominado por:
  - percentiles de F0 (altura tonal), con relación no monótona;
  - `F1frequency_amean` ↓ y `F2frequency_amean` ↓ (formantes);
  - `F0semitone_stddevNorm` ↓ (menor variabilidad de tono, es decir, prosodia más monótona).
- La monotonía prosódica es coherente con la literatura de depresión. Aquí es solo sugestiva.

## 5. Conclusión para el título

La única señal que cumple los tres criterios es la **expresión facial de tipo enojo/tensión (py-feat `anger`) más alta y más variable**, junto con una **menor activación de la ceja interna (AU1)**:

1. alta en SHAP y LIME, con la misma dirección;
2. discriminante por sí misma (AUC ≈ 0.72–0.75);
3. sobrevive (o casi) a la corrección por comparaciones múltiples.

Condiciones para redactar:

- El título debe hablar de **depresión**, no de ansiedad.
- Debe usar **"se asocian con" o "explican"**, no "son indicador". El modelo facial tiene AUC 0.63, n = 79 y casos mayormente leves.

Títulos propuestos:

1. *Expresiones faciales de enojo y menor elevación de la ceja interna como marcadores explicables del riesgo de depresión en jóvenes rurales: un enfoque multimodal con SHAP y LIME*
2. *¿Qué mira la IA en un rostro con riesgo de depresión? Tensión facial y baja activación de la ceja interna en un modelo audiovisual explicable*
3. *Tensión facial, menos cejas y voz monótona: explicabilidad SHAP/LIME de un modelo multimodal de riesgo de depresión en jóvenes de Samaná, Caldas*

El tercero solo si se acepta que la parte de voz es sugestiva y no significativa.

## 6. Pendientes antes de fijar el título

1. Reentrenar el modelo facial sin `n_frames_detected` (y probar sin `Yaw_mean`), para verificar que `anger` y `AU01` siguen arriba.
2. Si el artículo va a mencionar ansiedad, correr este mismo análisis sobre la matriz de ansiedad. Esta carpeta no la tiene.
3. Describir `anger` como "probabilidad de expresión de enojo estimada por py-feat", no como emoción sentida.
