# Resultados: selección del multimodal final

Fecha: 2026-09-28. Protocolo declarado **antes** de ver resultados: `INVENTARIO_Y_PROTOCOLO.md`.

Todo con CV totalmente anidada: `RepeatedStratifiedKFold(5, 5, semilla 2026)` externo, selección de configuración en CV interna solo con el train externo, n = 79 (ansiedad 19 positivos, depresión 21). La cifra principal es el **AUC agrupado por repetición** (79 participantes, media ± DE de 5 repeticiones).

Scripts:
- `preparar_audio_crudo.py`: audio crudo + Demucs.
- `extraer_embeddings.py`: wav2vec2-large-robust con mean pooling.
- `egemaps_audio_crudo.py`: eGeMAPS con la función de la plataforma.
- `evaluar_ramas.py`: evaluación anidada.
- `permutacion_anidada.py`: permutación que repite toda la selección.

## 0. Resumen en AUC estándar (métrica principal, 2026-09-30)

Por decisión de Mateo, la métrica principal es el **AUC estándar sobre todas las muestras juntas**: 79 participantes con audio, 5×5 anidado, media de las 5 repeticiones. El análisis por día de grabación (§3, §7–§11) queda solo como nota de robustez.

**Depresión**

| Modelo | Audio que necesita | AUC |
|---|---|---|
| **Voz micro-ventanas (R) + rostro** | editado (solo participante) | **0.683 ± 0.037** |
| Voz micro-ventanas sola | editado | 0.676 ± 0.035 |
| Voz R reentrenada con audio crudo + rostro | crudo (el que recibe la plataforma) | 0.636 ± 0.057 |
| eGeMAPS crudo + rostro | crudo | 0.635 ± 0.050 |
| Rostro solo | — | 0.616 ± 0.041 |
| eGeMAPS + rostro (lo que está hoy en la plataforma) | editado | 0.599 ± 0.067 |

**Ansiedad**

| Modelo | AUC |
|---|---|
| **Rostro solo** | **0.609 ± 0.014** |
| eGeMAPS + rostro (con audio crudo o editado) | 0.597–0.609 |
| Voz micro-ventanas + rostro | 0.581 ± 0.034 |

- En depresión, la voz de micro-ventanas es la mejor señal, y la fusión con el rostro es el mejor modelo. Rinde 0.683 con audio editado. Con audio crudo, reentrenada, baja a 0.636, pero sigue por encima del rostro solo.
- En ansiedad la voz no suma: el mejor modelo es el rostro solo.

> **CORRECCIÓN (ronda 4, §12): el 0.683 NO es robusto.** Depende del punto exacto donde se toman las micro-ventanas. Al desplazarlo 0.5–2.0 s, la voz cae a 0.50–0.59 y voz + rostro a 0.575–0.640. En promedio sobre 5 puntos de muestreo: voz 0.576 y voz + rostro 0.627. La cifra honesta del mejor modelo de depresión es la de la búsqueda AUTO: **0.646** (§12). En ansiedad, con 80 muestras: **0.549** (AUTO).

## 1. Resultado principal (ronda 1)

| Método | Ansiedad | Depresión |
|---|---|---|
| **V** rostro (AU) | **0.609 ± 0.014** | **0.616 ± 0.041** |
| A_ed voz eGeMAPS (audio editado) | 0.509 ± 0.070 | 0.524 ± 0.042 |
| A_crudo voz eGeMAPS (audio crudo) | 0.518 ± 0.069 | 0.604 ± 0.044 |
| E voz wav2vec (audio crudo) | 0.381 ± 0.076 | 0.514 ± 0.060 |
| **A_ed+V** (fusión actual en la plataforma) | 0.597 ± 0.021 | 0.599 ± 0.067 |
| E+V | 0.518 ± 0.058 | 0.583 ± 0.081 |
| A_ed+E+V | 0.526 ± 0.070 | 0.565 ± 0.078 |
| A_crudo+V | 0.607 ± 0.020 | 0.635 ± 0.050 |

**Decisión según la regla declarada: la rama de embeddings (E) NO entra en ningún eje.** En ambos ejes, E+V y A_ed+E+V quedan por debajo de A_ed+V en AUC medio. Por repetición, E+V le gana a A_ed+V en 0/5 (ansiedad) y 2/5 (depresión), y A_ed+E+V en 0/5 en ambos ejes: ninguna alcanza la mayoría que exige la regla.

## 2. Las cifras anteriores estaban infladas

| Eje | Antes (`pipeline_multimodal_xai`) | Ahora (totalmente anidada) |
|---|---|---|
| Ansiedad, soft vote A+V | 0.668 | 0.597 |
| Depresión, soft vote A+V | 0.674 | 0.599 |

Causa: en el pipeline anterior, los 10 folds del screening eran 10 de los 25 de la confirmación (misma semilla 42), y el ganador (soft vote) se eligió como el máximo de esa misma tabla. La permutación de entonces congelaba la arquitectura ganadora, así que su p-valor no incluía la selección.

## 3. Hallazgo central: la voz sigue la sesión de grabación, el rostro no

Las fechas de modificación de los `.mp4`/`.m4a` conservan la hora de grabación:
- **7 días en dos campañas**: 19–22 de noviembre y 9–11 de diciembre de 2025.
- El orden del código coincide en un 89% con el orden temporal.
- El primer día (26 participantes) tiene 12% de positivos; los demás, ~30%.
- **El orden de grabación, por sí solo, da AUC 0.60 (ansiedad) y 0.61 (depresión)**, lo mismo que los modelos.

Para separar señal real de efecto de sesión se calculó el **AUC intra-día**: solo cuenta pares positivo/negativo grabados el mismo día. Eso elimina cualquier diferencia entre sesiones.

| Método | Ansiedad global → intra-día | Depresión global → intra-día |
|---|---|---|
| **V** rostro | 0.609 → **0.606** | 0.616 → **0.625** |
| A_ed | 0.509 → 0.472 | 0.524 → 0.458 |
| A_crudo | 0.518 → 0.487 | 0.604 → 0.530 |
| E | 0.381 → 0.311 | 0.514 → 0.459 |
| A_ed+V | 0.597 → 0.584 | 0.599 → **0.532** |
| A_crudo+V | 0.607 → 0.593 | 0.635 → 0.594 |

- **El rostro conserva su AUC dentro de cada día.** Su score casi no correlaciona con el orden de grabación (ρ = 0.16 y 0.09).
- **Toda la voz cae a ≤ 0.53 dentro de cada día.** Sus scores correlacionan con el orden de grabación (ρ = 0.28–0.50). La aparente ventaja de A_crudo en depresión (0.604) era efecto de sesión.
- **Agregar voz al rostro empeora el resultado dentro de cada día**: en depresión, A_ed+V da 0.532 frente a 0.625 de V.

Por qué los embeddings del audio crudo fallan tanto (diagnóstico por PCA):
- **PC1 concentra el 86% de la varianza** y correlaciona con la fracción de audio que se quitó al editar (ρ = −0.52). Es decir, codifica la **voz del entrevistador**.
- **PC2, PC3 y PC5 correlacionan con el orden de grabación** (|ρ| = 0.44–0.54): sala, equipo o día.
- El AUC por debajo de 0.5 es la huella típica de una estructura de grupos ajena a la etiqueta bajo CV.

## 4. Desajuste de audio en la plataforma (exploratorio)

- Las eGeMAPS de entrenamiento salen de audio editado a mano. Se conserva en mediana el 67% de la duración cruda, y 29 de 79 participantes conservan menos del 60%.
- La plataforma calcula eGeMAPS sobre el audio crudo.
- Entre participantes, ambas versiones correlacionan con mediana 0.80, pero el 10% de las features tiene correlación ≈ 0.
- El modelo de voz desplegado (A_ed→crudo) da 0.530 en ansiedad y 0.541 en depresión: tan cerca del azar como con el audio editado.

## 5. Test de permutación (repite toda la selección anidada)

Candidato evaluado: V, el rostro solo, que es la única rama que sobrevive al control de sesión. Se usaron 200 permutaciones. En cada una se vuelve a elegir la configuración de rostro, entre 5, en la CV interna. Estadístico: AUC agrupado de 1 repetición 5-fold.

| Eje | AUC real (1×5) | Nulo: media | Nulo: percentil 95 | p |
|---|---|---|---|---|
| Ansiedad | 0.611 | 0.508 | 0.657 | **0.154** |
| Depresión | 0.586 | 0.504 | 0.684 | **0.209** |

**Ninguno es significativo.** Con N = 79 (19–21 positivos) y la selección de configuración incluida, un AUC de ~0.60 cae dentro de lo que el azar produce: el percentil 95 del nulo está en 0.66–0.68.

Matices:
- El estadístico usa una sola repetición y es ruidoso. En depresión, esa repetición dio 0.586, frente a 0.616 de media en 5 repeticiones.
- Una permutación que promedie las 5 repeticiones tendría más potencia, pero cuesta ~5 veces más (unas 4 h por eje).
- El resultado v3 de solo-video en depresión (AUC 0.662, p = 0.0033) usaba otra representación: 256 features temporales. Habría que verificar si su permutación repetía la elección entre las 12 configuraciones.

## 6. Conclusión de la ronda 1 (actualizada en §10)

1. **Con la evidencia de la ronda 1, el modelo final defendible es el rostro solo, un modelo por eje.** Es la única modalidad cuya señal sobrevive al control de sesión. La fusión con voz no mejora y, controlando sesión, empeora. Con honestidad: su AUC (~0.61) **no alcanza significancia** cuando la permutación incluye la selección (p = 0.15 y 0.21). Es una señal consistente pero débil, a presentar como exploratoria.
2. **La voz no está lista para entrar**, ni eGeMAPS ni wav2vec. Faltan dos cosas:
   - aislar la voz del participante de forma reproducible: diarización automática que también pueda correr en la plataforma, o conseguir los `NNN_editado.wav` del servidor para entrenar;
   - evaluar con control de sesión: AUC intra-día o CV agrupada por día.
3. Para el artículo y la tesis, el efecto de sesión es un hallazgo metodológico valioso en sí mismo. En N pequeño, **la voz aprende la sesión antes que al participante**, y solo se detecta con control por fecha de grabación.
4. Los `.joblib` de la plataforma (soft vote A+V) no se tocaron. Cambiarlos a rostro solo es decisión pendiente.

## 7. Ronda 2: audio editado a mano (protocolo en `INVENTARIO_Y_PROTOCOLO.md` §5)

Mateo trajo del servidor los 79 `NNN_recortado_denoised.wav`: editados sin entrevistador, recortados y con Demucs. Validación: las eGeMAPS recalculadas coinciden con las de entrenamiento (mismos segmentos; diferencia relativa p95 = 0.0001).

Nuestros embeddings (mean pooling, promedio de capas 1–24, ventanas completas de 5 s) sobre ese audio:

| Método (AUC intra-día / agrupado) | Ansiedad | Depresión |
|---|---|---|
| V rostro | 0.606 / 0.609 | 0.625 / 0.616 |
| E_ed | 0.247 / 0.354 | 0.429 / 0.509 |
| E_ed+V | 0.465 / 0.499 | 0.520 / 0.590 |
| A_ed+E_ed+V | 0.447 / 0.499 | 0.490 / 0.571 |

**Decisión según la regla: la voz no vuelve en ningún eje con esta representación.**
- El PC3 de E_ed está explicado en un 65% por el día de grabación, y toda su relación con la etiqueta es entre días.
- El AUC muy por debajo de 0.5 es el artefacto conocido de CV con datos agrupados y sin señal ("stratification bias", Parker et al. 2007): el modelo aprende la tasa de positivos de cada día, y al sacar a un participante para probarlo, la tasa de su día en el entrenamiento se mueve en contra de su etiqueta.

## 8. Ronda 3 (EXPLORATORIA): los embeddings originales de Psiquiatria

Mateo exportó desde la workstation sus embeddings wav2vec2-large-robust (`embeddings_v2`, campo `embedding`, todos los segmentos). Se evaluaron con el mismo protocolo (`evaluar_emb_psiquiatria.py`, `replicar_emb_psiquiatria.py`).

**Hallazgo (auditoría E11).** Su extractor recorta cada ventana de 5 s a **0.19 s** (`max_length=3000` muestras) y usa la última capa. Su representación son micro-muestras de 0.19 s tomadas cada 2.5 s. Se construyó una **réplica determinista (R)**: el mismo recorte, pero con el promedio de los 9 frames en lugar de los pesos aleatorios. R reproduce sus vectores (correlación entre participantes, mediana por dimensión 0.998), así que **sí es desplegable**.

| Método (AUC intra-día / agrupado) | Ansiedad | Depresión |
|---|---|---|
| V rostro | 0.606 / 0.609 | 0.625 / 0.616 |
| P, sus embeddings | 0.434 / 0.500 | 0.588 / 0.653 |
| R, réplica determinista | 0.462 / 0.520 | 0.607 / 0.676 |
| P+V | 0.547 / 0.575 | 0.642 / 0.669 (gana a V en 3/5 repeticiones) |
| R+V | 0.538 / 0.581 | 0.637 / 0.683 (gana a V en 2/5) |
| PAUSA (4 features de pausas) | 0.314 / 0.368 | 0.328 / 0.344 |

**Controles:**
- **Hipótesis de pausas: descartada.** Las pausas medidas directamente (fracción de silencio, pausas por minuto, duración media, fracción de micro-muestras en silencio) tienen AUC univariado de 0.53–0.55, y como rama quedan por debajo del azar (mismo artefacto de §7).
- **El sexo no explica la señal de voz en depresión.** El sexo solo da AUC 0.547 (29% de mujeres con riesgo, frente a 21% de hombres). La réplica R rinde 0.690 dentro de cada sexo. Controlando sexo y día a la vez: V 0.615, R 0.608, **R+V 0.644**. En ansiedad, R dentro de cada sexo = 0.504: sin señal.

**Lectura:**
- En **depresión**, la representación de micro-muestras de la última capa tiene señal propia que sobrevive al control de día y de sexo (~0.61), comparable a la del rostro. La fusión mejora poco y de forma inestable: +0.012 intra-día, gana en 2 de 5 repeticiones.
- En **ansiedad**, la voz no tiene señal con ninguna representación.
- No se sabe qué captura la última capa en 0.19 s; las pausas quedan descartadas. Es exploratorio: la representación salió de un error de código y se evaluó después de ver que P funcionaba. Por eso hace falta el test de §9.

## 9. Test de permutación intra-día (depresión)

Barajado de etiquetas **dentro de cada día**, lo que conserva el efecto de sesión en el nulo. Repite toda la selección anidada. Estadístico: AUC intra-día medio de las mismas 5 repeticiones de la evaluación. 100 permutaciones (`permutacion_anidada.py --intra-dia --n-rep 5`, ejecutado en lotes reanudables).

| Método | AUC intra-día real | Nulo: media | Nulo: percentil 95 | p |
|---|---|---|---|---|
| **R+V** (voz réplica + rostro) | **0.637** | 0.478 | 0.630 | **0.040** |

- El valor real reproduce exactamente la tabla de §8 (mismos folds).
- Bajo este nulo, el AUC **agrupado** promedia 0.542. Es decir, la voz aprovecha el efecto de día incluso sin señal: por eso el estadístico correcto es el intra-día.
- La media del nulo intra-día queda por debajo de 0.5 (0.478): es el artefacto de CV con grupos descrito en §7.

**Matices, a declarar si se publica:**
1. **Incertidumbre de Monte Carlo:** 3 de 100 permutaciones igualaron o superaron el real. Con 100 permutaciones, el p tiene un margen aproximado de 0.01–0.10.
2. **Es un resultado de la ronda exploratoria.** La permutación incluye la selección dentro del pipeline, pero no los caminos que probamos entre rondas: representaciones E, E_ed y R. Corrigiendo por 3 representaciones (Bonferroni), p ≈ 0.12. Lo honesto es presentarlo como **hallazgo prometedor a confirmar en una muestra nueva**, no como resultado confirmatorio.
3. La ganancia sobre el rostro solo es pequeña: +0.012 intra-día (+0.029 controlando sexo y día) y gana en 2 de 5 repeticiones.

## 10. Conclusión actualizada (tras las rondas 2 y 3)

1. **Ansiedad: rostro solo.** La voz no tiene señal en ansiedad con ninguna representación: eGeMAPS, embeddings propios o los de Psiquiatria. El rostro rinde ~0.61 (intra-día 0.606), pero no es significativo bajo permutación global (p = 0.15).
2. **Depresión: multimodal voz + rostro (R+V).** La voz, con la representación de micro-muestras de 0.19 s de la última capa de wav2vec2-large-robust (réplica determinista R, desplegable), aporta señal propia que sobrevive al control de día y de sexo. La fusión con el rostro da 0.637 intra-día, **p = 0.040** bajo un nulo que conserva el efecto de sesión. Es exploratorio (ver matices de §9).
3. **R NO funciona sobre audio crudo (§11).** La plataforma recibe audio crudo, así que hoy la voz no se puede desplegar. Para eso hace falta separar automáticamente la voz del participante (diarización), o que el audio llegue ya editado.
4. **Mejor modelo desplegable hoy (audio crudo): rostro solo en ambos ejes.** Depresión: 0.616 agrupado / 0.625 intra-día. Ansiedad: 0.609 / 0.606. El mejor modelo de investigación (con audio editado) es R+V en depresión: 0.683 / 0.637.
5. **Hallazgos metodológicos para el artículo:**
   - la voz aprende la sesión de grabación;
   - el AUC intra-día y el nulo con barajado intra-día son necesarios para detectarlo;
   - los AUC por debajo de 0.5 son artefacto de CV con grupos;
   - la representación útil de Psiquiatria surgió de un error de código (recorte a 0.19 s).

## 11. ¿Funciona R sobre audio crudo? (`evaluar_R_crudo.py`)

Misma receta de micro-ventanas sobre `features/audio_crudo_demucs/`: grabación completa, con entrevistador, pasada por Demucs, que es lo que recibe la plataforma. Mismo protocolo; las predicciones de V se reutilizan (mismos folds y semilla).

| Depresión (AUC intra-día / agrupado) | |
|---|---|
| V rostro | 0.625 / 0.616 |
| R_ed (entrena y prueba con editado; referencia) | 0.607 / 0.676 |
| R_crudo (entrena y prueba con crudo) | 0.531 / 0.584 |
| R_ed2crudo (entrena con editado, prueba con crudo = desplegar tal cual) | 0.478 / 0.547 |
| R_ed+V | 0.637 / 0.683 |
| R_crudo+V | 0.592 / 0.636 |
| R_ed2crudo+V | 0.557 / 0.607 |

- Los vectores de crudo y editado se parecen poco entre participantes: mediana por dimensión 0.64. Las micro-ventanas caen a menudo sobre la voz del entrevistador.
- **La señal de voz depende de tener solo la voz del participante.** Sobre audio crudo, la rama de voz pierde la señal y la fusión empeora al rostro solo.

**Implicación:** para desplegar la voz hay que automatizar lo que hoy hizo la edición manual, separando al entrevistador del participante (diarización). El audio editado a mano sirve como referencia para validar esa separación, comparando R_diarizado con R_ed y midiendo el AUC.

## 12. Ronda 4: búsqueda honesta del mejor modelo (AUC estándar; protocolo en `INVENTARIO_Y_PROTOCOLO.md` §6)

`buscar_mejor_modelo.py`: compiten las ramas y todas sus combinaciones. AUTO elige en cada fold con los datos de entrenamiento; su AUC es la cifra honesta de "quedarse con el mejor".

**Depresión (n = 79)**

| Modelo | AUC |
|---|---|
| R+V | 0.683 ± 0.037 |
| R | 0.676 ± 0.035 |
| R+RD+V | 0.666 ± 0.032 |
| **AUTO** | **0.646 ± 0.037** (eligió R+V en 14/25 folds) |
| RD+V | 0.638 ± 0.045 |
| V | 0.616 ± 0.036 |
| RD (micro-ventanas densas; la CV interna eligió 0.19 s en 21/25 folds) | 0.612 ± 0.048 |
| VT (256 temporales de julio) | 0.544 ± 0.099 (corrida con R, V, VT; AUTO de esa corrida: 0.649) |

**Ansiedad (n = 80, solo rostro)**: V 0.564 ± 0.060, V+VT 0.563, **AUTO 0.549**, VT 0.529.

**Robustez de R frente al punto de muestreo** (`robustez_offset_R.py`, depresión):

| Desplazamiento de la micro-ventana | R | R+V |
|---|---|---|
| 0 s (Psiquiatria / réplica) | 0.677 | 0.684 |
| 0.5 s | 0.559 | 0.620 |
| 1.0 s | 0.552 | 0.615 |
| 1.5 s | 0.588 | 0.640 |
| 2.0 s | 0.504 | 0.575 |
| **Promedio** | **0.576** | **0.627** |

**Lectura:**
- El buen resultado de la voz R depende de un punto de muestreo arbitrario (el de Psiquiatria); no hay un mecanismo que lo justifique. Con las micro-ventanas densas, que promedian todas las posiciones, queda en ~0.61.
- **La voz de micro-ventanas no es una señal robusta.** El p = 0.040 de §9 corresponde a esa configuración afortunada y no incluye la variabilidad del muestreo: no debe citarse como evidencia confirmatoria.
- **Conclusión honesta con estos datos:**
  - Depresión: ~0.62–0.65 (AUTO 0.646; rostro solo 0.616; voz + rostro promediada sobre muestreos 0.627).
  - Ansiedad: ~0.55–0.61.
  - Las cifras mayores que hemos visto (0.68–0.77) salen de configuraciones afortunadas o de fugas.
- **Palancas que quedan, con información genuinamente nueva y no una re-lectura de las mismas señales:**
  - las **transcripciones** de las entrevistas (80, corregidas, nunca usadas);
  - más participantes, sobre todo casos moderados y severos;
  - revisar cómo se construyó la etiqueta "DX IA".

## 13. Texto (transcripciones) y etiquetas del cuestionario validado (2026-09-30)

**Transcripción:**
- Whisper large-v3-turbo vía `transformers` (`transcribir.py`), sobre el audio editado (solo el participante). El 66 se transcribió desde el audio de su video, que incluye al entrevistador.
- 80 transcripciones; mediana de 329 palabras.

**Features de texto** (`texto_features.py`):
- embeddings `paraphrase-multilingual-mpnet-base-v2` (768 dimensiones);
- 7 features léxicas fijadas a priori: primera persona, emociones negativas y positivas, absolutistas, negaciones, número de palabras, palabras por minuto.
- La CV interna elige entre ambas representaciones.

**Etiquetas:**
- **"DX IA": de origen desconocido.** Mateo no sabe cómo se generó.
- No se relaciona con el cuestionario: PHQ-9 r = −0.04; GAD-7, AUC 0.59.
- Del Excel, lo más asociado son los antecedentes familiares (p = 0.005).
- **Etiquetas validadas:** PHQ-9 ≥ 5 (23 de 69) y GAD-7 ≥ 5 (19 de 69), es decir, síntomas al menos leves. Solo 69/80 tienen el cuestionario completo. Con el corte clínico (≥ 10) hay apenas 5 y 3 casos.
- "DX IA" de depresión es independiente de PHQ-9 ≥ 5: 33% de positivos en ambos grupos.

**Mejor modelo honesto (AUTO), AUC estándar:**

| Eje | Ramas | DX IA | Cuestionario (≥ 5) |
|---|---|---|---|
| Depresión | voz R, RD + rostro (n = 79/68) | **0.646** | 0.508 |
| Depresión | texto + rostro (n = 80/69) | 0.557 | 0.286 |
| Ansiedad | rostro (n = 80) | 0.549 | — |
| Ansiedad | voz R, RD + rostro (n = 68) | — | 0.499 |
| Ansiedad | texto + rostro (n = 80/69) | 0.514 | 0.444 |

El texto solo, contra DX IA: 0.551 (depresión) y 0.361 (ansiedad).

**Diagnóstico de los AUC muy por debajo de 0.5:**
- La alineación de etiquetas está verificada, sin error.
- Con etiquetas barajadas al azar, un modelo simple sobre el rostro (69 personas) da AUC entre **0.31 y 0.64** (percentiles 5–95). Los valores bajos están dentro de lo que produce el azar con este N.
- Además, rostro y texto reconocen la campaña de grabación (AUC 0.77 y 0.66, noviembre frente a diciembre), y la prevalencia de PHQ-9 ≥ 5 difiere entre campañas (26% frente a 46%). Esa estructura agrupada amplifica el sesgo de la CV.

## 14. Conclusión global (2026-09-30)

1. **Con estos datos, ninguna modalidad (rostro, voz o texto) predice depresión ni ansiedad de forma fiable**, ni contra "DX IA" ni contra el cuestionario validado. La mejor cifra honesta es depresión contra DX IA con voz + rostro, AUTO **0.646**. Queda en el borde de lo que produce el azar con N ≈ 80, cuyo rango del 90% llega a ~0.64.
2. **Las cifras altas vistas en el proyecto** (0.68–0.77 aquí y en Psiquiatria) **salieron de fugas, de selección sobre los mismos datos o de configuraciones afortunadas.** Ejemplo: la micro-ventana de voz cae de 0.68 a 0.50–0.59 con solo desplazarla medio segundo.
3. **Causas estructurales:**
   - N ≈ 80 con ~20 positivos: el AUC tiene una incertidumbre enorme.
   - La etiqueta principal ("DX IA") es de origen desconocido y no coincide con los instrumentos validados.
   - La muestra es subclínica: PHQ-9 medio de 3.4, 5 casos ≥ 10.
   - Hay efecto de campaña/sesión en las señales.
4. **Qué haría falta para métricas buenas y creíbles:**
   - más participantes, con casos moderados y severos;
   - una etiqueta de referencia validada (entrevista clínica, o PHQ-9/GAD-7 completos);
   - protocolo de grabación estandarizado (mismo equipo y sala);
   - tareas de habla estandarizadas (lectura, descripción de imagen).
5. **Actualización (§15):** con más datos de entrenamiento por fold (75/5) y 7 modelos, las mejores combinaciones de la tabla llegan a 0.70 (depresión) y 0.68 (ansiedad), pero **AUTO queda en 0.56–0.58**.
6. **Aporte defendible de la tesis con estos datos:**
   - una evaluación rigurosa y honesta de un pipeline multimodal y explicable;
   - la demostración de cuánto inflan el rendimiento las fugas, la selección, las etiquetas no validadas y las configuraciones afortunadas;
   - la plataforma como prototipo, con su rendimiento real declarado.

## 15. SHAP / LIME por muestra y todas las combinaciones (repo Psiquiatria, `Multimodal_XAI/`)

Pedido de Mateo:
- entrenar con 75 y probar con 5 (16 folds), agregando cada fold a los CSV, para ambos ejes;
- SHAP y LIME por muestra;
- explicabilidad de embeddings "si cabe";
- las mejores combinaciones de 2, 3, 4… modelos.

Se hizo con 5 repeticiones (80 folds por eje). La configuración de cada modelo se elige en la CV interna del train; la fusión es soft vote. Salidas copiadas a `shapLimePorMuestra/` (ver su `LEEME.md`).

| AUC estándar | Depresión | Ansiedad |
|---|---|---|
| `au` (rostro) | 0.605 | **0.660** |
| `rostro_temporal` | 0.451 | 0.446 |
| `egemaps` (voz) | 0.405 | **0.633** |
| `voz_microventanas` | **0.691** | 0.524 |
| `voz_densa` | 0.577 | 0.480 |
| `texto_emb` | 0.544 | 0.408 |
| `texto_lexico` | 0.346 | 0.375 |
| Mejor de 2 modelos | **au + voz_microventanas 0.701** | **au + egemaps 0.682** |
| Mejor de 3 | + texto_emb 0.678 | + voz_microventanas 0.652 |
| Mejor de 4 | 0.663 | 0.621 |
| Los 7 | 0.544 | 0.486 |
| **AUTO (honesto)** | **0.558** | **0.579** |

**Lectura:**
- En ambos ejes, **lo mejor son 2 modelos**. Agregar más empeora: los modelos sin señal diluyen el promedio.
- La brecha entre la mejor combinación de la tabla (0.68–0.70) y AUTO (0.56–0.58) es el costo de elegir entre 127 opciones con 80 personas.
- En depresión, la mejor combinación depende de `voz_microventanas`, que no es robusta (§12).
- En ansiedad, `au` y `egemaps` rinden más que con el esquema 5-fold. Hay más datos de entrenamiento por fold (75 frente a 63), pero también cambia la partición, así que no se puede separar un efecto del otro.

**Explicabilidad:**
- **Top SHAP, depresión:** `eye_squint_p90`, `slopeV0-500 stddevNorm`, `mfcc2`, `AU01_mean`, `anger_std`, `F2bandwidth`, `loudness FallingSlope`, `anger_mean`, `logRelF0-H1-H2`.
- **Top SHAP, ansiedad:** `AU15_std`, `mfcc2 stddevNorm`, `AU01_mean`, `AU20_std`, `primera_persona` (texto), `logRelF0-H1-H2`, `hammarbergIndex`, `n_palabras`, `AU05_std`, `frown_skew`.
- **Embeddings (proxy):**
  - `voz_densa` sigue los formantes y la dinámica de volumen (|ρ| ≈ 0.52 en depresión; 0.33 en ansiedad).
  - `voz_microventanas` sigue `mfcc2` (ρ ≈ 0.37–0.40).
  - `texto_emb` no se alinea con ninguna feature léxica.

## 16. XAI sobre embeddings: métodos 3 y 5 del estado del arte (`Psiquiatria/Multimodal_XAI/xai_embeddings.py`)

Base: `ESTADO_DEL_ARTE_XAI_EMBEDDINGS.md`. Papers en `papers/xai_embeddings/`.

**Método 3, probing dirigido a la tarea (Dixit et al. 2024).** Pasos: dimensiones importantes para el clasificador (|coef| de una logística, 16 folds) → probes Ridge que predicen cada eGeMAPS (o feature léxica) desde todas las dimensiones, desde las 50 importantes y desde 50 al azar.
- **Los embeddings sí codifican bien las propiedades acústicas**: R² medio por categoría de 0.52–0.77 desde todas las dimensiones.
- **Pero las 50 dimensiones que más usa el clasificador no codifican ninguna categoría mejor que 50 al azar** (ganancia ≈ 0 en ambos ejes). Coherente con que esos clasificadores no tienen señal robusta: AUC de la logística 0.52–0.59 en depresión y 0.36–0.49 en ansiedad.
- **Excepciones en depresión:** en texto, las dimensiones importantes codifican la **primera persona** (R² 0.33 frente a 0.18 al azar); en voz densa, la pendiente espectral en tramos sordos (`slopeUV0-500`, 0.75 frente a 0.59).
- En ansiedad las ganancias puntuales (`slopeV500-1500`, `mfcc2V`) salen de clasificadores sin señal: no interpretarlas.
- **Lectura para la tesis:** con estos datos, los modelos de embeddings no se apoyan en un patrón acústico identificable. Es un resultado negativo honesto, que el método permite afirmar con evidencia.

**Método 5, atribución temporal por oclusión + transcripción de Whisper.**
- 907 fragmentos (mediana de 7 por participante y de 2.9 s).
- Para cada participante, con modelos que no lo vieron: cuánto baja su riesgo al quitar cada fragmento, en voz (micro-ventanas, eGeMAPS) y en texto.
- **Produce explicaciones legibles para un clínico.** En depresión, los fragmentos que más suben el riesgo incluyen, por ejemplo, una mención a la pérdida de seres queridos (+0.11) y una expresión de tristeza por no lograr metas.
- **También muestra desacuerdos entre modalidades:** en ansiedad, un fragmento de contenido positivo (bienestar y satisfacción personal) sube el riesgo por la **voz** (eGeMAPS +0.41) mientras el **texto** lo baja (−0.19).
- Las frases textuales están solo en `shapLimePorMuestra/atribucion_temporal_<eje>.md`, que es local e ignorado por git (dato clínico).
- **Cautela:** estas atribuciones explican modelos de señal débil (AUC ~0.55–0.65). Son útiles para mostrar *cómo* razona el sistema y para la plataforma, no como evidencia clínica.
