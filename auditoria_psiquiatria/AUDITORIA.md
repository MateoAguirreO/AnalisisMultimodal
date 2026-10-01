# Auditoría del repositorio `Psiquiatria` (componente multimodal)

Fecha: 2026-09-28. Repositorio auditado: https://github.com/MateoAguirreO/Psiquiatria (commit `09b577a`), clonado en `../Psiquiatria`.

Alcance: el código multimodal que generó `shapLimePorMuestra/`, la rama de destilación (KD) y lo que haría falta para desplegar esos modelos en `../PlataformaMultimodal`. No se reentrenó nada del repo auditado. La fuga de KD se midió con los artefactos que el propio repo trae (`verificar_fuga_kd.py` en esta carpeta).

## 1. Qué hay en el repo

- **Multimodal existe solo para depresión** (`Depresion/Multimodal/`). No hay ninguna versión multimodal de ansiedad.
- Versiones sucesivas del constructor de la matriz OOF:

  | Script | Modelos | Nota |
  |---|---|---|
  | `build_stacking_oof_matrix.py` | ET-embeddings, BiLSTM, KD (AdaBoost), BiGRU sobre AU | 5×5 folds |
  | `buildStOof2.py` | Los mismos, con/sin KD | 5×10 folds, esquema train_fit/val_early/test |
  | `build_oof_matrix_v3.py` | + ET sobre eGeMAPS, ET sobre AU (reemplaza BiGRU) | Agrega SHAP/LIME |
  | `build_oof_matrix_v4_balanceado.py` | Quita ET-eGeMAPS, `class_weight="balanced"` en video | **Generó `shapLimePorMuestra/`** |

- Los CSV de fusión que trae el repo (`fusion_*.csv`) corresponden a la matriz vieja con BiGRU, **no a la v4**.
- **No hay `.joblib` ni checkpoints.** El `.gitignore` excluye `*.joblib`, `*.pt` y `*.pkl`, y tampoco existe script de exportación ni de inferencia.

Las etiquetas (`pacientes_finales.csv`) coinciden al 100% con las nuestras en ambos ejes: 79 pacientes, sin el 66. Ese frente está bien.

## 2. Errores de procedimiento

| # | Error | Dónde | Efecto | Evidencia |
|---|---|---|---|---|
| E1 | **Ansiedad sin multimodal.** `target_ansiedad` solo aparece en la lista de columnas a descartar. | `Depresion/Multimodal/*` | Cualquier conclusión multimodal del repo vale solo para depresión. | Comentario en v3/v4: una corrida anterior había usado `target_ansiedad` **como feature** del video de depresión (AUC 0.69 → 0.89). Ya estaba corregido en v3/v4. |
| E2 | **El profesor de ansiedad usa los hiperparámetros elegidos para depresión** ("se usa EXACTAMENTE esta configuración para ambas condiciones"). | `Kd_bilstm_lazypredict.py:154-163` | **AUC del profesor de ansiedad = 0.502 (azar).** Toda la destilación de ansiedad imita ruido. | `teacher_oof_ansiedad.csv`: AUC por repetición 0.502 ± 0.050. El de depresión da 0.769. |
| E3 | **Early stopping del profesor con la pérdida del mismo fold que se reporta.** El fold de validación sirve a la vez para elegir el checkpoint y como predicción OOF. | `Kd_bilstm_lazypredict.py:432-445` | Las OOF del profesor están infladas. Es el mismo bug que el proyecto corrigió en el BiLSTM (FIX-E/F), donde midió una caída de ~0.09 de AUC al eliminarlo. | El docstring dice "protocolo limpio", pero `loader_val` es el mismo en ambos usos. |
| E4 | **Fuga de segundo orden en el estudiante.** Se entrena con `p_teacher_OOF` de los pacientes de train, y esas probabilidades salen de profesores entrenados **con las etiquetas del fold de test del estudiante**. | `Kd_bilstm_lazypredict.py:628`, `build_oof_matrix_v4_balanceado.py:1298-1304` | **Medido: +0.078 de AUC.** Mismo estudiante y mismos folds: 0.657 con objetivos fugados, 0.580 con limpios (fugado > limpio en 13/15 folds). Entrenado con la etiqueta real da 0.540. | `verificar_fuga_kd.py`, con la caché `teacher_oof_inner_cache/` del propio repo |
| E5 | **Sesgo de selección (winner's curse).** Cada configuración "ganadora" es el máximo de muchas, evaluado sobre los mismos 79 pacientes con los que luego se reporta, sin CV anidada. | KD: mejor de 406 filas por eje. BiLSTM: mejor fila de un barrido. ET: "ExtraTrees optimizado". Video: mejor de LazyPredict × escaladores. Modelo SSL: mejor de 6. | Todos los AUC base son optimistas en una cantidad no medible sin los embeddings. | El propio repo lo reconoce en `optimize_adaboost_winner_champion_challenger.py`. |
| E6 | **Embeddings wav2vec no reproducibles.** `AttentionPooling` se instancia con pesos aleatorios: nunca se entrena, no tiene semilla y no se guarda. | `embedding_pipeline_5s.py:382-416` | Un paciente nuevo **no se puede proyectar al mismo espacio**. Las ramas BiLSTM, ET-embeddings y el profesor de KD **no son desplegables**. | `grep` de `manual_seed` / `seed`: sin resultados. |
| E7 | **Confusor `n_frames_detected` dentro del modelo de video.** | `build_oof_matrix_v4_balanceado.py:315` (no está en `DROP_COLS_VIDEO`) | Es la 6.ª feature por SHAP (ver `shapLimePorMuestra/ANALISIS_titulo_articulo.md`). | Nuestro pipeline ya la excluye. |
| E8 | **Circularidad KD ↔ BiLSTM.** El profesor de KD *es* el BiLSTM. | v4, `REUSE_BILSTM_AS_TEACHER=True` | La fusión cuenta dos veces la misma señal. | Advertido en el propio docstring de v4. |
| E9 | **AUC calculado sobre predicciones promediadas entre repeticiones.** | `fusion_resumen_final.csv` | "Voting" da 0.745 promediado vs **0.677 por repetición**, que es la cifra comparable. | `fusion_resumen_por_repeat.csv` |
| E10 | **Meta-modelo de stacking entrenado con OOF de la misma partición externa.** | `fusion_tardia_nested.py`, `late_fusion_nested.py` | Fuga leve para los fusores entrenados (logreg, MLP, perceptrón). "Voting" no se ve afectado. | — |
| E11 | **Cada ventana "de 5 s" se recorta a 0.19 s** antes de wav2vec: `processor(..., padding="max_length", max_length=3000, truncation=True)` corta en 3000 muestras a 16 kHz. El embedding solo ve los primeros 0.19 s (9 frames) de cada ventana, cada 2.5 s, en la última capa. | `embedding_pipeline_5s.py:460-467` | No es la representación que el proyecto dice usar. Paradójicamente, **es la que tiene señal en depresión**; la ventana completa de 5 s no la tiene (ver `experimento_embeddings/RESULTADOS.md` §8). | Réplica independiente: coincide con sus vectores (mismo número de segmentos; correlación entre participantes, mediana por dimensión = 0.998). |

Nota (2026-09-29):
- No existen modelos guardados de las ramas de embeddings de depresión (lista `modelos_guardados.txt` de la workstation). Solo hay modelos de ansiedad: espectrogramas de `fase3_viejo` y destilación de `KD_BiLSTM_XGBoost`.
- Los embeddings de las carpetas Depresión y Ansiedad son idénticos para el mismo audio.

## 3. Comparación honesta con nuestro multimodal

AUC por repetición (media ± DE) en CV repetida por paciente.

| Eje | Psiquiatria | Nuestro `pipeline_multimodal_xai` (soft vote eGeMAPS + AU) |
|---|---|---|
| **Depresión** | BiLSTM 0.669 ± 0.042 (E5) · ET-embeddings 0.670 ± 0.031 (E5) · KD 0.665 ± 0.035 (E4: limpio ≈ 0.58) · video 0.612 ± 0.054 (E7) · fusión voting 0.677 ± 0.032 (matriz vieja) | **0.674**, CV anidada, permutación p = 0.050 |
| **Ansiedad** | No existe. Profesor 0.502; mejor KD 0.61 (máximo de 406) | **0.668**, CV anidada, permutación p = 0.045 |

Aun con sus sesgos a favor, el multimodal de Psiquiatria **no supera** al nuestro en depresión. En ansiedad no existe.

## 4. Conclusión y recomendación

1. **El multimodal correcto hoy es el nuestro, uno por eje:** soft vote eGeMAPS + AU, en `pipeline_multimodal_xai/`.
   - Los `.joblib` de `../PlataformaMultimodal/backend/modelos/` son **idénticos** (md5) a los de `pipeline_multimodal_xai/modelos/`.
   - Son un par de modelos por eje (ansiedad: `anova_xgb` + `anova_logreg`; depresión: `l1logreg` + `anova_rf`) y excluyen `n_frames_detected`.
   - **No hace falta cambiar los `.joblib` de la plataforma.**
2. **Nada del lado wav2vec de Psiquiatria se puede llevar a la plataforma (E6).** Lo rescatable es la *idea*: los embeddings wav2vec2-large-robust dan ~0.67 en depresión incluso con estos sesgos. Una tercera rama de voz con embeddings de **mean pooling determinista**, evaluada con nuestra CV anidada por eje, es la única palanca con potencial real.
3. **Para el artículo:** `shapLimePorMuestra/` explica la v4. Solo cubre depresión, el video incluye `n_frames_detected` (E7) y el SHAP de voz explica un estudiante con fuga (E4). La matriz SHAP/LIME por muestra conviene regenerarla desde nuestro pipeline, para ambos ejes.

## 5. Reproducir

```bash
# desde la raíz de Multimodal/, con ../Psiquiatria clonado
python auditoria_psiquiatria/verificar_fuga_kd.py
```
