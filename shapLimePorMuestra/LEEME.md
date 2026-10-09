# shapLimePorMuestra: qué es cada archivo

## Nuevos (2026-09-30): ambos ejes, 7 modelos, 127 combinaciones

Generados por `../../Psiquiatria/Multimodal_XAI/xai_por_muestra.py` (ver su README). Esquema: entrena con 75, prueba con 5 (16 folds), repetido 5 veces. La configuración se elige solo con el train de cada fold.

| Archivo (`<eje>` = depresion / ansiedad) | Contenido |
|---|---|
| `oof_matrix_detail_by_repeat_<eje>.csv` | probabilidad de cada modelo y de AUTO, por participante y repetición |
| `oof_matrix_final_<eje>.csv` | promedio de las 5 repeticiones |
| `shap_matrix_<eje>.csv` | SHAP por participante y feature (`<modelo>__<feature>`), en probabilidad |
| `lime_matrix_<eje>.csv` | LIME por participante y feature (peso lineal local; el signo indica la dirección) |
| `contribucion_modelos_<eje>.csv` | aporte de cada modelo a la mejor combinación, por participante |
| `explicacion_embeddings_<eje>.csv` | a qué features interpretables se parecen las predicciones de los modelos de embeddings |
| `combinaciones_<eje>.csv` | AUC de las 127 combinaciones y de AUTO |
| `resumen_<eje>.txt` | resumen legible |

**Modelos:**
- `au`: rostro, AU de py-feat.
- `rostro_temporal`: rostro, 256 features de MediaPipe.
- `egemaps`: voz.
- `voz_microventanas`: voz, wav2vec en micro-ventanas de 0.19 s cada 2.5 s.
- `voz_densa`: voz, wav2vec en micro-ventanas contiguas.
- `texto_emb`: texto (Whisper), embeddings.
- `texto_lexico`: texto (Whisper), 7 features léxicas.

El participante 66 no tiene audio editado: sus modelos de voz quedan vacíos y la fusión usa sus demás modelos.

### XAI sobre embeddings (`../../Psiquiatria/Multimodal_XAI/xai_embeddings.py`)

| Archivo | Contenido |
|---|---|
| `probing_embeddings_<eje>.csv` | Método 3 (Dixit et al. 2024). Por feature interpretable (eGeMAPS o léxico): R² de predecirla desde todas las dimensiones del embedding, desde las 50 que más usa el clasificador y desde 50 al azar |
| `probing_resumen_<eje>.csv` | Lo mismo resumido por categoría (frecuencia, energía, espectral, temporal, léxico) |
| `atribucion_temporal_<eje>.csv` | Método 5. Por participante y fragmento de la entrevista (marcas de tiempo de Whisper): cuánto cambia el riesgo al quitar ese fragmento, por rama (voz_microventanas, egemaps, texto_emb). Modelos que no vieron al participante |
| `atribucion_temporal_<eje>.md` | Ejemplos legibles: los fragmentos que más subieron el riesgo en los positivos de mayor riesgo. **Contiene frases de las entrevistas: no compartir** |

## Viejos (v4 del repo Psiquiatria, solo depresión)

`oof_matrix_final.csv`, `oof_matrix_detail_by_repeat.csv`, `shap_matrix.csv`, `lime_matrix.csv`. Tienen los problemas descritos en `../auditoria_psiquiatria/AUDITORIA.md`: destilación con fuga, `n_frames_detected` y embeddings no reproducibles. Se conservan como referencia. `ANALISIS_titulo_articulo.md` analizó estos archivos viejos.

## Para leer los AUC
- **AUTO es la cifra honesta** de "quedarse con la mejor combinación".
- El máximo de la tabla de combinaciones es optimista.
- `voz_microventanas` depende del punto de muestreo: no es robusta.
- La etiqueta es "DX IA": tamizaje PHQ-9 / GAD-7 con corte en síntomas leves (≥ 5); coincide con las bandas del cuestionario en el 91–93% de los participantes.
