# Reporte de resultados — clasificación de riesgo SOLO con video

**Fecha:** 2026-06-02 · **Datos:** 80 clips (cara frontal, 1080×1920, 30 fps), **los 80 con etiqueta**.
**Tarea:** clasificación binaria de riesgo (Sin Riesgo vs Con Riesgo), **ansiedad y depresión por separado**.
**Validación:** RepeatedStratifiedKFold 5×5 (25 folds), a nivel de participante. **Solo video** (sin audio).
**Fuente de etiquetas:** `…/.MUESTRAS_SAMANÁ/Base de Datos Trabajo de campo RESULTADOS.xlsx` (columnas `DX DEPRESIÓN IA`, `DX DANSIEDAD IA`).

---

## ⚠️ Corrección importante respecto a la versión anterior

La versión previa de este reporte decía "11 participantes sin etiqueta → se entrenó con 69". **Eso era un bug, ya corregido.** El cruce se hacía por **cédula (documento)**, y para **11 participantes la cédula del nombre de la carpeta NO coincide con la del Excel** (errores de digitación; p. ej. carpeta `033_<cédula>` vs Excel `…742`; `040_…441` vs `…447`). La llave correcta es el **CÓDIGO de participante (1–80)**, presente en ambos lados. Cruzando por código, **los 80 quedan etiquetados** (ansiedad 61/19, depresión 59/21).

> **Acción sugerida para ti:** revisar esas 11 cédulas discrepantes (¿está mal el nombre de la carpeta o el Excel?). No afecta el entrenamiento (se usa el código), pero importa para trazabilidad y para cruzar consentimientos.

---

## TL;DR (resultado honesto, ya con etiquetas correctas)

> **El video solo carga una señal DÉBIL pero real (AUC ≈ 0.60) para ambos ejes** — _cuando_ se reduce la dimensionalidad. No es suficiente como detector autónomo, pero no es ruido.
>
> - **Mejor configuración (6 features clínicas + LogReg):** Ansiedad **AUC 0.62**, Depresión **AUC 0.60**.
> - Con las 48 features, el desempeño baja (ansiedad ~0.57, depresión ~0.52) por **sobreajuste/maldición de la dimensionalidad** (48 features vs solo 19–21 positivos).
>
> Conclusión: hay algo de señal en el comportamiento facial, pero es modesta. Esto **justifica con fuerza la fusión multimodal + selección de features**, no el uso de video solo.

---

## 1. Resultados cuantitativos

Línea base por mayoría (accuracy trivial): **ansiedad 0.76**, **depresión 0.74**. La métrica clave con clases desbalanceadas es **AUC** (azar = 0.50).

### Ansiedad (n=80; 61 sin riesgo / 19 con riesgo)

| Modelo                        | AUC       | F1-macro | Accuracy |
| ----------------------------- | --------- | -------- | -------- |
| Random Forest (48 feats)      | 0.572     | 0.447    | 0.760    |
| XGBoost (48 feats)            | 0.517     | 0.513    | 0.718    |
| SVM-RBF (48 feats)            | 0.575     | 0.537    | 0.703    |
| **LogReg (6 feats clínicas)** | **0.616** | 0.531    | —        |

### Depresión (n=80; 59 sin riesgo / 21 con riesgo)

| Modelo                        | AUC       | F1-macro | Accuracy |
| ----------------------------- | --------- | -------- | -------- |
| Random Forest (48 feats)      | 0.528     | 0.456    | 0.735    |
| XGBoost (48 feats)            | 0.513     | 0.511    | 0.693    |
| SVM-RBF (48 feats)            | 0.506     | 0.497    | 0.632    |
| **LogReg (6 feats clínicas)** | **0.599** | 0.501    | —        |

> Las 6 features clínicas usadas — ansiedad: `blink_rate_min, gaze_aversion_frac, eye_squint_mean, head_motion_std, mouth_press_mean, brow_down_mean`; depresión: `expresividad_mean, smile_mean, frown_mean, gaze_down_mean, head_motion_mean, blink_rate_min`.

## 2. Hallazgo metodológico: confound de duración (removido)

`dur_min`, `n_frames_proc` y `blink_count` aparecían como "importantes", pero **los "con riesgo" tienen clips más largos** (4.5 vs 3.9 min): es un artefacto de grabación, no un biomarcador. Se eliminaron esas variables (y las de calidad de detección) antes de entrenar. Todos los números de arriba ya están **sin ese confound**.

## 3. Qué dice el SHAP (interpretabilidad)

Las features que más pesan son **clínicamente plausibles** (el pipeline captura los conceptos correctos):

- **Depresión:** `eye_squint_std`, `frown_mean/std` (ceño/mueca), `brow_inner_up` (AU1, elevación interna de cejas → tristeza), `brow_down`, `gaze_up_std`. → marcadores de tristeza/afecto negativo.
- **Ansiedad:** `frown_mean`, `nose_sneer`, `brow_down` (ceño/tensión), `head_still_frac` y `pitch_mean` (inquietud de cabeza), `eye_squint`. → marcadores de tensión/estrés.

Ver `shap_ansiedad.png`, `shap_depresion.png`, `shap_importancia_*.csv`. El modelo "mira lo correcto"; el límite es la **magnitud** de la señal, no su dirección.

## 4. Por qué sigue siendo débil (limitaciones)

1. **Pocos positivos** (19–21) → poca potencia; con 48 features hay sobreajuste (por eso 6 features rinden mejor).
2. **Agregación por clip (media/desviación sobre ~4.4 min)** diluye expresiones transitorias.
3. **Tarea de monólogo:** caras relativamente neutras entre grupos.
4. **Manos casi nunca visibles (4.5%)** → micro-gestos corporales aportan poco en video selfie vertical.
5. **Etiqueta binaria auto-reportada** (PHQ/GAD) como verdad de terreno: ruidosa.
