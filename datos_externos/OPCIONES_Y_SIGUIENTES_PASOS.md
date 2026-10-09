# Ampliación de datos y validación externa: opciones y siguientes pasos

*Versión del 2026-10-05. Documento de trabajo; las decisiones marcadas como "propuesta" están pendientes de acordar.*

---

## 1. Motivación

El mejor modelo honesto con los datos de Samaná ronda un AUC de 0.62–0.65 en depresión y de 0.55–0.61 en ansiedad (ver `experimento_embeddings/RESULTADOS.md`). La pregunta es si incorporar datos externos (de Virtualmente o públicos) puede darle más validez estadística a la tesis.

Hay que separar dos problemas distintos:

1. **Precisión.** Con 79 participantes y unos 20 positivos, el intervalo de confianza del AUC es muy ancho (§4).
2. **Generalización.** Todo viene de un solo sitio y una sola población, sin validación externa.

Un N de 79 no invalida la tesis por sí solo. En el área es habitual: el conjunto de evaluación de DAIC-WOZ tiene 35–47 personas, D3TEC 62, DEPTALK 101 y ANDROIDS 118. Lo que sí limita es qué se puede afirmar. Las opciones de este documento se evalúan según cuál de los dos problemas atacan.

---

## 2. Hallazgo previo: origen de la etiqueta "DX IA"

### 2.1 La etiqueta es el tamizaje PHQ-9/GAD-7

`Base de Datos Trabajo de campo RESULTADOS.xlsx` (abril de 2026) es el archivo de enero (`Base de datos FINAL Trabajo de campo reconocimeineto de voz .xlsx`) con dos columnas añadidas: `DX DEPRESIÓN IA` y `DX DANSIEDAD IA`.

Las categorías de DX IA corresponden a las bandas estándar de severidad (Likert 0–3): <5 sin riesgo, 5–9 leve, 10–14 moderado.

| Comparación de DX IA contra… | Banda exacta (dep / ans) | Binario con umbral ≥5 (dep / ans) | r punto-biserial (dep / ans) |
|---|---|---|---|
| Puntajes del Excel de **enero** | 0.85 / 0.89 | **0.91 / 0.93** | **0.82 / 0.78** |
| Puntajes del Excel de **abril** | 0.57 / 0.71 | 0.58 / 0.72 | −0.04 / 0.15 |

Las ~10 discrepancias con enero están todas en la frontera (puntajes 4–7). Los 6 participantes sin cuestionario (códigos 5, 10, 19, 21, 30 y 38) quedaron como "Sin riesgo".

### 2.2 El Excel de abril tiene el bloque de cuestionarios desfasado

En RESULTADOS, las columnas desde `Frecuencia al beber` hasta `SDS_7` (alcohol, PHQ-9, GAD-7 y SDS) están corridas respecto a `CÓDIGO`. Alguien insertó celdas vacías solo en ese rango:

- el código 13 contiene las respuestas del 12, el 24 las del 22, …, y el 78 las del 73;
- quedan 5 filas sobrantes al final, sin código.

La demografía, la cédula y DX IA siguen alineadas con el código.

Hay una confirmación independiente. Una versión de noviembre (`investigacion/Muestras Samaná Noviembre 2025/Trabajo de campo Muestras.xlsx`, 54 participantes) coincide con el Excel de enero en el bloque de cuestionarios en **52/54** participantes, y con el de abril solo en **13/54**.

### 2.3 Consecuencias

- **Las etiquetas binarias con las que se entrenaron todos los modelos son correctas** (DX IA por código).
- **Quedan invalidados los análisis que usaron puntajes del Excel de abril:**
  - `features/puntajes_phq_gad.csv`;
  - `check_label_quality.py`;
  - la evaluación "contra el cuestionario validado" (`RESULTADOS.md` §14);
  - la afirmación de que "DX IA no se relaciona con el PHQ-9".

  Hay que rehacerlos con el Excel de enero (74/80 participantes con respuestas).
- La etiqueta queda definida como **tamizaje con instrumentos validados, con corte en síntomas leves (≥5)**, no como diagnóstico clínico.
- **Pendiente:** confirmar con el equipo de campo que el archivo de enero es el original, y qué criterio se aplicó en los casos de frontera.

---

## 3. Proyecto Virtualmente (Universidad de Caldas)

Lo que se puede verificar públicamente:

- **Nombre y financiación:** proyecto "Cabina Móvil VirtualMente", convocatoria MinCiencias 918-2022. Presupuesto total de 2.700 millones de pesos.
- **Ejecutores:** Facultad de Ciencias para la Salud, Corporación IPS Universitaria de Caldas y el Grupo de Investigación Telesalud. Coinvestigadora: Claudia Patricia Jaramillo Ángel.
- **Componentes:** cabinas móviles y teleconsultorios en Samaná, Riosucio y Manizales; alfabetización en salud mental; tamizaje con inteligencia artificial y alertas a especialistas; teleorientación psicológica.
- **Clausura (agosto de 2025), resultados anunciados:** 238 jóvenes de 18 a 28 años en 36 meses; 56% de mejoría en síntomas de depresión; 50% de reducción de ansiedad; **"92% de coincidencia diagnóstica entre inteligencia artificial y valoración clínica"**.
- No se encontró una publicación científica del proyecto con detalle metodológico.

Implicaciones para la tesis:

1. El 92% anunciado coincide con el 91–93% de §2.1. Sugiere que "DX IA" es el tamizaje automatizado de la cabina, y que **existe una valoración clínica separada**. Si la hay para los 80 participantes, sería una etiqueta de referencia mejor que el autoinforme.
2. Los 238 participantes del proyecto son más que los 80 grabados (noviembre–diciembre de 2025, después de la clausura). Hay que preguntar si alguno más tiene grabaciones.

### 3.1 Datos locales sin usar: prueba piloto (40 participantes)

- **Grabaciones:** `analisis-espectrogramas/.MUESTRAS_SAMANÁ/Pruebas protocolo/Muestras/` tiene **40 participantes distintos de los 80** (ninguna cédula en común), con mp4, wav y m4a, grabados entre el 6 y el 14 de noviembre de 2025.
- **Demografía:** su base (`investigacion/Muestras prueba piloto/BASE DE DATOS Grabación Muestras Protocolo.xlsx`) indica 17–28 años (mediana 21), 21 mujeres y 19 hombres, de varias regiones (Manizales 10, Nariño, Salamina, Ibagué, etc.).
- **Sin cuestionarios:** esa base **no tiene PHQ-9 ni GAD-7**. Si el equipo los tiene, el N subiría a 119.
- **Advertencias:** es otra campaña y otra población (universitarios de varias regiones, no jóvenes rurales de Samaná), y el rostro ya distingue campañas (AUC 0.77). Por eso conviene más usarla como **conjunto de validación externa** que mezclarla con los 79.

---

## 4. Precisión estadística con el N actual

Intervalo de confianza del 95% del AUC (Hanley y McNeil, 1982):

| AUC observado | Depresión (21 positivos / 58 negativos) | Ansiedad (19 / 60) |
|---|---|---|
| 0.60 | [0.45, 0.75] | [0.45, 0.75] |
| 0.65 | [0.51, 0.79] | [0.50, 0.80] |
| 0.70 | [0.56, 0.84] | [0.56, 0.84] |

N necesario para detectar un AUC distinto de 0.5 con potencia de 80% (α = 0.05, prevalencia de 26%):

| AUC verdadero | N total | Positivos |
|---|---|---|
| 0.60 | ≈ 343 | 89 |
| 0.65 | ≈ 152 | 40 |
| 0.70 | ≈ 83 | 22 |

Con el N actual solo se detectarían efectos de AUC ≥ 0.70. Agregando el piloto (N = 119), el intervalo para un AUC de 0.65 se estrecha de ±0.14 a ±0.12.

---

## 5. Datasets externos: inventario y adecuación

| Dataset | Idioma / contexto | N | Etiqueta | Modalidades | Acceso | Adecuación |
|---|---|---|---|---|---|---|
| **D3TEC** (México, 2024) | **Español**; preguntas abiertas, lectura y descripción de imágenes; micrófono profesional + **iPhone** | 62 | PHQ-9 (el artículo binariza en ≥10) | Solo audio | **Abierto, CC-BY** (IEEE DataPort) | **Alta para voz-depresión.** Mismo idioma, celular, solo la voz del participante |
| **E-DAIC** (EE. UU., AVEC 2019) | Inglés; entrevista semiestructurada con agente virtual | 275 | PHQ-8 (= PHQ-9 sin el ítem 9) | Audio, transcripciones, features de rostro OpenFace (AU, mirada, pose); sin video crudo | EULA académica (la firma un profesor); sin redistribución | **Alta para rostro-depresión.** Mismo tipo de contexto (entrevista pasiva) y el mismo instrumento |
| DAIC-WOZ | Inglés; entrevistador controlado por humano | 189 | PHQ-8 | Igual que E-DAIC | EULA académica | Subconjunto de lo anterior |
| **Bridge2AI-Voice** v3.1 (2026) | Inglés; vocales sostenidas, oraciones, tareas cognitivas (sin habla libre) | 833 adultos | **PHQ-9 y GAD-7** | Solo features (openSMILE, Praat, espectrogramas) | Credencial PhysioNet + DUA | Baja–media. **Única opción pública con GAD-7**, pero tareas y cohortes clínicas muy distintas |
| D-Vlog / LMVD | Vlogs de YouTube y redes chinas | 816 / 1475 | Débil (por título o contenido) | Features audiovisuales | Por solicitud / GitHub | Baja. La etiqueta no es un instrumento clínico |
| AVEC 2013/14 | Alemán; tareas en laboratorio | ~150 | BDI-II | Video y audio | Incierto hoy | Baja |
| DEPTALK (España, 2025) | **Español**; conversación casual con humanos virtuales | 101 | PHQ-9 | Audio, texto (¿video?) | **Privado** | Solo vía colaboración con UPV Valencia |
| DEPAC | Inglés; mTurk | 571 | PHQ-9 y GAD-7 | Audio | **Propietario** | No disponible |

Resumen:

- **Depresión:** hay dos opciones viables y complementarias, D3TEC para la voz y E-DAIC para el rostro.
- **Ansiedad:** no existe un dataset público con GAD-7 en contexto conversacional y con audio o video. Bridge2AI-Voice es lo único disponible y su contexto es muy distinto. La ansiedad seguirá dependiendo de los datos de Samaná y debe declararse como limitación.

---

## 6. Qué rendimiento esperar: evidencia de la literatura

- **Transferencia facial entre corpus** (Sadeghi et al., 2026), entre E-DAIC (N = 275) y un ensayo clínico alemán (N = 256), con features de OpenFace 2.2:
  - la clasificación transfiere con **AUC 0.65–0.70**;
  - la regresión de severidad transfiere peor (CCC 0.21–0.26), e incluso dentro de E-DAIC el CCC es de apenas 0.08;
  - los contextos pasivos (como una entrevista) transfieren mejor que las tareas activas;
  - los marcadores estables entre corpus son **la mirada (gaze)**, la entropía de **AU15**, la menor variabilidad del movimiento de cabeza y **AU04**. AU15 y AU04 ya aparecen en nuestros SHAP de ansiedad, y la pose de cabeza fue el núcleo estable en el v3 de video.
- **Voz con modelos fundacionales** (Gómez-Zaragozá et al., 2025): con 101–459 participantes, el F1 de depresión queda en 0.55–0.75. En DEPTALK (español) la voz da 0.61 y voz+texto 0.69. Los autores señalan que muchos resultados de DAIC-WOZ se calculan sobre 35 personas.
- **Entre idiomas** (Pattanayak et al., 2026), de inglés a mandarín: F1 ≈ 0.64. Un F1 previo de 0.954 estaba inflado por fuga del hablante.
- **Auditoría de benchmarks** (Ishikawa y Duke, 2026): la transferencia sin reentrenar a corpus externos es mucho más débil que el rendimiento dentro del mismo corpus, y los rankings de los leaderboards son inestables.

**Conclusión.** Ningún dataset externo va a llevar el AUC a 0.85. Nuestro rango de 0.62–0.65 es coherente con lo que reporta el área cuando la evaluación es honesta. Lo que sí aportan los datos externos es **validación externa**: mostrar que lo aprendido en Samaná se replica en otra población, o que los marcadores coinciden. Eso responde mejor a la objeción del tamaño muestral que un AUC más alto.

---

## 7. Opciones evaluadas

| # | Opción | Ataca | Costo | Dependencia | Valor esperado |
|---|---|---|---|---|---|
| A1 | Rehacer puntajes con el Excel de enero y **entrenar contra el puntaje continuo** (PHQ-9 0–27, GAD-7 0–21) | Precisión: usa más información que el binario con ~20 positivos | Bajo | Ninguna | Medio |
| A2 | **Intervalos de confianza + curvas de aprendizaje** de los modelos actuales | Rigor: muestra si más datos ayudarían | Bajo | Ninguna | Alto para la defensa |
| A3 | **D3TEC**: validación externa de la voz (Samaná→D3TEC y D3TEC→Samaná) y voz agrupada en español (N ≈ 141) | Generalización y precisión (voz-depresión) | Medio | Ninguna (es abierto) | Medio. Nuestra voz ya era débil |
| A4 | **Piloto (40)** como validación externa, si hay PHQ-9/GAD-7 | Generalización, ambos ejes, mismo idioma y protocolo | Medio (extraer features) | **Equipo Virtualmente** | **Alto** |
| A5 | **Valoración clínica** de los 80 como etiqueta alternativa | Validez de la etiqueta | Bajo | **Equipo Virtualmente** | **Alto** |
| A6 | **E-DAIC + OpenFace 2.2**: (i) aplicar a Samaná sin reentrenar; (ii) su probabilidad como feature en la CV anidada; (iii) Samaná→E-DAIC; (iv) SHAP comparado entre poblaciones | Generalización (rostro-depresión) y aporte de XAI | Alto (EULA + extracción) | Firma del director | Medio–alto |
| A7 | Agregar features de **mirada (gaze)** de OpenFace en Samaná | Señal nueva (el marcador más transferible) | Medio | Ninguna | Medio |
| A8 | Bridge2AI-Voice para voz-ansiedad | Ansiedad | Alto | Credencial PhysioNet | Bajo |
| A9 | Contactar a los autores de DEPTALK | Voz y texto en español | Incierto | Terceros | Bajo–medio |

Todas las opciones respetan las reglas ya fijadas:
- modelos independientes por eje, sin multitarea;
- AUC estándar sobre todas las muestras;
- protocolo declarado antes de ver resultados, al estilo de `experimento_embeddings/INVENTARIO_Y_PROTOCOLO.md`.

---

## 8. Siguientes pasos (propuesta)

**Fase 1: inmediato, sin depender de terceros**
1. Recalcular PHQ-9, PHQ-8 y GAD-7 desde el Excel de enero, y regenerar `features/puntajes_phq_gad.csv`.
2. Rehacer los análisis invalidados en §2.3.
3. A1: modelos con objetivo continuo, evaluados con AUC contra DX IA.
4. A2: intervalos de confianza (bootstrap estratificado) y curvas de aprendizaje de los modelos de la plataforma.
5. A3: descargar D3TEC y declarar el protocolo de validación externa de la voz.
6. Instalar OpenFace 2.2 y extraer AU, mirada y pose de los 80 participantes y de los 40 del piloto (sirve para A4, A6 y A7).

**Fase 2: preguntas al equipo de Virtualmente** (director, Claudia Jaramillo o el equipo de campo)
1. ¿Los 40 participantes de la prueba piloto respondieron PHQ-9/GAD-7? ¿Con qué protocolo de grabación?
2. ¿Existe una valoración clínica (psicólogo o psiquiatra) para los 80? ¿Es la "valoración clínica" del 92%?
3. ¿Cómo se generó DX IA y qué criterio se usó en los casos de frontera? ¿El archivo de enero es el original?
4. ¿Algún otro de los 238 participantes del proyecto tiene grabaciones de voz o video?
5. ¿Qué permite el consentimiento informado sobre publicación y combinación con datos externos?

**Fase 3: E-DAIC** (en paralelo; la aprobación tarda semanas)
1. Solicitar el acceso con correo institucional. La EULA la firma el director.
2. Declarar el protocolo de A6 antes de abrir los datos de Samaná con el modelo externo.

**Criterio de éxito para la tesis.** No se busca un AUC alto, sino al menos una de estas dos cosas:
- (a) que el modelo se sostenga en un conjunto externo (piloto, D3TEC o E-DAIC) con un intervalo que excluya 0.5;
- (b) que los marcadores explicados (SHAP) coincidan entre poblaciones.

Cualquiera de las dos convierte "0.65 en 79 personas de un solo sitio" en evidencia replicada.

---

## Referencias

- Gómez-Zaragozá, L., Marín-Morales, J., Alcañiz, M., y Soleymani, M. (2025). Speech and text foundation models for depression detection: Cross-task and cross-language evaluation. *Proc. Interspeech 2025*, 5253–5257. https://doi.org/10.21437/Interspeech.2025-1035
- Gratch, J., Artstein, R., Lucas, G., et al. (2014). The Distress Analysis Interview Corpus of human and computer interviews. *Proc. LREC 2014*, 3123–3128.
- Hanley, J. A., y McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology, 143*(1), 29–36. https://doi.org/10.1148/radiology.143.1.7063747
- Ishikawa, T., y Duke, J. (2026). A multi-probe audit of clinical-interview depression detection benchmarks. *arXiv:2605.23977*.
- LMVD (2025). A large-scale multimodal vlog dataset for depression detection in the wild. *Information Fusion, 126*, 103632. https://doi.org/10.1016/j.inffus.2025.103632
- Pattanayak, A., Kang, H., Chou, H.-C., Narayanan, S., y Kadiri, S. R. (2026). Layer-wise cross-lingual depression detection from speech: Analysis with contrastive alignment. *arXiv:2607.02920*.
- Protocolo D3TEC (2025). Data collection for automatic depression identification in Spanish speakers using deep learning algorithms: Protocol for a case-control study. *JMIR Research Protocols*. https://doi.org/10.2196/60439. Dataset: https://ieee-dataport.org/documents/d3tec-dataset
- Ringeval, F., Schuller, B., Valstar, M., et al. (2019). AVEC 2019 workshop and challenge: State-of-mind, detecting depression with AI, and cross-cultural affect recognition. *Proc. 9th AVEC Workshop*, 3–12.
- Sadeghi, M., Richer, R., Rupp, L. H., et al. (2026). Do depressive facial patterns transfer across cultures and contexts? Evidence from a German RCT and E-DAIC. *arXiv:2609.05543*.
- Tasnim, M., Ehghaghi, M., Diep, B., y Novikova, J. (2022). DEPAC: A corpus for depression and anxiety detection from speech. *Proc. CLPsych 2022*. arXiv:2306.12443.
- Bridge2AI-Voice (2026). An ethically-sourced, diverse voice dataset linked to health information, v3.1.0. PhysioNet. https://physionet.org/content/b2ai-voice/
- Universidad de Caldas (2023). Telesalud y Universidad de Caldas ganan convocatorias de MinCiencias… https://www.ucaldas.edu.co/portal/telesalud-y-universidad-de-caldas-ganan-convocatorias-de-minciencias-para-llevar-salud-tecnologia-y-realidad-virtual-a-zonas-rurales-de-caldas/
- Universidad de Caldas (2025). Universidad de Caldas clausura en Samaná el proyecto Cabina Móvil VirtualMente. https://www.ucaldas.edu.co/portal/universidad-de-caldas-clausura-en-samana-el-proyecto-cabina-movil-virtualmente/
- DAIC-WOZ / E-DAIC: https://dcapswoz.ict.usc.edu/
