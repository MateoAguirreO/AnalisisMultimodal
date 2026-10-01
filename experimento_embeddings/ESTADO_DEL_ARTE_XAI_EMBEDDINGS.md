# Explicabilidad (XAI) sobre embeddings: estado del arte y aplicación a la tesis

Fecha: 2026-09-30. Revisión dirigida (no sistemática) para responder: **¿cómo se explica un modelo que decide con embeddings** (wav2vec2 / WavLM para voz, sentence-transformers para texto), donde cada dimensión no significa nada por sí sola?

## 1. El problema

En un modelo sobre eGeMAPS o sobre AU, SHAP y LIME dan una lectura directa: "más variabilidad de `logRelF0-H1-H2` sube el riesgo". En un embedding de 1024 dimensiones, un SHAP por dimensión ("la dimensión 517 subió el riesgo") no significa nada para un clínico.

La literatura resuelve esto **cambiando el objeto que se explica**:
- la modalidad;
- la capa;
- conceptos humanos;
- features interpretables que el embedding codifica;
- momentos de la grabación;
- tokens del texto.

## 2. Taxonomía de métodos

| # | Familia | Qué responde | Referencias clave | ¿Aplica a nuestro pipeline? |
|---|---|---|---|---|
| 1 | **Contribución de modalidad** en la fusión | ¿Cuánto pesó la voz frente al rostro en *este* paciente? | Práctica común en fusión tardía; descomposición exacta del soft vote | **Hecho** (`contribucion_modelos_<eje>.csv`) |
| 2 | **Probing / análisis por capas**: sondas que predicen propiedades conocidas desde cada capa | ¿Qué información hay en qué capa? | Pasad et al. 2021 y 2023; Li et al. 2022 | Sí: justifica qué capa usar (Psiquiatria usaba la última) |
| 3 | **Explicar el embedding prediciendo features interpretables** (probing dirigido a la tarea) | ¿Qué propiedades acústicas usa el clasificador de embeddings? | **Dixit et al. 2024** (WavLM → eGeMAPS) | **Sí, recomendado** (versión rigurosa de nuestro "proxy") |
| 4 | **Conceptos (TCAV / CAV)**: direcciones del espacio latente para conceptos humanos | ¿Cuánto depende la decisión de "voz monótona", "habla lenta"…? | Kim et al. 2018; Asokan et al. 2022 (IEMOCAP multimodal); auditoría de sesgos en habla 2026 | Sí, con conceptos definidos a partir de eGeMAPS o de anotación |
| 5 | **Atribución temporal** (oclusión de segmentos / MIL) | ¿Qué *momentos* de la entrevista movieron la predicción? | Análisis de atención por enunciado en detección de depresión (2026); oclusión clásica | **Sí, muy útil**: alineable con las transcripciones de Whisper |
| 6 | **Gradientes (Integrated Gradients, saliencia)**, en el embedding o hasta la señal | ¿Qué subespacio del embedding o qué región tiempo-frecuencia importa? | Sundararajan et al. 2017; **Grósz et al. 2023** (IG sobre embeddings de wav2vec2 y BERT) | Parcial: requiere clasificador diferenciable (logística en torch) |
| 7 | **Features intrínsecamente interpretables** (autoencoders dispersos, SAE) | ¿Qué "detectores" aprendió el modelo? | **AudioSAE** (Aparin et al. 2026, Whisper/HuBERT) | Frontera; fuera del alcance (requiere entrenar SAE a gran escala) |
| 8 | **Pesos de atención** como explicación | ¿Dónde "miró" el modelo? | Jain y Wallace 2019 frente a Wiegreffe y Pinter 2019 | Con cautela: la atención no es explicación fiel por sí sola. El attention pooling de Psiquiatria (aleatorio y sin entrenar) no explica nada |
| 9 | **Texto: atribución a tokens y embeddings interpretables** | ¿Qué palabras o frases pesaron? | SHAP e IG para transformers; revisión de Opitz et al. 2025 (EMNLP) | Sí: SHAP de partición o IG sobre el encoder de frases, o léxico interpretable (hecho) |

### 2.1 Probing por capas (familia 2)

- **Pasad et al. (2021, 2023)**: en wav2vec2 y otros modelos auto-supervisados (SSL), las capas tempranas codifican señales acústicas de bajo nivel, las intermedias información fonética y las altas información léxica y semántica.
- **Li et al. (SLT 2022)**: con correlación canónica entre capas de wav2vec2 y eGeMAPS encontraron que el modelo "parece descartar información paralingüística menos útil para reconocer palabras". Las capas intermedias rinden tan bien como el promedio de capas para emoción, y **la última capa es la peor en algunos casos**.

**Para nosotros:** la representación de Psiquiatria usa la **última capa**, justo la que menos información paralingüística retiene según esta literatura. Es coherente con que su señal sea frágil.

### 2.2 Explicar embeddings prediciendo features interpretables (familia 3), recomendada

**Dixit, Low, Elbanna, Catania y Ghosh (2024)** proponen un "probing modificado" en tres pasos:
1. Identifican las dimensiones del embedding (WavLM) más importantes para cada emoción.
2. Predicen features interpretables (eGeMAPS: F0, volumen, espectrales, temporales) con **todas** las dimensiones y con **solo las importantes**.
3. Las features que se predicen igual de bien con el subconjunto importante son las que el modelo realmente usa.

**Resultado:** para emoción, las categorías energía, frecuencia, espectral y temporal aportan en ese orden decreciente.

**Nuestro "proxy" actual** (correlación entre la predicción del modelo de embeddings y las eGeMAPS) es una versión simple de esta idea. La versión rigurosa daría frases del tipo *"el modelo de voz se apoya en dimensiones que codifican la dinámica del volumen y los formantes"*, citables y comparables con la literatura.

### 2.3 Conceptos, TCAV (familia 4)

**Kim et al. (ICML 2018)** proponen representar un concepto humano como una dirección del espacio latente. Esa dirección se aprende con una regresión lineal que separa ejemplos con y sin el concepto. Después se mide la sensibilidad de la predicción a esa dirección.

Hay aplicaciones en:
- emoción multimodal (audio, texto y video en IEMOCAP; Asokan et al. 2022);
- auditoría de sesgos en evaluación del habla (2026) y en embeddings de música (2025).

**Para nosotros**, los conceptos podrían definirse con datos, sin anotadores. Por ejemplo, segmentos del cuartil alto frente al bajo de variabilidad de F0 ("voz monótona"), de volumen ("voz débil") o de jitter/shimmer ("voz inestable"). El resultado sería una frase del tipo *"la predicción de riesgo es sensible al concepto 'voz monótona' (puntaje TCAV = …)"*.

### 2.4 Atribución temporal (familia 5)

Detectores de depresión basados en SSL y atención por enunciado reportan más atención en los enunciados con indicios de varios síntomas depresivos (trabajo de 2026, guiado por síntomas).

**Aplicación directa a nuestro caso**, sin atención entrenada:
1. Por **oclusión**, quitar cada micro-ventana o segmento del promedio del participante.
2. Medir cuánto cambia la probabilidad.
3. **Alinear ese momento con el texto transcrito por Whisper** (tenemos las marcas de tiempo).

El resultado es una explicación legible para un clínico: *"el riesgo subió principalmente en el minuto 2:10, cuando dijo '…'"*. Además une voz y texto en la misma explicación.

### 2.5 Gradientes (familia 6)

**Grósz et al. (ACM MM 2023)** usaron Integrated Gradients sobre embeddings de BERT, wav2vec2, ELECTRA y ViT para **descubrir el subespacio relevante** de cada tarea. Con solo esas dimensiones entrenaron modelos más pequeños (~54% menos parámetros) con resultados iguales o mejores. Es una forma de reducir un embedding a sus dimensiones útiles antes de explicarlas (familia 3).

### 2.6 Riesgo específico de los embeddings: confusores (familia transversal)

**Yeh et al. (Interspeech 2026)** muestran que en detección de depresión por voz "las features relacionadas con depresión extraídas por los modelos actuales están fuertemente entrelazadas con la identidad del hablante". Con hablantes compartidos entre entrenamiento y prueba el rendimiento sube; con hablantes nuevos cae fuerte.

Esto coincide con nuestros hallazgos: los embeddings del audio crudo codificaban al entrevistador y la campaña de grabación. **Una sección de XAI sobre embeddings debe incluir estos controles**: qué tanto codifica el embedding el hablante, el sexo o la sesión, para mostrar que la explicación no descansa en un confusor.

## 3. Contexto de rendimiento en la literatura

**Maran et al. (JMIR Mental Health 2025)**, revisión sistemática y metaanálisis de 105 estudios de detección de depresión por voz:
- exactitud entre 0.66 y 0.81;
- heterogeneidad enorme (I² = 94–99%);
- 47.6% de los estudios con alto riesgo de sesgo;
- una advertencia explícita: *"small training sets may lead to inflated accuracy estimates"*;
- predominan etiquetas por autorreporte y no diagnóstico clínico.

Nuestro AUC honesto (~0.56–0.65 con N ≈ 80) no está "mal" frente a la literatura. Es lo que se espera cuando se evalúa sin las fuentes típicas de inflación.

## 4. Recomendación para la tesis, ordenada por valor y esfuerzo

1. **Ya hecho:**
   - contribución de modalidad por participante (familia 1);
   - SHAP y LIME sobre las ramas interpretables (AU, eGeMAPS, léxico);
   - proxy de correlación de los embeddings.
2. **Siguiente (~1–2 h), probing dirigido a la tarea al estilo Dixit et al. (2024)** sobre los modelos de voz de embeddings:
   - dimensiones importantes para el clasificador;
   - qué eGeMAPS codifican esas dimensiones.

   Convierte el "proxy" en un método publicado y citable.
3. **Muy recomendable (~2 h), atribución temporal por oclusión + alineación con la transcripción de Whisper.** Explica qué momentos de la entrevista movieron la predicción y qué dijo la persona en ese momento. Encaja con la plataforma y con el informe PDF (el loop jr/senior podría leerlo).
4. **Opcional:** TCAV con conceptos definidos a partir de eGeMAPS ("voz monótona", "voz débil", "voz inestable").
5. **Trabajo futuro:** autoencoders dispersos (AudioSAE) e Integrated Gradients de extremo a extremo sobre la señal.
   - Entrenar un SAE propio sí queda fuera de alcance: requiere millones de frames de corpus grandes y diversos; con 3.6 h de 79 personas aprendería las particularidades de la muestra.
   - **Pero AudioSAE publicó SAE ya entrenados** (licencia MIT; https://github.com/audiosae/audio-sae) para HuBERT-base/large y Whisper-small/large-v3/**large-v3-turbo**, que es el que usamos para transcribir.
   - Usarlos es factible como análisis exploratorio. El costo está en **interpretar** las features: 10.240 por capa, y el repositorio solo da herramientas (clips más activadores, espectrogramas), no etiquetas.
   - Con N = 79, buscar qué features distinguen a los grupos entre miles implica un problema serio de comparaciones múltiples.
6. **En la discusión:**
   - la atención no es explicación por sí sola (Jain y Wallace; Wiegreffe y Pinter);
   - la última capa de wav2vec2 pierde información paralingüística (Li et al.);
   - los embeddings de voz se entrelazan con el hablante y la sesión (Yeh et al.; nuestros resultados).

## Referencias

- Aparin, G. et al. (2026). *AudioSAE: Towards Understanding of Audio-Processing Models with Sparse AutoEncoders.* EACL 2026. https://arxiv.org/abs/2602.05027
- Asokan, A. R., Kumar, N., Ragam, A. V., & Sharath, S. S. (2022). *Interpretability for Multimodal Emotion Recognition using Concept Activation Vectors.* arXiv:2202.01072. https://arxiv.org/abs/2202.01072
- Dixit, S., Low, D. M., Elbanna, G., Catania, F., & Ghosh, S. S. (2024). *Explaining Deep Learning Embeddings for Speech Emotion Recognition by Predicting Interpretable Acoustic Features.* arXiv:2409.09511. https://arxiv.org/abs/2409.09511
- Dumpala, S. H. et al. (2024). *Self-Supervised Embeddings for Detecting Individual Symptoms of Depression.* Interspeech 2024. https://arxiv.org/abs/2406.17229
- Grósz, T., Virkkunen, A., Porjazovski, D., & Kurimo, M. (2023). *Discovering Relevant Sub-spaces of BERT, Wav2Vec 2.0, ELECTRA and ViT Embeddings for Humor and Mimicked Emotion Recognition with Integrated Gradients.* ACM Multimedia 2023. https://dl.acm.org/doi/10.1145/3606039.3613102
- Jain, S., & Wallace, B. C. (2019). *Attention is not Explanation.* NAACL 2019. https://arxiv.org/abs/1902.10186
- Kim, B. et al. (2018). *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV).* ICML 2018. https://proceedings.mlr.press/v80/kim18d/kim18d.pdf
- Länzlinger, J., Müller, K. O. E., Stiller, B., & Rodrigues, B. (2026). *Towards Interpretable Depression Detection: Linking Acoustic Features to DSM-5 Indicators.* arXiv:2608.26148. https://arxiv.org/abs/2608.26148
- Li, Y., Mohamied, Y., Bell, P., & Lai, C. (2022). *Exploration of A Self-Supervised Speech Model: A Study on Emotional Corpora.* SLT 2022. https://arxiv.org/abs/2210.02595
- Maran et al. (2025). *Performance of Automatic Speech Analysis in Detecting Depression: Systematic Review and Meta-Analysis.* JMIR Mental Health. https://pmc.ncbi.nlm.nih.gov/articles/PMC12590051/
- Opitz, J., Möller, L., Michail, A., Padó, S., & Clematide, S. (2025). *Interpretable Text Embeddings and Text Similarity Explanation: A Survey.* EMNLP 2025. https://arxiv.org/abs/2502.14862
- Pasad, A., Chou, J.-C., & Livescu, K. (2021). *Layer-wise Analysis of a Self-supervised Speech Representation Model.* ASRU 2021. https://arxiv.org/abs/2107.04734
- Pasad, A., Shi, B., & Livescu, K. (2023). *Comparative Layer-wise Analysis of Self-supervised Speech Models.* ICASSP 2023. https://arxiv.org/abs/2211.03929
- Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks (Integrated Gradients).* ICML 2017. https://arxiv.org/abs/1703.01365
- Wiegreffe, S., & Pinter, Y. (2019). *Attention is not not Explanation.* EMNLP 2019. https://aclanthology.org/D19-1002/
- Yeh, H.-C., Sun, L., Mahapatra, A., Suresh Chandra, S., Mower Provost, E., & Sisman, B. (2026). *Who is Speaking or Who is Depressed? A Controlled Study of Speaker Leakage in Speech-Based Depression Detection.* Interspeech 2026. https://arxiv.org/abs/2604.14354
- Revisión sistemática de métodos XAI en detección de deterioro cognitivo por voz (2025): SHAP, LIME y atención en 13 estudios. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12657886/
