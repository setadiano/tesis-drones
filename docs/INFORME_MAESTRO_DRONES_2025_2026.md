# Uso Operacional de Drones en Conflictos Modernos (2025-2026)
## Análisis de Viabilidad: Artículo Académico y Tesis Doctoral

**Autor:** Análisis preparatorio  
**Fecha:** Abril 2026  
**Clasificación:** Fuentes abiertas (OSINT)

---

## 1. ESTADO DEL ARTE Y CONTEXTO

### 1.1 El Nuevo Paradigma de la Guerra de Drones

El período 2025-2026 ha consolidado la guerra de drones como el fenómeno militar más transformador desde la introducción del carro de combate. Dos teatros de operaciones ofrecen laboratorios analíticos de naturaleza radicalmente distinta pero complementaria:

| Dimensión | Ucrania-Rusia (Táctico-Operacional) | Irán-Israel/GCC (Estratégico) |
|-----------|-------------------------------------|-------------------------------|
| Escala temporal | Continua (>3 años) | Episódica (oleadas) |
| Volumen | 54.538 Shahed solo en 2025 | Cientos por operación |
| Objetivo primario | Saturación defensa aérea + atricción | Disuasión estratégica + shock |
| Sistema defensivo | Mixto (EW + cinético + drones interceptores) | Multicapa de alta tecnología (Iron Dome, Arrow, THAAD) |
| Métrica principal | Tasa de intercepción diaria | Tasa por oleada/operación |
| Innovación clave | FPV fibra óptica; Geran-3 jet; interceptor drone | Drones pre-posicionados; swarm + BM coordinado |

### 1.2 Hallazgos Cuantitativos Clave (2025)

**Teatro ucraniano:**
- **54.538** Shahed-type lanzados en 2025 (5x el año anterior)
- **819.737** impactos drone confirmados por UA en 2025 (80%+ objetivos destruidos)
- **3.000.000** FPV recibidos por Ucrania en 2025
- Producción rusa: **50.000 FPV fibra óptica/mes** (septiembre 2025)
- Tasa intercepción: oscila entre **77-95%** según volumen del ataque
- **Paradoja de saturación:** ataques >400 UAVs reducen tasa hit al 5-8% pero agotan defensas para misiles más costosos
- Mayor ataque único: **823 UAVs** el 7 de septiembre de 2025

**Teatro iraní-israelí:**
- **Rising Lion (13 junio 2025):** 2.500 sorties israelíes, 6.000 municiones, destrucción del 80% de la DAA iraní
- **99%** intercepción de UAVs durante Rising Lion (Moshe Patel, Homa Directorate, UVID 2025)
- **92%** intercepción conjunta US-Israel en conflicto 2026
- **+93 oleadas** de TP4 hasta abril 2026
- Primera vez en historia: Iran ataca simultáneamente **todos los países GCC**
- **300 BM** iraníes disparados en los primeros 10 días de 2026; capacidad degradada >90% al día 10

---

## 2. FUENTES DE DATOS DISPONIBLES

### 2.1 Fuentes OSINT de Alta Calidad (acceso abierto)

| Fuente | URL | Tipo | Cobertura | Fiabilidad |
|--------|-----|------|-----------|------------|
| **ISIS-Online** | isis-online.org | Análisis mensual Shahed | Ene-Dic 2025 + tendencias | ⭐⭐⭐⭐⭐ |
| **ACLED** | acleddata.com | BD eventos georreferenciados | 2020-presente, descargable | ⭐⭐⭐⭐⭐ |
| **CSIS Futures Lab** | csis.org/analysis | Dashboard interactivo diario | 2022-presente | ⭐⭐⭐⭐⭐ |
| **Ukraine Air Force** | t.me/kpszsu | Reportes diarios (Telegram) | Diario 2022-presente | ⭐⭐⭐⭐ |
| **ShahedTracker** | @ShahedTracker (X) | Tracking en tiempo real | 2023-presente | ⭐⭐⭐⭐ |
| **Oryx** | oryxspioenkop.com | Pérdidas equipamiento foto-verificadas | 2022-presente | ⭐⭐⭐⭐⭐ |
| **OPFOR Journal** | opforjournal.com | Análisis situacional | 2024-presente | ⭐⭐⭐⭐ |
| **CriticalThreats/ISW** | criticalthreats.org | Actualizaciones Iran-Israel | 2023-presente | ⭐⭐⭐⭐⭐ |
| **Ukraine Arms Monitor** | Substack | Análisis anual tendencias | 2023-presente | ⭐⭐⭐⭐ |
| **Kyiv Independent** | kyivindependent.com | Noticias + datos MoD | Diario | ⭐⭐⭐⭐ |

### 2.2 Fuentes Académicas de Referencia Obligatoria

| Paper | Autores | Revista | Año | Relevancia |
|-------|---------|---------|-----|------------|
| Quantitative analysis drone/counter-drone Russia-Ukraine | Mittal & Goetz | Small Wars & Insurgencies | 2025 | **★★★★★ Metodología referente** |
| Spatiotemporal analysis drone ops (ACLED) | Kim & Cho | J. of Advanced Military Studies | 2023 | **★★★★★ Modelo espacio-temporal** |
| Underwater Drones SEM Model | Santhoshraja & Chandra | Turkish Epublication | 2024 | **★★★★ SEM para guerras asimétricas** |
| Evolution strike UAVs RU vs UA Jan-Jun 2025 | Oleksenko & Ikaiev | Ukrainian MoD Journal | 2025 | **★★★★★ Datos primarios únicos** |
| Evolution Drones & Counter-Drone Systems | Eprikian et al. | Defense & Science (Georgia) | 2025 | **★★★★ Marco conceptual** |
| Standardized Evaluation Counter-Drone Systems | Ollero et al. | Drones (MDPI) | 2025 | **★★★★ Métricas rendimiento** |
| Reinforcement Learning Drone Interception | ArXiv 2508.00641 | arXiv preprint | 2025 | **★★★★ Modelado RL** |
| Swarm Classification Neural Networks | ArXiv 2403.19572 | arXiv preprint | 2024 | **★★★★ NN para swarms** |

---

## 3. OPCIONES METODOLÓGICAS PARA EL ARTÍCULO/TESIS

### 3.1 MAPA METODOLÓGICO COMPLETO

```
PREGUNTA DE INVESTIGACIÓN
        │
        ├─► ¿Qué factores predicen la tasa de penetración?
        │         └─► REGRESIÓN MÚLTIPLE / GLM
        │
        ├─► ¿Cómo evolucionan los patrones en el tiempo?
        │         └─► SERIES TEMPORALES (ARIMA/SARIMA/VAR)
        │
        ├─► ¿Existen fases o regímenes distintos?
        │         └─► ANÁLISIS CAMBIO DE RÉGIMEN / Markov Switching
        │
        ├─► ¿Qué diferencia los dos teatros?
        │         └─► ANÁLISIS MULTIVARIANTE (PCA/Factor Analysis/MANOVA)
        │
        ├─► ¿Hay umbrales de saturación?
        │         └─► CLUSTERING (K-means) + REGRESIÓN UMBRAL (TAR)
        │
        └─► ¿Se pueden predecir oleadas futuras?
                  └─► REDES NEURONALES (LSTM/Transformer)
```

### 3.2 OPCIONES EN DETALLE

#### A. ANÁLISIS DE SERIES TEMPORALES (RECOMENDACIÓN PRIORITARIA)

**Fundamento:** La variable principal (lanzamientos diarios/mensuales) es una serie temporal con estructura clara.

| Técnica | Variables | Aplicación específica | Complejidad |
|---------|-----------|----------------------|-------------|
| **ARIMA/SARIMA** | Lanzamientos_t, intercepciones_t | Modelar la dinámica de escalada mensual | Media |
| **VAR (Vector Autoregresivo)** | [lanzamientos, intercepciones, hits, misiles_combinados] | Causalidad Granger entre producción y eficacia | Media-Alta |
| **GARCH** | Volatilidad en tasa de intercepción | Modelar períodos de alta/baja incertidumbre defensiva | Alta |
| **Markov Switching** | Cambios de régimen táctico | Identificar transiciones: ataque sostenido → ataque masivo | Alta |
| **Structural Break (Chow test)** | Serie lanzamientos Shahed | Detectar puntos de quiebre (sept 2024, jun 2025, sept 2025) | Baja-Media |

**Variable dependiente sugerida:** `tasa_hit_pct` (proxy de eficacia ofensiva)  
**Variables independientes:** `log(lanzamientos_total)`, `ratio_decoy`, `tipo_ataque_binario`, `estación`, `presión_frontline`

**Hipótesis testeable (H1):** "La tasa de impacto disminuye logarítmicamente con el volumen de lanzamientos (efecto saturación), pero aumenta con el ratio strike/decoy y con la complejidad táctica (multi-vector)."

---

#### B. ANÁLISIS MULTIVARIANTE

**Fundamento:** Comparar sistemáticamente los dos teatros en múltiples dimensiones simultáneamente.

| Técnica | Aplicación | Output |
|---------|-----------|--------|
| **PCA** | Reducción dimensional de variables operacionales | Identificar los 2-3 factores latentes que explican la varianza |
| **Cluster Analysis (K-means/DBSCAN)** | Tipología de ataques | Clasificar ataques en: "saturación pura", "combinado BM+UAV", "quirúrgico", "perturbación" |
| **MANOVA** | Diferencias Ucrania vs Israel | Test estadístico de diferencias en vector [tasa_intercepcion, tasa_hit, escala, objetivo] |
| **Análisis Discriminante** | Clasificación de escenarios | ¿Qué variables predicen si un ataque es táctico (UA-RU) vs estratégico (IR-IL)? |
| **Regresión Logística Multinomial** | Daño categorizado | Predecir nivel de daño: [ninguno, parcial, severo, crítico] |

**Hipótesis testeable (H2):** "Los factores de escala (volumen, frecuencia) son los determinantes principales de eficacia en el teatro táctico, mientras que la sorpresa estratégica y la saturación de C2 lo son en el teatro estratégico."

---

#### C. REDES NEURONALES Y MACHINE LEARNING

**Fundamento:** Capturar patrones no lineales y dependencias complejas en el tiempo.

| Técnica | Aplicación | Ventaja | Limitación |
|---------|-----------|---------|------------|
| **LSTM (Long Short-Term Memory)** | Predicción series temporales de lanzamientos | Captura dependencias largas (ciclos de producción) | Requiere +200 observaciones |
| **Random Forest / XGBoost** | Clasificación de tipo de ataque + predicción daño | Interpretable con SHAP; bueno con features mixtas | No captura secuencias |
| **CNN 1D** | Detección de patrones en oleadas | Detecta "firmas" táctimas en la secuencia temporal | Caja negra |
| **Transformer / Attention** | Predicción con contexto complejo | SOTA en séries temporales | Alta complejidad computacional |
| **Reinforcement Learning (POMDP)** | Modelado de intercepción óptima | Modela la decisión bajo incertidumbre (ArXiv 2508.00641) | Requiere simulación |

**Nota metodológica:** Con datos mensuales (17 filas UA-RU), los modelos LSTM/DL están **sub-especificados**. Necesitarías datos **diarios** (365+ observaciones) que existen en las fuentes primarias (Ukraine AF diario, ISIS-Online semanal). La tesis debe construir ese dataset granular.

**Hipótesis testeable (H3):** "Un modelo LSTM entrenado con datos diarios de lanzamientos, intercepciones, tipo de munición y condiciones climáticas puede predecir el volumen del siguiente ataque con error <15% (MAPE)."

---

#### D. ANÁLISIS DE UMBRAL Y SATURACIÓN (TU LÍNEA ACTUAL)

**Fundamento:** Continuación directa de tu trabajo previo con K-means y el umbral de 412 drones.

| Técnica | Aplicación | Resultado esperado |
|---------|-----------|-------------------|
| **TAR (Threshold Autoregressive)** | Umbral de saturación dinámica | Estimar el umbral crítico donde la defensa colapsa marginalmente |
| **SETAR (Self-Exciting TAR)** | Autoexcitación en el volumen | Modelar si ataques grandes generan más ataques grandes |
| **Análisis de supervivencia (Cox)** | Tiempo hasta colapso defensivo | ¿Cuánto dura una defensa antes de ser saturada? |
| **Non-linear regression (spline)** | Curva saturación-eficacia | Curva en U o J entre volumen y eficacia |

**Datos clave ya identificados para este análisis:**
- 100-200 drones/día: tasa hit 40-50% (strike UAVs: 70-80%)
- >400 drones/día: tasa hit cae a 5-8%
- Agosto 9, 2025: outlier excepcional (66% efectividad)
- Este patrón es **estadísticamente modelable** con regresión no lineal

---

### 3.3 DISEÑO DE TESIS DOCTORAL PROPUESTO

#### Título tentativo:
*"Dinámica de Saturación en la Guerra de Drones: Análisis Cuantitativo Comparado de los Conflictos Ucrania-Rusia e Irán-Israel (2022-2026)"*

#### Estructura propuesta:

```
CAPÍTULO 1: Marco teórico
  - Teoría de la atricción aérea (Boyd, Biddle, Krepinevich)
  - Saturación vs. precisión en guerra moderna
  - Hipótesis de saturación de defensa aérea (HSDA)

CAPÍTULO 2: Metodología y datos
  - Construcción del dataset (ACLED + Ukraine AF + ISIS-Online + IDF)
  - Variables operacionales: definición y operacionalización
  - Estrategia de análisis multinivel (oleada / campaña / teatro)

CAPÍTULO 3: Teatro táctico - Ucrania-Rusia
  - Análisis de series temporales: evolución 2022-2026
  - Modelo de umbral de saturación (TAR)
  - Curva eficacia-volumen (regresión no lineal)
  - Innovación táctica: FPV fibra óptica, Geran-3, interceptores

CAPÍTULO 4: Teatro estratégico - Irán-Israel
  - Análisis de oleadas (event study)
  - Comparativa True Promise 1-4 y Rising Lion
  - Degradación capacidad ofensiva: modelo de desgaste (Lanchester)
  - Sistema defensivo multicapa: Iron Dome, Arrow, THAAD

CAPÍTULO 5: Análisis comparativo y modelo integrado
  - PCA: factores latentes de eficacia en ambos teatros
  - MANOVA: diferencias táctico vs. estratégico
  - Modelo predictivo LSTM/VAR
  - Implicaciones doctrinales

CAPÍTULO 6: Conclusiones y proyección 2026-2030
  - Convergencia táctica-estratégica
  - Transferencia tecnológica Rusia-Irán-Hezbollah
  - Implicaciones para la OTAN y sistemas occidentales
```

#### Originalidad científica (gap en la literatura):
1. **Ningún paper publicado** compara cuantitativamente ambos teatros con los datos de 2025-2026
2. Los modelos de umbral dinámico (TAR/SETAR) **no han sido aplicados** a datos de saturación de defensa aérea
3. La construcción de un dataset longitudinal diario **con variables tácticas enriquecidas** es en sí misma una contribución metodológica
4. La dicotomía táctico/estratégico como variable independiente en modelos multivariantes es **novedosa**

---

## 4. BASES DE DATOS GENERADAS

### 4.1 Base de Datos Ucrania-Rusia (db_drones_ucrania_rusia_2025_2026.csv)
- **17 registros** (Ene 2025 - Mar 2026) a nivel oleada mensual
- **16 variables:** oleada_id, fecha, teatro, tipo_drone, lanzamientos_total, strike_uav, decoy_uav, intercepciones, hits, tasa_intercepcion_pct, tasa_hit_pct, regiones_objetivo, tipo_objetivo, daño_confirmado, fuente, notas
- **Datos cuantitativos verificados:** 72.115 lanzamientos totales registrados

### 4.2 Base de Datos Irán-Israel/GCC (db_drones_iran_israel_2025_2026.csv)
- **9 registros** de operaciones/oleadas principales 2025-2026
- **19 variables:** incluye campo adicional `sistema_defensa` y `operacion`
- Cubre: True Promise 2, Rising Lion, True Promise 4 (oleadas 1-93+)

### 4.3 Dataset Combinado JSON (db_drones_COMBINADO_2025_2026.json)
- Metadatos completos + ambas tablas
- Estructura anidada lista para pandas/R

---

## 5. PRÓXIMOS PASOS PARA AMPLIAR LA BD

### Datos pendientes de extracción (alta prioridad):

| Fuente | Datos a extraer | Granularidad | Método |
|--------|----------------|--------------|--------|
| Ukraine AF Telegram (@kpszsu) | Reportes diarios 2025 | Diario | Parser Python (Telethon) |
| ISIS-Online tablas Fig.1/Table 1 | Datos semanales Shahed | Semanal | Web scraping + OCR |
| ACLED Data Export Tool | Eventos georef. drones Ucrania | Por evento | Descarga CSV (registro gratuito) |
| ACLED Middle East 2025-2026 | Eventos Iran-Israel-GCC | Por evento | Descarga CSV |
| CSIS Interactive Dashboard | Series semanales Shahed | Semanal | Selenium/requests |
| IDF Press Office | Datos intercepción por operación | Por oleada | Web scraping |

### Variables adicionales a incorporar:
- **Coordenadas GPS** de impactos (disponible en ACLED)
- **Condiciones climáticas** (temperatura, viento, visibilidad) — efecto en EW y drones
- **Fase lunar / hora del ataque** (ataques nocturnos Shahed)
- **Presión de la línea de frente** (km avanzados en el período)
- **Precio del barril Brent** (proxy de presión económica sobre defensores)
- **Tipo de sistema defensivo disponible** (interceptores SAM restantes)

---

## 6. VIABILIDAD DEL ARTÍCULO Y LA TESIS

### 6.1 Viabilidad del Artículo (6-12 meses)

**Enfoque recomendado para artículo inicial:**
> *"Saturación logarítmica y eficacia ofensiva en la campaña Shahed 2025: análisis de series temporales y regresión de umbral"*

- **Datos:** ya disponibles (ISIS-Online Table 1 + Ukraine AF)
- **Metodología:** ARIMA + regresión umbral TAR + gráficas no lineales
- **Novedad:** primera aplicación de TAR a datos de saturación aérea en conflicto activo
- **Revistas objetivo:**
  - *Journal of Strategic Studies* (Q1 Defensa)
  - *Small Wars & Insurgencies* (Q2, ya tiene papers similares Mittal & Goetz)
  - *Defence & Security Analysis* (Q2)
  - *Journal of Military and Strategic Studies* (acceso abierto)
  - *Drones* (MDPI, Q2, interdisciplinar, muy citado)

**Calificación de viabilidad: ALTA ✓**

### 6.2 Viabilidad de la Tesis Doctoral

**Factores favorables:**
- ✅ Tema de máxima actualidad con escasez de literatura cuantitativa
- ✅ Datos OSINT accesibles y verificables (fuentes abiertas)
- ✅ Doble teatro ofrece comparación interna robusta
- ✅ Múltiples técnicas posibles → capítulos diferenciados
- ✅ El conflicto activo genera datos nuevos continuamente
- ✅ Conexión directa con necesidades del mundo de defensa (OTAN, SEDEF)

**Factores de riesgo:**
- ⚠️ Datos de alta granularidad (diario) requieren construcción manual
- ⚠️ Sesgo de fuente (datos ucranianos son oficiales, posiblemente optimistas)
- ⚠️ El conflicto puede cambiar rápidamente → metodología debe ser robusta al cambio
- ⚠️ Clasificación parcial de algunos eventos en teatro israelí

**Mitigaciones:**
- Usar múltiples fuentes cruzadas (triangulación)
- Distinguir claramente "datos verificados con vídeo" (Oryx) vs "declaraciones oficiales"
- Modelo con bandas de incertidumbre (intervalos de confianza bootstrap)

**Calificación de viabilidad: MUY ALTA ✓✓**

---

## 7. REFERENCIAS CLAVE

1. Mittal, V. & Goetz, J. (2025). "A quantitative analysis of the effects of drone and counter-drone systems on the Russia-Ukraine battlefield." *Small Wars & Insurgencies*. DOI: 10.1080/14751798.2025.2479973
2. Kim, H. & Cho, J. (2023). "Spatiotemporal analysis of drone operations using ACLED: Russia-Ukraine war." *Journal of Advanced Military Studies*, 6(3). DOI: 10.37944/jams.v6i3.230
3. ISIS-Online (2026). "A Comprehensive Analytical Review of Russian Shahed-type UAVs deployment against Ukraine in 2025." isis-online.org
4. CSIS Futures Lab (2025). "Drone Saturation: Russia's Shahed Campaign." csis.org/analysis/drone-saturation
5. CSIS (2026). "Russia-Ukraine War in 10 Charts." csis.org
6. CSIS (2025). "Operation Rising Lion and the New Way of War." csis.org/analysis/ungentlemanly-robots
7. ACLED (2026). "Middle East Special Issue: March 2026." acleddata.com
8. Oleksenko, O. & Ikaiev, D. (2025). "Evolution of strike UAVs by the Armed Forces of the Russian Federation (Jan-Jun 2025)." DOI: 10.37701/dndivsovt.26.2025.17
9. Eprikian, P. et al. (2025). "The Evolution of Drones and Counter-Drone challenges in contemporary warfare." *Defense and Science*, 4(2025). DOI: 10.61446/ds.4.2025.10449
10. ArXiv 2508.00641 (2025). "Reinforcement Learning for Decision-Level Interception of Drone Swarms."
11. Ukraine's Arms Monitor (2025). "Drone Warfare in Ukraine: Key Trends of 2025." Substack.
12. Wikipedia (2026). "2026 Iranian strikes on Israel." (con referencias a JINSA, France 24, ACLED)
13. JFeed (2025). "Israel Boasts Near-Perfect Drone Interception Rates in Operation Rising Lion." UVID 2025 conference data.
14. CriticalThreats/ISW (2026). "Iran Update Evening Special Report, March 5, 2026." criticalthreats.org
15. Santhoshraja V. & Chandra G. (2024). "Underwater Drones – Defence SEM Model." DOI: 10.53555/d00gas41

---

*Documento generado con fuentes OSINT verificadas. Para uso académico y analítico.*
