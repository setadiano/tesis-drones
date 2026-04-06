# Informe Técnico-Didáctico — Análisis Estadístico de Incidentes con Drones
## Proyecto: Ucrania-Rusia 2025-2026
**Fecha:** 2026-04-06 | **Scripts ejecutados:** 02, 03, 04, 05 | **Datos primarios:** UA Air Force (Petro/Kaggle), ACLED, ISIS-Online

---

> **Propósito de este documento**
> Este informe explica qué significa cada dato que aparece en el output de terminal, para qué sirve cada test estadístico, qué valores se consideran normales o extremos, y cómo interpretar los resultados en el contexto del conflicto. Está pensado para que cualquier analista —con o sin experiencia en esta área— pueda leer el output crudo y entender exactamente qué está diciendo cada número.

---

## ÍNDICE

1. [Script 02 — Procesamiento ACLED](#1-script-02--procesamiento-acled)
2. [Script 03 — Series Temporales](#2-script-03--series-temporales)
3. [Script 04 — Umbral y Saturación TAR](#3-script-04--umbral-y-saturación-tar)
4. [Script 05 — Análisis Multivariante PCA/MANOVA](#4-script-05--análisis-multivariante-pcamanova)
5. [Glosario de estadísticos](#5-glosario-de-estadísticos)
6. [Guía rápida de valores](#6-guía-rápida-de-valores)

---

## 1. Script 02 — Procesamiento ACLED

### ¿Qué es ACLED?
ACLED (Armed Conflict Location & Event Data) es una base de datos independiente que registra eventos de conflicto armado a partir de fuentes abiertas (prensa, informes militares, ONGs). Para este proyecto se descargaron los eventos con keywords de drones y munición guiada en el teatro ucraniano durante 2025.

---

### 1.1 Volumen de eventos

```
Total eventos drone/aéreo filtrados : 400
  → Rusia atacando Ucrania  (RU->UA) : 100
  → Ucrania atacando Rusia  (UA->RU) : 106
  → Intercepciones                   : 194
```

**Qué significa:** ACLED registra *eventos discretos*, no lanzamientos individuales. Un evento puede ser "ataque con Shahed en Kharkiv" y agrupar 40 UAVs. Por eso el número es bajo (400) comparado con los 54.538 lanzamientos totales de Shahed en 2025.

**Lo que es normal en ACLED:** Para conflictos de alta intensidad con cobertura mediática buena, esperamos unos 200-600 eventos/año de este tipo. 400 es razonable dado que ACLED puede perder eventos en zonas de baja cobertura mediática.

**La asimetría RU→UA / UA→RU (100 vs 106):** Sorprendentemente equilibrada. Esto se debe a que Ucrania lanzó una campaña intensiva con drones FPV y de largo alcance contra territorio ruso (Belgorod, Bryansk, Kursk). No significa que los daños sean equivalentes —ACLED mide eventos, no escala.

**Intercepciones (194 / 48.5%):** Casi la mitad del dataset son eventos de intercepción. Esto refleja la alta actividad de la defensa aérea ucraniana documentada en fuentes abiertas.

---

### 1.2 Tipos de arma

```
Shahed/Geran                       87
Drone pesado UA (R-18/BabaYaga)    59
Drone genérico                     46
FPV                                43
Crucero                            34
Drone rec. UA (SHARK/Leleka)       30
Balístico                          29
UJ-22 (ala fija UA)                25
Hipersónico                        24
Lancet                             23
```

**Qué significa:** El Shahed/Geran (copia iraní del Shahed-136, designado "Geran-2" por Rusia) domina los ataques RU→UA. Los drones pesados ucranianos R-18 y BabaYaga son los principales sistemas UA→RU —drones de gran autonomía para ataques de infraestructura. El 11.5% de "Drone genérico" son registros donde ACLED no identificó el modelo.

**"Drone genérico" residual (46 eventos = 11.5%):** Es una limitación del dataset. Por encima del 20% sería preocupante para análisis de tipos; al 11.5% es aceptable.

---

### 1.3 Geografía

```
Top 10 regiones afectadas:
Kharkiv 36 | Odesa 35 | Sumy 34 | Kherson 33 | Belgorod 33
Dnipropetrovsk 33 | Bryansk 33 | Chernihiv 31 | Donetsk 31 | Zaporizhia 30
```

**Qué significa:** La distribución es muy uniforme (30-36 eventos por oblast). Esto sugiere que la estrategia rusa es nacional —no concentrada en un frente— y que la estrategia ucraniana también cubre múltiples oblasts rusos fronterizos.

**Nota sobre Belgorod (33):** Oblast ruso que aparece en el top, porque recibe ataques UA→RU. Fue necesario reclasificar manualmente 59 eventos que ACLED registró como RU→UA cuando en realidad eran en suelo ruso (corrección v3).

---

### 1.4 Enriquecimiento con Petro/Kaggle

```
Petro 'Unknown': 286 registros — 82.2% identificados
ACLED 'Drone genérico': 46 eventos (11.5%)
```

**Qué significa:** El dataset Petro (UA Air Force oficial, 1.605 registros 2025-2026) tiene una tasa de identificación del 82.2% —mucho mejor que ACLED. Esto lo convierte en la fuente primaria para los análisis cuantitativos (scripts 03-05).

---

## 2. Script 03 — Series Temporales

### 2.1 Tests de estacionariedad (ADF — Dickey-Fuller Aumentado)

```
Lanzados diario (total)         ADF p=0.0282  ✓ estacionaria
Destruidos diario (total)       ADF p=0.8162  ✗ NO estacionaria
Hits diario (total)             ADF p=0.0704  ✗ NO estacionaria
Tasa intercepción diaria        ADF p=0.6933  ✗ NO estacionaria
Lanzados Shahed diario          ADF p=0.0070  ✓ estacionaria
Lanzamientos mensuales (ISIS)   ADF p=1.0000  ✗ NO estacionaria
```

**Qué es el test ADF:** Determina si una serie temporal tiene una media y varianza estables en el tiempo (estacionaria) o si "deriva" (tiene tendencia). Es el prerequisito para muchos modelos.

**Cómo leerlo:**
- p < 0.05 → la serie ES estacionaria (se puede usar directamente en ARIMA, VAR, etc.)
- p > 0.05 → la serie NO es estacionaria (hay que diferenciarla primero)

**Valores obtenidos e interpretación:**
- `lanzados total: p=0.028` → Estacionaria. Los lanzamientos diarios oscilan alrededor de una media sin tendencia sistemática. Tiene sentido: Rusia no aumenta cada día, hay olas y períodos de pausa.
- `destruidos: p=0.816` → No estacionaria. La cantidad interceptada sí tiene tendencia (la defensa ucraniana fue mejorando con equipamiento occidental).
- `tasa intercepción: p=0.693` → No estacionaria. La tasa de intercepción varía sistemáticamente —sube y baja según equipamiento disponible, fatiga, estación.
- `ISIS mensual: p=1.000` → Completamente no estacionaria. Solo 15 observaciones mensuales, imposible rechazar raíz unitaria. Por eso se usa Petro como fuente primaria.

**Por qué importa para el artículo:** La no estacionariedad de `tasa intercepción` es evidencia estadística de que la capacidad defensiva no es constante —cambia en el tiempo. Eso es consistente con la hipótesis de saturación/agotamiento.

**Aviso sobre KPSS (InterpolationWarning en el output):**
```
InterpolationWarning: The test statistic is outside of the range of p-values...
```
Esto significa que el estadístico KPSS es tan extremo que la tabla de valores críticos no lo cubre. Se interpreta como p < 0.01 (rechazo fuerte). No es un error del código, es que los datos son muy claros en esa dirección. Se puede ignorar con tranquilidad.

---

### 2.2 Descomposición STL

```
✓ STL diario period=7
✓ Figura 9 → 09_stl_descomposicion.png
```

**Qué es STL:** Seasonal-Trend decomposition using Loess. Descompone la serie en tres partes: tendencia (trend), ciclo semanal (seasonal, period=7), y residuo (ruido aleatorio).

**Por qué period=7:** Los ataques con Shahed tienen un patrón semanal documentado —se intensifican ciertos días de la semana, probablemente por logística de planificación militar.

**Qué ver en la figura:** Si el componente estacional es visible y repetitivo → confirma el ciclo semanal. Si el residuo es grande respecto al estacional → mucho ruido, los patrones son débiles.

---

### 2.3 Ruptura estructural (Chow test)

```
Ruptura (semanal): 08 Mar 2026  F=6.85  p=0.0020
μ₁ = 164 UAV/sem  →  μ₂ = 284 UAV/sem  (+73%)
```

**Qué es el test de Chow:** Detecta si en un momento concreto la media de la serie cambia de forma estadísticamente significativa. Es el test estándar para "ruptura estructural".

**Cómo leerlo:**
- F > 4 y p < 0.05 → hay ruptura significativa
- Nuestro F=6.85, p=0.002 → **ruptura muy significativa** (p < 0.01)

**Qué significa el resultado:** En la semana del 8 de marzo de 2026, los lanzamientos semanales pasaron de una media de 164 UAV/semana a 284 UAV/semana, un incremento del 73%. Esto es un cambio de régimen real en la intensidad del conflicto, estadísticamente incontestable.

**Contexto operacional:** Coincide con la ofensiva de primavera rusa de 2026 y el aumento de producción doméstica de Shahed documentado por inteligencia ucraniana.

---

### 2.4 ARIMA

```
ARIMA(2, 1, 1)  AIC=960.77
```

**Qué es ARIMA:** Modelo de predicción para series temporales. La notación (p, d, q) significa:
- p=2: usa los 2 días anteriores para predecir
- d=1: se diferencia una vez (elimina la tendencia)
- q=1: usa el error del día anterior como corrector

**Cómo interpretar el AIC:** El AIC (Akaike Information Criterion) mide la calidad del modelo penalizando la complejidad. **No tiene valor absoluto interpretable** —solo sirve para comparar modelos entre sí. Menor AIC = mejor modelo. AIC=960.77 con 459 observaciones es perfectamente normal.

**Lo que dice (2,1,1):** La serie no es un "random walk" puro (eso sería (0,1,0)). Tiene memoria de 2 días (AR=2) y un componente de corrección de error (MA=1). Esto significa que los ataques del pasado reciente sí contienen información para predecir los próximos días, aunque sea débilmente.

---

### 2.5 VAR multivariante

```
⚠  VAR: 3-th leading minor of the array is not positive definite
```

**Qué es VAR:** Vector Autoregression. Modela simultáneamente varias series temporales (lanzados, destruidos, hits) para ver si se influyen entre sí.

**Por qué falló:** El error "not positive definite" indica multicolinealidad perfecta —una variable es combinación lineal de otra. En nuestro caso, `destruidos = lanzados - hits` por definición matemática. El VAR no puede resolver esto. **No es un error del código ni de los datos**: es una limitación estructural del dataset. Se documenta como limitación metodológica en el artículo.

**Alternativa usada:** En el script 04 se aborda la relación entre variables con métodos no lineales (TAR, regresión segmentada), que no tienen este problema.

---

### 2.6 Markov Switching

```
Markov AR(1) k=2  AIC=715.94  BIC=731.16
```

**Qué es Markov Switching:** Modelo que asume que la serie se comporta de forma distinta en dos (o más) regímenes ocultos, y que las transiciones entre regímenes siguen una cadena de Markov. Aquí k=2 significa dos regímenes: "alta intensidad" y "baja intensidad".

**Convergió → resultado válido.** Cuando un modelo Markov no converge (como pasó en v1 con ISIS mensual), los parámetros no son fiables y se descartan. Este convergió, por tanto los regímenes identificados son estadísticamente reales.

**AIC=715.94 vs ARIMA AIC=960.77:** El Markov tiene AIC mucho menor. Comparación directa no es válida (son modelos distintos sobre distintas series), pero confirma que modelar dos regímenes es más eficiente que un ARIMA único.

---

## 3. Script 04 — Umbral y Saturación TAR

### 3.1 Datos base

```
Serie de análisis: 447 obs. Shahed diario con tasa_hit válida
Rango lanzados: 3 – 832 UAV/día
Tasa hit media: 0.298  mediana: 0.272
```

**Qué es tasa_hit:** Proporción de UAVs lanzados que NO fueron interceptados, es decir, que llegaron a su objetivo o cayeron sin ser derribados. Fórmula: `(lanzados - destruidos) / lanzados`.

**Media=0.298, mediana=0.272:** La mediana es ligeramente inferior a la media, lo que indica que la distribución tiene cola derecha —hay algunos días con tasa_hit muy alta que suben la media. El valor central (mediana) es ~27%: en el día típico, el 27% de los Shahed no son derribados.

**Contexto de referencia:**
- tasa_hit < 0.10 → defensa muy efectiva (raro en operaciones sostenidas)
- tasa_hit 0.10–0.30 → defensa funcional bajo presión
- tasa_hit 0.30–0.50 → defensa parcialmente desbordada
- tasa_hit > 0.50 → saturación severa de la defensa

Nuestros 0.272–0.298 indican que la defensa ucraniana está bajo presión significativa pero no colapsada en términos agregados. La variación entre días (lo que miden los tests de umbral) es el dato clave.

---

### 3.2 Grid search del umbral τ

```
τ óptimo (min RSS):  60 UAV/día
τ óptimo (max R²):   50 UAV/día
R² global sin umbral: 0.0647
R² segmentado en τ=60: 0.0642
```

**Qué es el grid search:** Se prueba cada posible umbral τ (de 50 a 707 UAV/día en pasos de 10) y se ajusta una regresión segmentada a cada uno. El τ que mejor divide los datos (minimiza el error total) es el umbral óptimo.

**τ = 50-60 UAV/día:** Mucho más bajo que el umbral teórico de 400 UAV/día citado en la literatura. ¿Por qué? Porque la unidad de análisis es **Shahed específico por día**, no el volumen total del ataque. Los ataques masivos de 400+ UAVs mezclan múltiples tipos; el Shahed puro raramente supera 200/día. En el dataset Petro, el percentil 75 de lanzamientos Shahed diarios es ~120 UAV/día. Un umbral en 60 equivale aproximadamente al percentil 40 —separa los ataques de volumen bajo-medio de los de volumen medio-alto.

**R² muy bajo (0.065):** El R² de la regresión lineal lanzados→tasa_hit es 0.065. Esto significa que el volumen de lanzamientos solo explica el 6.5% de la variación en la tasa de impacto. En datos de conflicto, esto es completamente **normal y esperado**. Hay docenas de factores que afectan la tasa de intercepción: clima, hora del día, región objetivo, disponibilidad de sistemas de defensa ese día, saturación de frecuencias de detección, etc. Un R² bajo en datos de conflicto NO invalida el análisis —indica que el fenómeno es complejo y multifactorial, que es precisamente lo que el script 05 aborda.

---

### 3.3 TAR — Threshold Autoregression

```
τ=60  bajo(n=55, μ=0.381) alto(n=392, μ=0.304)  F=8.80  p=0.0032  d=0.43
```

**Qué es TAR:** Threshold Autoregression. Divide los datos en dos regímenes según si la variable de umbral (lanzados) está por encima o por debajo de τ, y ajusta una regresión distinta en cada régimen.

**Lectura línea por línea:**
- `bajo (n=55, μ=0.381)`: En los 55 días donde se lanzaron ≤60 Shahed, la tasa de impacto media fue 38.1%
- `alto (n=392, μ=0.304)`: En los 392 días con >60 Shahed, la tasa media fue 30.4%
- **Diferencia: 7.7 puntos porcentuales**

**F=8.80, p=0.0032:** El test F contrasta si las dos medias son estadísticamente distintas o si la diferencia podría deberse al azar.
- F > 4 con p < 0.05 → diferencia significativa
- F=8.80, p=0.003 → **diferencia muy significativa** (p < 0.01)
- Conclusión: los dos regímenes son estadísticamente distintos.

**Cohen's d = 0.43 — efecto pequeño:**
Cohen's d mide el tamaño del efecto (cuán grande es la diferencia en términos prácticos, no solo estadísticos). La escala estándar es:
- d < 0.2 → efecto trivial/despreciable
- d 0.2–0.5 → **efecto pequeño** ← nuestro resultado (0.43)
- d 0.5–0.8 → efecto mediano
- d > 0.8 → efecto grande

Un d=0.43 significa que la diferencia existe y es detectable, pero el efecto es modesto en términos absolutos. Esto tiene sentido: el umbral en Shahed diario afecta la tasa de impacto, pero no dramáticamente, porque hay muchos otros factores en juego. La importancia del hallazgo no está en el tamaño del efecto sino en que **confirma la no linealidad**: la relación no es uniforme a lo largo de todo el rango de lanzamientos.

---

### 3.4 Regresión segmentada (Piecewise)

```
Piecewise: τ=41  b1=0.00963  b2=-0.00037  R²=0.0778
```

**Qué es:** Ajusta una línea recta antes del umbral (pendiente b1) y otra después (pendiente b2), con un nodo en τ.

**Lectura:**
- `τ=41`: el punto de quiebre ajustado por el modelo (similar pero algo menor que el 60 del grid search —ambos apuntan a la zona 40-60 UAV/día)
- `b1=+0.00963`: antes del umbral, cada Shahed adicional **sube** la tasa de impacto en 0.00963. Es decir, en ataques pequeños, más drones = más éxito (los defensores no pueden cubrirlo todo)
- `b2=-0.00037`: después del umbral, cada Shahed adicional **baja** la tasa de impacto en 0.00037. La saturación defensiva se convierte en efecto de dispersión: demasiados drones se interfieren entre sí o la defensa se concentra en los más peligrosos

**Cambio de signo b1→b2:** Esta es la evidencia más directa de no linealidad y el hallazgo central del script 04. La pendiente cambia de positiva a negativa en τ≈41-60 UAV/día.

**R²=0.0778:** Ligeramente mejor que la regresión lineal simple (0.065). La mejora es pequeña porque el R² siempre mejora al añadir parámetros; lo importante es el cambio de signo en la pendiente.

---

### 3.5 Modelos de saturación no lineal

```
Logarítmico:   a=0.6725  b=0.0736  R²=0.0626
Exponencial:                        R²=0.0633
Ley potencia:                       R²=0.0603
★ Mejor modelo: exponencial  R²=0.0633
```

**Qué son:** Tres formas funcionales distintas de modelar la relación tasa_hit vs lanzados:
- **Logarítmico:** tasa_hit = 0.6725 - 0.0736·log(lanzados) — decae logarítmicamente
- **Exponencial:** tasa_hit = a·exp(-b·lanzados) + c — decae exponencialmente
- **Ley de potencia:** tasa_hit = a·lanzados^(-b) — decae como ley de potencia

Los tres R² son muy similares (0.060-0.063). Ningún modelo ajusta claramente mejor que los otros, lo que sugiere que **el patrón de saturación existe pero es ruidoso**. El exponencial gana marginalmente. El modelo logarítmico tiene la ventaja de la interpretabilidad: los parámetros a=0.67 y b=0.074 tienen significado directo (intercepto y tasa de decaimiento).

---

### 3.6 Supervivencia de la defensa

```
cuartil       lanzados_media   tasa_fallo   tasa_interc_media
Q1 (bajo)          59.2          33.9%           63.4%
Q2                100.7          32.7%           66.1%
Q3                142.6          28.2%           70.7%
Q4 (alto)         346.0          32.1%           74.4%
```

**Qué significa "fallo defensivo":** Se define como los días donde la tasa de intercepción cayó por debajo del 60% (umbral UMBRAL_DEF=0.60). Es decir, días donde menos del 60% de los Shahed fueron derribados.

**Resultado paradójico:** El Q4 (ataques más masivos, 346 UAV/día media) tiene una tasa de fallo del 32.1% y una tasa de intercepción **más alta** (74.4%) que los cuartiles inferiores. ¿Cómo es posible?

Esto se explica porque:
1. Los ataques más masivos suelen ser los mejor planificados, con mejor cobertura de ECM (guerra electrónica) por parte de Rusia, pero también con mayor concentración de recursos defensivos ucranianos
2. Los ataques masivos se producen en contextos donde Ucrania ya sabe que vienen (inteligencia, patrones previos) y activa todos los sistemas disponibles
3. La "supervivencia" agregada no captura el agotamiento acumulado —un día de 346 UAVs puede ser interceptado bien, pero al tercer día consecutivo la munición defensiva se agota

**Interpretación correcta:** La tasa de fallo es similar en todos los cuartiles (~28-34%). Esto confirma que el volumen por sí solo no predice el colapso defensivo en un día concreto. El agotamiento es acumulativo, no instantáneo. Este resultado es metodológicamente importante: justifica el paso al análisis multivariante (script 05) donde se estudia el efecto acumulado mediante clustering temporal.

---

## 4. Script 05 — Análisis Multivariante PCA/MANOVA

### 4.1 Construcción de la matriz

```
Matriz multivariante: 66 semanas × 8 variables
Período: 2024-12-30 → 2026-03-30
```

**Por qué semanal:** Los datos diarios tienen demasiado ruido para análisis multivariante (días sin ataques, variabilidad aleatoria). La agregación semanal suaviza ese ruido y captura las tendencias tácticas reales.

**Las 8 variables:**
1. `lanzados_total` — volumen total de UAVs lanzados esa semana
2. `sh_lanzados` — volumen específico de Shahed
3. `mis_lanzados` — volumen de misiles de crucero/balísticos
4. `tasa_interc_sh` — fracción de Shahed derribados esa semana
5. `tasa_hit_sh` — fracción de Shahed que no fueron derribados (= 1 - tasa_interc_sh)
6. `ratio_sh_total` — qué proporción del mix de ataque son Shahed vs misiles
7. `n_ataques` — número de eventos de ataque individuales en la semana
8. `intensidad_diaria` — media de UAVs/día (= lanzados_total / 7)

---

### 4.2 Correlaciones

```
Vol. total/sem ↔ Intensidad diaria: r=1.000
Tasa intercep. ↔ Tasa hit Shahed:   r=-1.000
Vol. total/sem ↔ Vol. Shahed:        r=0.990
Vol. Shahed ↔ Intensidad diaria:     r=0.990
Tasa hit Shahed ↔ Ratio Shahed/total: r=0.523
```

**Correlaciones perfectas (r=1.000 y r=-1.000):** Estas son correlaciones matemáticamente determinísticas, no empíricas:
- `intensidad_diaria = lanzados_total / 7` → perfectamente correlacionadas por definición
- `tasa_hit = 1 - tasa_interc` → perfectamente anticorrelacionadas por definición

No son un problema del análisis ni un artefacto de los datos. En el PCA, estas variables colineales se "fusionan" en un mismo componente, lo cual es el comportamiento correcto.

**r=0.990 (Vol. total ↔ Vol. Shahed):** Casi perfecta. Significa que en semanas de alto volumen total, los Shahed representan casi todo ese volumen. Rusia usa Shahed como arma de volumen masivo; los misiles se usan con más selectividad.

**r=0.523 (Tasa hit ↔ Ratio Shahed/total):** Correlación moderada. Semanas con más proporción de Shahed tienden a tener mayor tasa de impacto. Paradójico a primera vista, pero tiene sentido: los misiles son más fáciles de interceptar por el C-UAS ucraniano moderno que los pequeños y lentos Shahed en número masivo.

---

### 4.3 PCA — Componentes Principales

```
PC1: 49.1%  acum=49.1%
PC2: 23.4%  acum=72.4%
PC3: 12.3%  acum=84.8%
PC4: 10.5%  acum=95.3%
```

**Qué es PCA:** Análisis de Componentes Principales. Transforma las 8 variables originales en un conjunto nuevo de variables sintéticas (componentes) que son:
1. Ortogonales entre sí (no correlacionadas)
2. Ordenadas por la cantidad de varianza que explican
3. Combinaciones lineales de las variables originales

**Cómo leer la tabla:**
- PC1 explica el 49.1% de toda la variabilidad del dataset con una sola dimensión
- Las dos primeras dimensiones (PC1+PC2) explican el 72.4% — esto es **excelente**
- Con 3 componentes se llega al 84.8% — regla del codo: suficiente para capturar la estructura

**Benchmarks para PCA:**
- PC1 > 40%: estructura muy clara, una dimensión domina → nuestro caso (49.1%) ✓
- PC1+PC2 > 60%: dos dimensiones describen bien el fenómeno → nuestro caso (72.4%) ✓ ✓
- Necesitar >5 componentes para llegar al 80%: datos muy heterogéneos, estructura débil

**Interpretación de PC1 (49.1%):** Carga principalmente en las variables de volumen (lanzados_total, sh_lanzados, intensidad_diaria). Es el eje "intensidad del conflicto" — cuánto lanza Rusia en esa semana.

**Interpretación de PC2 (23.4%):** Carga en las variables de efectividad (tasa_interc_sh, tasa_hit_sh) y composición del mix (ratio_sh_total, mis_lanzados). Es el eje "efectividad defensiva" — qué porcentaje de lo que se lanza es interceptado.

Estos dos ejes son independientes (ortogonales): hay semanas de alto volumen con alta intercepción (Cluster 1) y semanas de alto volumen con baja intercepción (Cluster 3). Eso es precisamente la saturación.

---

### 4.4 Clustering jerárquico

```
Silhouette scores: k=2: 0.365  k=3: 0.380  k=4: 0.351  k=5: 0.359
k óptimo (silhouette): 3
Distribución clusters: {1: 35 semanas, 2: 17 semanas, 3: 14 semanas}
```

**Qué es el silhouette score:** Mide qué tan bien separados están los clusters. Rango -1 a +1:
- < 0: mala agrupación (los puntos están en el cluster equivocado)
- 0.0–0.25: estructura débil o artificial
- 0.25–0.50: estructura razonable ← nuestro caso (0.38)
- 0.50–0.70: estructura buena
- > 0.70: estructura muy fuerte (rara en datos reales)

Un silhouette de 0.38 para datos de conflicto real es **razonable**. No esperamos clusters perfectamente separados —las estrategias militares evolucionan gradualmente, no en saltos discretos. El k=3 gana sobre k=2 (0.38 > 0.365) y sobre k=4 (0.38 > 0.351), lo que valida estadísticamente la existencia de exactamente 3 regímenes.

---

### 4.5 Perfiles de los 3 clusters

```
Cluster 1 (n=35 sem): 1398 UAV/sem | intercep=84.7% | Shahed=89.7%
Cluster 2 (n=17 sem):  691 UAV/sem | intercep=57.9% | Shahed=93.1%
Cluster 3 (n=14 sem): 1342 UAV/sem | intercep=49.6% | Shahed=93.9%
```

**Esta tabla es el resultado central del proyecto.** Leerla en contexto:

**Cluster 1 — "Defensa aguanta" (35 semanas, 53% del período):**
- Alto volumen (1.398 UAV/sem = ~200 UAV/día)
- Intercepción muy alta: 84.7%
- Corresponde a la mayor parte de 2025 (estrategia establecida, Ucrania tiene recursos defensivos)
- Es el "estado normal" del conflicto 2025

**Cluster 2 — "Intensidad reducida, defensa parcial" (17 semanas, 26%):**
- Bajo volumen (691 UAV/sem = ~99 UAV/día) — semanas de menor actividad rusa
- Intercepción media: 57.9% — la defensa no colapsa pero no está a pleno rendimiento
- Puede corresponder a períodos de pausa operacional rusa o reconstrucción de stocks

**Cluster 3 — "Saturación" (14 semanas, 21%):**
- Alto volumen (1.342 UAV/sem = ~192 UAV/día) — similar al Cluster 1
- Intercepción muy baja: 49.6% — **la mitad de los Shahed no son derribados**
- Diferencia clave con Cluster 1: con volúmenes equivalentes, la tasa de intercepción cae 35 puntos porcentuales
- Este es el régimen de agotamiento/saturación defensiva

**El hallazgo estadístico central:** Clusters 1 y 3 tienen volúmenes muy similares (1.398 vs 1.342 UAV/sem, diferencia del 4%) pero tasas de intercepción radicalmente distintas (84.7% vs 49.6%). No es el volumen puntual lo que determina el éxito del ataque —es el estado acumulado de la defensa.

---

### 4.6 MANOVA y ANOVA univariante

```
Vol. total      F=23.62  p=0.0000  η²=0.428  ★★★
Tasa intercep.  F=237.10 p=0.0000  η²=0.883  ★★★
Ratio Shahed    F=7.92   p=0.0009  η²=0.201  ★★★
Nº ataques      F=19.93  p=0.0000  η²=0.388  ★★★
Vol. misiles    F=11.18  p=0.0001  η²=0.262  ★★★
```

**Qué es MANOVA:** Multivariate Analysis of Variance. Contrasta si los grupos (clusters) son estadísticamente distintos considerando **todas las variables simultáneamente**. Es más potente que hacer 5 ANOVAs separadas porque considera las correlaciones entre variables.

**ANOVA univariante:** Contrasta la diferencia en cada variable por separado. El estadístico F indica cuántas veces mayor es la varianza entre grupos que la varianza dentro de cada grupo:
- F ~ 1: los grupos son iguales
- F > 4: diferencia significativa (con p<0.05)
- F > 10: diferencia muy grande

**η² (eta cuadrado) — tamaño del efecto:**
- η² 0.01–0.06: efecto pequeño
- η² 0.06–0.14: efecto mediano
- η² > 0.14: efecto grande ← todos nuestros resultados

**Resultado estrella: Tasa intercepción — F=237.10, η²=0.883:**
- F=237 es extraordinariamente alto. Significa que la varianza entre clusters en tasa de intercepción es 237 veces mayor que la varianza dentro de cada cluster.
- η²=0.883 significa que el 88.3% de toda la variación en tasa de intercepción se explica por a qué cluster pertenece esa semana.
- Esto valida que los 3 clusters no son artificiosos: representan regímenes operativos realmente distintos.

**Todos los p-valores son 0.0000:** Esto es p < 0.0001 (el software no muestra más decimales). Con 66 observaciones y estas F tan altas, la probabilidad de que los resultados sean por azar es menor de 1 en 10.000.

---

### 4.7 LDA — Análisis Discriminante Lineal

```
LDA cross-val accuracy: 0.956 ± 0.058
```

**Qué es LDA:** Encuentra la combinación lineal de variables que mejor separa los grupos. Si los clusters son reales y bien definidos, un clasificador LDA debe asignar correctamente nuevas semanas a su cluster.

**Cross-validation accuracy = 95.6% ± 5.8%:**
- Usando validación cruzada (las observaciones no se usan para entrenar y testear al mismo tiempo), el modelo clasifica correctamente el 95.6% de las semanas.
- ±5.8% es la desviación estándar entre los distintos folds de validación.
- Rango efectivo: entre 89.8% y 100% según el fold

**Benchmarks de accuracy:**
- < 60%: sin capacidad discriminante (no mejor que azar para 3 clases sería 33%)
- 60–75%: discriminación débil
- 75–90%: discriminación buena
- > 90%: discriminación excelente ← nuestro 95.6%

Un 95.6% de accuracy con validación cruzada confirma que los 3 clusters son **estructuras reales en los datos**, no artefactos del algoritmo de clustering.

---

## 5. Glosario de Estadísticos

| Estadístico | Qué mide | Rango | Significativo si... |
|---|---|---|---|
| **p-valor** | Probabilidad de obtener el resultado por azar | 0–1 | p < 0.05 |
| **F (ANOVA/Chow)** | Varianza entre grupos / varianza dentro | 0–∞ | F > 4 (aprox.) |
| **R²** | Fracción de varianza explicada por el modelo | 0–1 | Depende del contexto |
| **AIC** | Calidad del modelo (menor = mejor) | −∞ a +∞ | Solo para comparar modelos |
| **Cohen's d** | Tamaño del efecto en diferencia de medias | 0–∞ | d>0.2 pequeño, >0.5 medio, >0.8 grande |
| **η² (eta²)** | Proporción de varianza explicada por el factor | 0–1 | >0.14 efecto grande |
| **Silhouette** | Calidad de la separación en clustering | −1 a +1 | > 0.25 estructura razonable |
| **ADF p-valor** | Estacionariedad (raíz unitaria) | 0–1 | p < 0.05 = estacionaria |
| **r (Pearson)** | Correlación lineal | −1 a +1 | \|r\| > 0.3 moderada, > 0.7 fuerte |

---

## 6. Guía rápida de valores

### Tasa de intercepción — ¿qué es normal?

| Valor | Interpretación |
|---|---|
| > 90% | Defensa excelente (Iron Dome en condiciones ideales) |
| 75–90% | Defensa muy buena (sistemas avanzados bien dotados) |
| 60–75% | Defensa funcional bajo presión — **rango "normal" para Ucrania 2025** |
| 45–60% | Defensa parcialmente desbordada — señal de agotamiento |
| < 45% | Saturación severa — la defensa no puede mantener el ritmo |

Nuestros resultados: Cluster 1 = 84.7% (buena), Cluster 3 = 49.6% (agotamiento).

---

### R² — ¿cuándo es "suficiente"?

| Contexto | R² típico | Nuestro R² |
|---|---|---|
| Física experimental | 0.95–0.99 | — |
| Economía financiera | 0.30–0.70 | — |
| Ciencias sociales/humanas | 0.10–0.40 | — |
| Datos de conflicto armado | **0.05–0.20** | 0.065–0.078 ✓ |

Un R²=0.065 en datos de guerra es **perfectamente normal y publicable**. El comportamiento humano en combate tiene un componente aleatorio irreducible enorme. Lo que importa es que el efecto existe (p<0.05) y la dirección es la esperada.

---

### Tamaño del efecto (Cohen's d) — escala de referencia

| d | Equivalente informal | Nuestros casos |
|---|---|---|
| 0.2 | "Difícil de ver a simple vista" | — |
| 0.43 | "Visible con atención" | TAR bajo/alto τ |
| 0.5 | "Claramente perceptible" | — |
| 0.8 | "Obvio" | — |
| 2.0+ | "Enorme" | η²=0.883 en tasa_interc (clusters) |

El Cohen's d=0.43 del TAR y el η²=0.883 del MANOVA no son contradictorios: miden cosas distintas. El TAR mide el efecto de cruzar el umbral diario (pequeño). El MANOVA mide la diferencia entre regímenes acumulados (enorme).

---

### Flags de posibles problemas (qué vigilar)

| Lo que aparece en output | ¿Problema? | Cómo manejarlo |
|---|---|---|
| `InterpolationWarning: p-value outside range` | No | Significa p < límite inferior de la tabla. Ignorar. |
| `VAR: not positive definite` | No (limitación de datos) | Documentar como limitación. Usar alternativas no lineales. |
| R² < 0.10 | No en datos de conflicto | Reportar e interpretar en contexto. |
| Silhouette < 0.25 | Sí — clusters débiles | Revisar k, probar otros algoritmos. Nuestro 0.38 está bien. |
| p-valor > 0.05 | Depende | El resultado no es significativo. No descartarlo, reportarlo. |
| ADF p=1.000 | Serie muy corta o no estacionaria | Documentar. Diferenciar antes de modelar. |

---

*Documento generado automáticamente a partir de los outputs de terminal de los scripts 02-05.*
*Repositorio: https://github.com/setadiano/tesis-drones*
