# Tesis: Uso Operacional de Drones en Conflictos Modernos (2025-2026)

Análisis cuantitativo comparado de los teatros Ucrania-Rusia (táctico) e Irán-Israel/GCC (estratégico).

## Estructura del proyecto

```
tesis-drones/
├── data/
│   ├── raw/                  # Datos originales OSINT (no modificar)
│   │   ├── db_drones_ucrania_rusia_2025_2026.csv
│   │   ├── db_drones_iran_israel_2025_2026.csv
│   │   └── db_drones_COMBINADO_2025_2026.json
│   └── processed/            # Datos procesados y limpios
├── scripts/                  # Scripts Python de análisis
├── docs/                     # Documentación e informes
│   └── INFORME_MAESTRO_DRONES_2025_2026.md
└── outputs/
    ├── figures/              # Gráficos generados
    └── tables/               # Tablas de resultados
```

## Fuentes principales
- ISIS-Online (análisis mensual Shahed)
- ACLED (eventos georreferenciados)
- CSIS Futures Lab (dashboard interactivo)
- Ukraine Air Force (reportes diarios)
- CriticalThreats/ISW (teatro Irán-Israel)

## Metodología prevista
- Series temporales (ARIMA/VAR/Markov Switching)
- Análisis umbral de saturación (TAR/SETAR)
- Análisis multivariante comparativo (PCA/MANOVA)
- Modelos predictivos (LSTM)
