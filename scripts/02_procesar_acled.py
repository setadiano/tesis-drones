"""
=============================================================
TESIS: Uso Operacional de Drones en Conflictos Modernos
Script 02 — Procesamiento y Análisis ACLED
=============================================================
Input  : data/raw/acled_ukraine_2025_2026.csv
Outputs:
  data/processed/acled_drones_limpio.csv     ← solo eventos drone
  data/processed/acled_mensual_agregado.csv  ← agregado mensual
  outputs/figures/06_acled_eventos_diarios.png
  outputs/figures/07_acled_mapa_regiones.png
  outputs/figures/08_acled_tipos_eventos.png
  outputs/tables/03_acled_resumen_regiones.csv
  outputs/tables/04_acled_actores.csv
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ── Rutas ────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent.parent
DATA_R  = BASE / "data" / "raw"
DATA_P  = BASE / "data" / "processed"
FIGS    = BASE / "outputs" / "figures"
TABLES  = BASE / "outputs" / "tables"
DATA_P.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# ── Estilo ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
AZUL  = "#58a6ff"
VERDE = "#3fb950"
ROJO  = "#f85149"
AMBAR = "#d29922"
LILA  = "#bc8cff"

# ════════════════════════════════════════════════════════════
# 1. CARGA Y LIMPIEZA
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("PROCESAMIENTO ACLED — UCRANIA 2025-2026")
print("=" * 60)

df = pd.read_csv(DATA_R / "acled_ukraine_2025_2026.csv", low_memory=False)
df["event_date"] = pd.to_datetime(df["event_date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["event_date"])
df = df.sort_values("event_date").reset_index(drop=True)

print(f"\n✓ Total registros cargados : {len(df):,}")
print(f"  Rango fechas             : {df['event_date'].min().date()} → {df['event_date'].max().date()}")
print(f"  Columnas                 : {len(df.columns)}")
print(f"  Tipos de evento únicos   : {df['event_type'].nunique()}")
print(f"  Sub-eventos únicos       : {df['sub_event_type'].nunique()}")

# ── Filtrar eventos relacionados con drones/UAV/aéreos ───────
KEYWORDS_DRONE = [
    "drone", "uav", "shahed", "geran", "lancet", "fpv",
    "kamikaze", "unmanned", "air strike", "airstrike",
    "missile", "rocket", "loitering", "munition",
    "aerial", "air attack", "шахед", "герань", "дрон"
]

# Buscar en sub_event_type, notes, actor1, actor2
mask_drone = (
    df["sub_event_type"].str.lower().str.contains(
        "air|drone|missile|rocket|shelling", na=False
    ) |
    df["notes"].str.lower().str.contains(
        "|".join(KEYWORDS_DRONE), na=False, regex=True
    ) |
    df["event_type"].str.lower().str.contains(
        "explosion|remote|air", na=False
    )
)

df_drones = df[mask_drone].copy()

# Añadir columnas derivadas útiles
df_drones["anio"]         = df_drones["event_date"].dt.year
df_drones["mes"]          = df_drones["event_date"].dt.month
df_drones["mes_anio"]     = df_drones["event_date"].dt.to_period("M")
df_drones["semana"]       = df_drones["event_date"].dt.to_period("W")
df_drones["dia_semana"]   = df_drones["event_date"].dt.day_name()
df_drones["es_masivo"]    = df_drones["notes"].str.lower().str.contains(
    r"\b[3-9]\d{2}\b|\b[1-9]\d{3}\b", na=False, regex=True
)

# Clasificar tipo de arma por keywords en notes
def clasificar_arma(nota):
    if pd.isna(nota): return "desconocido"
    nota = nota.lower()
    if any(k in nota for k in ["shahed","geran","shaheed"]): return "Shahed/Geran"
    if any(k in nota for k in ["lancet","loitering"]): return "Lancet"
    if any(k in nota for k in ["fpv","first-person","fiber"]): return "FPV"
    if any(k in nota for k in ["ballistic","iskander","kinzhal"]): return "Balístico"
    if any(k in nota for k in ["cruise","kalibr","kh-101","x-101"]): return "Crucero"
    if any(k in nota for k in ["drone","uav","unmanned"]): return "Drone genérico"
    if any(k in nota for k in ["missile","misil","rocket"]): return "Misil/Cohete"
    return "otro"

df_drones["tipo_arma"] = df_drones["notes"].apply(clasificar_arma)

print(f"\n✓ Eventos drone/aéreos filtrados : {len(df_drones):,} ({len(df_drones)/len(df)*100:.1f}% del total)")
print(f"\n── Distribución tipo de arma ──")
print(df_drones["tipo_arma"].value_counts().to_string())

print(f"\n── Top 10 regiones afectadas (admin1) ──")
print(df_drones["admin1"].value_counts().head(10).to_string())

print(f"\n── Distribución sub_event_type ──")
print(df_drones["sub_event_type"].value_counts().head(10).to_string())

# ── Guardar dataset limpio ───────────────────────────────────
cols_guardar = ["event_id_cnty","event_date","anio","mes","mes_anio",
                "dia_semana","event_type","sub_event_type","actor1","actor2",
                "admin1","admin2","location","latitude","longitude",
                "fatalities","tipo_arma","es_masivo","notes","source","tags"]
cols_guardar = [c for c in cols_guardar if c in df_drones.columns]
df_drones[cols_guardar].to_csv(DATA_P / "acled_drones_limpio.csv", index=False)
print(f"\n✓ Dataset limpio guardado → acled_drones_limpio.csv ({len(df_drones):,} filas)")

# ════════════════════════════════════════════════════════════
# 2. AGREGADO MENSUAL
# ════════════════════════════════════════════════════════════
mensual = df_drones.groupby("mes_anio").agg(
    n_eventos        = ("event_id_cnty", "count"),
    n_bajas          = ("fatalities", "sum"),
    regiones_unicas  = ("admin1", "nunique"),
    localizaciones   = ("location", "nunique"),
    n_masivos        = ("es_masivo", "sum"),
).reset_index()
mensual["mes_anio_str"] = mensual["mes_anio"].astype(str)

mensual.to_csv(DATA_P / "acled_mensual_agregado.csv", index=False)
print(f"✓ Agregado mensual guardado → acled_mensual_agregado.csv")

# ── Resumen por región ───────────────────────────────────────
region_df = df_drones.groupby("admin1").agg(
    n_eventos   = ("event_id_cnty","count"),
    n_bajas     = ("fatalities","sum"),
    n_masivos   = ("es_masivo","sum"),
    lat_media   = ("latitude","mean"),
    lon_media   = ("longitude","mean"),
).sort_values("n_eventos", ascending=False).reset_index()
region_df.to_csv(TABLES / "03_acled_resumen_regiones.csv", index=False)
print(f"✓ Resumen regiones guardado → 03_acled_resumen_regiones.csv")

# ── Resumen por actor ────────────────────────────────────────
actor_df = df_drones.groupby("actor1").agg(
    n_eventos = ("event_id_cnty","count"),
    n_bajas   = ("fatalities","sum"),
).sort_values("n_eventos", ascending=False).head(20).reset_index()
actor_df.to_csv(TABLES / "04_acled_actores.csv", index=False)
print(f"✓ Resumen actores guardado → 04_acled_actores.csv")

# ════════════════════════════════════════════════════════════
# 3. FIGURA 6 — Eventos diarios ACLED en el tiempo
# ════════════════════════════════════════════════════════════
diario = df_drones.groupby("event_date").size().reset_index(name="n_eventos")
diario["rolling7"] = diario["n_eventos"].rolling(7, center=True).mean()

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(diario["event_date"], diario["n_eventos"],
       color=AZUL, alpha=0.4, width=1, label="Eventos diarios")
ax.plot(diario["event_date"], diario["rolling7"],
        color=ROJO, lw=2, label="Media móvil 7 días")

# Separador años
for anio in [2026]:
    ax.axvline(pd.Timestamp(f"{anio}-01-01"), color="#8b949e",
               lw=1, ls="--", alpha=0.7)
    ax.text(pd.Timestamp(f"{anio}-01-01"), ax.get_ylim()[1]*0.92,
            f" {anio}", color="#8b949e", fontsize=9)

ax.set_xlabel("Fecha")
ax.set_ylabel("Nº eventos")
ax.set_title("Eventos aéreos/drone diarios — ACLED Ucrania (datos verificados)", pad=12)
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(axis="y")
plt.tight_layout()
plt.savefig(FIGS / "06_acled_eventos_diarios.png", bbox_inches="tight")
plt.close()
print(f"\n✓ Figura 6 → 06_acled_eventos_diarios.png")

# ════════════════════════════════════════════════════════════
# 4. FIGURA 7 — Top regiones más afectadas
# ════════════════════════════════════════════════════════════
top_regiones = region_df.head(15)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top_regiones["admin1"][::-1],
               top_regiones["n_eventos"][::-1],
               color=AZUL, alpha=0.85)

# Colorear bajas en eje secundario
ax2 = ax.twiny()
ax2.barh(top_regiones["admin1"][::-1],
         top_regiones["n_bajas"][::-1],
         color=ROJO, alpha=0.4, label="Bajas confirmadas")
ax2.set_xlabel("Bajas confirmadas", color=ROJO)
ax2.tick_params(axis="x", colors=ROJO)

ax.set_xlabel("Nº de eventos")
ax.set_title("Top 15 regiones más afectadas — ACLED Ucrania", pad=12)

patch1 = mpatches.Patch(color=AZUL, alpha=0.85, label="Eventos")
patch2 = mpatches.Patch(color=ROJO, alpha=0.4,  label="Bajas")
ax.legend(handles=[patch1, patch2], fontsize=9, framealpha=0.3, loc="lower right")
ax.grid(axis="x")
plt.tight_layout()
plt.savefig(FIGS / "07_acled_regiones.png", bbox_inches="tight")
plt.close()
print(f"✓ Figura 7 → 07_acled_regiones.png")

# ════════════════════════════════════════════════════════════
# 5. FIGURA 8 — Tipos de arma y distribución temporal
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart tipos de arma
arma_counts = df_drones["tipo_arma"].value_counts()
colors_pie = [AZUL, ROJO, VERDE, AMBAR, LILA, "#ff7b72", "#79c0ff"]
axes[0].pie(arma_counts.values,
            labels=arma_counts.index,
            colors=colors_pie[:len(arma_counts)],
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "#e6edf3", "fontsize": 9})
axes[0].set_title("Distribución por tipo de arma\n(eventos ACLED)", pad=10)

# Evolución mensual por tipo arma (top 4)
top_armas = arma_counts.head(4).index.tolist()
df_arma_mes = df_drones[df_drones["tipo_arma"].isin(top_armas)].groupby(
    ["mes_anio","tipo_arma"]).size().unstack(fill_value=0)

colores_arma = [AZUL, ROJO, VERDE, AMBAR]
for i, arma in enumerate(top_armas):
    if arma in df_arma_mes.columns:
        axes[1].plot(range(len(df_arma_mes)),
                     df_arma_mes[arma],
                     marker="o", ms=4, lw=2,
                     color=colores_arma[i], label=arma)

axes[1].set_xticks(range(len(df_arma_mes)))
axes[1].set_xticklabels([str(p) for p in df_arma_mes.index],
                         rotation=45, ha="right", fontsize=7)
axes[1].set_ylabel("Nº eventos")
axes[1].set_title("Evolución mensual por tipo de arma", pad=10)
axes[1].legend(fontsize=8, framealpha=0.3)
axes[1].grid(axis="y")

plt.suptitle("Análisis de tipos de arma — ACLED Ucrania 2025-2026",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIGS / "08_acled_tipos_arma.png", bbox_inches="tight")
plt.close()
print(f"✓ Figura 8 → 08_acled_tipos_arma.png")

# ════════════════════════════════════════════════════════════
# 6. RESUMEN FINAL
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PROCESAMIENTO ACLED COMPLETADO")
print("=" * 60)
print(f"\nDataset completo   : {len(df):,} eventos totales")
print(f"Eventos drone/aéreo: {len(df_drones):,} eventos filtrados")
print(f"Rango temporal     : {df['event_date'].min().date()} → {df['event_date'].max().date()}")
print(f"Regiones cubiertas : {df_drones['admin1'].nunique()} oblasts")
print(f"Bajas totales      : {int(df_drones['fatalities'].sum()):,}")
print(f"\nArchivos generados:")
print(f"  data/processed/acled_drones_limpio.csv")
print(f"  data/processed/acled_mensual_agregado.csv")
print(f"  outputs/figures/06_acled_eventos_diarios.png")
print(f"  outputs/figures/07_acled_regiones.png")
print(f"  outputs/figures/08_acled_tipos_arma.png")
print(f"  outputs/tables/03_acled_resumen_regiones.csv")
print(f"  outputs/tables/04_acled_actores.csv")
print(f"\nPróximo: scripts/03_series_temporales.py")
