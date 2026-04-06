"""
=============================================================
TESIS: Uso Operacional de Drones en Conflictos Modernos
Script 01 — Exploración Inicial de Datos
=============================================================
Autor  : Análisis OSINT
Fecha  : Abril 2026
Fuente : ISIS-Online, ACLED, CSIS, Ukraine AF, IDF, ISW

Outputs generados (en outputs/):
  figures/01_lanzamientos_mensuales_UR.png
  figures/02_tasas_intercepcion_hit_UR.png
  figures/03_strike_vs_decoy_UR.png
  figures/04_operaciones_iran_israel.png
  figures/05_comparativa_teatros.png
  tables/01_estadisticos_descriptivos.csv
  tables/02_correlaciones_UR.csv
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Rutas ────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parent.parent
DATA    = BASE / "data" / "raw"
FIGS    = BASE / "outputs" / "figures"
TABLES  = BASE / "outputs" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# ── Estilo global ────────────────────────────────────────────
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

AZUL   = "#58a6ff"
VERDE  = "#3fb950"
ROJO   = "#f85149"
AMBAR  = "#d29922"
LILA   = "#bc8cff"

# ════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("TESIS DRONES — EXPLORACIÓN INICIAL")
print("=" * 60)

ur = pd.read_csv(DATA / "db_drones_ucrania_rusia_2025_2026.csv")
ii = pd.read_csv(DATA / "db_drones_iran_israel_2025_2026.csv")

# Parsear fechas (tomamos fecha de inicio del período)
ur["fecha_inicio"] = pd.to_datetime(ur["fecha"].str.split("/").str[0])
ur = ur.sort_values("fecha_inicio").reset_index(drop=True)

# Etiqueta mes-año para gráficos
ur["mes_label"] = ur["fecha_inicio"].dt.strftime("%b\n%Y")

print(f"\n✓ Dataset Ucrania-Rusia   : {len(ur)} registros")
print(f"✓ Dataset Irán-Israel     : {len(ii)} registros")

# ════════════════════════════════════════════════════════════
# 2. ESTADÍSTICOS DESCRIPTIVOS
# ════════════════════════════════════════════════════════════
print("\n── ESTADÍSTICOS DESCRIPTIVOS (Ucrania-Rusia) ──")

cols_num = ["lanzamientos_total", "strike_uav", "decoy_uav",
            "intercepciones", "hits",
            "tasa_intercepcion_pct", "tasa_hit_pct"]

desc = ur[cols_num].describe().round(2)
print(desc.to_string())

# Guardar tabla
desc.to_csv(TABLES / "01_estadisticos_descriptivos.csv")
print(f"\n✓ Tabla guardada → {TABLES / '01_estadisticos_descriptivos.csv'}")

# Correlaciones
corr = ur[cols_num].corr().round(3)
corr.to_csv(TABLES / "02_correlaciones_UR.csv")
print(f"✓ Correlaciones guardadas → {TABLES / '02_correlaciones_UR.csv'}")

# Datos clave a mano
print("\n── CIFRAS CLAVE 2025 ──")
ur_2025 = ur[ur["fecha_inicio"].dt.year == 2025]
print(f"  Total lanzamientos 2025       : {ur_2025['lanzamientos_total'].sum():,.0f}")
print(f"  Total hits 2025               : {ur_2025['hits'].sum():,.0f}")
print(f"  Tasa hit media 2025           : {ur_2025['tasa_hit_pct'].mean():.1f}%")
print(f"  Tasa intercepción media 2025  : {ur_2025['tasa_intercepcion_pct'].mean():.1f}%")
print(f"  Mes con más lanzamientos      : {ur_2025.loc[ur_2025['lanzamientos_total'].idxmax(), 'mes_label'].replace(chr(10),' ')}")
print(f"  Mes con mayor tasa hit        : {ur_2025.loc[ur_2025['tasa_hit_pct'].idxmax(), 'mes_label'].replace(chr(10),' ')} ({ur_2025['tasa_hit_pct'].max():.1f}%)")

# ════════════════════════════════════════════════════════════
# 3. FIGURA 1 — Lanzamientos mensuales totales
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))

x = range(len(ur))
bars = ax.bar(x, ur["lanzamientos_total"], color=AZUL, alpha=0.85, width=0.7, label="Total lanzamientos")
ax.plot(x, ur["strike_uav"], color=ROJO, marker="o", ms=5, lw=2, label="Strike UAV")
ax.plot(x, ur["decoy_uav"],  color=AMBAR, marker="s", ms=4, lw=1.5, ls="--", label="Decoy UAV")

# Línea de tendencia lanzamientos totales
z = np.polyfit(list(x), ur["lanzamientos_total"].fillna(0), 1)
p = np.poly1d(z)
ax.plot(x, p(list(x)), color=VERDE, lw=1.5, ls=":", alpha=0.8, label="Tendencia")

# Anotar máximo
idx_max = ur["lanzamientos_total"].idxmax()
ax.annotate(
    f"RÉCORD\n{int(ur.loc[idx_max,'lanzamientos_total']):,}",
    xy=(idx_max, ur.loc[idx_max, "lanzamientos_total"]),
    xytext=(idx_max - 1.5, ur.loc[idx_max, "lanzamientos_total"] + 300),
    color=ROJO, fontsize=9, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=ROJO, lw=1.2)
)

ax.set_xticks(list(x))
ax.set_xticklabels(ur["mes_label"], fontsize=8)
ax.set_ylabel("Nº de UAVs")
ax.set_title("Lanzamientos mensuales de Shahed/Geran — Teatro Ucrania-Rusia (2025-2026)", pad=12)
ax.legend(loc="upper left", fontsize=9, framealpha=0.3)
ax.grid(axis="y")

# Separador 2025/2026
idx_2026 = ur[ur["fecha_inicio"].dt.year == 2026].index[0]
ax.axvline(x=idx_2026 - 0.5, color="#8b949e", lw=1, ls="--", alpha=0.6)
ax.text(idx_2026 - 0.4, ax.get_ylim()[1] * 0.95, "2026 →", color="#8b949e", fontsize=8)

plt.tight_layout()
plt.savefig(FIGS / "01_lanzamientos_mensuales_UR.png", bbox_inches="tight")
plt.close()
print(f"\n✓ Figura 1 guardada → 01_lanzamientos_mensuales_UR.png")

# ════════════════════════════════════════════════════════════
# 4. FIGURA 2 — Tasas de intercepción y hit
# ════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(14, 5))

ax2 = ax1.twinx()

ax1.fill_between(x, ur["tasa_intercepcion_pct"], alpha=0.25, color=VERDE)
ax1.plot(x, ur["tasa_intercepcion_pct"], color=VERDE, marker="o", ms=5, lw=2,
         label="Tasa intercepción (%)")

ax2.fill_between(x, ur["tasa_hit_pct"], alpha=0.25, color=ROJO)
ax2.plot(x, ur["tasa_hit_pct"], color=ROJO, marker="^", ms=5, lw=2,
         label="Tasa hit/impacto (%)")

# Zona de saturación (>400 UAVs → tasa hit baja)
for i, row in ur.iterrows():
    if pd.notna(row["lanzamientos_total"]) and row["lanzamientos_total"] > 4000:
        ax1.axvspan(i - 0.4, i + 0.4, alpha=0.1, color=AMBAR)

ax1.set_xticks(list(x))
ax1.set_xticklabels(ur["mes_label"], fontsize=8)
ax1.set_ylabel("Tasa intercepción (%)", color=VERDE)
ax2.set_ylabel("Tasa hit (%)", color=ROJO)
ax1.tick_params(axis="y", colors=VERDE)
ax2.tick_params(axis="y", colors=ROJO)
ax1.set_title("Tasas de intercepción vs. impacto — Teatro Ucrania-Rusia\n(zonas naranjas = meses con >4.000 lanzamientos)", pad=12)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9, framealpha=0.3)
ax1.grid(axis="y")

plt.tight_layout()
plt.savefig(FIGS / "02_tasas_intercepcion_hit_UR.png", bbox_inches="tight")
plt.close()
print(f"✓ Figura 2 guardada → 02_tasas_intercepcion_hit_UR.png")

# ════════════════════════════════════════════════════════════
# 5. FIGURA 3 — Composición Strike vs Decoy
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))

ur_plot = ur.dropna(subset=["strike_uav", "decoy_uav"])
x3 = range(len(ur_plot))

ax.bar(x3, ur_plot["strike_uav"], color=ROJO,  alpha=0.85, width=0.7, label="Strike UAV (Shahed/Geran)")
ax.bar(x3, ur_plot["decoy_uav"],  color=AMBAR, alpha=0.85, width=0.7,
       bottom=ur_plot["strike_uav"], label="Decoy UAV (Gerbera/Parody)")

# Ratio decoy en eje secundario
ax2 = ax.twinx()
ratio = (ur_plot["decoy_uav"] / ur_plot["lanzamientos_total"] * 100).fillna(0)
ax2.plot(list(x3), ratio, color=LILA, marker="D", ms=4, lw=2,
         label="% Decoy / Total")
ax2.set_ylabel("% Decoy", color=LILA)
ax2.tick_params(axis="y", colors=LILA)
ax2.set_ylim(0, 60)

ax.set_xticks(list(x3))
ax.set_xticklabels(ur_plot["mes_label"], fontsize=8)
ax.set_ylabel("Nº de UAVs")
ax.set_title("Composición de oleadas: Strike vs. Decoy — Teatro Ucrania-Rusia", pad=12)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9, framealpha=0.3)
ax.grid(axis="y")

plt.tight_layout()
plt.savefig(FIGS / "03_strike_vs_decoy_UR.png", bbox_inches="tight")
plt.close()
print(f"✓ Figura 3 guardada → 03_strike_vs_decoy_UR.png")

# ════════════════════════════════════════════════════════════
# 6. FIGURA 4 — Teatro Irán-Israel: tasas por operación
# ════════════════════════════════════════════════════════════
ii_plot = ii.dropna(subset=["tasa_intercepcion_pct"]).copy()

fig, ax = plt.subplots(figsize=(12, 5))
x4 = range(len(ii_plot))

bars_int = ax.bar(x4, ii_plot["tasa_intercepcion_pct"],
                  color=VERDE, alpha=0.85, width=0.6, label="Tasa intercepción (%)")

# Colorear barra de Rising Lion diferente
for i, (idx, row) in enumerate(ii_plot.iterrows()):
    if "RISLION" in str(row.get("oleada_id", "")):
        bars_int[i].set_color(AZUL)
        bars_int[i].set_label("Rising Lion (Israel → Iran)")

ax.axhline(y=90, color=AMBAR, lw=1.5, ls="--", alpha=0.7, label="Umbral 90%")
ax.set_ylim(70, 105)
ax.set_xticks(list(x4))
ax.set_xticklabels(
    [r["oleada_id"].replace("II-","").replace("_","\n") for _, r in ii_plot.iterrows()],
    fontsize=8
)
ax.set_ylabel("Tasa de intercepción (%)")
ax.set_title("Tasa de intercepción por operación — Teatro Irán-Israel/GCC (2025-2026)", pad=12)
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(axis="y")

plt.tight_layout()
plt.savefig(FIGS / "04_operaciones_iran_israel.png", bbox_inches="tight")
plt.close()
print(f"✓ Figura 4 guardada → 04_operaciones_iran_israel.png")

# ════════════════════════════════════════════════════════════
# 7. FIGURA 5 — Comparativa tasas de intercepción ambos teatros
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

# Ucrania-Rusia
ur_int = ur["tasa_intercepcion_pct"].dropna()
# Irán-Israel
ii_int = ii_plot["tasa_intercepcion_pct"].dropna()

data_box = [ur_int.values, ii_int.values]
labels_box = ["Ucrania-Rusia\n(táctico)", "Irán-Israel/GCC\n(estratégico)"]
colors_box = [AZUL, ROJO]

bp = ax.boxplot(data_box, patch_artist=True, widths=0.4,
                medianprops=dict(color="white", lw=2),
                whiskerprops=dict(color="#8b949e"),
                capprops=dict(color="#8b949e"),
                flierprops=dict(marker="o", color="#8b949e", ms=5))

for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Overlay puntos individuales
for i, (data, color) in enumerate(zip(data_box, colors_box), 1):
    jitter = np.random.uniform(-0.1, 0.1, size=len(data))
    ax.scatter(np.full(len(data), i) + jitter, data,
               color=color, alpha=0.8, s=40, zorder=5)

# Medias
for i, data in enumerate(data_box, 1):
    ax.scatter(i, np.mean(data), color="white", s=80,
               marker="D", zorder=6, label=f"Media: {np.mean(data):.1f}%" if i == 1 else f"Media: {np.mean(data):.1f}%")

ax.set_xticklabels(labels_box, fontsize=11)
ax.set_ylabel("Tasa de intercepción (%)")
ax.set_title("Comparativa de tasas de intercepción por teatro\n(cada punto = una oleada/mes)", pad=12)
ax.legend(fontsize=9, framealpha=0.3)
ax.grid(axis="y")

plt.tight_layout()
plt.savefig(FIGS / "05_comparativa_teatros.png", bbox_inches="tight")
plt.close()
print(f"✓ Figura 5 guardada → 05_comparativa_teatros.png")

# ════════════════════════════════════════════════════════════
# 8. RESUMEN FINAL
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPLORACIÓN COMPLETADA")
print("=" * 60)
print(f"\nFiguras generadas en  : {FIGS}")
print(f"Tablas generadas en   : {TABLES}")
print("\nPróximos scripts:")
print("  02_series_temporales.py  — ARIMA + detección cambio régimen")
print("  03_umbral_saturacion.py  — Regresión no lineal + TAR")
print("  04_analisis_multivariante.py — PCA + clustering")
