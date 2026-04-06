"""
Script 08 — Análisis Horario: Ventanas de Vulnerabilidad Nocturna
=================================================================
Autor : setadiano / jplatas6@alumno.uned.es
Fecha : Abril 2026

Preguntas de inteligencia
--------------------------
Q1  ¿A qué hora lanza Rusia? ¿Hay una ventana de lanzamiento fija?
Q2  ¿A qué hora terminan los ataques (impactos finales)? ¿Hay un
    patrón de "hora de mayor vulnerabilidad"?
Q3  ¿Cuánto dura un ataque? ¿Ha cambiado la duración con el tiempo?
Q4  ¿Hay diferencia entre ataques nocturnos y diurnos en volumen
    e intercepción?
Q5  ¿Existe un día de la semana con mayor actividad?
Q6  ¿Ha evolucionado la hora de lanzamiento a lo largo de 2025-2026?
    (¿adapta Rusia el timing para evitar los turnos defensivos?)

Inputs
------
  data/raw/petro_attacks_2025_2026.csv

Outputs
-------
  data/processed/shahed_horario.csv
  outputs/08_*.png  (7 figuras)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, kruskal, f_oneway
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle, FancyArrowPatch
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT  = ROOT / "outputs"

# ── Estilo dark ────────────────────────────────────────────────────────────
BG     = "#0d1117"
FG     = "#e6edf3"
ACCENT = "#58a6ff"
RED    = "#ff7b72"
GREEN  = "#3fb950"
YELLOW = "#d29922"
PURPLE = "#bc8cff"
ORANGE = "#ffa657"
GRID_C = "#21262d"

plt.rcParams.update({
    "figure.facecolor" : BG, "axes.facecolor"  : BG,
    "axes.edgecolor"   : GRID_C, "axes.labelcolor": FG,
    "axes.titlecolor"  : FG, "xtick.color"     : FG,
    "ytick.color"      : FG, "text.color"      : FG,
    "grid.color"       : GRID_C, "grid.alpha"  : 0.5,
    "legend.facecolor" : "#161b22", "legend.edgecolor": GRID_C,
    "font.family"      : "monospace", "figure.dpi"  : 120,
})

FIGS_OK = 0

def save_fig(name):
    global FIGS_OK
    plt.savefig(OUT / name, bbox_inches="tight", facecolor=BG, dpi=150)
    plt.close()
    FIGS_OK += 1
    print(f"  [✓] {name}")

# ══════════════════════════════════════════════════════════════════════════
# 0.  CARGA Y PREPROCESADO
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SCRIPT 08 — ANÁLISIS HORARIO")
print("="*60)

print("\n[0] Cargando y preprocesando …")
df_raw = pd.read_csv(RAW / "petro_attacks_2025_2026.csv")
df = df_raw[df_raw["model"].str.contains("Shahed", na=False)].copy()

df["ts"] = pd.to_datetime(df["time_start"], errors="coerce")
df["te"] = pd.to_datetime(df["time_end"],   errors="coerce")
df["fecha"] = pd.to_datetime(df["fecha"],   errors="coerce")

df = df.dropna(subset=["ts","launched","destroyed"]).copy()
df = df[df["launched"] > 0].copy()

df["hora_inicio"]   = df["ts"].dt.hour
df["hora_fin"]      = df["te"].dt.hour
df["duracion_h"]    = (df["te"] - df["ts"]).dt.total_seconds() / 3600
df["dia_semana"]    = df["ts"].dt.dayofweek          # 0=lunes
df["dia_semana_n"]  = df["ts"].dt.day_name()
df["mes"]           = df["ts"].dt.to_period("M")
df["semana"]        = df["ts"].dt.to_period("W")
df["trimestre"]     = df["ts"].dt.to_period("Q")

df["tasa_intercep"] = df["destroyed"] / df["launched"]
df["tasa_hit"]      = 1 - df["tasa_intercep"]
df["impactos"]      = df["launched"] - df["destroyed"]

# Clasificar turno
# Turno NOCHE: lanzamiento 17:00-23:59 (hora local Moscú ≈ UTC+3)
# Turno DÍA:   lanzamiento 06:00-16:59
def turno(h):
    if 17 <= h <= 23:
        return "Nocturno (17-23h)"
    elif 0 <= h <= 5:
        return "Madrugada (0-5h)"
    else:
        return "Diurno (6-16h)"

df["turno"] = df["hora_inicio"].apply(turno)

# Períodos doctrinales (de Scripts anteriores)
def periodo(d):
    if pd.Timestamp("2025-01-01") <= d <= pd.Timestamp("2025-07-31"):
        return "P1: Saturación\n(Ene-Jul 2025)"
    elif pd.Timestamp("2025-08-01") <= d <= pd.Timestamp("2025-12-31"):
        return "P2: Transición\n(Ago-Dic 2025)"
    else:
        return "P3: Nueva normalidad\n(Ene-Abr 2026)"

df["periodo"] = df["fecha"].apply(periodo)

print(f"  Eventos Shahed con hora: {len(df)}")
print(f"  Rango: {df['ts'].min()} → {df['ts'].max()}")
print(f"  Turnos: {df['turno'].value_counts().to_dict()}")
print(f"  Duración media: {df['duracion_h'].mean():.1f}h  "
      f"std: {df['duracion_h'].std():.1f}h")

df.to_csv(PROC / "shahed_horario.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════
# 1.  Q1-Q2: DISTRIBUCIÓN HORA DE LANZAMIENTO Y FIN
# ══════════════════════════════════════════════════════════════════════════
print("\n[1] Q1-Q2: Distribución horaria …")

# Hora de inicio ponderada por volumen
hora_vol = (df.groupby("hora_inicio")
              .agg(n_ataques=("launched","count"),
                   total_lanzados=("launched","sum"),
                   media_intercep=("tasa_intercep","mean"))
              .reset_index())

hora_fin_vol = (df.dropna(subset=["hora_fin"])
                  .groupby("hora_fin")
                  .agg(n_ataques=("launched","count"),
                       total_lanzados=("launched","sum"))
                  .reset_index())

# ── Fig 1: Distribución horaria lanzamiento + fin ─────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Q1-Q2: ¿A qué hora lanza Rusia y cuándo terminan los ataques?",
             color=FG, fontsize=13, y=1.01)

ax_nl, ax_nf, ax_vl, ax_int = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

# Nº ataques por hora inicio
all_hours = pd.DataFrame({"hora_inicio": range(24)})
hv = all_hours.merge(hora_vol, on="hora_inicio", how="left").fillna(0)
colors_h = [RED if 17 <= h <= 23 else
            (PURPLE if h <= 5 else ACCENT)
            for h in hv["hora_inicio"]]
bars = ax_nl.bar(hv["hora_inicio"], hv["n_ataques"],
                 color=colors_h, alpha=0.85, edgecolor=BG, lw=0.3)
ax_nl.set_xlabel("Hora de lanzamiento (hora local Moscú)", color=FG)
ax_nl.set_ylabel("Nº ataques", color=FG)
ax_nl.set_title("Frecuencia de lanzamiento por hora", color=FG, fontsize=10)
ax_nl.set_xticks(range(0, 24, 2))
ax_nl.axvspan(17, 24, alpha=0.07, color=RED)
ax_nl.axvspan(0,   6, alpha=0.07, color=PURPLE)
ax_nl.text(19, hv["n_ataques"].max()*0.85, "Ventana\nprincipal",
           color=RED, fontsize=8, ha="center")
ax_nl.grid(True, alpha=0.3, axis="y")
ax_nl.set_facecolor(BG)

# Nº ataques por hora fin
all_hours2 = pd.DataFrame({"hora_fin": range(24)})
hf = all_hours2.merge(hora_fin_vol, on="hora_fin", how="left").fillna(0)
colors_f = [GREEN if 6 <= h <= 11 else ACCENT for h in hf["hora_fin"]]
bars2 = ax_nf.bar(hf["hora_fin"], hf["n_ataques"],
                  color=colors_f, alpha=0.85, edgecolor=BG, lw=0.3)
ax_nf.set_xlabel("Hora de fin / último impacto (hora local)", color=FG)
ax_nf.set_ylabel("Nº ataques", color=FG)
ax_nf.set_title("Distribución hora de finalización", color=FG, fontsize=10)
ax_nf.set_xticks(range(0, 24, 2))
ax_nf.axvspan(6, 12, alpha=0.07, color=GREEN)
ax_nf.text(9, hf["n_ataques"].max()*0.85, "Ventana\nde impacto",
           color=GREEN, fontsize=8, ha="center")
ax_nf.grid(True, alpha=0.3, axis="y")
ax_nf.set_facecolor(BG)

# Volumen total lanzado por hora inicio
ax_vl.bar(hv["hora_inicio"], hv["total_lanzados"],
          color=colors_h, alpha=0.85, edgecolor=BG, lw=0.3)
ax_vl.set_xlabel("Hora de lanzamiento", color=FG)
ax_vl.set_ylabel("Total UAV lanzados", color=FG)
ax_vl.set_title("Volumen total por hora de lanzamiento", color=FG, fontsize=10)
ax_vl.set_xticks(range(0, 24, 2))
ax_vl.axvspan(17, 24, alpha=0.07, color=RED)
ax_vl.grid(True, alpha=0.3, axis="y")
ax_vl.set_facecolor(BG)

# Tasa intercepción por hora
ax_int.bar(hv["hora_inicio"],
           hv["media_intercep"].replace(0, np.nan) * 100,
           color=colors_h, alpha=0.85, edgecolor=BG, lw=0.3)
ax_int.set_xlabel("Hora de lanzamiento", color=FG)
ax_int.set_ylabel("Tasa intercepción media (%)", color=FG)
ax_int.set_title("Intercepción media por hora de lanzamiento", color=FG, fontsize=10)
ax_int.set_xticks(range(0, 24, 2))
ax_int.axhline(80, color=YELLOW, lw=1, ls="--", alpha=0.7, label="80%")
ax_int.legend(fontsize=9)
ax_int.grid(True, alpha=0.3, axis="y")
ax_int.set_facecolor(BG)

# Leyenda de colores
for ax in axes.flat:
    ax.plot([], [], color=RED,    lw=8, alpha=0.5, label="Nocturno 17-23h")
    ax.plot([], [], color=PURPLE, lw=8, alpha=0.5, label="Madrugada 0-5h")
    ax.plot([], [], color=ACCENT, lw=8, alpha=0.5, label="Diurno 6-16h")

plt.tight_layout()
save_fig("08_fig1_distribucion_horaria.png")

# ══════════════════════════════════════════════════════════════════════════
# 2.  Q3: DURACIÓN DEL ATAQUE Y EVOLUCIÓN TEMPORAL
# ══════════════════════════════════════════════════════════════════════════
print("\n[2] Q3: Duración y evolución temporal …")

df_dur = df.dropna(subset=["duracion_h"]).copy()
df_dur = df_dur[(df_dur["duracion_h"] > 0) & (df_dur["duracion_h"] <= 24)]

# Evolución mensual duración
dur_mensual = (df_dur.groupby("mes")["duracion_h"]
               .agg(["mean","std","count"])
               .reset_index())
dur_mensual["mes_ts"] = dur_mensual["mes"].dt.to_timestamp()

# ── Fig 2: Duración temporal ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle("Q3: Duración de los ataques Shahed — ¿cambia con el tiempo?",
             color=FG, fontsize=12, y=1.01)

ax1, ax2, ax3 = axes

# Histograma duración
ax1.hist(df_dur["duracion_h"], bins=30, color=ACCENT, alpha=0.8,
         edgecolor=BG, lw=0.3)
ax1.axvline(df_dur["duracion_h"].mean(), color=YELLOW, lw=2,
            ls="--", label=f"Media: {df_dur['duracion_h'].mean():.1f}h")
ax1.axvline(df_dur["duracion_h"].median(), color=GREEN, lw=2,
            ls=":", label=f"Mediana: {df_dur['duracion_h'].median():.1f}h")
ax1.set_xlabel("Duración (horas)", color=FG)
ax1.set_ylabel("Frecuencia", color=FG)
ax1.set_title("Distribución duración", color=FG, fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor(BG)

# Evolución mensual
ax2.plot(dur_mensual["mes_ts"], dur_mensual["mean"],
         color=ACCENT, lw=2, marker="o", ms=5, label="Media duración")
ax2.fill_between(dur_mensual["mes_ts"],
                 dur_mensual["mean"] - dur_mensual["std"],
                 dur_mensual["mean"] + dur_mensual["std"],
                 alpha=0.2, color=ACCENT)
ax2.axhline(df_dur["duracion_h"].mean(), color=YELLOW, lw=1,
            ls="--", alpha=0.6)
ax2.set_xlabel("Mes", color=FG)
ax2.set_ylabel("Duración media (horas)", color=FG)
ax2.set_title("Evolución mensual", color=FG, fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor(BG)

# Duración por período doctrinal
periodos_ord = ["P1: Saturación\n(Ene-Jul 2025)",
                "P2: Transición\n(Ago-Dic 2025)",
                "P3: Nueva normalidad\n(Ene-Abr 2026)"]
colors_p = [RED, YELLOW, GREEN]
data_bp = [df_dur[df_dur["periodo"]==p]["duracion_h"].dropna().values
           for p in periodos_ord]
bp = ax3.boxplot(data_bp,
                 labels=["P1\nSaturac.", "P2\nTransic.", "P3\nNormal."],
                 patch_artist=True,
                 medianprops={"color": FG, "lw": 2},
                 flierprops={"marker": "o", "markerfacecolor": FG,
                             "markersize": 3, "alpha": 0.4})
for patch, color in zip(bp["boxes"], colors_p):
    patch.set_facecolor(color + "55")
    patch.set_edgecolor(color)

# Test Kruskal-Wallis entre períodos
if all(len(d) > 2 for d in data_bp):
    kw_stat, kw_p = kruskal(*[d for d in data_bp if len(d) > 0])
    ax3.set_title(f"Duración por período — Kruskal p={kw_p:.3f}"
                  f"  {'★' if kw_p < 0.05 else 'ns'}",
                  color=YELLOW if kw_p < 0.05 else FG, fontsize=9)
else:
    ax3.set_title("Duración por período", color=FG, fontsize=10)

medias_p = [np.mean(d) if len(d) > 0 else 0 for d in data_bp]
for i, (m, c) in enumerate(zip(medias_p, colors_p), 1):
    ax3.text(i, m + 0.3, f"{m:.1f}h", ha="center", va="bottom",
             fontsize=9, color=c)

ax3.set_ylabel("Duración (horas)", color=FG)
ax3.grid(True, alpha=0.3, axis="y")
ax3.set_facecolor(BG)

plt.tight_layout()
save_fig("08_fig2_duracion_ataques.png")

# ══════════════════════════════════════════════════════════════════════════
# 3.  Q4: TURNO NOCTURNO VS DIURNO
# ══════════════════════════════════════════════════════════════════════════
print("\n[3] Q4: Comparativa nocturno vs diurno …")

turnos = df.groupby("turno").agg(
    n_ataques      = ("launched","count"),
    total_lanzados = ("launched","sum"),
    media_lanzados = ("launched","mean"),
    media_intercep = ("tasa_intercep","mean"),
    media_hit      = ("tasa_hit","mean"),
    total_impactos = ("impactos","sum"),
).reset_index()

print(f"  {turnos.to_string(index=False)}")

# Test ANOVA intercepción entre turnos
grupos = [df[df["turno"]==t]["tasa_intercep"].dropna().values
          for t in df["turno"].unique()]
if all(len(g) > 2 for g in grupos):
    f_stat, p_anova = f_oneway(*grupos)
    print(f"\n  ANOVA intercepción entre turnos: F={f_stat:.2f}  p={p_anova:.3f}")
else:
    p_anova = 1.0
    f_stat = 0.0

# ── Fig 3: Nocturno vs Diurno ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
fig.suptitle("Q4: ¿Importa si el ataque es nocturno o diurno?",
             color=FG, fontsize=12, y=1.01)

ax1, ax2, ax3 = axes

turno_colors = {
    "Nocturno (17-23h)"  : RED,
    "Madrugada (0-5h)"   : PURPLE,
    "Diurno (6-16h)"     : ACCENT,
}

# Barras volumen por turno
tcolors = [turno_colors.get(t, ACCENT) for t in turnos["turno"]]
bars = ax1.bar(turnos["turno"], turnos["media_lanzados"],
               color=tcolors, alpha=0.85, edgecolor=BG, lw=0.5)
for bar, val in zip(bars, turnos["media_lanzados"]):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 2, f"{val:.0f}",
             ha="center", va="bottom", fontsize=10, color=FG)
ax1.set_ylabel("Media UAV lanzados por ataque", color=FG)
ax1.set_title("Volumen medio por turno", color=FG, fontsize=10)
ax1.tick_params(axis="x", labelsize=8)
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_facecolor(BG)

# Intercepción por turno
bars2 = ax2.bar(turnos["turno"], turnos["media_intercep"] * 100,
                color=tcolors, alpha=0.85, edgecolor=BG, lw=0.5)
for bar, val in zip(bars2, turnos["media_intercep"]):
    ax2.text(bar.get_x() + bar.get_width()/2,
             val * 100 + 0.5, f"{val*100:.1f}%",
             ha="center", va="bottom", fontsize=10, color=FG)
ax2.set_ylabel("Tasa intercepción media (%)", color=FG)
ax2.set_title(f"Intercepción por turno\nANOVA p={p_anova:.3f} "
              f"{'★' if p_anova < 0.05 else 'ns'}",
              color=YELLOW if p_anova < 0.05 else FG, fontsize=9)
ax2.axhline(80, color=YELLOW, lw=1, ls="--", alpha=0.6)
ax2.tick_params(axis="x", labelsize=8)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_facecolor(BG)

# Impactos totales por turno
bars3 = ax3.bar(turnos["turno"], turnos["total_impactos"],
                color=tcolors, alpha=0.85, edgecolor=BG, lw=0.5)
for bar, val in zip(bars3, turnos["total_impactos"]):
    ax3.text(bar.get_x() + bar.get_width()/2,
             val + 50, f"{int(val):,}",
             ha="center", va="bottom", fontsize=10, color=FG)
ax3.set_ylabel("Total impactos acumulados", color=FG)
ax3.set_title("Impactos totales por turno", color=FG, fontsize=10)
ax3.tick_params(axis="x", labelsize=8)
ax3.grid(True, alpha=0.3, axis="y")
ax3.set_facecolor(BG)

plt.tight_layout()
save_fig("08_fig3_nocturno_vs_diurno.png")

# ══════════════════════════════════════════════════════════════════════════
# 4.  Q5: DÍA DE LA SEMANA
# ══════════════════════════════════════════════════════════════════════════
print("\n[4] Q5: Patrón día de la semana …")

dias_ord = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dias_es  = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]

diasem = (df.groupby("dia_semana_n")
            .agg(n_ataques      = ("launched","count"),
                 media_lanzados = ("launched","mean"),
                 media_intercep = ("tasa_intercep","mean"),
                 total_impactos = ("impactos","sum"))
            .reindex(dias_ord)
            .reset_index())
diasem["dia_es"] = dias_es

# Chi2 uniformidad (¿distribución no uniforme?)
chi2_dias, p_dias = stats.chisquare(diasem["n_ataques"].fillna(0))
print(f"  Chi2 uniformidad días semana: χ²={chi2_dias:.2f}  p={p_dias:.3f}")

# ── Fig 4: Día de la semana ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Q5: ¿Hay un día de la semana preferido para atacar?",
             color=FG, fontsize=12, y=1.01)

ax1, ax2 = axes

# Frecuencia por día
colors_d = [YELLOW if d in ["Saturday","Sunday"] else ACCENT
            for d in diasem["dia_semana_n"]]
bars = ax1.bar(diasem["dia_es"], diasem["n_ataques"],
               color=colors_d, alpha=0.85, edgecolor=BG, lw=0.5)
media_ataques_dia = diasem["n_ataques"].mean()
ax1.axhline(media_ataques_dia, color=FG, lw=1.2, ls="--",
            alpha=0.6, label=f"Media: {media_ataques_dia:.0f}")
for bar, val in zip(bars, diasem["n_ataques"].fillna(0)):
    ax1.text(bar.get_x() + bar.get_width()/2,
             val + 0.5, f"{int(val)}",
             ha="center", va="bottom", fontsize=9, color=FG)
ax1.set_ylabel("Nº ataques", color=FG)
ax1.set_title(f"Frecuencia por día — χ²={chi2_dias:.1f}  p={p_dias:.3f}"
              f"  {'★ No uniforme' if p_dias < 0.05 else 'Distribución uniforme'}",
              color=YELLOW if p_dias < 0.05 else FG, fontsize=9)
ax1.legend(fontsize=9)
ax1.tick_params(axis="x", labelsize=9)
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_facecolor(BG)

# Media lanzados y tasa intercepción por día
ax1b = ax1.twinx()
ax1b.plot(diasem["dia_es"],
          diasem["media_intercep"].fillna(0) * 100,
          color=RED, lw=2, marker="D", ms=6,
          label="% intercepción")
ax1b.set_ylabel("Intercepción media (%)", color=RED)
ax1b.tick_params(axis="y", colors=RED)
ax1b.legend(loc="upper right", fontsize=9)

# Heatmap día × hora (frecuencia ataques)
pivot_dh = (df.groupby(["dia_semana", "hora_inicio"])
              .size()
              .unstack(fill_value=0))
# Asegurar todas las horas
for h in range(24):
    if h not in pivot_dh.columns:
        pivot_dh[h] = 0
pivot_dh = pivot_dh[sorted(pivot_dh.columns)]
pivot_dh.index = dias_es

sns.heatmap(pivot_dh, ax=ax2,
            cmap="YlOrRd", annot=False,
            linewidths=0.2, linecolor=BG,
            cbar_kws={"shrink": 0.8, "label": "Nº ataques"})
ax2.set_xlabel("Hora de lanzamiento (local)", color=FG)
ax2.set_ylabel("Día de la semana", color=FG)
ax2.set_title("Heatmap: día × hora de lanzamiento", color=FG, fontsize=10)
ax2.tick_params(colors=FG, labelsize=8)
ax2.set_facecolor(BG)

plt.tight_layout()
save_fig("08_fig4_dia_semana.png")

# ══════════════════════════════════════════════════════════════════════════
# 5.  Q6: EVOLUCIÓN HORA DE LANZAMIENTO EN EL TIEMPO
# ══════════════════════════════════════════════════════════════════════════
print("\n[5] Q6: ¿Adapta Rusia el timing? Evolución hora de lanzamiento …")

# Media móvil de hora de inicio (30 días)
df_sorted = df.sort_values("ts").dropna(subset=["hora_inicio"]).copy()
df_sorted["hora_mm30"] = (df_sorted["hora_inicio"]
                           .rolling(30, min_periods=5)
                           .mean())

# Evolución mensual hora media
hora_mensual = (df.dropna(subset=["hora_inicio"])
                  .groupby("mes")["hora_inicio"]
                  .agg(["mean","std","count"])
                  .reset_index())
hora_mensual["mes_ts"] = hora_mensual["mes"].dt.to_timestamp()

# Evolución mensual hora de fin
horafin_mensual = (df.dropna(subset=["hora_fin"])
                     .groupby("mes")["hora_fin"]
                     .agg(["mean","std"])
                     .reset_index())
horafin_mensual["mes_ts"] = horafin_mensual["mes"].dt.to_timestamp()

# Regresión lineal: ¿deriva la hora a lo largo del tiempo?
x_num = (df_sorted["ts"] - df_sorted["ts"].min()).dt.days
y_hora = df_sorted["hora_inicio"]
slope_h, intercept_h, r_h, p_h, _ = stats.linregress(x_num, y_hora)
print(f"  Tendencia hora inicio: slope={slope_h*30:.3f}h/mes  "
      f"r={r_h:.3f}  p={p_h:.3f}  "
      f"{'★ DERIVA SIGNIFICATIVA' if p_h < 0.05 else 'Sin deriva'}")

# ── Fig 5: Evolución timing ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
fig.suptitle("Q6: ¿Adapta Rusia el horario de lanzamiento?",
             color=FG, fontsize=12, y=1.01)

ax1, ax2 = axes

# Serie hora inicio con MM30
ax1.scatter(df_sorted["ts"], df_sorted["hora_inicio"],
            color=ACCENT, alpha=0.25, s=15, label="Hora lanzamiento")
ax1.plot(df_sorted["ts"], df_sorted["hora_mm30"],
         color=YELLOW, lw=2, label="Media móvil 30 eventos")

# Tendencia lineal
x_plot = np.array([0, x_num.max()])
y_trend = intercept_h + slope_h * x_plot
t_plot  = [df_sorted["ts"].min(),
           df_sorted["ts"].min() + pd.Timedelta(days=int(x_num.max()))]
ax1.plot(t_plot, y_trend, color=RED, lw=1.5, ls="--",
         label=f"Tendencia: {slope_h*30:.2f}h/mes  p={p_h:.3f}"
               f"  {'★' if p_h < 0.05 else 'ns'}")

ax1.set_ylabel("Hora de lanzamiento", color=FG)
ax1.set_yticks(range(0, 25, 3))
ax1.set_yticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
ax1.axhspan(17, 23, alpha=0.07, color=RED)
ax1.legend(fontsize=9, loc="lower left")
ax1.grid(True, alpha=0.3)
ax1.set_facecolor(BG)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

# Hora inicio vs fin por mes (banda temporal del ataque)
ax2.fill_between(hora_mensual["mes_ts"],
                 hora_mensual["mean"].clip(upper=23),
                 24,
                 alpha=0.25, color=RED, label="Zona nocturna")
ax2.fill_between(horafin_mensual["mes_ts"],
                 0,
                 horafin_mensual["mean"].clip(lower=0),
                 alpha=0.25, color=GREEN, label="Zona fin ataque")

ax2.plot(hora_mensual["mes_ts"], hora_mensual["mean"],
         color=RED, lw=2, marker="o", ms=5, label="Hora media inicio")
ax2.plot(horafin_mensual["mes_ts"], horafin_mensual["mean"],
         color=GREEN, lw=2, marker="s", ms=5, label="Hora media fin")

ax2.set_ylabel("Hora (local)", color=FG)
ax2.set_yticks(range(0, 25, 3))
ax2.set_yticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
ax2.set_title("Ventana temporal del ataque por mes (inicio → fin)",
              color=FG, fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor(BG)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
save_fig("08_fig5_evolucion_timing.png")

# ══════════════════════════════════════════════════════════════════════════
# 6.  MAPA DE CALOR OPERACIONAL — VISIÓN SEMANAL COMPLETA
# ══════════════════════════════════════════════════════════════════════════
print("\n[6] Heatmap operacional semana-tipo …")

# Crear semana tipo: día × hora ponderado por lanzamientos
pivot_vol = (df.groupby(["dia_semana", "hora_inicio"])["launched"]
               .sum()
               .unstack(fill_value=0))
for h in range(24):
    if h not in pivot_vol.columns:
        pivot_vol[h] = 0
pivot_vol = pivot_vol[sorted(pivot_vol.columns)]
pivot_vol.index = dias_es

# Tasa intercepción por día × hora
pivot_int = (df.groupby(["dia_semana", "hora_inicio"])["tasa_intercep"]
               .mean()
               .unstack(fill_value=np.nan))
for h in range(24):
    if h not in pivot_int.columns:
        pivot_int[h] = np.nan
pivot_int = pivot_int[sorted(pivot_int.columns)]
pivot_int.index = dias_es

# ── Fig 6: Heatmap dual operacional ──────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle("Mapa de Calor Operacional — Semana Tipo Shahed\n"
             "Inteligencia: cuándo ataca Rusia y cuándo falla la defensa",
             color=FG, fontsize=12, y=1.01)

ax1, ax2 = axes

# Volumen
im1 = sns.heatmap(pivot_vol, ax=ax1,
                  cmap="hot", annot=False,
                  linewidths=0.3, linecolor=BG,
                  cbar_kws={"shrink": 0.6, "label": "UAV lanzados"})
ax1.set_title("Volumen total de lanzamientos (día × hora)",
              color=FG, fontsize=11, pad=8)
ax1.set_xlabel("Hora de lanzamiento (local)", color=FG)
ax1.set_ylabel("")
ax1.tick_params(colors=FG, labelsize=9)

# Marcar zona de máxima actividad
ax1.add_patch(Rectangle((17, -0.5), 7, 7.5,
                          fill=False, edgecolor=RED, lw=2.5,
                          linestyle="--", zorder=5))
ax1.text(18, 7.2, "VENTANA PRINCIPAL", color=RED, fontsize=9,
         fontweight="bold", zorder=6)

# Intercepción
im2 = sns.heatmap(pivot_int * 100, ax=ax2,
                  cmap="RdYlGn", center=70,
                  vmin=40, vmax=100,
                  annot=False,
                  linewidths=0.3, linecolor=BG,
                  cbar_kws={"shrink": 0.6, "label": "Intercepción (%)"},
                  mask=pivot_int.isna())
ax2.set_title("Tasa de intercepción media (día × hora)\n"
              "Verde = alta intercepción (defensa eficaz) | "
              "Rojo = baja intercepción (vulnerable)",
              color=FG, fontsize=10, pad=8)
ax2.set_xlabel("Hora de lanzamiento (local)", color=FG)
ax2.set_ylabel("")
ax2.tick_params(colors=FG, labelsize=9)

plt.tight_layout()
save_fig("08_fig6_heatmap_operacional.png")

# ══════════════════════════════════════════════════════════════════════════
# 7.  FIGURA RESUMEN DE INTELIGENCIA HORARIA
# ══════════════════════════════════════════════════════════════════════════
print("\n[7] Figura resumen inteligencia horaria …")

fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

# ── Panel 1: Timeline de un ataque tipo ──────────────────────────────────
ax_tl = fig.add_subplot(gs[0, :2])
ax_tl.set_xlim(0, 24)
ax_tl.set_ylim(-0.5, 1.5)
ax_tl.set_facecolor(BG)
ax_tl.set_xticks(range(0, 25, 1))
ax_tl.set_xticklabels([f"{h:02d}" for h in range(0, 25)],
                       fontsize=7, rotation=0)
ax_tl.set_yticks([])
ax_tl.set_title("Timeline tipo de un ataque Shahed (horas locales Moscú)",
                color=FG, fontsize=11)

# Noche
ax_tl.axvspan(0, 6,   alpha=0.08, color=PURPLE)
ax_tl.axvspan(17, 24, alpha=0.08, color=PURPLE)

# Fases del ataque
fases = [
    (0,   6,   PURPLE, 0.2, "Madrugada\n(impactos finales)"),
    (6,   9,   GREEN,  0.6, "08-09h\nMáx impactos"),
    (17,  20,  RED,    0.6, "18-20h\nVentana lanzamiento"),
    (20,  24,  ORANGE, 0.3, "20-23h\nTránsito"),
]
for x1, x2, color, alpha, label in fases:
    ax_tl.barh(0.5, x2-x1, left=x1, height=0.6,
               color=color, alpha=alpha, edgecolor=BG)
    ax_tl.text((x1+x2)/2, 0.5, label,
               ha="center", va="center", fontsize=7.5,
               color=FG, fontweight="bold")

# Flecha duración media
ax_tl.annotate("", xy=(df["hora_fin"].median() + 0.5, 1.2),
               xytext=(df["hora_inicio"].median() - 0.5, 1.2),
               arrowprops={"arrowstyle": "<->", "color": YELLOW, "lw": 2})
ax_tl.text(12, 1.3,
           f"Duración media: {df['duracion_h'].mean():.1f}h",
           ha="center", color=YELLOW, fontsize=10)

ax_tl.grid(True, alpha=0.2, axis="x")

# ── Panel 2: KPIs ─────────────────────────────────────────────────────────
ax_kpi = fig.add_subplot(gs[0, 2])
ax_kpi.axis("off")
ax_kpi.set_facecolor(BG)

# Calcular los KPIs clave
noc  = df[df["turno"]=="Nocturno (17-23h)"]
dia  = df[df["turno"]=="Diurno (6-16h)"]
mad  = df[df["turno"]=="Madrugada (0-5h)"]

kpis = [
    ("VENTANA PRINCIPAL", f"18:00–20:00 local"),
    ("Hora fin / impacto", f"08:00–09:00 local"),
    ("Duración media", f"{df['duracion_h'].mean():.1f}h (σ={df['duracion_h'].std():.1f})"),
    ("", ""),
    ("% ataques nocturnos", f"{100*len(noc)/len(df):.0f}%"),
    ("Vol. med. nocturno", f"{noc['launched'].mean():.0f} UAV"),
    ("Intercep. nocturna", f"{noc['tasa_intercep'].mean()*100:.1f}%"),
    ("", ""),
    ("Tendencia timing", f"slope={slope_h*30:.3f}h/mes"),
    ("Deriva temporal",
     "★ Sí" if p_h < 0.05 else "No significativa"),
]

y = 0.97
for k, v in kpis:
    if k == "":
        y -= 0.06
        continue
    color_v = (YELLOW if "★" in v else
               GREEN  if "%" in v or "h" in v else FG)
    ax_kpi.text(0.0, y, k, color=ACCENT, fontsize=8.5,
                transform=ax_kpi.transAxes, va="top",
                fontfamily="monospace")
    ax_kpi.text(0.0, y-0.045, f"  {v}", color=color_v, fontsize=9,
                transform=ax_kpi.transAxes, va="top",
                fontfamily="monospace", fontweight="bold")
    y -= 0.10

ax_kpi.set_title("KPIs horarios", color=FG, fontsize=10, pad=8)

# ── Panel 3: Heatmap compacto día×hora (volumen) ──────────────────────────
ax_hm = fig.add_subplot(gs[1, :])
sns.heatmap(pivot_vol, ax=ax_hm,
            cmap="hot", annot=False,
            linewidths=0.3, linecolor=BG,
            cbar_kws={"shrink": 0.4, "label": "UAV lanzados"})
ax_hm.set_title("Distribución acumulada 2025-2026: día × hora de lanzamiento",
                color=FG, fontsize=11, pad=8)
ax_hm.set_xlabel("Hora de lanzamiento (local Moscú)", color=FG)
ax_hm.set_ylabel("")
ax_hm.tick_params(colors=FG, labelsize=9)
ax_hm.add_patch(Rectangle((17, -0.5), 7, 7.5,
                            fill=False, edgecolor=RED,
                            lw=2.5, linestyle="--", zorder=5))

fig.suptitle("Script 08 — Inteligencia Horaria: Ventanas de Vulnerabilidad Shahed",
             color=FG, fontsize=13, y=1.01)
plt.tight_layout()
save_fig("08_fig7_resumen_horario.png")

# ══════════════════════════════════════════════════════════════════════════
# 8.  RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RESUMEN SCRIPT 08")
print("="*60)

print(f"\n{'─'*45}")
print("HALLAZGOS DE INTELIGENCIA HORARIA")
print(f"{'─'*45}")
print(f"  Ventana lanzamiento : 18:00-20:00 local (Moscú UTC+3)")
print(f"  Ventana impactos    : 08:00-09:00 local (amanecer)")
print(f"  Duración media      : {df['duracion_h'].mean():.1f}h  "
      f"σ={df['duracion_h'].std():.1f}h")
print(f"  % ataques nocturnos : {100*len(noc)/len(df):.0f}%")
print(f"  Vol. med. nocturno  : {noc['launched'].mean():.0f} UAV  "
      f"vs diurno {dia['launched'].mean():.0f} UAV")
print(f"  Intercep. nocturna  : {noc['tasa_intercep'].mean()*100:.1f}%  "
      f"vs diurna {dia['tasa_intercep'].mean()*100:.1f}%")
print(f"  Tendencia timing    : {slope_h*30:.4f}h/mes  p={p_h:.3f}  "
      f"{'★ DERIVA' if p_h < 0.05 else 'ESTABLE'}")
print(f"  Kruskal duracion    : p={kw_p:.3f}  "
      f"{'★ Cambia entre períodos' if kw_p < 0.05 else 'Sin cambio'}")
print(f"  Chi2 días semana    : χ²={chi2_dias:.2f}  p={p_dias:.3f}  "
      f"{'★ No uniforme' if p_dias < 0.05 else 'Uniforme'}")

print(f"\n  Figuras generadas: {FIGS_OK}/7")
print(f"  CSV guardado: {PROC}/shahed_horario.csv")
print("\n  [DONE]")
