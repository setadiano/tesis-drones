"""
Script 07 — Variables Externas: Meteorología + Red Eléctrica
=============================================================
Autor : setadiano / jplatas6@alumno.uned.es
Fecha : Abril 2026

Hipótesis analizadas
--------------------
H_METEO : Rusia lanza olas masivas cuando el viento en Primorsko-Akhtarsk
           sopla en dirección favorable (E / SE, 90-180°) y velocidad
           moderada (< 10 m/s), minimizando la deriva lateral del Shahed.

H_GRID  : La degradación acumulada de la red eléctrica ucraniana
           (proxy: importaciones netas UA←UE vía ENTSO-E / Energy Charts)
           aumenta en las semanas posteriores a olas con alta tasa de impacto
           (baja intercepción).

Fuentes de datos (todas gratuitas, sin API-KEY)
-----------------------------------------------
  Meteorología : Open-Meteo archive API  →  lat 46.05, lon 38.15
  Red eléctrica: energy-charts.info API  →  datos ENTSO-E para Ucrania
  Ataques      : petro_attacks_2025_2026.csv  (ya en repo)

Outputs
-------
  data/processed/meteo_primorsko.csv
  data/processed/entso_ukraine_imports.csv
  data/processed/weekly_combined.csv
  outputs/07_*.png  (8 figuras)
"""

import os
import sys
import warnings
import json
import time
from pathlib import Path

import requests
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Rutas ──────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
RAW    = ROOT / "data" / "raw"
PROC   = ROOT / "data" / "processed"
OUT    = ROOT / "outputs"
PROC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# ── Estilo dark ────────────────────────────────────────────────────────────
BG      = "#0d1117"
FG      = "#e6edf3"
ACCENT  = "#58a6ff"
RED     = "#ff7b72"
GREEN   = "#3fb950"
YELLOW  = "#d29922"
PURPLE  = "#bc8cff"
ORANGE  = "#ffa657"
GRID_C  = "#21262d"

plt.rcParams.update({
    "figure.facecolor"  : BG,
    "axes.facecolor"    : BG,
    "axes.edgecolor"    : GRID_C,
    "axes.labelcolor"   : FG,
    "axes.titlecolor"   : FG,
    "xtick.color"       : FG,
    "ytick.color"       : FG,
    "text.color"        : FG,
    "grid.color"        : GRID_C,
    "grid.alpha"        : 0.5,
    "legend.facecolor"  : "#161b22",
    "legend.edgecolor"  : GRID_C,
    "font.family"       : "monospace",
    "figure.dpi"        : 120,
})

FIGS_OK = 0

def save_fig(name):
    global FIGS_OK
    path = OUT / name
    plt.savefig(path, bbox_inches="tight", facecolor=BG, dpi=150)
    plt.close()
    FIGS_OK += 1
    print(f"  [✓] {name}")

# ══════════════════════════════════════════════════════════════════════════
# 0.  CARGAR DATOS DE ATAQUES
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SCRIPT 07 — VARIABLES EXTERNAS")
print("="*60)

print("\n[0] Cargando petro_attacks_2025_2026.csv …")
df_raw = pd.read_csv(RAW / "petro_attacks_2025_2026.csv")

# Filtrar solo Shahed y solo ataques con datos útiles
# NOTA: is_shahed NO es booleano, almacena nº de Shahed en eventos mixtos
# → filtrar por model string
df = df_raw[df_raw["model"].str.contains("Shahed", na=False)].copy()
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df = df.dropna(subset=["fecha", "launched", "destroyed"])
df = df[df["launched"] > 0].copy()
df["tasa_intercep"] = df["destroyed"] / df["launched"]
df["tasa_hit"]      = 1 - df["tasa_intercep"]
df["impactos"]      = df["launched"] - df["destroyed"]

print(f"  Shahed con datos completos: {len(df)} eventos")
print(f"  Rango: {df['fecha'].min().date()} → {df['fecha'].max().date()}")

# Agregado diario (puede haber varios registros por día)
diario = (df.groupby("fecha")
            .agg(launched=("launched","sum"),
                 destroyed=("destroyed","sum"),
                 impactos=("impactos","sum"))
            .reset_index())
diario["tasa_hit"]      = diario["impactos"]  / diario["launched"]
diario["tasa_intercep"] = diario["destroyed"] / diario["launched"]

# Agregado semanal (lunes como inicio)
diario.set_index("fecha", inplace=True)
semanal = diario.resample("W-MON", label="left", closed="left").agg(
    launched   =("launched","sum"),
    destroyed  =("destroyed","sum"),
    impactos   =("impactos","sum"),
    n_ataques  =("launched","count")
).reset_index()
semanal["tasa_hit"]      = semanal["impactos"]  / semanal["launched"]
semanal["tasa_intercep"] = semanal["destroyed"] / semanal["launched"]
semanal = semanal[(semanal["fecha"] >= "2025-01-01") &
                  (semanal["fecha"] <= "2026-04-06")].copy()
print(f"  Semanas Shahed: {len(semanal)}")

# ══════════════════════════════════════════════════════════════════════════
# 1.  DESCARGAR METEOROLOGÍA — OPEN-METEO
# ══════════════════════════════════════════════════════════════════════════
print("\n[1] Descargando meteorología Open-Meteo (Primorsko-Akhtarsk) …")

meteo_cache = PROC / "meteo_primorsko.csv"

if meteo_cache.exists():
    print("  Usando caché:", meteo_cache)
    meteo = pd.read_csv(meteo_cache, parse_dates=["date"])
else:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=46.05&longitude=38.15"
        "&start_date=2025-01-01&end_date=2026-04-05"
        "&hourly=windspeed_10m,winddirection_10m,windgusts_10m,"
        "windspeed_100m,winddirection_100m,precipitation,temperature_2m"
        "&wind_speed_unit=ms"
        "&timezone=Europe%2FMoscow"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()

    hourly = data["hourly"]
    meteo_h = pd.DataFrame({
        "datetime"         : pd.to_datetime(hourly["time"]),
        "wspeed_10m"       : hourly["windspeed_10m"],
        "wdir_10m"         : hourly["winddirection_10m"],
        "wgusts_10m"       : hourly["windgusts_10m"],
        "wspeed_100m"      : hourly["windspeed_100m"],
        "wdir_100m"        : hourly["winddirection_100m"],
        "precip"           : hourly["precipitation"],
        "temp"             : hourly["temperature_2m"],
    })

    # Componentes vectoriales del viento (para promedio diario correcto)
    for lvl in ["10m", "100m"]:
        spd_col = f"wspeed_{lvl}"
        dir_col = f"wdir_{lvl}"
        meteo_h[f"wu_{lvl}"] = -meteo_h[spd_col] * np.sin(np.radians(meteo_h[dir_col]))
        meteo_h[f"wv_{lvl}"] = -meteo_h[spd_col] * np.cos(np.radians(meteo_h[dir_col]))

    meteo_h["date"] = meteo_h["datetime"].dt.date

    # Agregado diario
    meteo = (meteo_h.groupby("date")
             .agg(
                 wspeed_10m_mean = ("wspeed_10m","mean"),
                 wspeed_10m_max  = ("wspeed_10m","max"),
                 wspeed_100m_mean= ("wspeed_100m","mean"),
                 wgusts_max      = ("wgusts_10m","max"),
                 wu_10m_mean     = ("wu_10m","mean"),
                 wv_10m_mean     = ("wv_10m","mean"),
                 wu_100m_mean    = ("wu_100m","mean"),
                 wv_100m_mean    = ("wv_100m","mean"),
                 precip_sum      = ("precip","sum"),
                 temp_mean       = ("temp","mean"),
             )
             .reset_index())
    meteo["date"] = pd.to_datetime(meteo["date"])

    # Reconstruir dirección media vectorial
    for lvl in ["10m", "100m"]:
        meteo[f"wdir_{lvl}_mean"] = (
            np.degrees(np.arctan2(-meteo[f"wu_{lvl}_mean"],
                                  -meteo[f"wv_{lvl}_mean"])) % 360
        )

    meteo.to_csv(meteo_cache, index=False)
    print(f"  Descargados {len(meteo)} días → {meteo_cache}")

# Definir "viento favorable" para Shahed:
#   Primorsko-Akhtarsk está al SE de Ucrania central
#   Shahed vuela ~NW desde allí; viento de cola = viento del SE (90-180°)
#   Condición favorable: wdir 90-180° AND wspeed < 10 m/s
meteo["viento_favorable"] = (
    (meteo["wdir_10m_mean"] >= 80) &
    (meteo["wdir_10m_mean"] <= 195) &
    (meteo["wspeed_10m_mean"] < 10)
).astype(int)

meteo["wdir_bin"] = pd.cut(
    meteo["wdir_10m_mean"],
    bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
    labels=["N","NE","E","SE","S","SO","O","NO"],
    right=False
)

print(f"  Días con viento favorable (SE<10m/s): "
      f"{meteo['viento_favorable'].sum()} de {len(meteo)} "
      f"({100*meteo['viento_favorable'].mean():.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# 2.  DESCARGAR RED ELÉCTRICA — ENERGY CHARTS (ENTSO-E proxy)
# ══════════════════════════════════════════════════════════════════════════
print("\n[2] Descargando datos red eléctrica (energy-charts.info) …")

entso_cache = PROC / "entso_ukraine_imports.csv"

if entso_cache.exists():
    print("  Usando caché:", entso_cache)
    entso = pd.read_csv(entso_cache, parse_dates=["date"])
else:
    # Energy Charts API — cross-border flows UA
    url_ec = (
        "https://api.energy-charts.info/cross_border_flows"
        "?country=UA&start=2025-01-01&end=2026-04-05"
    )
    try:
        r2 = requests.get(url_ec, timeout=60,
                          headers={"User-Agent": "research-script/1.0"})
        r2.raise_for_status()
        ec_data = r2.json()
        print(f"  Energy Charts keys: {list(ec_data.keys())[:6]}")
    except Exception as e:
        print(f"  Energy Charts falló ({e}), intentando endpoint alternativo…")
        ec_data = None

    if ec_data and "unix_seconds" in ec_data:
        ts  = pd.to_datetime(ec_data["unix_seconds"], unit="s", utc=True)
        ts  = ts.tz_convert("Europe/Kyiv")

        # Construir df con todos los flows disponibles
        rows = []
        for entry in ec_data.get("cross_border_flows", []):
            partner = entry.get("name","")
            values  = entry.get("data",[])
            for t, v in zip(ts, values):
                rows.append({"datetime": t, "partner": partner, "mwh": v})

        if rows:
            ec_df = pd.DataFrame(rows)
            ec_df["date"] = ec_df["datetime"].dt.date
            # Neto = suma de todos los partners (positivo = importación neta UA)
            entso_h = (ec_df.groupby(["date"])["mwh"].sum().reset_index())
            entso_h.columns = ["date","net_import_mwh"]
            entso_h["date"] = pd.to_datetime(entso_h["date"])
            entso = entso_h.copy()
        else:
            ec_data = None

    if ec_data is None or not rows if 'rows' in dir() else True:
        # Fallback: construir serie sintética realista basada en
        # los patrones documentados en la literatura (GDU Report, ENTSO-E)
        # para no bloquear el script — se marca claramente como estimación
        print("  [FALLBACK] Construyendo proxy de importaciones basado en")
        print("  patrones documentados (GDU Winter 2025/26, ENTSO-E histórico)")
        print("  NOTA: datos estimados, NO descargar para publicación")

        dates = pd.date_range("2025-01-01", "2026-04-05", freq="D")
        n = len(dates)

        # Curva basada en evidencia documental:
        # Ene-Feb 2025: importación moderada (invierno, ataques sostenidos)
        # Mar-May 2025: reducción (reparaciones H1-2025)
        # Jun-Sep 2025: exportador neto o equilibrio (reconstrucción)
        # Oct-Dic 2025: pico importación (ataques CHP masivos)
        # Ene-Mar 2026: importación alta (invierno + daños acumulados)

        base = np.zeros(n)
        for i, d in enumerate(dates):
            m = d.month
            if m in [1, 2]:
                base[i] = 800 + np.random.normal(0, 100)   # Invierno 2025
            elif m in [3, 4, 5]:
                base[i] = 400 + np.random.normal(0, 80)    # Reparaciones
            elif m in [6, 7, 8]:
                base[i] = -50 + np.random.normal(0, 120)   # Exportador
            elif m == 9:
                base[i] = 200 + np.random.normal(0, 80)
            elif m in [10, 11]:
                base[i] = 1100 + np.random.normal(0, 200)  # Pico ataques
            elif m == 12:
                base[i] = 1400 + np.random.normal(0, 250)  # Invierno pico
            else:  # 2026 Q1
                base[i] = 1200 + np.random.normal(0, 200)

        entso = pd.DataFrame({"date": dates, "net_import_mwh": base})
        entso["is_estimated"] = True

    entso.to_csv(entso_cache, index=False)
    print(f"  {len(entso)} días guardados → {entso_cache}")

# Semanal ENTSO-E
entso["date"] = pd.to_datetime(entso["date"])
entso_s = (entso.set_index("date")
               .resample("W-MON", label="left", closed="left")["net_import_mwh"]
               .sum().reset_index())
entso_s.columns = ["fecha", "net_import_mwh_week"]
print(f"  Semanas importación: {len(entso_s)}")

# ══════════════════════════════════════════════════════════════════════════
# 3.  CONSTRUIR DATASET SEMANAL COMBINADO
# ══════════════════════════════════════════════════════════════════════════
print("\n[3] Construyendo dataset semanal combinado …")

# Meteo semanal
meteo["date"] = pd.to_datetime(meteo["date"])
meteo_s = (meteo.set_index("date")
               .resample("W-MON", label="left", closed="left")
               .agg(
                   wspeed_mean     = ("wspeed_10m_mean","mean"),
                   wspeed_max      = ("wspeed_10m_max","max"),
                   wgusts_max      = ("wgusts_max","max"),
                   wdir_mean       = ("wdir_10m_mean","mean"),
                   precip_sum      = ("precip_sum","sum"),
                   dias_favorable  = ("viento_favorable","sum"),
                   temp_mean       = ("temp_mean","mean"),
               )
               .reset_index())
meteo_s.columns = ["fecha"] + list(meteo_s.columns[1:])

# Merge
combined = semanal.merge(meteo_s, on="fecha", how="left")
combined = combined.merge(entso_s, on="fecha", how="left")

# Limpiar NaN residuales
combined["launched"]           = combined["launched"].fillna(0)
combined["tasa_hit"]           = combined["tasa_hit"].bfill().ffill()
combined["wspeed_mean"]        = combined["wspeed_mean"].bfill().ffill()
combined["net_import_mwh_week"]= combined["net_import_mwh_week"].bfill().ffill()
combined["dias_favorable"]     = combined["dias_favorable"].fillna(0)

# Variable táctica clave: semana de volumen alto (> mediana)
med_vol = combined["launched"].median()
combined["vol_alto"] = (combined["launched"] > med_vol).astype(int)

# Lag importación (H_GRID: daño aparece 1-2 semanas después)
combined["import_lag1"] = combined["net_import_mwh_week"].shift(-1)
combined["import_lag2"] = combined["net_import_mwh_week"].shift(-2)

combined_path = PROC / "weekly_combined.csv"
combined.to_csv(combined_path, index=False)
print(f"  Dataset combinado: {len(combined)} semanas × {combined.shape[1]} variables")
print(f"  Guardado → {combined_path}")

# ══════════════════════════════════════════════════════════════════════════
# 4.  ANÁLISIS H_METEO
# ══════════════════════════════════════════════════════════════════════════
print("\n[4] Analizando H_METEO: ¿viento favorable → más lanzamientos? …")

# Unir meteo DIARIO con lanzamientos diarios
diario_reset = diario.reset_index()
diario_reset.columns = ["date", "launched", "destroyed", "impactos",
                        "tasa_hit", "tasa_intercep"]
meteo2 = meteo[["date","wspeed_10m_mean","wdir_10m_mean",
                 "viento_favorable","wdir_bin","precip_sum"]].copy()
meteo2["date"] = pd.to_datetime(meteo2["date"])
diario_reset["date"] = pd.to_datetime(diario_reset["date"])

daily_merged = meteo2.merge(diario_reset, on="date", how="left")
daily_merged["launched"] = daily_merged["launched"].fillna(0)
daily_merged["ataque"]   = (daily_merged["launched"] > 0).astype(int)
daily_merged["ataque_masivo"] = (daily_merged["launched"] > 50).astype(int)

# Test 1: correlación velocidad de viento vs lanzamientos
mask = daily_merged["launched"] > 0
r_spear, p_spear = spearmanr(
    daily_merged.loc[mask, "wspeed_10m_mean"],
    daily_merged.loc[mask, "launched"]
)

# Test 2: días ataque vs viento favorable (chi-cuadrado)
ct = pd.crosstab(daily_merged["ataque_masivo"],
                 daily_merged["viento_favorable"])
chi2, p_chi2, dof, _ = chi2_contingency(ct)

# Test 3: media lanzamientos en días favorables vs desfavorables
fav   = daily_merged[daily_merged["viento_favorable"]==1]["launched"]
nofav = daily_merged[daily_merged["viento_favorable"]==0]["launched"]
t_stat, p_ttest = stats.ttest_ind(
    fav[fav > 0], nofav[nofav > 0], equal_var=False
)

# Test 4: distribución por dirección del viento (solo días con ataque)
wdir_ataque = (daily_merged[daily_merged["ataque_masivo"]==1]
               .groupby("wdir_bin")["launched"].agg(["mean","count","sum"])
               .reset_index())

print(f"  Spearman r(viento_speed, lanzamientos) = {r_spear:.3f}  p={p_spear:.3f}")
print(f"  Chi2 (ataque_masivo × viento_favorable) = {chi2:.2f}  p={p_chi2:.3f}")
print(f"  t-test lanzamientos: fav={fav[fav>0].mean():.1f}  "
      f"nofav={nofav[nofav>0].mean():.1f}  p={p_ttest:.3f}")
print(f"  Distribución ataques masivos por dir. viento:")
print(f"  {wdir_ataque.to_string(index=False)}")

# ── Fig 1: Tiempo — viento + lanzamientos ─────────────────────────────────
print("\n  Generando figuras H_METEO …")

fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                          gridspec_kw={"height_ratios": [1.5, 1.5, 1]})
fig.suptitle("H_METEO: Viento en Primorsko-Akhtarsk vs Ataques Shahed",
             color=FG, fontsize=13, y=1.01)

ax1, ax2, ax3 = axes

# Serie lanzamientos diarios
days_all = daily_merged["date"]
launched_all = daily_merged["launched"]
ax1.fill_between(days_all, launched_all, alpha=0.5, color=RED, label="UAV lanzados/día")
ax1.plot(days_all, launched_all, color=RED, lw=0.5, alpha=0.7)
ax1.set_ylabel("UAV lanzados", color=FG)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)

# Marcar días con viento favorable
fav_days = daily_merged[daily_merged["viento_favorable"]==1]["date"]
ymax = launched_all.max() * 0.15
for d in fav_days:
    ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.12,
                color=GREEN, zorder=0)
ax1.plot([], [], color=GREEN, alpha=0.4, lw=8,
         label="Viento favorable (SE, <10m/s)")
ax1.legend(loc="upper left", fontsize=9)

# Velocidad viento
ax2.plot(meteo["date"], meteo["wspeed_10m_mean"], color=ACCENT, lw=1.2,
         label="Vel. viento 10m (m/s)")
ax2.fill_between(meteo["date"], meteo["wspeed_10m_mean"],
                 alpha=0.2, color=ACCENT)
ax2.axhline(10, color=YELLOW, lw=1.2, ls="--", label="Umbral 10 m/s")
ax2.set_ylabel("Velocidad viento (m/s)", color=FG)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)

# Dirección viento (rosa reducida como scatter)
sc = ax3.scatter(meteo["date"], meteo["wdir_10m_mean"],
                 c=meteo["wspeed_10m_mean"], cmap="plasma",
                 s=4, alpha=0.6, vmin=0, vmax=15)
ax3.axhspan(80, 195, alpha=0.12, color=GREEN, label="Zona favorable SE (80-195°)")
ax3.set_ylabel("Dir. viento (°)", color=FG)
ax3.set_yticks([0, 90, 180, 270, 360])
ax3.set_yticklabels(["N", "E", "S", "O", "N"])
ax3.legend(loc="upper right", fontsize=9)
ax3.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax3, label="m/s", pad=0.01)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_facecolor(BG)

plt.tight_layout()
save_fig("07_fig1_meteo_vs_lanzamientos.png")

# ── Fig 2: Distribución lanzamientos por dir. viento (rosa de ataques) ────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                          subplot_kw={"projection": "polar"})
fig.suptitle("Rosa de Ataques: Dirección del Viento en Día de Lanzamiento",
             color=FG, fontsize=12)

# Todos los días con ataque
attack_days = daily_merged[daily_merged["launched"] > 0].copy()
attack_days_massive = daily_merged[daily_merged["ataque_masivo"] == 1].copy()

bins_deg = np.arange(0, 361, 22.5)
bins_rad = np.radians(bins_deg)

for ax, data, title in zip(
    axes,
    [attack_days, attack_days_massive],
    ["Todos los días con ataque\n(lanzados > 0)", "Ataques masivos\n(lanzados > 50 UAV)"]
):
    wdirs = data["wdir_10m_mean"].dropna()
    hist, _ = np.histogram(wdirs, bins=bins_deg)
    width    = np.radians(22.5)
    theta    = np.radians(bins_deg[:-1] + 11.25)

    bars = ax.bar(theta, hist, width=width, bottom=0,
                  color=ACCENT, alpha=0.7, edgecolor=BG, lw=0.5)

    # Colorear sector favorable
    fav_idx = [(i, b) for i, b in enumerate(bins_deg[:-1])
               if 80 <= b <= 195]
    for i, b in fav_idx:
        ax.patches[i].set_facecolor(GREEN)
        ax.patches[i].set_alpha(0.85)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    ax.set_title(title, color=FG, pad=15, fontsize=10)
    ax.spines["polar"].set_color(GRID_C)
    ax.grid(color=GRID_C, alpha=0.5)

plt.tight_layout()
save_fig("07_fig2_rosa_ataques_viento.png")

# ── Fig 3: Boxplot lanzamientos × condición viento ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("Lanzamientos por Condición de Viento",
             color=FG, fontsize=12)

ax1, ax2 = axes

# Boxplot viento favorable vs no favorable
fav_l   = fav[fav > 0].values
nofav_l = nofav[nofav > 0].values
bp = ax1.boxplot([nofav_l, fav_l],
                 labels=["Viento NO\nfavorable", "Viento\nfavorable (SE<10)"],
                 patch_artist=True,
                 medianprops={"color": YELLOW, "lw": 2},
                 flierprops={"marker": "o", "markerfacecolor": FG,
                             "markersize": 3, "alpha": 0.4})
bp["boxes"][0].set_facecolor(RED + "55")
bp["boxes"][1].set_facecolor(GREEN + "55")

mean_fav   = np.mean(fav_l)
mean_nofav = np.mean(nofav_l)
ax1.axhline(mean_nofav, color=RED,   ls="--", alpha=0.6, lw=1.2,
            label=f"Media NF: {mean_nofav:.0f}")
ax1.axhline(mean_fav,   color=GREEN, ls="--", alpha=0.6, lw=1.2,
            label=f"Media F:  {mean_fav:.0f}")
ax1.set_ylabel("UAV lanzados", color=FG)
ax1.set_title(f"t-test p={p_ttest:.3f}  "
              f"{'★ sig.' if p_ttest < 0.05 else 'n.s.'}",
              color=YELLOW if p_ttest < 0.05 else FG, fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor(BG)

# Media lanzamientos por octante de viento
wdir_all = (daily_merged[daily_merged["launched"] > 0]
            .groupby("wdir_bin")["launched"]
            .agg(["mean","count"])
            .reset_index())
colors_oct = [GREEN if b in ["E","SE"] else ACCENT
              for b in wdir_all["wdir_bin"].astype(str)]
bars = ax2.bar(wdir_all["wdir_bin"].astype(str),
               wdir_all["mean"],
               color=colors_oct, alpha=0.85, edgecolor=BG, lw=0.5)
for bar, cnt in zip(bars, wdir_all["count"]):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 3,
             f"n={cnt}", ha="center", va="bottom", fontsize=8, color=FG)
ax2.set_ylabel("Media UAV lanzados (días con ataque)", color=FG)
ax2.set_title("Media por octante (verde = zona favorable)", color=FG, fontsize=10)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_facecolor(BG)

# Anotación chi2
ax1.text(0.98, 0.98,
         f"χ²={chi2:.2f}  p={p_chi2:.3f}\nn_fav={len(fav_l)}  n_nofav={len(nofav_l)}",
         transform=ax1.transAxes, ha="right", va="top",
         fontsize=9, color=FG,
         bbox={"boxstyle": "round", "facecolor": "#21262d", "alpha": 0.8})

plt.tight_layout()
save_fig("07_fig3_boxplot_viento.png")

# ══════════════════════════════════════════════════════════════════════════
# 5.  ANÁLISIS H_GRID
# ══════════════════════════════════════════════════════════════════════════
print("\n[5] Analizando H_GRID: ¿impactos → degradación eléctrica? …")

# Comprobar si los datos son estimados
is_estimated = False
if "is_estimated" in entso.columns and entso["is_estimated"].all():
    is_estimated = True
    print("  NOTA: datos de importación son estimados (proxy documental)")
    print("  → los tests estadísticos son indicativos, NO para publicación")

# Correlaciones entre impactos y importación (lag 0, 1, 2 semanas)
c_df = combined.dropna(subset=["impactos","net_import_mwh_week",
                                 "import_lag1","import_lag2"]).copy()

r0, p0 = pearsonr(c_df["impactos"], c_df["net_import_mwh_week"])
r1, p1 = pearsonr(c_df["impactos"], c_df["import_lag1"].dropna()
                  if len(c_df["import_lag1"].dropna()) == len(c_df) else
                  c_df["import_lag1"].fillna(c_df["net_import_mwh_week"].mean()))
r2, p2 = pearsonr(c_df["impactos"], c_df["import_lag2"].fillna(
    c_df["net_import_mwh_week"].mean()))

print(f"  Pearson r(impactos, import_lag0) = {r0:.3f}  p={p0:.3f}")
print(f"  Pearson r(impactos, import_lag1) = {r1:.3f}  p={p1:.3f}")
print(f"  Pearson r(impactos, import_lag2) = {r2:.3f}  p={p2:.3f}")

# Regresión: ¿la tasa_hit de ataques predice importación futura?
hit_mask = c_df["tasa_hit"].notna() & c_df["import_lag2"].notna()
from scipy.stats import linregress
slope_g, intercept_g, r_g, p_g, se_g = linregress(
    c_df.loc[hit_mask, "tasa_hit"],
    c_df.loc[hit_mask, "import_lag2"].fillna(c_df["net_import_mwh_week"].mean())
)
print(f"  Regresión: import_lag2 ~ tasa_hit  "
      f"β={slope_g:.1f}  r={r_g:.3f}  p={p_g:.3f}")

# ── Fig 4: Serie dual — impactos + importación ────────────────────────────
fig, ax1 = plt.subplots(figsize=(14, 5.5))
ax2 = ax1.twinx()

label_import = ("Importación neta UA←UE (MWh/sem)"
                if not is_estimated else
                "Importación neta [ESTIMADA] (MWh/sem)")

ax1.fill_between(combined["fecha"], combined["impactos"],
                 alpha=0.55, color=RED, label="Impactos Shahed/sem")
ax1.plot(combined["fecha"], combined["impactos"],
         color=RED, lw=0.8, alpha=0.8)
ax1.set_ylabel("Shahed que impactan/semana", color=RED)
ax1.tick_params(axis="y", colors=RED)

ax2.plot(combined["fecha"], combined["net_import_mwh_week"],
         color=ACCENT, lw=1.8, label=label_import, alpha=0.85)
ax2.fill_between(combined["fecha"], combined["net_import_mwh_week"],
                 alpha=0.15, color=ACCENT)
ax2.axhline(0, color=FG, lw=0.8, ls="--", alpha=0.4)
ax2.set_ylabel(label_import, color=ACCENT)
ax2.tick_params(axis="y", colors=ACCENT)

# Períodos clave según literatura
for start, end, label in [
    ("2025-03-01", "2025-09-30", "Reconstrucción"),
    ("2025-10-01", "2025-12-31", "Ataques CHP"),
]:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                alpha=0.08, color=YELLOW)
    ax1.text(pd.Timestamp(start) + pd.Timedelta(weeks=2),
             combined["impactos"].max() * 0.85,
             label, color=YELLOW, fontsize=8, alpha=0.9)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="upper left", fontsize=9)

ax1.set_title("H_GRID: Impactos Shahed vs Degradación Red Eléctrica Ucraniana",
              color=FG, fontsize=12)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax1.grid(True, alpha=0.3)
ax1.set_facecolor(BG)
fig.patch.set_facecolor(BG)

if is_estimated:
    ax1.text(0.5, 0.02,
             "⚠ Datos de importación basados en proxy documental — no usar para publicación",
             transform=ax1.transAxes, ha="center", va="bottom",
             fontsize=8, color=YELLOW, alpha=0.85)

plt.tight_layout()
save_fig("07_fig4_impactos_vs_red.png")

# ── Fig 5: Scatter correlación con lag ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Correlación Impactos Shahed → Importación Eléctrica (lags 0-2 sem)",
             color=FG, fontsize=11)

for ax, (lag_col, lag_label, r_val, p_val) in zip(
    axes,
    [("net_import_mwh_week", "Lag 0", r0, p0),
     ("import_lag1",         "Lag +1 sem", r1, p1),
     ("import_lag2",         "Lag +2 sem", r2, p2)]
):
    y_col = c_df[lag_col].fillna(c_df["net_import_mwh_week"].mean())
    ax.scatter(c_df["impactos"], y_col,
               color=ACCENT, alpha=0.55, s=30, edgecolors="none")

    # Línea de tendencia
    if len(c_df) > 3:
        z = np.polyfit(c_df["impactos"], y_col, 1)
        xline = np.linspace(c_df["impactos"].min(), c_df["impactos"].max(), 100)
        ax.plot(xline, np.polyval(z, xline),
                color=YELLOW, lw=1.5, ls="--")

    sig_label = "★" if p_val < 0.05 else "n.s."
    color_r   = GREEN if p_val < 0.05 else FG
    ax.set_title(f"{lag_label}\nr={r_val:.3f}  p={p_val:.3f} {sig_label}",
                 color=color_r, fontsize=10)
    ax.set_xlabel("Impactos Shahed/sem", color=FG)
    ax.set_ylabel("Importación neta (MWh)", color=FG)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(BG)

plt.tight_layout()
save_fig("07_fig5_scatter_lag_grid.png")

# ══════════════════════════════════════════════════════════════════════════
# 6.  ANÁLISIS INTEGRADO — 3 VARIABLES JUNTAS
# ══════════════════════════════════════════════════════════════════════════
print("\n[6] Análisis integrado …")

# Clasificar semanas en 3 regímenes (heredado de Script 05 clusters)
# Cluster 3: Feb-Jul 2025 (alta saturación)
# Cluster 2: volumen bajo
# Cluster 1: Jul 2025-Mar 2026 (alta intercepción)
def asignar_cluster(row):
    d = row["fecha"]
    if pd.Timestamp("2025-02-01") <= d <= pd.Timestamp("2025-07-31"):
        return 3
    elif pd.Timestamp("2025-08-01") <= d <= pd.Timestamp("2026-04-06"):
        return 1
    else:
        return 2

combined["cluster"] = combined.apply(asignar_cluster, axis=1)

cluster_colors = {1: GREEN, 2: ACCENT, 3: RED}
cluster_labels = {1: "C1 Alta intercepción", 2: "C2 Vol. bajo", 3: "C3 Saturación"}

# ── Fig 6: Scatter 3D proyectado — viento × impactos × import ─────────────
fig, ax = plt.subplots(figsize=(11, 7))

for cl in [1, 2, 3]:
    sub = combined[combined["cluster"] == cl].dropna(
        subset=["wspeed_mean","impactos","net_import_mwh_week"])
    sc = ax.scatter(sub["wspeed_mean"], sub["impactos"],
                    s=sub["net_import_mwh_week"].clip(lower=0) / 200 + 20,
                    c=cluster_colors[cl], alpha=0.7, label=cluster_labels[cl],
                    edgecolors=BG, lw=0.5)

ax.set_xlabel("Velocidad media viento semana (m/s)", color=FG)
ax.set_ylabel("Impactos Shahed/semana", color=FG)
ax.set_title("Viento × Impactos × Importación eléctrica\n"
             "(tamaño burbuja = importación neta MWh)",
             color=FG, fontsize=11)
ax.axvline(10, color=YELLOW, lw=1.2, ls="--", alpha=0.7, label="Umbral viento 10m/s")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_facecolor(BG)
plt.tight_layout()
save_fig("07_fig6_scatter3var.png")

# ── Fig 7: Heatmap correlaciones ──────────────────────────────────────────
corr_cols = ["launched","destroyed","tasa_hit","impactos",
             "wspeed_mean","wgusts_max","dias_favorable",
             "wdir_mean","net_import_mwh_week"]
corr_labels = ["UAV lanzados","UAV destruidos","Tasa hit",
               "Impactos","Viento (m/s)","Ráfagas (m/s)",
               "Días viento\nfavorable","Dir. viento (°)",
               "Import. eléct."]

corr_data = combined[corr_cols].dropna()
corr_matrix = corr_data.corr(method="spearman")
corr_matrix.index   = corr_labels
corr_matrix.columns = corr_labels

fig, ax = plt.subplots(figsize=(11, 9))
mask_up = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, ax=ax, mask=mask_up,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            linewidths=0.4, linecolor=BG,
            cbar_kws={"shrink": 0.8, "label": "Spearman ρ"})

ax.set_title("Correlaciones Spearman — Ataques + Meteorología + Red Eléctrica",
             color=FG, fontsize=11, pad=12)
ax.tick_params(colors=FG, labelsize=9)
ax.set_facecolor(BG)
plt.tight_layout()
save_fig("07_fig7_heatmap_correlaciones.png")

# ── Fig 8: Resumen ejecutivo de inteligencia ─────────────────────────────
print("\n[7] Generando figura resumen de inteligencia …")

fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Tasa hit mensual + viento
ax_thr = fig.add_subplot(gs[0, :2])
monthly = (diario_reset.copy()
           .assign(mes=lambda x: x["date"].dt.to_period("M"))
           .groupby("mes")
           .agg(launched=("launched","sum"),
                destroyed=("destroyed","sum"))
           .assign(tasa_hit=lambda x: (x["launched"]-x["destroyed"])/x["launched"]))
monthly.index = monthly.index.to_timestamp()

meteo_m = (meteo.set_index("date")
               .resample("MS")
               .agg(wspeed_mean=("wspeed_10m_mean","mean"),
                    dias_fav=("viento_favorable","sum"))
               .reset_index())

ax_thr.plot(monthly.index, monthly["tasa_hit"]*100,
            color=RED, lw=2, marker="o", ms=5, label="Tasa hit Shahed (%)")
ax_thr.axhline(20, color=GREEN, lw=1, ls="--", alpha=0.6, label="20% umbral bajo")
ax_thr.set_ylabel("Tasa hit (%)", color=FG)
ax_thr.set_ylim(0, 60)

ax_tw = ax_thr.twinx()
ax_tw.bar(meteo_m["date"], meteo_m["dias_fav"],
          width=20, alpha=0.3, color=ACCENT, label="Días viento favorable/mes")
ax_tw.set_ylabel("Días viento favorable", color=ACCENT)
ax_tw.tick_params(axis="y", colors=ACCENT)

lines_a, labels_a = ax_thr.get_legend_handles_labels()
lines_b, labels_b = ax_tw.get_legend_handles_labels()
ax_thr.legend(lines_a + lines_b, labels_a + labels_b,
              loc="upper right", fontsize=8)
ax_thr.set_title("Tasa Hit mensual + Días de viento favorable",
                 color=FG, fontsize=10)
ax_thr.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax_thr.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax_thr.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax_thr.grid(True, alpha=0.3)
ax_thr.set_facecolor(BG)

# Panel 2: KPIs resumen
ax_kpi = fig.add_subplot(gs[0, 2])
ax_kpi.axis("off")
kpi_text = [
    ("H_METEO", ""),
    (f"Spearman r: {r_spear:.3f}", "★" if abs(r_spear) > 0.1 and p_spear < 0.05 else "ns"),
    (f"χ² p={p_chi2:.3f}", "★" if p_chi2 < 0.05 else "ns"),
    (f"Días fav. lanzados: {100*daily_merged[daily_merged['ataque_masivo']==1]['viento_favorable'].mean():.0f}%", ""),
    ("", ""),
    ("H_GRID", ""),
    (f"r(imp, lag0): {r0:.3f}", "★" if p0 < 0.05 else "ns"),
    (f"r(imp, lag1): {r1:.3f}", "★" if p1 < 0.05 else "ns"),
    (f"r(imp, lag2): {r2:.3f}", "★" if p2 < 0.05 else "ns"),
]
y_pos = 0.97
for txt, sig in kpi_text:
    if txt in ["H_METEO", "H_GRID"]:
        ax_kpi.text(0.05, y_pos, txt, color=ACCENT,
                    fontsize=11, fontweight="bold",
                    transform=ax_kpi.transAxes, va="top")
    else:
        color = GREEN if sig == "★" else (FG if sig == "" else FG + "88")
        ax_kpi.text(0.05, y_pos, f"  {txt} {sig}", color=color,
                    fontsize=9, transform=ax_kpi.transAxes, va="top",
                    fontfamily="monospace")
    y_pos -= 0.09
ax_kpi.set_facecolor(BG)
ax_kpi.set_title("KPIs de hipótesis", color=FG, fontsize=10, pad=8)

# Panel 3: importación eléctrica + lanzamientos (serie temporal compacta)
ax_grid = fig.add_subplot(gs[1, :])
ax_g2   = ax_grid.twinx()

ax_grid.fill_between(combined["fecha"], combined["impactos"],
                     alpha=0.45, color=RED, label="Impactos/sem")
ax_g2.plot(combined["fecha"], combined["net_import_mwh_week"],
           color=ACCENT, lw=1.5, alpha=0.85,
           label="Import. eléctrica (MWh/sem)")

ax_grid.set_ylabel("Impactos Shahed", color=RED)
ax_g2.set_ylabel("Import. neta (MWh)", color=ACCENT)
ax_g2.tick_params(axis="y", colors=ACCENT)
ax_grid.tick_params(axis="y", colors=RED)

lines1, labels1 = ax_grid.get_legend_handles_labels()
lines2, labels2 = ax_g2.get_legend_handles_labels()
ax_grid.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=8)
ax_grid.set_title("H_GRID: Degradación acumulada de la red eléctrica",
                  color=FG, fontsize=10)
ax_grid.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax_grid.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax_grid.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax_grid.grid(True, alpha=0.3)
ax_grid.set_facecolor(BG)
fig.patch.set_facecolor(BG)

if is_estimated:
    fig.text(0.5, 0.01,
             "⚠ Datos importación eléctrica = proxy documental. Pendiente API ENTSO-E.",
             ha="center", fontsize=8, color=YELLOW)

fig.suptitle("Script 07 — Inteligencia Operacional: Meteorología + Red Eléctrica",
             color=FG, fontsize=13, y=1.01)

plt.tight_layout()
save_fig("07_fig8_resumen_inteligencia.png")

# ══════════════════════════════════════════════════════════════════════════
# 7.  RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RESUMEN SCRIPT 07")
print("="*60)

print(f"\n{'─'*40}")
print("H_METEO — Condicionamiento meteorológico")
print(f"{'─'*40}")
print(f"  Spearman r(viento_speed, lanzamientos) = {r_spear:.3f}  p={p_spear:.3f}"
      f"  {'★ SIGNIFICATIVO' if p_spear < 0.05 else 'NO significativo'}")
print(f"  χ²(ataque_masivo × viento_fav)         = {chi2:.2f}   p={p_chi2:.3f}"
      f"  {'★ SIGNIFICATIVO' if p_chi2 < 0.05 else 'NO significativo'}")
print(f"  Media lanzamientos: fav={np.mean(fav_l):.0f}  nofav={np.mean(nofav_l):.0f}"
      f"  t-test p={p_ttest:.3f}"
      f"  {'★' if p_ttest < 0.05 else 'ns'}")
pct_masivos_fav = 100*daily_merged[daily_merged["ataque_masivo"]==1]["viento_favorable"].mean()
print(f"  % ataques masivos en día de viento favorable: {pct_masivos_fav:.1f}%")

print(f"\n{'─'*40}")
print("H_GRID — Degradación acumulada red eléctrica")
print(f"{'─'*40}")
print(f"  r(impactos, import_lag0) = {r0:.3f}  p={p0:.3f}"
      f"  {'★' if p0 < 0.05 else 'ns'}")
print(f"  r(impactos, import_lag1) = {r1:.3f}  p={p1:.3f}"
      f"  {'★' if p1 < 0.05 else 'ns'}")
print(f"  r(impactos, import_lag2) = {r2:.3f}  p={p2:.3f}"
      f"  {'★' if p2 < 0.05 else 'ns'}")
if is_estimated:
    print("  ⚠ AVISO: datos importación son estimados. Necesita ENTSO-E real.")

print(f"\n  Figuras generadas: {FIGS_OK}/8")
print(f"  Dataset guardado:  {combined_path}")
print("\n  [DONE]")
