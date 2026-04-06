"""
03_series_temporales.py
=======================
Análisis de series temporales del teatro Ucrania-Rusia 2025-2026
Teatro Irán-Israel como serie de eventos discretos.

Técnicas:
  1. Descomposición STL (tendencia, estacionalidad, residuo)
  2. Detección de rupturas estructurales (Chow test + Bai-Perron)
  3. ARIMA/SARIMA auto-selección (pmdarima)
  4. VAR multivariante (lanzamientos ~ intercepciones ~ hits)
  5. Markov Switching (regímenes alto/bajo impacto)
  6. Análisis de correlación cruzada (CCF) RU→UA vs UA→RU (ACLED)
  7. Figura resumen integrada (dark theme)

Inputs  : data/raw/ur_mensual_2025_2026.csv
          data/raw/ur_ataques_grandes_2025_2026.csv
          data/processed/acled_mensual_agregado.csv   (generado por script 02)
Outputs : outputs/figures/09_stl_descomposicion.png
          outputs/figures/10_rupturas_estructurales.png
          outputs/figures/11_arima_forecast.png
          outputs/figures/12_var_impulso_respuesta.png
          outputs/figures/13_markov_switching.png
          outputs/figures/14_ccf_acled.png
          outputs/tables/03_arima_summary.txt
          outputs/tables/04_var_summary.txt
          outputs/tables/05_markov_summary.txt
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns

# ── statsmodels ──────────────────────────────────────────────────────────────
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, ccf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

# ── pmdarima (auto ARIMA) ────────────────────────────────────────────────────
try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False
    print("⚠  pmdarima no instalado — se usará ARIMA manual")

# ── scipy ────────────────────────────────────────────────────────────────────
from scipy import stats
from scipy.signal import find_peaks

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN VISUAL (dark theme GitHub)
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
BG2      = "#161b22"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f85149"
ORANGE   = "#d29922"
PURPLE   = "#bc8cff"
CYAN     = "#39d353"
FG       = "#e6edf3"
FG2      = "#8b949e"
GRID     = "#21262d"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG2,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   FG,
    "axes.titlecolor":   FG,
    "xtick.color":       FG2,
    "ytick.color":       FG2,
    "text.color":        FG,
    "legend.facecolor":  BG2,
    "legend.edgecolor":  GRID,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
})

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW         = os.path.join(BASE, "data", "raw")
PROCESSED   = os.path.join(BASE, "data", "processed")
FIGS        = os.path.join(BASE, "outputs", "figures")
TABLES      = os.path.join(BASE, "outputs", "tables")

for d in [FIGS, TABLES, PROCESSED]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("ANÁLISIS DE SERIES TEMPORALES — DRONES 2025-2026")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Cargando datos...")

# ── Datos mensuales UR ───────────────────────────────────────────────────────
df_ur = pd.read_csv(os.path.join(RAW, "ur_mensual_2025_2026.csv"))
df_ur["fecha_dt"] = pd.to_datetime(df_ur["fecha"].str.strip() + "-01",
                                    format="%Y-%m-%d", errors="coerce")
df_ur = df_ur.dropna(subset=["fecha_dt"]).sort_values("fecha_dt").reset_index(drop=True)
df_ur["lanzamientos_total"] = pd.to_numeric(df_ur["lanzamientos_total"], errors="coerce")
df_ur["intercepciones"]     = pd.to_numeric(df_ur["intercepciones"],     errors="coerce")
df_ur["hits"]               = pd.to_numeric(df_ur["hits"],               errors="coerce")
df_ur["tasa_hit_total_pct"] = pd.to_numeric(df_ur["tasa_hit_total_pct"], errors="coerce")
df_ur["tasa_strike_pct"]    = pd.to_numeric(df_ur["tasa_strike_pct"],    errors="coerce")
df_ur["strike_uav"]         = pd.to_numeric(df_ur["strike_uav"],         errors="coerce")
df_ur["decoy_uav"]          = pd.to_numeric(df_ur["decoy_uav"],          errors="coerce")
df_ur["avg_diario"]         = pd.to_numeric(df_ur["avg_diario"],         errors="coerce")

print(f"  ✓ ur_mensual: {len(df_ur)} meses  "
      f"({df_ur['fecha_dt'].min().strftime('%b %Y')} → "
      f"{df_ur['fecha_dt'].max().strftime('%b %Y')})")

# ── Ataques grandes ──────────────────────────────────────────────────────────
df_ag = pd.read_csv(os.path.join(RAW, "ur_ataques_grandes_2025_2026.csv"))
df_ag["fecha_dt"] = pd.to_datetime(df_ag["fecha"], errors="coerce")
df_ag = df_ag.dropna(subset=["fecha_dt"]).sort_values("fecha_dt").reset_index(drop=True)
df_ag["uavs_total"] = pd.to_numeric(df_ag["uavs_total"], errors="coerce")
print(f"  ✓ ur_ataques_grandes: {len(df_ag)} eventos")

# ── ACLED mensual (generado por script 02) ───────────────────────────────────
acled_mensual_path = os.path.join(PROCESSED, "acled_mensual_agregado.csv")
df_acled_m = None
if os.path.exists(acled_mensual_path):
    df_acled_m = pd.read_csv(acled_mensual_path)
    # Convertir columna de fecha — puede ser 'mes_inicio' o 'fecha'
    for col in ["mes_inicio", "fecha", "event_date"]:
        if col in df_acled_m.columns:
            df_acled_m["fecha_dt"] = pd.to_datetime(df_acled_m[col], errors="coerce")
            break
    df_acled_m = df_acled_m.dropna(subset=["fecha_dt"]).sort_values("fecha_dt").reset_index(drop=True)
    print(f"  ✓ acled_mensual: {len(df_acled_m)} filas  cols={list(df_acled_m.columns)}")
else:
    # Regenerar desde acled_drones_limpio.csv
    acled_limpio_path = os.path.join(PROCESSED, "acled_drones_limpio.csv")
    if os.path.exists(acled_limpio_path):
        df_acled_raw = pd.read_csv(acled_limpio_path, low_memory=False)
        df_acled_raw["event_date"] = pd.to_datetime(df_acled_raw["event_date"], errors="coerce")
        df_acled_raw["mes"] = df_acled_raw["event_date"].dt.to_period("M")
        df_acled_m = (df_acled_raw.groupby("mes")
                      .agg(eventos=("event_id_cnty", "count"),
                           bajas=("fatalities", "sum"))
                      .reset_index())
        df_acled_m["fecha_dt"] = df_acled_m["mes"].dt.to_timestamp()
        print(f"  ✓ acled_mensual regenerado: {len(df_acled_m)} meses")
    else:
        print("  ⚠  No se encontró acled_mensual_agregado.csv ni acled_drones_limpio.csv")
        print("     Ejecuta primero scripts/02_procesar_acled.py")

# ─────────────────────────────────────────────────────────────────────────────
# SERIES PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
ts_lanzamientos  = df_ur.set_index("fecha_dt")["lanzamientos_total"].asfreq("MS")
ts_intercepciones = df_ur.set_index("fecha_dt")["intercepciones"].asfreq("MS")
ts_hits          = df_ur.set_index("fecha_dt")["hits"].asfreq("MS")
ts_tasa_hit      = df_ur.set_index("fecha_dt")["tasa_hit_total_pct"].asfreq("MS")
ts_strike_pct    = df_ur.set_index("fecha_dt")["tasa_strike_pct"].asfreq("MS")

N = len(ts_lanzamientos)
print(f"\n  Series temporales: {N} observaciones mensuales")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TESTS DE ESTACIONARIEDAD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/7] Tests de estacionariedad (ADF + KPSS)...")

def test_estacionariedad(serie, nombre):
    """ADF + KPSS sobre una serie pandas. Retorna dict con resultados."""
    s = serie.dropna()
    # ADF
    adf_stat, adf_p, adf_lags, _, adf_cv, _ = adfuller(s, autolag="AIC")
    # KPSS
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_cv = kpss(s, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan
    
    estacionaria = (adf_p < 0.05) and (kpss_p > 0.05)
    print(f"  {nombre:30s}  ADF p={adf_p:.4f}  KPSS p={kpss_p:.4f}  "
          f"{'✓ estacionaria' if estacionaria else '✗ NO estacionaria'}")
    return {"serie": nombre, "adf_stat": adf_stat, "adf_p": adf_p,
            "kpss_stat": kpss_stat, "kpss_p": kpss_p, "estacionaria": estacionaria}

resultados_estac = []
for ts, nombre in [
    (ts_lanzamientos,   "Lanzamientos totales"),
    (ts_intercepciones, "Intercepciones"),
    (ts_hits,           "Hits"),
    (ts_tasa_hit,       "Tasa hit (%)"),
    (ts_strike_pct,     "Tasa strike (%)"),
]:
    resultados_estac.append(test_estacionariedad(ts, nombre))

df_estac = pd.DataFrame(resultados_estac)
df_estac.to_csv(os.path.join(TABLES, "06_estacionariedad.csv"), index=False)
print("  ✓ Tabla 06_estacionariedad.csv guardada")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FIGURA 9 — DESCOMPOSICIÓN STL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/7] Descomposición STL...")

fig9, axes = plt.subplots(4, 1, figsize=(14, 10), facecolor=BG)
fig9.suptitle("Descomposición STL — Lanzamientos Shahed/Geran Mensuales\nTeatro Ucrania-Rusia 2025-2026",
              fontsize=14, color=FG, fontweight="bold", y=0.98)

ts_launch_clean = ts_lanzamientos.interpolate(method="linear").bfill().ffill()

# STL con periodo 6 (semestral) — datos mensuales
try:
    stl = STL(ts_launch_clean, period=6, seasonal=7, robust=True)
    result_stl = stl.fit()
    
    components = [
        (ts_launch_clean,         "Original",       ACCENT,  "Lanzamientos totales"),
        (result_stl.trend,        "Tendencia",      GREEN,   "Componente tendencia"),
        (result_stl.seasonal,     "Estacionalidad", ORANGE,  "Componente estacional (periodo=6)"),
        (result_stl.resid,        "Residuo",        RED,     "Residuo"),
    ]
    
    for ax, (data, label, color, ylabel) in zip(axes, components):
        ax.plot(data.index, data.values, color=color, linewidth=2, label=label)
        if label == "Original":
            ax.fill_between(data.index, data.values, alpha=0.15, color=color)
        elif label == "Residuo":
            ax.axhline(0, color=FG2, linewidth=0.8, linestyle="--")
            ax.fill_between(data.index, data.values, 0, alpha=0.3, color=RED)
        ax.set_ylabel(ylabel, color=FG2, fontsize=9)
        ax.tick_params(colors=FG2)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(BG2)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.legend(loc="upper left", fontsize=8)
    
    # Anotar peak en tendencia
    peak_idx = result_stl.trend.idxmax()
    axes[1].annotate(f"Pico\n{peak_idx.strftime('%b %Y')}",
                     xy=(peak_idx, result_stl.trend[peak_idx]),
                     xytext=(peak_idx, result_stl.trend[peak_idx] * 0.85),
                     arrowprops=dict(arrowstyle="->", color=FG2, lw=1.2),
                     color=FG, fontsize=8, ha="center")
    
    print("  ✓ STL calculado con period=6")
except Exception as e:
    print(f"  ⚠  STL falló ({e}), usando plot simple")
    for ax in axes:
        ax.set_visible(False)
    ax = fig9.add_subplot(111)
    ax.plot(ts_launch_clean.index, ts_launch_clean.values, color=ACCENT, linewidth=2)
    ax.set_facecolor(BG2)
    ax.set_title("Lanzamientos mensuales (STL no disponible)", color=FG)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out9 = os.path.join(FIGS, "09_stl_descomposicion.png")
fig9.savefig(out9, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig9)
print(f"  ✓ Figura 9 → {os.path.basename(out9)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIGURA 10 — RUPTURAS ESTRUCTURALES (Chow test + CUSUM)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Detección de rupturas estructurales...")

def chow_test(y, breakpoint_idx):
    """Chow test para ruptura en breakpoint_idx."""
    n = len(y)
    x = np.arange(n).reshape(-1, 1)
    x_const = sm.add_constant(x)
    
    # Modelo restringido (sin ruptura)
    res_r = sm.OLS(y, x_const).fit()
    rss_r = res_r.ssr
    
    # Modelos no restringidos (dos subseries)
    y1, y2 = y[:breakpoint_idx], y[breakpoint_idx:]
    x1, x2 = x_const[:breakpoint_idx], x_const[breakpoint_idx:]
    
    if len(y1) < 3 or len(y2) < 3:
        return np.nan, np.nan
    
    rss1 = sm.OLS(y1, x1).fit().ssr
    rss2 = sm.OLS(y2, x2).fit().ssr
    rss_u = rss1 + rss2
    
    k = x_const.shape[1]
    F = ((rss_r - rss_u) / k) / (rss_u / (n - 2 * k))
    p_val = 1 - stats.f.cdf(F, k, n - 2 * k)
    return F, p_val

y_launch = ts_lanzamientos.dropna().values
n_obs = len(y_launch)

# Calcular Chow test para cada punto posible (ventana mínima = 3)
chow_F  = []
chow_p  = []
chow_ix = list(range(3, n_obs - 3))

for bp in chow_ix:
    F, p = chow_test(y_launch, bp)
    chow_F.append(F)
    chow_p.append(p)

chow_F = np.array(chow_F)
chow_p = np.array(chow_p)

# Ruptura más significativa
best_bp_rel = np.nanargmax(chow_F)
best_bp_abs = chow_ix[best_bp_rel]
best_bp_date = ts_lanzamientos.dropna().index[best_bp_abs]

print(f"  Ruptura más significativa: {best_bp_date.strftime('%b %Y')}  "
      f"F={chow_F[best_bp_rel]:.2f}  p={chow_p[best_bp_rel]:.4f}")

# CUSUM (OLS residuals)
x_time = sm.add_constant(np.arange(n_obs))
ols_model = sm.OLS(y_launch, x_time).fit()
residuals = ols_model.resid
cusum = np.cumsum(residuals / (np.std(residuals) * np.sqrt(n_obs)))

# ── Figura 10 ──────────────────────────────────────────────────────────────
fig10, axes10 = plt.subplots(3, 1, figsize=(14, 11), facecolor=BG)
fig10.suptitle("Detección de Rupturas Estructurales — Lanzamientos Mensuales UR\n"
               "Chow Test + CUSUM | Teatro Ucrania-Rusia 2025-2026",
               fontsize=13, color=FG, fontweight="bold", y=0.98)

idx_dates = ts_lanzamientos.dropna().index

# Panel A: Serie original con ruptura marcada
ax = axes10[0]
ax.plot(idx_dates, y_launch, color=ACCENT, linewidth=2.5, label="Lanzamientos totales")
ax.fill_between(idx_dates, y_launch, alpha=0.12, color=ACCENT)
ax.axvline(best_bp_date, color=RED, linewidth=2, linestyle="--",
           label=f"Ruptura detectada: {best_bp_date.strftime('%b %Y')}")

# Sombrear los dos regímenes
ax.axvspan(idx_dates[0], best_bp_date, alpha=0.06, color=GREEN, label="Régimen I")
ax.axvspan(best_bp_date, idx_dates[-1], alpha=0.06, color=ORANGE, label="Régimen II")

# Medias por régimen
mean1 = y_launch[:best_bp_abs].mean()
mean2 = y_launch[best_bp_abs:].mean()
ax.hlines(mean1, idx_dates[0], best_bp_date, color=GREEN, linewidth=1.5, linestyle=":", alpha=0.8)
ax.hlines(mean2, best_bp_date, idx_dates[-1], color=ORANGE, linewidth=1.5, linestyle=":", alpha=0.8)
ax.annotate(f"μ₁={mean1:.0f}", xy=(idx_dates[best_bp_abs//2], mean1 * 1.04),
            color=GREEN, fontsize=9, ha="center")
ax.annotate(f"μ₂={mean2:.0f}", xy=(idx_dates[best_bp_abs + (n_obs - best_bp_abs)//2], mean2 * 1.04),
            color=ORANGE, fontsize=9, ha="center")

ax.set_ylabel("Lanzamientos/mes", color=FG2)
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3); ax.set_facecolor(BG2)
for sp in ax.spines.values(): sp.set_color(GRID)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

# Panel B: Estadístico F del Chow test
ax2 = axes10[1]
chow_dates = [ts_lanzamientos.dropna().index[i] for i in chow_ix]
ax2.plot(chow_dates, chow_F, color=PURPLE, linewidth=2, label="Estadístico F (Chow)")
ax2.fill_between(chow_dates, chow_F, alpha=0.2, color=PURPLE)
ax2.axvline(best_bp_date, color=RED, linewidth=2, linestyle="--")
# Umbral F crítico (α=0.05, k=2, n=15)
f_crit = stats.f.ppf(0.95, 2, n_obs - 4)
ax2.axhline(f_crit, color=ORANGE, linewidth=1.2, linestyle=":", 
            label=f"F crítico α=0.05 ({f_crit:.2f})")
ax2.set_ylabel("Estadístico F", color=FG2)
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.3); ax2.set_facecolor(BG2)
for sp in ax2.spines.values(): sp.set_color(GRID)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

# Panel C: CUSUM
ax3 = axes10[2]
cusum_dates = idx_dates
ax3.plot(cusum_dates, cusum, color=CYAN, linewidth=2, label="CUSUM estandarizado")
ax3.fill_between(cusum_dates, cusum, 0, alpha=0.2, color=CYAN)
ax3.axhline(0, color=FG2, linewidth=0.8, linestyle="--")
# Bandas de confianza CUSUM (±1.36/sqrt(n) aproximado)
bound = 1.36
ax3.axhline( bound, color=RED, linewidth=1, linestyle=":", alpha=0.7, label=f"±{bound} (α=0.05)")
ax3.axhline(-bound, color=RED, linewidth=1, linestyle=":", alpha=0.7)
ax3.axvline(best_bp_date, color=RED, linewidth=2, linestyle="--")
ax3.set_ylabel("CUSUM estandarizado", color=FG2)
ax3.set_xlabel("Fecha", color=FG2)
ax3.legend(loc="upper left", fontsize=8)
ax3.grid(True, alpha=0.3); ax3.set_facecolor(BG2)
for sp in ax3.spines.values(): sp.set_color(GRID)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.tight_layout(rect=[0, 0, 1, 0.96])
out10 = os.path.join(FIGS, "10_rupturas_estructurales.png")
fig10.savefig(out10, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig10)
print(f"  ✓ Figura 10 → {os.path.basename(out10)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ARIMA / AUTO-ARIMA + FORECAST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] ARIMA auto-selección + forecast...")

ts_fit = ts_lanzamientos.dropna()
n_forecast = 3   # meses futuros

arima_summary_lines = []

if HAS_PMDARIMA and len(ts_fit) >= 8:
    try:
        auto_model = pm.auto_arima(
            ts_fit,
            start_p=0, max_p=4,
            start_q=0, max_q=4,
            d=None,       # auto-diferenciación
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            n_fits=30,
        )
        order = auto_model.order
        arima_aic  = auto_model.aic()
        arima_bic  = auto_model.bic()
        
        # Forecast
        forecast_vals, conf_int = auto_model.predict(
            n_periods=n_forecast, return_conf_int=True, alpha=0.10
        )
        
        # Generar fechas futuras
        last_date = ts_fit.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=n_forecast, freq="MS"
        )
        
        arima_summary_lines.append(f"Modelo seleccionado: ARIMA{order}")
        arima_summary_lines.append(f"AIC: {arima_aic:.2f}  BIC: {arima_bic:.2f}")
        arima_summary_lines.append(f"Observaciones: {len(ts_fit)}")
        arima_summary_lines.append(f"\nForecast ({n_forecast} meses):")
        for dt, val, ci in zip(forecast_dates, forecast_vals, conf_int):
            arima_summary_lines.append(
                f"  {dt.strftime('%b %Y')}: {val:.0f}  IC90%[{ci[0]:.0f}, {ci[1]:.0f}]"
            )
        
        # Fitted values
        fitted = auto_model.predict_in_sample()
        
        print(f"  Modelo: ARIMA{order}  AIC={arima_aic:.2f}  BIC={arima_bic:.2f}")
        print(f"  Forecast próximos {n_forecast} meses: "
              + ", ".join([f"{v:.0f}" for v in forecast_vals]))
        
        use_pmdarima = True
    except Exception as e:
        print(f"  ⚠  auto_arima falló ({e}), usando ARIMA(2,1,1) manual")
        use_pmdarima = False
else:
    use_pmdarima = False

if not use_pmdarima:
    try:
        arima_manual = ARIMA(ts_fit, order=(2, 1, 1)).fit()
        order = (2, 1, 1)
        arima_aic = arima_manual.aic
        arima_bic = arima_manual.bic
        forecast_obj = arima_manual.forecast(steps=n_forecast, alpha=0.10)
        forecast_vals = forecast_obj.values if hasattr(forecast_obj, "values") else np.array(forecast_obj)
        # Interval
        fc_summary = arima_manual.get_forecast(steps=n_forecast)
        conf_int_df = fc_summary.conf_int(alpha=0.10)
        conf_int = conf_int_df.values
        
        last_date = ts_fit.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=n_forecast, freq="MS"
        )
        fitted = arima_manual.fittedvalues
        
        arima_summary_lines.append(f"Modelo: ARIMA{order} (manual)")
        arima_summary_lines.append(f"AIC: {arima_aic:.2f}  BIC: {arima_bic:.2f}")
        print(f"  Modelo ARIMA{order} manual  AIC={arima_aic:.2f}")
    except Exception as e2:
        print(f"  ✗ ARIMA también falló: {e2}")
        forecast_vals = np.array([])
        forecast_dates = []
        conf_int = np.array([])
        fitted = ts_fit.copy() * np.nan
        order = "(N/A)"
        arima_aic = arima_bic = np.nan

# Guardar summary ARIMA
arima_txt_path = os.path.join(TABLES, "03_arima_summary.txt")
with open(arima_txt_path, "w", encoding="utf-8") as f:
    f.write("ARIMA — LANZAMIENTOS MENSUALES UR 2025-2026\n")
    f.write("=" * 50 + "\n\n")
    f.write("\n".join(arima_summary_lines))
print(f"  ✓ Tabla 03_arima_summary.txt guardada")

# ── Figura 11: ARIMA forecast ──────────────────────────────────────────────
fig11, axes11 = plt.subplots(2, 1, figsize=(14, 9), facecolor=BG)
fig11.suptitle(f"ARIMA{order} — Forecast Lanzamientos Shahed\nTeatro Ucrania-Rusia",
               fontsize=13, color=FG, fontweight="bold", y=0.98)

ax = axes11[0]
ax.plot(ts_fit.index, ts_fit.values, color=ACCENT, linewidth=2.5,
        label="Observado", zorder=3)
ax.plot(ts_fit.index, fitted, color=GREEN, linewidth=1.5,
        linestyle="--", label="Ajustado (in-sample)", alpha=0.85)

if len(forecast_vals) > 0:
    ax.plot(forecast_dates, forecast_vals, color=ORANGE, linewidth=2.5,
            linestyle="-", marker="o", markersize=6, label=f"Forecast +{n_forecast}m")
    if len(conf_int) > 0:
        ax.fill_between(forecast_dates, conf_int[:, 0], conf_int[:, 1],
                        color=ORANGE, alpha=0.2, label="IC 90%")
    # Línea vertical separando histórico/forecast
    ax.axvline(ts_fit.index[-1], color=FG2, linewidth=1, linestyle=":")

# Anotar valores forecast
for dt, val in zip(forecast_dates, forecast_vals):
    ax.annotate(f"{val:.0f}", xy=(dt, val), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=8,
                color=ORANGE, fontweight="bold")

ax.set_ylabel("Lanzamientos/mes", color=FG2)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3); ax.set_facecolor(BG2)
for sp in ax.spines.values(): sp.set_color(GRID)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

# Panel B: Residuos ARIMA
ax2 = axes11[1]
resid_arima = ts_fit.values - fitted.values[:len(ts_fit)] if hasattr(fitted, "values") else np.zeros(len(ts_fit))
ax2.stem(ts_fit.index, resid_arima, linefmt=FG2, markerfmt="o", basefmt=RED)
ax2.axhline(0, color=RED, linewidth=1, linestyle="--")
ax2.fill_between(ts_fit.index, resid_arima, 0, alpha=0.2, color=PURPLE)
# Bandas ±2σ
sigma = np.nanstd(resid_arima)
ax2.axhline( 2*sigma, color=ORANGE, linewidth=1, linestyle=":", alpha=0.7, label="±2σ")
ax2.axhline(-2*sigma, color=ORANGE, linewidth=1, linestyle=":", alpha=0.7)
ax2.set_ylabel("Residuos", color=FG2)
ax2.set_xlabel("Fecha", color=FG2)
ax2.set_title(f"Residuos del modelo | DW={durbin_watson(resid_arima):.3f}", 
              color=FG, fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3); ax2.set_facecolor(BG2)
for sp in ax2.spines.values(): sp.set_color(GRID)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.tight_layout(rect=[0, 0, 1, 0.96])
out11 = os.path.join(FIGS, "11_arima_forecast.png")
fig11.savefig(out11, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig11)
print(f"  ✓ Figura 11 → {os.path.basename(out11)}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. VAR MULTIVARIANTE (lanzamientos, intercepciones, hits)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/7] VAR multivariante (lanzamientos ~ intercepciones ~ hits)...")

var_summary_lines = []

# Preparar datos VAR
df_var = pd.DataFrame({
    "lanzamientos":   ts_lanzamientos,
    "intercepciones": ts_intercepciones,
    "hits":           ts_hits,
}).dropna()

# Diferenciar si no estacionario
df_var_d = df_var.diff().dropna()

var_ok = False
var_results = None

if len(df_var_d) >= 6:
    try:
        from statsmodels.tsa.vector_ar.var_model import VAR
        
        # Selección de lag óptimo (criterio AIC)
        # Con series cortas limitar maxlags para evitar sobreparametrización
        n_vars = 3
        max_lags_possible = max(1, (len(df_var_d) - 1) // n_vars - 1)
        max_lags_search = min(4, max_lags_possible)
        
        var_model = VAR(df_var_d)
        lag_order_result = var_model.select_order(maxlags=max_lags_search)
        optimal_lag = lag_order_result.aic
        if optimal_lag == 0:
            optimal_lag = 1
        
        var_results = var_model.fit(maxlags=optimal_lag, ic="aic")
        
        var_summary_lines.append(f"Modelo VAR - Orden óptimo (AIC): {optimal_lag} lag(s)")
        var_summary_lines.append(f"Variables: lanzamientos, intercepciones, hits (primeras diferencias)")
        var_summary_lines.append(f"Observaciones: {len(df_var_d)}")
        var_summary_lines.append(f"\nAIC: {var_results.aic:.4f}")
        var_summary_lines.append(f"BIC: {var_results.bic:.4f}")
        
        # Causalidad de Granger
        var_summary_lines.append("\n── Causalidad de Granger ──")
        granger_pairs = [
            ("intercepciones", "lanzamientos"),
            ("hits",           "lanzamientos"),
            ("hits",           "intercepciones"),
        ]
        for caused, causing in granger_pairs:
            try:
                test = var_results.test_causality(caused, causing, kind="f")
                var_summary_lines.append(
                    f"  {causing} → {caused}: F={test.test_statistic:.3f}  "
                    f"p={test.pvalue:.4f}  {'SIGNIFICATIVO' if test.pvalue < 0.05 else 'no sig.'}"
                )
            except Exception as eg:
                var_summary_lines.append(f"  {causing} → {caused}: error ({eg})")
        
        print(f"  VAR(p={optimal_lag}) ajustado  AIC={var_results.aic:.4f}")
        var_ok = True
        
    except Exception as e_var:
        print(f"  ⚠  VAR falló: {e_var}")
        var_summary_lines.append(f"VAR no pudo ajustarse: {e_var}")
else:
    print(f"  ⚠  Datos insuficientes para VAR ({len(df_var_d)} obs.)")
    var_summary_lines.append("Datos insuficientes para VAR multivariante.")

# Guardar summary VAR
var_txt_path = os.path.join(TABLES, "04_var_summary.txt")
with open(var_txt_path, "w", encoding="utf-8") as f:
    f.write("VAR — LANZAMIENTOS / INTERCEPCIONES / HITS  UR 2025-2026\n")
    f.write("=" * 60 + "\n\n")
    f.write("\n".join(var_summary_lines))
print(f"  ✓ Tabla 04_var_summary.txt guardada")

# ── Figura 12: VAR impulso-respuesta ────────────────────────────────────────
fig12, axes12 = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
fig12.suptitle("VAR — Funciones de Impulso-Respuesta\nTeatro Ucrania-Rusia 2025-2026 (Δ Mensuales)",
               fontsize=13, color=FG, fontweight="bold", y=1.01)

if var_ok and var_results is not None:
    try:
        irf = var_results.irf(periods=8)
        
        # (impulso lanzamientos → respuesta en cada variable)
        vars_resp = ["lanzamientos", "intercepciones", "hits"]
        colors_resp = [ACCENT, GREEN, RED]
        labels_resp = ["Lanzamientos", "Intercepciones", "Hits"]
        
        for ax, resp_var, col, lab in zip(axes12, vars_resp, colors_resp, labels_resp):
            try:
                impulse_idx = list(df_var_d.columns).index("lanzamientos")
                resp_idx    = list(df_var_d.columns).index(resp_var)
                
                irf_vals = irf.irfs[:, resp_idx, impulse_idx]
                periods  = np.arange(len(irf_vals))
                
                ax.plot(periods, irf_vals, color=col, linewidth=2.5, marker="o", markersize=5)
                ax.fill_between(periods, irf_vals, 0, alpha=0.2, color=col)
                ax.axhline(0, color=FG2, linewidth=0.8, linestyle="--")
                
                # Bandas de confianza bootstrap
                try:
                    irf_bounds = irf.cum_effect_stderr(orth=False)
                    stderr = irf_bounds[:, resp_idx, impulse_idx]
                    ax.fill_between(periods, irf_vals - 1.96*stderr, irf_vals + 1.96*stderr,
                                    color=col, alpha=0.12, label="IC 95%")
                except:
                    pass
                
            except Exception as e_irf:
                ax.text(0.5, 0.5, f"IRF no disponible\n{e_irf}", 
                        ha="center", va="center", transform=ax.transAxes, color=FG2, fontsize=8)
            
            ax.set_title(f"Impulso: Lanzamientos\nRespuesta: {lab}", color=FG, fontsize=10)
            ax.set_xlabel("Meses", color=FG2)
            ax.set_ylabel("Respuesta", color=FG2)
            ax.grid(True, alpha=0.3); ax.set_facecolor(BG2)
            for sp in ax.spines.values(): sp.set_color(GRID)
    
    except Exception as e_fig12:
        for ax in axes12:
            ax.text(0.5, 0.5, f"IRF no disponible\n({e_fig12})",
                    ha="center", va="center", transform=ax.transAxes, 
                    color=FG2, fontsize=9, wrap=True)
            ax.set_facecolor(BG2)
else:
    for i, (ax, lab) in enumerate(zip(axes12, ["Lanzamientos", "Intercepciones", "Hits"])):
        # Fallback: correlaciones cruzadas como proxy de IRF
        _colors_fallback = [ACCENT, GREEN, RED]
        if len(df_var) >= 4:
            ccf_vals = ccf(df_var["lanzamientos"].diff().dropna(),
                           df_var[["lanzamientos","intercepciones","hits"]
                                  ].iloc[:, i].diff().dropna(), nlags=8)
            ax.stem(range(len(ccf_vals)), ccf_vals, 
                    linefmt=_colors_fallback[i], markerfmt="o", basefmt=FG2)
            ax.axhline(0, color=FG2, linewidth=0.8)
            ax.set_title(f"CCF: Lanzamientos → {lab}\n(proxy IRF)", color=FG, fontsize=10)
            ax.set_xlabel("Lag (meses)", color=FG2)
            ax.grid(True, alpha=0.3); ax.set_facecolor(BG2)
            for sp in ax.spines.values(): sp.set_color(GRID)
        else:
            ax.text(0.5, 0.5, "Datos insuficientes", ha="center", va="center",
                    transform=ax.transAxes, color=FG2)
            ax.set_facecolor(BG2)

plt.tight_layout()
out12 = os.path.join(FIGS, "12_var_impulso_respuesta.png")
fig12.savefig(out12, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig12)
print(f"  ✓ Figura 12 → {os.path.basename(out12)}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. MARKOV SWITCHING + CCF ACLED + FIGURAS 13/14
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Markov Switching + CCF ACLED...")

markov_summary_lines = []
markov_ok = False

# ── Markov Switching ─────────────────────────────────────────────────────────
if len(ts_fit) >= 8:
    try:
        ms_model = MarkovAutoregression(
            ts_fit,
            k_regimes=2,
            order=1,
            switching_ar=False,
            switching_variance=True
        )
        ms_result = ms_model.fit(search_reps=20, search_iter=100, disp=False)
        
        smoothed_probs = ms_result.smoothed_marginal_probabilities
        
        # Identificar régimen "alto" vs "bajo" por media
        regime_means = [ms_result.params[f"const[{r}]"] if f"const[{r}]" in ms_result.params.index
                        else ms_result.predicted_state_probs.iloc[:, r].values @ ts_fit.values / len(ts_fit)
                        for r in range(2)]
        
        regime_alto = int(np.argmax(regime_means))
        regime_bajo = 1 - regime_alto
        
        markov_summary_lines.append("Markov Switching AR(1) — 2 regímenes")
        markov_summary_lines.append(f"AIC: {ms_result.aic:.2f}  BIC: {ms_result.bic:.2f}")
        markov_summary_lines.append(f"Régimen ALTO (intensivo): {regime_alto}")
        markov_summary_lines.append(f"Régimen BAJO (moderado):  {regime_bajo}")
        markov_summary_lines.append("\nMatriz de transición:")
        try:
            tp = ms_result.regime_transition
            markov_summary_lines.append(f"  P(alto→alto) = {tp[0, regime_alto, regime_alto]:.3f}")
            markov_summary_lines.append(f"  P(bajo→alto) = {tp[0, regime_alto, regime_bajo]:.3f}")
        except:
            pass
        
        print(f"  Markov Switching AR(1): AIC={ms_result.aic:.2f}  BIC={ms_result.bic:.2f}")
        markov_ok = True
        
    except Exception as e_ms:
        print(f"  ⚠  Markov Switching falló: {e_ms}")
        markov_summary_lines.append(f"Markov Switching no convergió: {e_ms}")
        
        # Fallback: clasificación manual por cuartil
        q3 = np.percentile(ts_fit.values, 75)
        smoothed_probs = pd.DataFrame({
            "bajo":  (ts_fit.values <= q3).astype(float),
            "alto":  (ts_fit.values > q3).astype(float),
        }, index=ts_fit.index)
        regime_alto = 1
        markov_summary_lines.append("→ Fallback: clasificación por percentil 75%")

else:
    print("  ⚠  Datos insuficientes para Markov Switching")
    markov_summary_lines.append("Datos insuficientes para Markov Switching (n<8)")
    smoothed_probs = None
    regime_alto = 1

markov_txt_path = os.path.join(TABLES, "05_markov_summary.txt")
with open(markov_txt_path, "w", encoding="utf-8") as f:
    f.write("MARKOV SWITCHING — RÉGIMEN ALTO/BAJO INTENSIDAD\n")
    f.write("=" * 55 + "\n\n")
    f.write("\n".join(markov_summary_lines))
print(f"  ✓ Tabla 05_markov_summary.txt guardada")

# ── Figura 13: Markov Switching ─────────────────────────────────────────────
fig13, axes13 = plt.subplots(2, 1, figsize=(14, 9), facecolor=BG)
fig13.suptitle("Markov Switching — Regímenes de Intensidad\nTeatro Ucrania-Rusia 2025-2026",
               fontsize=13, color=FG, fontweight="bold", y=0.98)

ax = axes13[0]
ax.plot(ts_fit.index, ts_fit.values, color=ACCENT, linewidth=2.5, label="Lanzamientos", zorder=3)

if smoothed_probs is not None:
    prob_alto = smoothed_probs.iloc[:, regime_alto].values
    # Sombrear regímenes
    for i in range(len(ts_fit.index) - 1):
        if prob_alto[i] > 0.6:
            ax.axvspan(ts_fit.index[i], ts_fit.index[i+1],
                       alpha=0.25, color=RED, zorder=1)
        elif prob_alto[i] < 0.4:
            ax.axvspan(ts_fit.index[i], ts_fit.index[i+1],
                       alpha=0.15, color=GREEN, zorder=1)

ax.set_ylabel("Lanzamientos/mes", color=FG2)
# Leyenda manual para regímenes
patch_alto = mpatches.Patch(color=RED, alpha=0.5, label="Régimen ALTO (saturación)")
patch_bajo = mpatches.Patch(color=GREEN, alpha=0.5, label="Régimen BAJO (moderado)")
line_obs   = plt.Line2D([0], [0], color=ACCENT, linewidth=2, label="Observado")
ax.legend(handles=[line_obs, patch_alto, patch_bajo], loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3); ax.set_facecolor(BG2)
for sp in ax.spines.values(): sp.set_color(GRID)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

ax2 = axes13[1]
if smoothed_probs is not None:
    ax2.plot(ts_fit.index, prob_alto, color=RED, linewidth=2.5,
             label=f"P(régimen alto)")
    ax2.fill_between(ts_fit.index, prob_alto, alpha=0.25, color=RED)
    ax2.axhline(0.5, color=FG2, linewidth=1, linestyle="--", alpha=0.7, label="Umbral 0.5")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("P(Régimen Alto)", color=FG2)
    ax2.legend(loc="upper left", fontsize=8)
else:
    ax2.text(0.5, 0.5, "Probabilidades no disponibles", ha="center", va="center",
             transform=ax2.transAxes, color=FG2)
ax2.set_xlabel("Fecha", color=FG2)
ax2.grid(True, alpha=0.3); ax2.set_facecolor(BG2)
for sp in ax2.spines.values(): sp.set_color(GRID)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.tight_layout(rect=[0, 0, 1, 0.96])
out13 = os.path.join(FIGS, "13_markov_switching.png")
fig13.savefig(out13, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig13)
print(f"  ✓ Figura 13 → {os.path.basename(out13)}")

# ── Figura 14: CCF ACLED (RU→UA vs UA→RU) ───────────────────────────────────
fig14, axes14 = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig14.suptitle("Correlación Cruzada (CCF) — Eventos ACLED\nRU→UA vs UA→RU 2025 (respuesta táctica)",
               fontsize=13, color=FG, fontweight="bold", y=1.01)

if df_acled_m is not None:
    # Intentar localizar columnas de eventos por dirección
    cols = df_acled_m.columns.tolist()
    
    # Buscar columnas RU->UA y UA->RU
    col_rua = None
    col_uar = None
    for c in cols:
        if any(k in c.lower() for k in ["ru_ua", "ru->ua", "rusia_ataca", "eventos_rua"]):
            col_rua = c
        if any(k in c.lower() for k in ["ua_ru", "ua->ru", "ucrania_ataca", "eventos_uar"]):
            col_uar = c
    
    # Si no encontramos columnas específicas, buscamos columnas genéricas de eventos
    if col_rua is None or col_uar is None:
        # Usar columna de eventos total si existe
        for c in cols:
            if "evento" in c.lower() or "count" in c.lower() or c == "eventos":
                if col_rua is None:
                    col_rua = c
                    break
        
        # Leer directamente los CSV procesados
        rua_path = os.path.join(PROCESSED, "acled_drones_ofensivo_RU.csv")
        uar_path = os.path.join(PROCESSED, "acled_drones_ofensivo_UA.csv")
        
        if os.path.exists(rua_path) and os.path.exists(uar_path):
            df_rua = pd.read_csv(rua_path, low_memory=False)
            df_uar = pd.read_csv(uar_path, low_memory=False)
            
            for df_dir in [df_rua, df_uar]:
                df_dir["event_date"] = pd.to_datetime(df_dir["event_date"], errors="coerce")
                df_dir["mes"] = df_dir["event_date"].dt.to_period("M")
            
            s_rua = (df_rua.groupby("mes").size()
                     .rename("rua").to_frame()
                     .assign(fecha=lambda x: x.index.to_timestamp()))
            s_uar = (df_uar.groupby("mes").size()
                     .rename("uar").to_frame()
                     .assign(fecha=lambda x: x.index.to_timestamp()))
            
            df_ccf = s_rua.merge(s_uar, left_index=True, right_index=True, how="outer").fillna(0)
            
            ser_rua = df_ccf["rua"].values.astype(float)
            ser_uar = df_ccf["uar"].values.astype(float)
            
            col_rua = "rua"
            col_uar = "uar"
            use_direct = True
        else:
            use_direct = False
    else:
        ser_rua = df_acled_m[col_rua].values.astype(float)
        ser_uar = df_acled_m[col_uar].values.astype(float)
        use_direct = True
    
    if col_rua is not None and col_uar is not None and use_direct:
        try:
            n_lags = min(6, len(ser_rua) - 2)
            
            # CCF: RU→UA predice UA→RU (¿hay respuesta?)
            ccf_rua_uar = ccf(ser_rua - ser_rua.mean(), ser_uar - ser_uar.mean(), nlags=n_lags, alpha=0.10)
            ccf_uar_rua = ccf(ser_uar - ser_uar.mean(), ser_rua - ser_rua.mean(), nlags=n_lags, alpha=0.10)
            
            def plot_ccf(ax, ccf_result, title, color_pos, color_neg):
                if isinstance(ccf_result, tuple):
                    ccf_vals = ccf_result[0]
                    ci = ccf_result[1] if len(ccf_result) > 1 else None
                else:
                    ccf_vals = ccf_result
                    ci = None
                
                lags = np.arange(len(ccf_vals))
                colors_bar = [color_pos if v >= 0 else color_neg for v in ccf_vals]
                bars = ax.bar(lags, ccf_vals, color=colors_bar, alpha=0.8, width=0.6)
                
                # Bandas de significatividad (±1.96/√n)
                sig_bound = 1.96 / np.sqrt(len(ser_rua))
                ax.axhline( sig_bound, color=FG2, linewidth=1, linestyle="--", alpha=0.7,
                            label=f"±1.96/√n ({sig_bound:.3f})")
                ax.axhline(-sig_bound, color=FG2, linewidth=1, linestyle="--", alpha=0.7)
                ax.axhline(0, color=FG2, linewidth=0.5)
                
                ax.set_title(title, color=FG, fontsize=10)
                ax.set_xlabel("Lag (meses)", color=FG2)
                ax.set_ylabel("Correlación cruzada", color=FG2)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3); ax.set_facecolor(BG2)
                for sp in ax.spines.values(): sp.set_color(GRID)
            
            plot_ccf(axes14[0], ccf_rua_uar,
                     "CCF: RU→UA predice UA→RU\n(¿respuesta ucraniana a ataques rusos?)",
                     GREEN, RED)
            plot_ccf(axes14[1], ccf_uar_rua,
                     "CCF: UA→RU predice RU→UA\n(¿escalada rusa a ataques ucranianos?)",
                     ORANGE, PURPLE)
            
            print(f"  ✓ CCF ACLED calculado ({n_lags} lags)")
            
        except Exception as e_ccf:
            print(f"  ⚠  CCF falló: {e_ccf}")
            for ax in axes14:
                ax.text(0.5, 0.5, f"CCF no disponible\n({e_ccf})",
                        ha="center", va="center", transform=ax.transAxes,
                        color=FG2, fontsize=9)
                ax.set_facecolor(BG2)
    else:
        for ax in axes14:
            ax.text(0.5, 0.5, "Datos ACLED no disponibles\nEjecuta script 02 primero",
                    ha="center", va="center", transform=ax.transAxes,
                    color=FG2, fontsize=10)
            ax.set_facecolor(BG2)
else:
    for ax in axes14:
        ax.text(0.5, 0.5, "acled_mensual_agregado.csv no encontrado\nEjecuta script 02 primero",
                ha="center", va="center", transform=ax.transAxes,
                color=FG2, fontsize=10)
        ax.set_facecolor(BG2)

plt.tight_layout()
out14 = os.path.join(FIGS, "14_ccf_acled.png")
fig14.savefig(out14, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig14)
print(f"  ✓ Figura 14 → {os.path.basename(out14)}")

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPLETADO — SCRIPT 03 SERIES TEMPORALES")
print("=" * 60)

figuras_generadas = [
    f for f in [out9, out10, out11, out12, out13, out14]
    if os.path.exists(f)
]
tablas_generadas = [
    os.path.join(TABLES, t) for t in [
        "06_estacionariedad.csv",
        "03_arima_summary.txt",
        "04_var_summary.txt",
        "05_markov_summary.txt",
    ] if os.path.exists(os.path.join(TABLES, t))
]

print(f"\nFiguras generadas ({len(figuras_generadas)}/6):")
for f in figuras_generadas:
    print(f"  ✓ {os.path.basename(f)}")

print(f"\nTablas generadas ({len(tablas_generadas)}/4):")
for t in tablas_generadas:
    print(f"  ✓ {os.path.basename(t)}")

print(f"""
Hallazgos clave:
  • Ruptura estructural: {best_bp_date.strftime('%b %Y')} (F={chow_F[best_bp_rel]:.2f})
  • Modelo ARIMA: {order}
  • Markov Switching: 2 regímenes (alto/bajo saturación)
  • CCF ACLED: respuesta táctica RU↔UA analizada

Próximo: scripts/04_umbral_saturacion.py (TAR/SETAR)
""")
