"""
03_series_temporales.py  (v2)
==============================
Análisis de series temporales — Teatro Ucrania-Rusia 2025-2026

FUENTES:
  - Primaria  : data/raw/petro_attacks_2025_2026.csv  (455 obs. diarias, UA Air Force)
  - Secundaria: data/raw/ur_mensual_2025_2026.csv     (15 obs. mensuales, ISIS-Online)
  - ACLED     : data/processed/acled_mensual_agregado.csv

TÉCNICAS:
  1. STL  — descomposición tendencia/estacionalidad/residuo (serie diaria)
  2. Rupturas estructurales — Chow test + CUSUM (serie diaria agregada)
  3. ARIMA — auto-selección sobre serie diaria Shahed (pmdarima o manual)
  4. VAR   — lanzados ~ destruidos ~ hits (serie diaria, n≈455 → converge)
  5. Markov Switching AR(1) — regímenes alto/bajo (serie diaria)
  6. CCF   — correlación cruzada RU→UA / UA→RU (ACLED mensual)
  7. Figura resumen comparativa mensual ISIS-Online vs Petro

OUTPUTS:
  outputs/figures/09_stl_descomposicion.png
  outputs/figures/10_rupturas_estructurales.png
  outputs/figures/11_arima_forecast.png
  outputs/figures/12_var_impulso_respuesta.png
  outputs/figures/13_markov_switching.png
  outputs/figures/14_ccf_acled.png
  outputs/figures/15_comparativa_fuentes.png
  outputs/tables/03_arima_summary.txt
  outputs/tables/04_var_summary.txt
  outputs/tables/05_markov_summary.txt
  outputs/tables/06_estacionariedad.csv
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, ccf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.stattools import durbin_watson

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

# ─── Rutas ────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW     = os.path.join(BASE, "data", "raw")
PROC    = os.path.join(BASE, "data", "processed")
FIGS    = os.path.join(BASE, "outputs", "figures")
TABLES  = os.path.join(BASE, "outputs", "tables")
for d in [FIGS, TABLES, PROC]:
    os.makedirs(d, exist_ok=True)

# ─── Paleta dark theme ────────────────────────────────────────
BG    = "#0d1117"; BG2   = "#161b22"; GRID  = "#21262d"
FG    = "#e6edf3"; FG2   = "#8b949e"
AZUL  = "#58a6ff"; VERDE = "#3fb950"; ROJO  = "#f85149"
AMBAR = "#d29922"; LILA  = "#bc8cff"; CYAN  = "#39d353"

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": BG2,
    "axes.edgecolor": GRID,  "axes.labelcolor": FG,
    "axes.titlecolor": FG,   "xtick.color": FG2,
    "ytick.color": FG2,      "text.color": FG,
    "legend.facecolor": BG2, "legend.edgecolor": GRID,
    "grid.color": GRID,      "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
    "font.size": 10,         "axes.titlesize": 12,
})

def style_ax(ax):
    ax.set_facecolor(BG2)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

# ═════════════════════════════════════════════════════════════
# 1. CARGA Y PREPARACIÓN
# ═════════════════════════════════════════════════════════════
print("=" * 60)
print("SERIES TEMPORALES — DRONES 2025-2026  (v2)")
print("=" * 60)
print("\n[1/7] Cargando datos...")

# ── Petro diario ──────────────────────────────────────────────
petro_path = os.path.join(RAW, "petro_attacks_2025_2026.csv")
df_petro = pd.read_csv(petro_path)
df_petro["fecha"] = pd.to_datetime(df_petro["time_start"].str[:10], errors="coerce")
df_petro = df_petro.dropna(subset=["fecha"])
df_petro["launched"]  = pd.to_numeric(df_petro["launched"],  errors="coerce").fillna(0)
df_petro["destroyed"] = pd.to_numeric(df_petro["destroyed"], errors="coerce").fillna(0)

# Agregar por día (todos los modelos)
diario_total = (df_petro.groupby("fecha")
                .agg(lanzados=("launched","sum"),
                     destruidos=("destroyed","sum"))
                .reindex(pd.date_range(df_petro["fecha"].min(),
                                       df_petro["fecha"].max(), freq="D"))
                .fillna(0))
diario_total.index.name = "fecha"
diario_total["hits"]         = (diario_total["lanzados"] - diario_total["destruidos"]).clip(lower=0)
diario_total["tasa_interc"]  = diario_total["destruidos"] / diario_total["lanzados"].replace(0, np.nan)
diario_total["tasa_hit"]     = diario_total["hits"]       / diario_total["lanzados"].replace(0, np.nan)

# Solo Shahed/Geran
shahed_mask = df_petro["model"].str.contains("Shahed|Geran|Harpy", na=False, case=False)
diario_shahed = (df_petro[shahed_mask].groupby("fecha")
                 .agg(lanzados=("launched","sum"),
                      destruidos=("destroyed","sum"))
                 .reindex(diario_total.index)
                 .fillna(0))
diario_shahed["hits"]       = (diario_shahed["lanzados"] - diario_shahed["destruidos"]).clip(lower=0)
diario_shahed["tasa_interc"] = diario_shahed["destruidos"] / diario_shahed["lanzados"].replace(0, np.nan)

N_diario = len(diario_total)
print(f"  ✓ Petro diario: {N_diario} obs. "
      f"({diario_total.index.min().strftime('%d %b %Y')} → "
      f"{diario_total.index.max().strftime('%d %b %Y')})")

# ── ISIS-Online mensual ───────────────────────────────────────
df_ur = pd.read_csv(os.path.join(RAW, "ur_mensual_2025_2026.csv"))
df_ur["fecha_dt"] = pd.to_datetime(df_ur["fecha"].str.strip() + "-01", errors="coerce")
df_ur = df_ur.dropna(subset=["fecha_dt"]).sort_values("fecha_dt")
for col in ["lanzamientos_total","intercepciones","hits","tasa_hit_total_pct","tasa_strike_pct"]:
    df_ur[col] = pd.to_numeric(df_ur[col], errors="coerce")
ts_mensual = df_ur.set_index("fecha_dt")["lanzamientos_total"].asfreq("MS")
print(f"  ✓ ISIS-Online mensual: {len(ts_mensual)} obs.")

# ── ACLED mensual ─────────────────────────────────────────────
acled_path = os.path.join(PROC, "acled_mensual_agregado.csv")
df_acled_m = None
if os.path.exists(acled_path):
    df_acled_m = pd.read_csv(acled_path)
    df_acled_m["fecha_dt"] = pd.to_datetime(
        df_acled_m.get("mes_inicio", df_acled_m.get("fecha", "")), errors="coerce")
    df_acled_m = df_acled_m.dropna(subset=["fecha_dt"]).sort_values("fecha_dt")
    print(f"  ✓ ACLED mensual: {len(df_acled_m)} obs.")

# ═════════════════════════════════════════════════════════════
# 2. TESTS DE ESTACIONARIEDAD (serie diaria lanzados)
# ═════════════════════════════════════════════════════════════
print("\n[2/7] Tests de estacionariedad...")

series_test = {
    "Lanzados diario (total)":        diario_total["lanzados"],
    "Destruidos diario (total)":      diario_total["destruidos"],
    "Hits diario (total)":            diario_total["hits"],
    "Tasa intercepción diaria":       diario_total["tasa_interc"],
    "Lanzados Shahed diario":         diario_shahed["lanzados"],
    "Lanzamientos mensuales (ISIS)":  ts_mensual.dropna(),
}

rows_estac = []
for nombre, serie in series_test.items():
    s = serie.dropna()
    s = s[s > 0] if "tasa" not in nombre.lower() else s.dropna()
    if len(s) < 8:
        continue
    try:
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    except:
        adf_stat, adf_p = np.nan, np.nan
    try:
        kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    except:
        kpss_stat, kpss_p = np.nan, np.nan
    estac = (adf_p < 0.05) if not np.isnan(adf_p) else False
    print(f"  {nombre:40s}  ADF p={adf_p:.4f}  "
          f"{'✓ estacionaria' if estac else '✗ NO estacionaria'}")
    rows_estac.append({"serie": nombre, "n": len(s),
                       "adf_p": adf_p, "kpss_p": kpss_p,
                       "estacionaria": estac})

pd.DataFrame(rows_estac).to_csv(os.path.join(TABLES, "06_estacionariedad.csv"), index=False)
print("  ✓ 06_estacionariedad.csv")

# ═════════════════════════════════════════════════════════════
# 3. FIGURA 9 — STL sobre serie diaria
# ═════════════════════════════════════════════════════════════
print("\n[3/7] Descomposición STL (serie diaria)...")

# Usar lanzados Shahed diario con media móvil 7d para reducir ruido
serie_stl = diario_shahed["lanzados"].copy()
serie_stl_smooth = serie_stl.rolling(7, center=True, min_periods=4).mean().bfill().ffill()

fig9, axes9 = plt.subplots(4, 1, figsize=(14, 11), facecolor=BG)
fig9.suptitle("Descomposición STL — Lanzamientos Shahed/Geran Diarios\n"
              "Teatro Ucrania-Rusia 2025-2026  (Fuente: UA Air Force / Petro Ivaniuk)",
              fontsize=13, color=FG, fontweight="bold", y=0.99)

try:
    stl = STL(serie_stl_smooth, period=7, seasonal=13, robust=True)
    res_stl = stl.fit()
    componentes = [
        (serie_stl_smooth,   "Original (media 7d)", AZUL),
        (res_stl.trend,      "Tendencia",            VERDE),
        (res_stl.seasonal,   "Estacionalidad (sem)", AMBAR),
        (res_stl.resid,      "Residuo",              ROJO),
    ]
    for ax, (data, label, color) in zip(axes9, componentes):
        ax.plot(data.index, data.values, color=color, lw=1.5, label=label)
        if label == "Original (media 7d)":
            ax.fill_between(data.index, data.values, alpha=0.12, color=color)
        elif label == "Residuo":
            ax.axhline(0, color=FG2, lw=0.8, ls="--")
            ax.fill_between(data.index, data.values, 0, alpha=0.25, color=ROJO)
        elif label == "Tendencia":
            # Anotar máximo de tendencia
            peak = data.idxmax()
            ax.annotate(f"Máx\n{peak.strftime('%b %Y')}",
                        xy=(peak, data[peak]),
                        xytext=(peak, data[peak]*0.88),
                        arrowprops=dict(arrowstyle="->", color=FG2, lw=1),
                        color=FG, fontsize=8, ha="center")
        ax.set_ylabel(label, color=FG2, fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        style_ax(ax)
    print("  ✓ STL diario period=7")
except Exception as e:
    print(f"  ⚠  STL falló: {e}")
    axes9[0].plot(serie_stl_smooth.index, serie_stl_smooth.values, color=AZUL)
    for ax in axes9[1:]:
        ax.set_visible(False)
    style_ax(axes9[0])

plt.tight_layout(rect=[0, 0, 1, 0.97])
out9 = os.path.join(FIGS, "09_stl_descomposicion.png")
fig9.savefig(out9, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig9)
print(f"  ✓ Figura 9 → {os.path.basename(out9)}")

# ═════════════════════════════════════════════════════════════
# 4. FIGURA 10 — RUPTURAS ESTRUCTURALES (Chow + CUSUM)
# ═════════════════════════════════════════════════════════════
print("\n[4/7] Rupturas estructurales...")

# Trabajar con media semanal para reducir ruido (más robusto que diario)
semanal = diario_total["lanzados"].resample("W").mean().dropna()
y_sem   = semanal.values
n_sem   = len(y_sem)

def chow_test(y, bp):
    n = len(y)
    if bp < 3 or bp > n - 3:
        return np.nan, np.nan
    x = sm.add_constant(np.arange(n).reshape(-1, 1))
    rss_r = sm.OLS(y, x).fit().ssr
    rss1  = sm.OLS(y[:bp],  x[:bp]).fit().ssr
    rss2  = sm.OLS(y[bp:],  x[bp:]).fit().ssr
    k = 2
    F = ((rss_r - rss1 - rss2) / k) / ((rss1 + rss2) / (n - 2*k))
    return F, 1 - stats.f.cdf(F, k, n - 2*k)

chow_F, chow_p, chow_ix = [], [], list(range(4, n_sem - 4))
for bp in chow_ix:
    F, p = chow_test(y_sem, bp)
    chow_F.append(F); chow_p.append(p)

chow_F = np.array(chow_F)
chow_p = np.array(chow_p)
best_rel  = int(np.nanargmax(chow_F))
best_abs  = chow_ix[best_rel]
best_date = semanal.index[best_abs]

print(f"  Ruptura (semanal): {best_date.strftime('%d %b %Y')}  "
      f"F={chow_F[best_rel]:.2f}  p={chow_p[best_rel]:.4f}")

# CUSUM
x_t   = sm.add_constant(np.arange(n_sem))
resid = sm.OLS(y_sem, x_t).fit().resid
cusum = np.cumsum(resid / (np.std(resid) * np.sqrt(n_sem)))

fig10, axes10 = plt.subplots(3, 1, figsize=(14, 11), facecolor=BG)
fig10.suptitle("Detección de Rupturas Estructurales — Lanzamientos Semanales\n"
               f"Teatro Ucrania-Rusia 2025-2026  |  Ruptura: {best_date.strftime('%b %Y')}",
               fontsize=13, color=FG, fontweight="bold", y=0.99)

# Panel A: serie + ruptura
ax = axes10[0]
ax.plot(semanal.index, y_sem, color=AZUL, lw=2, label="Media semanal lanzados")
ax.fill_between(semanal.index, y_sem, alpha=0.1, color=AZUL)
ax.axvline(best_date, color=ROJO, lw=2, ls="--",
           label=f"Ruptura: {best_date.strftime('%b %Y')}")
ax.axvspan(semanal.index[0],  best_date, alpha=0.05, color=VERDE)
ax.axvspan(best_date, semanal.index[-1], alpha=0.05, color=AMBAR)
mu1 = y_sem[:best_abs].mean(); mu2 = y_sem[best_abs:].mean()
ax.hlines(mu1, semanal.index[0],  best_date, color=VERDE, lw=1.5, ls=":")
ax.hlines(mu2, best_date, semanal.index[-1], color=AMBAR, lw=1.5, ls=":")
ax.annotate(f"μ₁={mu1:.0f} UAV/sem", xy=(semanal.index[best_abs//2], mu1*1.05),
            color=VERDE, fontsize=9, ha="center")
ax.annotate(f"μ₂={mu2:.0f} UAV/sem",
            xy=(semanal.index[best_abs + (n_sem-best_abs)//2], mu2*1.05),
            color=AMBAR, fontsize=9, ha="center")
ax.set_ylabel("UAV lanzados/semana", color=FG2)
ax.legend(loc="upper left", fontsize=9)
style_ax(ax)

# Panel B: estadístico F
ax2 = axes10[1]
chow_dates = [semanal.index[i] for i in chow_ix]
ax2.plot(chow_dates, chow_F, color=LILA, lw=2, label="Estadístico F (Chow)")
ax2.fill_between(chow_dates, chow_F, alpha=0.2, color=LILA)
ax2.axvline(best_date, color=ROJO, lw=2, ls="--")
f_crit = stats.f.ppf(0.95, 2, n_sem - 4)
ax2.axhline(f_crit, color=AMBAR, lw=1.2, ls=":",
            label=f"F crítico α=0.05 ({f_crit:.2f})")
ax2.set_ylabel("Estadístico F", color=FG2)
ax2.legend(fontsize=8)
style_ax(ax2)

# Panel C: CUSUM
ax3 = axes10[2]
ax3.plot(semanal.index, cusum, color=CYAN, lw=2, label="CUSUM estandarizado")
ax3.fill_between(semanal.index, cusum, 0, alpha=0.2, color=CYAN)
ax3.axhline(0, color=FG2, lw=0.8, ls="--")
ax3.axhline( 1.36, color=ROJO, lw=1, ls=":", alpha=0.7, label="±1.36 (α=0.05)")
ax3.axhline(-1.36, color=ROJO, lw=1, ls=":", alpha=0.7)
ax3.axvline(best_date, color=ROJO, lw=2, ls="--")
ax3.set_ylabel("CUSUM estandarizado", color=FG2)
ax3.set_xlabel("Semana", color=FG2)
ax3.legend(fontsize=8)
style_ax(ax3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out10 = os.path.join(FIGS, "10_rupturas_estructurales.png")
fig10.savefig(out10, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig10)
print(f"  ✓ Figura 10 → {os.path.basename(out10)}")

# ═════════════════════════════════════════════════════════════
# 5. ARIMA — serie diaria Shahed (media móvil semanal)
# ═════════════════════════════════════════════════════════════
print("\n[5/7] ARIMA (serie diaria Shahed)...")

ts_arima = diario_shahed["lanzados"].resample("W").sum().dropna()
ts_arima = ts_arima[ts_arima > 0]
n_forecast = 8  # semanas

arima_lines = []
order, fitted, forecast_vals, forecast_dates, conf_int = None, None, [], [], np.array([])

if HAS_PMDARIMA and len(ts_arima) >= 12:
    try:
        model = pm.auto_arima(
            ts_arima, start_p=0, max_p=4, start_q=0, max_q=4,
            d=None, seasonal=True, m=52,
            information_criterion="aic", stepwise=True,
            suppress_warnings=True, error_action="ignore", n_fits=50,
        )
        order     = model.order
        s_order   = model.seasonal_order
        aic, bic  = model.aic(), model.bic()
        fvals, ci = model.predict(n_periods=n_forecast,
                                  return_conf_int=True, alpha=0.10)
        forecast_vals  = fvals
        conf_int       = ci
        fitted         = pd.Series(model.predict_in_sample(),
                                   index=ts_arima.index)
        last = ts_arima.index[-1]
        forecast_dates = pd.date_range(
            start=last + pd.DateOffset(weeks=1),
            periods=n_forecast, freq="W")
        arima_lines += [
            f"Modelo: ARIMA{order}x{s_order}",
            f"AIC={aic:.2f}  BIC={bic:.2f}",
            f"N observaciones: {len(ts_arima)}",
            f"\nForecast +{n_forecast} semanas:",
        ] + [f"  {d.strftime('%d %b %Y')}: {v:.0f} UAV  IC90%[{c[0]:.0f},{c[1]:.0f}]"
             for d, v, c in zip(forecast_dates, forecast_vals, conf_int)]
        print(f"  pmdarima ARIMA{order}  AIC={aic:.2f}")
    except Exception as e:
        print(f"  ⚠  pmdarima: {e}")
        HAS_PMDARIMA = False

if not HAS_PMDARIMA or fitted is None:
    for ord_ in [(2,1,1), (1,1,1), (0,1,1), (1,1,0)]:
        try:
            m = ARIMA(ts_arima, order=ord_).fit()
            order = ord_
            fitted = m.fittedvalues
            fc_obj = m.get_forecast(steps=n_forecast)
            forecast_vals  = fc_obj.predicted_mean.values
            conf_int       = fc_obj.conf_int(alpha=0.10).values
            last = ts_arima.index[-1]
            forecast_dates = pd.date_range(
                start=last + pd.DateOffset(weeks=1),
                periods=n_forecast, freq="W")
            arima_lines += [
                f"Modelo: ARIMA{order} (manual)",
                f"AIC={m.aic:.2f}  BIC={m.bic:.2f}",
            ]
            print(f"  ARIMA{order} manual  AIC={m.aic:.2f}")
            break
        except:
            continue

with open(os.path.join(TABLES, "03_arima_summary.txt"), "w", encoding="utf-8") as f:
    f.write("ARIMA — SHAHED SEMANAL 2025-2026\n" + "="*50 + "\n\n")
    f.write("\n".join(arima_lines))
print("  ✓ 03_arima_summary.txt")

fig11, axes11 = plt.subplots(2, 1, figsize=(14, 9), facecolor=BG)
fig11.suptitle(f"ARIMA{order} — Forecast Lanzamientos Shahed Semanales\n"
               "Teatro Ucrania-Rusia  (Fuente: UA Air Force)",
               fontsize=13, color=FG, fontweight="bold", y=0.99)

ax = axes11[0]
ax.plot(ts_arima.index, ts_arima.values,
        color=AZUL, lw=2.5, label="Observado (semanal)")
if fitted is not None:
    ax.plot(fitted.index, fitted.values,
            color=VERDE, lw=1.5, ls="--", alpha=0.85, label="Ajustado")
if len(forecast_vals) > 0:
    ax.plot(forecast_dates, forecast_vals,
            color=AMBAR, lw=2.5, marker="o", ms=5, label=f"Forecast +{n_forecast} sem.")
    if conf_int.size > 0:
        ax.fill_between(forecast_dates, conf_int[:,0], conf_int[:,1],
                        color=AMBAR, alpha=0.2, label="IC 90%")
    ax.axvline(ts_arima.index[-1], color=FG2, lw=1, ls=":")
    for d, v in zip(forecast_dates, forecast_vals):
        ax.annotate(f"{v:.0f}", xy=(d, v), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=7, color=AMBAR, fontweight="bold")
ax.set_ylabel("UAV Shahed lanzados/semana", color=FG2)
ax.legend(loc="upper left", fontsize=9)
style_ax(ax)

ax2 = axes11[1]
if fitted is not None:
    common = ts_arima.index.intersection(fitted.index)
    resid  = ts_arima[common].values - fitted[common].values
    ax2.stem(common, resid, linefmt=FG2+"80", markerfmt="o", basefmt=ROJO)
    ax2.axhline(0, color=ROJO, lw=1, ls="--")
    sig = np.nanstd(resid)
    ax2.axhline( 2*sig, color=AMBAR, lw=1, ls=":", alpha=0.7, label="±2σ")
    ax2.axhline(-2*sig, color=AMBAR, lw=1, ls=":", alpha=0.7)
    dw = durbin_watson(resid)
    ax2.set_title(f"Residuos  |  Durbin-Watson={dw:.3f}", color=FG, fontsize=10)
    ax2.legend(fontsize=8)
ax2.set_ylabel("Residuo (UAV/sem)", color=FG2)
ax2.set_xlabel("Semana", color=FG2)
style_ax(ax2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out11 = os.path.join(FIGS, "11_arima_forecast.png")
fig11.savefig(out11, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig11)
print(f"  ✓ Figura 11 → {os.path.basename(out11)}")

# ═════════════════════════════════════════════════════════════
# 6. VAR — lanzados / destruidos / hits  (semanal, n≈66)
# ═════════════════════════════════════════════════════════════
print("\n[6/7] VAR multivariante (semanal)...")

from statsmodels.tsa.vector_ar.var_model import VAR

df_var = pd.DataFrame({
    "lanzados":   diario_total["lanzados"].resample("W").sum(),
    "destruidos": diario_total["destruidos"].resample("W").sum(),
    "hits":       diario_total["hits"].resample("W").sum(),
}).dropna()

var_lines = []
var_ok    = False
var_res   = None

df_var_d = df_var.diff().dropna()
n_var    = len(df_var_d)
max_lags = max(1, min(4, (n_var - 1) // 3 - 1))

try:
    vm  = VAR(df_var_d)
    sel = vm.select_order(maxlags=max_lags)
    p   = sel.aic if sel.aic > 0 else 1
    var_res = vm.fit(maxlags=p, ic="aic")
    var_ok  = True
    var_lines += [
        f"VAR(p={p}) sobre primeras diferencias semanales",
        f"N obs: {n_var}   Variables: lanzados, destruidos, hits",
        f"AIC={var_res.aic:.4f}  BIC={var_res.bic:.4f}",
        "\n── Causalidad de Granger ──",
    ]
    for caused, causing in [("destruidos","lanzados"),
                             ("hits",      "lanzados"),
                             ("hits",      "destruidos")]:
        try:
            t = var_res.test_causality(caused, causing, kind="f")
            sig = "SIGNIFICATIVO ★" if t.pvalue < 0.05 else "no sig."
            var_lines.append(
                f"  {causing:12s} → {caused:12s}  "
                f"F={t.test_statistic:.3f}  p={t.pvalue:.4f}  {sig}")
        except Exception as eg:
            var_lines.append(f"  {causing} → {caused}: error ({eg})")
    print(f"  VAR(p={p})  n={n_var}  AIC={var_res.aic:.4f}")
except Exception as e:
    var_lines.append(f"VAR no estimado: {e}")
    print(f"  ⚠  VAR: {e}")

with open(os.path.join(TABLES, "04_var_summary.txt"), "w", encoding="utf-8") as f:
    f.write("VAR — LANZADOS/DESTRUIDOS/HITS SEMANAL\n" + "="*50 + "\n\n")
    f.write("\n".join(var_lines))
print("  ✓ 04_var_summary.txt")

fig12, axes12 = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
fig12.suptitle("VAR — Funciones de Impulso-Respuesta  (Δ Semanal)\n"
               "Impulso: Lanzamientos  →  Respuesta en cada variable",
               fontsize=13, color=FG, fontweight="bold", y=1.02)

resp_vars  = ["lanzados","destruidos","hits"]
resp_labs  = ["Lanzados","Destruidos","Hits"]
resp_cols  = [AZUL, VERDE, ROJO]

if var_ok and var_res is not None:
    try:
        irf = var_res.irf(periods=12)
        for ax, rv, lab, col in zip(axes12, resp_vars, resp_labs, resp_cols):
            ii = list(df_var_d.columns).index("lanzados")
            ri = list(df_var_d.columns).index(rv)
            vals = irf.irfs[:, ri, ii]
            xs   = np.arange(len(vals))
            ax.plot(xs, vals, color=col, lw=2.5, marker="o", ms=4)
            ax.fill_between(xs, vals, 0, alpha=0.15, color=col)
            ax.axhline(0, color=FG2, lw=0.8, ls="--")
            ax.set_title(f"Respuesta: {lab}", color=FG, fontsize=11)
            ax.set_xlabel("Semanas", color=FG2)
            ax.set_ylabel("Respuesta IRF", color=FG2)
            style_ax(ax); ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x,p: f"{int(x)}"))
    except Exception as e_irf:
        for ax, lab, col in zip(axes12, resp_labs, resp_cols):
            # Fallback: CCF semanal
            try:
                s_imp = df_var_d["lanzados"].values
                s_res = df_var_d[lab.lower()].values
                c     = ccf(s_imp - s_imp.mean(),
                            s_res - s_res.mean(), nlags=12)
                ax.bar(range(len(c)), c, color=col, alpha=0.75, width=0.6)
                ax.axhline(1.96/np.sqrt(n_var), color=AMBAR,
                           lw=1, ls=":", label="±1.96/√n")
                ax.axhline(-1.96/np.sqrt(n_var), color=AMBAR, lw=1, ls=":")
                ax.axhline(0, color=FG2, lw=0.5)
            except:
                pass
            ax.set_title(f"CCF proxy: {lab}", color=FG, fontsize=11)
            ax.set_xlabel("Lag (sem)", color=FG2)
            style_ax(ax); ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x,p: f"{int(x)}"))
else:
    for ax, lab, col in zip(axes12, resp_labs, resp_cols):
        ax.text(0.5, 0.5, "VAR no disponible", ha="center", va="center",
                transform=ax.transAxes, color=FG2)
        ax.set_facecolor(BG2)

plt.tight_layout()
out12 = os.path.join(FIGS, "12_var_impulso_respuesta.png")
fig12.savefig(out12, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig12)
print(f"  ✓ Figura 12 → {os.path.basename(out12)}")

# ═════════════════════════════════════════════════════════════
# 7. MARKOV SWITCHING + CCF ACLED + FIGURA COMPARATIVA
# ═════════════════════════════════════════════════════════════
print("\n[7/7] Markov Switching + CCF ACLED...")

from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

ts_ms  = diario_total["lanzados"].resample("W").mean().dropna()
ms_ok  = False
ms_res = None
ms_lines = []

if len(ts_ms) >= 20:
    for k_reg in [2, 3]:
        try:
            ms_m  = MarkovAutoregression(
                ts_ms, k_regimes=k_reg, order=1,
                switching_ar=False, switching_variance=True)
            ms_res = ms_m.fit(search_reps=30, search_iter=200, disp=False)
            ms_ok  = True
            ms_lines += [
                f"Markov Switching AR(1) — {k_reg} regímenes",
                f"AIC={ms_res.aic:.2f}  BIC={ms_res.bic:.2f}",
                f"N obs. semanales: {len(ts_ms)}",
            ]
            print(f"  Markov AR(1) k={k_reg}  AIC={ms_res.aic:.2f}  BIC={ms_res.bic:.2f}")
            break
        except Exception as e_ms:
            ms_lines.append(f"k={k_reg} falló: {e_ms}")
            continue

if not ms_ok:
    print("  ⚠  Markov Switching no convergió — usando clasificación cuantil")
    q75 = np.percentile(ts_ms.values, 75)
    q50 = np.percentile(ts_ms.values, 50)
    ms_lines.append(f"Fallback: cuantiles — alto>q75({q75:.0f}), medio q50-q75, bajo<q50({q50:.0f})")

with open(os.path.join(TABLES, "05_markov_summary.txt"), "w", encoding="utf-8") as f:
    f.write("MARKOV SWITCHING — RÉGIMEN INTENSIDAD\n" + "="*50 + "\n\n")
    f.write("\n".join(ms_lines))
print("  ✓ 05_markov_summary.txt")

# ── Figura 13: Markov ─────────────────────────────────────────
fig13, axes13 = plt.subplots(2, 1, figsize=(14, 9), facecolor=BG)
fig13.suptitle("Markov Switching — Regímenes de Intensidad  (Semanal)\n"
               "Teatro Ucrania-Rusia 2025-2026",
               fontsize=13, color=FG, fontweight="bold", y=0.99)

ax = axes13[0]
ax.plot(ts_ms.index, ts_ms.values, color=AZUL, lw=2, label="Lanzados/semana")

if ms_ok and ms_res is not None:
    probs = ms_res.smoothed_marginal_probabilities
    k_reg = probs.shape[1]
    # Identificar régimen "más alto" por media de los estados
    n_probs = len(probs)
    ts_ms_aligned = ts_ms.iloc[:n_probs]
    means = []
    for r in range(k_reg):
        idx_r = probs.iloc[:, r].values > 0.5
        means.append(ts_ms_aligned.values[idx_r].mean() if idx_r.sum() > 0 else 0)
    reg_alto = int(np.argmax(means))
    prob_alto = probs.iloc[:, reg_alto].values

    for i in range(len(prob_alto) - 1):
        if prob_alto[i] > 0.65:
            ax.axvspan(ts_ms_aligned.index[i], ts_ms_aligned.index[i+1],
                       alpha=0.25, color=ROJO, zorder=1)
        elif prob_alto[i] < 0.35 and k_reg == 2:
            ax.axvspan(ts_ms_aligned.index[i], ts_ms_aligned.index[i+1],
                       alpha=0.12, color=VERDE, zorder=1)

    ax2b = axes13[1]
    ax2b.plot(ts_ms_aligned.index, prob_alto, color=ROJO, lw=2.5,
              label=f"P(régimen alto)")
    ax2b.fill_between(ts_ms_aligned.index, prob_alto, alpha=0.2, color=ROJO)
    ax2b.axhline(0.5, color=FG2, lw=1, ls="--", alpha=0.7, label="Umbral 0.5")
    ax2b.set_ylim(-0.05, 1.05)
    ax2b.set_ylabel("P(Régimen Alto)", color=FG2)
    ax2b.legend(fontsize=8); style_ax(ax2b)
else:
    # Fallback visual por cuantil
    q75_v = np.percentile(ts_ms.values, 75)
    q50_v = np.percentile(ts_ms.values, 50)
    for i, (t, v) in enumerate(zip(ts_ms.index[:-1], ts_ms.values[:-1])):
        if v > q75_v:
            ax.axvspan(t, ts_ms.index[i+1], alpha=0.25, color=ROJO, zorder=1)
        elif v < q50_v:
            ax.axvspan(t, ts_ms.index[i+1], alpha=0.12, color=VERDE, zorder=1)
    axes13[1].plot(ts_ms.index, ts_ms.values,
                   color=AMBAR, lw=1.5, alpha=0.7)
    axes13[1].axhline(q75_v, color=ROJO, lw=1.5, ls="--",
                      label=f"Q75={q75_v:.0f} (umbral alto)")
    axes13[1].axhline(q50_v, color=VERDE, lw=1.5, ls="--",
                      label=f"Q50={q50_v:.0f} (umbral bajo)")
    axes13[1].legend(fontsize=8)
    axes13[1].set_ylabel("UAV/semana", color=FG2)
    style_ax(axes13[1])

patch_a = mpatches.Patch(color=ROJO, alpha=0.5, label="Régimen ALTO (saturación)")
patch_b = mpatches.Patch(color=VERDE, alpha=0.3, label="Régimen BAJO (moderado)")
ax.legend(handles=[plt.Line2D([0],[0],color=AZUL,lw=2,label="Observado"),
                   patch_a, patch_b], fontsize=8, loc="upper left")
ax.set_ylabel("UAV lanzados/semana", color=FG2)
style_ax(ax)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out13 = os.path.join(FIGS, "13_markov_switching.png")
fig13.savefig(out13, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig13)
print(f"  ✓ Figura 13 → {os.path.basename(out13)}")

# ── Figura 14: CCF ACLED ──────────────────────────────────────
fig14, axes14 = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig14.suptitle("Correlación Cruzada (CCF) — ACLED Ucrania 2025\n"
               "¿Los ataques RU→UA predicen respuesta UA→RU?",
               fontsize=13, color=FG, fontweight="bold", y=1.01)

ccf_ok = False
if df_acled_m is not None and "n_ru_ua" in df_acled_m.columns and "n_ua_ru" in df_acled_m.columns:
    try:
        s_ru = df_acled_m["n_ru_ua"].values.astype(float)
        s_ua = df_acled_m["n_ua_ru"].values.astype(float)
        n_ccf = len(s_ru)
        n_lags = min(6, n_ccf - 2)
        sig_b  = 1.96 / np.sqrt(n_ccf)

        for ax, (s_x, s_y, titulo, col) in zip(axes14, [
            (s_ru, s_ua, "RU→UA predice UA→RU\n(respuesta ucraniana)", VERDE),
            (s_ua, s_ru, "UA→RU predice RU→UA\n(escalada rusa)",        AMBAR),
        ]):
            c = ccf(s_x - s_x.mean(), s_y - s_y.mean(), nlags=n_lags, alpha=0.10)
            cvals = c[0] if isinstance(c, tuple) else c
            lags  = np.arange(len(cvals))
            colors_bar = [col if v >= 0 else ROJO for v in cvals]
            ax.bar(lags, cvals, color=colors_bar, alpha=0.8, width=0.6)
            ax.axhline( sig_b, color=FG2, lw=1.2, ls="--",
                        label=f"±1.96/√n ({sig_b:.2f})")
            ax.axhline(-sig_b, color=FG2, lw=1.2, ls="--")
            ax.axhline(0, color=FG2, lw=0.5)
            ax.set_title(titulo, color=FG, fontsize=10)
            ax.set_xlabel("Lag (meses)", color=FG2)
            ax.set_ylabel("Correlación cruzada", color=FG2)
            ax.legend(fontsize=8); style_ax(ax)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f"{int(x)}"))
        ccf_ok = True
        print(f"  ✓ CCF ACLED  n={n_ccf} meses  {n_lags} lags")
    except Exception as e_ccf:
        print(f"  ⚠  CCF: {e_ccf}")

if not ccf_ok:
    for ax in axes14:
        ax.text(0.5, 0.5, "Ejecuta script 02 primero",
                ha="center", va="center", transform=ax.transAxes, color=FG2)
        ax.set_facecolor(BG2)

plt.tight_layout()
out14 = os.path.join(FIGS, "14_ccf_acled.png")
fig14.savefig(out14, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig14)
print(f"  ✓ Figura 14 → {os.path.basename(out14)}")

# ── Figura 15: comparativa fuentes ISIS vs Petro ──────────────
fig15, axes15 = plt.subplots(2, 1, figsize=(14, 9), facecolor=BG)
fig15.suptitle("Validación Cruzada de Fuentes — ISIS-Online vs UA Air Force\n"
               "Lanzamientos mensuales Shahed 2025-2026",
               fontsize=13, color=FG, fontweight="bold", y=0.99)

# Petro mensual
petro_m = (diario_total["lanzados"]
           .resample("MS").sum()
           .rename("petro_lanzados"))
isis_m  = ts_mensual.rename("isis_lanzados")

df_comp = pd.DataFrame({"ISIS-Online": isis_m, "UA Air Force (Petro)": petro_m}).dropna()

ax = axes15[0]
ax.plot(df_comp.index, df_comp["ISIS-Online"],
        color=AZUL, lw=2.5, marker="o", ms=5, label="ISIS-Online (Shahed total)")
ax.plot(df_comp.index, df_comp["UA Air Force (Petro)"],
        color=VERDE, lw=2.5, marker="s", ms=5, ls="--",
        label="UA Air Force / Petro (todos modelos)")
ax.fill_between(df_comp.index,
                df_comp["ISIS-Online"],
                df_comp["UA Air Force (Petro)"],
                alpha=0.08, color=AMBAR, label="Diferencia entre fuentes")
ax.set_ylabel("UAV lanzados/mes", color=FG2)
ax.legend(loc="upper left", fontsize=9); style_ax(ax)

# Diferencia relativa
ax2 = axes15[1]
diff = ((df_comp["UA Air Force (Petro)"] - df_comp["ISIS-Online"])
        / df_comp["ISIS-Online"] * 100)
colors_diff = [VERDE if v >= 0 else ROJO for v in diff.values]
ax2.bar(df_comp.index, diff.values, color=colors_diff, alpha=0.8, width=20)
ax2.axhline(0, color=FG2, lw=0.8, ls="--")
ax2.axhline( 15, color=AMBAR, lw=1, ls=":", alpha=0.7, label="±15%")
ax2.axhline(-15, color=AMBAR, lw=1, ls=":", alpha=0.7)
ax2.set_ylabel("Diferencia relativa (%)", color=FG2)
ax2.set_xlabel("Mes", color=FG2)
ax2.set_title("Petro vs ISIS-Online  (% diferencia)", color=FG, fontsize=10)
ax2.legend(fontsize=8); style_ax(ax2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out15 = os.path.join(FIGS, "15_comparativa_fuentes.png")
fig15.savefig(out15, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig15)
print(f"  ✓ Figura 15 → {os.path.basename(out15)}")

# ═════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COMPLETADO — SCRIPT 03 v2")
print("=" * 60)

figs_ok = [f for f in [out9,out10,out11,out12,out13,out14,out15]
           if os.path.exists(f)]
print(f"\nFiguras: {len(figs_ok)}/7")
for f in figs_ok:
    print(f"  ✓ {os.path.basename(f)}")

print(f"""
Hallazgos clave (datos reales):
  • Serie diaria UA Air Force: {N_diario} observaciones
  • Ruptura estructural (semanal): {best_date.strftime('%d %b %Y')}
    μ₁={mu1:.0f} UAV/sem  →  μ₂={mu2:.0f} UAV/sem  (+{100*(mu2-mu1)/mu1:.0f}%)
  • ARIMA{order}: forecast +{n_forecast} semanas disponible
  • VAR semanal: {'convergió' if var_ok else 'no convergió'}
  • Markov Switching: {'convergió' if ms_ok else 'clasificación por cuantil'}
  • CCF ACLED: {'calculado' if ccf_ok else 'no disponible'}

Próximo: scripts/04_umbral_saturacion.py (TAR/SETAR)
""")
