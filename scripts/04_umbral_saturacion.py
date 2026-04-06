"""
04_umbral_saturacion.py
========================
Análisis de umbral y saturación — Teatro Ucrania-Rusia 2025-2026

HIPÓTESIS CENTRAL:
  La relación entre volumen de ataque (lanzados) y efectividad (tasa_hit)
  es NO LINEAL: existe un umbral τ a partir del cual la tasa de impacto
  cae bruscamente por saturación de la defensa aérea.

  H1: tasa_hit ~ f(lanzados)  con cambio de régimen en τ ≈ 400-500 UAV/día
  H2: por encima de τ, la función de daño sigue una curva logarítmica/asintótica
  H3: la relación destruidos/lanzados también cambia de régimen en τ

TÉCNICAS:
  1. Análisis exploratorio no lineal (scatter + LOWESS)
  2. Búsqueda del umbral óptimo τ por grid search (R² condicional)
  3. TAR — Threshold Autoregression (regímenes arriba/abajo del umbral)
  4. SETAR — Self-Exciting TAR (umbral endógeno sobre la propia serie)
  5. Regresión segmentada (piecewise linear regression)
  6. Ajuste logarítmico: tasa_hit = a - b·log(lanzados) para lanzados > τ
  7. Análisis de supervivencia: ¿cuánto aguanta la defensa antes de saturar?
  8. Comparativa ataques <τ vs >τ (t-test, effect size Cohen's d)

FUENTES:
  data/raw/petro_attacks_2025_2026.csv     (diario, 459 obs.)
  data/raw/ur_mensual_2025_2026.csv        (mensual, 15 obs.)
  data/raw/ur_ataques_grandes_2025_2026.csv (ataques individuales)

OUTPUTS:
  outputs/figures/16_scatter_umbral_lowess.png
  outputs/figures/17_grid_search_umbral.png
  outputs/figures/18_tar_regimenes.png
  outputs/figures/19_regresion_segmentada.png
  outputs/figures/20_curva_logaritmica.png
  outputs/figures/21_supervivencia_defensa.png
  outputs/tables/07_umbral_resultados.csv
  outputs/tables/08_tar_coeficientes.txt
  outputs/tables/09_comparativa_regimenes.csv
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
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# ─── Rutas ────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW    = os.path.join(BASE, "data", "raw")
PROC   = os.path.join(BASE, "data", "processed")
FIGS   = os.path.join(BASE, "outputs", "figures")
TABLES = os.path.join(BASE, "outputs", "tables")
for d in [FIGS, TABLES]:
    os.makedirs(d, exist_ok=True)

# ─── Paleta dark theme ────────────────────────────────────────
BG    = "#0d1117"; BG2   = "#161b22"; GRID  = "#21262d"
FG    = "#e6edf3"; FG2   = "#8b949e"
AZUL  = "#58a6ff"; VERDE = "#3fb950"; ROJO  = "#f85149"
AMBAR = "#d29922"; LILA  = "#bc8cff"; CYAN  = "#39d353"
PALETTE = [AZUL, VERDE, ROJO, AMBAR, LILA, CYAN]

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

# ═════════════════════════════════════════════════════════════
# 1. CARGA Y PREPARACIÓN
# ═════════════════════════════════════════════════════════════
print("=" * 60)
print("ANÁLISIS DE UMBRAL Y SATURACIÓN — DRONES 2025-2026")
print("=" * 60)
print("\n[1/8] Cargando datos...")

# ── Petro diario ──────────────────────────────────────────────
df_petro = pd.read_csv(os.path.join(RAW, "petro_attacks_2025_2026.csv"))
df_petro["fecha"] = pd.to_datetime(df_petro["time_start"].str[:10], errors="coerce")
df_petro = df_petro.dropna(subset=["fecha"])
df_petro["launched"]  = pd.to_numeric(df_petro["launched"],  errors="coerce").fillna(0)
df_petro["destroyed"] = pd.to_numeric(df_petro["destroyed"], errors="coerce").fillna(0)

# Agregar por día
diario = (df_petro.groupby("fecha")
          .agg(lanzados=("launched","sum"),
               destruidos=("destroyed","sum"))
          .reset_index())
diario["hits"]        = (diario["lanzados"] - diario["destruidos"]).clip(lower=0)
diario["tasa_interc"] = diario["destruidos"] / diario["lanzados"].replace(0, np.nan)
diario["tasa_hit"]    = diario["hits"]       / diario["lanzados"].replace(0, np.nan)
diario = diario[diario["lanzados"] > 0].copy()

print(f"  ✓ Petro diario: {len(diario)} obs. con lanzados > 0")
print(f"    Rango lanzados: {diario['lanzados'].min():.0f} – {diario['lanzados'].max():.0f} UAV/día")
print(f"    Tasa hit media: {diario['tasa_hit'].mean():.3f}  "
      f"mediana: {diario['tasa_hit'].median():.3f}")

# ── Shahed específico ─────────────────────────────────────────
shahed_mask = df_petro["model"].str.contains("Shahed|Geran|Harpy", na=False, case=False)
diario_sh = (df_petro[shahed_mask].groupby("fecha")
             .agg(lanzados=("launched","sum"),
                  destruidos=("destroyed","sum"))
             .reset_index())
diario_sh["hits"]        = (diario_sh["lanzados"] - diario_sh["destruidos"]).clip(lower=0)
diario_sh["tasa_hit"]    = diario_sh["hits"]       / diario_sh["lanzados"].replace(0, np.nan)
diario_sh["tasa_interc"] = diario_sh["destruidos"] / diario_sh["lanzados"].replace(0, np.nan)
diario_sh = diario_sh[diario_sh["lanzados"] > 0].copy()

# ── ISIS-Online mensual ───────────────────────────────────────
df_ur = pd.read_csv(os.path.join(RAW, "ur_mensual_2025_2026.csv"))
df_ur["fecha_dt"] = pd.to_datetime(df_ur["fecha"].str.strip() + "-01", errors="coerce")
for col in ["lanzamientos_total","intercepciones","hits","tasa_hit_total_pct"]:
    df_ur[col] = pd.to_numeric(df_ur[col], errors="coerce")

# ── Ataques grandes ───────────────────────────────────────────
df_ag = pd.read_csv(os.path.join(RAW, "ur_ataques_grandes_2025_2026.csv"))
df_ag["fecha_dt"]   = pd.to_datetime(df_ag["fecha"], errors="coerce")
df_ag["uavs_total"] = pd.to_numeric(df_ag["uavs_total"], errors="coerce")
df_ag["tasa_hit"]   = pd.to_numeric(df_ag["tasa_hit_pct"], errors="coerce") / 100
df_ag = df_ag.dropna(subset=["fecha_dt","uavs_total"])

print(f"  ✓ ISIS-Online mensual: {len(df_ur)} obs.")
print(f"  ✓ Ataques grandes: {len(df_ag)} eventos")

# Variable de trabajo principal: Shahed diario (más limpio)
X_all  = diario_sh["lanzados"].values
Y_hit  = diario_sh["tasa_hit"].values
Y_int  = diario_sh["tasa_interc"].values
mask_v = ~(np.isnan(X_all) | np.isnan(Y_hit))
X = X_all[mask_v]
Y = Y_hit[mask_v]
Yi = Y_int[mask_v]
N = len(X)
print(f"\n  Serie de análisis: {N} obs. Shahed diario con tasa_hit válida")

# ═════════════════════════════════════════════════════════════
# 2. EXPLORACIÓN VISUAL + LOWESS
# ═════════════════════════════════════════════════════════════
print("\n[2/8] Exploración no lineal + LOWESS...")

# LOWESS sobre scatter lanzados vs tasa_hit
lw_hit  = lowess(Y,  X, frac=0.35, return_sorted=True)
lw_int  = lowess(Yi, X, frac=0.35, return_sorted=True)

fig16, axes16 = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig16.suptitle("Relación Volumen de Ataque vs Efectividad — LOWESS\n"
               "Teatro Ucrania-Rusia 2025-2026  (Shahed diario)",
               fontsize=13, color=FG, fontweight="bold", y=1.01)

# ── Panel izquierdo: lanzados vs tasa_hit ────────────────────
ax = axes16[0]
sc = ax.scatter(X, Y, c=AZUL, alpha=0.25, s=20, zorder=2, label="Observado")
ax.plot(lw_hit[:,0], lw_hit[:,1], color=AMBAR, lw=3, zorder=3, label="LOWESS")

# Línea τ = 400 como referencia teórica
ax.axvline(400, color=ROJO, lw=1.8, ls="--", alpha=0.8, label="τ=400 (ref. teórico)")
ax.axvline(200, color=LILA, lw=1.2, ls=":", alpha=0.6, label="τ=200")

ax.set_xlabel("UAV Shahed lanzados/día", color=FG2)
ax.set_ylabel("Tasa de impacto (hits/lanzados)", color=FG2)
ax.set_title("Volumen → Tasa de Impacto", color=FG)
ax.legend(fontsize=9); style_ax(ax)
ax.set_ylim(-0.02, None)

# ── Panel derecho: lanzados vs tasa_intercepción ─────────────
ax2 = axes16[1]
ax2.scatter(X, Yi, c=VERDE, alpha=0.25, s=20, zorder=2, label="Observado")
ax2.plot(lw_int[:,0], lw_int[:,1], color=AMBAR, lw=3, zorder=3, label="LOWESS")
ax2.axvline(400, color=ROJO, lw=1.8, ls="--", alpha=0.8, label="τ=400 (ref.)")

ax2.set_xlabel("UAV Shahed lanzados/día", color=FG2)
ax2.set_ylabel("Tasa de intercepción (destruidos/lanzados)", color=FG2)
ax2.set_title("Volumen → Tasa de Intercepción", color=FG)
ax2.legend(fontsize=9); style_ax(ax2)

plt.tight_layout()
out16 = os.path.join(FIGS, "16_scatter_umbral_lowess.png")
fig16.savefig(out16, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig16)
print(f"  ✓ Figura 16 → {os.path.basename(out16)}")

# ═════════════════════════════════════════════════════════════
# 3. GRID SEARCH DEL UMBRAL ÓPTIMO τ
# ═════════════════════════════════════════════════════════════
print("\n[3/8] Grid search umbral óptimo τ...")

# Para cada τ candidato: ajustar dos regresiones lineales (segmentada)
# y calcular R² total. El τ que maximiza R² es el umbral óptimo.
tau_range = np.arange(50, int(X.max() * 0.85), 10)
r2_scores = []
rss_scores = []

def r2_segmentada(X, Y, tau):
    """R² de regresión piecewise en τ."""
    mask1 = X <= tau
    mask2 = X > tau
    if mask1.sum() < 3 or mask2.sum() < 3:
        return np.nan, np.nan
    def r2_seg(x, y):
        if len(x) < 2:
            return 0
        x_ = x.reshape(-1, 1)
        m  = LinearRegression().fit(x_, y)
        return r2_score(y, m.predict(x_))
    r2_1 = r2_seg(X[mask1], Y[mask1])
    r2_2 = r2_seg(X[mask2], Y[mask2])
    n1, n2 = mask1.sum(), mask2.sum()
    r2_w = (n1 * r2_1 + n2 * r2_2) / (n1 + n2)
    rss = (np.sum((Y[mask1] - LinearRegression().fit(X[mask1].reshape(-1,1), Y[mask1])
                   .predict(X[mask1].reshape(-1,1)))**2) +
           np.sum((Y[mask2] - LinearRegression().fit(X[mask2].reshape(-1,1), Y[mask2])
                   .predict(X[mask2].reshape(-1,1)))**2))
    return r2_w, rss

for tau in tau_range:
    r2w, rss = r2_segmentada(X, Y, tau)
    r2_scores.append(r2w)
    rss_scores.append(rss)

r2_scores  = np.array(r2_scores)
rss_scores = np.array(rss_scores)

# Umbral óptimo por RSS mínimo (más robusto con ruido)
best_idx   = int(np.nanargmin(rss_scores))
tau_opt    = tau_range[best_idx]
tau_r2_max = tau_range[int(np.nanargmax(r2_scores))]

print(f"  τ óptimo (min RSS):  {tau_opt} UAV/día")
print(f"  τ óptimo (max R²):   {tau_r2_max} UAV/día")
print(f"  R² global sin umbral: {r2_score(Y, LinearRegression().fit(X.reshape(-1,1), Y).predict(X.reshape(-1,1))):.4f}")
print(f"  R² segmentado en τ={tau_opt}: {r2_scores[best_idx]:.4f}")

# ── Figura 17: grid search ────────────────────────────────────
fig17, axes17 = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig17.suptitle(f"Grid Search del Umbral Óptimo τ\n"
               f"τ (min RSS) = {tau_opt} UAV/día  |  τ (max R²) = {tau_r2_max} UAV/día",
               fontsize=13, color=FG, fontweight="bold", y=1.01)

ax = axes17[0]
ax.plot(tau_range, r2_scores, color=AZUL, lw=2, label="R² ponderado")
ax.axvline(tau_r2_max, color=AMBAR, lw=2, ls="--",
           label=f"τ óptimo R² = {tau_r2_max}")
ax.axvline(400, color=ROJO, lw=1.5, ls=":", alpha=0.7, label="τ=400 teórico")
ax.set_xlabel("Umbral τ (UAV/día)", color=FG2)
ax.set_ylabel("R² ponderado", color=FG2)
ax.set_title("R² según umbral", color=FG)
ax.legend(fontsize=9); style_ax(ax)

ax2 = axes17[1]
ax2.plot(tau_range, rss_scores, color=VERDE, lw=2, label="RSS total")
ax2.axvline(tau_opt, color=AMBAR, lw=2, ls="--",
            label=f"τ óptimo RSS = {tau_opt}")
ax2.axvline(400, color=ROJO, lw=1.5, ls=":", alpha=0.7, label="τ=400 teórico")
ax2.set_xlabel("Umbral τ (UAV/día)", color=FG2)
ax2.set_ylabel("RSS (suma residuos²)", color=FG2)
ax2.set_title("RSS según umbral (menor = mejor)", color=FG)
ax2.legend(fontsize=9); style_ax(ax2)

plt.tight_layout()
out17 = os.path.join(FIGS, "17_grid_search_umbral.png")
fig17.savefig(out17, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig17)
print(f"  ✓ Figura 17 → {os.path.basename(out17)}")

# ═════════════════════════════════════════════════════════════
# 4. TAR — REGÍMENES ARRIBA/ABAJO DEL UMBRAL
# ═════════════════════════════════════════════════════════════
print("\n[4/8] TAR — regímenes alto/bajo...")

tau = tau_opt  # usar el óptimo encontrado

mask_bajo = X <= tau
mask_alto = X > tau

# Regresión lineal por régimen
def fit_regime(x, y, nombre):
    if len(x) < 3:
        return None, None, None
    x_  = sm.add_constant(x)
    m   = sm.OLS(y, x_).fit()
    return m, m.params, m.rsquared

m_bajo, p_bajo, r2_bajo = fit_regime(X[mask_bajo], Y[mask_bajo], "Bajo")
m_alto, p_alto, r2_alto = fit_regime(X[mask_alto], Y[mask_alto], "Alto")

n_bajo = mask_bajo.sum()
n_alto = mask_alto.sum()

tar_lines = [
    f"TAR — Threshold Autoregression",
    f"Umbral τ = {tau} UAV/día",
    f"",
    f"Régimen BAJO (lanzados ≤ {tau}):  n={n_bajo}",
]
if p_bajo is not None:
    tar_lines += [
        f"  Intercepto = {p_bajo[0]:.4f}",
        f"  Pendiente  = {p_bajo[1]:.6f}",
        f"  R²         = {r2_bajo:.4f}",
        f"  Interpretación: cada UAV adicional {'aumenta' if p_bajo[1]>0 else 'reduce'} "
        f"la tasa de impacto en {abs(p_bajo[1]):.4f}",
    ]

tar_lines += [f"", f"Régimen ALTO (lanzados > {tau}):  n={n_alto}"]
if p_alto is not None:
    tar_lines += [
        f"  Intercepto = {p_alto[0]:.4f}",
        f"  Pendiente  = {p_alto[1]:.6f}",
        f"  R²         = {r2_alto:.4f}",
        f"  Interpretación: cada UAV adicional {'aumenta' if p_alto[1]>0 else 'reduce'} "
        f"la tasa de impacto en {abs(p_alto[1]):.4f}",
    ]

# Test de cambio de pendiente (Chow adaptado)
if p_bajo is not None and p_alto is not None:
    F_tar, p_tar = stats.f_oneway(Y[mask_bajo], Y[mask_alto])
    tar_lines += [
        f"",
        f"── Test diferencia de medias (F) ──",
        f"  F={F_tar:.3f}  p={p_tar:.4f}  "
        f"{'Diferencia SIGNIFICATIVA ★' if p_tar < 0.05 else 'No significativa'}",
        f"  Media tasa_hit bajo τ:  {Y[mask_bajo].mean():.4f}",
        f"  Media tasa_hit alto τ:  {Y[mask_alto].mean():.4f}",
        f"  Diferencia media:       {Y[mask_bajo].mean() - Y[mask_alto].mean():.4f}",
    ]
    # Cohen's d
    pooled_std = np.sqrt((Y[mask_bajo].var() * (n_bajo-1) +
                          Y[mask_alto].var() * (n_alto-1)) / (n_bajo + n_alto - 2))
    d_cohen = abs(Y[mask_bajo].mean() - Y[mask_alto].mean()) / pooled_std if pooled_std > 0 else 0
    tar_lines.append(f"  Cohen's d = {d_cohen:.3f}  "
                     f"({'grande' if d_cohen>0.8 else 'medio' if d_cohen>0.5 else 'pequeño'})")
    print(f"  τ={tau}  bajo(n={n_bajo}, μ={Y[mask_bajo].mean():.3f}) "
          f"alto(n={n_alto}, μ={Y[mask_alto].mean():.3f})  "
          f"F={F_tar:.2f} p={p_tar:.4f}  d={d_cohen:.2f}")

with open(os.path.join(TABLES, "08_tar_coeficientes.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(tar_lines))
print("  ✓ 08_tar_coeficientes.txt")

# ── Figura 18: TAR regímenes ──────────────────────────────────
fig18, axes18 = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig18.suptitle(f"TAR — Regímenes de Impacto según Umbral τ={tau} UAV/día\n"
               "Teatro Ucrania-Rusia 2025-2026",
               fontsize=13, color=FG, fontweight="bold", y=1.01)

ax = axes18[0]
ax.scatter(X[mask_bajo], Y[mask_bajo], color=VERDE, alpha=0.35, s=22,
           label=f"Régimen BAJO (n={n_bajo}, μ={Y[mask_bajo].mean():.3f})", zorder=2)
ax.scatter(X[mask_alto], Y[mask_alto], color=ROJO,  alpha=0.35, s=22,
           label=f"Régimen ALTO (n={n_alto}, μ={Y[mask_alto].mean():.3f})", zorder=2)

x_fit_b = np.linspace(X[mask_bajo].min(), tau, 100)
x_fit_a = np.linspace(tau, X[mask_alto].max(), 100)
if p_bajo is not None:
    ax.plot(x_fit_b, p_bajo[0] + p_bajo[1]*x_fit_b,
            color=VERDE, lw=2.5, ls="-", label=f"OLS bajo  R²={r2_bajo:.3f}")
if p_alto is not None:
    ax.plot(x_fit_a, p_alto[0] + p_alto[1]*x_fit_a,
            color=ROJO, lw=2.5, ls="-", label=f"OLS alto  R²={r2_alto:.3f}")

ax.axvline(tau, color=AMBAR, lw=2.5, ls="--", label=f"τ = {tau} UAV/día")
ax.set_xlabel("UAV Shahed lanzados/día", color=FG2)
ax.set_ylabel("Tasa de impacto (hits/lanzados)", color=FG2)
ax.set_title("Lanzados vs Tasa de Impacto por Régimen", color=FG)
ax.legend(fontsize=8, loc="upper right"); style_ax(ax)

# Panel derecho: distribución de tasa_hit por régimen
ax2 = axes18[1]
parts = ax2.violinplot([Y[mask_bajo], Y[mask_alto]],
                       positions=[1, 2], showmedians=True, showextrema=True)
for pc, col in zip(parts["bodies"], [VERDE, ROJO]):
    pc.set_facecolor(col); pc.set_alpha(0.5)
parts["cmedians"].set_color(AMBAR)
parts["cmedians"].set_linewidth(2)
ax2.set_xticks([1, 2])
ax2.set_xticklabels([f"Bajo τ\n(≤{tau} UAV/d)", f"Alto τ\n(>{tau} UAV/d)"], color=FG2)
ax2.set_ylabel("Tasa de impacto", color=FG2)
ax2.set_title(f"Distribución tasa_hit por régimen\n"
              f"F={F_tar:.2f}  p={p_tar:.4f}  Cohen's d={d_cohen:.2f}",
              color=FG, fontsize=10)
if p_tar < 0.05:
    ax2.text(1.5, ax2.get_ylim()[1] * 0.95, "★ p < 0.05",
             ha="center", color=AMBAR, fontsize=11, fontweight="bold")
style_ax(ax2)

plt.tight_layout()
out18 = os.path.join(FIGS, "18_tar_regimenes.png")
fig18.savefig(out18, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig18)
print(f"  ✓ Figura 18 → {os.path.basename(out18)}")

# ═════════════════════════════════════════════════════════════
# 5. REGRESIÓN SEGMENTADA (Piecewise linear)
# ═════════════════════════════════════════════════════════════
print("\n[5/8] Regresión segmentada...")

def piecewise_linear(x, tau, a1, b1, b2):
    """Modelo lineal por tramos con nodo en tau."""
    return np.where(x <= tau,
                    a1 + b1 * x,
                    a1 + b1 * tau + b2 * (x - tau))

try:
    from scipy.optimize import curve_fit as cf_scipy
    x_sort = np.sort(X)
    p0 = [tau, Y[X.argmin()], 0.001, -0.001]
    bounds = ([X.min(), -1, -0.1, -0.1],
              [X.max(),  1,  0.1,  0.1])
    popt, pcov = cf_scipy(piecewise_linear, X, Y, p0=p0, bounds=bounds, maxfev=10000)
    tau_pw, a1_pw, b1_pw, b2_pw = popt
    y_pred_pw = piecewise_linear(X, *popt)
    r2_pw = r2_score(Y, y_pred_pw)
    print(f"  Piecewise: τ={tau_pw:.0f}  b1={b1_pw:.5f}  b2={b2_pw:.5f}  R²={r2_pw:.4f}")
    pw_ok = True
except Exception as e:
    print(f"  ⚠  Piecewise: {e} — usando τ fijo")
    tau_pw = tau
    b1_pw  = np.polyfit(X[mask_bajo], Y[mask_bajo], 1)[0] if mask_bajo.sum() > 2 else 0
    b2_pw  = np.polyfit(X[mask_alto], Y[mask_alto], 1)[0] if mask_alto.sum() > 2 else 0
    a1_pw  = np.mean(Y[mask_bajo]) - b1_pw * np.mean(X[mask_bajo])
    y_pred_pw = piecewise_linear(X, tau_pw, a1_pw, b1_pw, b2_pw)
    r2_pw  = r2_score(Y, y_pred_pw) if len(y_pred_pw) > 0 else 0
    pw_ok  = False

# ─── Figura 19 ────────────────────────────────────────────────
fig19, axes19 = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig19.suptitle(f"Regresión Segmentada (Piecewise)  τ={tau_pw:.0f} UAV/día\n"
               "Cambio de pendiente en el umbral de saturación",
               fontsize=13, color=FG, fontweight="bold", y=1.01)

ax = axes19[0]
ax.scatter(X, Y, color=AZUL, alpha=0.2, s=18, zorder=1, label="Observado")
x_line = np.linspace(X.min(), X.max(), 500)
y_line = piecewise_linear(x_line, tau_pw, a1_pw, b1_pw, b2_pw)
ax.plot(x_line, y_line, color=AMBAR, lw=3, zorder=3,
        label=f"Piecewise R²={r2_pw:.3f}")
ax.plot(lw_hit[:,0], lw_hit[:,1], color=LILA, lw=2, ls="--",
        alpha=0.8, label="LOWESS", zorder=2)
ax.axvline(tau_pw, color=ROJO, lw=2, ls="--",
           label=f"τ = {tau_pw:.0f} UAV/día")
ax.set_xlabel("UAV Shahed lanzados/día", color=FG2)
ax.set_ylabel("Tasa de impacto", color=FG2)
ax.set_title("Ajuste piecewise vs LOWESS", color=FG)
ax.legend(fontsize=9); style_ax(ax)

# Residuos del modelo segmentado
ax2 = axes19[1]
resid_pw = Y - y_pred_pw
ax2.scatter(X, resid_pw, color=VERDE, alpha=0.3, s=18)
ax2.axhline(0, color=ROJO, lw=1.5, ls="--")
ax2.axhline( 2*resid_pw.std(), color=AMBAR, lw=1, ls=":", alpha=0.7, label="±2σ")
ax2.axhline(-2*resid_pw.std(), color=AMBAR, lw=1, ls=":", alpha=0.7)
ax2.axvline(tau_pw, color=ROJO, lw=1.5, ls="--", alpha=0.5)
ax2.set_xlabel("UAV Shahed lanzados/día", color=FG2)
ax2.set_ylabel("Residuo", color=FG2)
ax2.set_title(f"Residuos piecewise  σ={resid_pw.std():.4f}", color=FG)
ax2.legend(fontsize=8); style_ax(ax2)

plt.tight_layout()
out19 = os.path.join(FIGS, "19_regresion_segmentada.png")
fig19.savefig(out19, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig19)
print(f"  ✓ Figura 19 → {os.path.basename(out19)}")

# ═════════════════════════════════════════════════════════════
# 6. CURVA LOGARÍTMICA  tasa_hit = a - b·log(lanzados)
# ═════════════════════════════════════════════════════════════
print("\n[6/8] Ajuste función logarítmica de saturación...")

def log_sat(x, a, b):
    """tasa_hit = a - b * log(x)  — curva de saturación."""
    return a - b * np.log(x)

def exp_decay(x, a, b, c):
    """tasa_hit = a * exp(-b * x) + c  — decaimiento exponencial."""
    return a * np.exp(-b * x) + c

def power_law(x, a, b):
    """tasa_hit = a * x^(-b)  — ley de potencia."""
    return a * np.power(x, -b)

# Filtrar outliers extremos para el ajuste (percentil 2-98)
p2, p98 = np.percentile(X, 2), np.percentile(X, 98)
fit_mask = (X >= p2) & (X <= p98) & (Y >= 0) & (Y <= 1)
Xf, Yf  = X[fit_mask], Y[fit_mask]

modelos_fit = {}

# Logarítmico
try:
    popt_log, _ = curve_fit(log_sat, Xf, Yf, p0=[0.5, 0.05], maxfev=5000)
    y_log = log_sat(Xf, *popt_log)
    r2_log = r2_score(Yf, y_log)
    modelos_fit["logarítmico"] = (log_sat, popt_log, r2_log,
                                   f"a={popt_log[0]:.4f}, b={popt_log[1]:.4f}")
    print(f"  Logarítmico: a={popt_log[0]:.4f} b={popt_log[1]:.4f}  R²={r2_log:.4f}")
except Exception as e:
    print(f"  ⚠  Log: {e}")

# Exponencial
try:
    popt_exp, _ = curve_fit(exp_decay, Xf, Yf,
                            p0=[0.3, 0.002, 0.02],
                            bounds=([0,0,0],[1,0.1,0.5]), maxfev=10000)
    y_exp = exp_decay(Xf, *popt_exp)
    r2_exp = r2_score(Yf, y_exp)
    modelos_fit["exponencial"] = (exp_decay, popt_exp, r2_exp,
                                   f"a={popt_exp[0]:.4f}, b={popt_exp[1]:.5f}, c={popt_exp[2]:.4f}")
    print(f"  Exponencial: R²={r2_exp:.4f}")
except Exception as e:
    print(f"  ⚠  Exp: {e}")

# Ley de potencia
try:
    popt_pow, _ = curve_fit(power_law, Xf, Yf + 1e-6,
                            p0=[5, 0.5], bounds=([0,0],[100,5]), maxfev=10000)
    y_pow = power_law(Xf, *popt_pow)
    r2_pow = r2_score(Yf, y_pow)
    modelos_fit["potencia"] = (power_law, popt_pow, r2_pow,
                                f"a={popt_pow[0]:.4f}, b={popt_pow[1]:.4f}")
    print(f"  Ley potencia: R²={r2_pow:.4f}")
except Exception as e:
    print(f"  ⚠  Potencia: {e}")

mejor_modelo = max(modelos_fit.items(), key=lambda x: x[1][2]) if modelos_fit else None
if mejor_modelo:
    print(f"  ★ Mejor modelo: {mejor_modelo[0]}  R²={mejor_modelo[1][2]:.4f}")

# ─── Figura 20 ────────────────────────────────────────────────
fig20, axes20 = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig20.suptitle("Modelos de Saturación — Ajuste No Lineal\n"
               "tasa_hit vs lanzados Shahed",
               fontsize=13, color=FG, fontweight="bold", y=1.01)

ax = axes20[0]
ax.scatter(Xf, Yf, color=AZUL, alpha=0.2, s=18, zorder=1, label="Observado")

colores_modelo = [AMBAR, VERDE, LILA, CYAN]
x_curve = np.linspace(Xf.min(), Xf.max(), 500)

for (nombre, (func, popt, r2, params)), col in zip(modelos_fit.items(), colores_modelo):
    try:
        y_curve = func(x_curve, *popt)
        valid = np.isfinite(y_curve) & (y_curve >= -0.1) & (y_curve <= 1.1)
        ax.plot(x_curve[valid], y_curve[valid], color=col, lw=2.5,
                label=f"{nombre}  R²={r2:.3f}")
    except:
        pass

ax.axvline(tau, color=ROJO, lw=1.8, ls="--", alpha=0.7, label=f"τ={tau}")
ax.set_xlabel("UAV Shahed lanzados/día", color=FG2)
ax.set_ylabel("Tasa de impacto", color=FG2)
ax.set_title("Comparativa modelos de saturación", color=FG)
ax.set_ylim(-0.02, None)
ax.legend(fontsize=8); style_ax(ax)

# Panel derecho: evolución temporal con régimen marcado
ax2 = axes20[1]
dates_sh = diario_sh["fecha"].values
x_t = diario_sh["lanzados"].values
y_t = diario_sh["tasa_hit"].values

# Colorear puntos por régimen
colors_t = [ROJO if v > tau else VERDE for v in x_t]
ax2.scatter(dates_sh, y_t, c=colors_t, alpha=0.35, s=15, zorder=2)

# Media móvil 14 días
roll_hit = pd.Series(y_t, index=pd.to_datetime(dates_sh)).rolling(14, center=True, min_periods=7).mean()
ax2.plot(roll_hit.index, roll_hit.values, color=AMBAR, lw=2.5, zorder=3,
         label="Media móvil 14d")

ax2.axhline(Y[mask_bajo].mean(), color=VERDE, lw=1.5, ls="--", alpha=0.7,
            label=f"μ bajo τ = {Y[mask_bajo].mean():.3f}")
ax2.axhline(Y[mask_alto].mean(), color=ROJO, lw=1.5, ls="--", alpha=0.7,
            label=f"μ alto τ = {Y[mask_alto].mean():.3f}")

ax2.set_xlabel("Fecha", color=FG2)
ax2.set_ylabel("Tasa de impacto diaria", color=FG2)
ax2.set_title("Evolución temporal — rojo=alto τ, verde=bajo τ", color=FG)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax2.legend(fontsize=8, loc="upper right"); style_ax(ax2)

patch_a = mpatches.Patch(color=ROJO, alpha=0.5, label=f"Ataques > τ ({mask_alto.sum()})")
patch_b = mpatches.Patch(color=VERDE, alpha=0.5, label=f"Ataques ≤ τ ({mask_bajo.sum()})")
ax2.legend(handles=[ax2.lines[0], patch_a, patch_b], fontsize=8, loc="upper right")

plt.tight_layout()
out20 = os.path.join(FIGS, "20_curva_logaritmica.png")
fig20.savefig(out20, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig20)
print(f"  ✓ Figura 20 → {os.path.basename(out20)}")

# ═════════════════════════════════════════════════════════════
# 7. ANÁLISIS DE SUPERVIVENCIA DE LA DEFENSA
# ═════════════════════════════════════════════════════════════
print("\n[7/8] Análisis de supervivencia de la defensa...")

# Definir "evento" = día en que la defensa NO aguanta (tasa_interc < umbral_def)
# Umbral defensivo: tasa_intercepción cae por debajo del 60%
UMBRAL_DEF = 0.60

# Para cada ataque: ¿aguantó (1) o falló (0) la defensa?
diario_sh_v = diario_sh[diario_sh["tasa_interc"].notna()].copy()
diario_sh_v["defensa_aguanta"] = (diario_sh_v["tasa_interc"] >= UMBRAL_DEF).astype(int)
diario_sh_v["fallo_defensa"]   = 1 - diario_sh_v["defensa_aguanta"]

# Curva Kaplan-Meier simplificada por cuartiles de volumen
cuartiles = pd.qcut(diario_sh_v["lanzados"], q=4,
                    labels=["Q1 (bajo)", "Q2", "Q3", "Q4 (alto)"])
diario_sh_v["cuartil"] = cuartiles

# Tasa de fallo por cuartil
resumen_cuartil = (diario_sh_v.groupby("cuartil", observed=True)
                   .agg(n=("fallo_defensa","count"),
                        fallos=("fallo_defensa","sum"),
                        tasa_fallo=("fallo_defensa","mean"),
                        lanzados_media=("lanzados","mean"),
                        tasa_interc_media=("tasa_interc","mean"))
                   .reset_index())
resumen_cuartil["tasa_exito"] = 1 - resumen_cuartil["tasa_fallo"]

print("  Tasa de fallo defensivo por cuartil de volumen:")
print(resumen_cuartil[["cuartil","lanzados_media","tasa_fallo",
                         "tasa_interc_media"]].to_string(index=False))

# Curva de supervivencia acumulada (% días que la defensa aguanta >= X%)
umbral_range = np.linspace(0.3, 0.99, 50)
surv_curve = [(diario_sh_v["tasa_interc"] >= u).mean() for u in umbral_range]

# ─── Figura 21 ────────────────────────────────────────────────
fig21, axes21 = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig21.suptitle(f"Supervivencia de la Defensa Aérea\n"
               f"¿Cuándo falla la intercepción? (umbral = {UMBRAL_DEF:.0%})",
               fontsize=13, color=FG, fontweight="bold", y=1.01)

ax = axes21[0]
colores_q = [VERDE, AZUL, AMBAR, ROJO]
for i, row in resumen_cuartil.iterrows():
    ax.bar(i, row["tasa_fallo"], color=colores_q[i], alpha=0.8, width=0.6,
           label=f"{row['cuartil']}: {row['lanzados_media']:.0f} UAV/d")
    ax.text(i, row["tasa_fallo"] + 0.01, f"{row['tasa_fallo']:.1%}",
            ha="center", color=FG2, fontsize=9)

ax.axhline(0.5, color=AMBAR, lw=1.5, ls="--", alpha=0.7, label="50% fallo")
ax.set_xticks(range(len(resumen_cuartil)))
ax.set_xticklabels(resumen_cuartil["cuartil"].astype(str), color=FG2, fontsize=9)
ax.set_ylabel(f"Tasa de fallo (tasa_interc < {UMBRAL_DEF:.0%})", color=FG2)
ax.set_title("Tasa de fallo por cuartil de volumen", color=FG)
ax.legend(fontsize=8, loc="upper left"); style_ax(ax)

ax2 = axes21[1]
ax2.plot(umbral_range, surv_curve, color=AZUL, lw=3, label="Curva de supervivencia")
ax2.fill_between(umbral_range, surv_curve, alpha=0.15, color=AZUL)
ax2.axvline(UMBRAL_DEF, color=AMBAR, lw=2, ls="--",
            label=f"Umbral referencia {UMBRAL_DEF:.0%}")
ax2.axvline(0.80, color=VERDE, lw=1.5, ls=":", alpha=0.8, label="80% intercepción")
ax2.axvline(0.95, color=ROJO, lw=1.5, ls=":", alpha=0.8, label="95% intercepción")
ax2.set_xlabel("Umbral de tasa de intercepción", color=FG2)
ax2.set_ylabel("Fracción de días que se supera el umbral", color=FG2)
ax2.set_title("Curva de 'supervivencia' de la defensa\n"
              "(% días que la intercepción supera X%)", color=FG)
ax2.legend(fontsize=9); style_ax(ax2)

plt.tight_layout()
out21 = os.path.join(FIGS, "21_supervivencia_defensa.png")
fig21.savefig(out21, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig21)
print(f"  ✓ Figura 21 → {os.path.basename(out21)}")

# ═════════════════════════════════════════════════════════════
# 8. TABLA RESUMEN + COMPARATIVA REGÍMENES
# ═════════════════════════════════════════════════════════════
print("\n[8/8] Tablas de resultados...")

# Tabla 07: resultados del umbral
umbral_results = {
    "tau_grid_rss":    tau_opt,
    "tau_grid_r2":     tau_r2_max,
    "tau_piecewise":   round(tau_pw),
    "n_bajo_tau":      int(n_bajo),
    "n_alto_tau":      int(n_alto),
    "media_hit_bajo":  round(float(Y[mask_bajo].mean()), 4),
    "media_hit_alto":  round(float(Y[mask_alto].mean()), 4),
    "r2_bajo":         round(float(r2_bajo) if r2_bajo else 0, 4),
    "r2_alto":         round(float(r2_alto) if r2_alto else 0, 4),
    "r2_piecewise":    round(float(r2_pw), 4),
    "f_stat":          round(float(F_tar), 4),
    "p_valor":         round(float(p_tar), 6),
    "cohen_d":         round(float(d_cohen), 4),
}
if "logarítmico" in modelos_fit:
    umbral_results["r2_log"] = round(modelos_fit["logarítmico"][2], 4)
if "exponencial" in modelos_fit:
    umbral_results["r2_exp"] = round(modelos_fit["exponencial"][2], 4)
if mejor_modelo:
    umbral_results["mejor_modelo_nonlineal"] = mejor_modelo[0]
    umbral_results["r2_mejor"] = round(mejor_modelo[1][2], 4)

pd.DataFrame([umbral_results]).T.rename(columns={0: "valor"}).to_csv(
    os.path.join(TABLES, "07_umbral_resultados.csv"))

# Tabla 09: comparativa regímenes por mes
diario_sh_m = diario_sh.copy()
diario_sh_m["fecha"] = pd.to_datetime(diario_sh_m["fecha"])
diario_sh_m["mes"] = diario_sh_m["fecha"].dt.to_period("M")
diario_sh_m["regimen"] = np.where(diario_sh_m["lanzados"] > tau, "alto", "bajo")

comp_mensual = (diario_sh_m.groupby(["mes","regimen"])
                .agg(n_dias=("lanzados","count"),
                     lanzados_media=("lanzados","mean"),
                     tasa_hit_media=("tasa_hit","mean"),
                     tasa_interc_media=("tasa_interc","mean"))
                .reset_index())
comp_mensual["mes_str"] = comp_mensual["mes"].astype(str)
comp_mensual.to_csv(os.path.join(TABLES, "09_comparativa_regimenes.csv"), index=False)

print("  ✓ 07_umbral_resultados.csv")
print("  ✓ 09_comparativa_regimenes.csv")

# ═════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ═════════════════════════════════════════════════════════════
figs_ok = [f for f in [out16,out17,out18,out19,out20,out21] if os.path.exists(f)]
mejor_nm = mejor_modelo[0] if mejor_modelo else 'N/A'
mejor_r2 = mejor_modelo[1][2] if mejor_modelo else 0.0

print("\n" + "=" * 60)
print("COMPLETADO — SCRIPT 04 UMBRAL SATURACIÓN")
print("=" * 60)
print(f"\nFiguras: {len(figs_ok)}/6")
for f in figs_ok:
    print(f"  ✓ {os.path.basename(f)}")

print(f"""
Hallazgos clave:
  • Umbral óptimo τ (min RSS):  {tau_opt} UAV/día
  • Umbral óptimo τ (max R²):   {tau_r2_max} UAV/día
  • Umbral piecewise ajustado:  {tau_pw:.0f} UAV/día
  • Media tasa_hit  BAJO τ:     {Y[mask_bajo].mean():.3f}  ({Y[mask_bajo].mean():.1%})
  • Media tasa_hit  ALTO τ:     {Y[mask_alto].mean():.3f}  ({Y[mask_alto].mean():.1%})
  • Test F diferencia regímenes: F={F_tar:.2f}  p={p_tar:.4f}
  • Cohen's d: {d_cohen:.3f}  ({'grande' if d_cohen>0.8 else 'medio' if d_cohen>0.5 else 'pequeño'})
  • Mejor modelo no lineal: {mejor_nm}  R²={mejor_r2:.4f}

Próximo: scripts/05_analisis_multivariante.py (PCA/MANOVA)
""")
