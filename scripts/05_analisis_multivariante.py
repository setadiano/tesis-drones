"""
05_analisis_multivariante.py
=============================
Análisis multivariante — Teatro Ucrania-Rusia 2025-2026

HIPÓTESIS:
  Los patrones de ataque (volumen, tipo, efectividad, ritmo) se estructuran
  en dimensiones latentes que capturan estrategias diferenciadas (PCA),
  y los grupos de intensidad presentan perfiles multivariantes distintos (MANOVA).

TÉCNICAS:
  1. Construcción de matriz multivariante por período (semanal/mensual)
  2. Análisis de correlación con mapa de calor
  3. PCA — componentes principales: ¿cuántas dimensiones explican el conflicto?
  4. Biplot PCA con interpretación de componentes
  5. Clustering jerárquico (Ward) sobre scores PCA
  6. MANOVA — diferencia multivariante entre grupos de intensidad
  7. Perfiles de grupo (radar chart) por cluster
  8. Análisis discriminante (LDA) como validación

FUENTES:
  data/raw/petro_attacks_2025_2026.csv       (diario, 459 obs.)
  data/raw/ur_mensual_2025_2026.csv          (mensual, 15 obs.)
  data/processed/acled_mensual_agregado.csv  (mensual, 12 obs.)

OUTPUTS:
  outputs/figures/22_heatmap_correlacion.png
  outputs/figures/23_pca_varianza_explicada.png
  outputs/figures/24_pca_biplot.png
  outputs/figures/25_clustering_jerarquico.png
  outputs/figures/26_manova_perfiles.png
  outputs/figures/27_lda_discriminante.png
  outputs/tables/10_pca_loadings.csv
  outputs/tables/11_manova_resultados.txt
  outputs/tables/12_clusters_descripcion.csv
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
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA

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
CLUSTER_COLORS = [AZUL, AMBAR, ROJO, VERDE, LILA]

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
# 1. CARGA Y CONSTRUCCIÓN DE MATRIZ MULTIVARIANTE
# ═════════════════════════════════════════════════════════════
print("=" * 60)
print("ANÁLISIS MULTIVARIANTE — DRONES 2025-2026")
print("=" * 60)
print("\n[1/8] Construyendo matriz multivariante semanal...")

# ── Petro: agregar por semana ──────────────────────────────
df_petro = pd.read_csv(os.path.join(RAW, "petro_attacks_2025_2026.csv"))
df_petro["fecha"] = pd.to_datetime(df_petro["time_start"].str[:10], errors="coerce")
df_petro = df_petro.dropna(subset=["fecha"])
for col in ["launched","destroyed","not_reach_goal","is_shahed","turbojet","turbojet_destroyed"]:
    df_petro[col] = pd.to_numeric(df_petro[col], errors="coerce").fillna(0)

# Shahed específico
sh_mask = df_petro["model"].str.contains("Shahed|Geran|Harpy", na=False, case=False)
df_sh   = df_petro[sh_mask].copy()

# Misiles (turbojet/Iskander/Kalibr/Kh-)
mis_mask = (df_petro["turbojet"] > 0) | df_petro["model"].str.contains(
    "Iskander|Kalibr|Kh-|Kinzhal|Zircon|S-300|S-400|Onyx|Kh", na=False, case=False)
df_mis = df_petro[mis_mask].copy()

df_petro["semana"] = df_petro["fecha"].dt.to_period("W")
df_sh["semana"]    = df_sh["fecha"].dt.to_period("W")
df_mis["semana"]   = df_mis["fecha"].dt.to_period("W")

# Agregar por semana
semanal_total = (df_petro.groupby("semana")
    .agg(
        lanzados_total   = ("launched",  "sum"),
        destruidos_total = ("destroyed", "sum"),
        n_ataques        = ("launched",  "count"),
    ).reset_index())

semanal_sh = (df_sh.groupby("semana")
    .agg(
        sh_lanzados  = ("launched",  "sum"),
        sh_destruidos= ("destroyed", "sum"),
    ).reset_index())

semanal_mis = (df_mis.groupby("semana")
    .agg(
        mis_lanzados  = ("launched",  "sum"),
        mis_destruidos= ("turbojet_destroyed", "sum"),
    ).reset_index())

# Merge
semanal = (semanal_total
    .merge(semanal_sh,  on="semana", how="left")
    .merge(semanal_mis, on="semana", how="left"))

semanal = semanal.fillna(0)

# Variables derivadas
semanal["tasa_interc_sh"]  = (semanal["sh_destruidos"]  / semanal["sh_lanzados"].replace(0, np.nan)).fillna(0)
semanal["tasa_hit_sh"]     = 1 - semanal["tasa_interc_sh"]
semanal["ratio_sh_total"]  = semanal["sh_lanzados"]  / semanal["lanzados_total"].replace(0, np.nan)
semanal["ratio_mis_total"] = semanal["mis_lanzados"] / semanal["lanzados_total"].replace(0, np.nan)
semanal["intensidad_diaria"]= semanal["lanzados_total"] / 7
semanal["semana_dt"]        = semanal["semana"].dt.start_time

# Variables finales para PCA (numéricas, sin NaN)
VARS_PCA = [
    "lanzados_total",      # volumen total semanal
    "sh_lanzados",         # volumen Shahed
    "mis_lanzados",        # volumen misiles
    "tasa_interc_sh",      # efectividad defensa ucraniana
    "tasa_hit_sh",         # efectividad ofensiva rusa
    "ratio_sh_total",      # proporción Shahed en mix
    "n_ataques",           # frecuencia de ataques
    "intensidad_diaria",   # intensidad promedio diaria
]
LABELS_PCA = [
    "Vol. total/sem",
    "Vol. Shahed",
    "Vol. Misiles",
    "Tasa intercep.",
    "Tasa hit Shahed",
    "Ratio Shahed/total",
    "Nº ataques/sem",
    "Intensidad diaria",
]

# Filtrar semanas con datos suficientes
df_mv = semanal[semanal["lanzados_total"] >= 10].copy()
df_mv = df_mv.dropna(subset=VARS_PCA)
df_mv = df_mv.reset_index(drop=True)

X_raw = df_mv[VARS_PCA].values
N_sem = len(df_mv)
print(f"  ✓ Matriz multivariante: {N_sem} semanas × {len(VARS_PCA)} variables")
print(f"    Período: {df_mv['semana_dt'].min().strftime('%Y-%m-%d')} → "
      f"{df_mv['semana_dt'].max().strftime('%Y-%m-%d')}")

# Estandarizar
scaler = StandardScaler()
X_std  = scaler.fit_transform(X_raw)

# ═════════════════════════════════════════════════════════════
# 2. CORRELACIONES
# ═════════════════════════════════════════════════════════════
print("\n[2/8] Mapa de correlaciones...")

corr_matrix = pd.DataFrame(X_raw, columns=LABELS_PCA).corr()

fig22, ax = plt.subplots(figsize=(11, 9), facecolor=BG)
ax.set_facecolor(BG2)

# Heatmap manual con colormap personalizado
import matplotlib.colors as mcolors
cmap = plt.cm.RdBu_r

im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors=FG2)

# Anotaciones
for i in range(len(LABELS_PCA)):
    for j in range(len(LABELS_PCA)):
        val = corr_matrix.values[i, j]
        color = "white" if abs(val) > 0.5 else FG2
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color=color, fontweight="bold" if abs(val) > 0.7 else "normal")

ax.set_xticks(range(len(LABELS_PCA)))
ax.set_yticks(range(len(LABELS_PCA)))
ax.set_xticklabels(LABELS_PCA, rotation=40, ha="right", fontsize=9, color=FG2)
ax.set_yticklabels(LABELS_PCA, fontsize=9, color=FG2)
ax.set_title("Mapa de Correlaciones — Variables Multivariantes\n"
             "Teatro Ucrania-Rusia 2025-2026  (semanal)",
             fontsize=12, color=FG, fontweight="bold", pad=15)
for sp in ax.spines.values():
    sp.set_color(GRID)

plt.tight_layout()
out22 = os.path.join(FIGS, "22_heatmap_correlacion.png")
fig22.savefig(out22, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig22)
print(f"  ✓ Figura 22 → {os.path.basename(out22)}")

# Correlaciones más altas
corr_vals = [(LABELS_PCA[i], LABELS_PCA[j], corr_matrix.values[i,j])
             for i in range(len(LABELS_PCA)) for j in range(i+1, len(LABELS_PCA))]
corr_vals.sort(key=lambda x: abs(x[2]), reverse=True)
print("  Correlaciones más altas:")
for v1, v2, r in corr_vals[:5]:
    print(f"    {v1} ↔ {v2}: r={r:.3f}")

# ═════════════════════════════════════════════════════════════
# 3. PCA
# ═════════════════════════════════════════════════════════════
print("\n[3/8] PCA — componentes principales...")

pca = PCA(n_components=min(len(VARS_PCA), N_sem - 1))
scores = pca.fit_transform(X_std)
loadings = pca.components_   # shape: (n_components, n_features)

var_exp     = pca.explained_variance_ratio_
var_cum     = np.cumsum(var_exp)
n_comp_80   = int(np.searchsorted(var_cum, 0.80)) + 1
n_comp_90   = int(np.searchsorted(var_cum, 0.90)) + 1

print(f"  Componentes para 80% varianza: {n_comp_80}")
print(f"  Componentes para 90% varianza: {n_comp_90}")
for i, (ve, vc) in enumerate(zip(var_exp[:6], var_cum[:6])):
    print(f"  PC{i+1}: {ve:.3f} ({ve:.1%})  acum={vc:.1%}")

# ─── Figura 23: varianza explicada ────────────────────────
fig23, axes23 = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig23.suptitle("PCA — Varianza Explicada por Componente\n"
               "Análisis Multivariante Drones 2025-2026",
               fontsize=13, color=FG, fontweight="bold", y=1.01)

n_show = min(8, len(var_exp))
ax = axes23[0]
bars = ax.bar(range(1, n_show+1), var_exp[:n_show]*100,
              color=AZUL, alpha=0.85, width=0.6)
ax.plot(range(1, n_show+1), var_cum[:n_show]*100,
        color=AMBAR, lw=2.5, marker="o", ms=6, label="% acumulado")
ax.axhline(80, color=VERDE, lw=1.5, ls="--", alpha=0.8, label="80%")
ax.axhline(90, color=ROJO,  lw=1.5, ls="--", alpha=0.8, label="90%")
for bar, val in zip(bars, var_exp[:n_show]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{val:.1%}", ha="center", fontsize=8, color=FG2)
ax.set_xlabel("Componente Principal", color=FG2)
ax.set_ylabel("% Varianza explicada", color=FG2)
ax.set_title("Scree plot + varianza acumulada", color=FG)
ax.legend(fontsize=9); style_ax(ax)
ax.set_xticks(range(1, n_show+1))

# Loadings de PC1 y PC2
ax2 = axes23[1]
x_ = np.arange(len(LABELS_PCA))
w = 0.35
bars1 = ax2.bar(x_ - w/2, loadings[0], w, color=AZUL,  alpha=0.85, label="PC1")
bars2 = ax2.bar(x_ + w/2, loadings[1], w, color=VERDE, alpha=0.85, label="PC2")
ax2.axhline(0, color=FG2, lw=0.8)
ax2.set_xticks(x_)
ax2.set_xticklabels(LABELS_PCA, rotation=40, ha="right", fontsize=8, color=FG2)
ax2.set_ylabel("Loading (peso)", color=FG2)
ax2.set_title(f"Loadings PC1 ({var_exp[0]:.1%}) y PC2 ({var_exp[1]:.1%})", color=FG)
ax2.legend(fontsize=9); style_ax(ax2)

plt.tight_layout()
out23 = os.path.join(FIGS, "23_pca_varianza_explicada.png")
fig23.savefig(out23, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig23)
print(f"  ✓ Figura 23 → {os.path.basename(out23)}")

# ═════════════════════════════════════════════════════════════
# 4. BIPLOT PCA (PC1 × PC2)
# ═════════════════════════════════════════════════════════════
print("\n[4/8] Biplot PCA...")

# Colorear por período (cuatrimestres)
dates_mv = pd.to_datetime(df_mv["semana_dt"])
periodo_labels = []
periodo_colors = []
for d in dates_mv:
    if d < pd.Timestamp("2025-04-01"):
        periodo_labels.append("Q1 2025")
        periodo_colors.append(AZUL)
    elif d < pd.Timestamp("2025-07-01"):
        periodo_labels.append("Q2 2025")
        periodo_colors.append(VERDE)
    elif d < pd.Timestamp("2025-10-01"):
        periodo_labels.append("Q3 2025")
        periodo_colors.append(AMBAR)
    elif d < pd.Timestamp("2026-01-01"):
        periodo_labels.append("Q4 2025")
        periodo_colors.append(ROJO)
    else:
        periodo_labels.append("2026")
        periodo_colors.append(LILA)

fig24, ax = plt.subplots(figsize=(13, 10), facecolor=BG)
ax.set_facecolor(BG2)

# Scatter de observaciones
for label, color in zip(["Q1 2025","Q2 2025","Q3 2025","Q4 2025","2026"],
                         [AZUL, VERDE, AMBAR, ROJO, LILA]):
    mask = [p == label for p in periodo_labels]
    if any(mask):
        idx = [i for i, m in enumerate(mask) if m]
        ax.scatter(scores[idx, 0], scores[idx, 1],
                   c=color, s=55, alpha=0.75, zorder=3, label=label)

# Flechas de loadings (escaladas)
scale = max(abs(scores[:, :2].max()), abs(scores[:, :2].min())) * 0.45
for i, (label, lx, ly) in enumerate(zip(LABELS_PCA, loadings[0], loadings[1])):
    ax.annotate("", xy=(lx*scale, ly*scale), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=AMBAR, lw=1.8, mutation_scale=12))
    offset_x = 0.15 * np.sign(lx) if abs(lx) > 0.05 else 0
    offset_y = 0.15 * np.sign(ly) if abs(ly) > 0.05 else 0
    ax.text(lx*scale + offset_x, ly*scale + offset_y, label,
            color=AMBAR, fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, alpha=0.7, edgecolor="none"))

ax.axhline(0, color=GRID, lw=0.8, ls="--")
ax.axvline(0, color=GRID, lw=0.8, ls="--")
ax.set_xlabel(f"PC1 — {var_exp[0]:.1%} varianza explicada", color=FG2, fontsize=11)
ax.set_ylabel(f"PC2 — {var_exp[1]:.1%} varianza explicada", color=FG2, fontsize=11)
ax.set_title(f"Biplot PCA — Teatro Ucrania-Rusia 2025-2026\n"
             f"PC1+PC2 explican {var_cum[1]:.1%} de la varianza total",
             fontsize=13, color=FG, fontweight="bold")
ax.legend(title="Período", fontsize=9, title_fontsize=9)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.grid(True, alpha=0.2)

plt.tight_layout()
out24 = os.path.join(FIGS, "24_pca_biplot.png")
fig24.savefig(out24, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig24)
print(f"  ✓ Figura 24 → {os.path.basename(out24)}")

# ═════════════════════════════════════════════════════════════
# 5. CLUSTERING JERÁRQUICO (sobre scores PCA)
# ═════════════════════════════════════════════════════════════
print("\n[5/8] Clustering jerárquico...")

# Usar los primeros n_comp_80 componentes para clustering
n_clust_comp = min(n_comp_80, scores.shape[1])
X_clust = scores[:, :n_clust_comp]

# Dendrograma y cut en k=3 clusters
Z = linkage(X_clust, method="ward")
k_clusters = 3
cluster_labels = fcluster(Z, k_clusters, criterion="maxclust")

# Silhouette para validar
sil_scores = {}
for k in range(2, min(6, N_sem)):
    lbl = fcluster(Z, k, criterion="maxclust")
    if len(np.unique(lbl)) > 1:
        sil_scores[k] = silhouette_score(X_clust, lbl)

best_k = max(sil_scores, key=sil_scores.get) if sil_scores else k_clusters
print(f"  Silhouette scores: " + "  ".join(f"k={k}: {s:.3f}" for k, s in sil_scores.items()))
print(f"  k óptimo (silhouette): {best_k}")

# Re-cortar con k óptimo
cluster_labels = fcluster(Z, best_k, criterion="maxclust")
cluster_sizes  = {k: (cluster_labels == k).sum() for k in np.unique(cluster_labels)}
print(f"  Distribución clusters: {cluster_sizes}")

# ─── Figura 25: dendrograma + scatter ─────────────────────
fig25, axes25 = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig25.suptitle(f"Clustering Jerárquico (Ward, k={best_k})\n"
               "Semanas de ataque agrupadas por perfil multivariante",
               fontsize=13, color=FG, fontweight="bold", y=1.01)

# Dendrograma
ax = axes25[0]
ax.set_facecolor(BG2)
dend = dendrogram(Z, ax=ax, color_threshold=Z[-best_k+1, 2] * 1.01,
                  above_threshold_color=FG2, no_labels=True,
                  link_color_func=lambda k: PALETTE[k % len(PALETTE)])
ax.axhline(Z[-best_k+1, 2] * 1.01, color=AMBAR, lw=2, ls="--",
           label=f"Corte k={best_k}")
ax.set_xlabel("Semanas", color=FG2)
ax.set_ylabel("Distancia (Ward)", color=FG2)
ax.set_title("Dendrograma jerárquico", color=FG)
ax.legend(fontsize=9)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.grid(True, alpha=0.2, axis="y")

# Scatter PC1 × PC2 coloreado por cluster
ax2 = axes25[1]
ax2.set_facecolor(BG2)
for c in np.unique(cluster_labels):
    mask = cluster_labels == c
    mean_lanz = df_mv.loc[mask, "lanzados_total"].mean()
    ax2.scatter(scores[mask, 0], scores[mask, 1],
                c=CLUSTER_COLORS[(c-1) % len(CLUSTER_COLORS)],
                s=65, alpha=0.8, zorder=3,
                label=f"Cluster {c}  (n={mask.sum()}, μ={mean_lanz:.0f} UAV/sem)")
    # Centroide
    cx, cy = scores[mask, 0].mean(), scores[mask, 1].mean()
    ax2.scatter(cx, cy, c=CLUSTER_COLORS[(c-1) % len(CLUSTER_COLORS)],
                s=200, marker="*", zorder=5, edgecolors="white", lw=0.8)

ax2.axhline(0, color=GRID, lw=0.8, ls="--")
ax2.axvline(0, color=GRID, lw=0.8, ls="--")
ax2.set_xlabel(f"PC1 ({var_exp[0]:.1%})", color=FG2)
ax2.set_ylabel(f"PC2 ({var_exp[1]:.1%})", color=FG2)
ax2.set_title(f"Clusters en espacio PCA\n(★ = centroide)", color=FG)
ax2.legend(fontsize=8, loc="best")
for sp in ax2.spines.values():
    sp.set_color(GRID)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
out25 = os.path.join(FIGS, "25_clustering_jerarquico.png")
fig25.savefig(out25, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig25)
print(f"  ✓ Figura 25 → {os.path.basename(out25)}")

# ═════════════════════════════════════════════════════════════
# 6. MANOVA
# ═════════════════════════════════════════════════════════════
print("\n[6/8] MANOVA — diferencia multivariante entre clusters...")

df_mv["cluster"] = cluster_labels

manova_lines = ["MANOVA — Diferencia multivariante entre clusters", "=" * 50, ""]

# Variables dependientes para MANOVA (seleccionar las más informativas y no colineales)
VARS_MANOVA = ["lanzados_total", "tasa_interc_sh", "ratio_sh_total",
               "n_ataques", "mis_lanzados"]
LABELS_MANOVA = ["Vol. total", "Tasa intercep.", "Ratio Shahed", "Nº ataques", "Vol. misiles"]

df_manova = df_mv[VARS_MANOVA + ["cluster"]].dropna()

# Verificar varianza no nula
vars_ok = [v for v in VARS_MANOVA if df_manova[v].std() > 0]
if len(vars_ok) < 2:
    manova_lines.append("MANOVA: insuficientes variables con varianza > 0")
    manova_ok = False
else:
    try:
        formula = " + ".join(vars_ok) + " ~ C(cluster)"
        mv_result = MANOVA.from_formula(formula, data=df_manova)
        mv_test   = mv_result.mv_test()
        manova_lines.append(f"Variables dependientes: {', '.join(vars_ok)}")
        manova_lines.append(f"Factor: cluster (k={best_k})")
        manova_lines.append("")
        # Pillai's trace (el más robusto)
        manova_str = str(mv_test)
        manova_lines.append(manova_str[:3000])  # truncar si es muy largo
        manova_ok = True
        print("  ✓ MANOVA ejecutado")
    except Exception as e:
        manova_lines.append(f"MANOVA error: {e}")
        manova_ok = False
        print(f"  ⚠  MANOVA: {e}")

# Tests univariantes por variable
manova_lines += ["", "── Tests univariantes (ANOVA por variable) ──"]
anova_results = []
for var, label in zip(VARS_MANOVA, LABELS_MANOVA):
    grupos = [df_mv.loc[df_mv["cluster"]==c, var].dropna().values
              for c in np.unique(cluster_labels)]
    grupos = [g for g in grupos if len(g) >= 2]
    if len(grupos) >= 2:
        F, p = stats.f_oneway(*grupos)
        ss_within = sum(((g - g.mean())**2).sum() for g in grupos)
        all_vals   = df_mv[var].dropna().values
        ss_total   = ((all_vals - all_vals.mean())**2).sum()
        eta2 = 1 - ss_within / ss_total if ss_total > 0 else 0
        sig = "★★★" if p < 0.001 else "★★" if p < 0.01 else "★" if p < 0.05 else "ns"
        anova_results.append({"variable": label, "F": round(F,3), "p": round(p,5),
                               "eta2": round(eta2,4), "sig": sig})
        manova_lines.append(f"  {label:<20} F={F:.2f}  p={p:.4f}  η²={eta2:.3f}  {sig}")
        print(f"  {label:<20} F={F:.2f}  p={p:.4f}  {sig}")

with open(os.path.join(TABLES, "11_manova_resultados.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(manova_lines))
print("  ✓ 11_manova_resultados.txt")

# ═════════════════════════════════════════════════════════════
# 7. PERFILES DE GRUPO — RADAR CHART
# ═════════════════════════════════════════════════════════════
print("\n[7/8] Radar chart — perfiles de cluster...")

# Normalizar a 0-1 para radar
vars_radar = VARS_MANOVA
labels_radar = LABELS_MANOVA

cluster_profiles = {}
for c in np.unique(cluster_labels):
    mask = df_mv["cluster"] == c
    profile = df_mv.loc[mask, vars_radar].mean()
    cluster_profiles[c] = profile

# Normalizar 0-1
all_means = pd.DataFrame(cluster_profiles).T
col_min = all_means.min()
col_max = all_means.max()
col_range = (col_max - col_min).replace(0, 1)
all_norm = (all_means - col_min) / col_range

N_var = len(vars_radar)
angles = np.linspace(0, 2 * np.pi, N_var, endpoint=False).tolist()
angles += angles[:1]  # cerrar

fig26, axes26 = plt.subplots(1, best_k, figsize=(5*best_k, 5),
                              facecolor=BG,
                              subplot_kw=dict(polar=True))
if best_k == 1:
    axes26 = [axes26]

fig26.suptitle(f"Perfiles de Cluster — Radar Chart  (k={best_k})\n"
               "Caracterización multivariante de estrategias de ataque",
               fontsize=12, color=FG, fontweight="bold", y=1.05)

for idx, (c, ax) in enumerate(zip(np.unique(cluster_labels), axes26)):
    ax.set_facecolor(BG2)
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID, alpha=0.4)

    vals = all_norm.loc[c].values.tolist()
    vals += vals[:1]
    col = CLUSTER_COLORS[(c-1) % len(CLUSTER_COLORS)]

    ax.fill(angles, vals, color=col, alpha=0.25)
    ax.plot(angles, vals, color=col, lw=2.5, marker="o", ms=5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, size=8, color=FG2)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], size=6, color=FG2)
    ax.tick_params(colors=FG2)

    # Descripción del cluster
    n_c = (cluster_labels == c).sum()
    mean_lanz = all_means.loc[c, "lanzados_total"]
    mean_interc = all_means.loc[c, "tasa_interc_sh"]
    descr = (f"Cluster {c}  (n={n_c})\n"
             f"Vol: {mean_lanz:.0f} UAV/sem\n"
             f"Intercep: {mean_interc:.2%}")
    ax.set_title(descr, color=col, fontsize=9, fontweight="bold", pad=18)

plt.tight_layout()
out26 = os.path.join(FIGS, "26_manova_perfiles.png")
fig26.savefig(out26, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig26)
print(f"  ✓ Figura 26 → {os.path.basename(out26)}")

# ═════════════════════════════════════════════════════════════
# 8. LDA DISCRIMINANTE + TABLA PCA LOADINGS
# ═════════════════════════════════════════════════════════════
print("\n[8/8] LDA discriminante + tablas...")

# LDA
lda = LinearDiscriminantAnalysis(n_components=min(best_k - 1, 2))
lda_scores = lda.fit_transform(X_std, cluster_labels)

# Accuracy del clasificador LDA
from sklearn.model_selection import cross_val_score
lda_cv = LinearDiscriminantAnalysis()
cv_scores = cross_val_score(lda_cv, X_std, cluster_labels,
                            cv=min(5, N_sem // best_k), scoring="accuracy")
print(f"  LDA cross-val accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ─── Figura 27: LDA + evolución temporal clusters ─────────
fig27, axes27 = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig27.suptitle("LDA — Análisis Discriminante Lineal\n"
               "Validación de clusters y evolución temporal",
               fontsize=13, color=FG, fontweight="bold", y=1.01)

ax = axes27[0]
ax.set_facecolor(BG2)

if lda_scores.shape[1] >= 2:
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        ax.scatter(lda_scores[mask, 0], lda_scores[mask, 1],
                   c=CLUSTER_COLORS[(c-1) % len(CLUSTER_COLORS)],
                   s=60, alpha=0.8, zorder=3, label=f"Cluster {c}")
    ax.set_xlabel("LD1", color=FG2)
    ax.set_ylabel("LD2", color=FG2)
else:
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        ax.hist(lda_scores[mask, 0], bins=12, alpha=0.6,
                color=CLUSTER_COLORS[(c-1) % len(CLUSTER_COLORS)],
                label=f"Cluster {c}")
    ax.set_xlabel("LD1", color=FG2)
    ax.set_ylabel("Frecuencia", color=FG2)

ax.set_title(f"Espacio discriminante LDA\n"
             f"Accuracy CV = {cv_scores.mean():.1%} ± {cv_scores.std():.1%}",
             color=FG)
ax.legend(fontsize=9)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.grid(True, alpha=0.2)

# Panel derecho: evolución temporal de clusters
ax2 = axes27[1]
ax2.set_facecolor(BG2)

for c in np.unique(cluster_labels):
    mask_c = cluster_labels == c
    dates_c = df_mv.loc[mask_c, "semana_dt"]
    lanz_c  = df_mv.loc[mask_c, "lanzados_total"]
    ax2.scatter(dates_c, lanz_c,
                c=CLUSTER_COLORS[(c-1) % len(CLUSTER_COLORS)],
                s=50, alpha=0.75, zorder=3, label=f"Cluster {c}")

# Media móvil general
roll = df_mv.set_index("semana_dt")["lanzados_total"].rolling(4, center=True, min_periods=2).mean()
ax2.plot(roll.index, roll.values, color=FG2, lw=2, ls="--", alpha=0.6,
         label="Media móvil 4sem", zorder=2)

ax2.set_xlabel("Fecha", color=FG2)
ax2.set_ylabel("UAV lanzados / semana", color=FG2)
ax2.set_title("Evolución temporal de clusters\n(color = cluster asignado)",
              color=FG)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax2.legend(fontsize=8, loc="upper left")
for sp in ax2.spines.values():
    sp.set_color(GRID)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
out27 = os.path.join(FIGS, "27_lda_discriminante.png")
fig27.savefig(out27, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig27)
print(f"  ✓ Figura 27 → {os.path.basename(out27)}")

# ── Tabla 10: PCA loadings ─────────────────────────────────
n_comp_tabla = min(5, loadings.shape[0])
df_loadings = pd.DataFrame(
    loadings[:n_comp_tabla].T,
    index=LABELS_PCA,
    columns=[f"PC{i+1}" for i in range(n_comp_tabla)]
)
df_loadings["comunalidad"] = (loadings[:n_comp_tabla]**2).sum(axis=0).mean()
# comunalidad real: suma de cargas² por variable
df_loadings["comunalidad"] = [(loadings[:n_comp_tabla, i]**2).sum()
                               for i in range(len(LABELS_PCA))]
df_loadings.index.name = "variable"

# Añadir fila de varianza explicada
row_ve = pd.DataFrame([[f"{v:.4f}" for v in var_exp[:n_comp_tabla]] + [""]],
                       index=["% var. explicada"],
                       columns=df_loadings.columns)
df_loadings_str = df_loadings.round(4).astype(str)
df_loadings_full = pd.concat([df_loadings_str, row_ve])
df_loadings_full.to_csv(os.path.join(TABLES, "10_pca_loadings.csv"))
print("  ✓ 10_pca_loadings.csv")

# ── Tabla 12: descripción de clusters ─────────────────────
cluster_desc = []
desc_vars = VARS_PCA + ["cluster"]
for c in np.unique(cluster_labels):
    mask_c = df_mv["cluster"] == c
    row = {"cluster": int(c), "n_semanas": int(mask_c.sum())}
    for v in VARS_PCA:
        row[f"{v}_media"] = round(float(df_mv.loc[mask_c, v].mean()), 3)
        row[f"{v}_std"]   = round(float(df_mv.loc[mask_c, v].std()),  3)
    # Fechas del cluster
    row["fecha_inicio"] = df_mv.loc[mask_c, "semana_dt"].min().strftime("%Y-%m-%d")
    row["fecha_fin"]    = df_mv.loc[mask_c, "semana_dt"].max().strftime("%Y-%m-%d")
    cluster_desc.append(row)

pd.DataFrame(cluster_desc).to_csv(
    os.path.join(TABLES, "12_clusters_descripcion.csv"), index=False)
print("  ✓ 12_clusters_descripcion.csv")

# ═════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ═════════════════════════════════════════════════════════════
figs_ok = [f for f in [out22,out23,out24,out25,out26,out27] if os.path.exists(f)]

# Construir descripción de clusters
cluster_desc_str = ""
for c in np.unique(cluster_labels):
    mask_c = df_mv["cluster"] == c
    n_c    = mask_c.sum()
    lanz_m = df_mv.loc[mask_c, "lanzados_total"].mean()
    int_m  = df_mv.loc[mask_c, "tasa_interc_sh"].mean()
    rsh_m  = df_mv.loc[mask_c, "ratio_sh_total"].mean()
    cluster_desc_str += f"  Cluster {c} (n={n_c}): {lanz_m:.0f} UAV/sem | interc={int_m:.1%} | Shahed={rsh_m:.1%}\n"

print("\n" + "=" * 60)
print("COMPLETADO — SCRIPT 05 ANÁLISIS MULTIVARIANTE")
print("=" * 60)
print(f"\nFiguras: {len(figs_ok)}/6")
for f in figs_ok:
    print(f"  ✓ {os.path.basename(f)}")

print(f"""
Hallazgos clave:
  • Semanas analizadas:          {N_sem}
  • Variables multivariantes:    {len(VARS_PCA)}
  • PC1 varianza explicada:      {var_exp[0]:.1%}
  • PC1+PC2 varianza explicada:  {var_cum[1]:.1%}
  • Componentes para 80% var:    {n_comp_80}
  • Clusters óptimos (silhouette): k={best_k}
  • LDA accuracy (CV):           {cv_scores.mean():.1%} ± {cv_scores.std():.1%}

Perfiles de cluster:
{cluster_desc_str}
ANOVA univariante por variable:""")
for r in anova_results:
    print(f"  {r['variable']:<22} F={r['F']:.2f}  p={r['p']:.4f}  η²={r['eta2']:.3f}  {r['sig']}")

print("""
Próximo: INFORME_MAESTRO (actualización) o entrega final.
""")
