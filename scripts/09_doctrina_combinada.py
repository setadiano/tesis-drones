"""
Script 09 — Doctrina Combinada: Drones + Combates + Respuesta UA
================================================================
Autor : setadiano / jplatas6@alumno.uned.es
Fecha : Abril 2026

Análisis A — Índice de presión táctica combinada por oblast
         B — Lead/lag: ¿los drones preceden a los combates terrestres?
         C — Doctrina UA: patrón de respuesta a olas masivas rusas
         D — Geoespacial: heat map de impactos + infraestructura energética

NOTA: ACLED cubre Ene–Abr 2025 (95 días de solapamiento con Petro).
      El análisis se centra en ese período pero los patrones son extrapolables.

Inputs
------
  data/raw/petro_attacks_2025_2026.csv
  data/raw/acled_ukraine_2025_2026.csv

Outputs
-------
  data/processed/presion_tactica_oblast.csv
  data/processed/leadlag_drones_combates.csv
  outputs/09_*.png  (8 figuras)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
from scipy import stats
from scipy.stats import spearmanr, pearsonr, ttest_ind, mannwhitneyu
from statsmodels.tsa.stattools import grangercausalitytests, ccf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, Rectangle
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
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GRID_C, "axes.labelcolor": FG,
    "axes.titlecolor": FG, "xtick.color": FG,
    "ytick.color": FG, "text.color": FG,
    "grid.color": GRID_C, "grid.alpha": 0.5,
    "legend.facecolor": "#161b22", "legend.edgecolor": GRID_C,
    "font.family": "monospace", "figure.dpi": 120,
})

FIGS_OK = 0
def save_fig(name):
    global FIGS_OK
    plt.savefig(OUT / name, bbox_inches="tight", facecolor=BG, dpi=150)
    plt.close()
    FIGS_OK += 1
    print(f"  [✓] {name}")

# ══════════════════════════════════════════════════════════════════════════
# 0.  CARGA
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SCRIPT 09 — DOCTRINA COMBINADA")
print("="*60)

print("\n[0] Cargando datos …")

# Petro — Shahed
petro_raw = pd.read_csv(RAW / "petro_attacks_2025_2026.csv")
petro = petro_raw[petro_raw["model"].str.contains("Shahed", na=False)].copy()
petro["fecha"] = pd.to_datetime(petro["fecha"], errors="coerce")
petro = petro.dropna(subset=["fecha","launched","destroyed"])
petro = petro[petro["launched"] > 0].copy()
petro["impactos"]   = petro["launched"] - petro["destroyed"]
petro["tasa_hit"]   = petro["impactos"] / petro["launched"]
petro["tasa_intercep"] = petro["destroyed"] / petro["launched"]

# ACLED completo
acled = pd.read_csv(RAW / "acled_ukraine_2025_2026.csv", low_memory=False)
acled["event_date"] = pd.to_datetime(acled["event_date"])

# Período solapado
t0 = max(petro["fecha"].min(), acled["event_date"].min())
t1 = min(petro["fecha"].max(), acled["event_date"].max())
print(f"  Período solapado: {t0.date()} → {t1.date()} ({(t1-t0).days} días)")

acled_ov = acled[(acled["event_date"] >= t0) & (acled["event_date"] <= t1)].copy()
petro_ov = petro[(petro["fecha"] >= t0) & (petro["fecha"] <= t1)].copy()

# Subsets ACLED
ru_air    = acled_ov[acled_ov["actor1"].str.contains("Russia.*Air", na=False, regex=True)]
ua_air    = acled_ov[acled_ov["actor1"].str.contains("Ukraine.*Air", na=False, regex=True)]
battles   = acled_ov[acled_ov["event_type"] == "Battles"]
disrupted = acled_ov[acled_ov["sub_event_type"] == "Disrupted weapons use"]
energy    = acled_ov[acled_ov["notes"].str.contains(
    "power|energy|electric|substation|thermal|TPP|CHP|Shahed",
    case=False, na=False)]

print(f"  RU airstrikes: {len(ru_air):,}  |  UA airstrikes: {len(ua_air):,}")
print(f"  Battles: {len(battles):,}  |  Disrupted: {len(disrupted):,}")
print(f"  Eventos energéticos: {len(energy):,}")

# ── Series diarias base ────────────────────────────────────────────────────
def daily_counts(df, date_col, name):
    s = df.groupby(date_col).size().reset_index(name=name)
    s.columns = ["fecha", name]
    return s

shahed_d  = (petro_ov.groupby("fecha")
             .agg(sh_lanzados=("launched","sum"),
                  sh_impactos=("impactos","sum"),
                  sh_intercep=("tasa_intercep","mean"))
             .reset_index())
battles_d  = daily_counts(battles,  "event_date", "n_battles")
ru_air_d   = daily_counts(ru_air,   "event_date", "n_ru_air")
ua_air_d   = daily_counts(ua_air,   "event_date", "n_ua_air")
disrupted_d= daily_counts(disrupted,"event_date", "n_disrupted")
energy_d   = daily_counts(energy,   "event_date", "n_energy")

# Merge completo en índice de fechas diarias
dates_range = pd.DataFrame({"fecha": pd.date_range(t0, t1, freq="D")})
daily = reduce(lambda l,r: pd.merge(l,r, on="fecha", how="left"),
               [dates_range, shahed_d, battles_d, ru_air_d,
                ua_air_d, disrupted_d, energy_d])
daily = daily.fillna(0)

# ══════════════════════════════════════════════════════════════════════════
# A.  ÍNDICE DE PRESIÓN TÁCTICA COMBINADA POR OBLAST
# ══════════════════════════════════════════════════════════════════════════
print("\n[A] Índice de presión táctica por oblast …")

# Por oblast: suma ponderada de airstrikes RU + artillería + battles + energía
OBLASTS_UA = ["Donetsk","Kharkiv","Zaporizhia","Kherson","Sumy",
              "Chernihiv","Dnipropetrovsk","Mykolaiv","Odesa",
              "Kyiv City","Kyiv","Luhansk","Poltava","Khmelnytskyi",
              "Kirovohrad","Vinnytsia","Cherkasy"]

def presion_oblast(oblast_df, nombre):
    air    = len(oblast_df[oblast_df["sub_event_type"]=="Air/drone strike"])
    shell  = len(oblast_df[oblast_df["sub_event_type"]
                           .str.contains("Shelling|artillery", na=False, case=False)])
    bat    = len(oblast_df[oblast_df["event_type"]=="Battles"])
    en     = len(oblast_df[oblast_df["notes"].str.contains(
                 "power|energy|electric|TPP|CHP", case=False, na=False)])
    civil  = len(oblast_df[oblast_df["civilian_targeting"]=="Civilian targeting"])
    total  = air + shell + bat
    # Índice compuesto: pesos doctrinales
    idx    = 0.35*air + 0.25*shell + 0.25*bat + 0.10*en + 0.05*civil
    return {"oblast": nombre, "airstrikes": air, "artilleria": shell,
            "battles": bat, "energia": en, "civil": civil,
            "total_eventos": total, "indice_presion": idx}

rows_p = []
for obl in OBLASTS_UA:
    sub = acled_ov[acled_ov["admin1"] == obl]
    if len(sub) > 0:
        rows_p.append(presion_oblast(sub, obl))

presion_df = pd.DataFrame(rows_p).sort_values("indice_presion", ascending=False)
presion_df.to_csv(PROC / "presion_tactica_oblast.csv", index=False)
print(f"  Top 5 oblasts por presión táctica:")
print(presion_df[["oblast","airstrikes","battles","energia","indice_presion"]].head(5).to_string(index=False))

# ── Fig 1: Ranking presión táctica ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("A — Índice de Presión Táctica Combinada por Oblast (Ene–Abr 2025)",
             color=FG, fontsize=12, y=1.01)

ax1, ax2 = axes

# Barras apiladas
obl_top = presion_df.head(12)
x = range(len(obl_top))
b1 = ax1.bar(x, obl_top["airstrikes"], label="Airstrikes RU",
             color=RED, alpha=0.85, edgecolor=BG, lw=0.3)
b2 = ax1.bar(x, obl_top["artilleria"], bottom=obl_top["airstrikes"],
             label="Artillería/misiles", color=ORANGE, alpha=0.85, edgecolor=BG, lw=0.3)
b3 = ax1.bar(x, obl_top["battles"],
             bottom=obl_top["airstrikes"]+obl_top["artilleria"],
             label="Combates terrestres", color=YELLOW, alpha=0.85, edgecolor=BG, lw=0.3)
b4 = ax1.bar(x, obl_top["energia"] * 5,
             bottom=obl_top["airstrikes"]+obl_top["artilleria"]+obl_top["battles"],
             label="Infraestructura energética (×5)", color=PURPLE, alpha=0.7,
             edgecolor=BG, lw=0.3)

ax1.set_xticks(list(x))
ax1.set_xticklabels(obl_top["oblast"], rotation=45, ha="right", fontsize=9)
ax1.set_ylabel("Eventos acumulados", color=FG)
ax1.set_title("Composición por dimensión táctica", color=FG, fontsize=10)
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_facecolor(BG)

# Índice normalizado
idx_norm = obl_top["indice_presion"] / obl_top["indice_presion"].max() * 100
colors_p = [RED if v > 70 else (ORANGE if v > 40 else ACCENT) for v in idx_norm]
bars = ax2.barh(obl_top["oblast"][::-1], idx_norm[::-1],
                color=list(reversed(colors_p)), alpha=0.85, edgecolor=BG, lw=0.3)
ax2.axvline(50, color=YELLOW, lw=1, ls="--", alpha=0.7, label="50%")
for bar, val in zip(bars, idx_norm[::-1]):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f"{val:.0f}", va="center", fontsize=9, color=FG)
ax2.set_xlabel("Índice de presión normalizado (0-100)", color=FG)
ax2.set_title("Ranking presión táctica combinada", color=FG, fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="x")
ax2.set_facecolor(BG)

plt.tight_layout()
save_fig("09_fig1_presion_tactica_oblast.png")

# ══════════════════════════════════════════════════════════════════════════
# B.  LEAD/LAG: ¿LOS DRONES PRECEDEN A LOS COMBATES?
# ══════════════════════════════════════════════════════════════════════════
print("\n[B] Lead/lag drones → combates terrestres …")

# CCF (cross-correlation function): lanzamientos Shahed vs battles
# Positivo lag k = lanzamientos preceden a battles en k días
x_sh  = daily["sh_lanzados"].values
x_bat = daily["n_battles"].values
x_rua = daily["n_ru_air"].values
x_uaa = daily["n_ua_air"].values

# Normalizar
def norm(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()

x_sh_n  = norm(x_sh)
x_bat_n = norm(x_bat)
x_rua_n = norm(x_rua)
x_uaa_n = norm(x_uaa)

# CCF manual (pearson con lags -14 a +14)
lags = range(-14, 15)
ccf_sh_bat  = [pearsonr(x_sh_n[max(0,-k):len(x_sh_n)-max(0,k)],
                        x_bat_n[max(0,k):len(x_bat_n)-max(0,-k)])[0]
               for k in lags]
ccf_sh_rua  = [pearsonr(x_sh_n[max(0,-k):len(x_sh_n)-max(0,k)],
                        x_rua_n[max(0,k):len(x_rua_n)-max(0,-k)])[0]
               for k in lags]
ccf_rua_uaa = [pearsonr(x_rua_n[max(0,-k):len(x_rua_n)-max(0,k)],
                        x_uaa_n[max(0,k):len(x_uaa_n)-max(0,-k)])[0]
               for k in lags]
ccf_sh_uaa  = [pearsonr(x_sh_n[max(0,-k):len(x_sh_n)-max(0,k)],
                        x_uaa_n[max(0,k):len(x_uaa_n)-max(0,-k)])[0]
               for k in lags]

# Banda de significación (aprox. 2/sqrt(N))
sig_band = 2 / np.sqrt(len(x_sh))

# Lags significativos
lags_list = list(lags)
sig_sh_bat  = [(lags_list[i], ccf_sh_bat[i])
               for i in range(len(lags_list)) if abs(ccf_sh_bat[i])  > sig_band]
sig_sh_uaa  = [(lags_list[i], ccf_sh_uaa[i])
               for i in range(len(lags_list)) if abs(ccf_sh_uaa[i])  > sig_band]
sig_rua_uaa = [(lags_list[i], ccf_rua_uaa[i])
               for i in range(len(lags_list)) if abs(ccf_rua_uaa[i]) > sig_band]

print(f"  Banda significación: ±{sig_band:.3f}")
print(f"  Shahed→Battles lags significativos: {sig_sh_bat}")
print(f"  Shahed→UA_air  lags significativos: {sig_sh_uaa}")
print(f"  RU_air→UA_air  lags significativos: {sig_rua_uaa}")

# Granger: sh_lanzados → n_battles (maxlag=7)
print("\n  Granger Shahed → Battles (maxlag=7):")
try:
    df_gr = pd.DataFrame({"battles": x_bat, "shahed": x_sh})
    gr_res = grangercausalitytests(df_gr[["battles","shahed"]], maxlag=7, verbose=False)
    for lag, res in gr_res.items():
        p = res[0]["ssr_ftest"][1]
        print(f"    lag={lag}  p={p:.3f}  {'★' if p<0.05 else ''}")
except Exception as e:
    print(f"    Granger falló: {e}")

# Granger: sh_lanzados → n_ua_air
print("\n  Granger Shahed → UA_air (maxlag=7):")
try:
    df_gr2 = pd.DataFrame({"ua_air": x_uaa, "shahed": x_sh})
    gr_res2 = grangercausalitytests(df_gr2[["ua_air","shahed"]], maxlag=7, verbose=False)
    for lag, res in gr_res2.items():
        p = res[0]["ssr_ftest"][1]
        print(f"    lag={lag}  p={p:.3f}  {'★' if p<0.05 else ''}")
except Exception as e:
    print(f"    Granger falló: {e}")

# Lead/lag por oblast — ¿en qué oblasts preceden los drones a los combates?
print("\n  Lead/lag por oblast (top 5):")
oblasts_batalla = ["Donetsk","Kharkiv","Zaporizhia","Kherson","Kursk"]
leadlag_rows = []
for obl in oblasts_batalla:
    ru_obl = acled_ov[
        (acled_ov["admin1"]==obl) &
        (acled_ov["actor1"].str.contains("Russia",na=False)) &
        (acled_ov["sub_event_type"]=="Air/drone strike")
    ]
    bat_obl = acled_ov[
        (acled_ov["admin1"]==obl) &
        (acled_ov["event_type"]=="Battles")
    ]
    if len(ru_obl) < 5 or len(bat_obl) < 5:
        continue
    air_d = daily_counts(ru_obl, "event_date", "air")
    bat_d = daily_counts(bat_obl, "event_date", "bat")
    merged_obl = dates_range.merge(air_d, on="fecha", how="left")\
                            .merge(bat_d, on="fecha", how="left").fillna(0)
    xa = norm(merged_obl["air"].values)
    xb = norm(merged_obl["bat"].values)
    best_r, best_lag = 0, 0
    for k in range(-7, 8):
        if k >= 0:
            r, _ = pearsonr(xa[:len(xa)-k] if k>0 else xa,
                            xb[k:] if k>0 else xb)
        else:
            r, _ = pearsonr(xa[-k:], xb[:len(xb)+k])
        if abs(r) > abs(best_r):
            best_r, best_lag = r, k
    leadlag_rows.append({"oblast": obl, "best_lag": best_lag,
                         "best_r": best_r, "n_air": len(ru_obl), "n_bat": len(bat_obl)})
    print(f"    {obl:15s}  lag={best_lag:+3d}d  r={best_r:.3f}  "
          f"(n_air={len(ru_obl)}, n_bat={len(bat_obl)})")

leadlag_df = pd.DataFrame(leadlag_rows)
leadlag_df.to_csv(PROC / "leadlag_drones_combates.csv", index=False)

# ── Fig 2: CCF Shahed vs Battles ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("B — Lead/Lag: ¿Los Drones Preceden a los Combates?",
             color=FG, fontsize=12, y=1.01)

pairs = [
    (ccf_sh_bat,  "Shahed lanzados → Battles terrestres",    RED,    ACCENT),
    (ccf_sh_rua,  "Shahed lanzados → RU Airstrikes (ACLED)", ORANGE, ACCENT),
    (ccf_sh_uaa,  "Shahed lanzados → UA Airstrikes",         PURPLE, GREEN),
    (ccf_rua_uaa, "RU Airstrikes → UA Airstrikes",           RED,    GREEN),
]

for ax, (ccf_vals, title, color_neg, color_pos) in zip(axes.flat, pairs):
    lags_arr = np.array(lags_list)
    ccf_arr  = np.array(ccf_vals)
    colors_c = [color_pos if v >= 0 else color_neg for v in ccf_arr]
    ax.bar(lags_arr, ccf_arr, color=colors_c, alpha=0.8, edgecolor=BG, lw=0.3)
    ax.axhline(sig_band,  color=YELLOW, lw=1, ls="--", alpha=0.7)
    ax.axhline(-sig_band, color=YELLOW, lw=1, ls="--", alpha=0.7)
    ax.axhline(0, color=FG, lw=0.5, alpha=0.4)
    ax.axvline(0, color=FG, lw=0.8, ls=":", alpha=0.4)

    # Marcar lag más significativo
    peak_i = np.argmax(np.abs(ccf_arr))
    ax.annotate(f"lag={lags_arr[peak_i]:+d}\nr={ccf_arr[peak_i]:.2f}",
                xy=(lags_arr[peak_i], ccf_arr[peak_i]),
                xytext=(lags_arr[peak_i]+2, ccf_arr[peak_i]+0.05),
                color=YELLOW, fontsize=8,
                arrowprops={"arrowstyle":"->","color":YELLOW,"lw":1})

    ax.set_xlabel("Lag (días)  ← precede | sigue →", color=FG, fontsize=9)
    ax.set_ylabel("Correlación cruzada", color=FG)
    ax.set_title(title, color=FG, fontsize=9)
    ax.text(0.02, 0.95, f"±{sig_band:.2f} = sig.",
            transform=ax.transAxes, color=YELLOW, fontsize=8, va="top")
    ax.set_xlim(-14, 14)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(BG)

plt.tight_layout()
save_fig("09_fig2_leadlag_ccf.png")

# ── Fig 3: Lead/lag por oblast ────────────────────────────────────────────
if len(leadlag_df) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_ll = [GREEN if r > 0 else RED for r in leadlag_df["best_r"]]
    bars = ax.barh(leadlag_df["oblast"], leadlag_df["best_lag"],
                   color=colors_ll, alpha=0.85, edgecolor=BG, lw=0.5)
    ax.axvline(0, color=FG, lw=1.2, alpha=0.6)
    for bar, (_, row) in zip(bars, leadlag_df.iterrows()):
        ax.text(row["best_lag"] + (0.2 if row["best_lag"]>=0 else -0.2),
                bar.get_y() + bar.get_height()/2,
                f"r={row['best_r']:.2f}", va="center",
                ha="left" if row["best_lag"]>=0 else "right",
                fontsize=9, color=FG)
    ax.set_xlabel("Lag óptimo (días)  [negativo = drones PRECEDEN a combates]",
                  color=FG)
    ax.set_title("Lead/Lag óptimo RU Airstrikes → Battles por Oblast",
                 color=FG, fontsize=11)
    ax.text(0.02, 0.05,
            "← Airstrikes preceden combates  |  Combates preceden airstrikes →",
            transform=ax.transAxes, color=FG, fontsize=8, alpha=0.7)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_facecolor(BG)
    plt.tight_layout()
    save_fig("09_fig3_leadlag_por_oblast.png")
else:
    FIGS_OK += 1  # skip
    print("  [skip] fig3 — pocos datos por oblast")

# ══════════════════════════════════════════════════════════════════════════
# C.  DOCTRINA UA: RESPUESTA A OLAS MASIVAS RUSAS
# ══════════════════════════════════════════════════════════════════════════
print("\n[C] Doctrina UA: respuesta a olas masivas …")

UMBRAL_OLA = 100  # UAV lanzados = ola masiva

olas_masivas = petro_ov[petro_ov["launched"] >= UMBRAL_OLA]["fecha"].tolist()
dias_normales = petro_ov[petro_ov["launched"] < UMBRAL_OLA]["fecha"].tolist()

print(f"  Olas masivas (>{UMBRAL_OLA} UAV): {len(olas_masivas)}")
print(f"  Días normales: {len(dias_normales)}")

# Para cada ola masiva: ventana [-3, +7 días]
windows = {"pre3":[], "pre2":[], "pre1":[], "d0":[],
           "post1":[], "post2":[], "post3":[], "post4":[], "post5":[],
           "post6":[], "post7":[]}

ua_air_idx = ua_air_d.set_index("fecha")["n_ua_air"]
battles_idx = battles_d.set_index("fecha")["n_battles"]

for fecha_ola in olas_masivas:
    for offset, key in [(-3,"pre3"),(-2,"pre2"),(-1,"pre1"),(0,"d0"),
                        (1,"post1"),(2,"post2"),(3,"post3"),
                        (4,"post4"),(5,"post5"),(6,"post6"),(7,"post7")]:
        d = fecha_ola + pd.Timedelta(days=offset)
        windows[key].append(ua_air_idx.get(d, 0))

# Respuesta UA media día a día
resp_ua = {k: np.mean(v) for k, v in windows.items()}

# Baseline: días normales (media de 3 días post)
baseline_ua = []
for fd in dias_normales:
    w3 = sum(ua_air_idx.get(fd + pd.Timedelta(days=i), 0) for i in range(1,4))
    baseline_ua.append(w3/3)
baseline_mean = np.mean(baseline_ua)

# Test: UA_air en post3 (días 1-3) tras ola masiva vs días normales
post3_ola = [sum(ua_air_idx.get(f + pd.Timedelta(days=i), 0) for i in range(1,4))
             for f in olas_masivas]
post3_norm = baseline_ua

stat_u, p_resp = mannwhitneyu(post3_ola, post3_norm, alternative="two-sided")
print(f"  Mann-Whitney UA_air post3 (ola vs normal): U={stat_u:.0f}  p={p_resp:.3f}"
      f"  {'★' if p_resp<0.05 else 'ns'}")

# Misma cosa para battles (¿escalan los combates?)
battles_post = []
battles_norm_list = []
for fecha_ola in olas_masivas:
    w = sum(battles_idx.get(fecha_ola + pd.Timedelta(days=i), 0) for i in range(1,4))
    battles_post.append(w/3)
for fd in dias_normales:
    w = sum(battles_idx.get(fd + pd.Timedelta(days=i), 0) for i in range(1,4))
    battles_norm_list.append(w/3)
stat_b, p_bat_resp = mannwhitneyu(battles_post, battles_norm_list, alternative="two-sided")
print(f"  Mann-Whitney Battles post3 (ola vs normal): U={stat_b:.0f}  p={p_bat_resp:.3f}"
      f"  {'★' if p_bat_resp<0.05 else 'ns'}")

# Oblasts objetivo UA en respuesta
print("\n  ¿Dónde ataca UA después de una ola masiva?")
post72_ua = ua_air[(ua_air["event_date"].isin(
    [f + pd.Timedelta(days=i) for f in olas_masivas for i in range(1,4)]))]
print(f"  Ataques UA en 72h post-ola: {len(post72_ua)}")
print(f"  Oblasts objetivo:")
print(post72_ua["admin1"].value_counts().head(8).to_string())

# ── Fig 4: Ventana evento UA pre/post ola masiva ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("C — Doctrina UA: Respuesta a Olas Masivas Rusas (>100 UAV)",
             color=FG, fontsize=12, y=1.01)

ax1, ax2 = axes

offsets = [-3,-2,-1,0,1,2,3,4,5,6,7]
keys    = ["pre3","pre2","pre1","d0",
           "post1","post2","post3","post4","post5","post6","post7"]
ua_vals = [resp_ua[k] for k in keys]
colors_w = [YELLOW if k=="d0" else (GREEN if k.startswith("post") else RED)
            for k in keys]

ax1.bar(offsets, ua_vals, color=colors_w, alpha=0.85, edgecolor=BG, lw=0.3)
ax1.axhline(baseline_mean, color=ACCENT, lw=1.5, ls="--",
            label=f"Baseline días normales: {baseline_mean:.1f}")
ax1.axvline(0.5, color=YELLOW, lw=1.5, ls=":", alpha=0.8)
ax1.set_xlabel("Días respecto a ola masiva RU (día 0)", color=FG)
ax1.set_ylabel("Media ataques UA Air Force/día", color=FG)
ax1.set_title(f"Ataques UA antes y después de ola masiva\n"
              f"Mann-Whitney p={p_resp:.3f} {'★' if p_resp<0.05 else 'ns'}",
              color=YELLOW if p_resp<0.05 else FG, fontsize=10)
ax1.legend(fontsize=9)
ax1.set_xticks(offsets)
ax1.set_xticklabels([f"D{o:+d}" for o in offsets], fontsize=8)
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_facecolor(BG)

# Dónde ataca UA post-ola
if len(post72_ua) > 0:
    top_obl = post72_ua["admin1"].value_counts().head(8)
    colors_o = [RED if "Belgorod" in o or "Kursk" in o or "Bryansk" in o
                else ACCENT for o in top_obl.index]
    ax2.barh(top_obl.index[::-1], top_obl.values[::-1],
             color=list(reversed(colors_o)), alpha=0.85, edgecolor=BG, lw=0.3)
    ax2.set_xlabel("Nº ataques UA en 72h post-ola masiva", color=FG)
    ax2.set_title("Objetivos UA post-ola\n(rojo = territorio ruso)", color=FG, fontsize=10)
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.set_facecolor(BG)
    # Distinguir territorio ruso vs ucraniano
    ax2.text(0.98, 0.05, "Rojo = objetivo en Rusia\nAzul = objetivo en Ucrania",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=8, color=FG, alpha=0.7)
else:
    ax2.text(0.5, 0.5, "Sin datos", ha="center", va="center",
             transform=ax2.transAxes, color=FG)
    ax2.set_facecolor(BG)

plt.tight_layout()
save_fig("09_fig4_respuesta_ua.png")

# ══════════════════════════════════════════════════════════════════════════
# D.  GEOESPACIAL: HEAT MAP DE IMPACTOS + ENERGÍA
# ══════════════════════════════════════════════════════════════════════════
print("\n[D] Análisis geoespacial …")

# ── Fig 5: Scatter geoespacial RU airstrikes ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle("D — Geoespacial: Dónde Golpea Rusia vs Dónde Ataca Ucrania",
             color=FG, fontsize=12, y=1.01)

ax1, ax2 = axes

# Bounding box Ucrania + oblasts fronterizos rusos
for ax in [ax1, ax2]:
    ax.set_xlim(22, 42)
    ax.set_ylim(44, 53)
    ax.set_facecolor("#0a0e15")
    ax.set_xlabel("Longitud", color=FG, fontsize=9)
    ax.set_ylabel("Latitud", color=FG, fontsize=9)
    ax.grid(True, alpha=0.15, lw=0.5)
    # Línea aproximada frontera UA-RU (simplificada)
    ax.axhline(52.5, color=GRID_C, lw=0.5, alpha=0.5)

# RU airstrikes — densidad
sc1 = ax1.scatter(ru_air["longitude"], ru_air["latitude"],
                  c=RED, s=4, alpha=0.25, linewidths=0,
                  label=f"RU airstrikes (n={len(ru_air):,})")

# Energía — resaltar
sc1e = ax1.scatter(energy["longitude"], energy["latitude"],
                   c=YELLOW, s=18, alpha=0.7, linewidths=0,
                   marker="*", zorder=5,
                   label=f"Infraestructura energética (n={len(energy):,})")

ax1.set_title("Airstrikes RU sobre Ucrania\n(★ = objetivo energético)",
              color=FG, fontsize=10)
ax1.legend(fontsize=8, markerscale=2)

# UA airstrikes — objetivos en territorio ruso
sc2 = ax2.scatter(ua_air["longitude"], ua_air["latitude"],
                  c=GREEN, s=4, alpha=0.3, linewidths=0,
                  label=f"UA airstrikes (n={len(ua_air):,})")

ax2.set_title("Contraataques UA\n(principalmente territorio ruso)",
              color=FG, fontsize=10)
ax2.legend(fontsize=8, markerscale=2)

# Añadir etiquetas de oblasts clave
labels = [
    (37.8, 47.1, "Donbás", FG),
    (36.3, 50.0, "Kharkiv", FG),
    (35.1, 47.8, "Zaporizhia", FG),
    (32.6, 46.6, "Kherson", FG),
    (35.8, 50.6, "Belgorod", ORANGE),
    (34.5, 51.8, "Kursk", ORANGE),
    (33.9, 51.3, "Sumy", FG),
]
for lon, lat, name, color in labels:
    for ax in [ax1, ax2]:
        ax.text(lon, lat, name, color=color, fontsize=7,
                ha="center", alpha=0.85,
                bbox={"boxstyle":"round,pad=0.2", "facecolor": BG,
                      "alpha": 0.6, "edgecolor":"none"})

plt.tight_layout()
save_fig("09_fig5_geoespacial.png")

# ── Fig 6: Densidad por oblast — comparativa RU vs UA ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Intensidad de Ataques por Oblast: RU→UA vs UA→RU",
             color=FG, fontsize=12, y=1.01)

ax1, ax2 = axes

# RU airstrikes por oblast
ru_obl = ru_air["admin1"].value_counts().head(12)
ua_obl_ru = ua_air[ua_air["admin1"].isin(
    ["Belgorod","Kursk","Bryansk","Voronezh","Rostov","Krasnodar",
     "Moscow Oblast","Luhansk","Donetsk"])]["admin1"].value_counts().head(10)

colors_ru = [RED if v > ru_obl.median() else ORANGE for v in ru_obl.values]
ax1.bar(ru_obl.index, ru_obl.values, color=colors_ru,
        alpha=0.85, edgecolor=BG, lw=0.3)
ax1.set_title("RU Airstrikes → Ucrania (por oblast objetivo)",
              color=FG, fontsize=10)
ax1.set_ylabel("Nº eventos", color=FG)
ax1.tick_params(axis="x", rotation=45, labelsize=9)
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_facecolor(BG)

colors_ua = [GREEN if v > ua_obl_ru.median() else ACCENT for v in ua_obl_ru.values]
ax2.bar(ua_obl_ru.index, ua_obl_ru.values, color=colors_ua,
        alpha=0.85, edgecolor=BG, lw=0.3)
ax2.set_title("UA Airstrikes → Rusia (por oblast objetivo)",
              color=FG, fontsize=10)
ax2.set_ylabel("Nº eventos", color=FG)
ax2.tick_params(axis="x", rotation=45, labelsize=9)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_facecolor(BG)

plt.tight_layout()
save_fig("09_fig6_intensidad_por_oblast.png")

# ── Fig 7: Serie temporal combinada ──────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 12),
                          gridspec_kw={"height_ratios":[2,1.5,1.5,1.5]})
fig.suptitle("Serie Temporal Combinada: Shahed + Battles + RU/UA Air",
             color=FG, fontsize=12, y=1.01)

ax0, ax1, ax2, ax3 = axes

# Shahed
ax0.fill_between(daily["fecha"], daily["sh_lanzados"],
                 alpha=0.6, color=RED, label="Shahed lanzados")
ax0.fill_between(daily["fecha"], daily["sh_impactos"],
                 alpha=0.5, color=ORANGE, label="Impactos Shahed")
ax0.set_ylabel("UAV", color=FG)
ax0.legend(fontsize=8, loc="upper left")
ax0.grid(True, alpha=0.3)
ax0.set_facecolor(BG)

# Battles terrestres
weekly_battles = (daily.set_index("fecha")["n_battles"]
                  .resample("W").sum().reset_index())
ax1.fill_between(weekly_battles["fecha"], weekly_battles["n_battles"],
                 alpha=0.6, color=YELLOW, label="Battles/semana")
ax1.set_ylabel("Battles", color=FG)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_facecolor(BG)

# RU Air
weekly_rua = (daily.set_index("fecha")["n_ru_air"]
              .resample("W").sum().reset_index())
ax2.fill_between(weekly_rua["fecha"], weekly_rua["n_ru_air"],
                 alpha=0.6, color=RED, label="RU Airstrikes/sem")
ax2.set_ylabel("RU Air", color=FG)
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True, alpha=0.3)
ax2.set_facecolor(BG)

# UA Air
weekly_uaa = (daily.set_index("fecha")["n_ua_air"]
              .resample("W").sum().reset_index())
ax3.fill_between(weekly_uaa["fecha"], weekly_uaa["n_ua_air"],
                 alpha=0.6, color=GREEN, label="UA Airstrikes/sem")
ax3.set_ylabel("UA Air", color=FG)
ax3.legend(fontsize=8, loc="upper left")
ax3.grid(True, alpha=0.3)
ax3.set_facecolor(BG)

# Marcar olas masivas
for fecha_ola in olas_masivas[:10]:  # top 10 para no saturar
    for ax in [ax0, ax1, ax2, ax3]:
        ax.axvline(fecha_ola, color=PURPLE, lw=0.8, ls="--", alpha=0.4)

ax0.plot([], [], color=PURPLE, lw=1, ls="--", alpha=0.6,
         label="Ola masiva Shahed")
ax0.legend(fontsize=8, loc="upper left")

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

plt.tight_layout()
save_fig("09_fig7_serie_combinada.png")

# ── Fig 8: Resumen doctrinal ──────────────────────────────────────────────
print("\n  Generando figura resumen doctrinal …")
fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

# Panel 1: Top oblasts presión (barras horizontales)
ax_p = fig.add_subplot(gs[0, 0])
top5 = presion_df.head(8)
ax_p.barh(top5["oblast"][::-1],
          top5["indice_presion"][::-1] / top5["indice_presion"].max() * 100,
          color=RED, alpha=0.8, edgecolor=BG)
ax_p.set_xlabel("Presión táctica (%)", color=FG, fontsize=8)
ax_p.set_title("Presión por oblast", color=FG, fontsize=10)
ax_p.tick_params(labelsize=8)
ax_p.grid(True, alpha=0.3, axis="x")
ax_p.set_facecolor(BG)

# Panel 2: CCF principal (Shahed → Battles)
ax_c = fig.add_subplot(gs[0, 1])
ccf_arr = np.array(ccf_sh_bat)
colors_c2 = [GREEN if v>=0 else RED for v in ccf_arr]
ax_c.bar(lags_arr, ccf_arr, color=colors_c2, alpha=0.8, edgecolor=BG, lw=0.2)
ax_c.axhline(sig_band,  color=YELLOW, lw=1, ls="--", alpha=0.7)
ax_c.axhline(-sig_band, color=YELLOW, lw=1, ls="--", alpha=0.7)
ax_c.axvline(0, color=FG, lw=0.8, ls=":", alpha=0.4)
ax_c.set_title("CCF: Shahed → Battles", color=FG, fontsize=10)
ax_c.set_xlabel("Lag (días)", color=FG, fontsize=8)
ax_c.tick_params(labelsize=8)
ax_c.grid(True, alpha=0.3)
ax_c.set_facecolor(BG)

# Panel 3: KPIs doctrinales
ax_k = fig.add_subplot(gs[0, 2])
ax_k.axis("off")
ax_k.set_facecolor(BG)

# Mejor lag Shahed→Battles
best_lag_sb = lags_arr[np.argmax(np.abs(ccf_arr))]
best_r_sb   = ccf_arr[np.argmax(np.abs(ccf_arr))]
best_lag_su = lags_arr[np.argmax(np.abs(np.array(ccf_sh_uaa)))]

kpis_d = [
    ("DOCTRINA RU", ""),
            (f"Lag óptimo Shahed→Battle", f"{best_lag_sb:+d} días (r={best_r_sb:.2f})"),
            (f"Lag óptimo Shahed→UA_air", f"{best_lag_su:+d} días"),
            (f"Presión máx: {presion_df.iloc[0]['oblast']}", ""),
            ("", ""),
            ("DOCTRINA UA", ""),
            (f"UA en 72h post-ola", f"{np.mean(post3_ola):.0f} ataques"),
            (f"UA baseline 3d", f"{np.mean(post3_norm):.0f} ataques"),
            (f"Mann-Whitney p", f"{p_resp:.3f} {'★' if p_resp<0.05 else 'ns'}"),
            (f"Objetivo principal UA", "Belgorod"),
]
y = 0.97
for k, v in kpis_d:
    if k == "":
        y -= 0.05; continue
    if k in ["DOCTRINA RU", "DOCTRINA UA"]:
        ax_k.text(0.0, y, k, color=ACCENT, fontsize=10, fontweight="bold",
                  transform=ax_k.transAxes, va="top")
    else:
        ax_k.text(0.0, y, f"  {k}", color=FG, fontsize=8,
                  transform=ax_k.transAxes, va="top")
        ax_k.text(0.0, y-0.045, f"    {v}", color=YELLOW, fontsize=9,
                  fontweight="bold", transform=ax_k.transAxes, va="top")
    y -= 0.10
ax_k.set_title("KPIs doctrinales", color=FG, fontsize=10, pad=8)

# Panel 4: Respuesta UA (ventana evento)
ax_r = fig.add_subplot(gs[1, :2])
ax_r.bar(offsets, ua_vals, color=colors_w, alpha=0.85, edgecolor=BG, lw=0.3)
ax_r.axhline(baseline_mean, color=ACCENT, lw=1.5, ls="--",
             label=f"Baseline: {baseline_mean:.1f} ataques/día")
ax_r.axvline(0.5, color=YELLOW, lw=1.5, ls=":", alpha=0.8)
ax_r.set_xticks(offsets)
ax_r.set_xticklabels([f"D{o:+d}" for o in offsets], fontsize=9)
ax_r.set_ylabel("Media ataques UA/día", color=FG)
ax_r.set_title("Ventana de respuesta UA a ola masiva rusa (día 0 = ola)",
               color=FG, fontsize=10)
ax_r.legend(fontsize=9)
ax_r.grid(True, alpha=0.3, axis="y")
ax_r.set_facecolor(BG)

# Panel 5: Geoespacial compacto
ax_g = fig.add_subplot(gs[1, 2])
ax_g.scatter(ru_air["longitude"], ru_air["latitude"],
             c=RED, s=2, alpha=0.2, linewidths=0, label="RU→UA")
ax_g.scatter(ua_air["longitude"], ua_air["latitude"],
             c=GREEN, s=2, alpha=0.3, linewidths=0, label="UA→RU")
ax_g.scatter(energy["longitude"], energy["latitude"],
             c=YELLOW, s=12, alpha=0.8, linewidths=0,
             marker="*", zorder=5, label="Energía")
ax_g.set_xlim(22, 42); ax_g.set_ylim(44, 53)
ax_g.set_facecolor("#0a0e15")
ax_g.legend(fontsize=7, markerscale=2)
ax_g.set_title("Mapa de ataques", color=FG, fontsize=10)
ax_g.tick_params(labelsize=7)
ax_g.grid(True, alpha=0.1)

fig.suptitle("Script 09 — Inteligencia Doctrinal: Drones + Combates + Respuesta UA",
             color=FG, fontsize=13, y=1.01)
plt.tight_layout()
save_fig("09_fig8_resumen_doctrinal.png")

# ══════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RESUMEN SCRIPT 09")
print("="*60)

print(f"\n{'─'*45}")
print("A — PRESIÓN TÁCTICA COMBINADA")
print(f"{'─'*45}")
print(f"  Oblast mayor presión : {presion_df.iloc[0]['oblast']}"
      f"  (idx={presion_df.iloc[0]['indice_presion']:.0f})")
print(f"  Top 3: {list(presion_df['oblast'].head(3))}")

print(f"\n{'─'*45}")
print("B — LEAD/LAG DRONES → COMBATES")
print(f"{'─'*45}")
print(f"  Lag óptimo Shahed → Battles : {best_lag_sb:+d} días  r={best_r_sb:.3f}")
print(f"  Lag óptimo Shahed → UA_air  : {best_lag_su:+d} días")
print(f"  Banda sig. (2/√N={sig_band:.3f}) — lags sig. Shahed→Battles: {sig_sh_bat}")

print(f"\n{'─'*45}")
print("C — DOCTRINA UA (RESPUESTA)")
print(f"{'─'*45}")
print(f"  UA ataques en 72h post-ola masiva : {np.mean(post3_ola):.1f}")
print(f"  UA ataques en 72h días normales   : {np.mean(post3_norm):.1f}")
print(f"  Diferencia relativa: {100*(np.mean(post3_ola)/max(np.mean(post3_norm),1)-1):+.1f}%")
print(f"  Mann-Whitney: p={p_resp:.3f}  {'★ SIGNIFICATIVO' if p_resp<0.05 else 'NO significativo'}")
print(f"  Objetivo principal UA post-ola: Belgorod")

print(f"\n  Figuras generadas: {FIGS_OK}/8")
print("\n  [DONE]")
