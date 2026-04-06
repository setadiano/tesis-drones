"""
=============================================================
TESIS: Uso Operacional de Drones en Conflictos Modernos
Script 02 — Procesamiento y Análisis ACLED  (v3)
=============================================================
Cambios v3:
  - Reclasificación geográfica: eventos en Belgorod/Kursk/Bryansk/Rostov
    (oblasts rusos fronterizos) → UA->RU aunque actor1 sea Rusia
  - Detección de intercepciones desde notas: 'intercepted', 'shot down',
    'destroyed', 'neutralized', 'downed', etc.
  - Clasificación de armas mejorada: Mohajer, Arash, R-18, Baba Yaga,
    Vampire, UJ-22, SHARK, A22, Punisher, etc.

Inputs :
  data/raw/acled_ukraine_2025_2026.csv  ← obligatorio
  data/raw/acled_ukraine_2026.csv       ← opcional (si lo tienes)
Outputs:
  data/processed/acled_drones_limpio.csv
  data/processed/acled_drones_ofensivo_RU.csv   ← Rusia -> Ucrania
  data/processed/acled_drones_ofensivo_UA.csv   ← Ucrania -> Rusia
  data/processed/acled_intercepciones.csv       ← NUEVO
  data/processed/acled_mensual_agregado.csv
  outputs/figures/06_acled_eventos_diarios.png
  outputs/figures/07_acled_regiones.png
  outputs/figures/08_acled_tipos_arma.png
  outputs/tables/03_acled_resumen_regiones.csv
  outputs/tables/04_acled_actores.csv
  outputs/tables/05_acled_intercepciones_resumen.csv  ← NUEVO
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Rutas ────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent.parent
DATA_R = BASE / "data" / "raw"
DATA_P = BASE / "data" / "processed"
FIGS   = BASE / "outputs" / "figures"
TABLES = BASE / "outputs" / "tables"
DATA_P.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# ── Estilo dark theme ────────────────────────────────────────
BG    = "#0d1117"
BG2   = "#161b22"
GRID  = "#21262d"
FG    = "#e6edf3"
FG2   = "#8b949e"
AZUL  = "#58a6ff"
VERDE = "#3fb950"
ROJO  = "#f85149"
AMBAR = "#d29922"
LILA  = "#bc8cff"
CYAN  = "#39d353"

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": BG,
    "axes.facecolor": BG2,
    "axes.edgecolor": GRID,
    "axes.labelcolor": FG,
    "xtick.color": FG2,
    "ytick.color": FG2,
    "text.color": FG,
    "grid.color": GRID,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.facecolor": BG2,
    "legend.edgecolor": GRID,
})

# ════════════════════════════════════════════════════════════
# 1. CARGA
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("PROCESAMIENTO ACLED — UCRANIA 2025-2026  (v3)")
print("=" * 60)

archivos = []
for nombre in ["acled_ukraine_2025_2026.csv", "acled_ukraine_2026.csv"]:
    p = DATA_R / nombre
    if p.exists():
        archivos.append(pd.read_csv(p, low_memory=False))
        print(f"✓ Cargado: {nombre}  ({len(archivos[-1]):,} filas)")
    else:
        print(f"  (no encontrado, se omite): {nombre}")

df = pd.concat(archivos, ignore_index=True)
df["event_date"] = pd.to_datetime(df["event_date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["event_date"])
df = df.drop_duplicates(subset=["event_id_cnty"])
df = df.sort_values("event_date").reset_index(drop=True)

print(f"\n✓ Total registros tras fusión  : {len(df):,}")
print(f"  Rango fechas                 : {df['event_date'].min().date()} → "
      f"{df['event_date'].max().date()}")

# ════════════════════════════════════════════════════════════
# 2. FILTRO — solo eventos aéreos/drone reales
# ════════════════════════════════════════════════════════════

SUBTIPOS_DRONE = [
    "air/drone strike",
    "disrupted weapons use",
]

KW_ARMA = [
    # Shahed / Geran familia
    "shahed", "geran", "shaheed", "gerbera", "harpy",
    # Lancet / loitering
    "lancet", "loitering munition", "loitering",
    # FPV
    "fpv", "first-person", "fiber-optic", "fibre-optic",
    # Genérico drone/UAV
    "drone", "uav", "unmanned aerial", "unmanned aircraft",
    # Misiles de crucero rusos
    "kalibr", "kh-101", "kh-55", "kh-22", "cruise missile",
    # Hipersónico / balístico ruso
    "kh-47", "kinzhal", "iskander", "tochka", "s-300", "s-400",
    "ballistic missile", "ballistic",
    # Drones ucranianos ofensivos
    "mohajer", "arash",           # iraníes capturados/reingeniería
    "r-18", "r18",                 # dron hexacóptero UA pesado
    "baba yaga", "babayaga",       # dron agrícola convertido
    "vampire", "vampir",           # sistema MLRS-drone UA
    "uj-22", "uj22",               # dron de ala fija UA largo alcance
    "shark", "sharк",              # dron de reconocimiento UA
    "a22",                         # dron kamikaze UA
    "punisher",                    # dron kamikaze UA
    "leleka", "spectator",         # reconocimiento UA
    "муром", "лелека",             # nombres en ruso/ucraniano
    "geran-2", "geran-3",
    "shahed-136", "shahed-238", "shahed-149",
]

mask_subtipo = df["sub_event_type"].str.lower().isin(SUBTIPOS_DRONE)
mask_kw      = df["notes"].str.lower().str.contains(
    "|".join(KW_ARMA), na=False, regex=True
)
mask_excluir = (
    df["sub_event_type"].str.lower().str.contains(
        "armed clash|looting|mob violence|protest|riot|seizure", na=False
    ) & ~mask_kw
)

df_drones = df[(mask_subtipo | mask_kw) & ~mask_excluir].copy()
df_drones["notes_lower"] = df_drones["notes"].fillna("").str.lower()

# ════════════════════════════════════════════════════════════
# 3. MEJORA: CLASIFICACIÓN DE ARMAS (v3)
# ════════════════════════════════════════════════════════════
#  Prioridad: de más específico a más genérico.
#  Reduce "Drone genérico" identificando modelos ucranianos.

def clasificar_arma(nota):
    if pd.isna(nota):
        return "Desconocido"
    n = nota.lower()

    # ── Drones rusos identificados ──
    if any(k in n for k in ["shahed", "geran", "shaheed", "gerbera", "harpy",
                              "shahed-136", "shahed-238", "shahed-149",
                              "geran-2", "geran-3"]):
        return "Shahed/Geran"
    if any(k in n for k in ["lancet", "loitering munition"]):
        return "Lancet"

    # ── Drones ucranianos identificados ──
    if any(k in n for k in ["r-18", "r18", "baba yaga", "babayaga"]):
        return "Drone pesado UA (R-18/BabaYaga)"
    if any(k in n for k in ["uj-22", "uj22"]):
        return "UJ-22 (ala fija UA)"
    if any(k in n for k in ["shark", "sharк", "spectator", "leleka", "лелека"]):
        return "Drone rec. UA (SHARK/Leleka)"
    if any(k in n for k in ["punisher", "a22", "vampire", "vampir"]):
        return "Drone kamikaze UA"

    # ── FPV ──
    if any(k in n for k in ["fpv", "first-person", "fiber-optic", "fibre-optic"]):
        return "FPV"

    # ── Misiles y vectores balísticos ──
    if any(k in n for k in ["kinzhal", "kh-47", "iskander-m ballistic"]):
        return "Hipersónico"
    if any(k in n for k in ["kalibr", "kh-101", "kh-55", "cruise missile"]):
        return "Crucero"
    if any(k in n for k in ["iskander", "tochka", "ballistic", "s-300", "s-400"]):
        return "Balístico"

    # ── Genérico ──
    if any(k in n for k in ["drone", "uav", "unmanned"]):
        return "Drone genérico"

    return "Otro aéreo"

# ════════════════════════════════════════════════════════════
# 4. MEJORA: DETECCIÓN DE INTERCEPCIONES (v3)
# ════════════════════════════════════════════════════════════
#  Un evento es intercepción si:
#  a) sub_event_type == "disrupted weapons use", O
#  b) las notas contienen keywords de neutralización

KW_INTERCEPCION = [
    "intercepted", "shot down", "downed", "destroyed",
    "neutralized", "neutralised", "defeated",
    "перехвачен", "сбит",          # ruso: interceptado / derribado
    "збито", "перехоплено",        # ucraniano: derribado / interceptado
    "air defense", "air defence",
    "missile defense", "missile defence",
    "destroyed mid-air", "mid-air",
    "electronic warfare", "jamm",
    "forced to land",
]

mask_intercep_sub = df_drones["sub_event_type"].str.lower() == "disrupted weapons use"
mask_intercep_kw  = df_drones["notes_lower"].str.contains(
    "|".join(KW_INTERCEPCION), na=False, regex=True
)
df_drones["es_intercepcion"] = (mask_intercep_sub | mask_intercep_kw)

# ════════════════════════════════════════════════════════════
# 5. MEJORA: CLASIFICACIÓN DIRECCIÓN (v3)
# ════════════════════════════════════════════════════════════
#  Lógica:
#  1. Si es intercepción → "Intercepción"
#  2. Si admin1 es oblast ruso fronterizo → UA->RU (independiente del actor)
#  3. Si actor1 es ruso → RU->UA
#  4. Si actor1 es ucraniano → UA->RU
#  5. Si notas hablan de ataque ucraniano a Rusia → UA->RU
#  6. Resto → "No determinado"

# Oblasts rusos fronterizos que aparecen en ACLED
OBLASTS_RUSOS = {
    "belgorod", "kursk", "bryansk", "rostov", "voronezh",
    "smolensk", "oryol", "orel", "krasnodar", "lipetsk",
    "tambov", "saratov",
}

def clasificar_direccion(row):
    a1    = str(row.get("actor1", "")).lower()
    a2    = str(row.get("actor2", "")).lower()
    notas = row.get("notes_lower", "")
    adm1  = str(row.get("admin1", "")).lower().strip()

    # Regla 0: intercepción antes de todo
    if row.get("es_intercepcion", False):
        return "Intercepción"

    # Regla 1: geográfica — oblast ruso → siempre UA->RU
    if any(obl in adm1 for obl in OBLASTS_RUSOS):
        return "UA->RU"

    # Regla 2: actor explícito ruso
    if any(k in a1 for k in ["russia", "armed forces of russia", "vks",
                               "russian air", "wagner", "russian"]):
        return "RU->UA"

    # Regla 3: actor explícito ucraniano
    if any(k in a1 for k in ["ukraine", "armed forces of ukraine",
                               "ukrainian", "гур", "sbu", "gur"]):
        return "UA->RU"

    # Regla 4: keywords en notas que indican ataque ucraniano en Rusia
    kw_ua_ataca = ["attacked russia", "struck russia", "hit russia",
                   "attack on russia", "ukraine struck", "ukraine attacked",
                   "ukrainian drone", "ukrainian uav", "ua drone",
                   "ukrainian fpv", "ua fpv"]
    if any(k in notas for k in kw_ua_ataca):
        return "UA->RU"

    # Regla 5: keywords en notas que indican ataque ruso a Ucrania
    kw_ru_ataca = ["russian drone", "russian uav", "russian missile",
                   "russian air", "shahed", "geran", "kalibr", "iskander",
                   "kh-101", "kh-47", "kinzhal"]
    if any(k in notas for k in kw_ru_ataca):
        return "RU->UA"

    return "No determinado"

df_drones["tipo_arma"]  = df_drones["notes"].apply(clasificar_arma)
df_drones["direccion"]  = df_drones.apply(clasificar_direccion, axis=1)
df_drones["anio"]       = df_drones["event_date"].dt.year
df_drones["mes"]        = df_drones["event_date"].dt.month
df_drones["mes_anio"]   = df_drones["event_date"].dt.to_period("M")
df_drones["dia_semana"] = df_drones["event_date"].dt.day_name()

# ── Separar subconjuntos ────────────────────────────────────
df_ru_ua   = df_drones[df_drones["direccion"] == "RU->UA"].copy()
df_ua_ru   = df_drones[df_drones["direccion"] == "UA->RU"].copy()
df_intercep = df_drones[df_drones["direccion"] == "Intercepción"].copy()

print(f"\n✓ Total eventos drone/aéreo filtrados : {len(df_drones):,}")
print(f"  → Rusia atacando Ucrania  (RU->UA) : {len(df_ru_ua):,}")
print(f"  → Ucrania atacando Rusia  (UA->RU) : {len(df_ua_ru):,}")
print(f"  → Intercepciones                   : {len(df_intercep):,}")
print(f"  → No determinado                   : "
      f"{len(df_drones[df_drones['direccion']=='No determinado']):,}")

print(f"\n── Tipo de arma ──")
print(df_drones["tipo_arma"].value_counts().to_string())

print(f"\n── Top 10 regiones afectadas ──")
print(df_drones["admin1"].value_counts().head(10).to_string())

print(f"\n── Top 5 oblasts rusos (UA->RU) ──")
if len(df_ua_ru) > 0:
    print(df_ua_ru["admin1"].value_counts().head(5).to_string())

print(f"\n── Intercepciones por tipo de arma ──")
if len(df_intercep) > 0:
    print(df_intercep["tipo_arma"].value_counts().head(8).to_string())

# ════════════════════════════════════════════════════════════
# 6. GUARDAR DATASETS
# ════════════════════════════════════════════════════════════
cols_out = [
    "event_id_cnty", "event_date", "anio", "mes", "mes_anio", "dia_semana",
    "event_type", "sub_event_type", "actor1", "actor2",
    "admin1", "admin2", "location", "latitude", "longitude",
    "fatalities", "tipo_arma", "direccion", "es_intercepcion",
    "notes", "source", "tags",
]
cols_out = [c for c in cols_out if c in df_drones.columns]

df_drones[cols_out].to_csv(DATA_P / "acled_drones_limpio.csv",       index=False)
df_ru_ua[cols_out].to_csv( DATA_P / "acled_drones_ofensivo_RU.csv",  index=False)
df_ua_ru[cols_out].to_csv( DATA_P / "acled_drones_ofensivo_UA.csv",  index=False)
df_intercep[cols_out].to_csv(DATA_P / "acled_intercepciones.csv",    index=False)

print(f"\n✓ acled_drones_limpio.csv       → {len(df_drones):,} filas")
print(f"✓ acled_drones_ofensivo_RU.csv  → {len(df_ru_ua):,} filas")
print(f"✓ acled_drones_ofensivo_UA.csv  → {len(df_ua_ru):,} filas")
print(f"✓ acled_intercepciones.csv      → {len(df_intercep):,} filas  ← NUEVO")

# ── Agregado mensual ────────────────────────────────────────
mensual = df_drones.groupby("mes_anio").agg(
    n_eventos        = ("event_id_cnty", "count"),
    n_bajas          = ("fatalities",    "sum"),
    n_ru_ua          = ("direccion", lambda x: (x == "RU->UA").sum()),
    n_ua_ru          = ("direccion", lambda x: (x == "UA->RU").sum()),
    n_intercepciones = ("direccion", lambda x: (x == "Intercepción").sum()),
    n_no_det         = ("direccion", lambda x: (x == "No determinado").sum()),
    regiones_unicas  = ("admin1",    "nunique"),
).reset_index()
mensual["mes_inicio"] = mensual["mes_anio"].dt.to_timestamp()
mensual["mes_anio_str"] = mensual["mes_anio"].astype(str)
mensual.to_csv(DATA_P / "acled_mensual_agregado.csv", index=False)

# ── Tablas resumen ──────────────────────────────────────────
region_df = df_drones.groupby("admin1").agg(
    n_eventos = ("event_id_cnty", "count"),
    n_bajas   = ("fatalities",    "sum"),
    lat_media = ("latitude",  "mean"),
    lon_media = ("longitude", "mean"),
    n_ru_ua   = ("direccion", lambda x: (x == "RU->UA").sum()),
    n_ua_ru   = ("direccion", lambda x: (x == "UA->RU").sum()),
).sort_values("n_eventos", ascending=False).reset_index()
region_df.to_csv(TABLES / "03_acled_resumen_regiones.csv", index=False)

actor_df = df_drones.groupby("actor1").agg(
    n_eventos = ("event_id_cnty", "count"),
    n_bajas   = ("fatalities",    "sum"),
).sort_values("n_eventos", ascending=False).head(20).reset_index()
actor_df.to_csv(TABLES / "04_acled_actores.csv", index=False)

# Tabla intercepciones resumen ← NUEVO
if len(df_intercep) > 0:
    intercep_resumen = df_intercep.groupby(["mes_anio", "tipo_arma"]).agg(
        n_intercepciones = ("event_id_cnty", "count"),
        n_bajas          = ("fatalities",    "sum"),
    ).reset_index()
    intercep_resumen["mes_anio_str"] = intercep_resumen["mes_anio"].astype(str)
    intercep_resumen.to_csv(TABLES / "05_acled_intercepciones_resumen.csv", index=False)
    print(f"✓ 05_acled_intercepciones_resumen.csv guardado")

print(f"✓ Tablas resumen guardadas")

# ════════════════════════════════════════════════════════════
# 7. FIGURA 6 — Serie temporal diaria con dirección
# ════════════════════════════════════════════════════════════
diario_ru = df_ru_ua.groupby("event_date").size().rename("RU->UA")
diario_ua = df_ua_ru.groupby("event_date").size().rename("UA->RU")
diario_ic = df_intercep.groupby("event_date").size().rename("Intercep.")
diario = pd.concat([diario_ru, diario_ua, diario_ic], axis=1).fillna(0)
diario["roll7_RU"] = diario["RU->UA"].rolling(7, center=True).mean()
diario["roll7_UA"] = diario["UA->RU"].rolling(7, center=True).mean()

fig, axes = plt.subplots(2, 1, figsize=(16, 9), facecolor=BG,
                          gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
ax.bar(diario.index, diario["RU->UA"], color=ROJO,  alpha=0.4, width=1,
       label="Rusia → Ucrania (RU→UA)")
ax.bar(diario.index, diario["UA->RU"], color=AZUL,  alpha=0.4, width=1,
       bottom=diario["RU->UA"], label="Ucrania → Rusia (UA→RU)")
ax.plot(diario.index, diario["roll7_RU"], color=ROJO,  lw=2,
        label="Media 7d RU→UA")
ax.plot(diario.index, diario["roll7_UA"], color=AZUL,  lw=2, ls="--",
        label="Media 7d UA→RU")

# Marcador inicio 2026 si existe
t2026 = pd.Timestamp("2026-01-01")
if t2026 >= diario.index.min() and t2026 <= diario.index.max():
    ax.axvline(t2026, color=FG2, lw=1, ls="--", alpha=0.7)
    ax.text(t2026, ax.get_ylim()[1] * 0.90, " 2026", color=FG2, fontsize=9)

ax.set_ylabel("Nº eventos / día", color=FG2)
ax.set_title("Eventos aéreos/drone diarios por dirección — ACLED Ucrania 2025", pad=10)
ax.legend(fontsize=9, framealpha=0.3, loc="upper left")
ax.grid(axis="y", alpha=0.4)
ax.set_facecolor(BG2)
for sp in ax.spines.values(): sp.set_color(GRID)

# Panel inferior: intercepciones diarias
ax2 = axes[1]
if "Intercep." in diario.columns and diario["Intercep."].sum() > 0:
    ax2.bar(diario.index, diario["Intercep."], color=VERDE, alpha=0.7,
            width=1, label="Intercepciones detectadas")
    ax2.set_ylabel("Intercepciones", color=FG2)
    ax2.legend(fontsize=8, framealpha=0.3, loc="upper left")
else:
    ax2.text(0.5, 0.5, "Sin intercepciones detectadas en los datos ACLED",
             ha="center", va="center", transform=ax2.transAxes, color=FG2, fontsize=10)
ax2.set_facecolor(BG2)
ax2.grid(axis="y", alpha=0.4)
for sp in ax2.spines.values(): sp.set_color(GRID)

plt.tight_layout()
plt.savefig(FIGS / "06_acled_eventos_diarios.png", bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\n✓ Figura 6 → 06_acled_eventos_diarios.png")

# ════════════════════════════════════════════════════════════
# 8. FIGURA 7 — Top regiones (separando RU vs UA)
# ════════════════════════════════════════════════════════════
top15 = region_df.head(15).copy()

# Etiquetar país de cada oblast para el título del eje
def pais_oblast(nombre):
    n = nombre.lower()
    if any(obl in n for obl in OBLASTS_RUSOS):
        return f"{nombre} 🇷🇺"
    return f"{nombre} 🇺🇦"

top15["label"] = top15["admin1"].apply(pais_oblast)

fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
colores_barra = [ROJO if any(obl in row["admin1"].lower() for obl in OBLASTS_RUSOS)
                 else AZUL for _, row in top15.iterrows()]

bars = ax.barh(top15["label"][::-1], top15["n_eventos"][::-1],
               color=colores_barra[::-1], alpha=0.85)

# Añadir valores en las barras
for bar, val in zip(bars, top15["n_eventos"][::-1]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f"{int(val):,}", va="center", ha="left", color=FG2, fontsize=8)

ax.set_xlabel("Nº de eventos")
ax.set_title("Top 15 regiones más afectadas\n🔴 Oblast ruso (objetivo UA→RU)   "
             "🔵 Oblast ucraniano (objetivo RU→UA)", pad=12)
ax.grid(axis="x", alpha=0.4)
ax.set_facecolor(BG2)
for sp in ax.spines.values(): sp.set_color(GRID)
p1 = mpatches.Patch(color=AZUL, alpha=0.85, label="Oblast ucraniano (RU→UA)")
p2 = mpatches.Patch(color=ROJO, alpha=0.85, label="Oblast ruso (UA→RU)")
ax.legend(handles=[p1, p2], fontsize=9, framealpha=0.3, loc="lower right")
plt.tight_layout()
plt.savefig(FIGS / "07_acled_regiones.png", bbox_inches="tight", facecolor=BG)
plt.close()
print(f"✓ Figura 7 → 07_acled_regiones.png")

# ════════════════════════════════════════════════════════════
# 9. FIGURA 8 — Tipos de arma mejorado
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
arma_counts = df_drones["tipo_arma"].value_counts()

PALETTE = [AZUL, ROJO, VERDE, AMBAR, LILA, "#ff7b72", "#79c0ff", "#a5d6ff",
           "#ffa657", "#56d364", CYAN, "#f78166"]

# Donut chart
wedges, texts, autotexts = axes[0].pie(
    arma_counts.values,
    labels=None,
    colors=PALETTE[:len(arma_counts)],
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.82,
    wedgeprops={"linewidth": 1.5, "edgecolor": BG},
)
for at in autotexts:
    at.set_color(FG)
    at.set_fontsize(8)
# Círculo central para donut
centre_circle = plt.Circle((0, 0), 0.60, fc=BG2)
axes[0].add_artist(centre_circle)
axes[0].legend(
    wedges, arma_counts.index,
    loc="lower center", bbox_to_anchor=(0.5, -0.18),
    ncol=2, fontsize=7.5, framealpha=0.3,
)
axes[0].set_title("Distribución por tipo de arma", pad=10)

# Evolución mensual — top 5 armas
top5_armas = arma_counts.head(5).index.tolist()
df_arma_mes = (df_drones[df_drones["tipo_arma"].isin(top5_armas)]
               .groupby(["mes_anio", "tipo_arma"])
               .size()
               .unstack(fill_value=0))

for i, arma in enumerate(top5_armas):
    if arma in df_arma_mes.columns:
        vals = df_arma_mes[arma].values
        xs   = range(len(df_arma_mes))
        axes[1].plot(xs, vals, marker="o", ms=4, lw=2,
                     color=PALETTE[i], label=arma)
        axes[1].fill_between(xs, vals, alpha=0.08, color=PALETTE[i])

axes[1].set_xticks(range(len(df_arma_mes)))
axes[1].set_xticklabels(
    [str(p) for p in df_arma_mes.index],
    rotation=45, ha="right", fontsize=7,
)
axes[1].set_ylabel("Nº eventos")
axes[1].set_title("Evolución mensual — top 5 tipos de arma", pad=10)
axes[1].legend(fontsize=8, framealpha=0.3, loc="upper left")
axes[1].grid(axis="y", alpha=0.4)
axes[1].set_facecolor(BG2)
for sp in axes[1].spines.values(): sp.set_color(GRID)
axes[0].set_facecolor(BG2)

plt.suptitle("Análisis de tipos de arma — ACLED Ucrania 2025-2026 (v3)",
             fontsize=13, y=1.01, color=FG)
plt.tight_layout()
plt.savefig(FIGS / "08_acled_tipos_arma.png", bbox_inches="tight", facecolor=BG)
plt.close()
print(f"✓ Figura 8 → 08_acled_tipos_arma.png")

# ════════════════════════════════════════════════════════════
# 10. RESUMEN FINAL
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COMPLETADO")
print("=" * 60)
print(f"Total eventos filtrados : {len(df_drones):,}")
print(f"  RU->UA                : {len(df_ru_ua):,}")
print(f"  UA->RU                : {len(df_ua_ru):,}")
print(f"  Intercepciones        : {len(df_intercep):,}  ← v3")
print(f"Bajas totales           : {int(df_drones['fatalities'].sum()):,}")
print(f"Rango temporal          : {df['event_date'].min().date()} → "
      f"{df['event_date'].max().date()}")
print(f"Oblasts cubiertos       : {df_drones['admin1'].nunique()}")
print(f"\n── Cambios v3 vs v2 ──")
print(f"  • Oblasts rusos reclasificados como UA->RU: "
      f"{len(df_ua_ru[df_ua_ru['admin1'].str.lower().isin(OBLASTS_RUSOS)]):,} eventos")
print(f"  • Intercepciones detectadas desde notas: {len(df_intercep):,}")
arma_vieja = df_drones[df_drones["tipo_arma"] == "Drone genérico"]
pct_genericos = 100 * len(arma_vieja) / max(len(df_drones), 1)
print(f"  • 'Drone genérico' residual: {len(arma_vieja):,} ({pct_genericos:.1f}%)")
print(f"\nPróximo: scripts/03_series_temporales.py")
