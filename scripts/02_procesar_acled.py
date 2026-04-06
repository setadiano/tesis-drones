"""
=============================================================
TESIS: Uso Operacional de Drones en Conflictos Modernos
Script 02 — Procesamiento y Análisis ACLED  (v2)
=============================================================
Inputs :
  data/raw/acled_ukraine_2025_2026.csv  ← obligatorio
  data/raw/acled_ukraine_2026.csv       ← opcional (si lo tienes)
Outputs:
  data/processed/acled_drones_limpio.csv
  data/processed/acled_drones_ofensivo_RU.csv   ← Rusia -> Ucrania
  data/processed/acled_drones_ofensivo_UA.csv   ← Ucrania -> Rusia
  data/processed/acled_mensual_agregado.csv
  outputs/figures/06_acled_eventos_diarios.png
  outputs/figures/07_acled_regiones.png
  outputs/figures/08_acled_tipos_arma.png
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

# ── Estilo ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22", "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "text.color": "#e6edf3",
    "grid.color": "#21262d", "grid.linestyle": "--", "grid.alpha": 0.6,
    "font.family": "DejaVu Sans", "axes.titlesize": 13, "axes.labelsize": 11,
})
AZUL = "#58a6ff"; VERDE = "#3fb950"; ROJO = "#f85149"
AMBAR = "#d29922"; LILA = "#bc8cff"

# ════════════════════════════════════════════════════════════
# 1. CARGA — fusiona 2025 y 2026 si existen ambos ficheros
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("PROCESAMIENTO ACLED — UCRANIA 2025-2026  (v2)")
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
print(f"  Rango fechas                 : {df['event_date'].min().date()} → {df['event_date'].max().date()}")

# ════════════════════════════════════════════════════════════
# 2. FILTRO PRECISO — solo eventos aéreos/drone reales
# ════════════════════════════════════════════════════════════

# A) Por sub_event_type — los más relevantes
SUBTIPOS_DRONE = [
    "air/drone strike",
    "disrupted weapons use",   # intercepciones
]

# B) Por keywords en notes — armas aéreas específicas
KW_ARMA = [
    "shahed", "geran", "shaheed", "harpy",
    "lancet", "loitering munition",
    "fpv", "first-person", "fiber-optic", "fibre-optic",
    "drone", "uav", "unmanned aerial",
    "kalibr", "kh-101", "kh-22", "kh-47", "kinzhal",
    "iskander", "tochka", "s-300", "s-400",
    "cruise missile", "ballistic missile",
    "geran-2", "geran-3", "shahed-136", "shahed-238",
]

mask_subtipo = df["sub_event_type"].str.lower().isin(SUBTIPOS_DRONE)
mask_kw      = df["notes"].str.lower().str.contains(
    "|".join(KW_ARMA), na=False, regex=True
)
# Excluir artillería pura y choques armados que no son aéreos
mask_excluir = (
    df["sub_event_type"].str.lower().str.contains(
        "armed clash|looting|mob violence|protest|riot|seizure", na=False
    ) & ~mask_kw
)

df_drones = df[(mask_subtipo | mask_kw) & ~mask_excluir].copy()

# ── Clasificar tipo de arma ──────────────────────────────────
def clasificar_arma(nota):
    if pd.isna(nota): return "Desconocido"
    n = nota.lower()
    if any(k in n for k in ["shahed","geran","shaheed","harpy","gerbera"]): return "Shahed/Geran"
    if any(k in n for k in ["lancet","loitering"]): return "Lancet"
    if any(k in n for k in ["fpv","first-person","fiber","fibre"]): return "FPV"
    if any(k in n for k in ["kinzhal","kh-47","iskander-m ballistic"]): return "Hipersónico"
    if any(k in n for k in ["kalibr","kh-101","kh-55","cruise"]): return "Crucero"
    if any(k in n for k in ["iskander","tochka","ballistic","s-300","s-400"]): return "Balístico"
    if any(k in n for k in ["drone","uav","unmanned"]): return "Drone genérico"
    return "Otro aéreo"

# ── Clasificar dirección del ataque ─────────────────────────
def clasificar_direccion(actor1, actor2, notas):
    a1 = str(actor1).lower() if pd.notna(actor1) else ""
    a2 = str(actor2).lower() if pd.notna(actor2) else ""
    n  = str(notas).lower()  if pd.notna(notas)  else ""
    if any(k in a1 for k in ["russia","armed forces of russia","vks","russian"]):
        return "RU->UA"
    if any(k in a1 for k in ["ukraine","armed forces of ukraine","ukrainian"]):
        return "UA->RU"
    if "disrupted" in str(actor1).lower():
        return "Intercepción"
    return "No determinado"

df_drones["tipo_arma"]    = df_drones["notes"].apply(clasificar_arma)
df_drones["direccion"]    = df_drones.apply(
    lambda r: clasificar_direccion(r["actor1"], r["actor2"], r["notes"]), axis=1
)
df_drones["anio"]         = df_drones["event_date"].dt.year
df_drones["mes"]          = df_drones["event_date"].dt.month
df_drones["mes_anio"]     = df_drones["event_date"].dt.to_period("M")
df_drones["dia_semana"]   = df_drones["event_date"].dt.day_name()

# ── Separar ofensiva rusa vs ucraniana ───────────────────────
df_ru_ua = df_drones[df_drones["direccion"] == "RU->UA"].copy()
df_ua_ru = df_drones[df_drones["direccion"] == "UA->RU"].copy()

print(f"\n✓ Total eventos drone/aéreo filtrados : {len(df_drones):,}")
print(f"  → Rusia atacando Ucrania  (RU->UA) : {len(df_ru_ua):,}")
print(f"  → Ucrania atacando Rusia  (UA->RU) : {len(df_ua_ru):,}")
print(f"  → Intercepciones                   : {len(df_drones[df_drones['direccion']=='Intercepción']):,}")
print(f"  → No determinado                   : {len(df_drones[df_drones['direccion']=='No determinado']):,}")

print(f"\n── Tipo de arma ──")
print(df_drones["tipo_arma"].value_counts().to_string())

print(f"\n── Top 10 regiones afectadas ──")
print(df_drones["admin1"].value_counts().head(10).to_string())

# ── Guardar datasets ─────────────────────────────────────────
cols = ["event_id_cnty","event_date","anio","mes","mes_anio","dia_semana",
        "event_type","sub_event_type","actor1","actor2","admin1","admin2",
        "location","latitude","longitude","fatalities","tipo_arma",
        "direccion","notes","source","tags"]
cols = [c for c in cols if c in df_drones.columns]

df_drones[cols].to_csv(DATA_P / "acled_drones_limpio.csv",       index=False)
df_ru_ua[cols].to_csv( DATA_P / "acled_drones_ofensivo_RU.csv",  index=False)
df_ua_ru[cols].to_csv( DATA_P / "acled_drones_ofensivo_UA.csv",  index=False)
print(f"\n✓ acled_drones_limpio.csv       → {len(df_drones):,} filas")
print(f"✓ acled_drones_ofensivo_RU.csv  → {len(df_ru_ua):,} filas")
print(f"✓ acled_drones_ofensivo_UA.csv  → {len(df_ua_ru):,} filas")

# ── Agregado mensual ─────────────────────────────────────────
mensual = df_drones.groupby("mes_anio").agg(
    n_eventos       =("event_id_cnty","count"),
    n_bajas         =("fatalities","sum"),
    n_ru_ua         =("direccion", lambda x: (x=="RU->UA").sum()),
    n_ua_ru         =("direccion", lambda x: (x=="UA->RU").sum()),
    n_intercepciones=("direccion", lambda x: (x=="Intercepción").sum()),
    regiones_unicas =("admin1","nunique"),
).reset_index()
mensual["mes_anio_str"] = mensual["mes_anio"].astype(str)
mensual.to_csv(DATA_P / "acled_mensual_agregado.csv", index=False)

# ── Tablas resumen ───────────────────────────────────────────
region_df = df_drones.groupby("admin1").agg(
    n_eventos=("event_id_cnty","count"),
    n_bajas  =("fatalities","sum"),
    lat_media=("latitude","mean"),
    lon_media=("longitude","mean"),
).sort_values("n_eventos", ascending=False).reset_index()
region_df.to_csv(TABLES / "03_acled_resumen_regiones.csv", index=False)

actor_df = df_drones.groupby("actor1").agg(
    n_eventos=("event_id_cnty","count"),
    n_bajas  =("fatalities","sum"),
).sort_values("n_eventos", ascending=False).head(20).reset_index()
actor_df.to_csv(TABLES / "04_acled_actores.csv", index=False)
print(f"✓ Tablas resumen guardadas")

# ════════════════════════════════════════════════════════════
# 3. FIGURA 6 — Serie temporal diaria con dirección
# ════════════════════════════════════════════════════════════
diario_ru = df_ru_ua.groupby("event_date").size().rename("RU->UA")
diario_ua = df_ua_ru.groupby("event_date").size().rename("UA->RU")
diario    = pd.concat([diario_ru, diario_ua], axis=1).fillna(0)
diario["roll7_RU"] = diario["RU->UA"].rolling(7, center=True).mean()
diario["roll7_UA"] = diario["UA->RU"].rolling(7, center=True).mean()

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(diario.index, diario["RU->UA"], color=ROJO,  alpha=0.4, width=1, label="Rusia → Ucrania")
ax.bar(diario.index, diario["UA->RU"], color=AZUL,  alpha=0.4, width=1,
       bottom=diario["RU->UA"], label="Ucrania → Rusia")
ax.plot(diario.index, diario["roll7_RU"], color=ROJO,  lw=2, label="Media 7d RU")
ax.plot(diario.index, diario["roll7_UA"], color=AZUL,  lw=2, ls="--", label="Media 7d UA")

for anio in [2026]:
    t = pd.Timestamp(f"{anio}-01-01")
    if t >= diario.index.min() and t <= diario.index.max():
        ax.axvline(t, color="#8b949e", lw=1, ls="--", alpha=0.7)
        ax.text(t, ax.get_ylim()[1]*0.92, f" {anio}", color="#8b949e", fontsize=9)

ax.set_xlabel("Fecha"); ax.set_ylabel("Nº eventos")
ax.set_title("Eventos aéreos/drone diarios por dirección — ACLED Ucrania", pad=12)
ax.legend(fontsize=9, framealpha=0.3); ax.grid(axis="y")
plt.tight_layout()
plt.savefig(FIGS / "06_acled_eventos_diarios.png", bbox_inches="tight")
plt.close()
print(f"\n✓ Figura 6 → 06_acled_eventos_diarios.png")

# ════════════════════════════════════════════════════════════
# 4. FIGURA 7 — Top regiones
# ════════════════════════════════════════════════════════════
top15 = region_df.head(15)
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(top15["admin1"][::-1], top15["n_eventos"][::-1], color=AZUL, alpha=0.85, label="Eventos")
ax2 = ax.twiny()
ax2.barh(top15["admin1"][::-1], top15["n_bajas"][::-1],  color=ROJO, alpha=0.4,  label="Bajas")
ax2.set_xlabel("Bajas confirmadas", color=ROJO)
ax2.tick_params(axis="x", colors=ROJO)
ax.set_xlabel("Nº de eventos")
ax.set_title("Top 15 regiones más afectadas — ACLED Ucrania", pad=12)
p1 = mpatches.Patch(color=AZUL, alpha=0.85, label="Eventos")
p2 = mpatches.Patch(color=ROJO, alpha=0.4,  label="Bajas")
ax.legend(handles=[p1,p2], fontsize=9, framealpha=0.3, loc="lower right")
ax.grid(axis="x")
plt.tight_layout()
plt.savefig(FIGS / "07_acled_regiones.png", bbox_inches="tight")
plt.close()
print(f"✓ Figura 7 → 07_acled_regiones.png")

# ════════════════════════════════════════════════════════════
# 5. FIGURA 8 — Tipos de arma + evolución mensual
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
arma_counts = df_drones["tipo_arma"].value_counts()
cols_pie = [AZUL,ROJO,VERDE,AMBAR,LILA,"#ff7b72","#79c0ff","#a5d6ff"]
axes[0].pie(arma_counts.values, labels=arma_counts.index,
            colors=cols_pie[:len(arma_counts)],
            autopct="%1.1f%%", startangle=90,
            textprops={"color":"#e6edf3","fontsize":9})
axes[0].set_title("Distribución por tipo de arma", pad=10)

top_armas = arma_counts.head(4).index.tolist()
df_arma_mes = df_drones[df_drones["tipo_arma"].isin(top_armas)].groupby(
    ["mes_anio","tipo_arma"]).size().unstack(fill_value=0)
for i, arma in enumerate(top_armas):
    if arma in df_arma_mes.columns:
        axes[1].plot(range(len(df_arma_mes)), df_arma_mes[arma],
                     marker="o", ms=4, lw=2,
                     color=cols_pie[i], label=arma)
axes[1].set_xticks(range(len(df_arma_mes)))
axes[1].set_xticklabels([str(p) for p in df_arma_mes.index],
                         rotation=45, ha="right", fontsize=7)
axes[1].set_ylabel("Nº eventos"); axes[1].set_title("Evolución mensual por tipo de arma", pad=10)
axes[1].legend(fontsize=8, framealpha=0.3); axes[1].grid(axis="y")
plt.suptitle("Análisis de tipos de arma — ACLED Ucrania 2025-2026", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIGS / "08_acled_tipos_arma.png", bbox_inches="tight")
plt.close()
print(f"✓ Figura 8 → 08_acled_tipos_arma.png")

# ════════════════════════════════════════════════════════════
# 6. RESUMEN
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("COMPLETADO")
print("="*60)
print(f"Total eventos filtrados : {len(df_drones):,}")
print(f"  RU->UA                : {len(df_ru_ua):,}")
print(f"  UA->RU                : {len(df_ua_ru):,}")
print(f"Bajas totales           : {int(df_drones['fatalities'].sum()):,}")
print(f"Rango temporal          : {df['event_date'].min().date()} → {df['event_date'].max().date()}")
print(f"Oblasts cubiertos       : {df_drones['admin1'].nunique()}")
print(f"\nPróximo: scripts/03_series_temporales.py")
