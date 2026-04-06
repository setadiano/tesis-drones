"""
Script 10 — Teatro España: Modelo de Saturación + Amenaza Regional
===================================================================
Autor : setadiano / jplatas6@alumno.uned.es
Fecha : Abril 2026

Análisis A — Capacidades UAV amenaza (Marruecos, Argelia, actores no-estatales)
         B — Defensa antiaérea española (Ceuta/Melilla) + umbrales saturación
         C — Geometría del teatro (distancias, tiempos de vuelo, ventanas)
         D — Transferencia de lecciones: Ucrania → España
         E — Modelo de saturación aplicado (TAR ucraniano → Estrecho)

Fuentes OSINT
-------------
  SIPRI Arms Transfers Database 2024
  IISS Military Balance 2024/2025
  Jane's Land-Based Air Defence
  Baykar, MBDA, Kongsberg documentación oficial
  Real Instituto Elcano, Zona Militar, fuerzasmilitares.es
  Ejercito.defensa.gob.es (RAMIX 30, RAMIX 32)
  Lecciones empíricas: Scripts 03-09 (datos Ucrania 2025-2026)

NOTA: Todos los datos son OSINT (fuentes abiertas).
      Los umbrales de saturación son estimaciones analíticas,
      no datos clasificados.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
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
THREAT = "#ff4444"
SAFE   = "#22dd44"

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

print("\n" + "="*60)
print("SCRIPT 10 — TEATRO ESPAÑA: SATURACIÓN + AMENAZA REGIONAL")
print("="*60)

# ══════════════════════════════════════════════════════════════════════════
# A. BASE DE DATOS: CAPACIDADES UAV AMENAZA
# ══════════════════════════════════════════════════════════════════════════
print("\n[A] Cargando base de datos capacidades amenaza …")

amenaza = pd.DataFrame([
    # Marruecos — sistemas ofensivos confirmados OSINT
    {"pais":"Marruecos","sistema":"Bayraktar TB2",   "tipo":"MALE UCAV",
     "unidades":23,  "vcruise_kmh":130, "alcance_km":300,  "payload_kg":150,
     "endurance_h":27, "armado":1, "origen":"Turquía",
     "estado":"Operativo/combate activo",
     "capacidad_simultan":8, "base_norte":1, "dist_ceuta_km":35.5,
     "dist_melilla_km":81.3, "nota":"TB2 base Tetuán confirmada"},
    {"pais":"Marruecos","sistema":"Bayraktar Akinci","tipo":"MALE UCAV estratégico",
     "unidades":3,   "vcruise_kmh":220, "alcance_km":7500, "payload_kg":1500,
     "endurance_h":24, "armado":1, "origen":"Turquía",
     "estado":"Entrega iniciada feb.2025",
     "capacidad_simultan":2, "base_norte":1, "dist_ceuta_km":35.5,
     "dist_melilla_km":81.3, "nota":"Misil SOM antibuque, radar AESA"},
    {"pais":"Marruecos","sistema":"Wing Loong II",   "tipo":"MALE UCAV",
     "unidades":4,   "vcruise_kmh":150, "alcance_km":1000, "payload_kg":480,
     "endurance_h":32, "armado":1, "origen":"China",
     "estado":"Operativo (Sahara Occidental)",
     "capacidad_simultan":3, "base_norte":0, "dist_ceuta_km":486,
     "dist_melilla_km":582, "nota":"SATCOM, alcance >1000km"},
    {"pais":"Marruecos","sistema":"ThunderB/SpyX",   "tipo":"Munición merodeadora",
     "unidades":150, "vcruise_kmh":160, "alcance_km":100,  "payload_kg":10,
     "endurance_h":2, "armado":1, "origen":"Israel",
     "estado":"Operativo (lote 2021+2024)",
     "capacidad_simultan":30,"base_norte":1, "dist_ceuta_km":35.5,
     "dist_melilla_km":7.6,  "nota":"Base Gurugú a 7.6km de Melilla"},
    {"pais":"Marruecos","sistema":"IAI Hermes 900",  "tipo":"MALE ISR/UCAV",
     "unidades":4,   "vcruise_kmh":220, "alcance_km":1000, "payload_kg":300,
     "endurance_h":36, "armado":1, "origen":"Israel",
     "estado":"Operativo",
     "capacidad_simultan":2, "base_norte":1, "dist_ceuta_km":486,
     "dist_melilla_km":486,  "nota":"ISR principal"},
    # Argelia — orientación Sahel/Marruecos, no España
    {"pais":"Argelia",  "sistema":"CASC CH-4",        "tipo":"MALE UCAV",
     "unidades":8,   "vcruise_kmh":150, "alcance_km":345,  "payload_kg":345,
     "endurance_h":12, "armado":1, "origen":"China",
     "estado":"Operativo",
     "capacidad_simultan":4, "base_norte":0, "dist_ceuta_km":435,
     "dist_melilla_km":213,  "nota":"Base Orán → Melilla 213km"},
    {"pais":"Argelia",  "sistema":"WJ-700 Falcon",    "tipo":"HALE UCAV",
     "unidades":4,   "vcruise_kmh":700, "alcance_km":2000, "payload_kg":800,
     "endurance_h":20, "armado":1, "origen":"China",
     "estado":"En pruebas finales (entregado mar.2024)",
     "capacidad_simultan":2, "base_norte":0, "dist_ceuta_km":435,
     "dist_melilla_km":213,  "nota":"Capacidad antibuque C705KD"},
    {"pais":"Argelia",  "sistema":"Anka-S/Aksungur",  "tipo":"MALE UCAV",
     "unidades":16,  "vcruise_kmh":150, "alcance_km":300,  "payload_kg":325,
     "endurance_h":24, "armado":1, "origen":"Turquía",
     "estado":"Pedido confirmado (10+6)",
     "capacidad_simultan":6, "base_norte":0, "dist_ceuta_km":435,
     "dist_melilla_km":213,  "nota":"Primer operador nordafricano Aksungur"},
])

amenaza.to_csv(PROC / "amenaza_uav_regional.csv", index=False)
print(f"  {len(amenaza)} sistemas registrados")
print(f"  Marruecos: {len(amenaza[amenaza['pais']=='Marruecos'])} sistemas")
print(f"  Argelia:   {len(amenaza[amenaza['pais']=='Argelia'])} sistemas")

# Resumen capacidad ofensiva marroquí norte
mar_norte = amenaza[(amenaza["pais"]=="Marruecos") & (amenaza["base_norte"]==1)]
cap_simultan_total = mar_norte["capacidad_simultan"].sum()
print(f"\n  Capacidad lanzamiento simultáneo Marruecos (desde norte):")
print(f"  {cap_simultan_total} vectores simultáneos teóricos")

# ══════════════════════════════════════════════════════════════════════════
# B. DEFENSA ESPAÑOLA — CAPACIDADES Y UMBRALES
# ══════════════════════════════════════════════════════════════════════════
print("\n[B] Modelando defensa española …")

defensa = pd.DataFrame([
    {"plaza":"Ceuta", "sistema":"Mistral II/III",   "tipo":"VSHORAD",
     "lanzadores":12, "misiles_por_lanzador":2, "cadencia_s":45,
     "alcance_km":8,  "altitud_m":6000, "canales_fuego":4,
     "efectividad_uav":0.65, "estado":"Operativo (moderniz. Mistral3 desde 2026)",
     "nota":"RAMIX 30, bat. II/30"},
    {"plaza":"Ceuta", "sistema":"Oerlikon 35/90",   "tipo":"AAA cañón",
     "lanzadores":3,  "misiles_por_lanzador":500, "cadencia_s":2,
     "alcance_km":4,  "altitud_m":3000, "canales_fuego":3,
     "efectividad_uav":0.55, "estado":"Operativo",
     "nota":"RAMIX 30, munición AHEAD"},
    {"plaza":"Ceuta", "sistema":"Cervus III C-UAS",  "tipo":"C-UAS",
     "lanzadores":1,  "misiles_por_lanzador":0, "cadencia_s":1,
     "alcance_km":10, "altitud_m":500,  "canales_fuego":6,
     "efectividad_uav":0.70, "estado":"NO documentado en Ceuta (desplegado OTAN Este)",
     "nota":"Gap identificado"},
    {"plaza":"Melilla","sistema":"Mistral II/III",   "tipo":"VSHORAD",
     "lanzadores":10, "misiles_por_lanzador":2, "cadencia_s":45,
     "alcance_km":8,  "altitud_m":6000, "canales_fuego":4,
     "efectividad_uav":0.65, "estado":"Operativo",
     "nota":"RAMIX 32, bat. II/32"},
    {"plaza":"Melilla","sistema":"Oerlikon 35/90",   "tipo":"AAA cañón",
     "lanzadores":2,  "misiles_por_lanzador":500, "cadencia_s":2,
     "alcance_km":4,  "altitud_m":3000, "canales_fuego":2,
     "efectividad_uav":0.55, "estado":"Operativo",
     "nota":"RAMIX 32"},
    {"plaza":"Melilla","sistema":"Cervus III C-UAS",  "tipo":"C-UAS",
     "lanzadores":1,  "misiles_por_lanzador":0, "cadencia_s":1,
     "alcance_km":10, "altitud_m":500,  "canales_fuego":6,
     "efectividad_uav":0.70, "estado":"NO documentado en Melilla",
     "nota":"Gap crítico — Gurugú a 7.6km"},
])

defensa.to_csv(PROC / "defensa_espana_capacidades.csv", index=False)

# Umbral de saturación por plaza
print("\n  Calculando umbrales de saturación …")

for plaza in ["Ceuta", "Melilla"]:
    d = defensa[defensa["plaza"] == plaza]
    # Canales de fuego simultáneos
    canales = d["canales_fuego"].sum()
    # UAV por ciclo de 60 segundos que puede batir
    ciclo_60s = sum(
        row["canales_fuego"] * (60 / max(row["cadencia_s"], 1))
        for _, row in d.iterrows()
        if row["tipo"] != "C-UAS"
    )
    # Efectividad media ponderada
    ef_media = (d["efectividad_uav"] * d["lanzadores"]).sum() / d["lanzadores"].sum()
    # Umbral: cuando la tasa de llegada supera la tasa de enganche
    tau_low  = int(canales * 0.7)   # saturación parcial (70% canales ocupados)
    tau_high = int(canales * 1.1)   # saturación total
    print(f"\n  {plaza}:")
    print(f"    Canales de fuego totales: {canales}")
    print(f"    UAV batibles en 60s: ~{ciclo_60s:.0f}")
    print(f"    Efectividad media vs UAV: {ef_media*100:.0f}%")
    print(f"    Umbral saturación parcial (τ_low):  ~{tau_low} UAV simultáneos")
    print(f"    Umbral saturación total  (τ_high): ~{tau_high} UAV simultáneos")

# Cargar umbral ucraniano del Script 04 para comparación
tau_ucrania = 41  # τ bajo Script 04
tau_ucrania_alto = 60
print(f"\n  REFERENCIA UCRANIA (Script 04): τ = {tau_ucrania}–{tau_ucrania_alto} UAV/día")
print(f"  NOTA: umbral ucraniano era diario; umbral España es simultáneo (8-12 min ventana)")

# ══════════════════════════════════════════════════════════════════════════
# C. GEOMETRÍA DEL TEATRO
# ══════════════════════════════════════════════════════════════════════════
print("\n[C] Geometría del teatro …")

rutas = pd.DataFrame([
    {"desde":"Monte Gurugú",         "hacia":"Melilla",   "dist_km":7.6,
     "sistema":"ThunderB/SpyX",      "vcruise_kmh":160,   "pais":"Marruecos",
     "amenaza":"CRÍTICA"},
    {"desde":"Tetuán (base TB2)",     "hacia":"Ceuta",     "dist_km":35.5,
     "sistema":"Bayraktar TB2",       "vcruise_kmh":130,   "pais":"Marruecos",
     "amenaza":"ALTA"},
    {"desde":"Tánger",                "hacia":"Tarifa",    "dist_km":33.6,
     "sistema":"Bayraktar TB2",       "vcruise_kmh":130,   "pais":"Marruecos",
     "amenaza":"ALTA"},
    {"desde":"El Araoui/Nador",       "hacia":"Melilla",   "dist_km":81.3,
     "sistema":"Bayraktar TB2",       "vcruise_kmh":130,   "pais":"Marruecos",
     "amenaza":"MEDIA-ALTA"},
    {"desde":"Tánger",                "hacia":"Algeciras", "dist_km":52.7,
     "sistema":"Bayraktar TB2",       "vcruise_kmh":130,   "pais":"Marruecos",
     "amenaza":"MEDIA"},
    {"desde":"Orán",                  "hacia":"Melilla",   "dist_km":213,
     "sistema":"CH-4 / Anka-S",       "vcruise_kmh":150,   "pais":"Argelia",
     "amenaza":"BAJA-MEDIA"},
    {"desde":"Laayoune (Sahara Occ.)","hacia":"Canarias",  "dist_km":420,
     "sistema":"Wing Loong II",       "vcruise_kmh":150,   "pais":"Marruecos",
     "amenaza":"BAJA"},
])

rutas["tiempo_vuelo_min"] = (rutas["dist_km"] / rutas["vcruise_kmh"]) * 60
rutas["ventana_defensa_min"] = rutas["tiempo_vuelo_min"]

# Tiempo de reacción defensa española estimado
T_DETECCION = 2.0    # minutos: alerta radar → confirmación
T_ENGAGEMENT = 1.5   # minutos: confirmación → primer disparo
T_REACCION = T_DETECCION + T_ENGAGEMENT  # = 3.5 min total

rutas["tiempo_util_defensa_min"] = (rutas["tiempo_vuelo_min"] - T_REACCION).clip(lower=0)
rutas["misiles_defensivos_posibles"] = (rutas["tiempo_util_defensa_min"] / (45/60)).astype(int)

print(f"\n  Tiempo de reacción defensa estimado: {T_REACCION} min")
print(f"\n  Rutas críticas:")
print(rutas[["desde","hacia","dist_km","tiempo_vuelo_min",
             "tiempo_util_defensa_min","amenaza"]].to_string(index=False))

# ── Fig 1: Mapa del teatro ────────────────────────────────────────────────
print("\n  Generando figuras …")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_facecolor("#0a0e15")
fig.patch.set_facecolor(BG)

# Coordenadas de los puntos clave
puntos = {
    "Melilla":           (  -2.94, 35.29, "ESP"),
    "Ceuta":             (  -5.31, 35.89, "ESP"),
    "Tarifa":            (  -5.61, 36.01, "ESP"),
    "Algeciras":         (  -5.45, 36.13, "ESP"),
    "Gibraltar":         (  -5.36, 36.14, "GBR"),
    "Tetuán":            (  -5.37, 35.57, "MAR"),
    "Tánger":            (  -5.80, 35.77, "MAR"),
    "Nador/El Araoui":   (  -2.93, 35.16, "MAR"),
    "Monte Gurugú":      (  -2.95, 35.23, "MAR"),
    "Orán":              (  -0.63, 35.69, "ARG"),
}

colors_pais = {"ESP": GREEN, "MAR": RED, "GBR": ACCENT, "ARG": ORANGE}
markers_pais= {"ESP": "^", "MAR": "v", "GBR": "s", "ARG": "D"}

# Dibujar puntos
for nombre, (lon, lat, pais) in puntos.items():
    ax.scatter(lon, lat, c=colors_pais[pais], s=80,
               marker=markers_pais[pais], zorder=5,
               edgecolors=BG, lw=1)
    offset_x = 0.08 if lon > -4 else -0.08
    ha = "left" if lon > -4 else "right"
    ax.text(lon + offset_x, lat + 0.04, nombre,
            color=colors_pais[pais], fontsize=8, ha=ha,
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

# Dibujar rutas de amenaza
ruta_coords = {
    "Monte Gurugú→Melilla":     ((-2.95,35.23),(-2.94,35.29)),
    "Tetuán→Ceuta":             ((-5.37,35.57),(-5.31,35.89)),
    "Tánger→Tarifa":            ((-5.80,35.77),(-5.61,36.01)),
    "Tánger→Algeciras":         ((-5.80,35.77),(-5.45,36.13)),
    "El Araoui→Melilla":        ((-2.93,35.16),(-2.94,35.29)),
    "Orán→Melilla":             ((-0.63,35.69),(-2.94,35.29)),
}

colores_ruta = {
    "Monte Gurugú→Melilla": THREAT,
    "Tetuán→Ceuta":         RED,
    "Tánger→Tarifa":        RED,
    "Tánger→Algeciras":     ORANGE,
    "El Araoui→Melilla":    ORANGE,
    "Orán→Melilla":         YELLOW,
}

for nombre, ((x1,y1),(x2,y2)) in ruta_coords.items():
    r = rutas[
        rutas.apply(lambda r: nombre.split("→")[0].strip() in r["desde"], axis=1)
    ]
    t = r["tiempo_vuelo_min"].values[0] if len(r) else 0
    color = colores_ruta.get(nombre, ACCENT)
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops={"arrowstyle":"->","color":color,
                            "lw":2.5 if "Gurugú" in nombre else 1.8,
                            "alpha":0.85})
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx, my, f"~{t:.0f}min", color=color,
            fontsize=7.5, ha="center",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

# Círculos de alcance defensivo (radio ≈ alcance Mistral en grados)
MISTRAL_DEG = 8 / 111  # 8km en grados
for nombre, (lon, lat, pais) in puntos.items():
    if pais == "ESP" and nombre in ["Ceuta","Melilla"]:
        circ = Circle((lon, lat), MISTRAL_DEG, fill=False,
                      edgecolor=GREEN, lw=1.2, ls="--", alpha=0.5)
        ax.add_patch(circ)
        ax.text(lon, lat - MISTRAL_DEG - 0.06,
                "Cobertura\nMistral 8km",
                color=GREEN, fontsize=6.5, ha="center", alpha=0.7)

# Estrecho
ax.text(-5.5, 35.95, "ESTRECHO\nDE GIBRALTAR",
        color=ACCENT, fontsize=9, ha="center", alpha=0.6,
        style="italic")

# Leyenda
legend_elems = [
    Line2D([0],[0], marker="^", color="w", markerfacecolor=GREEN,
           markersize=8, label="España"),
    Line2D([0],[0], marker="v", color="w", markerfacecolor=RED,
           markersize=8, label="Marruecos (base UAV)"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor=ORANGE,
           markersize=8, label="Argelia"),
    Line2D([0],[0], color=THREAT, lw=2.5, label="Amenaza CRÍTICA (<5min)"),
    Line2D([0],[0], color=RED,    lw=1.8, label="Amenaza ALTA (15-25min)"),
    Line2D([0],[0], color=ORANGE, lw=1.8, label="Amenaza MEDIA (30-60min)"),
    Line2D([0],[0], color=YELLOW, lw=1.5, label="Amenaza BAJA (>60min)"),
    Line2D([0],[0], color=GREEN,  lw=1.2, ls="--", label="Cobertura Mistral 8km"),
]
ax.legend(handles=legend_elems, loc="upper right", fontsize=8,
          facecolor="#161b22", edgecolor=GRID_C)

ax.set_xlim(-7.5, 1.5)
ax.set_ylim(34.5, 37.5)
ax.set_xlabel("Longitud", color=FG)
ax.set_ylabel("Latitud", color=FG)
ax.set_title("Teatro de Operaciones: Estrecho de Gibraltar + Ceuta/Melilla\n"
             "Tiempos de vuelo UAV marroquíes y cobertura AA española (OSINT)",
             color=FG, fontsize=11)
ax.grid(True, alpha=0.12)

plt.tight_layout()
save_fig("10_fig1_mapa_teatro.png")

# ── Fig 2: Inventario amenaza UAV ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("A — Inventario UAV Amenaza Regional: Marruecos + Argelia (OSINT SIPRI/IISS 2025)",
             color=FG, fontsize=11, y=1.01)

ax1, ax2 = axes

# Barras por sistema y país
mar = amenaza[amenaza["pais"]=="Marruecos"].sort_values("unidades", ascending=False)
arg = amenaza[amenaza["pais"]=="Argelia"].sort_values("unidades", ascending=False)

bar_colors_m = [THREAT if r["base_norte"] else ORANGE
                for _, r in mar.iterrows()]
b = ax1.barh(mar["sistema"][::-1], mar["unidades"][::-1],
             color=list(reversed(bar_colors_m)), alpha=0.85, edgecolor=BG)
for bar, (_, row) in zip(b, mar[::-1].iterrows()):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{row['vcruise_kmh']}km/h · {row['alcance_km']}km",
             va="center", fontsize=8, color=FG)
ax1.set_xlabel("Unidades estimadas (OSINT)", color=FG)
ax1.set_title("MARRUECOS — Rojo oscuro = base norte (proximidad España)",
              color=FG, fontsize=9)
ax1.grid(True, alpha=0.3, axis="x")
ax1.set_facecolor(BG)

bar_colors_a = [ORANGE for _ in arg.iterrows()]
b2 = ax2.barh(arg["sistema"][::-1], arg["unidades"][::-1],
              color=bar_colors_a, alpha=0.85, edgecolor=BG)
for bar, (_, row) in zip(b2, arg[::-1].iterrows()):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f"{row['vcruise_kmh']}km/h · {row['alcance_km']}km",
             va="center", fontsize=8, color=FG)
ax2.set_xlabel("Unidades estimadas (OSINT)", color=FG)
ax2.set_title("ARGELIA — Orientación Sahel/Marruecos (amenaza contingente)",
              color=FG, fontsize=9)
ax2.grid(True, alpha=0.3, axis="x")
ax2.set_facecolor(BG)

plt.tight_layout()
save_fig("10_fig2_inventario_amenaza.png")

# ── Fig 3: Tiempos vuelo vs ventana defensiva ─────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

rutas_sorted = rutas.sort_values("tiempo_vuelo_min")
colores_r    = [THREAT if t < 5 else
                (RED    if t < 20 else
                 (ORANGE if t < 40 else YELLOW))
                for t in rutas_sorted["tiempo_vuelo_min"]]

labels_r = [f"{r['desde'].split('(')[0].strip()}\n→{r['hacia']}\n{r['sistema']}"
            for _, r in rutas_sorted.iterrows()]

bars = ax.bar(range(len(rutas_sorted)), rutas_sorted["tiempo_vuelo_min"],
              color=colores_r, alpha=0.85, edgecolor=BG, lw=0.3)

# Tiempo de reacción defensiva
ax.axhline(T_REACCION, color=GREEN, lw=2, ls="--",
           label=f"Tiempo reacción defensiva: {T_REACCION:.1f} min")
ax.axhline(8, color=YELLOW, lw=1.5, ls=":",
           label="8 min: tiempo mín. para 1 salva Mistral")
ax.axhspan(0, T_REACCION, alpha=0.08, color=THREAT,
           label="Zona sin respuesta posible")

for bar, (_, row) in zip(bars, rutas_sorted.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f"{row['tiempo_vuelo_min']:.0f}min",
            ha="center", va="bottom", fontsize=9, color=FG)

ax.set_xticks(range(len(rutas_sorted)))
ax.set_xticklabels(labels_r, fontsize=8)
ax.set_ylabel("Tiempo de vuelo hasta objetivo (minutos)", color=FG)
ax.set_title("B — Ventana Defensiva: Tiempo de Vuelo vs Tiempo de Reacción Española\n"
             "Cuanto más bajo → menos tiempo tiene la defensa para actuar",
             color=FG, fontsize=10)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3, axis="y")
ax.set_facecolor(BG)

plt.tight_layout()
save_fig("10_fig3_ventana_defensiva.png")

# ══════════════════════════════════════════════════════════════════════════
# D. LECCIONES UCRANIA → ESPAÑA
# ══════════════════════════════════════════════════════════════════════════
print("\n[D] Transfiriendo lecciones ucranianas …")

lecciones = [
    # (id, lección_Ucrania, valor_Ucrania, aplicación_España, valor_España, nivel_riesgo)
    ("L1", "Umbral saturación defensiva",
     "τ=41-60 UAV/día → tasa hit sube a 50%",
     "Ceuta: τ~15-20 UAV simultáneos\nMelilla: τ~12-16 UAV simultáneos",
     "Marruecos puede lanzar 30-40 simultáneos desde norte",
     "CRÍTICO"),
    ("L2", "Ventana temporal ataque nocturno",
     "18:00-20:00 lanzamiento → 08:00-09:00 impactos\n13.2h ventana defensiva",
     "Tetuán→Ceuta: 16 min TB2\nGurugú→Melilla: 3 min loitering",
     "Ventana defensiva colapsada a minutos, no horas",
     "CRÍTICO"),
    ("L3", "Zona lanzamiento fija (95-100%)",
     "Primorsko-Akhtarsk inamovible\n→ Strike preventivo viable en escenario bélico",
     "Base Tetuán + Monte Gurugú identificadas\nCoordenadas OSINT disponibles",
     "Posible targeting preventivo (marco legal OTAN)",
     "ALTO"),
    ("L4", "Inyección externa cambia el régimen",
     "Agosto 2025: hardware OTAN → intercepción 43%→84.7%\nCambio estructural en 1 semana",
     "Cervus III / CROW no documentados en plazas\nNASAMS en Báltico, no en España",
     "Gap C-UAS: sin inyección externa la defensa\nno aguantaría campaña sostenida",
     "ALTO"),
    ("L5", "Adaptación timing adversario",
     "-0.12h/mes deriva lanzamiento\n→ adversario ajusta para evitar turnos defensivos",
     "Con solo 16 min de vuelo, 0.12h/mes\n= ajuste de 7 min en 1 mes = indetectable",
     "Monitoreo continuo patrones horarios\nes esencial desde el primer incidente",
     "MEDIO"),
    ("L6", "Represalia UA + 261% post-ola masiva",
     "Belgorod = 82% de ataques de respuesta UA\nCiclo acción-reacción sistemático",
     "Respuesta española: solo defensiva\nsin capacidad strike transfronterizo",
     "Asimetría: Marruecos puede escalar,\nEspaña solo puede absorber o disuadir",
     "ALTO"),
    ("L7", "Degradación acumulada de infraestructura",
     "4 GW destruidos en fase saturación (C3)\n→ déficit estructural persiste aunque intercepción mejore",
     "Ceuta 18km²: toda infraestructura crítica\nen radio de 10km desde costa marroquí",
     "Sin profundidad estratégica: un ataque\nsostenido de 48-72h destruye infraestructura clave",
     "CRÍTICO"),
]

print(f"  {len(lecciones)} lecciones transferidas")
riesgo_counts = {}
for l in lecciones:
    r = l[5]
    riesgo_counts[r] = riesgo_counts.get(r, 0) + 1
print(f"  Nivel riesgo: {riesgo_counts}")

# ── Fig 4: Matriz de lecciones ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 9))
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)
ax.axis("off")

ax.set_title("D — Transferencia de Lecciones: Ucrania (2025-2026) → España\n"
             "Aplicación al teatro Ceuta/Melilla/Estrecho",
             color=FG, fontsize=12, pad=15)

col_colors = {"CRÍTICO": THREAT, "ALTO": RED, "MEDIO": YELLOW, "BAJO": GREEN}
col_headers = ["ID", "Lección\n(Ucrania)", "Evidencia\nEmpírica", "Aplicación\nEspaña", "Riesgo"]
col_widths   = [0.04, 0.22, 0.22, 0.25, 0.07]

# Encabezados
y = 0.96
x_starts = [0.01, 0.06, 0.30, 0.54, 0.82]
for hdr, xs, w in zip(col_headers, x_starts, col_widths):
    ax.text(xs + w/2, y, hdr, color=ACCENT, fontsize=9,
            fontweight="bold", ha="center", va="top",
            transform=ax.transAxes)

y = 0.91
row_h = 0.116
for lid, luc, val_uc, lap, val_sp, riesgo in lecciones:
    color_r = col_colors.get(riesgo, FG)
    # Fondo de fila
    fbox = FancyBboxPatch((0.01, y - row_h + 0.008), 0.97, row_h - 0.012,
                          boxstyle="round,pad=0.005",
                          facecolor=color_r + "11",
                          edgecolor=color_r + "44", lw=0.8,
                          transform=ax.transAxes)
    ax.add_patch(fbox)
    # ID
    ax.text(x_starts[0] + col_widths[0]/2, y - row_h/2,
            lid, color=color_r, fontsize=10, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    # Lección Ucrania
    ax.text(x_starts[1], y - 0.01, luc,
            color=FG, fontsize=7.5, va="top", transform=ax.transAxes,
            wrap=True)
    # Valor empírico
    ax.text(x_starts[2], y - 0.01, val_uc,
            color=YELLOW, fontsize=7, va="top", transform=ax.transAxes,
            fontfamily="monospace")
    # Aplicación España
    ax.text(x_starts[3], y - 0.01, lap,
            color=ACCENT, fontsize=7, va="top", transform=ax.transAxes)
    # Riesgo badge
    ax.text(x_starts[4] + col_widths[4]/2, y - row_h/2,
            riesgo, color=BG, fontsize=7.5, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes,
            bbox={"boxstyle":"round,pad=0.3", "facecolor": color_r,
                  "alpha": 0.9, "edgecolor":"none"})
    y -= row_h

plt.tight_layout()
save_fig("10_fig4_lecciones_ucrania_espana.png")

# ══════════════════════════════════════════════════════════════════════════
# E. MODELO DE SATURACIÓN ADAPTADO AL TEATRO ESPAÑOL
# ══════════════════════════════════════════════════════════════════════════
print("\n[E] Modelo de saturación teatro español …")

# Basado en TAR/SETAR del Script 04, adaptado a ventana de 8-16 min
# En vez de días, trabajamos en "salvas" (cada ~3 min un ciclo defensivo)

n_sim = 10000
np.random.seed(42)

# Parámetros defensivos Ceuta
canales_ceuta   = 19  # suma canales_fuego (tabla defensa)
ef_ceuta        = 0.62
tau_ceuta       = 15  # umbral saturación parcial simultánea

# Parámetros defensivos Melilla
canales_melilla = 14
ef_melilla      = 0.60
tau_melilla     = 12  # umbral saturación parcial

# Capacidad ofensiva marroquí (estimación conservadora)
# TB2 x20 + loitering x40: lanzamiento escalonado en 30 min
ola_small   = np.random.poisson(10,  n_sim)  # ataque menor
ola_medium  = np.random.poisson(25,  n_sim)  # ataque coordinado
ola_large   = np.random.poisson(45,  n_sim)  # ataque masivo TB2 + loitering
ola_saturat = np.random.poisson(80,  n_sim)  # saturación total

def tasa_intercep_espana(n_uav, canales, ef_base, tau):
    """
    Modelo TAR adaptado: si n_uav < tau, defensa opera en régimen normal.
    Si n_uav >= tau, defensa se satura y efectividad cae.
    Inspirado en Script 04 (τ ucraniano) pero calibrado para
    ventana temporal colapsada (minutos, no horas).
    """
    ef = np.where(
        n_uav < tau,
        ef_base + (tau - n_uav) * 0.005,  # mejora marginal con volumen bajo
        ef_base * (tau / n_uav) ** 0.7    # degradación por saturación
    )
    return np.clip(ef, 0.05, 0.98)

for plaza, canales, ef, tau, olas_list in [
    ("Ceuta",   canales_ceuta,   ef_ceuta,   tau_ceuta,
     [ola_small, ola_medium, ola_large, ola_saturat]),
    ("Melilla", canales_melilla, ef_melilla, tau_melilla,
     [ola_small, ola_medium, ola_large, ola_saturat]),
]:
    print(f"\n  {plaza} (τ={tau}, canales={canales}):")
    for ola, nombre in zip(olas_list,
                           ["Ataque menor (~10)", "Coordinado (~25)",
                            "Masivo (~45)","Saturación (~80)"]):
        ef_sim = tasa_intercep_espana(ola, canales, ef, tau)
        impactos = ola * (1 - ef_sim)
        print(f"    {nombre:22s}: intercep={ef_sim.mean()*100:.1f}%  "
              f"impactos={impactos.mean():.1f}  "
              f"P(>10 impactos)={100*(impactos>10).mean():.0f}%")

# ── Fig 5: Curva saturación España vs Ucrania ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("E — Modelo de Saturación: España vs Ucrania\n"
             "Transferencia del marco TAR/SETAR (Script 04) al teatro Estrecho",
             color=FG, fontsize=11, y=1.01)

ax1, ax2 = axes

n_uav_range = np.arange(1, 101)

# Curvas para distintas configuraciones
configs = [
    ("España — Ceuta (actual)",       canales_ceuta,  ef_ceuta,  tau_ceuta,   RED),
    ("España — Ceuta (con C-UAS)",    canales_ceuta+8,0.78,      tau_ceuta+10,GREEN),
    ("España — Melilla (actual)",     canales_melilla,ef_melilla,tau_melilla,  ORANGE),
    ("España — Melilla (con C-UAS)",  canales_melilla+8,0.76,    tau_melilla+8,ACCENT),
]

for label, ch, ef, tau, color in configs:
    ef_curve = tasa_intercep_espana(n_uav_range, ch, ef, tau)
    ax1.plot(n_uav_range, ef_curve * 100, color=color, lw=2,
             label=label, alpha=0.9)

# Referencia Ucrania C1 y C3
ax1.axhline(84.7, color=FG,    lw=1.5, ls="--", alpha=0.5,
            label="Ucrania C1 post-OTAN (84.7%)")
ax1.axhline(49.6, color=PURPLE, lw=1.5, ls="--", alpha=0.5,
            label="Ucrania C3 saturación (49.6%)")
ax1.axvline(tau_ceuta,    color=RED,    lw=1, ls=":", alpha=0.6)
ax1.axvline(tau_melilla,  color=ORANGE, lw=1, ls=":", alpha=0.6)
ax1.text(tau_ceuta + 0.5, 85, f"τ Ceuta={tau_ceuta}", color=RED, fontsize=8)
ax1.text(tau_melilla+0.5, 79, f"τ Melilla={tau_melilla}", color=ORANGE, fontsize=8)

# Capacidad ofensiva marroquí
ax1.axvspan(8, 15,  alpha=0.08, color=YELLOW, label="Rango ola pequeña MA")
ax1.axvspan(20, 35, alpha=0.08, color=ORANGE, label="Rango ola coordinada MA")
ax1.axvspan(40, 60, alpha=0.08, color=RED,    label="Rango ola masiva MA")

ax1.set_xlabel("UAV simultáneos en el teatro", color=FG)
ax1.set_ylabel("Tasa de intercepción estimada (%)", color=FG)
ax1.set_title("Curva de saturación por configuración defensiva",
              color=FG, fontsize=10)
ax1.legend(fontsize=7, loc="upper right")
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor(BG)

# Impactos esperados por escenario
escenarios = ["Ola\npequeña\n(~10)", "Coordinada\n(~25)",
              "Masiva\n(~45)", "Saturación\n(~80)"]
olas_medias = [10, 25, 45, 80]

for plaza, canales, ef, tau, color, linestyle in [
    ("Ceuta (actual)",      canales_ceuta,  ef_ceuta,  tau_ceuta,  RED,   "-"),
    ("Ceuta (C-UAS)",       canales_ceuta+8,0.78,      tau_ceuta+10,GREEN,"-"),
    ("Melilla (actual)",    canales_melilla,ef_melilla,tau_melilla,ORANGE,"-"),
    ("Melilla (C-UAS)",     canales_melilla+8,0.76,    tau_melilla+8,ACCENT,"-"),
]:
    impactos = [n * (1 - tasa_intercep_espana(
        np.array([n]), canales, ef, tau).mean())
        for n in olas_medias]
    ax2.plot(escenarios, impactos, color=color, lw=2,
             marker="o", ms=6, label=plaza)

ax2.set_ylabel("Impactos esperados", color=FG)
ax2.set_title("Impactos por escenario y configuración defensiva",
              color=FG, fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor(BG)
ax2.axhspan(0, 5,   alpha=0.06, color=GREEN, label="Daño asumible")
ax2.axhspan(5, 15,  alpha=0.06, color=YELLOW)
ax2.axhspan(15, 60, alpha=0.06, color=RED,   label="Daño severo")

plt.tight_layout()
save_fig("10_fig5_modelo_saturacion.png")

# ── Fig 6: Timeline defensivo Ceuta (16 min TB2) ──────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
ax.set_xlim(0, 20); ax.set_ylim(-0.5, 3)
ax.set_yticks([]); ax.set_xlabel("Minutos desde lanzamiento", color=FG)
ax.set_title("Timeline Defensivo — Escenario TB2 Tetuán → Ceuta (35.5 km, 16 min)\n"
             "Comparado con Ucrania: Shahed Primorsko-Akhtarsk → Kyiv (13.2h)",
             color=FG, fontsize=10)
ax.grid(True, alpha=0.2, axis="x")

# Fases España
fases_esp = [
    (0,   1.5,  "#1a3a1a", "Lanzamiento\ndetección radar"),
    (1.5, 3.5,  "#2a2a1a", "Confirmación\nthreat ID"),
    (3.5, 5.5,  "#3a1a1a", "Engagement\n1ª salva Mistral"),
    (5.5, 10,   "#4a1a1a", "Engagement\n2ª-4ª salva"),
    (10,  14,   "#5a0a0a", "Saturación\ndefensiva probable"),
    (14,  16,   "#7a0000", "IMPACTO\nCeuta"),
]
for x1, x2, col, label in fases_esp:
    ax.barh(2, x2-x1, left=x1, height=0.5, color=col, edgecolor=GRID_C, lw=0.5)
    ax.text((x1+x2)/2, 2, label, ha="center", va="center",
            fontsize=7, color=FG, fontweight="bold")
ax.text(-0.1, 2.35, "CEUTA (TB2, 16 min)", color=RED, fontsize=8,
        ha="left", fontweight="bold")

# Fases Ucrania (comprimidas a escala logarítmica para visualización)
fases_ua = [
    (0, 3,   "#0a1a3a", "Lanzamiento"),
    (3, 7,   "#0a2a3a", "Detección\n(min. 2-3h)"),
    (7, 12,  "#0a3a3a", "Intercepción\nsostenida"),
    (12, 16, "#0a4a2a", "Defensa\nmadura"),
]
for x1, x2, col, label in fases_ua:
    ax.barh(0.8, x2-x1, left=x1, height=0.5, color=col, edgecolor=GRID_C, lw=0.5)
    ax.text((x1+x2)/2, 0.8, label, ha="center", va="center",
            fontsize=7, color=FG)
ax.text(-0.1, 1.15, "UCRANIA (Shahed, 13.2h)", color=GREEN, fontsize=8,
        ha="left", fontweight="bold")
ax.text(16.5, 0.8, "→ 13.2h real\n(escala comprimida)", color=FG,
        fontsize=7, alpha=0.6)

# Anotación crítica
ax.axvline(T_REACCION, color=GREEN, lw=2, ls="--", alpha=0.8)
ax.text(T_REACCION + 0.1, 2.8, f"T reacción = {T_REACCION} min",
        color=GREEN, fontsize=8)
ax.axvline(8, color=YELLOW, lw=1.5, ls=":", alpha=0.7)
ax.text(8.1, 2.8, "1ª salva\ncompleta", color=YELLOW, fontsize=7)

plt.tight_layout()
save_fig("10_fig6_timeline_defensivo.png")

# ── Fig 7: Resumen ejecutivo de inteligencia ─────────────────────────────
print("\n  Generando resumen ejecutivo …")
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

# Panel 1: Mapa compacto
ax_m = fig.add_subplot(gs[0, :2])
ax_m.set_facecolor("#0a0e15")
for nombre, (lon, lat, pais) in puntos.items():
    ax_m.scatter(lon, lat, c=colors_pais[pais], s=50,
                 marker=markers_pais[pais], zorder=5, edgecolors=BG, lw=0.8)
    ax_m.text(lon+0.06, lat+0.03, nombre.split("(")[0][:10],
              color=colors_pais[pais], fontsize=6.5)
for nombre, ((x1,y1),(x2,y2)) in ruta_coords.items():
    color = colores_ruta.get(nombre, ACCENT)
    ax_m.annotate("", xy=(x2,y2), xytext=(x1,y1),
                  arrowprops={"arrowstyle":"->","color":color,"lw":2,"alpha":0.8})
ax_m.set_xlim(-7, 1); ax_m.set_ylim(34.6, 37.2)
ax_m.set_title("Teatro Estrecho", color=FG, fontsize=9)
ax_m.grid(True, alpha=0.1)
ax_m.tick_params(labelsize=7)

# Panel 2: Curva saturación compacta
ax_s = fig.add_subplot(gs[0, 2:])
for label, ch, ef, tau, color in configs[:2]:
    ef_c = tasa_intercep_espana(n_uav_range, ch, ef, tau)
    ax_s.plot(n_uav_range, ef_c*100, color=color, lw=2, label=label.replace("España — ",""))
ax_s.axhline(84.7, color=FG, lw=1, ls="--", alpha=0.4, label="UA post-OTAN 84.7%")
ax_s.axhline(49.6, color=PURPLE, lw=1, ls="--", alpha=0.4, label="UA saturación 49.6%")
ax_s.axvspan(40,60, alpha=0.1, color=RED, label="Rango ola masiva MA")
ax_s.set_xlabel("UAV simultáneos", color=FG, fontsize=8)
ax_s.set_ylabel("% Intercepción", color=FG, fontsize=8)
ax_s.set_title("Curva saturación Ceuta", color=FG, fontsize=9)
ax_s.legend(fontsize=6.5); ax_s.grid(True, alpha=0.3); ax_s.set_facecolor(BG)

# Panel 3: KPIs ejecutivos
ax_k = fig.add_subplot(gs[1, :])
ax_k.axis("off"); ax_k.set_facecolor(BG)

kpi_data = [
    ("AMENAZA", THREAT,
     [("UAV combate MAR (norte)", "~43 vectores simultáneos teóricos"),
      ("Más próximo", "Gurugú→Melilla: 7.6 km / ~3 min loitering"),
      ("TB2 Tetuán→Ceuta", "35.5 km / 16 min"),
      ("Experiencia combate", "4+ años operaciones Sahara Occidental")]),
    ("DEFENSA", GREEN,
     [("Ceuta: canales fuego", f"{canales_ceuta} (Mistral+Oerlikon)"),
      ("Ceuta: τ saturación", f"~{tau_ceuta}-20 UAV simultáneos"),
      ("Melilla: τ saturación", f"~{tau_melilla}-16 UAV simultáneos"),
      ("Gap crítico", "Sin C-UAS dedicado en las plazas")]),
    ("LECCIONES UA", YELLOW,
     [("Umbral saturación ucraniano", "τ=41-60 UAV/día → aplica en minutos aquí"),
      ("Inyección C-UAS OTAN", "+35% intercepción en 1 semana"),
      ("Zona fija lanzamiento", "Primorsko≡Tetuán: targeting preventivo viable"),
      ("Ventana defensiva", "UA 13.2h → España 8-16 min (×50 menos)")]),
]

col_start = [0.01, 0.34, 0.67]
for i, (titulo, color, items) in enumerate(kpi_data):
    xs = col_start[i]
    ax_k.text(xs, 0.98, titulo, color=color, fontsize=11, fontweight="bold",
              transform=ax_k.transAxes, va="top")
    y = 0.82
    for k, v in items:
        ax_k.text(xs, y, f"▸ {k}", color=FG, fontsize=8,
                  transform=ax_k.transAxes, va="top")
        ax_k.text(xs+0.01, y-0.12, f"   {v}", color=color, fontsize=8,
                  fontweight="bold", transform=ax_k.transAxes, va="top",
                  fontfamily="monospace")
        y -= 0.24

# Panel 4: Semáforo de riesgo por lección
ax_sem = fig.add_subplot(gs[2, :])
ax_sem.axis("off"); ax_sem.set_facecolor(BG)
ax_sem.set_title("Semáforo de Riesgo — Lecciones Ucrania aplicadas a España",
                 color=FG, fontsize=10, pad=5)

col_riesgo = {"CRÍTICO": THREAT, "ALTO": RED, "MEDIO": YELLOW, "BAJO": GREEN}
x_pos = np.linspace(0.05, 0.95, len(lecciones))
for i, (lid, luc, val_uc, lap, val_sp, riesgo) in enumerate(lecciones):
    color_l = col_riesgo[riesgo]
    circ = Circle((x_pos[i], 0.65), 0.045,
                  facecolor=color_l, edgecolor=BG, lw=1.5,
                  transform=ax_sem.transAxes)
    ax_sem.add_patch(circ)
    ax_sem.text(x_pos[i], 0.65, lid, color=BG, fontsize=9,
                fontweight="bold", ha="center", va="center",
                transform=ax_sem.transAxes)
    ax_sem.text(x_pos[i], 0.42, riesgo, color=color_l, fontsize=7,
                ha="center", va="center", transform=ax_sem.transAxes)
    # Descripción corta
    short = luc[:25] + ("…" if len(luc)>25 else "")
    ax_sem.text(x_pos[i], 0.22, short, color=FG, fontsize=6,
                ha="center", va="center", transform=ax_sem.transAxes,
                wrap=True)

fig.suptitle("Script 10 — Inteligencia Operacional: Amenaza UAV Regional → España\n"
             "Basado en análisis empírico Ucrania 2025-2026 (Scripts 03-09) + OSINT",
             color=FG, fontsize=12, y=1.01)
plt.tight_layout()
save_fig("10_fig7_resumen_ejecutivo.png")

# ══════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RESUMEN SCRIPT 10")
print("="*60)

print(f"""
AMENAZA MARRUECOS (OSINT SIPRI/IISS 2025)
  UAV combate confirmados: ~43 (TB2×23 + Akinci×3 + Wing Loong×4 + loitering×150)
  Capacidad simultánea (norte): ~43 vectores
  Base más próxima a Melilla: Monte Gurugú, 7.6 km, tiempo vuelo ~3 min
  Base más próxima a Ceuta:   Tetuán, 35.5 km, TB2 16 min
  Experiencia combate: 4+ años Sahara Occidental (Wing Loong, TB2)
  Producción local (Atlas Defence): potencial 1.000 drones/año

DEFENSA ESPAÑOLA ACTUAL (OSINT)
  Ceuta:   {canales_ceuta} canales de fuego  τ~{tau_ceuta}-20 UAV simultáneos
  Melilla: {canales_melilla} canales de fuego  τ~{tau_melilla}-16 UAV simultáneos
  GAP CRÍTICO: sin C-UAS dedicado en las plazas
  GAP CRÍTICO: NASAMS en Báltico, no en España
  GAP CRÍTICO: Hawk PIP III donado a Ucrania (2 baterías)

LECCIONES UCRANIA APLICADAS
  L1 CRÍTICO: τ ucraniano (41-60/día) → τ España (12-20 simultáneos)
              Marruecos PUEDE superar τ desde el primer ataque
  L2 CRÍTICO: Ventana defensiva colapsada de 13.2h → 3-16 min
  L4 ALTO:    Inyección C-UAS OTAN = +35% intercepción en 1 semana
              España no tiene ese recurso actualmente en las plazas
  L7 CRÍTICO: Sin profundidad estratégica — toda infraestructura crítica
              de Ceuta dentro de radio de 10 km desde costa marroquí

  Figuras generadas: {FIGS_OK}/7
  [DONE]
""")
