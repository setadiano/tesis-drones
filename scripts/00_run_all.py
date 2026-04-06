"""
Script 00 — Runner maestro
==========================
Ejecuta todos los scripts en orden y reporta estado.

Uso:
    python scripts/00_run_all.py
    python scripts/00_run_all.py --desde 05   # empezar desde script 05
    python scripts/00_run_all.py --solo 07 08  # solo scripts específicos
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

PIPELINE = [
    ("02", "02_procesar_acled.py",       "Procesamiento ACLED"),
    ("03", "03_series_temporales.py",    "Series temporales (ARIMA/VAR/Markov)"),
    ("04", "04_umbral_saturacion.py",    "Umbral saturación (TAR/SETAR)"),
    ("05", "05_analisis_multivariante.py","PCA / MANOVA / Clusters"),
    ("06", "06_retroalimentacion_tactica.py", "Retroalimentación táctica (RL/Granger)"),
    ("07", "07_variables_externas.py",   "Variables externas (meteo + red eléctrica)"),
    ("08", "08_analisis_horario.py",     "Análisis horario / ventanas vulnerabilidad"),
    ("09", "09_doctrina_combinada.py",    "Doctrina combinada (ACLED + presión táctica)"),
    ("10", "10_teatro_espana.py",         "Teatro España: saturación + amenaza regional"),
]

def run_script(num, filename, descripcion):
    path = SCRIPTS / filename
    print(f"\n{'='*60}")
    print(f"  [{num}] {descripcion}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
        capture_output=False,   # deja que el output fluya en tiempo real
    )
    elapsed = time.time() - t0
    status = "✓ OK" if result.returncode == 0 else "✗ FALLO"
    return result.returncode, elapsed, status

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desde", type=str, default=None,
                        help="Empezar desde script número (ej: --desde 05)")
    parser.add_argument("--solo", nargs="+", default=None,
                        help="Ejecutar solo estos números (ej: --solo 07 08)")
    args = parser.parse_args()

    # Filtrar pipeline
    pipeline = PIPELINE
    if args.solo:
        pipeline = [(n,f,d) for n,f,d in PIPELINE if n in args.solo]
    elif args.desde:
        pipeline = [(n,f,d) for n,f,d in PIPELINE if n >= args.desde]

    print(f"\n{'#'*60}")
    print(f"  TESIS DRONES — PIPELINE COMPLETO")
    print(f"  Scripts a ejecutar: {len(pipeline)}")
    print(f"{'#'*60}")

    resumen = []
    t_total = time.time()

    for num, filename, descripcion in pipeline:
        code, elapsed, status = run_script(num, filename, descripcion)
        resumen.append((num, descripcion, status, f"{elapsed:.0f}s"))

    # Resumen final
    print(f"\n{'#'*60}")
    print("  RESUMEN FINAL")
    print(f"{'#'*60}")
    for num, desc, status, t in resumen:
        color = "" 
        print(f"  [{num}] {status}  {t:>5}  {desc}")

    total = time.time() - t_total
    fallos = sum(1 for _,_,s,_ in resumen if "FALLO" in s)
    print(f"\n  Total: {total:.0f}s  |  OK: {len(resumen)-fallos}  |  Fallos: {fallos}")

    if fallos:
        sys.exit(1)

if __name__ == "__main__":
    main()
