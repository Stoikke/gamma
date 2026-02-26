"""
Pipeline COMPLET : gtselect (par bin énergie) + gtbin (LC) + gtexposure.
"""

import os
import subprocess
import argparse

# ==============================================================================
# 0. PARAMÈTRES PAR DÉFAUT
# ==============================================================================

DEFAULT_RA     = 324  #  valeur pour la source 338.1849
DEFAULT_DEC    = -3 # valeur pour la source 11.718641
DEFAULT_RAD    = 10 # valeurs pour la source 4
DEFAULT_ZMAX   = 90.0
INPUT_FITS     = "data/Photon_projet/lat_photon_weekly_all.fits"
FT2_FILE       = "data/Photon_projet/spacecraft_projet/L2602050555366B2DD36C49_SC00.fits"
RESULTS_DIR    = "results_simple"
OUTPUT_BASE_DIR = "SED_output"

TIME_INTERVALS = [
    (5.03377224e+08, 5.03593224e+08, "activite_1"),
]

BIN_WIDTH_SEC = 10800
PHOTON_INDEX = -2.1
IRFS         = "CALDB"
SRC_MODEL    = "none"

ENERGY_BINS = [
    ( 100.0,    169.86),
    ( 169.86,   288.54),
    ( 288.54,   490.13),
    ( 490.13,   832.5),
    ( 832.5,   1414.21),
    (1414.21,  2402.25),
    (2402.25,  4080.57),
    (4080.57,  6931.45),
    (6931.45, 11774.08),
    (11774.08, 20000.0),
]

# ==============================================================================
# 1. UTILITAIRES
# ==============================================================================

def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def run_cmd(cmd: list, dry_run: bool = False):
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"  >>> {cmd_str}")
    if not dry_run:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [ERREUR] {result.stderr.strip()}")
        else:
            print(f"  [OK]")
        return result
    return None

def fmt(val: float) -> str:
    return f"{val:.2f}".rstrip("0").rstrip(".")

# ==============================================================================
# 2. GTSELECT PAR BIN ÉNERGIE
# ==============================================================================

def run_gtselect(ra, dec, rad, dry_run: bool = False):
    """Un fichier gtselect par bin énergie × période → results_simple/"""
    make_dirs(RESULTS_DIR)

    # dictionnaire : (tlabel, emin, emax) → chemin fichier gtselect
    gtselect_files = {}

    print("\n" + "="*70)
    print("GTSELECT : sélection photons par bin énergie")
    print("="*70)

    for tstart, tstop, tlabel in TIME_INTERVALS:
        for emin, emax in ENERGY_BINS:
            out_name = f"selected_region_fond_3_{fmt(emin)}_{fmt(emax)}.fits" # attention a bien prendre le bon nom
            out_path = os.path.join(RESULTS_DIR, out_name)

            print(f"  [{emin} – {emax}] MeV")

            run_cmd([
                "gtselect",
                f"infile={INPUT_FITS}",
                f"outfile={out_path}",
                f"ra={ra}", f"dec={dec}", f"rad={rad}",
                f"tmin={tstart}", f"tmax={tstop}",
                f"emin={emin}", f"emax={emax}",
                f"zmax={DEFAULT_ZMAX}",
            ], dry_run=dry_run)

            # clé = tuple, pas une string → pas de problème de split
            gtselect_files[(tlabel, emin, emax)] = out_path

    print(f"\nFichiers gtselect dans : {RESULTS_DIR}/")
    return gtselect_files

# ==============================================================================
# 3. GTBIN + GTEXPOSURE
# ==============================================================================

def build_lc_and_exposure(gtselect_files: dict, prefix: str, dry_run: bool = False):
    gtbin_dir = os.path.join(OUTPUT_BASE_DIR, "gtbin")
    make_dirs(gtbin_dir)

    for (tlabel, emin, emax), ft1_file in gtselect_files.items():
        tstart, tstop, _ = next(t for t in TIME_INTERVALS if t[2] == tlabel)

        lc_name = f"{prefix}_fond_3_{fmt(emin)}_{fmt(emax)}.fits" #  modifier lorsque on fait le fond ou la source
        lc_path = os.path.join(gtbin_dir, lc_name)

        print(f"\n{'='*50}")
        print(f"  Bin : [{emin} – {emax}] MeV  →  {lc_name}")
        print(f"{'='*50}")

        # ── gtbin ─────────────────────────────────────────────────────────────
        print("  → gtbin (LC)")
        run_cmd([
            "gtbin",
            f"evfile={ft1_file}",
            f"outfile={lc_path}",
            f"scfile={FT2_FILE}",
            "algorithm=LC",
            "tbinalg=LIN",
            f"tstart={tstart}",
            f"tstop={tstop}",
            f"dtime={BIN_WIDTH_SEC}",
        ], dry_run=dry_run)

        # ── gtexposure ────────────────────────────────────────────────────────
        print("  → gtexposure (EXPOSURE in-place)")
        run_cmd([
            "gtexposure",
            f"infile={lc_path}",
            f"scfile={FT2_FILE}",
            f"irfs={IRFS}",
            f"srcmdl={SRC_MODEL}",
            f"specin={PHOTON_INDEX}",
        ], dry_run=dry_run)

    print(f"\nTerminé. Fichiers dans : {gtbin_dir}/")

# ==============================================================================
# 4. PARSER
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline GTSELECT+GTBIN+GTEXPOSURE")
    parser.add_argument("--ra",      type=float, default=DEFAULT_RA,  help="RA [degr]")
    parser.add_argument("--dec",     type=float, default=DEFAULT_DEC, help="DEC [degr]")
    parser.add_argument("--rad",     type=float, default=DEFAULT_RAD, help="Rayon ROI [degr]")
    parser.add_argument("--prefix",  default="gt_bin_activite_1",     help="Prefix fichiers gtbin")
    parser.add_argument("--dry-run", action="store_true",             help="Mode test sans exécution")

    args = parser.parse_args()

    print(f"CONFIG : RA={args.ra}, DEC={args.dec}, RAD={args.rad}° | Prefix={args.prefix}")

    # Étape 1 : gtselect par bin énergie
    gtselect_files = run_gtselect(args.ra, args.dec, args.rad, args.dry_run)

    # Étape 2 : gtbin + gtexposure
    build_lc_and_exposure(gtselect_files, args.prefix, args.dry_run)
