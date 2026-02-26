#!/usr/bin/env python3
"""
Pipeline gtbin (LC) + gtexposure pour la SED d'un GRB.
gtexposure ajoute la colonne EXPOSURE directement dans le fichier gtbin (in-place).
"""

import os
import subprocess

# ==============================================================================
# 1. CONFIGURATION MANUELLE
# ==============================================================================

FT1_FILE        = "results_simple/fond_3.fits" #selected_region choisir le fichier ft1 à utiliser (fond_3 ou slected_region) pour faire les gtbin et gtexposure
FT2_FILE        = "data/Photon_projet/spacecraft_projet/L2602050555366B2DD36C49_SC00.fits"
OUTPUT_BASE_DIR = "SED_output"

TIME_INTERVALS = [
    (5.03377224e+08, 5.03593224e+08, "activite_1"),
    # (5.04057624e+08, 5.04338424e+08, "activite_2"),
]

BIN_WIDTH_SEC = 10800   # largeur des bins temporels en secondes

# Paramètres gtexposure
PHOTON_INDEX = -2.1     # "Photon index for spectral weighting" = specin
IRFS         = "CALDB"  # Response functions
SRC_MODEL    = "none"   # Source model XML file

# ==============================================================================
# 2. BINS EN ÉNERGIE
# ==============================================================================

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
# 3. UTILITAIRES
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
    """Formate un float en chaîne propre pour les noms de fichiers."""
    s = f"{val:.2f}".rstrip("0").rstrip(".")
    return s

# ==============================================================================
# 4. BOUCLE PRINCIPALE
# ==============================================================================

def build_lc_and_exposure(dry_run: bool = False):
    gtbin_dir = os.path.join(OUTPUT_BASE_DIR, "gtbin")
    make_dirs(gtbin_dir)  # un seul dossier : gtexposure écrit dans gtbin in-place

    for tstart, tstop, tlabel in TIME_INTERVALS:
        print(f"\n{'='*70}")
        print(f"Période : {tlabel}  |  MET [{tstart:.6e} – {tstop:.6e}]")
        print(f"{'='*70}")

        for emin, emax in ENERGY_BINS:
            lc_name = f"gt_bin_fond_3_{fmt(emin)}_{fmt(emax)}.fits"#{tlabel} a remplacer lorsque on est pas sur le fond
            lc_name_gtselect = f"gtselect_{
            lc_path = os.path.join(gtbin_dir, lc_name)
            lc_path_gtselect = os.path.join(gtbin_dir, f"gtselect_{lc_name}")
            print(f"\n  Bin : [{emin} – {emax}] MeV  →  {lc_name}")
            # ── gtselect (LC) ────────────────────────────────────────────────────
            print("  → gtselect (LC)")
            run_cmd([
                "gtselect",
                f"infile={FT1_FILE}",
                f"outfile={lc_path_gtselect}",
                f"ra=0", f"dec=0", f"rad=180",  # sélectionne tout le ciel
                f"tmin={tstart}", f"tmax={tstop}",
                f"emin={emin}", f"emax={emax}",
                "zmax=90",  # angle de zenith max pour éviter la Terre
            ], dry_run=dry_run)
            # ── gtbin (LC) ────────────────────────────────────────────────────
            print("  → gtbin (LC)")
            run_cmd([
                "gtbin",
                f"evfile={FT1_FILE}",
                f"outfile={lc_path}",
                f"scfile={FT2_FILE}",
                "algorithm=LC",
                "tbinalg=LIN",
                f"tstart={tstart}",
                f"tstop={tstop}",
                f"dtime={BIN_WIDTH_SEC}",
                f"emin={emin}",
                f"emax={emax}",
            ], dry_run=dry_run)

            # ── gtexposure (modifie lc_path in-place) ────────────────────────
            # Paramètres : infile, scfile, irfs, srcmdl, specin
            # PAS de outfile : gtexposure ajoute la colonne EXPOSURE dans infile
            print("  → gtexposure (ajout colonne EXPOSURE in-place)")
            run_cmd([
                "gtexposure",
                f"infile={lc_path}",
                f"scfile={FT2_FILE}",
                f"irfs={IRFS}",
                f"srcmdl={SRC_MODEL}",
                f"specin={PHOTON_INDEX}",
            ], dry_run=dry_run)

    print(f"\n{'='*70}")
    print(f"Terminé. Fichiers dans : {gtbin_dir}/")

# ==============================================================================
# 5. POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    build_lc_and_exposure(dry_run=False)
