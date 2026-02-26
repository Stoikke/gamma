#!/usr/bin/env python3
"""
Script de génération automatique des fichiers gtbin (LC) + gtexposure pour la SED d'un GRB.
Auteur : généré pour l'analyse cat102
Usage  : python run_gtbin_gtexposure.py
"""

import os
import subprocess

# ==============================================================================
# 1. CONFIGURATION MANUELLE
# ==============================================================================

# --- Fichiers d'entrée communs ---
FT1_FILE        = "photons.fits"          # fichier FT1 (sortie gtselect)
FT2_FILE        = "spacecraft.fits"       # fichier FT2 / spacecraft
OUTPUT_BASE_DIR = "SED_output"

# --- Gammes de temps : une liste de tuples (tstart_MET, tstop_MET, label)
#     Ajoutez ou modifiez autant de périodes que nécessaire
TIME_INTERVALS = [
    (5.03377224e+08, 5.03593224e+08, "activite_1"),
    # (5.04000000e+08, 5.04200000e+08, "activite_2"),  # exemple période 2
]

# --- Paramètre gtbin ---
BIN_WIDTH_SEC = 10800   # largeur des intervalles de temps (secondes), configurable

# --- Paramètres gtexposure ---
PHOTON_INDEX  = -2.1    # indice spectral pour la pondération
IRFS          = "CALDB" # fonctions de réponse (toujours CALDB)
SRC_MODEL     = "none"  # modèle XML source (jamais utilisé ici)

# ==============================================================================
# 2. BINS EN ÉNERGIE (issus du fichier Bin-Emin-MeV-Emax-MeV.txt)
# ==============================================================================

ENERGY_BINS = [
    ( 100.0,    199.5),
    ( 199.5,    398.1),
    ( 398.1,    794.3),
    ( 794.3,   1584.9),
    (1584.9,   3162.3),
    (3162.3,   6309.6),
    (6309.6,  12589.3),
    (12589.3,  25118.9),
    (25118.9,  50118.7),
    (50118.7, 100000.0),
]

# ==============================================================================
# 3. FONCTIONS UTILITAIRES
# ==============================================================================

def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def run_cmd(cmd: list[str], dry_run: bool = False):
    """Exécute une commande shell ou l'affiche en mode dry_run."""
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


# ==============================================================================
# 4. BOUCLE PRINCIPALE
# ==============================================================================

def build_lc_and_exposure(dry_run: bool = True):
    """
    Pour chaque période temporelle × chaque bin en énergie :
      1. gtbin  (mode LC) → fichier light-curve
      2. gtexposure      → fichier d'exposition
    """
    gtbin_dir    = os.path.join(OUTPUT_BASE_DIR, "gtbin")
    gtexpose_dir = os.path.join(OUTPUT_BASE_DIR, "gtexposure")
    make_dirs(gtbin_dir, gtexpose_dir)

    for tstart, tstop, tlabel in TIME_INTERVALS:
        print(f"
{'='*70}")
        print(f"Période : {tlabel}  |  MET [{tstart:.6e} – {tstop:.6e}]")
        print(f"{'='*70}")

        for emin, emax in ENERGY_BINS:
            # Formatage des noms de fichiers (cohérent avec votre convention)
            emin_str = str(emin).replace(".0", "").rstrip("0").rstrip(".")
            emax_str = str(emax).replace(".0", "").rstrip("0").rstrip(".")

            lc_name   = f"gt_bin_{tlabel}_{emin_str}_{emax_str}.fits"
            expo_name = f"gt_exposure_{tlabel}_{emin_str}_{emax_str}.fits"

            lc_path   = os.path.join(gtbin_dir,    lc_name)
            expo_path = os.path.join(gtexpose_dir, expo_name)

            print(f"
  Bin énergie : [{emin} – {emax}] MeV")

            # ------------------------------------------------------------------
            # 4a. gtbin en mode LC
            # ------------------------------------------------------------------
            print("  → gtbin (LC)")
            gtbin_cmd = [
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
            ]
            run_cmd(gtbin_cmd, dry_run=dry_run)

            # ------------------------------------------------------------------
            # 4b. gtexposure
            # ------------------------------------------------------------------
            print("  → gtexposure")
            gtexposure_cmd = [
                "gtexposure",
                f"infile={lc_path}",
                f"scfile={FT2_FILE}",
                f"irfs={IRFS}",
                f"srcmdl={SRC_MODEL}",
                f"specin={PHOTON_INDEX}",
                f"outfile={expo_path}",
            ]
            run_cmd(gtexposure_cmd, dry_run=dry_run)

    print(f"
{'='*70}")
    print("Terminé. Vérifiez les fichiers dans :")
    print(f"  gtbin      : {os.path.join(OUTPUT_BASE_DIR, 'gtbin')}")
    print(f"  gtexposure : {os.path.join(OUTPUT_BASE_DIR, 'gtexposure')}")


# ==============================================================================
# 5. POINT D'ENTRÉE
# ==============================================================================

if __name__ == "__main__":
    # dry_run=True  → affiche les commandes SANS les exécuter (test)
    # dry_run=False → exécute réellement gtbin + gtexposure
    build_lc_and_exposure(dry_run=True)
