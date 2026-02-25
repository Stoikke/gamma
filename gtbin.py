import os
import csv
import subprocess

try:
    from GtApp import GtApp
    USE_GTAPP = True
    print("✅ GtApp disponible")
except ImportError:
    USE_GTAPP = False
    print("⚠ GtApp non disponible, fallback subprocess")

# ==============================================================
# ==================== CONFIGURATION ==========================
# ==============================================================

CONFIG_CSV = 'runs_config.csv'   # ← Fichier de config (optionnel)
EV_DIR     = 'SED_output'             # dossier contenant les fichiers FT1
OUTDIR     = 'SED_output/gtbin_output'      # ← Dossier de sortie

# Paramètres gtbin globaux (identiques pour tous les runs)
ALGORITHM = 'LC'
TBINALG   = 'LIN'
SCFILE    = 'NONE'

# valeurs par défaut utilisées si on ne connaît pas tstart/tstop/dtime
DEFAULT_TSTART = 501643824
DEFAULT_TSTOP  = 505267085
DEFAULT_DTIME  = 5400

# ==============================================================
# ==================== LECTURE CONFIG =========================
# ==============================================================

def load_runs_from_csv(csv_file):
    """
    Lit le CSV de config et retourne une liste de dicts.
    Colonnes attendues: evfile, label, tstart, tstop, dtime
    Ce fichier est entièrement optionnel ; si absent on peut scanner
    un dossier pour trouver des .fits.
    """
    runs = []
    if not os.path.exists(csv_file):
        # Ne pas alerter encore, on traitera plus bas
        return runs

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append({
                "evfile": row['evfile'].strip(),
                "label":  row['label'].strip(),
                "tstart": float(row['tstart']),
                "tstop":  float(row['tstop']),
                "dtime":  float(row['dtime']),
            })

    print(f"✅ {len(runs)} runs chargés depuis {csv_file}")
    for r in runs:
        print(f"  {r['label']:15s} | evfile={os.path.basename(r['evfile'])} "
              f"| [{r['tstart']:.4e}, {r['tstop']:.4e}] | dtime={r['dtime']:.0f}s")
    return runs


def load_runs_from_dir(directory):
    """Constitue une liste simple de runs à partir de tous les fichiers
    `.fits` présents dans `directory`. Les valeurs temporelles sont
    initialisées aux constantes DEFAULT_*, il faut les ajuster si vous
    souhaitez autre chose.
    """
    runs = []
    if not os.path.isdir(directory):
        return runs
    for fname in os.listdir(directory):
        if fname.lower().endswith('.fits'):
            evfile = os.path.join(directory, fname)
            label = os.path.splitext(fname)[0]
            runs.append({
                "evfile": evfile,
                "label": label,
                "tstart": DEFAULT_TSTART,
                "tstop": DEFAULT_TSTOP,
                "dtime": DEFAULT_DTIME,
            })
    if runs:
        print(f"✅ {len(runs)} fichiers .fits trouvés dans {directory}")
    return runs

# ==============================================================
# ==================== FONCTION GTBIN =========================
# ==============================================================

def run_gtbin(evfile, outfile, scfile, algorithm, tbinalg, tstart, tstop, dtime):
    """Lance gtbin LC via GtApp ou subprocess stdin"""
    print(f"\n  evfile  : {os.path.basename(evfile)}")
    print(f"  outfile : {os.path.basename(outfile)}")
    print(f"  tstart={tstart:.4e}  tstop={tstop:.4e}  dtime={dtime:.0f}s")

    if not os.path.exists(evfile):
        print(f"  ❌ evfile introuvable : {evfile}")
        return False

    try:
        if USE_GTAPP:
            gtbin = GtApp('gtbin')
            gtbin['evfile']    = evfile
            gtbin['outfile']   = outfile
            gtbin['scfile']    = scfile
            gtbin['algorithm'] = algorithm
            gtbin['tbinalg']   = tbinalg
            gtbin['tstart']    = tstart
            gtbin['tstop']     = tstop
            gtbin['dtime']     = str(dtime)
            gtbin['clobber']   = 'yes'
            gtbin.run()
        else:
            params = '\n'.join([
                algorithm, evfile, outfile, scfile,
                tbinalg, str(tstart), str(tstop), str(dtime), 'yes'
            ])
            proc = subprocess.Popen(
                ['gtbin'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True
            )
            out, err = proc.communicate(input=params)
            if proc.returncode != 0:
                print(f"  ❌ gtbin ERREUR:\n{err.strip()[-300:]}")
                return False

        print(f"  ✅ OK → {os.path.basename(outfile)}")
        return True

    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

# ==============================================================
# ==================== MAIN ===================================
# ==============================================================

os.makedirs(OUTDIR, exist_ok=True)

# charge les runs depuis CSV si présent
RUNS = load_runs_from_csv(CONFIG_CSV)
if not RUNS:
    # si le CSV est absent ou vide, on prend tous les FITS de EV_DIR
    print(f"⚠ Pas de configuration CSV ({CONFIG_CSV}) ; balayage de {EV_DIR}")
    RUNS = load_runs_from_dir(EV_DIR)
    if not RUNS:
        print("❌ Aucun run à traiter, arrêt.")
        exit(1)

print(f"\n{'='*60}")
print(f"GTBIN BATCH — {len(RUNS)} fichiers | algo={ALGORITHM} | tbinalg={TBINALG}")
print(f"{'='*60}")

success, failed = [], []

for i, run in enumerate(RUNS):
    print(f"\n[{i+1}/{len(RUNS)}] ── {run['label']}")

    outfile = os.path.join(
        OUTDIR,
        f"gtbin_{ALGORITHM}_{run['label']}_dt{int(run['dtime'])}s.fits"
    )

    ok = run_gtbin(
        evfile    = run['evfile'],
        outfile   = outfile,
        scfile    = SCFILE,
        algorithm = ALGORITHM,
        tbinalg   = TBINALG,
        tstart    = run['tstart'],
        tstop     = run['tstop'],
        dtime     = run['dtime'],
    )

    if ok: success.append(run['label'])
    else:  failed.append(run['label'])

# Résumé final
print(f"\n{'='*60}")
print(f"RÉSUMÉ : {len(success)}/{len(RUNS)} OK")
if success: print(f"  ✅ OK    : {', '.join(success)}")
if failed:  print(f"  ❌ ÉCHEC : {', '.join(failed)}")
print(f"{'='*60}")
print(f"✅ Fichiers dans → {OUTDIR}/")
