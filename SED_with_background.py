import os
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit

# ==============================================================================
# 1. PARAMÈTRES
# ==============================================================================

OUTPUT_BASE_DIR = "SED_output"
activite        = "1"

gtbin_dir   = os.path.join(OUTPUT_BASE_DIR, "gtbin")
fond_dir    = OUTPUT_BASE_DIR  # dossier racine pour gtbin_fond_x.fits

# Patterns
pattern_signal = re.compile(rf"^gt_bin_activite_{activite}_(\d+\.?\d*)_(\d+\.?\d*)\.fits$")
pattern_fond   = re.compile(rf"^gt_bin_activite_{activite}_fond_(\d+\.?\d*)_(\d+\.?\d*)\.fits$")

# ==============================================================================
# 2. LECTURE FOND : moyenne des flux par bin énergie
# ==============================================================================

fond_flux_moyens = {}
fond_flux_erreurs = {}

if os.path.isdir(fond_dir):
    fond_files = [f for f in os.listdir(fond_dir) if pattern_fond.match(f)]
    print(f"{len(fond_files)} fichier(s) de fond trouvé(s) dans {fond_dir}/")

    for fichier in fond_files:
        match = pattern_fond.match(fichier)
        emin, emax = float(match.group(1)), float(match.group(2))
        dE    = emax - emin
        emean = np.sqrt(emin * emax)
        fond_path = os.path.join(fond_dir, fichier)

        with fits.open(fond_path) as hdul:
            cols_upper = [c.upper() for c in hdul[1].columns.names]
            data = Table(hdul[1].data)

            if "COUNTS" not in cols_upper or "EXPOSURE" not in cols_upper:
                print(f"  [FOND SKIP] {fichier} : {cols_upper}")
                continue

            counts   = data["COUNTS"].data.astype(float).flatten()
            exposure = data["EXPOSURE"].data.astype(float).flatten()

            if "ERROR" in cols_upper:
                errors = data["ERROR"].data.astype(float).flatten()
            else:
                errors = np.sqrt(counts)

        mask = exposure > 0
        if mask.sum() == 0:
            continue

        flux     = counts[mask]   / exposure[mask]
        flux_unc = errors[mask]   / exposure[mask]

        N         = len(flux)
        flux_mean = np.mean(flux)
        flux_err  = np.sqrt(np.sum(flux_unc**2)) / N

        dphi_dE     = flux_mean / dE
        dphi_dE_err = flux_err  / dE

        fond_flux_moyens[emean] = dphi_dE
        fond_flux_erreurs[emean] = dphi_dE_err

        print(f"  [FOND] [{emin:>6.1f}-{emax:>7.1f}] MeV  "
              f"| dPhi/dE={dphi_dE:.3e} ± {dphi_dE_err:.3e}")

else:
    print(f"Dossier fond {fond_dir} introuvable")

print("\n" + "="*80 + "\n")

# ==============================================================================
# 3. LECTURE SIGNAL
# ==============================================================================

energies_mean = []
flux_moyens   = []
flux_erreurs  = []

if not os.path.isdir(gtbin_dir):
    raise FileNotFoundError(f"Dossier introuvable : {gtbin_dir}")

fichiers = sorted([f for f in os.listdir(gtbin_dir) if pattern_signal.match(f)])

if not fichiers:
    raise FileNotFoundError(f"Aucun gt_bin_activite_{activite}_*.fits dans {gtbin_dir}")

print(f"{len(fichiers)} fichier(s) signal trouvé(s) dans {gtbin_dir}/")
dE_tableau = []
for fichier in fichiers:
    match = pattern_signal.match(fichier)
    emin, emax = float(match.group(1)), float(match.group(2))
    dE    = emax - emin
    dE_tableau.append(dE)
    emean = np.sqrt(emin * emax)
    dE_tableau
    lc_path = os.path.join(gtbin_dir, fichier)

    with fits.open(lc_path) as hdul:
        cols_upper = [c.upper() for c in hdul[1].columns.names]
        data = Table(hdul[1].data)

        if "COUNTS" not in cols_upper or "EXPOSURE" not in cols_upper:
            print(f"  [SIGNAL SKIP] {fichier}")
            continue

        counts   = data["COUNTS"].data.astype(float).flatten()
        exposure = data["EXPOSURE"].data.astype(float).flatten()

        if "ERROR" in cols_upper:
            errors = data["ERROR"].data.astype(float).flatten()
        else:
            errors = np.sqrt(counts)

    mask = exposure > 0
    if mask.sum() == 0:
        continue

    flux     = counts[mask] / exposure[mask]
    flux_unc = errors[mask] / exposure[mask]

    N         = len(flux)
    flux_mean = np.mean(flux)
    flux_err  = np.sqrt(np.sum(flux_unc**2)) / N

    # ── SOUSTRACTION DU FOND ─────────────────────────────
    fond_dphi_dE = fond_flux_moyens.get(emean, 0.0)
    net_dphi_dE  = flux_mean / dE - fond_dphi_dE
    net_dphi_dE_err = np.sqrt((flux_err / dE)**2 + fond_flux_erreurs.get(emean, 0.0)**2)

    # Filtre : flux net positif
    if net_dphi_dE > 0:
        energies_mean.append(emean)
        flux_moyens.append(net_dphi_dE)
        flux_erreurs.append(net_dphi_dE_err)

        print(f"  [{emin:>10.2f} – {emax:>10.2f}] MeV  "
              f"| E_mean={emean:>9.2f} MeV  "
              f"| net dPhi/dE={net_dphi_dE:.3e} ± {net_dphi_dE_err:.3e}")

energies_mean = np.array(energies_mean)
flux_moyens   = np.array(flux_moyens)
flux_erreurs  = np.array(flux_erreurs)

# ==============================================================================
# 4. FILTRAGE + FIT
# ==============================================================================

masque_valid = (flux_moyens > 0) & (flux_erreurs > 0) & (flux_erreurs < flux_moyens)

E_valid    = energies_mean[masque_valid]
F_valid    = flux_moyens[masque_valid]
Ferr_valid = flux_erreurs[masque_valid]

if len(E_valid) == 0:
    raise ValueError("Aucun point valide après soustraction fond.")

def loi_puissance_log(logE, alpha, logN0):
    return alpha * logE + logN0

logE     = np.log10(E_valid)
logF     = np.log10(F_valid)
logF_err = Ferr_valid / (F_valid * np.log(10))

popt, pcov = curve_fit(loi_puissance_log, logE, logF, sigma=logF_err, absolute_sigma=True)
perr       = np.sqrt(np.diag(pcov))
alpha, logN0 = popt

print(f"\n=== FIT LOI DE PUISSANCE (signal - fond) ===")
print(f"  Indice spectral  alpha = {alpha:.3f} ± {perr[0]:.3f}")
print(f"  log10(N0)              = {logN0:.3f} ± {perr[1]:.3f}")
print(f"  N0                     = {10**logN0:.3e}")
print("dE_tableau",dE_tableau)
# ==============================================================================
# 5. TRACÉ
# ==============================================================================

E_fit = np.logspace(np.log10(E_valid.min()), np.log10(E_valid.max()), 300)
F_fit = 10**(loi_puissance_log(np.log10(E_fit), *popt))

label_fit = r"Loi de puissance : $\alpha=" + f"{alpha:.2f}\\pm{perr[0]:.2f}$"

fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(E_valid, F_valid, yerr=Ferr_valid,xerr= dE_tableau,
            fmt="o", color="steelblue", capsize=4, label="Signal - Fond")

if len(fond_flux_moyens) > 0:
    E_fond = np.array(list(fond_flux_moyens.keys()))
    F_fond = np.array(list(fond_flux_moyens.values()))
    ax.errorbar(E_fond, F_fond, yerr=np.array(list(fond_flux_erreurs.values())),xerr= dE_tableau,
                fmt="s", color="orange", capsize=3, label="Fond (moyenne)")

ax.plot(E_fit, F_fit, "r-", linewidth=2, label=label_fit)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Énergie (MeV)")
ax.set_ylabel(r"$d\Phi/dE$ (ph cm$^{-2}$ s$^{-1}$ MeV$^{-1}$)")
ax.set_title(f"SED — Activité {activite} (GRB cat102, fond soustrait)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig(f"SED_activite_{activite}_net.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSED nette sauvegardée : SED_activite_{activite}_net.png")
