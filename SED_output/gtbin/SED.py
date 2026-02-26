import os
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit

# ── PARAMÈTRES ──────────────────────────────────────────────────────────────
dossier  = "."
activite = "1"

pattern = re.compile(rf"gt_bin_activite_{activite}_(\d+\.?\d*)_(\d+\.?\d*)\.fits")

# ── LECTURE DES FICHIERS ─────────────────────────────────────────────────────
energies_mean = []
flux_moyens   = []
flux_erreurs  = []

fichiers = sorted([f for f in os.listdir(dossier) if pattern.match(f)])

for fichier in fichiers:
    match = pattern.match(fichier)
    emin  = float(match.group(1))
    emax  = float(match.group(2))
    emean = np.sqrt(emin * emax)

    with fits.open(os.path.join(dossier, fichier)) as hdul:
        cols = [c.upper() for c in hdul[1].columns.names]
        data = Table(hdul[1].data)
        if "COUNTS" in cols and "EXPOSURE" in cols:
            counts   = data["COUNTS"].data.astype(float).flatten()
            exposure = data["EXPOSURE"].data.astype(float).flatten()
        else:
            print(f"[SKIP] Colonnes non trouvées dans {fichier}")
            continue

    mask      = exposure > 0
    flux      = counts[mask] / exposure[mask]
    N         = len(flux)
    flux_mean = np.mean(flux)
    flux_err  = np.std(flux, ddof=1) / np.sqrt(N) if N > 1 else np.sqrt(np.sum(counts[mask])) / np.sum(exposure[mask])

    energies_mean.append(emean)
    flux_moyens.append(flux_mean)
    flux_erreurs.append(flux_err)

energies_mean = np.array(energies_mean)
flux_moyens   = np.array(flux_moyens)
flux_erreurs  = np.array(flux_erreurs)

# ── FILTRAGE ─────────────────────────────────────────────────────────────────
masque_valid = (flux_moyens > 0) & (flux_erreurs < flux_moyens)

E_valid    = energies_mean[masque_valid]
F_valid    = flux_moyens[masque_valid]
Ferr_valid = flux_erreurs[masque_valid]

E_ul = energies_mean[~masque_valid]
F_ul = flux_erreurs[~masque_valid]

# ── NORMALISATION PAR L'ÉNERGIE ───────────────────────────────────────────────
# dN/dE = flux / E  (ph cm⁻² s⁻¹ MeV⁻¹)
F_norm    = F_valid    / E_valid
Ferr_norm = Ferr_valid / E_valid

# Upper limits normalisés
F_ul_norm = F_ul / E_ul

print(f"\n{masque_valid.sum()} points valides, {(~masque_valid).sum()} upper limits exclus")

print("\n=== Energies valides (MeV) ===")
print(E_valid)
print("\n=== Flux / E (ph cm⁻² s⁻¹ MeV⁻¹) ===")
print(F_norm)
print("\n=== Erreurs associées ===")
print(Ferr_norm)

# ── AJUSTEMENT LOI DE PUISSANCE EN LOG-LOG ───────────────────────────────────
def loi_puissance_log(logE, a, b):
    return a * logE + b

logE     = np.log10(E_valid)
logF     = np.log10(F_norm)
logF_err = Ferr_norm / (F_norm * np.log(10))

popt, pcov = curve_fit(loi_puissance_log, logE, logF, sigma=logF_err, absolute_sigma=True)
perr = np.sqrt(np.diag(pcov))
a, b = popt
print(f"\n=== Ajustement loi de puissance ===")
print(f"  Indice spectral α = {a:.3f} ± {perr[0]:.3f}")
print(f"  Normalisation log(N0) = {b:.3f} ± {perr[1]:.3f}")

# ── TRACÉ SED ────────────────────────────────────────────────────────────────
E_fit = np.logspace(np.log10(E_valid.min()), np.log10(E_valid.max()), 200)
F_fit = 10**(loi_puissance_log(np.log10(E_fit), *popt))

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(E_valid, F_norm, yerr=Ferr_norm,
            fmt='o', color='steelblue', capsize=4, label='Données')

if len(E_ul) > 0:
    ax.errorbar(E_ul, F_ul_norm, yerr=0.3*F_ul_norm,
                fmt='v', color='gray', capsize=0,
                label='Upper limits', uplims=True)

ax.plot(E_fit, F_fit, 'r-',
        label=rf'Loi de puissance : $\alpha={a:.2f}\pm{perr[0]:.2f}$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Énergie (MeV)', fontsize=13)
ax.set_ylabel('dϕ /dE (ph cm$^{-2}$ s$^{-1}$ MeV$^{-1}$)', fontsize=13)
ax.set_title(f'SED — Activité {activite}', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(f"SED_activite_{activite}.png", dpi=150)
plt.show()
print(f"\nFigure sauvegardée : SED_activite_{activite}.png")
