import os
import re
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit

# ==============================================================================
# 1. PARAMÈTRES
# ==============================================================================

OUTPUT_BASE_DIR = "SED_output"
activite        = "1"

gtbin_dir = os.path.join(OUTPUT_BASE_DIR, "gtbin")

# Pattern matche les fichiers dans gtbin_dir (sans chemin)
pattern = re.compile(
    rf"^gt_bin_activite_{activite}_(\d+\.?\d*)_(\d+\.?\d*)\.fits$"
)

# ==============================================================================
# 2. LECTURE DES FICHIERS
# ==============================================================================

energies_mean = []
flux_moyens   = []
flux_erreurs  = []

if not os.path.isdir(gtbin_dir):
    raise FileNotFoundError(f"Dossier introuvable : {gtbin_dir}")

fichiers = sorted([f for f in os.listdir(gtbin_dir) if pattern.match(f)])

if not fichiers:
    raise FileNotFoundError(
        f"Aucun fichier gt_bin_activite_{activite}_*.fits dans {gtbin_dir}"
    )

print(f"{len(fichiers)} fichier(s) gtbin trouvé(s) dans {gtbin_dir}/")

for fichier in fichiers:
    match = pattern.match(fichier)
    emin  = float(match.group(1))
    emax  = float(match.group(2))
    dE    = emax - emin                  # largeur du bin en énergie [MeV]
    emean = np.sqrt(emin * emax)         # centre géométrique [MeV]

    lc_path = os.path.join(gtbin_dir, fichier)

    with fits.open(lc_path) as hdul:
        cols_upper = [c.upper() for c in hdul[1].columns.names]
        data = Table(hdul[1].data)

        if "COUNTS" not in cols_upper or "EXPOSURE" not in cols_upper:
            print(f"  [SKIP] Colonnes manquantes dans {fichier} : {cols_upper}")
            continue

        counts   = data["COUNTS"].data.astype(float).flatten()
        exposure = data["EXPOSURE"].data.astype(float).flatten()

        # Colonne ERROR si disponible, sinon erreur poissonienne
        if "ERROR" in cols_upper:
            errors = data["ERROR"].data.astype(float).flatten()
        else:
            errors = np.sqrt(counts)

    # ── Flux par bin temporel : counts / exposure [ph cm⁻² s⁻¹]
    mask = exposure > 0
    if mask.sum() == 0:
        print(f"  [SKIP] Exposition nulle dans {fichier}")
        continue

    flux     = counts[mask]   / exposure[mask]          # ph cm⁻² s⁻¹
    flux_unc = errors[mask]   / exposure[mask]          # incertitude propagée

    # ── Moyenne sur les bins temporels
    N         = len(flux)
    flux_mean = np.mean(flux)

    # Incertitude sur la moyenne : propagation quadratique des sigma_i/sqrt(N)
    flux_err  = np.sqrt(np.sum(flux_unc**2)) / N

    # ── dPhi/dE [ph cm⁻² s⁻¹ MeV⁻¹]
    dphi_dE     = flux_mean / dE 
    dphi_dE_err = flux_err  / dE

    energies_mean.append(emean)
    flux_moyens.append(dphi_dE)
    flux_erreurs.append(dphi_dE_err)

    print(f"  [{emin:>10.2f} – {emax:>10.2f}] MeV  "
          f"| E_mean={emean:>9.2f} MeV  "
          f"| dPhi/dE={dphi_dE:.3e} ± {dphi_dE_err:.3e} ph/cm²/s/MeV")

energies_mean = np.array(energies_mean)
flux_moyens   = np.array(flux_moyens)
flux_erreurs  = np.array(flux_erreurs)

# ==============================================================================
# 3. FILTRAGE
# ==============================================================================

masque_valid = (flux_moyens > 0) & (flux_erreurs > 0) & (flux_erreurs < flux_moyens)

E_valid    = energies_mean[masque_valid]
F_valid    = flux_moyens[masque_valid]
Ferr_valid = flux_erreurs[masque_valid]

E_ul   = energies_mean[~masque_valid]
F_ul   = flux_erreurs[~masque_valid]

print(f"\n{masque_valid.sum()} point(s) valide(s), {(~masque_valid).sum()} upper limit(s)")

if len(E_valid) == 0:
    raise ValueError("Aucun point valide après filtrage.")

print("\n=== dPhi/dE (ph cm⁻² s⁻¹ MeV⁻¹) ===")
for e, f, fe in zip(E_valid, F_valid, Ferr_valid):
    print(f"  E={e:>9.2f} MeV  dPhi/dE = {f:.3e} ± {fe:.3e}")

# ==============================================================================
# 4. AJUSTEMENT LOI DE PUISSANCE EN LOG-LOG
# ==============================================================================
# log10(dPhi/dE) = alpha * log10(E) + log10(N0)

def loi_puissance_log(logE, alpha, logN0):
    return alpha * logE + logN0

logE     = np.log10(E_valid)
logF     = np.log10(F_valid)
logF_err = Ferr_valid / (F_valid * np.log(10))   # propagation en log

popt, pcov = curve_fit(
    loi_puissance_log, logE, logF,
    sigma=logF_err, absolute_sigma=True
)
perr       = np.sqrt(np.diag(pcov))
alpha, logN0 = popt

print(f"\n=== Ajustement loi de puissance ===")
print(f"  Indice spectral  alpha = {alpha:.3f} ± {perr[0]:.3f}")
print(f"  log10(N0)              = {logN0:.3f} ± {perr[1]:.3f}")
print(f"  N0                     = {10**logN0:.3e} ph cm⁻² s⁻¹ MeV⁻¹")

# ==============================================================================
# 5. TRACÉ SED
# ==============================================================================

E_fit = np.logspace(np.log10(E_valid.min()), np.log10(E_valid.max()), 300)
F_fit = 10**(loi_puissance_log(np.log10(E_fit), *popt))

# Label sans mélange rf'' : on construit la string séparément
label_fit = (r"Loi de puissance : $\alpha=" +
             f"{alpha:.2f}" + r"\pm" + f"{perr[0]:.2f}$")

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(
    E_valid, F_valid, yerr=Ferr_valid,
    fmt="o", color="steelblue", capsize=4,
    label="Données", zorder=5
)

if len(E_ul) > 0:
    F_ul_norm = F_ul   # déjà en dPhi/dE
    ax.errorbar(
        E_ul, F_ul_norm, yerr=0.3 * F_ul_norm,
        fmt="v", color="gray", capsize=0,
        label="Upper limits", uplims=True
    )

ax.plot(E_fit, F_fit, "r-", label=label_fit)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energie (MeV)", fontsize=13)
ax.set_ylabel(r"$d\Phi/dE$ (ph cm$^{-2}$ s$^{-1}$ MeV$^{-1}$)", fontsize=13)
ax.set_title(f"SED — Activite {activite} (GRB cat102)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, which="both", alpha=0.3)
plt.savefig(f"SED_activite_{activite}.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nFigure sauvegardee : SED_activite_{activite}.png")
