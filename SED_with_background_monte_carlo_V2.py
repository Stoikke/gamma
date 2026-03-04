import os
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.odr import ODR, Model, RealData

# ==============================================================================
# 1. PARAMÈTRES
# ==============================================================================
OUTPUT_BASE_DIR      = "SED_output"
OUTPUT_BASE_DIR_FOND = "SED_output/gtbin"
activite             = "2"

gtbin_dir = os.path.join(OUTPUT_BASE_DIR, "gtbin")
fond_dir  = OUTPUT_BASE_DIR_FOND

pattern_signal = re.compile(rf"^gt_bin_activite_{activite}_(\d+\.?\d*)_(\d+\.?\d*)\.fits$")
pattern_fond   = re.compile(rf"^gt_bin_activite_{activite}_fond_[123]_(\d+\.?\d*)_(\d+\.?\d*)\.fits$")

# ==============================================================================
# 2. LECTURE FOND
# ==============================================================================
fond_flux_moyens  = {}
fond_flux_erreurs = {}
fond_dE_half      = {}

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
                print(f"  [FOND SKIP] {fichier}")
                continue
            counts   = data["COUNTS"].data.astype(float).flatten()
            exposure = data["EXPOSURE"].data.astype(float).flatten()
            errors   = (data["ERROR"].data.astype(float).flatten()
                        if "ERROR" in cols_upper else np.sqrt(counts))

        mask = exposure > 0
        if mask.sum() == 0:
            continue

        flux     = counts[mask] / exposure[mask]
        flux_unc = errors[mask] / exposure[mask]
        N        = len(flux)
        flux_mean = np.mean(flux)
        flux_err  = np.sqrt(np.sum(flux_unc**2)) / N

        dphi_dE     = flux_mean / dE
        dphi_dE_err = flux_err  / dE

        # Moyenne pondérée si le bin existe déjà (plusieurs fichiers fond)
        if emean in fond_flux_moyens:
            old_f, old_e = fond_flux_moyens[emean], fond_flux_erreurs[emean]
            w1, w2 = 1/old_e**2, 1/dphi_dE_err**2
            fond_flux_moyens[emean]  = (w1*old_f + w2*dphi_dE) / (w1 + w2)
            fond_flux_erreurs[emean] = 1.0 / np.sqrt(w1 + w2)
        else:
            fond_flux_moyens[emean]  = dphi_dE
            fond_flux_erreurs[emean] = dphi_dE_err
            fond_dE_half[emean]      = dE / 2.0

        print(f"  [FOND] [{emin:>6.1f}-{emax:>7.1f}] MeV  "
              f"| dPhi/dE={dphi_dE:.3e} ± {dphi_dE_err:.3e}")
else:
    print(f"Dossier fond {fond_dir} introuvable")

fond_energies = np.array(sorted(fond_flux_moyens.keys()))

def get_fond_nearest(emean, tol=0.10):
    """Retourne (fond, err_fond) du bin le plus proche dans ±tol relatif."""
    if len(fond_energies) == 0:
        return 0.0, 0.0
    idx = np.argmin(np.abs(fond_energies - emean))
    e_near = fond_energies[idx]
    if np.abs(e_near - emean) / emean < tol:
        return fond_flux_moyens[e_near], fond_flux_erreurs[e_near]
    print(f"  [FOND MANQUANT] emean={emean:.2f} MeV (plus proche={e_near:.2f}, >{tol*100:.0f}%)")
    return 0.0, 0.0

print("\n" + "="*80 + "\n")

# ==============================================================================
# 3. LECTURE SIGNAL
# ==============================================================================
energies_mean = []
flux_moyens   = []
flux_erreurs  = []
dE_half_list  = []

if not os.path.isdir(gtbin_dir):
    raise FileNotFoundError(f"Dossier introuvable : {gtbin_dir}")

fichiers = sorted([f for f in os.listdir(gtbin_dir) if pattern_signal.match(f)])
if not fichiers:
    raise FileNotFoundError(f"Aucun gt_bin_activite_{activite}_*.fits dans {gtbin_dir}")

print(f"{len(fichiers)} fichier(s) signal trouvé(s) dans {gtbin_dir}/")

for fichier in fichiers:
    match = pattern_signal.match(fichier)
    emin, emax = float(match.group(1)), float(match.group(2))
    dE    = emax - emin
    emean = np.sqrt(emin * emax)
    lc_path = os.path.join(gtbin_dir, fichier)

    with fits.open(lc_path) as hdul:
        cols_upper = [c.upper() for c in hdul[1].columns.names]
        data = Table(hdul[1].data)
        if "COUNTS" not in cols_upper or "EXPOSURE" not in cols_upper:
            print(f"  [SIGNAL SKIP] {fichier}")
            continue
        counts   = data["COUNTS"].data.astype(float).flatten()
        exposure = data["EXPOSURE"].data.astype(float).flatten()
        errors   = (data["ERROR"].data.astype(float).flatten()
                    if "ERROR" in cols_upper else np.sqrt(counts))

    mask = exposure > 0
    if mask.sum() == 0:
        continue

    flux     = counts[mask] / exposure[mask]
    flux_unc = errors[mask] / exposure[mask]
    N         = len(flux)
    flux_mean = np.mean(flux)
    flux_err  = np.sqrt(np.sum(flux_unc**2)) / N

    fond_dphi_dE, fond_dphi_dE_err = get_fond_nearest(emean)

    net_dphi_dE     = flux_mean / dE - fond_dphi_dE
    net_dphi_dE_err = np.sqrt((flux_err / dE)**2 + fond_dphi_dE_err**2)

    print(f"  [{emin:>10.2f} – {emax:>10.2f}] MeV  "
          f"| E_mean={emean:>9.2f} MeV  "
          f"| net dPhi/dE={net_dphi_dE:.3e} ± {net_dphi_dE_err:.3e}"
          f"{'  [<0 ignoré]' if net_dphi_dE <= 0 else ''}")

    if net_dphi_dE > 0:
        energies_mean.append(emean)
        flux_moyens.append(net_dphi_dE)
        flux_erreurs.append(net_dphi_dE_err)
        dE_half_list.append(dE / 2.0)

energies_mean = np.array(energies_mean)
flux_moyens   = np.array(flux_moyens)
flux_erreurs  = np.array(flux_erreurs)
dE_half_arr   = np.array(dE_half_list)

print(f"\n{len(energies_mean)} points signal positifs")

# ==============================================================================
# 4. FILTRAGE
# ==============================================================================
masque_valid = (flux_moyens > 0) & (flux_erreurs > 0) & (flux_erreurs < flux_moyens)
print(f"{masque_valid.sum()} points après masque (err < flux)")

E_valid    = energies_mean[masque_valid]
F_valid    = flux_moyens[masque_valid]
Ferr_valid = flux_erreurs[masque_valid]
Dx_valid   = dE_half_arr[masque_valid]

if len(E_valid) == 0:
    print("\n=== DEBUG ===")
    print(f"  energies_mean : {energies_mean}")
    print(f"  flux_moyens   : {flux_moyens}")
    print(f"  flux_erreurs  : {flux_erreurs}")
    print(f"  flux>0        : {flux_moyens > 0}")
    print(f"  err>0         : {flux_erreurs > 0}")
    print(f"  err<flux      : {flux_erreurs < flux_moyens}")
    raise ValueError("Aucun point valide après soustraction fond — voir debug ci-dessus.")

# ==============================================================================
# 5. FIT ODR
# ==============================================================================
def loi_puissance_odr(params, x):
    return params[0] * x + params[1]

logE     = np.log10(E_valid)
logF     = np.log10(F_valid)
logF_err = Ferr_valid / (F_valid * np.log(10))
logE_err = Dx_valid   / (E_valid * np.log(10))

model    = Model(loi_puissance_odr)
data_odr = RealData(logE, logF, sx=logE_err, sy=logF_err)
odr_fit  = ODR(data_odr, model, beta0=[-2.0, -10.0])
result   = odr_fit.run()

alpha, logN0 = result.beta
perr_odr     = result.sd_beta

print(f"\n=== FIT ODR ===")
print(f"  alpha  = {alpha:.3f} ± {perr_odr[0]:.3f}")
print(f"  logN0  = {logN0:.3f} ± {perr_odr[1]:.3f}")
print(f"  N0     = {10**logN0:.3e}")

# ==============================================================================
# 6. MONTE-CARLO
# ==============================================================================
N_MC = 100000
rng  = np.random.default_rng(42)

alpha_mc = np.empty(N_MC)
logN0_mc = np.empty(N_MC)

for k in range(N_MC):
    logF_mc = logF + rng.standard_normal(len(logF)) * logF_err
    logE_mc = logE + rng.standard_normal(len(logE)) * logE_err
    try:
        data_mc = RealData(logE_mc, logF_mc, sx=logE_err, sy=logF_err)
        odr_mc  = ODR(data_mc, model, beta0=[alpha, logN0])
        res_mc  = odr_mc.run()
        alpha_mc[k], logN0_mc[k] = res_mc.beta
    except Exception:
        alpha_mc[k], logN0_mc[k] = np.nan, np.nan

valid_mc = np.isfinite(alpha_mc) & np.isfinite(logN0_mc)
alpha_mc = alpha_mc[valid_mc]
logN0_mc = logN0_mc[valid_mc]

alpha_mc_std  = np.std(alpha_mc, ddof=1)
logN0_mc_std  = np.std(logN0_mc, ddof=1)

print(f"\n=== MONTE-CARLO ({N_MC} tirages, {valid_mc.sum()} valides) ===")
print(f"  alpha  = {np.mean(alpha_mc):.3f} ± {alpha_mc_std:.3f}  (1σ MC)")
print(f"  logN0  = {np.mean(logN0_mc):.3f} ± {logN0_mc_std:.3f}  (1σ MC)")

# ==============================================================================
# 7. FIGURE
# ==============================================================================
masque_plot = (flux_moyens > 0) & (flux_erreurs > 0)
E_plot    = energies_mean[masque_plot]
F_plot    = flux_moyens[masque_plot]
Ferr_plot = flux_erreurs[masque_plot]
Dx_plot   = dE_half_arr[masque_plot]

E_fit = np.logspace(np.log10(E_valid.min()), np.log10(E_valid.max()), 300)
F_fit = 10**(loi_puissance_odr([alpha, logN0], np.log10(E_fit)))

fig, ax = plt.subplots(figsize=(10, 6))

# Courbes MC
N_PLOT   = 1000
idx_plot = rng.choice(len(alpha_mc), size=N_PLOT, replace=False)
for i, k in enumerate(idx_plot):
    ax.plot(E_fit,
            10**(loi_puissance_odr([alpha_mc[k], logN0_mc[k]], np.log10(E_fit))),
            color="red", alpha=0.01, linewidth=0.5,
            label="Réalisations MC" if i == 0 else None)

# Points détectés (S/N > 1)
mask_det = F_plot > Ferr_plot
mask_ul  = ~mask_det

if mask_det.any():
    ax.errorbar(E_plot[mask_det], F_plot[mask_det],
                yerr=Ferr_plot[mask_det], xerr=Dx_plot[mask_det],
                fmt="o", color="steelblue", capsize=4, label="Signal - Fond")

if mask_ul.any():
    ul_values = 3 * Ferr_plot[mask_ul]
    ax.errorbar(E_plot[mask_ul], ul_values,
                xerr=Dx_plot[mask_ul], yerr=ul_values * 0.3,
                fmt="o", color="steelblue", uplims=True,
                capsize=4, alpha=0.6, label="Upper limit (3σ)")

label_fit = (r"Fit : $\alpha=" + f"{alpha:.2f}" +
             r" \pm " + f"{alpha_mc_std:.2f}$" +
             r"  $\log_{{10}}(N_0)=" + f"{logN0:.2f}" +
             r" \pm " + f"{logN0_mc_std:.2f}$")

ax.plot(E_fit, F_fit, "r-", linewidth=2, label=label_fit)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Énergie (MeV)")
ax.set_ylabel(r"$d\Phi/dE$ (ph cm$^{{-2}}$ s$^{{-1}}$ MeV$^{{-1}}$)")
ax.set_title(f"SED — Activité {activite} (GRB cat102, fond soustrait)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.savefig(f"SED_activite_{activite}_net_MC.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSED sauvegardée : SED_activite_{activite}_net_MC.png")