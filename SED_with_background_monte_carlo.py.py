import os
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData
OUTPUT_BASE_DIR = "SED_output"
OUTPUT_BASE_DIR_FOND = "SED_output/gtbin"
activite        = "1"

gtbin_dir   = os.path.join(OUTPUT_BASE_DIR, "gtbin")
fond_dir    = OUTPUT_BASE_DIR_FOND

pattern_signal = re.compile(rf"^gt_bin_activite_{activite}_(\d+\.?\d*)_(\d+\.?\d*)\.fits$")
pattern_fond   = re.compile(rf"^gt_bin_activite_{activite}_fond_[123]_(\d+\.?\d*)_(\d+\.?\d*)\.fits$")

fond_flux_moyens = {}
fond_flux_erreurs = {}
fond_dE_half = {}

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

        fond_flux_moyens[emean]  = dphi_dE
        fond_flux_erreurs[emean] = dphi_dE_err
        fond_dE_half[emean]      = dE / 2.0

        print(f"  [FOND] [{emin:>6.1f}-{emax:>7.1f}] MeV  "
              f"| dPhi/dE={dphi_dE:.3e} ± {dphi_dE_err:.3e}")
else:
    print(f"Dossier fond {fond_dir} introuvable")

print("" + "="*80 + "")

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

    fond_dphi_dE     = fond_flux_moyens.get(emean, 0.0)
    fond_dphi_dE_err = fond_flux_erreurs.get(emean, 0.0)

    net_dphi_dE     = flux_mean / dE - fond_dphi_dE #* dE**2
    net_dphi_dE_err = np.sqrt((flux_err / dE)**2 + fond_dphi_dE_err**2)

    if net_dphi_dE > 0:
        energies_mean.append(emean)
        flux_moyens.append(net_dphi_dE)
        flux_erreurs.append(net_dphi_dE_err)
        dE_half_list.append(dE / 2.0)

        print(f"  [{emin:>10.2f} – {emax:>10.2f}] MeV  "
              f"| E_mean={emean:>9.2f} MeV  "
              f"| net dPhi/dE={net_dphi_dE:.3e} ± {net_dphi_dE_err:.3e}")

energies_mean = np.array(energies_mean)
flux_moyens   = np.array(flux_moyens)
flux_erreurs  = np.array(flux_erreurs)
dE_half_arr   = np.array(dE_half_list)

masque_valid = (flux_moyens > 0) & (flux_erreurs > 0) & (flux_erreurs < flux_moyens)

E_valid    = energies_mean[masque_valid]
F_valid    = flux_moyens[masque_valid]
Ferr_valid = flux_erreurs[masque_valid]
Dx_valid   = dE_half_arr[masque_valid]

if len(E_valid) == 0:
    raise ValueError("Aucun point valide après soustraction fond.")

def loi_puissance_log(logE, alpha, logN0):
    return alpha * logE + logN0

def loi_puissance_odr(params, x):
    alpha, logN0 = params
    return alpha * x + logN0

logE     = np.log10(E_valid)
logF     = np.log10(F_valid)
logF_err = Ferr_valid / (F_valid * np.log(10))
logE_err = Dx_valid   / (E_valid * np.log(10))   # erreur en x en log

# ==============================================================================
# FIT ODR
# ==============================================================================
model   = Model(loi_puissance_odr)
data_odr = RealData(logE, logF, sx=logE_err, sy=logF_err)
odr_fit = ODR(data_odr, model, beta0=[-2.0, -10.0])
result  = odr_fit.run()

alpha, logN0 = result.beta
perr_odr     = result.sd_beta

print("=== FIT ODR ===")
print(f"  alpha  = {alpha:.3f} ± {perr_odr[0]:.3f}")
print(f"  logN0  = {logN0:.3f} ± {perr_odr[1]:.3f}")
print(f"  N0     = {10**logN0:.3e}")

# ==============================================================================
# MONTE-CARLO : tirages gaussiens sur barres d'erreur X et Y
# ==============================================================================
N_MC = 100000
rng  = np.random.default_rng(42)

alpha_mc = np.empty(N_MC)
logN0_mc = np.empty(N_MC)

for k in range(N_MC):
    # Tirage gaussien sur logF (erreur Y) et logE (erreur X)
    logF_mc = logF + rng.standard_normal(len(logF)) * logF_err
    logE_mc = logE + rng.standard_normal(len(logE)) * logE_err

    try:
        data_mc = RealData(logE_mc, logF_mc, sx=logE_err, sy=logF_err)
        odr_mc  = ODR(data_mc, model, beta0=[alpha, logN0])
        res_mc  = odr_mc.run()
        alpha_mc[k], logN0_mc[k] = res_mc.beta
    except Exception:
        alpha_mc[k], logN0_mc[k] = np.nan, np.nan

# Filtrage des runs divergents
valid_mc  = np.isfinite(alpha_mc) & np.isfinite(logN0_mc)
alpha_mc  = alpha_mc[valid_mc]
logN0_mc  = logN0_mc[valid_mc]

alpha_mc_mean = np.mean(alpha_mc)
alpha_mc_std  = np.std(alpha_mc,  ddof=1)
logN0_mc_mean = np.mean(logN0_mc)
logN0_mc_std  = np.std(logN0_mc,  ddof=1)

print(f"\n=== MONTE-CARLO ({N_MC} tirages, {valid_mc.sum()} valides) ===")
print(f"  alpha  = {alpha_mc_mean:.3f} ± {alpha_mc_std:.3f}  (1σ MC)")
print(f"  logN0  = {logN0_mc_mean:.3f} ± {logN0_mc_std:.3f}  (1σ MC)")
print(f"  N0     = {10**logN0_mc_mean:.3e}")

# ==============================================================================
# COURBE + ENVELOPPE MC
# ==============================================================================
masque_valid = (flux_moyens > 0) & (flux_erreurs > 0)

E_valid    = energies_mean[masque_valid]
F_valid    = flux_moyens[masque_valid]
Ferr_valid = flux_erreurs[masque_valid]
Dx_valid   = dE_half_arr[masque_valid]

E_fit  = np.logspace(np.log10(E_valid.min()), np.log10(E_valid.max()), 300)
F_fit  = 10**(loi_puissance_odr([alpha, logN0], np.log10(E_fit)))


# ==============================================================================
# FIGURE
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# ── Courbes MC ────────────────────────────────────────────────────────────────
N_PLOT   = 1000
idx_plot = np.random.choice(len(alpha_mc), size=N_PLOT, replace=False)

for i, k in enumerate(idx_plot):
    ax.plot(E_fit,
            10**(loi_puissance_odr([alpha_mc[k], logN0_mc[k]], np.log10(E_fit))),
            color="red", alpha=0.01, linewidth=0.5,
            label="Réalisations MC" if i == 0 else None)

# ── Données ───────────────────────────────────────────────────────────────────
mask_det = F_valid > 1 * Ferr_valid
mask_ul  = ~mask_det

if mask_det.any():
    ax.errorbar(E_valid[mask_det], F_valid[mask_det],
                yerr=np.abs(Ferr_valid[mask_det]),
                xerr=Dx_valid[mask_det],
                fmt="o", color="steelblue", capsize=4,
                label="Signal - Fond")

if mask_ul.any():
    ul_values = 3 * Ferr_valid[mask_ul]
    ax.errorbar(E_valid[mask_ul], ul_values,
                xerr=Dx_valid[mask_ul],
                yerr=ul_values * 0.3,
                fmt="o", color="steelblue",
                uplims=True, capsize=4, alpha=0.6,
                label="Upper limit (3σ)")

# ── Fit central ───────────────────────────────────────────────────────────────
label_fit = (r"Fit : $\alpha=" + f"{alpha:.2f}" +
             r" \pm " + f"{alpha_mc_std:.2f}$"  +
             r"  $\log_{10}(N_0)=" + f"{logN0:.2f}" +
             r" \pm " + f"{logN0_mc_std:.2f}$")

ax.plot(E_fit, F_fit, "r-", linewidth=2, label=label_fit)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Énergie (MeV)")
ax.set_ylabel(r"$d\Phi/dE$ (ph cm$^{-2}$ s$^{-1}$ MeV$^{-1}$)")
ax.set_title(f"SED — Activité {activite} (GRB cat102, fond soustrait)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig(f"SED_activite_{activite}_net_MC.png", dpi=150, bbox_inches="tight")
plt.show()
