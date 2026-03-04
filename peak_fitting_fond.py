import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit
import corner

# =====================================================
# PARAMÈTRES UTILISATEUR
# =====================================================
N_FLARES = 2
N_MC     = 1000   # nombre de tirages Monte-Carlo

filename       = "results_simple/lat_photon_flare1article_selected_radius4_LC_p6h.fits"
filenames_fond = [
    "SED_output/FOND_peak/gt_bin_activite_1_fond_1_100_20000.fits",
    "SED_output/FOND_peak/gt_bin_activite_1_fond_2_100_20000.fits",
    "SED_output/FOND_peak/gt_bin_activite_1_fond_3_100_20000.fits",
]

T_MIN = 503280004+13*3600  # 5.03277224e+08
T_MAX = 503280004+4*86400
PEAK_TIME_CONSTRAINTS = [
    (5.033985e+8, 5.033327e+08, 5.034666e+08),   # Peak 1
    (5.035719e+08, 5.034866e+08, 5.036153e+08),   # Peak 2
]

# =====================================================
# LECTURE FOND
# =====================================================
flux_fond_mean = 0.0
flux_fond_err  = 0.0

def lire_fond(filepath):
    with fits.open(filepath) as hdul:
        cols_upper = [c.upper() for c in hdul[1].columns.names]
        data_fond  = Table(hdul[1].data)
        if "FLUX" in cols_upper and "FLUX_ERR" in cols_upper:
            f  = data_fond["FLUX"].data.astype(float).flatten()
            fe = data_fond["FLUX_ERR"].data.astype(float).flatten()
        elif "COUNTS" in cols_upper and "EXPOSURE" in cols_upper:
            counts   = data_fond["COUNTS"].data.astype(float).flatten()
            exposure = data_fond["EXPOSURE"].data.astype(float).flatten()
            errors   = (data_fond["ERROR"].data.astype(float).flatten()
                        if "ERROR" in cols_upper else np.sqrt(counts))
            mask_exp = exposure > 0
            f, fe    = counts[mask_exp]/exposure[mask_exp], errors[mask_exp]/exposure[mask_exp]
        else:
            raise ValueError(f"Colonnes manquantes : {cols_upper}")
    mask = np.isfinite(f) & np.isfinite(fe) & (fe > 0)
    f, fe = f[mask], fe[mask]
    return np.mean(f), np.sqrt(np.sum(fe**2)) / len(f)

try:
    fonds_mean, fonds_err = [], []
    for fp in filenames_fond:
        fm, fe = lire_fond(fp)
        fonds_mean.append(fm)
        fonds_err.append(fe)
        print(f"  Fond {fp.split('/')[-1]} : {fm:.3e} ± {fe:.3e}")
    weights        = 1.0 / np.array(fonds_err)**2
    flux_fond_mean = np.average(fonds_mean, weights=weights)
    flux_fond_err  = 1.0 / np.sqrt(np.sum(weights))
    print(f"\nFond moyen (pondéré) = {flux_fond_mean:.3e} ± {flux_fond_err:.3e}")
except FileNotFoundError as e:
    print(f"[ATTENTION] {e}\n  → fond désactivé")

# =====================================================
# LECTURE SIGNAL
# =====================================================
with fits.open(filename) as hdul:
    data     = hdul[1].data
    t        = data["TIME"].astype(float)
    flux     = data["FLUX"].astype(float)
    flux_err = data["FLUX_ERR"].astype(float)

mask = ((t >= T_MIN) & (t <= T_MAX) &
        np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0))
t, flux, flux_err = t[mask], flux[mask], flux_err[mask]

flux_net     = flux - flux_fond_mean
flux_net_err = np.sqrt(flux_err**2 + flux_fond_err**2)

print(f"\n{len(t)} bins | flux net : min={flux_net.min():.3e}  max={flux_net.max():.3e}")

# =====================================================
# MODÈLE
# =====================================================
def asym_exp_flare(t, A, t0, tau_r, tau_d):
    return A / (np.exp((t0 - t) / tau_r) + np.exp((t - t0) / tau_d))

def multi_flare_model(t, *params):
    model = np.zeros_like(t, dtype=float)
    for i in range(N_FLARES):
        if i == 0:
            A, t0, tr, td = params[4*i:4*i+4]
            model += asym_exp_flare(t, A, t0, tr, tr)
        else:
            A, t0, tr, td = params[4*i:4*i+4]
            model += asym_exp_flare(t, A, t0, tr, td)
    return model
# =====================================================
# GUESSES + BORNES  
# =====================================================
p0           = []
lower_bounds = []
upper_bounds = []

dt = t[-1] - t[0]

for i in range(N_FLARES):
    t0_guess, t0_min, t0_max = PEAK_TIME_CONSTRAINTS[i]

    # Amplitude : flux net au bin le plus proche de t0_guess
    idx_closest = np.argmin(np.abs(t - t0_guess))
    A_guess = max(flux_net[idx_closest], 1e-8)

    p0.extend([A_guess, t0_guess, 28800.0, 25200.0])

    lower_bounds.extend([0,      t0_min, 10, 10])
    upper_bounds.extend([1e-3,   t0_max, dt, dt])

print("\nGuesses initiaux :")
for i in range(N_FLARES):
    t0_guess, t0_min, t0_max = PEAK_TIME_CONSTRAINTS[i]
    print(f"  Peak {i+1} : A={p0[4*i]:.3e}  t0={p0[4*i+1]:.6e}"
          f"  t0 ∈ [{t0_min:.3e}, {t0_max:.3e}]"
          f"  tau_r={p0[4*i+2]:.0f}s  tau_d={p0[4*i+3]:.0f}s")

# =====================================================
# FIT PRINCIPAL, maxfev=50000
# =====================================================
def do_fit(flux_data, flux_sigma):
    return curve_fit(
        multi_flare_model, t, flux_data,
        p0=p0, sigma=flux_sigma, absolute_sigma=True,
        bounds=(lower_bounds, upper_bounds),maxfev=50000
    )

try:
    popt, pcov = do_fit(flux_net, flux_net_err)
    perr       = np.sqrt(np.diag(pcov))
    fit_ok     = True
except RuntimeError as e:
    print(f"\n[FIT ÉCHOUÉ] {e}")
    popt, perr, fit_ok = np.array(p0), [np.nan]*len(p0), False

# =====================================================
# MONTE-CARLO : tirages gaussiens sur flux_net
# =====================================================
rng     = np.random.default_rng(42)
n_par   = 4 * N_FLARES
mc_params = np.full((N_MC, n_par), np.nan)

print(f"\nMonte-Carlo : {N_MC} tirages...")
n_ok = 0
for k in range(N_MC):
    # Tirage sur le flux net (erreur signal + fond)
    flux_mc = flux_net + rng.standard_normal(len(flux_net)) * flux_net_err
    # Tirage sur le fond (propage l'incertitude du fond)
    fond_mc = flux_fond_mean + rng.standard_normal() * flux_fond_err
    flux_mc_net = flux_mc - (fond_mc - flux_fond_mean)  # double soustraction cohérente

    try:
        popt_mc, _ = do_fit(flux_mc_net, flux_net_err)
        mc_params[k] = popt_mc
        n_ok += 1
    except Exception:
        pass
valid_mc  = np.all(np.isfinite(mc_params), axis=1)
mc_params = mc_params[valid_mc]
print(f"  {mc_params.shape[0]}/{N_MC} fits MC valides")

# Médiane et MAD par paramètre (sur les tirages MC)
mc_median = np.median(mc_params, axis=0)
mc_mad    = np.median(np.abs(mc_params - mc_median), axis=0)

# Filtre : garder les tirages où TOUS les paramètres sont dans médiane ± 1 MAD
bonfit = np.any(np.abs(mc_params - mc_median) < 3 * mc_mad, axis=1)  # shape (N,)
mc_params = mc_params[bonfit]   # reste une matrice (N_valid, n_params)

# Recalcul sur les tirages filtrés
mc_median = np.median(mc_params, axis=0)
mc_mad    = np.median(np.abs(mc_params - mc_median), axis=0)

print(f"  Tirages retenus après filtre MAD : {mc_params.shape[0]}/{bonfit.size}")


# =====================================================
# RÉSULTATS
# =====================================================
print("\n========== RÉSULTATS (médiane MC ± MAD) ==========")
for i in range(N_FLARES):
    A,  t0, tr,  td   = mc_median[4*i:4*i+4]
    dA, dt0, dtr, dtd = mc_mad[4*i:4*i+4]
    print(f"\nPeak {i+1}")
    print(f"  A      = {A:.3e}  ±  {dA:.3e}  ph cm⁻² s⁻¹  (MAD)")
    print(f"  t0     = {t0:.6e}  ±  {dt0:.3e}  s (MET)")
    print(f"  tau_r  = {tr:.1f}  ±  {dtr:.1f}  s")
    print(f"  tau_d  = {td:.1f}  ±  {dtd:.1f}  s")


# =====================================================
# PLOT
# =====================================================
t_fit  = np.linspace(t.min(), t.max(), 2000)
N_PLOT = min(500, len(mc_params))
idx_plot = rng.choice(len(mc_params), size=N_PLOT, replace=False)

fig, ax = plt.subplots(figsize=(10, 5))

# Réalisations MC (fond + transparence)
for ii, k in enumerate(idx_plot):
    ax.plot(t_fit, multi_flare_model(t_fit, *mc_params[k]),
            color="gray", alpha=0.02, linewidth=0.5,
            label="Réalisations MC" if ii == 0 else None)

# Données
ax.errorbar(t, flux, yerr=flux_err,
            fmt="o", ms=4, color="gray", capsize=3, alpha=0.4,
            label="Données brutes")
ax.errorbar(t, flux_net, yerr=flux_net_err,
            fmt="o", ms=4, color="steelblue", capsize=3,
            label="Signal - Fond")

# Niveau du fond
ax.axhline(flux_fond_mean, color="orange", linestyle="--", linewidth=1.2,
           label=f"Fond = {flux_fond_mean:.2e}")
ax.axhspan(flux_fond_mean - flux_fond_err,
           flux_fond_mean + flux_fond_err,
           color="orange", alpha=0.15)

# Fit central
if fit_ok:
    ax.plot(t_fit, multi_flare_model(t_fit, *popt),
            color="black", linewidth=2, label="Fit total")
    for i in range(N_FLARES):
        ax.plot(t_fit, asym_exp_flare(t_fit, *popt[4*i:4*i+4]),
                linestyle="--", linewidth=1.5, label=f"Peak {i+1}")
print(max(t))
ax.set_xlabel("Temps (s, MET)")
ax.set_ylabel(r"Flux (ph cm$^{-2}$ s$^{-1}$)")
# ax.set_ylim(0e-5,2.5e-5)
ax.set_title(f"Courbe de lumière — {N_FLARES} peak(s), fond soustrait, MC={N_MC}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.savefig("light_curve_fit_6h_MC_article.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure sauvegardée : light_curve_fit_MC.png")