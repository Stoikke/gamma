import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit

# =====================================================
# PARAMÈTRES UTILISATEUR
# =====================================================
N_MC = 1000

filename       = "results_simple/lat_photon_flare1article_selected_radius4_LC_p6h.fits"
filenames_fond = [
    "SED_output/FOND_peak/gt_bin_activite_1_fond_1_100_20000.fits",
    "SED_output/FOND_peak/gt_bin_activite_1_fond_2_100_20000.fits",
    "SED_output/FOND_peak/gt_bin_activite_1_fond_3_100_20000.fits",
]

T_MIN = 5.03277224e+08
T_MAX = 503660000.0

# Plages temporelles indépendantes par pic : (t0_guess, t_min, t_max)
PEAKS = [
    (5.034026e+08, 5.033272e+08, 5.034747e+08),  # Peak 1
    (5.035561e+08, 5.034757e+08, 5.036444e+08),  # Peak 2
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
        fonds_mean.append(fm); fonds_err.append(fe)
        print(f"  Fond {fp.split('/')[-1]} : {fm:.3e} ± {fe:.3e}")
    weights        = 1.0 / np.array(fonds_err)**2
    flux_fond_mean = np.average(fonds_mean, weights=weights)
    flux_fond_err  = 1.0 / np.sqrt(np.sum(weights))
    print(f"\nFond moyen (pondéré) = {flux_fond_mean:.3e} ± {flux_fond_err:.3e}")
except FileNotFoundError as e:
    print(f"[ATTENTION] {e}\n  → fond désactivé")

# =====================================================
# LECTURE SIGNAL (fenêtre globale)
# =====================================================
with fits.open(filename) as hdul:
    data = hdul[1].data
    t_all        = data["TIME"].astype(float)
    flux_all     = data["FLUX"].astype(float)
    flux_err_all = data["FLUX_ERR"].astype(float)

mask_global = ((t_all >= T_MIN) & (t_all <= T_MAX) &
               np.isfinite(flux_all) & np.isfinite(flux_err_all) & (flux_err_all > 0))
t_all        = t_all[mask_global]
flux_all     = flux_all[mask_global]
flux_err_all = flux_err_all[mask_global]

flux_net_all     = flux_all - flux_fond_mean
flux_net_err_all = np.sqrt(flux_err_all**2 + flux_fond_err**2)

# =====================================================
# MODÈLE : un seul flare
# =====================================================
def asym_exp_flare(t, A, t0, tau_r, tau_d):
    return A / (np.exp((t0 - t) / tau_r) + np.exp((t - t0) / tau_d))

# =====================================================
# FIT INDÉPENDANT PAR PIC
# =====================================================
rng = np.random.default_rng(42)

results   = []   # stocke (popt, mc_std) par pic
mc_all    = []   # stocke mc_params par pic (pour le plot)

for i, (t0_guess, t_min, t_max) in enumerate(PEAKS):
    print(f"\n{'='*55}")
    print(f"  Peak {i+1}  |  fenêtre [{t_min:.4e} – {t_max:.4e}] s")
    print(f"{'='*55}")

    # ── Sélection de la fenêtre de ce pic ────────────────
    mask_peak = (t_all >= t_min) & (t_all <= t_max)
    t_p   = t_all[mask_peak]
    f_p   = flux_net_all[mask_peak]
    fe_p  = flux_net_err_all[mask_peak]

    if len(t_p) < 4:
        print(f"  [SKIP] seulement {len(t_p)} bins dans la fenêtre")
        results.append(None)
        mc_all.append(None)
        continue

    # ── Guess + bornes ────────────────────────────────────
    idx_c   = np.argmin(np.abs(t_p - t0_guess))
    A_guess = max(f_p[idx_c], 1e-8)
    dt_peak = t_max - t_min

    p0_i = [A_guess, t0_guess, dt_peak / 3, dt_peak / 3]
    lb_i = [0,      t_min,    10,           10          ]
    ub_i = [1e-3,   t_max,    dt_peak,      dt_peak     ]

    print(f"  Guess : A={A_guess:.3e}  t0={t0_guess:.4e}  "
          f"tau_r={p0_i[2]:.0f}s  tau_d={p0_i[3]:.0f}s")

    def do_fit_peak(flux_data, flux_sigma):
        return curve_fit(
            asym_exp_flare, t_p, flux_data,
            p0=p0_i, sigma=flux_sigma, absolute_sigma=True,
            bounds=(lb_i, ub_i), maxfev=50000
        )

    # ── Fit principal ─────────────────────────────────────
    try:
        popt_i, pcov_i = do_fit_peak(f_p, fe_p)
        fit_ok_i = True
    except RuntimeError as e:
        print(f"  [FIT ÉCHOUÉ] {e}")
        popt_i, fit_ok_i = np.array(p0_i), False

    # ── Monte-Carlo ───────────────────────────────────────
    mc_i = np.full((N_MC, 4), np.nan)
    for k in range(N_MC):
        f_mc   = f_p + rng.standard_normal(len(f_p)) * fe_p
        fond_k = flux_fond_mean + rng.standard_normal() * flux_fond_err
        f_mc  -= (fond_k - flux_fond_mean)
        try:
            popt_mc, _ = do_fit_peak(f_mc, fe_p)
            mc_i[k]   = popt_mc
        except Exception:
            pass

    valid_i = np.all(np.isfinite(mc_i), axis=1)
    mc_i    = mc_i[valid_i]
    mc_std_i = np.std(mc_i, ddof=1, axis=0)

    print(f"  MC : {mc_i.shape[0]}/{N_MC} valides")

    # ── Résultats ─────────────────────────────────────────
    A, t0, tr, td       = popt_i
    dA, dt0, dtr, dtd   = mc_std_i
    print(f"  A      = {A:.3e}  ±  {dA:.3e}  ph cm⁻² s⁻¹")
    print(f"  t0     = {t0:.6e}  ±  {dt0:.3e}  s (MET)")
    print(f"  tau_r  = {tr:.1f}  ±  {dtr:.1f}  s")
    print(f"  tau_d  = {td:.1f}  ±  {dtd:.1f}  s")

    results.append((popt_i, mc_std_i, fit_ok_i, t_p, f_p, fe_p, lb_i, ub_i))
    mc_all.append(mc_i)

# =====================================================
# PLOT
# =====================================================
colors = ["tomato", "mediumseagreen", "mediumpurple", "goldenrod"]
t_fit_global = np.linspace(t_all.min(), t_all.max(), 3000)

fig, ax = plt.subplots(figsize=(12, 5))

# Données globales
ax.errorbar(t_all, flux_all, yerr=flux_err_all,
            fmt="o", ms=4, color="gray", capsize=3, alpha=0.4,
            label="Données brutes")
ax.errorbar(t_all, flux_net_all, yerr=flux_net_err_all,
            fmt="o", ms=4, color="steelblue", capsize=3,
            label="Signal - Fond")

# Fond
ax.axhline(flux_fond_mean, color="orange", linestyle="--", linewidth=1.2,
           label=f"Fond = {flux_fond_mean:.2e}")
ax.axhspan(flux_fond_mean - flux_fond_err,
           flux_fond_mean + flux_fond_err,
           color="orange", alpha=0.15)

# Par pic
for i, res in enumerate(results):
    if res is None:
        continue
    popt_i, mc_std_i, fit_ok_i, t_p, f_p, fe_p, lb_i, ub_i = res
    mc_i   = mc_all[i]
    color  = colors[i % len(colors)]
    t_min_p, t_max_p = PEAKS[i][1], PEAKS[i][2]
    t_fit_i = np.linspace(t_min_p, t_max_p, 1000)

    # Fenêtre temporelle
    ax.axvspan(t_min_p, t_max_p, alpha=0.07, color=color)

    # Réalisations MC
    N_PLOT = min(300, len(mc_i))
    idx_plot = rng.choice(len(mc_i), size=N_PLOT, replace=False)
    for ii, k in enumerate(idx_plot):
        ax.plot(t_fit_i, asym_exp_flare(t_fit_i, *mc_i[k]),
                color=color, alpha=0.02, linewidth=0.5,
                label=f"MC Peak {i+1}" if ii == 0 else None)

    # Fit central
    if fit_ok_i:
        ax.plot(t_fit_i, asym_exp_flare(t_fit_i, *popt_i),
                color=color, linewidth=2,
                label=f"Peak {i+1} : t0={popt_i[1]:.3e}s")

ax.set_xlabel("Temps (s, MET)")
ax.set_ylabel(r"Flux (ph cm$^{-2}$ s$^{-1}$)")
ax.set_title(f"Courbe de lumière — {len(PEAKS)} pics indépendants (fond soustrait, MC={N_MC})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.savefig("light_curve_independent_fits.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure sauvegardée : light_curve_independent_fits.png")