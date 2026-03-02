import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

# =====================================================
# PARAMÈTRES UTILISATEUR
# =====================================================
N_FLARES = 2   # <-- change ici (1, 2, 3...)

filename = "results_simple/gtbin_selected_region_6h_bin.fits"

# Fenêtre temporelle pour zoomer sur la zone d'intérêt
T_MIN = 5.03277224e+08# 501643824.0 
T_MAX = 503660000.0  #5.03593224e+08

# =====================================================
# LECTURE FITS
# =====================================================
with fits.open(filename) as hdul:
    data     = hdul[1].data
    t        = data["TIME"].astype(float)       # centre de chaque bin [s MET]
    flux     = data["FLUX"].astype(float)
    flux_err = data["FLUX_ERR"].astype(float)

# ── Fenêtrage temporel + masque qualité ──────────────
mask = (
    (t >= T_MIN) & (t <= T_MAX) &
    np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
)
t        = t[mask]
flux     = flux[mask]
flux_err = flux_err[mask]

print(f"{len(t)} bins temporels après masquage")
print(f"t min = {t.min():.6e}  t max = {t.max():.6e}")
print(f"flux  : min={flux.min():.3e}  max={flux.max():.3e}")

# =====================================================
# MODÈLE : flare exponentiel asymétrique
# =====================================================
def asym_exp_flare(t, A, t0, tau_r, tau_d):
    """Flare de Norris : A / (exp((t0-t)/tau_r) + exp((t-t0)/tau_d))"""
    return A / (np.exp((t0 - t) / tau_r) + np.exp((t - t0) / tau_d))

def multi_flare_model(t, *params):
    model = np.zeros_like(t, dtype=float)
    for i in range(N_FLARES):
        A, t0, tr, td = params[4*i:4*i+4]
        model += asym_exp_flare(t, A, t0, tr, td)
    return model

# =====================================================
# GUESSES AUTOMATIQUES
# =====================================================
# Sélectionne les N_FLARES pics les plus élevés comme point de départ
sorted_idx = np.argsort(flux)[-N_FLARES:]
sorted_idx = np.sort(sorted_idx)   # ordre chronologique

p0 = []
for idx in sorted_idx:
    p0.extend([
        flux[idx],        # A     : amplitude au pic
        t[idx],           # t0    : position temporelle du pic
        28800,           # tau_r : temps de montée [s] (~8h)
        25200.0,           # tau_d : temps de descente [s] (~7h)
    ])

print(f"\nGuesses initiaux :")
for i in range(N_FLARES):
    print(f"  Peak {i+1} : A={p0[4*i]:.3e}  t0={p0[4*i+1]:.6e}  "
          f"tau_r={p0[4*i+2]:.0f}s  tau_d={p0[4*i+3]:.0f}s")

# =====================================================
# BORNES
# =====================================================
dt = t[-1] - t[0]
lower_bounds, upper_bounds = [], []
for i in range(N_FLARES):
    lower_bounds.extend([0,       t.min(),        10,   10  ])
    upper_bounds.extend([1e-3,    t.max(),        dt,   dt  ])

# =====================================================
# FIT
# =====================================================
try:
    popt, pcov = curve_fit(
        multi_flare_model,
        t, flux,
        p0=p0,
        sigma=flux_err,
        absolute_sigma=True,
        bounds=(lower_bounds, upper_bounds),
        maxfev=50000
    )
    perr = np.sqrt(np.diag(pcov))
    fit_ok = True
except RuntimeError as e:
    print(f"\n[FIT ÉCHOUÉ] {e}")
    popt = p0
    perr = [np.nan] * len(p0)
    fit_ok = False

# =====================================================
# RÉSULTATS
# =====================================================
print("\n========== RÉSULTATS ==========")
for i in range(N_FLARES):
    A,  t0, tr,  td  = popt[4*i:4*i+4]
    dA, dt0, dtr, dtd = perr[4*i:4*i+4]
    print(f"\nPeak {i+1}")
    print(f"  A      = {A:.3e}  ±  {dA:.3e}  ph cm⁻² s⁻¹")
    print(f"  t0     = {t0:.6e}  ±  {dt0:.3e}  s (MET)")
    print(f"  tau_r  = {tr:.1f}  ±  {dtr:.1f}  s")
    print(f"  tau_d  = {td:.1f}  ±  {dtd:.1f}  s")

# =====================================================
# PLOT
# =====================================================
t_fit = np.linspace(t.min(), t.max(), 2000)

fig, ax = plt.subplots(figsize=(10, 5))

ax.errorbar(t, flux, yerr=flux_err,
            fmt="o", ms=4, color="steelblue",
            capsize=3, label="Données")

if fit_ok:
    ax.plot(t_fit, multi_flare_model(t_fit, *popt),
            color="black", linewidth=2, label="Fit total")

    for i in range(N_FLARES):
        ax.plot(t_fit, asym_exp_flare(t_fit, *popt[4*i:4*i+4]),
                linestyle="--", linewidth=1.5, label=f"Peak {i+1}")

ax.set_xlabel("Temps (s, MET)")
ax.set_ylabel(r"Flux (ph cm$^{-2}$ s$^{-1}$)")
ax.set_title(f"Courbe de lumière — fit {N_FLARES} peak(s)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig("light_curve_fit_3h.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure sauvegardée : light_curve_fit.png")
