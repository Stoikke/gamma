import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.stats import median_abs_deviation

LCFILE = 'results_simple/gtbin_selected_region.fits'
figure_name = 'flares_detected_mad_flux_version.png'
label_name = "Flux"
HDU = 1
COL_TIME = 'TIME'
COL_FLUX = 'EXPOSURE'
COL_COUNT = 'COUNTS'
COL_ERR = 'ERROR'  # ← Colonne erreur gtbin
t_start_activite_source_secondaire = 5.03625626e+08
t_stop_activite_source_secondaire  =  5.04025226e+08


hdul = fits.open(LCFILE)
lc = Table.read(LCFILE, hdu=HDU)
print("Colonnes disponibles:", lc.colnames)

time         = lc[COL_TIME].value
counts_val   = lc[COL_COUNT].value
exposure_val = lc[COL_FLUX].value
error_col    = lc[COL_ERR].value   # Erreur en counts (gtbin)
flux         = counts_val / exposure_val

# Erreur Poisson propagée : σ_flux = ERROR / exposure
err_flux = np.where(exposure_val > 0,
                    error_col / exposure_val,
                    np.nan)

# INTERVALLE DE RÉFÉRENCE
t_ref_start = 501643824.0
t_ref_stop  = 503200000.0

mask_ref      = (time >= t_ref_start) & (time <= t_ref_stop) 
flux_ref      = flux[mask_ref]
err_flux_ref  = err_flux[mask_ref]

# Stats robustes
median_flux = np.nanmedian(flux_ref)
mad_flux    = median_abs_deviation(flux_ref, nan_policy='omit', scale=1.4826)
seuil_3mad  = median_flux + 3 * mad_flux

# Erreur sur le seuil (propagation via barres d'erreur de référence)
mad_err        = np.sqrt(np.nansum(err_flux_ref**2)) / np.sum(mask_ref)
seuil_3mad_err = 3 * np.sqrt(mad_flux**2 + mad_err**2)

print(f"\n--- STATISTIQUES ROBUSTES ---")
print(f"N bins référence : {np.sum(mask_ref)}")
print(f"Médiane          : {median_flux:.8e}")
print(f"MAD (normalisé)  : {mad_flux:.8e}")
print(f"Seuil 3×MAD      : {seuil_3mad:.8e} ± {seuil_3mad_err:.8e}")

# Seuil sur tout l'échantillon
time_full = time
flux_full = flux
err_full  = err_flux

mask_time_valid = (time_full >= t_start_activite_source_secondaire) & (time_full <= t_stop_activite_source_secondaire) & (time_full <= time_full.max())
mask_flare = (flux_full > seuil_3mad + seuil_3mad_err) &~ mask_time_valid
flare_times = time_full[mask_flare] 
flare_flux  = flux_full[mask_flare]
flare_err   = err_full[mask_flare]

print(f"\n--- FLARES DÉTECTÉS ({np.sum(mask_flare)} bins > 3×MAD) ---")
for t, f, e in zip(flare_times, flare_flux, flare_err):
    print(f"  t = {t:.8e}  |  flux = {f:.8e} ± {e:.8e}  |  σ = {(f - median_flux)/mad_flux:.2f}σ")

# PLOT
fig, ax = plt.subplots(figsize=(14, 5))

# Zone référence
ax.axvspan(t_ref_start, t_ref_stop, alpha=0.1, color='blue', label='Intervalle référence')

# Courbe avec barres d'erreur
ax.errorbar(time_full, flux_full, yerr=err_full,
            fmt='.', lw=0.5, alpha=0.6,
            elinewidth=0.8, capsize=2, capthick=0.8,
            color='steelblue', ecolor='gray', label=label_name)

# Seuils
ax.axhline(median_flux,              color='green',      ls='-',  lw=1.5, label=f'Médiane = {median_flux:.4e}')
ax.axhline(median_flux + mad_flux,   color='orange',     ls='--', lw=1,   label='1×MAD')
ax.axhline(median_flux + 2*mad_flux, color='darkorange', ls='--', lw=1,   label='2×MAD')
ax.axhline(seuil_3mad,               color='red',        ls='--', lw=1.5, label=f'3×MAD = {seuil_3mad:.4e}')

# Bande erreur sur le seuil 3×MAD
ax.fill_between(time_full,
                seuil_3mad - seuil_3mad_err,
                seuil_3mad + seuil_3mad_err,
                color='red', alpha=0.15, label=f'±err 3×MAD')
# Source secondaire active
ax.axvspan(t_start_activite_source_secondaire, t_stop_activite_source_secondaire, alpha=0.1, color='black', label='Intervalle source secondaire')

# Flares avec barres d'erreur
if np.sum(mask_flare) > 0:
    ax.errorbar(flare_times, flare_flux, yerr=flare_err,
                fmt='o', color='red', ms=8, zorder=5,
                elinewidth=1.5, capsize=4, label='Flares >3×MAD')

ax.set_xlabel('Temps (MET s)')
ax.set_ylabel(label_name)
ax.set_title('LC robuste (médiane + MAD) avec barres d\'erreur')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figure_name, dpi=150)
plt.show()

# Export avec erreurs
np.savetxt('flares_detected_mad.txt',
           np.column_stack([flare_times, flare_flux, flare_err]),
           header='time_MET  flux  err_flux  (bins > 3*MAD)',
           fmt='%.8e')
print("\n✅ flares_detected_mad.txt + figure OK")
hdul.close()
