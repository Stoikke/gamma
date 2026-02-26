import os, glob, sys
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from astropy.io import fits
from astropy.table import Table
import numpy as np 
from astropy.time import Time

fitsfile = 'data/Photon_projet/lat_photon_weekly_all.fits'
f1 = fits.open(fitsfile)
events = Table.read(fitsfile, hdu=1)

hdr = f1[1].header
print('START:', Time(hdr['TSTART'], format='unix').iso[:10], 
      'STOP:', Time(hdr['TSTOP'], format='unix').iso[:10])

energy = events['ENERGY']
l = events['L']
b = events['B']
time = events['TIME']
zenith_angle = events['ZENITH_ANGLE']

zenith_angle_mask = zenith_angle < 90
emin_mask = energy > 100
tot_mask = emin_mask & zenith_angle_mask

# PAS DE TEMPS 
step_sec = 3600 * 3 # ← Change ici (ex: 1 jour = 3600*24)
tmin, tmax = time.min(), time.max()
tstarts = np.arange(501643824, 504661409, step_sec)  # Tes bounds
tstarts = tstarts[(tstarts >= tmin) & (tstarts < tmax)]
n_steps = len(tstarts)
print(f"{n_steps} étapes (step={step_sec/3600/24:.1f} jours)")

# Calcul Grid dynamique
ncols = 5  # Fixe colonnes
nrows = math.ceil(n_steps / ncols)
print(f"Grid: {nrows}x{ncols}")

# === FIGURE 1: MAPS ABSOLUES ===
fig1, axs1 = plt.subplots(nrows, ncols, figsize=(20, 3*nrows))
axs1 = axs1.flatten()  # Aplatir pour itérer facile

for i in range(n_steps):
    tstart = tstarts[i]
    tstop = min(tstart + step_sec, tmax)
    mask = (time >= tstart) & (time < tstop) & tot_mask
    l_step = l[mask]
    b_step = b[mask]
    
    # Plot
    im = axs1[i].hist2d(l_step, b_step, bins=[360, 180], norm=LogNorm(), cmap='hot')[3]  # Focus direct
    axs1[i].set_title(f'{Time(tstart, format="unix").jd:.1f} ({len(l_step)})', fontsize=8)
    axs1[i].axis('off')  # Clean look

# Cache les axes vides
for j in range(i+1, len(axs1)):
    axs1[j].axis('off')

plt.suptitle(f'Maps absolues (step={step_sec/3600/24:.1f}j)', fontsize=16)
plt.tight_layout()
plt.savefig('grid_absolues.png', dpi=150)
plt.show(block=False)

# Ref pour diffs (step 0)
mask0 = (time >= tstarts[0]) & (time < tstarts[0]+step_sec) & tot_mask
l0, b0 = l[mask0], b[mask0]
histo_ref, xedges, yedges = np.histogram2d(l0, b0, bins=[360,180])

# === FIGURE 2: DIFFÉRENCES ===
fig2, axs2 = plt.subplots(nrows, ncols, figsize=(20, 3*nrows))
axs2 = axs2.flatten()

# Step 0 = Ref (vide ou ref)
axs2[0].text(0.5, 0.5, "REFERENCE", ha='center')
axs2[0].axis('off')

for i in range(1, n_steps):
    tstart = tstarts[i]
    tstop = min(tstart + step_sec, tmax)
    mask = (time >= tstart) & (time < tstop) & tot_mask
    l_step = l[mask]
    b_step = b[mask]
    
    histo_i, _, _ = np.histogram2d(l_step, b_step, bins=[360, 180])
    diff = histo_i - histo_ref
    
    vmax = max(1, np.max(np.abs(diff)))
    axs2[i].pcolormesh(xedges, yedges, diff.T, cmap='RdBu_r', 
                      norm=SymLogNorm(1, vmin=-vmax, vmax=vmax))
    axs2[i].set_title(f'Δ {Time(tstart, format="unix").jd:.1f}', fontsize=8)
    axs2[i].axis('off')

for j in range(n_steps, len(axs2)):
    axs2[j].axis('off')

plt.suptitle(f'Différences vs Step 0 (Rouge=+, Bleu=-)', fontsize=16)
plt.tight_layout()
plt.savefig('grid_diffs.png', dpi=150)
plt.show()

f1.close()
