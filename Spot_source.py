import os, glob, sys

# Set up matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider

from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
import numpy as np				### numarray or numpy depends on the python install


fitsfile='results_simple/selected_region.fits'    # put here the path to your favorite file
#vela-16y-emin1000-roi10-zmax180
f1 = fits.open(fitsfile)

fits.info(fitsfile)  # renvoie des infos sur les extensions
events = Table.read(fitsfile,hdu=1)  #selectionne l'extension 1 du fichier

#print header
hdr = f1[1].header
#print(hdr)   #imprime une tonne d'information sur le contenu de l'extension

# start and stop of date
print('START: ',hdr['TSTART'],'STOP: ',hdr['TSTOP'])  # donne le debut et la fin des data


print(events.columns)  # print column names -> En Tête
print(events['ENERGY'].unit)  # donne les unités de la colonne sélectionee
print(events['ENERGY'][0]) # donne la valeur de l'ENERGY pour le 1er evenement du fichier

fig, axs = plt.subplots(1, 2, figsize=(20, 6))

energy = events['ENERGY']# lit et stocke dans la table "energy" l'energie de tous les evenements
l = events['L'] # lit et stocke dans la table "l" la longitude galactique de tous les evenements
b = events['B'] # idem pour la latitude galactique
ra = events['RA'] # idem pour l'ascension droite
dec = events['DEC'] # idem pour la declinaison
time = events['TIME']
zenith_angle = events['ZENITH_ANGLE']

# definition de masque (coupure, critere de selection) pour filtrer les evenements
zenith_angle_mask = zenith_angle < 90 # ne garde que les evenements dont le zenith_angle est plus grand que 180
emin_mask = energy > 30# ne garde que les evenements dont l'energie est sup à 8000 MeV
#l_mask = (l > -50 ) & (l < -30 )
#b_mask = (b > 65 ) & (b < 85 )

def time_mask(t0,dt):
    time_mask = (time > t0) & (time < t0 + dt)
    tot_mask = zenith_angle_mask & emin_mask & time_mask #& b_mask & l_mask   # les 2 masques a la fois
    l_cut = l[tot_mask]  # on applique le masque a la longitude galactique
    b_cut = b[tot_mask]  # masque la latitude galac.
    return l_cut, b_cut

def build_map(t0, dt):

    H, xedges, yedges = np.histogram2d(
        time_mask(t0,dt)[0],
        time_mask(t0,dt)[1],
        bins=[10*360, 10*180],
    )

    return H.T

# partie graphique
t0_init =time[0]
dt_init = 86400

fig, ax = plt.subplots()

plt.subplots_adjust(bottom=0.3)

img = ax.imshow(
    build_map(t0_init, dt_init),
    origin="lower",
    norm=LogNorm()
)
cbar = plt.colorbar(img, ax=ax)
title = ax.set_title("")


ax_t0 = plt.axes([0.2, 0.18, 0.6, 0.03])
slider_t0 = Slider(
    ax_t0, 't₀',
    time.min(), time.max(),
    valinit=t0_init
)

ax_dt = plt.axes([0.2, 0.12, 0.6, 0.03])
slider_dt = Slider(
    ax_dt, 'Δt',
    3*3600, (time.max() - time.min())/4,
    valinit=dt_init
)

def update(val):
    t0 = slider_t0.val
    dt = slider_dt.val

    new_map = build_map(t0, dt)
    img.set_data(new_map)
    #img.autoscale()

    title.set_text(
        f"t₀ = {t0:.89} | Δt = {dt:.9e} "
    )

    fig.canvas.draw_idle()


slider_t0.on_changed(update)
slider_dt.on_changed(update)

def on_key(event):
    if event.key == 'right':
        slider_t0.set_val(slider_t0.val + slider_dt.val)
    elif event.key == 'left':
        slider_t0.set_val(slider_t0.val - slider_dt.val)
    elif event.key == 'up':
        slider_dt.set_val(slider_dt.val * 1.2)
    elif event.key == 'down':
        slider_dt.set_val(slider_dt.val / 1.2)

fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
