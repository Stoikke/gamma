import os, glob, sys

# Set up matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.io import fits
from astropy.table import Table
import numpy as np				### numarray or numpy depends on the python install

fitsfile='vela-16y-emin1000-roi10-zmax180.fits'    # put here the path to your favorite file
f1 = fits.open(fitsfile)

fits.info(fitsfile)  # renvoie des infos sur les extensions 
events = Table.read(fitsfile,hdu=1)  #selectionne l'extension 1 du fichier

#print header
hdr = f1[1].header
#print(hdr)   #imprime une tonne d'information sur le contenu de l'extension

# start and stop of date
print('START: ',hdr['TSTART'],'STOP: ',hdr['TSTOP'])  # donne le debut et la fin des data


print(events.columns)  # print column names
print(events['ENERGY'].unit)  # donne les unités de la colonne sélectionee
print(events['ENERGY'][0]) # donne la valeur de l'ENERGY pour le 1er evenement du fichier

fig, axs = plt.subplots(2, 1, figsize=(8, 8))

energy = events['ENERGY']# lit et stocke dans la table "energy" l'energie de tous les evenements
l = events['L'] # lit et stocke dans la table "l" la longitude galactique de tous les evenements 
b = events['B'] # idem pour la latitude galactique
ra = events['RA'] # idem pour l'ascension droite
dec = events['DEC'] # idem pour la declinaison
zenith_angle = events['ZENITH_ANGLE']

# definition de masque (coupure, critere de selection) pour filtrer les evenements
zenith_angle_mask = zenith_angle < 180 # ne garde que les evenements dont le zenith_angle est plus grand que 180
emin_mask = energy > 1000 # ne garde que les evenements dont l'energie est sup à 100 MeV
tot_mask = zenith_angle_mask & emin_mask  # les 2 masques a la fois


# partie graphique

axs[0].set_xlabel('L (deg.)', fontsize=10)
axs[0].set_ylabel('B (deg.)', fontsize=10)
# this is to fill and plot a 2D histogram
#counts0, xedges0, yedges0, im0 = axs[0].hist2d(l,b,bins=[360,180],norm=LogNorm())
counts0, xedges0, yedges0, im0 = axs[0].hist2d(ra,dec,bins=[360,180],norm=LogNorm())
fig.colorbar(im0,ax=axs[0])

# this is to display a scatter plot
#axs[1].scatter(l,b,marker='o',s=0.0001)

l_cut = l[tot_mask] # on applique le masque a la longitude galactique
b_cut = b[tot_mask] # masque la latitude galac.
axs[1].set_xlabel('L (deg.)', fontsize=10)
axs[1].set_ylabel('B (deg.)', fontsize=10)
# this is to fill and plot a 2D histogram
#counts1, xedges1, yedges1, im1 = axs[1].hist2d(l_cut,b_cut,bins=[360,180],norm=LogNorm())
counts1, xedges1, yedges1, im1 = axs[1].hist2d(l,b,bins=[360,180])
fig.colorbar(im1,ax=axs[1])

# this is to display a scatter plot
#axs[1].scatter(l,b,marker='o',s=0.0001)

plt.show()

exit(0)
