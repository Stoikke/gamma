import os, subprocess, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit

import subprocess
try:
    from GtApp import GtApp
    USE_GTAPP = True
    print("✅ GtApp disponible")
except ImportError:
    USE_GTAPP = False
    print("⚠ GtApp non disponible, fallback subprocess")

# ==============================================================
# ==================== CONFIGURATION ==========================
# ==============================================================

# --- Fichiers ---
EVFILE_FT1   = 'results_simple/selected_region.fits'
SCFILE       = 'data/Photon_projet/spacecraft_projet/L2602050555366B2DD36C49_SC00.fits'
FLARES_FILE  = 'flares_detected_mad.txt'   # ← Ton fichier de flares
OUTDIR       = 'SED_output'

# --- Source ---
RA_SRC  = 338.93076
DEC_SRC =  11.900912
RAD_ROI =  4       # degrés

# --- Énergie ---
EMIN    = 100.0    # MeV
EMAX    = 300000.0 # MeV
N_EBINS = 10       # Bins log-espacés

# --- IRF ---
IRF      = 'CALDB'
SPEC_IDX = -2.1

# --- Période calme de référence (MET) ---
T_QUIET_START = 501643824.0
T_QUIET_STOP  = 502500000.0
T_QUIET_COLOR = "steelblue"

# --- Gap max entre bins contigus pour grouper les flares (secondes) ---
# Les bins dans ton fichier sont espacés de ~10800s (3h)
# On considère que 2 bins consécutifs appartiennent au même groupe
# si leur écart est <= GAP_MAX
GAP_MAX = 32400.0  # 3 * 10800s = tolérance 3 bins manquants

# --- Couleurs automatiques pour les zones actives ---
ACTIVE_COLORS = ["red", "darkorange", "purple", "crimson", "gold", "green"]

# --- Options gtbin ---
GTBIN_OPTS = {
    'algorithm': 'LC',
    'tbinalg':   'LIN',
    'dtime':     '10800',
    'scfile':    SCFILE,

}

# --- Options gtexposure ---
GTEXP_OPTS = {
    'irfs':   IRF,
    'srcmdl': 'none',
    'specin': str(SPEC_IDX),
    'ra':     str(RA_SRC),
    'dec':    str(DEC_SRC),
    'rad':    str(RAD_ROI),
    'apcorr': 'no',
}

# ==============================================================
# ==================== LECTURE FICHIER FLARES =================
# ==============================================================

def load_periods_from_flares(flares_file, gap_max, colors):
    """
    Lit le fichier de flares et regroupe les bins contigus
    en zones d'activité (périodes).
    Retourne une liste de dicts {"label", "tmin", "tmax", "color"}
    """
    data = np.loadtxt(flares_file, comments='#')
    times = data[:, 0]
    times_sorted = np.sort(times)

    print(f"\n{'='*60}")
    print(f"DÉTECTION DES ZONES D'ACTIVITÉ depuis {flares_file}")
    print(f"  {len(times_sorted)} bins de flares, gap_max={gap_max:.0f}s")
    print(f"{'='*60}")

    # Regroupe par gap
    groups = []
    current_group = [times_sorted[0]]
    for i in range(1, len(times_sorted)):
        if times_sorted[i] - times_sorted[i-1] <= gap_max:
            current_group.append(times_sorted[i])
        else:
            groups.append(current_group)
            current_group = [times_sorted[i]]
    groups.append(current_group)

    # Construit PERIODS
    periods = []
    for i, group in enumerate(groups):
        tmin  = min(group)
        tmax  = max(group)
        label = f"active{i+1}"
        color = colors[i % len(colors)]
        periods.append({"label": label, "tmin": tmin, "tmax": tmax, "color": color})
        print(f"  Zone {i+1}: {label} | [{tmin:.6e}, {tmax:.6e}] | {len(group)} bins")

    return periods

# Chargement automatique des périodes actives
ACTIVE_PERIODS = load_periods_from_flares(FLARES_FILE, GAP_MAX, ACTIVE_COLORS)

# Ajoute la période calme
QUIET_PERIOD = {"label": "quiet", "tmin": T_QUIET_START,
                "tmax": T_QUIET_STOP, "color": T_QUIET_COLOR}

# Toutes les périodes
ALL_PERIODS = ACTIVE_PERIODS + [QUIET_PERIOD]

print(f"\nPériodes totales à traiter: {len(ALL_PERIODS)}")
for p in ALL_PERIODS:
    print(f"  {p['label']:10s} [{p['tmin']:.4e}, {p['tmax']:.4e}]  color={p['color']}")

# ==============================================================
# ==================== FONCTIONS OUTILS =======================
# ==============================================================

def run_cmd(cmd, label=""):
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  ❌ ERREUR {label}:\n{res.stderr.strip()[-300:]}")
        return False
    print(f"  ✅ {label} OK")
    return True

def gtselect_energy(infile, outfile, emin, emax, tmin, tmax):
    cmd = ['gtselect',
           f'infile={infile}', f'outfile={outfile}',
           f'ra={RA_SRC}', f'dec={DEC_SRC}', f'rad={RAD_ROI}',
           f'tmin={tmin}', f'tmax={tmax}',
           f'emin={emin:.4f}', f'emax={emax:.4f}',
           'evclass=128', 'zmax=90', 'clobber=yes']
    return run_cmd(cmd, f"gtselect E=[{emin:.0f},{emax:.0f}]MeV")

def gtbin_lc(evfile, outfile, tmin, tmax):
    try:
        if USE_GTAPP:
            gtbin = GtApp('gtbin')
            gtbin['evfile']    = evfile
            gtbin['outfile']   = outfile
            gtbin['scfile']    = GTBIN_OPTS['scfile']
            gtbin['algorithm'] = GTBIN_OPTS['algorithm']
            gtbin['tbinalg']   = GTBIN_OPTS['tbinalg']
            gtbin['tstart']    = tmin
            gtbin['tstop']     = tmax
            gtbin['dtime']     = GTBIN_OPTS['dtime']
            gtbin['clobber']   = 'yes'
            gtbin.run()
        else:
            params = '\n'.join([evfile, outfile, GTBIN_OPTS['scfile'],
                                GTBIN_OPTS['algorithm'], str(tmin), str(tmax),
                                GTBIN_OPTS['tbinalg'], GTBIN_OPTS['dtime'], 'yes'])
            proc = subprocess.Popen(['gtbin'], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = proc.communicate(input=params)
            if proc.returncode != 0:
                print(f"  ❌ gtbin:\n{err[-200:]}")
                return False
        print(f"  ✅ gtbin LC OK")
        return True
    except Exception as e:
        print(f"  ❌ gtbin exception: {e}")
        return False

def gtexposure(lcfile, scfile, emin, emax):
    try:
        if USE_GTAPP:
            gtexp = GtApp('gtexposure')
            gtexp['infile']  = lcfile
            gtexp['scfile']  = scfile
            gtexp['irfs']    = GTEXP_OPTS['irfs']
            gtexp['srcmdl']  = GTEXP_OPTS['srcmdl']
            gtexp['specin']  = GTEXP_OPTS['specin']
            gtexp['ra']      = GTEXP_OPTS['ra']
            gtexp['dec']     = GTEXP_OPTS['dec']
            gtexp['rad']     = GTEXP_OPTS['rad']
            gtexp['emin']    = emin
            gtexp['emax']    = emax
            gtexp['apcorr']  = GTEXP_OPTS['apcorr']
            gtexp['clobber'] = 'yes'
            gtexp.run()
        else:
            params = '\n'.join([lcfile, scfile, GTEXP_OPTS['irfs'],
                                GTEXP_OPTS['srcmdl'], GTEXP_OPTS['specin'],
                                GTEXP_OPTS['ra'], GTEXP_OPTS['dec'], GTEXP_OPTS['rad'],
                                str(emin), str(emax), GTEXP_OPTS['apcorr'], 'yes'])
            proc = subprocess.Popen(['gtexposure'], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = proc.communicate(input=params)
            if proc.returncode != 0:
                print(f"  ❌ gtexposure:\n{err[-200:]}")
                return False
        print(f"  ✅ gtexposure OK")
        return True
    except Exception as e:
        print(f"  ❌ gtexposure exception: {e}")
        return False

def read_lc_flux(lcfile, emin, emax):
    try:
        lc = Table.read(lcfile, hdu=1)
        c  = float(lc['COUNTS'][0])
        ex = float(lc['EXPOSURE'][0])
        dE = emax - emin
        if ex > 0 and c >= 0:
            F  = c / ex
            dF = np.sqrt(max(c, 1)) / ex
            return c, ex, F/dE, dF/dE
    except Exception as e:
        print(f"  Erreur lecture {lcfile}: {e}")
    return 0, 0, np.nan, np.nan

# ==============================================================
# ==================== PIPELINE SED ===========================
# ==============================================================

def run_sed_pipeline(period_label, tmin, tmax):
    print(f"\n{'='*60}")
    print(f"SED PIPELINE — {period_label} [{tmin:.4e}, {tmax:.4e}]")
    print(f"{'='*60}")

    period_dir = os.path.join(OUTDIR, period_label)
    os.makedirs(period_dir, exist_ok=True)

    e_edges = np.logspace(np.log10(EMIN), np.log10(EMAX), N_EBINS + 1)
    e_cents = np.sqrt(e_edges[:-1] * e_edges[1:])
    energies, fluxes, dfluxes, counts_list, expo_list = [], [], [], [], []

    for j in range(N_EBINS):
        emin_j, emax_j = e_edges[j], e_edges[j+1]
        print(f"\n  Bin {j+1:02d}/{N_EBINS}: E=[{emin_j:.1f}, {emax_j:.1f}] MeV")

        sel_file = os.path.join(period_dir, f"gtselect_{period_label}_E{emin_j:.0f}-{emax_j:.0f}MeV.fits")
        lc_file  = os.path.join(period_dir, f"gtbin_lc_{period_label}_E{emin_j:.0f}-{emax_j:.0f}MeV.fits")

        if not gtselect_energy(EVFILE_FT1, sel_file, emin_j, emax_j, tmin, tmax): continue
        try:
            ev = Table.read(sel_file, hdu=1)
            if len(ev) == 0: print("  ⚠ 0 photon, skip"); continue
            print(f"  {len(ev)} photons")
        except: continue
        if not gtbin_lc(sel_file, lc_file, tmin, tmax): continue
        if not gtexposure(lc_file, SCFILE, emin_j, emax_j): continue

        c, ex, dPhidE, ddPhidE = read_lc_flux(lc_file, emin_j, emax_j)
        if np.isnan(dPhidE): continue

        energies.append(e_cents[j]); fluxes.append(dPhidE)
        dfluxes.append(ddPhidE); counts_list.append(c); expo_list.append(ex)
        print(f"  counts={c:.0f}, exp={ex:.2e} → dΦ/dE={dPhidE:.4e}")

    if len(energies) > 0:
        out_table = os.path.join(period_dir, f"SED_{period_label}.txt")
        np.savetxt(out_table,
                   np.column_stack([energies, fluxes, dfluxes, counts_list, expo_list]),
                   header='Energy_MeV  dPhi_dE  err_dPhi_dE  counts  exposure',
                   fmt='%.6e')
        print(f"\n  ✅ SED → {out_table}")

    return np.array(energies), np.array(fluxes), np.array(dfluxes)

# ==============================================================
# ==================== LC SUMMARY =============================
# ==============================================================

def plot_lc_summary(period_label, tmin, tmax):
    period_dir = os.path.join(OUTDIR, period_label)
    e_edges = np.logspace(np.log10(EMIN), np.log10(EMAX), N_EBINS + 1)
    e_cents = np.sqrt(e_edges[:-1] * e_edges[1:])
    energies, counts_arr, expo_arr, flux_arr, err_arr = [], [], [], [], []

    for j in range(N_EBINS):
        emin_j, emax_j = e_edges[j], e_edges[j+1]
        lc_file = os.path.join(period_dir, f"gtbin_lc_{period_label}_E{emin_j:.0f}-{emax_j:.0f}MeV.fits")
        if not os.path.exists(lc_file): continue
        try:
            lc = Table.read(lc_file, hdu=1)
            c, ex = float(lc['COUNTS'][0]), float(lc['EXPOSURE'][0])
            if ex > 0:
                energies.append(e_cents[j]); counts_arr.append(c); expo_arr.append(ex)
                flux_arr.append(c/ex); err_arr.append(np.sqrt(max(c,1))/ex)
        except Exception as e:
            print(f"  Erreur: {e}")

    energies = np.array(energies); counts_arr = np.array(counts_arr)
    expo_arr = np.array(expo_arr); flux_arr = np.array(flux_arr); err_arr = np.array(err_arr)

    if len(energies) == 0:
        print(f"  ⚠ Aucun fichier LC pour {period_label}"); return

    dE_matched = np.array([e_edges[j+1]-e_edges[j] for j in range(N_EBINS) if e_cents[j] in energies])
    dPhidE_plot = flux_arr / dE_matched if len(dE_matched)==len(flux_arr) else flux_arr/(energies*0.5)

    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(f'LC Résumé — {period_label.upper()}\n[{tmin:.4e}, {tmax:.4e}] MET', fontsize=13)

    axs[0].step(energies, counts_arr, where='mid', color='steelblue', lw=2)
    axs[0].scatter(energies, counts_arr, color='steelblue', s=40, zorder=5)
    axs[0].set_ylabel('Counts'); axs[0].set_yscale('log')
    axs[0].set_title('Counts par bin énergie'); axs[0].grid(True, which='both', alpha=0.3)

    axs[1].step(energies, expo_arr, where='mid', color='darkorange', lw=2)
    axs[1].scatter(energies, expo_arr, color='darkorange', s=40, zorder=5)
    axs[1].set_ylabel('Exposure (cm² s)'); axs[1].set_yscale('log')
    axs[1].set_title('Exposition gtexposure'); axs[1].grid(True, which='both', alpha=0.3)

    axs[2].errorbar(energies, flux_arr, yerr=err_arr, fmt='o-', color='green', lw=2, ms=6, capsize=4)
    axs[2].set_ylabel('Flux (ph cm⁻² s⁻¹)'); axs[2].set_yscale('log')
    axs[2].set_title('Flux = Counts / Exposure'); axs[2].grid(True, which='both', alpha=0.3)

    axs[3].errorbar(energies, dPhidE_plot, yerr=err_arr/np.maximum(dE_matched,1),
                   fmt='o-', color='red', lw=2, ms=6, capsize=4)
    axs[3].set_ylabel('dΦ/dE (ph cm⁻² s⁻¹ MeV⁻¹)'); axs[3].set_xlabel('Énergie (MeV)')
    axs[3].set_yscale('log'); axs[3].set_title('SED : dΦ/dE'); axs[3].grid(True, which='both', alpha=0.3)

    for ax in axs: ax.set_xscale('log')
    plt.tight_layout()
    outfile = os.path.join(OUTDIR, f'LC_summary_{period_label}.png')
    plt.savefig(outfile, dpi=150); plt.show()
    print(f"  ✅ LC summary → {outfile}")

# ==============================================================
# ==================== SED FIT + PLOT =========================
# ==============================================================

def power_law(E, A, alpha):
    return A * E**(-alpha)

def broken_power_law(E, A, alpha1, alpha2, E_break):
    return np.where(E < E_break,
                    A * E**(-alpha1),
                    A * E_break**(alpha2-alpha1) * E**(-alpha2))

def fit_and_plot_sed(E, F, dF, ax, label, color):
    if len(E) < 3:
        ax.text(0.5, 0.5, 'Pas assez de points', transform=ax.transAxes, ha='center')
        return
    ax.errorbar(E, F, yerr=dF, fmt='o', color=color, ms=7, label='Data', zorder=5)
    E_fit = np.logspace(np.log10(E.min()), np.log10(E.max()), 300)

    try:
        popt, pcov = curve_fit(power_law, E, F, sigma=dF, p0=[1e-10, 2.0], maxfev=10000)
        err = np.sqrt(np.diag(pcov))
        ax.plot(E_fit, power_law(E_fit, *popt), '--', color='orange', lw=2,
                label=f'PL: α={popt[1]:.3f}±{err[1]:.3f}')
        print(f"  PL: A={popt[0]:.4e}, α={popt[1]:.4f}±{err[1]:.4f}")
    except Exception as e:
        print(f"  Fit PL échoué: {e}")

    try:
        popt_b, pcov_b = curve_fit(broken_power_law, E, F, sigma=dF,
                                    p0=[1e-10, 1.5, 2.8, np.median(E)], maxfev=20000)
        err_b = np.sqrt(np.diag(pcov_b))
        ax.plot(E_fit, broken_power_law(E_fit, *popt_b), '-', color='green', lw=2,
                label=f'BPL: α₁={popt_b[1]:.3f} α₂={popt_b[2]:.3f} Eb={popt_b[3]:.0f}MeV')
        print(f"  BPL: α₁={popt_b[1]:.4f}±{err_b[1]:.4f}, α₂={popt_b[2]:.4f}±{err_b[2]:.4f}")
    except Exception as e:
        print(f"  Fit BPL échoué: {e}")

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Énergie (MeV)', fontsize=12)
    ax.set_ylabel('dΦ/dE (ph cm⁻² s⁻¹ MeV⁻¹)', fontsize=11)
    ax.set_title(f'SED — {label}', fontsize=13)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, which='both', alpha=0.3)

# ==============================================================
# ==================== MAIN ===================================
# ==============================================================

os.makedirs(OUTDIR, exist_ok=True)

# Lance pipelines SED pour toutes les périodes
results = {}
for period in ALL_PERIODS:
    E, F, dF = run_sed_pipeline(period["label"], period["tmin"], period["tmax"])
    results[period["label"]] = {"E": E, "F": F, "dF": dF,
                                 "color": period["color"],
                                 "tmin": period["tmin"], "tmax": period["tmax"]}

# LC summary
print("\n" + "="*60 + "\nRÉSUMÉ COURBES DE LUMIÈRE\n" + "="*60)
for period in ALL_PERIODS:
    plot_lc_summary(period["label"], period["tmin"], period["tmax"])

# SED comparaison
n = len(ALL_PERIODS)
fig, axs = plt.subplots(1, n, figsize=(7*n, 6))
if n == 1: axs = [axs]

for ax, period in zip(axs, ALL_PERIODS):
    lbl = period["label"]
    r = results[lbl]
    print(f"\nFit SED {lbl.upper()}:")
    fit_and_plot_sed(r["E"], r["F"], r["dF"], ax, lbl.upper(), r["color"])

plt.suptitle('SED Fermi-LAT — PL (orange) vs BPL (vert)', fontsize=13)
plt.tight_layout()
outplot = os.path.join(OUTDIR, 'SED_comparison.png')
plt.savefig(outplot, dpi=150)
plt.show()
print(f"\n✅ {outplot}")
print(f"✅ Fichiers dans {OUTDIR}/")
