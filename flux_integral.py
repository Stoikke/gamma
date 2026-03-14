import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import os

# ==============================================================
# ==================== CONFIGURATION ==========================
# ==============================================================

OUTDIR    = "pchip_output"
TIME_UNIT = "s"
FLUX_UNIT = "ph/cm²/s"

# ── Fichiers SOURCE (1 source × 2 périodes) ───────────────────
SOURCE_FILES = {
    "period1": "results_simple/lat_photon_flare1article_selected_radius4_LC_p6h.fits",
    "period2": "results_simple/lat_photon_flare2article_selected_radius4_LC_p6h.fits",
}

# ── Fichiers FOND (3 régions × 2 périodes) ────────────────────
FOND_FILES = {
    "fond1": {
        "period1": "SED_output/FOND_peak/gt_bin_activite_1_fond_1_100_20000.fits",
        "period2": "Flux_integral/gt_bin_activite_2_fond_1_100_300000.fits",
    },
    "fond2": {
        "period1": "SED_output/FOND_peak/gt_bin_activite_1_fond_1_100_20000.fits",
        "period2": "Flux_integral/gt_bin_activite_2_fond_2_100_300000.fits",
    },
    "fond3": {
        "period1": "SED_output/FOND_peak/gt_bin_activite_1_fond_1_100_20000.fits",
        "period2": "Flux_integral/gt_bin_activite_2_fond_3_100_300000.fits",
    },
}

# ── Zones actives (intégration PCHIP) ─────────────────────────
ACTIVE_ZONES = {
    "period1": {"t_start": 503280004+13*3600, "t_stop": 503280004+4*86400},
    "period2": {"t_start": 503712004, "t_stop": 504403204},
}

# ── Cosmologie & paramètres source ────────────────────────────
COSMO      = FlatLambdaCDM(H0=69.6, Om0=0.27)
REDSHIFT   = 1.037       # ← redshift de la source
GAMMA_SPEC = -1.92          # ← indice spectral Γ
E_MIN_MEV  = 100.0        # ← même bande que gtbin
E_MAX_MEV  = 300000.0

# Couleurs par période
PERIOD_COLORS = {"period1": "steelblue", "period2": "darkorange"}

# ==============================================================
# ==================== LECTURE DONNÉES ========================
# ==============================================================

def load_lightcurve(filepath):
    """Charge FITS gtbin : flux = COUNTS/EXPOSURE, err = ERROR/EXPOSURE."""
    print(f"  📂 {os.path.basename(filepath)}")
    with fits.open(filepath) as hdul:
        data     = hdul[1].data
        time     = data['TIME'].astype(float)
        counts   = data['COUNTS'].astype(float)
        exposure = data['EXPOSURE'].astype(float)
        try:
            err_raw = data['ERROR'].astype(float)
        except KeyError:
            err_raw = np.sqrt(np.maximum(counts, 1.0))

    with np.errstate(invalid='ignore', divide='ignore'):
        flux = np.where(exposure > 0, counts  / exposure, np.nan)
        ferr = np.where(exposure > 0, err_raw / exposure, np.nan)

    mask = np.isfinite(flux) & np.isfinite(ferr) & np.isfinite(time) & (flux >= 0)
    time, flux, ferr = time[mask], flux[mask], ferr[mask]
    print(f"     → {len(time)} points valides")
    return time, flux, ferr

# ==============================================================
# ==================== SOUSTRACTION DE FOND ===================
# ==============================================================

def compute_background(period):
    """
    Charge les 3 fichiers fond pour une période.
    Retourne la moyenne des 3 fonds et l'erreur combinée.
    """
    bkg_means, bkg_stds = [], []
    for fname, ffiles in FOND_FILES.items():
        _, flux_f, ferr_f = load_lightcurve(ffiles[period])
        m = np.mean(flux_f)
        s = np.sqrt(np.sum(ferr_f**2)) / len(ferr_f)
        bkg_means.append(m)
        bkg_stds.append(s)
        print(f"     {fname:6s} : {m:.4e} ± {s:.2e} {FLUX_UNIT}")

    bkg_total     = np.mean(bkg_means)
    bkg_total_err = np.sqrt(np.sum(np.array(bkg_stds)**2)) / len(bkg_stds)
    print(f"  🔹 Fond moyen : {bkg_total:.4e} ± {bkg_total_err:.2e} {FLUX_UNIT}")
    return bkg_total, bkg_total_err

def subtract_background(flux, ferr, bkg_mean, bkg_err):
    """Soustrait le fond, propage l'erreur quadratiquement."""
    return flux- bkg_mean , np.sqrt(ferr**2 + bkg_err**2)#- bkg_mean

# ==============================================================
# ==================== PCHIP + INTÉGRATION ====================
# ==============================================================

def compute_pchip_fluence(time, flux, ferr, t_start, t_stop, label=""):
    """
    PCHIP + intégration avec normalisation temporelle pour éviter NaN.
    """
    mask = (time >= t_start) & (time <= t_stop)
    idx  = np.where(mask)[0]

    if len(idx) < 2:
        print(f"  ❌ Pas assez de points dans la zone active pour {label}")
        return None, 0.0, 0.0, 0.0

    i0 = max(idx[0] - 1, 0)
    i1 = min(idx[-1] + 1, len(time) - 1)

    t_sel = time[i0:i1+1]
    f_sel = flux[i0:i1+1]
    e_sel = ferr[i0:i1+1]

    print(f"  🔍 t_sel   : [{t_sel.min():.6e}, {t_sel.max():.6e}]")
    print(f"  🔍 active  : [{t_start:.6e}, {t_stop:.6e}]")
    print(f"  🔍 points  : {len(t_sel)}")

    # Clip des bornes à la plage réelle
    t0_int = max(t_start, t_sel.min())
    t1_int = min(t_stop,  t_sel.max())

    if t0_int >= t1_int:
        print(f"  ❌ Zone active hors plage pour {label}")
        return None, 0.0, 0.0, 0.0

    # Normalisation temporelle (évite roundoff sur ~5×10⁸)
    t_ref  = t_sel[0]
    t_norm = t_sel  - t_ref
    t0_n   = t0_int - t_ref
    t1_n   = t1_int - t_ref

    pchip_norm = PchipInterpolator(t_norm, f_sel, extrapolate=False)

    fluence, _ = quad(pchip_norm, t0_n, t1_n,
                      limit=500, epsabs=0.0, epsrel=1e-6)

    # Erreur propagée (σ_i × Δt_i en quadrature)
    dt_bins     = np.diff(t_norm)
    dt_mid      = np.concatenate([[dt_bins[0]/2],
                                   (dt_bins[:-1]+dt_bins[1:])/2,
                                   [dt_bins[-1]/2]])
    fluence_err = np.sqrt(np.sum((e_sel * dt_mid)**2))

    dt        = t1_int - t0_int
    mean_flux = fluence / dt if dt > 0 else np.nan

    print(f"  ✅ [{label}] Fluence   = {fluence:.4e} ± {fluence_err:.4e} "
          f"{FLUX_UNIT}·{TIME_UNIT}")
    print(f"  ✅ [{label}] Flux moy. = {mean_flux:.4e} {FLUX_UNIT}")

    # PCHIP en temps absolu pour le plot
    pchip_abs = PchipInterpolator(t_sel, f_sel, extrapolate=False)
    return pchip_abs, fluence, fluence_err, mean_flux

# ==============================================================
# ==================== CONVERSION SPECTRALE ====================
# ==============================================================

def photon_to_energy_flux(photon_flux, gamma=GAMMA_SPEC,
                           e_min=E_MIN_MEV, e_max=E_MAX_MEV):
    """
    Convertit un flux photon [ph/cm²/s] en flux énergie [erg/cm²/s].
    Énergie moyenne pour une loi de puissance dN/dE ∝ E^(-Γ).
    """
    e1, e2 = e_min, e_max
    if abs(gamma - 2.0) < 1e-6:
        mean_E_MeV = np.log(e2/e1) / (1.0/e1 - 1.0/e2)
    elif abs(gamma - 1.0) < 1e-6:
        mean_E_MeV = (e2 - e1) / np.log(e2/e1)
    else:
        num        = (e2**(2.0-gamma) - e1**(2.0-gamma)) / (2.0-gamma)
        denom      = (e2**(1.0-gamma) - e1**(1.0-gamma)) / (1.0-gamma)
        mean_E_MeV = num / denom

    MEV_TO_ERG  = 1.60218e-6
    energy_flux = photon_flux * mean_E_MeV * MEV_TO_ERG
    return energy_flux, mean_E_MeV

# ==============================================================
# ==================== LUMINOSITÉ =============================
# ==============================================================

def compute_luminosity(mean_flux, mean_flux_err,
                       fluence, fluence_err,
                       redshift=REDSHIFT, gamma=GAMMA_SPEC,
                       e_min=E_MIN_MEV, e_max=E_MAX_MEV,
                       label=""):
    """
    Calcule :
      - Luminosité isotrope  L = 4π d_L² × F_erg × K(z)      [erg/s]
      - Énergie isotrope     E = 4π d_L² × Flu_erg × K(z)/(1+z) [erg]
    """
    print(f"\n  ── Luminosité [{label}] ──────────────────────────────")

    # Distance de luminosité
    d_L_cm = COSMO.luminosity_distance(redshift).to(u.cm).value
    print(f"  🔭 d_L = {d_L_cm/3.085677581e24:.4f} Mpc = {d_L_cm:.4e} cm")

    # K-correction loi de puissance
    K = (1.0 + redshift)**(gamma - 2.0)
    print(f"  ⚡ K-correction = {K:.6f}  (z={redshift}, Γ={gamma})")

    # ── Flux énergie et luminosité ──
    F_erg, mean_E = photon_to_energy_flux(mean_flux, gamma, e_min, e_max)
    F_erg_err     = photon_to_energy_flux(mean_flux_err, gamma, e_min, e_max)[0]

    L_iso     = 4.0 * np.pi * d_L_cm**2 * F_erg     * K
    L_iso_err = 4.0 * np.pi * d_L_cm**2 * F_erg_err * K

    print(f"  ⚡ <E_photon>    = {mean_E:.2f} MeV")
    print(f"  ⚡ Flux énergie  = {F_erg:.4e} ± {F_erg_err:.2e} erg/cm²/s")
    print(f"  ✅ Luminosité    = {L_iso:.4e} ± {L_iso_err:.2e} erg/s")
    print(f"  ✅ (M=10⁸M☉)    = {L_iso/1.26e46:.4f} L_Edd")

    # ── Fluence énergie et énergie totale ──
    MEV_TO_ERG      = 1.60218e-6
    fluence_erg     = fluence     * mean_E * MEV_TO_ERG
    fluence_erg_err = fluence_err * mean_E * MEV_TO_ERG

    E_iso     = 4.0 * np.pi * d_L_cm**2 * fluence_erg     * K / (1.0 + redshift)
    E_iso_err = 4.0 * np.pi * d_L_cm**2 * fluence_erg_err * K / (1.0 + redshift)

    print(f"  ✅ Énergie iso.  = {E_iso:.4e} ± {E_iso_err:.2e} erg")
    print(f"  ✅               = {E_iso/1e51:.4e} × 10⁵¹ erg")

    return {
        "L_iso":         L_iso,
        "L_iso_err":     L_iso_err,
        "F_erg":         F_erg,
        "F_erg_err":     F_erg_err,
        "E_iso":         E_iso,
        "E_iso_err":     E_iso_err,
        "fluence_erg":   fluence_erg,
        "fluence_erg_e": fluence_erg_err,
        "mean_E_MeV":    mean_E,
    }

# ==============================================================
# ==================== VISUALISATION ==========================
# ==============================================================

def plot_all(results, outdir):
    """
    2 panneaux côte à côte + panneau résumé luminosité en bas.
    """
    os.makedirs(outdir, exist_ok=True)
    periods = list(ACTIVE_ZONES.keys())

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Spline — Source (fond soustrait) + Luminosité",
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1],
                           hspace=0.45, wspace=0.30)

    for pi, per in enumerate(periods):
        ax = fig.add_subplot(gs[0, pi])
        r  = results[per]
        if r is None:
            ax.set_title(f"{per} — ❌")
            continue

        time    = r['time']
        flux    = r['flux']
        ferr    = r['ferr']
        pchip   = r['pchip']
        t_start = r['t_start']
        t_stop  = r['t_stop']
        fluence = r['fluence']
        flu_err = r['fluence_err']
        color   = PERIOD_COLORS[per]

        ax.errorbar(time, flux, yerr=ferr,
                    fmt='o', ms=3.5, color=color, alpha=0.6,
                    elinewidth=0.9, capsize=2.5,
                    label='Source (fond soustrait)', zorder=2)
        ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axvspan(t_start, t_stop, color='gold', alpha=0.15, label='Zone active')

        t_fine = np.linspace(t_start, t_stop, 3000)
        f_fine = pchip(t_fine)
        ax.plot(t_fine, f_fine, '-', color='crimson', lw=2.0,
                label='Spline', zorder=3)
        ax.fill_between(t_fine, 0, f_fine, alpha=0.20, color='crimson',
                        label=f'Fluence = {fluence:.3e} ± {flu_err:.1e}')

        lum = r.get('lum', {})
        if lum:
            ax.annotate(
                f"L = {lum['L_iso']:.2e} erg/s\n"
                f"E = {lum['E_iso']:.2e} erg\n"
                f"Fond : {r['bkg_mean']:.2e} {FLUX_UNIT}",
                xy=(0.02, 0.97), xycoords='axes fraction',
                fontsize=7.5, va='top', color='dimgray',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
            )

        pnum = per.replace('period', 'Période ')
        ax.set_title(f"{pnum}\nFluence = {fluence:.3e} ± {flu_err:.1e} "
                     f"{FLUX_UNIT}·{TIME_UNIT}", fontsize=9)
        ax.set_xlabel(f"Temps [{TIME_UNIT}]", fontsize=9)
        ax.set_ylabel(f"Flux [{FLUX_UNIT}]", fontsize=9)
        ax.legend(loc='upper right', fontsize=7.5)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(time.min(), time.max())
        ax.tick_params(labelsize=7)

    # ── Panneau résumé luminosité ──
    ax_sum = fig.add_subplot(gs[1, :])
    ax_sum.axis('off')

    rows = [["Période", "Fluence [ph/cm²]", "Flux moy [ph/cm²/s]",
             "F_erg [erg/cm²/s]", "L_iso [erg/s]", "E_iso [erg]"]]

    total_flu = total_E = 0.0
    for per in periods:
        r = results[per]
        if r is None:
            continue
        lum = r.get('lum', {})
        rows.append([
            per.replace('period', 'P'),
            f"{r['fluence']:.3e} ± {r['fluence_err']:.1e}",
            f"{r['mean_flux']:.3e}",
            f"{lum.get('F_erg', 0):.3e}" if lum else "—",
            f"{lum.get('L_iso', 0):.3e} ± {lum.get('L_iso_err', 0):.1e}" if lum else "—",
            f"{lum.get('E_iso', 0):.3e} ± {lum.get('E_iso_err', 0):.1e}" if lum else "—",
        ])
        if lum:
            total_flu += r['fluence']
            total_E   += lum.get('E_iso', 0)

    rows.append(["TOTAL", f"{total_flu:.3e}", "—", "—", "—",
                 f"{total_E:.3e} erg"])

    table = ax_sum.table(cellText=rows[1:], colLabels=rows[0],
                         loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    # En-tête coloré
    for j in range(len(rows[0])):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Ligne TOTAL
    for j in range(len(rows[0])):
        table[len(rows)-1, j].set_facecolor('#f0f0f0')
        table[len(rows)-1, j].set_text_props(fontweight='bold')

    ax_sum.set_title(
        f"Résumé  |  z = {REDSHIFT}  |  Γ = {GAMMA_SPEC}  |  "
        f"{E_MIN_MEV:.0f} MeV – {E_MAX_MEV/1000:.0f} GeV",
        fontsize=9, pad=8
    )

    plt.savefig(os.path.join(outdir, "pchip_source_luminosity.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Figure → {outdir}/pchip_source_luminosity.png")

# ==============================================================
# ==================== SAUVEGARDE TEXTE =======================
# ==============================================================

def save_results(results, total_fluence, total_flu_err,
                 total_E, total_E_err, outdir):
    os.makedirs(outdir, exist_ok=True)
    result_file = os.path.join(outdir, "fluence_luminosity.txt")
    with open(result_file, 'w') as f:
        f.write("# FLUENCE + LUMINOSITÉ SOURCE\n")
        f.write(f"# z = {REDSHIFT}  |  Gamma = {GAMMA_SPEC}\n")
        f.write(f"# Bande : {E_MIN_MEV:.0f} MeV – {E_MAX_MEV/1000:.0f} GeV\n\n")

        for per, r in results.items():
            if r is None:
                continue
            az  = ACTIVE_ZONES[per]
            lum = r.get('lum', {})
            f.write(f"[{per}]\n")
            f.write(f"  t_start        = {az['t_start']:.6e} {TIME_UNIT}\n")
            f.write(f"  t_stop         = {az['t_stop']:.6e} {TIME_UNIT}\n")
            f.write(f"  delta_t        = {az['t_stop']-az['t_start']:.6e} {TIME_UNIT}\n")
            f.write(f"  bkg_mean       = {r['bkg_mean']:.6e} {FLUX_UNIT}\n")
            f.write(f"  fluence        = {r['fluence']:.6e} ± {r['fluence_err']:.2e} "
                    f"{FLUX_UNIT}·{TIME_UNIT}\n")
            f.write(f"  mean_flux      = {r['mean_flux']:.6e} {FLUX_UNIT}\n")
            if lum:
                f.write(f"  mean_E_photon  = {lum['mean_E_MeV']:.2f} MeV\n")
                f.write(f"  F_energy       = {lum['F_erg']:.6e} ± "
                        f"{lum['F_erg_err']:.2e} erg/cm²/s\n")
                f.write(f"  fluence_energy = {lum['fluence_erg']:.6e} ± "
                        f"{lum['fluence_erg_e']:.2e} erg/cm²\n")
                f.write(f"  L_iso          = {lum['L_iso']:.6e} ± "
                        f"{lum['L_iso_err']:.2e} erg/s\n")
                f.write(f"  E_iso          = {lum['E_iso']:.6e} ± "
                        f"{lum['E_iso_err']:.2e} erg\n")
            f.write("\n")

        f.write("[TOTAL]\n")
        f.write(f"  fluence_total  = {total_fluence:.6e} ± "
                f"{total_flu_err:.2e} {FLUX_UNIT}·{TIME_UNIT}\n")
        f.write(f"  E_iso_total    = {total_E:.6e} ± {total_E_err:.2e} erg\n")
        f.write(f"  E_iso_total    = {total_E/1e51:.4e} × 10⁵¹ erg\n")

    print(f"  💾 Résultats → {result_file}")

# ==============================================================
# ==================== MAIN ===================================
# ==============================================================

if __name__ == "__main__":

    os.makedirs(OUTDIR, exist_ok=True)
    periods = list(ACTIVE_ZONES.keys())

    results       = {}
    total_fluence = 0.0
    total_flu_err = 0.0
    total_E       = 0.0
    total_E_err   = 0.0

    print("=" * 65)
    print("  PCHIP SOURCE — fond soustrait + luminosité")
    print(f"  z = {REDSHIFT}  |  Γ = {GAMMA_SPEC}  |  "
          f"{E_MIN_MEV:.0f} MeV – {E_MAX_MEV/1000:.0f} GeV")
    print("=" * 65)

    for per in periods:
        az  = ACTIVE_ZONES[per]
        lbl = per.upper()

        print(f"\n{'─'*55}")
        print(f"  ▶ {lbl}")

        # ── 1. Source ──
        print("  SOURCE :")
        time, flux, ferr = load_lightcurve(SOURCE_FILES[per])

        # ── 2. Fond ──
        print("  FOND (3 régions) :")
        bkg_mean, bkg_err = compute_background(per)

        # ── 3. Soustraction ──
        flux_sub, ferr_sub = subtract_background(flux, ferr, bkg_mean, bkg_err)

        # ── 4. PCHIP + fluence ──
        pchip, fluence, flu_err, mean_flux = compute_pchip_fluence(
            time, flux_sub, ferr_sub,
            az['t_start'], az['t_stop'],
            label=lbl
        )

        if pchip is None:
            results[per] = None
            continue

        # ── 5. Luminosité ──
        mean_flux_err = flu_err / (az['t_stop'] - az['t_start'])
        lum = compute_luminosity(
            mean_flux     = mean_flux,
            mean_flux_err = mean_flux_err,
            fluence       = fluence,
            fluence_err   = flu_err,
            label         = lbl
        )

        results[per] = {
            "time":        time,
            "flux":        flux_sub,
            "ferr":        ferr_sub,
            "pchip":       pchip,
            "fluence":     fluence,
            "fluence_err": flu_err,
            "mean_flux":   mean_flux,
            "t_start":     az['t_start'],
            "t_stop":      az['t_stop'],
            "bkg_mean":    bkg_mean,
            "bkg_err":     bkg_err,
            "lum":         lum,
        }

        # ── 6. Accumulation ──
        total_fluence += fluence
        total_flu_err  = np.sqrt(total_flu_err**2 + flu_err**2)
        total_E       += lum['E_iso']
        total_E_err    = np.sqrt(total_E_err**2 + lum['E_iso_err']**2)

    # ── 7. Résumé terminal ────────────────────────────────────
    print(f"\n{'='*65}")
    print("  RÉSUMÉ FINAL")
    print(f"{'='*65}")
    for per in periods:
        r = results.get(per)
        if r:
            lum = r['lum']
            print(f"\n  {per.upper()}")
            print(f"    Fluence  = {r['fluence']:.4e} ± {r['fluence_err']:.2e} "
                  f"{FLUX_UNIT}·{TIME_UNIT}")
            print(f"    L_iso    = {lum['L_iso']:.4e} ± {lum['L_iso_err']:.2e} erg/s")
            print(f"    E_iso    = {lum['E_iso']:.4e} ± {lum['E_iso_err']:.2e} erg")

    print(f"\n  ── TOTAL PÉRIODE 1 + 2 ──")
    print(f"  Fluence totale = {total_fluence:.4e} ± {total_flu_err:.2e} "
          f"{FLUX_UNIT}·{TIME_UNIT}")
    print(f"  E_iso totale   = {total_E:.4e} ± {total_E_err:.2e} erg")
    print(f"                 = {total_E/1e51:.4e} × 10⁵¹ erg")

    # ── 8. Sauvegarde + figure ────────────────────────────────
    save_results(results, total_fluence, total_flu_err,
                 total_E, total_E_err, OUTDIR)
    plot_all(results, OUTDIR)
