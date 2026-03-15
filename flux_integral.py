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

# ── Fichiers SOURCE ───────────────────────────────────────────
SOURCE_FILES = {
    "period1": "results_simple/lat_photon_flare1article_selected_radius4_LC_p6h.fits",
    "period2": "results_simple/lat_photon_flare2article_selected_radius4_LC_p6h.fits",
}

# ── Fichiers FOND (3 régions × 2 périodes) ────────────────────
FOND_FILES = {
    "fond1": {
        "period1": "SED_output/FOND_peak/gt_bin_activite_1_fond_1_100_20000.fits",
        "period2": "Flux_integral/gt_bin_activite_1_fond_1_100_300000.fits",
    },
    "fond2": {
        "period1": "SED_output/FOND_peak/gt_bin_activite_1_fond_2_100_20000.fits",
        "period2": "Flux_integral/gt_bin_activite_1_fond_2_100_300000.fits",
    },
    "fond3": {
        "period1": "SED_output/FOND_peak/gt_bin_activite_1_fond_3_100_20000.fits",
        "period2": "Flux_integral/gt_bin_activite_1_fond_3_100_300000.fits",
    },
}

# ── Zones actives ─────────────────────────────────────────────
ACTIVE_ZONES = {
    "period1": {"t_start": 503280004+13*3600, "t_stop": 503280004+4*86400},
    "period2": {"t_start": 503712004, "t_stop": 504403204},
}

# ── Cosmologie ────────────────────────────────────────────────
COSMO         = FlatLambdaCDM(H0=67.4, Om0=0.315)
REDSHIFT      = 1.037
DELTA_DOPPLER = 22.3          # Jorstad et al. 2005
E_MIN_MEV     = 100.0
E_MAX_MEV     = 300000.0

# ── Indice spectral Γ ± δΓ par période (depuis ton analyse SED)
GAMMA_PER_PERIOD = {
    "period1": {"gamma": 1.85, "gamma_err": 0.10},   # ← à adapter
    "period2": {"gamma": 1.92, "gamma_err": 0.14},   # ← à adapter
}

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
    """Moyenne des 3 fonds pour une période. Retourne (mean, err)."""
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
    return flux - bkg_mean, np.sqrt(ferr**2 + bkg_err**2)

# ==============================================================
# ==================== PCHIP + INTÉGRATION ====================
# ==============================================================

def compute_pchip_fluence(time, flux, ferr, t_start, t_stop, label=""):
    """PCHIP + intégration avec normalisation temporelle."""
    mask = (time >= t_start) & (time <= t_stop)
    idx  = np.where(mask)[0]

    if len(idx) < 2:
        print(f"  ❌ Pas assez de points dans la zone active pour {label}")
        return None, 0.0, 0.0, 0.0

    i0    = max(idx[0] - 1, 0)
    i1    = min(idx[-1] + 1, len(time) - 1)
    t_sel = time[i0:i1+1]
    f_sel = flux[i0:i1+1]
    e_sel = ferr[i0:i1+1]

    print(f"  🔍 t_sel   : [{t_sel.min():.6e}, {t_sel.max():.6e}]")
    print(f"  🔍 active  : [{t_start:.6e}, {t_stop:.6e}]")
    print(f"  🔍 points  : {len(t_sel)}")

    t0_int = max(t_start, t_sel.min())
    t1_int = min(t_stop,  t_sel.max())
    if t0_int >= t1_int:
        print(f"  ❌ Zone active hors plage pour {label}")
        return None, 0.0, 0.0, 0.0

    # Normalisation (évite roundoff sur ~5×10⁸)
    t_ref      = t_sel[0]
    t_norm     = t_sel  - t_ref
    pchip_norm = PchipInterpolator(t_norm, f_sel, extrapolate=False)

    fluence, _ = quad(pchip_norm, t0_int - t_ref, t1_int - t_ref,
                      limit=500, epsabs=0.0, epsrel=1e-6)

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

    pchip_abs = PchipInterpolator(t_sel, f_sel, extrapolate=False)
    return pchip_abs, fluence, fluence_err, mean_flux

# ==============================================================
# ========== CONVERSION SPECTRALE + PROPAGATION δΓ ============
# ==============================================================

def mean_energy_and_deriv(gamma, e_min, e_max, dgamma=1e-5):
    """
    Retourne <E>(Γ) en MeV et d<E>/dΓ par différences finies.
    <E> = ∫ E^(1-Γ) dE / ∫ E^(-Γ) dE
    """
    def _mean_E(g):
        e1, e2 = e_min, e_max
        if abs(g - 2.0) < 1e-6:
            return np.log(e2/e1) / (1.0/e1 - 1.0/e2)
        elif abs(g - 1.0) < 1e-6:
            return (e2 - e1) / np.log(e2/e1)
        else:
            num   = (e2**(2.0-g) - e1**(2.0-g)) / (2.0-g)
            denom = (e2**(1.0-g) - e1**(1.0-g)) / (1.0-g)
            return num / denom

    mE       = _mean_E(gamma)
    dmE_dG   = (_mean_E(gamma + dgamma) - _mean_E(gamma - dgamma)) / (2 * dgamma)
    return mE, dmE_dG


def photon_to_energy_flux_with_err(photon_flux, photon_flux_err,
                                   gamma, gamma_err,
                                   e_min=E_MIN_MEV, e_max=E_MAX_MEV):
    """
    F_erg = F_ph × <E>(Γ) × MEV_TO_ERG
    Propagation :
      σ²(F_erg) = (<E>·MEV_TO_ERG)² σ²(F_ph)
                + (F_ph·MEV_TO_ERG·d<E>/dΓ)² σ²(Γ)
    """
    MEV_TO_ERG    = 1.60218e-6
    mE, dmE_dG    = mean_energy_and_deriv(gamma, e_min, e_max)

    F_erg         = photon_flux * mE * MEV_TO_ERG
    sigma_from_F  = (mE * MEV_TO_ERG * photon_flux_err)**2
    sigma_from_G  = (photon_flux * MEV_TO_ERG * dmE_dG * gamma_err)**2
    F_erg_err     = np.sqrt(sigma_from_F + sigma_from_G)

    return F_erg, F_erg_err, mE, dmE_dG


def k_correction_with_err(gamma, gamma_err, redshift):
    """
    K(z,Γ) = (1+z)^(Γ-2)
    dK/dΓ  = ln(1+z) × K
    σ(K)   = |ln(1+z)| × K × σ(Γ)
    """
    K      = (1.0 + redshift)**(gamma - 2.0)
    dK_dG  = np.log(1.0 + redshift) * K
    K_err  = abs(dK_dG) * gamma_err
    return K, K_err

# ==============================================================
# ==================== LUMINOSITÉ =============================
# ==============================================================

def compute_luminosity(mean_flux, mean_flux_err,
                       fluence, fluence_err,
                       gamma, gamma_err,
                       redshift=REDSHIFT,
                       e_min=E_MIN_MEV, e_max=E_MAX_MEV,
                       delta=DELTA_DOPPLER,
                       label=""):
    """
    Calcule L_iso, L_int, E_iso, E_int avec propagation complète
    des incertitudes sur F_ph, Γ.

    Formules :
      L_iso = 4π d_L² F_erg K(z)
      L_int = L_iso / (δ⁴ (1+z))          [Urry & Padovani 1995]
      E_iso = 4π d_L² Flu_erg K(z)/(1+z)
      E_int = E_iso / (δ⁴ (1+z))

    Propagation Γ :
      σ²(L_iso) = (4π d_L²)² [ K² σ²(F_erg) + F_erg² σ²(K) ]
    """
    print(f"\n  ── Luminosité [{label}]  Γ = {gamma} ± {gamma_err} ──")

    d_L_cm = COSMO.luminosity_distance(redshift).to(u.cm).value
    print(f"  🔭 d_L = {d_L_cm/3.085677581e24:.4f} Mpc")

    # K-correction + incertitude
    K, K_err = k_correction_with_err(gamma, gamma_err, redshift)
    print(f"  ⚡ K(z,Γ) = {K:.6f} ± {K_err:.2e}  (z={redshift})")

    # Flux énergie + incertitude
    F_erg, F_erg_err, mE, dmE_dG = photon_to_energy_flux_with_err(
        mean_flux, mean_flux_err, gamma, gamma_err, e_min, e_max
    )
    print(f"  ⚡ <E>     = {mE:.2f} MeV  |  d<E>/dΓ = {dmE_dG:.2f} MeV")
    print(f"  ⚡ F_erg   = {F_erg:.4e} ± {F_erg_err:.2e} erg/cm²/s")
    print(f"     └ σ(F_ph) → {(mE*1.60218e-6*mean_flux_err):.2e}  "
          f"σ(Γ)  → {(mean_flux*1.60218e-6*dmE_dG*gamma_err):.2e} erg/cm²/s")

    # ── Luminosité isotrope ────────────────────────────────────
    # σ²(L_iso) = (4π dL²)² [K² σ²(F_erg) + F_erg² σ²(K)]
    coeff     = 4.0 * np.pi * d_L_cm**2
    L_iso     = coeff * F_erg * K
    L_iso_err = coeff * np.sqrt((K * F_erg_err)**2 + (F_erg * K_err)**2)

    # ── Luminosité intrinsèque ────────────────────────────────
    doppler_factor = delta**4 * (1.0 + redshift)
    L_int          = L_iso     / doppler_factor
    L_int_err      = L_iso_err / doppler_factor

    print(f"\n  ✅ L_iso          = {L_iso:.4e} ± {L_iso_err:.2e} erg/s")
    print(f"  ✅ L_int (δ={delta}) = {L_int:.4e} ± {L_int_err:.2e} erg/s")

    # ── Énergie isotrope et intrinsèque ──────────────────────
    MEV_TO_ERG    = 1.60218e-6
    # Propagation Γ sur la fluence énergie
    flu_erg       = fluence     * mE * MEV_TO_ERG
    flu_erg_err   = np.sqrt(
        (mE * MEV_TO_ERG * fluence_err)**2 +          # σ(fluence)
        (fluence * MEV_TO_ERG * dmE_dG * gamma_err)**2  # σ(Γ)
    )

    E_iso     = coeff * flu_erg     * K / (1.0 + redshift)
    E_iso_err = coeff / (1.0 + redshift) * np.sqrt(
        (K * flu_erg_err)**2 + (flu_erg * K_err)**2
    )
    E_int     = E_iso     / doppler_factor
    E_int_err = E_iso_err / doppler_factor

    print(f"  ✅ E_iso          = {E_iso:.4e} ± {E_iso_err:.2e} erg")
    print(f"  ✅ E_int (δ={delta}) = {E_int:.4e} ± {E_int_err:.2e} erg")

    return {
        "gamma":          gamma,
        "gamma_err":      gamma_err,
        "K":              K,           "K_err":          K_err,
        "F_erg":          F_erg,       "F_erg_err":      F_erg_err,
        "mean_E_MeV":     mE,          "dmE_dG":         dmE_dG,
        "L_iso":          L_iso,       "L_iso_err":      L_iso_err,
        "L_int":          L_int,       "L_int_err":      L_int_err,
        "fluence_erg":    flu_erg,     "fluence_erg_err": flu_erg_err,
        "E_iso":          E_iso,       "E_iso_err":      E_iso_err,
        "E_int":          E_int,       "E_int_err":      E_int_err,
        "doppler_factor": doppler_factor,
    }

# ==============================================================
# ==================== VISUALISATION ==========================
# ==============================================================

def plot_all(results, outdir):
    os.makedirs(outdir, exist_ok=True)
    periods = list(ACTIVE_ZONES.keys())

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("PCHIP — Source (fond soustrait) | Luminosité isotrope & intrinsèque",
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(3, 2,
                           height_ratios=[2.5, 0.85, 0.85],
                           hspace=0.55, wspace=0.30)

    # ── Courbes de lumière ────────────────────────────────────
    for pi, per in enumerate(periods):
        ax = fig.add_subplot(gs[0, pi])
        r  = results[per]
        if r is None:
            ax.set_title(f"{per} — ❌"); continue

        time    = r['time'];    flux = r['flux'];  ferr = r['ferr']
        pchip   = r['pchip'];  t_start = r['t_start']; t_stop = r['t_stop']
        fluence = r['fluence']; flu_err = r['fluence_err']
        lum     = r.get('lum', {})
        color   = PERIOD_COLORS[per]

        ax.errorbar(time, flux, yerr=ferr, fmt='o', ms=3.5, color=color,
                    alpha=0.6, elinewidth=0.9, capsize=2.5,
                    label='Source (fond soustrait)', zorder=2)
        ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axvspan(t_start, t_stop, color='gold', alpha=0.15, label='Zone active')

        t_fine = np.linspace(t_start, t_stop, 3000)
        f_fine = pchip(t_fine)
        ax.plot(t_fine, f_fine, '-', color='crimson', lw=2.0, label='PCHIP', zorder=3)
        ax.fill_between(t_fine, 0, f_fine, alpha=0.20, color='crimson',
                        label=f'Fluence = {fluence:.3e} ± {flu_err:.1e}')

        if lum:
            ax.annotate(
                f"Γ = {lum['gamma']} ± {lum['gamma_err']}\n"
                f"$L_{{\\rm iso}}$ = {lum['L_iso']:.2e} ± {lum['L_iso_err']:.1e} erg/s\n"
                f"$L_{{\\rm int}}$ = {lum['L_int']:.2e} ± {lum['L_int_err']:.1e} erg/s\n"
                f"Fond : {r['bkg_mean']:.2e} ph/cm²/s",
                xy=(0.02, 0.97), xycoords='axes fraction',
                fontsize=7.5, va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85)
            )

        pnum = per.replace('period', 'Période ')
        ax.set_title(
            f"{pnum}  |  Γ = {r['gamma']} ± {r['gamma_err']}\n"
            f"Fluence = {fluence:.3e} ± {flu_err:.1e} ph/cm²",
            fontsize=9)
        ax.set_xlabel("Temps [s]", fontsize=9)
        ax.set_ylabel("Flux [ph/cm²/s]", fontsize=9)
        ax.legend(loc='upper right', fontsize=7.5)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(time.min(), time.max())
        ax.tick_params(labelsize=7)

    # ── Tableau luminosité ISOTROPE ───────────────────────────
    ax_iso = fig.add_subplot(gs[1, :])
    ax_iso.axis('off')

    header_iso = ["Période", "Γ ± δΓ", "Fluence [ph/cm²]",
                  "F_erg [erg/cm²/s]", "L_iso [erg/s]", "E_iso [erg]"]
    rows_iso = []
    for per in periods:
        r = results[per]
        if r is None: continue
        lum = r['lum']
        rows_iso.append([
            per.replace('period', 'P'),
            f"{lum['gamma']} ± {lum['gamma_err']}",
            f"{r['fluence']:.3e} ± {r['fluence_err']:.1e}",
            f"{lum['F_erg']:.3e} ± {lum['F_erg_err']:.1e}",
            f"{lum['L_iso']:.3e} ± {lum['L_iso_err']:.1e}",
            f"{lum['E_iso']:.3e} ± {lum['E_iso_err']:.1e}",
        ])

    tbl = ax_iso.table(cellText=rows_iso, colLabels=header_iso,
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.55)
    for j in range(len(header_iso)):
        tbl[0, j].set_facecolor('#1a5276')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    gamma_str = "  |  ".join(
        [f"Γ({per.replace('period','P')})={results[per]['gamma']} ± {results[per]['gamma_err']}"
         for per in periods if results[per]]
    )
    ax_iso.set_title(
        r"$L_{\rm iso} = 4\pi\,d_L^2\,F_{\rm erg}\,K(z,\Gamma)$"
        r"    $K(z,\Gamma)=(1+z)^{\Gamma-2}$"
        f"    z={REDSHIFT}  |  {gamma_str}",
        fontsize=9, pad=6)

    # ── Tableau luminosité INTRINSÈQUE ────────────────────────
    ax_int = fig.add_subplot(gs[2, :])
    ax_int.axis('off')

    header_int = ["Période", "Γ ± δΓ", "K(z,Γ) ± δK",
                  "L_int [erg/s]", "E_int [erg]", "δ⁴(1+z)"]
    rows_int = []
    for per in periods:
        r = results[per]
        if r is None: continue
        lum = r['lum']
        rows_int.append([
            per.replace('period', 'P'),
            f"{lum['gamma']} ± {lum['gamma_err']}",
            f"{lum['K']:.4f} ± {lum['K_err']:.4f}",
            f"{lum['L_int']:.3e} ± {lum['L_int_err']:.1e}",
            f"{lum['E_int']:.3e} ± {lum['E_int_err']:.1e}",
            f"{lum['doppler_factor']:.3e}",
        ])

    tbl2 = ax_int.table(cellText=rows_int, colLabels=header_int,
                        loc='center', cellLoc='center')
    tbl2.auto_set_font_size(False); tbl2.set_fontsize(8.5); tbl2.scale(1, 1.55)
    for j in range(len(header_int)):
        tbl2[0, j].set_facecolor('#117a65')
        tbl2[0, j].set_text_props(color='white', fontweight='bold')

    ax_int.set_title(
        r"$L_{\rm int} = L_{\rm iso}\,/\,(\delta^4\,(1+z))$"
        r"    $\sigma(L_{\rm int})$ propagée depuis $\sigma(F_{\rm ph}),\,\sigma(\Gamma)$"
        f"    δ = {DELTA_DOPPLER}  [Urry & Padovani 1995, PASP 107, 803]",
        fontsize=9, pad=6)

    plt.savefig(os.path.join(outdir, "pchip_source_luminosity.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  📊 Figure → {outdir}/pchip_source_luminosity.png")

# ==============================================================
# ==================== SAUVEGARDE TEXTE =======================
# ==============================================================

def save_results(results, total_fluence, total_flu_err,
                 total_E_iso, total_E_iso_err,
                 total_E_int, total_E_int_err, outdir):
    os.makedirs(outdir, exist_ok=True)
    result_file = os.path.join(outdir, "fluence_luminosity.txt")

    with open(result_file, 'w') as f:
        f.write("# ============================================================\n")
        f.write("# FLUENCE + LUMINOSITÉ SOURCE — propagation incertitude sur Γ\n")
        f.write("# ============================================================\n")
        f.write(f"# z              = {REDSHIFT}\n")
        f.write(f"# delta (Doppler)= {DELTA_DOPPLER}  [Jorstad et al. 2005]\n")
        f.write(f"# Bande          : {E_MIN_MEV:.0f} MeV – {E_MAX_MEV/1000:.0f} GeV\n")
        f.write("#\n")
        f.write("# Formules :\n")
        f.write("#   F_erg = F_ph × <E>(Γ) × MEV_TO_ERG\n")
        f.write("#   K(z,Γ) = (1+z)^(Γ-2)\n")
        f.write("#   L_iso = 4π dL² × F_erg × K(z,Γ)\n")
        f.write("#   L_int = L_iso / (δ⁴ × (1+z))  [Urry & Padovani 1995, PASP 107, 803]\n")
        f.write("#   E_iso = 4π dL² × Flu_erg × K(z,Γ) / (1+z)\n")
        f.write("#   E_int = E_iso / (δ⁴ × (1+z))\n")
        f.write("#\n")
        f.write("# Propagation σ(Γ) :\n")
        f.write("#   σ²(F_erg) = (<E>×c)² σ²(F_ph) + (F_ph×c×d<E>/dΓ)² σ²(Γ)\n")
        f.write("#   σ²(K)     = (ln(1+z)×K)² σ²(Γ)\n")
        f.write("#   σ²(L_iso) = (4πdL²)² [K² σ²(F_erg) + F_erg² σ²(K)]\n")
        f.write("# ============================================================\n\n")

        for per, r in results.items():
            if r is None: continue
            az  = ACTIVE_ZONES[per]
            lum = r['lum']
            dt  = az['t_stop'] - az['t_start']

            f.write(f"[{per}]\n")
            f.write(f"  # Paramètres temporels\n")
            f.write(f"  t_start          = {az['t_start']:.6e} s\n")
            f.write(f"  t_stop           = {az['t_stop']:.6e} s\n")
            f.write(f"  delta_t          = {dt:.6e} s\n\n")
            f.write(f"  # Indice spectral\n")
            f.write(f"  gamma            = {lum['gamma']} ± {lum['gamma_err']}\n")
            f.write(f"  K_correction     = {lum['K']:.6f} ± {lum['K_err']:.6f}\n\n")
            f.write(f"  # Fond\n")
            f.write(f"  bkg_mean         = {r['bkg_mean']:.6e} {FLUX_UNIT}\n\n")
            f.write(f"  # Flux photon\n")
            f.write(f"  fluence          = {r['fluence']:.6e} ± {r['fluence_err']:.2e} ph/cm²\n")
            f.write(f"  mean_flux        = {r['mean_flux']:.6e} {FLUX_UNIT}\n\n")
            f.write(f"  # Conversion spectrale\n")
            f.write(f"  mean_E_photon    = {lum['mean_E_MeV']:.4f} MeV\n")
            f.write(f"  dE_dGamma        = {lum['dmE_dG']:.4f} MeV\n")
            f.write(f"  F_energy         = {lum['F_erg']:.6e} ± {lum['F_erg_err']:.2e} erg/cm²/s\n")
            f.write(f"  fluence_energy   = {lum['fluence_erg']:.6e} ± {lum['fluence_erg_err']:.2e} erg/cm²\n\n")
            f.write(f"  # Luminosité isotrope : L_iso = 4pi dL^2 F_erg K(z,Gamma)\n")
            f.write(f"  L_iso            = {lum['L_iso']:.6e} ± {lum['L_iso_err']:.2e} erg/s\n")
            f.write(f"  E_iso            = {lum['E_iso']:.6e} ± {lum['E_iso_err']:.2e} erg\n\n")
            f.write(f"  # Luminosité intrinsèque : L_int = L_iso / (delta^4 * (1+z))\n")
            f.write(f"  # Urry & Padovani 1995, PASP 107, 803\n")
            f.write(f"  delta_doppler    = {DELTA_DOPPLER}\n")
            f.write(f"  doppler_factor   = {lum['doppler_factor']:.6e}\n")
            f.write(f"  L_int            = {lum['L_int']:.6e} ± {lum['L_int_err']:.2e} erg/s\n")
            f.write(f"  E_int            = {lum['E_int']:.6e} ± {lum['E_int_err']:.2e} erg\n\n")

        f.write("[TOTAL]\n")
        f.write(f"  fluence_total    = {total_fluence:.6e} ± {total_flu_err:.2e} ph/cm²\n")
        f.write(f"  E_iso_total      = {total_E_iso:.6e} ± {total_E_iso_err:.2e} erg\n")
        f.write(f"  E_int_total      = {total_E_int:.6e} ± {total_E_int_err:.2e} erg\n")

    print(f"  💾 Résultats → {result_file}")

# ==============================================================
# ==================== MAIN ===================================
# ==============================================================

if __name__ == "__main__":

    os.makedirs(OUTDIR, exist_ok=True)
    periods = list(ACTIVE_ZONES.keys())

    results        = {}
    total_fluence  = 0.0;  total_flu_err  = 0.0
    total_E_iso    = 0.0;  total_E_iso_err = 0.0
    total_E_int    = 0.0;  total_E_int_err = 0.0

    print("=" * 65)
    print("  PCHIP SOURCE — fond soustrait + luminosité + σ(Γ)")
    print(f"  z={REDSHIFT}  δ={DELTA_DOPPLER}  "
          f"{E_MIN_MEV:.0f} MeV–{E_MAX_MEV/1000:.0f} GeV")
    print("=" * 65)

    for per in periods:
        az         = ACTIVE_ZONES[per]
        lbl        = per.upper()
        gamma      = GAMMA_PER_PERIOD[per]['gamma']
        gamma_err  = GAMMA_PER_PERIOD[per]['gamma_err']

        print(f"\n{'─'*55}")
        print(f"  ▶ {lbl}  |  Γ = {gamma} ± {gamma_err}")

        # 1. Source
        print("  SOURCE :")
        time, flux, ferr = load_lightcurve(SOURCE_FILES[per])

        # 2. Fond
        print("  FOND (3 régions) :")
        bkg_mean, bkg_err = compute_background(per)

        # 3. Soustraction
        flux_sub, ferr_sub = subtract_background(flux, ferr, bkg_mean, bkg_err)

        # 4. PCHIP + fluence
        pchip, fluence, flu_err, mean_flux = compute_pchip_fluence(
            time, flux_sub, ferr_sub,
            az['t_start'], az['t_stop'], label=lbl
        )
        if pchip is None:
            results[per] = None; continue

        # 5. Luminosité avec propagation σ(Γ)
        mean_flux_err = flu_err / (az['t_stop'] - az['t_start'])
        lum = compute_luminosity(
            mean_flux     = mean_flux,
            mean_flux_err = mean_flux_err,
            fluence       = fluence,
            fluence_err   = flu_err,
            gamma         = gamma,
            gamma_err     = gamma_err,
            label         = lbl
        )

        results[per] = {
            "time":        time,       "flux":        flux_sub,
            "ferr":        ferr_sub,   "pchip":       pchip,
            "fluence":     fluence,    "fluence_err": flu_err,
            "mean_flux":   mean_flux,
            "t_start":     az['t_start'], "t_stop":  az['t_stop'],
            "bkg_mean":    bkg_mean,   "bkg_err":    bkg_err,
            "gamma":       gamma,      "gamma_err":  gamma_err,
            "lum":         lum,
        }

        # Accumulation
        total_fluence   += fluence
        total_flu_err    = np.sqrt(total_flu_err**2   + flu_err**2)
        total_E_iso     += lum['E_iso']
        total_E_iso_err  = np.sqrt(total_E_iso_err**2 + lum['E_iso_err']**2)
        total_E_int     += lum['E_int']
        total_E_int_err  = np.sqrt(total_E_int_err**2 + lum['E_int_err']**2)

    # ── Résumé terminal ───────────────────────────────────────
    print(f"\n{'='*65}")
    print("  RÉSUMÉ FINAL")
    print(f"{'='*65}")
    for per in periods:
        r = results.get(per)
        if not r: continue
        lum = r['lum']
        print(f"\n  {per.upper()}  |  Γ = {lum['gamma']} ± {lum['gamma_err']}")
        print(f"    Fluence  = {r['fluence']:.4e} ± {r['fluence_err']:.2e} ph/cm²")
        print(f"    L_iso    = {lum['L_iso']:.4e} ± {lum['L_iso_err']:.2e} erg/s")
        print(f"    L_int    = {lum['L_int']:.4e} ± {lum['L_int_err']:.2e} erg/s")
        print(f"    E_iso    = {lum['E_iso']:.4e} ± {lum['E_iso_err']:.2e} erg")
        print(f"    E_int    = {lum['E_int']:.4e} ± {lum['E_int_err']:.2e} erg")

    print(f"\n  ── TOTAL ──")
    print(f"  Fluence  = {total_fluence:.4e} ± {total_flu_err:.2e} ph/cm²")
    print(f"  E_iso    = {total_E_iso:.4e} ± {total_E_iso_err:.2e} erg")
    print(f"  E_int    = {total_E_int:.4e} ± {total_E_int_err:.2e} erg")

    # ── Sorties ───────────────────────────────────────────────
    save_results(results,
                 total_fluence, total_flu_err,
                 total_E_iso, total_E_iso_err,
                 total_E_int, total_E_int_err,
                 OUTDIR)
    plot_all(results, OUTDIR)
