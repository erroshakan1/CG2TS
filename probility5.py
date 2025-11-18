#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import os

INPUT_FILE = "merge.txt"
OUT_PNG = "diagnostics_tm_DPPC318_final.png"
FIT_MIN, FIT_MAX = 3, 16  # Makale standardı (Tablo S4)
BIN_WIDTH = 1  # 1.0 Ų² - makale önerisi

def read_numeric_values(path):
    vals = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for tok in line.split():
                try:
                    vals.append(float(tok))
                except:
                    continue
    return np.array(vals, dtype=float)

def compute_histogram(areas, bin_width=1):
    """Compute probability distribution for selected area range"""
    maxA = np.nanmax(areas)
    bins = np.arange(FIT_MIN, FIT_MAX, bin_width)
    counts, edges = np.histogram(areas, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pmf = counts / counts.sum()
    return centers, pmf, counts

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(INPUT_FILE)

    # --- Read all numeric values ---
    areas_all = read_numeric_values(INPUT_FILE)

    # --- Use only range FIT_MIN--FIT_MAX Ų² ---
    areas = areas_all[(areas_all >= FIT_MIN) & (areas_all <= FIT_MAX)]
    if areas.size == 0:
        raise ValueError(f"No data within {FIT_MIN}--{FIT_MAX} Ų² range.")

    # --- Histogram & log(p) ---
    centers, pmf, counts = compute_histogram(areas, BIN_WIDTH)
    x_all = centers

    total_counts = counts.sum() if counts.sum() > 0 else 1.0
    pseudo = 1.0 / total_counts
    counts_pseudo = counts.astype(float) + pseudo
    pmf_pseudo = counts_pseudo / counts_pseudo.sum()
    positive_mask = counts > 0
    x_pos = x_all[positive_mask]
    y_pos = np.log(pmf[positive_mask])

    # --- Fit: use only positive-count bins inside the fit range with y > 1e-4 ---
    fit_mask = (x_pos >= FIT_MIN) & (x_pos <= FIT_MAX) & (np.exp(y_pos) > 1e-4)
    x_fit = x_pos[fit_mask]
    y_fit = y_pos[fit_mask]
    if x_fit.size == 0:
        raise ValueError("No positive-count bins found in fit range; cannot fit model.")
    X_fit = sm.add_constant(x_fit)
    model = sm.OLS(y_fit, X_fit).fit()
    print(model.summary())

    # --- Defect constant π ---
    slope = model.params[1]
    slope_err = model.bse[1]
    if abs(slope) < 1e-12:
        raise ZeroDivisionError("Fitted slope is too close to zero; cannot compute defect constant π reliably.")
    defect_constant = -1.0 / slope
    defect_constant_err = slope_err / (slope ** 2)

    intercept = model.params[0]
    intercept_err = model.bse[0]
    b = np.exp(intercept)
    b_err = b * intercept_err

    print(f"\nFitted model: p(A) ≈ {b:.4g} · exp(-A/{defect_constant:.2f})")
    print(f"π = {defect_constant:.3f} ± {defect_constant_err:.3f} Ų² (fit range {FIT_MIN}-{FIT_MAX})")

    # --- Predictions ---
    x_plot = np.linspace(FIT_MIN, FIT_MAX, 200)
    y_plot = model.predict(sm.add_constant(x_plot))

    # --- Influence diagnostics ---
    influence = model.get_influence()
    summary = influence.summary_frame()
    leverage = summary["hat_diag"].values
    student_resid = summary["student_resid"].values
    standard_resid = summary["standard_resid"].values
    fitted = model.fittedvalues
    resid = model.resid

    # === FIGURE ===
    fig = plt.figure(figsize=(9, 13))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.35)

    # (a) ln(p) vs Area
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x_fit, y_fit, s=20, alpha=0.7,
                facecolors='none', edgecolors='black',
                label=f"range {FIT_MIN}–{FIT_MAX} Ų²")
    ax1.plot(x_plot, y_plot, 'r-', lw=2, label="fit")
    ax1.set_xlabel("Area (Ų$^2$)")
    ax1.set_ylabel("ln(Probability)")
    ax1.set_title("(a) ln(p) vs Area")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) Residuals vs Fitted
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(fitted, resid, s=30, alpha=0.7,
                facecolors='none', edgecolors='black')
    ax2.axhline(0, color='r', lw=1.5)
    lowess = sm.nonparametric.lowess(resid, fitted, frac=0.4)
    ax2.plot(lowess[:, 0], lowess[:, 1], 'r-', lw=1.5)
    ax2.set_xlabel("Fitted values (ln p)")
    ax2.set_ylabel("Residuals")
    ax2.set_title("(b) Residuals vs Fitted")
    ax2.grid(True, alpha=0.3)

    # (c) Normal Q-Q (tamamen düzenlendi)
    ax3 = fig.add_subplot(gs[1, 0])
    osm, osr = stats.probplot(standard_resid, dist="norm", fit=False)
    ax3.scatter(osm, osr, s=30, facecolors='none', edgecolors='black')
    slope, intercept, r_value, _, _ = stats.linregress(osm, osr)
    ax3.plot(osm, slope * osm + intercept, 'r-', lw=1.5)
    ax3.set_title("(c) Normal Q-Q (standardized residuals)")
    ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Standardized Residuals")
    ax3.grid(True, alpha=0.3)

    # (d) Scale-Location
    ax4 = fig.add_subplot(gs[1, 1])
    y_scale = np.sqrt(np.abs(student_resid))
    ax4.scatter(fitted, y_scale, s=30, alpha=0.7,
                facecolors='none', edgecolors='black')
    lowess2 = sm.nonparametric.lowess(y_scale, fitted, frac=0.4)
    ax4.plot(lowess2[:, 0], lowess2[:, 1], 'r-', lw=1.5)
    ax4.set_xlabel("Fitted values (ln p)")
    ax4.set_ylabel("sqrt(|studentized residuals|)")
    ax4.set_title("(d) Scale-Location")
    ax4.grid(True, alpha=0.3)

    # (e) Residuals vs Leverage
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    ax5.scatter(leverage, student_resid, s=30, alpha=0.7,
                facecolors='none', edgecolors='black')
    ax5.axhline(0, color='r', lw=1.5)
    lowess3 = sm.nonparametric.lowess(student_resid, leverage, frac=0.4)
    ax5.plot(lowess3[:, 0], lowess3[:, 1], 'r-', lw=1.5)
    ax5.set_box_aspect(1)
    ax5.set_xlim(0, max(leverage) * 1.05)
    ax5.set_ylim(student_resid.min() * 1.1, student_resid.max() * 1.1)
    ax5.set_xlabel("Leverage")
    ax5.set_ylabel("Studentized residuals")
    ax5.set_title("(e) Residuals vs Leverage")
    ax5.grid(True, alpha=0.3)

    # Make all axes square boxes
    for a in (ax1, ax2, ax3, ax4, ax5):
        try:
            a.set_box_aspect(1)
        except Exception:
            a.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()
    print(f"Diagnostics figure saved to: {OUT_PNG}")

if __name__ == "__main__":
    main()
