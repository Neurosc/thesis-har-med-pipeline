"""
DSE/DVARS toolbox — Python port of Afyouni & Nichols (2018) MATLAB toolbox.

Functions
---------
dse_decomposition  — port of DSEvars.m
dvars_calc         — port of DVARSCalc.m
fmri_diag_plot     — five-panel QC figure (design based on fMRIDiag_plot.m /
                     mu_test_dvars.m usage pattern)

Reference: Afyouni S. & Nichols T.E., Insights and inference for DVARS,
           NeuroImage 172 (2018) 585-601.
Source:    https://github.com/asoroosh/DVARS
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2 as _chi2, norm as _spnorm

try:
    import nibabel as nib
    _NIF = True
except ImportError:
    _NIF = False


# ─── shared internal helpers ──────────────────────────────────────────────────

def _load_bold(V0):
    """Return (I, T) float64 from NIfTI path, 4-D array, or 2-D (I×T) array."""
    if isinstance(V0, (str, Path)):
        if not _NIF:
            raise ImportError("nibabel required for NIfTI input")
        d = nib.load(str(V0)).get_fdata(dtype=np.float64)
        T = d.shape[3] if d.ndim == 4 else d.shape[-1]
        return d.reshape(-1, T)
    Y = np.asarray(V0, dtype=np.float64)
    if Y.ndim == 4:
        return Y.reshape(-1, Y.shape[3])
    if Y.ndim != 2 or Y.shape[0] <= Y.shape[1]:
        raise ValueError("Y must be IxT with I > T")
    return Y


def _remove_bad_voxels(Y):
    """Remove all-zero and NaN-containing voxels. Returns (Y_clean, kept_idx)."""
    # DSEvars.m L190-L194, DVARSCalc.m L213-L218
    row_sum = Y.sum(axis=1)
    bad = np.union1d(np.where(np.isnan(row_sum))[0],
                     np.where(row_sum == 0)[0])
    idx = np.setdiff1d(np.arange(Y.shape[0]), bad)
    return Y[idx], idx


def _iqrsd(x):
    """(Q75-Q25)/1.349 — IQR-based robust std. DVARSCalc.m L265."""
    return (np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349


def _h_iqrsd(x):
    """(Q50-Q25)/1.349*2 — half-IQR robust std. DVARSCalc.m L266."""
    return (np.quantile(x, 0.50) - np.quantile(x, 0.25)) / 1.349 * 2


# ─── 1. DSE decomposition ─────────────────────────────────────────────────────

def dse_decomposition(Y, norm_scale=100):
    """
    DSE variance decomposition: D (fast), S (slow), E (edge) components.
    Faithful port of DSEvars.m (Afyouni & Nichols 2018).

    Reference: Afyouni S. & Nichols T.E., NeuroImage 172 (2018) 585-601.
    Source:    https://github.com/asoroosh/DVARS  (DSEvars.m)

    Parameters
    ----------
    Y : (I, T) array-like or NIfTI path.
    norm_scale : float
        Intensity normalisation target (default 100). Equivalent to DSEvars
        'Norm' option: Y = (Y / median(mean_image)) * norm_scale.

    Returns
    -------
    dict with keys:
        Avar_ts, Dvar_ts, Svar_ts, Evar_ts          shapes (T,), (T-1,), (T-1,), (2,)
        g_Avar_ts, g_Dvar_ts, g_Svar_ts, g_Evar_ts  global components (same shapes)
        ng_Avar_ts, ng_Dvar_ts, ng_Svar_ts, ng_Evar_ts  non-global components
        w_Avar, w_Dvar, w_Svar, w_Evar               whole-brain sums (scalars)
        g_Avar, g_Dvar, g_Svar, g_Evar               global sums
        ng_Avar, ng_Dvar, ng_Svar, ng_Evar           non-global sums
        g_Ats, g_Dts, g_Sts                          global signal ts for vis
        Stat : dict  {Labels, SS, MS, RMS, Prntg, RelVar, DeltapDvar, pDvar, dim, dim0}
    """
    Y = _load_bold(Y)
    I0, T0 = Y.shape

    # DSEvars.m L188 — whole-image mean before masking (for reporting only)
    mvY_WholeImage = Y.mean(axis=1)

    # Remove zero/NaN voxels — DSEvars.m L190-L195
    Y, _idx = _remove_bad_voxels(Y)
    I1 = Y.shape[0]

    mvY_Untouched = Y.mean(axis=1)

    # Intensity normalisation — DSEvars.m L201-L215
    # 'Norm' path: md = median(mean(Y,2)), Y = (Y/md)*scl
    md  = np.median(Y.mean(axis=1))          # median of voxel-wise temporal mean
    Y   = (Y / md) * norm_scale

    # Centre (demean each voxel) — DSEvars.m L217-L221
    mvY_NormInt  = Y.mean(axis=1)            # grand mean post-norm (used for reporting)
    Y            = Y - mvY_NormInt[:, np.newaxis]   # repmat equivalent
    mvY_Demeaned = Y.mean(axis=1)

    # Global signal — DSEvars.m L231
    Ybar = Y.sum(axis=0) / I1               # (T0,)

    # Lagged images — DSEvars.m L233-L241
    D    = Y[:, :-1] - Y[:, 1:]             # (I1, T0-1)  fast-variance image
    Dbar = D.sum(axis=0) / I1               # (T0-1,)
    S    = Y[:, :-1] + Y[:, 1:]             # (I1, T0-1)  slow-variance image
    Sbar = S.sum(axis=0) / I1               # (T0-1,)
    Ytail = Y[:, -1]                        # (I1,)
    Yhead = Y[:, 0]                         # (I1,)
    Ytbar = Ytail.sum() / I1                # scalar
    Y1bar = Yhead.sum() / I1                # scalar

    # Time-resolved variance components — DSEvars.m L244-L257
    # V_Img.Avar_ts = Y.^2,        V.Avar_ts = mean(V_Img.Avar_ts) along voxels
    Avar_ts  = (Y ** 2).mean(axis=0)                             # (T0,)
    # V_Img.Dvar_ts = D.^2./4
    Dvar_ts  = (D ** 2).mean(axis=0) / 4                        # (T0-1,)
    # V_Img.Svar_ts = S.^2./4
    Svar_ts  = (S ** 2).mean(axis=0) / 4                        # (T0-1,)
    # V_Img.Evar_ts = [Yhead,Ytail].^2./2  — I1×2 matrix, then mean over voxels
    Evar_ts  = np.column_stack([Yhead, Ytail]) ** 2 / 2         # (I1, 2)
    Evar_ts  = Evar_ts.mean(axis=0)                             # (2,)

    # Whole-brain scalars — DSEvars.m L298-L301
    w_Avar = float(Avar_ts.sum())
    w_Dvar = float(Dvar_ts.sum())
    w_Svar = float(Svar_ts.sum())
    w_Evar = float(Evar_ts.sum())

    # Global variance components — DSEvars.m L303-L316
    g_Avar_ts = Ybar ** 2                                        # (T0,)
    g_Dvar_ts = Dbar ** 2 / 4                                   # (T0-1,)
    g_Svar_ts = Sbar ** 2 / 4                                   # (T0-1,)
    g_Evar_ts = Ybar[[0, T0 - 1]] ** 2 / 2                     # (2,)

    g_Avar = float(g_Avar_ts.sum())
    g_Dvar = float(g_Dvar_ts.sum())
    g_Svar = float(g_Svar_ts.sum())
    g_Evar = float(g_Evar_ts.sum())

    # Non-global variance components — DSEvars.m L324-L332
    ng_Avar_ts = ((Y - Ybar[np.newaxis, :]) ** 2).mean(axis=0)             # (T0,)
    ng_Dvar_ts = ((D - Dbar[np.newaxis, :]) ** 2).mean(axis=0) / 4        # (T0-1,)
    ng_Svar_ts = ((S - Sbar[np.newaxis, :]) ** 2).mean(axis=0) / 4        # (T0-1,)
    _ng_e = np.column_stack([(Yhead - Y1bar) ** 2, (Ytail - Ytbar) ** 2])
    ng_Evar_ts = _ng_e.mean(axis=0) / 2                                    # (2,)

    ng_Avar = float(ng_Avar_ts.sum())
    ng_Dvar = float(ng_Dvar_ts.sum())
    ng_Svar = float(ng_Svar_ts.sum())
    ng_Evar = float(ng_Evar_ts.sum())

    # ANOVA table — DSEvars.m L350-L357
    # SS = I1 * [w_Avar, w_Dvar, w_Svar, w_Evar, g_Avar, g_Dvar, g_Svar, g_Evar]
    SS    = I1 * np.array([w_Avar, w_Dvar, w_Svar, w_Evar,
                            g_Avar, g_Dvar, g_Svar, g_Evar])
    MS    = SS / I1 / T0
    RMS   = np.sqrt(MS)
    Prntg = RMS ** 2 / RMS[0] ** 2 * 100
    # Expected values for iid case — DSEvars.m L355-L356
    _eb   = np.array([1.0, (T0-1)/T0/2, (T0-1)/T0/2, 1.0/T0])
    Expd  = np.concatenate([_eb, _eb / I1])
    RelVar = Prntg / 100.0 / Expd
    Labels = ['Avar','Dvar','Svar','Evar','g_Avar','g_Dvar','g_Svar','g_Evar']

    # Standardised DVARS — DSEvars.m L386-L387
    # DeltapDvar = (Dvar_ts - median(Dvar_ts)) / mean(Avar_ts) * 100
    DeltapDvar = (Dvar_ts - np.median(Dvar_ts)) / Avar_ts.mean() * 100
    # pDvar = Dvar_ts / mean(Avar_ts) * 100
    pDvar = Dvar_ts / Avar_ts.mean() * 100

    Stat = dict(
        Labels=Labels, SS=SS, MS=MS, RMS=RMS, Prntg=Prntg, RelVar=RelVar,
        DeltapDvar=DeltapDvar, pDvar=pDvar,
        dim=(I1, T0), dim0=(I0, T0),
        GrandMean_Untouched=float(mvY_Untouched.mean()),
        GrandMean_NormInt=float(mvY_NormInt.mean()),
        GrandMean_Demeaned=float(mvY_Demeaned.mean()),
        GranMean_WholeBrain=float(mvY_WholeImage.mean()),
    )

    return dict(
        Avar_ts=Avar_ts, Dvar_ts=Dvar_ts, Svar_ts=Svar_ts, Evar_ts=Evar_ts,
        g_Avar_ts=g_Avar_ts, g_Dvar_ts=g_Dvar_ts,
        g_Svar_ts=g_Svar_ts, g_Evar_ts=g_Evar_ts,
        ng_Avar_ts=ng_Avar_ts, ng_Dvar_ts=ng_Dvar_ts,
        ng_Svar_ts=ng_Svar_ts, ng_Evar_ts=ng_Evar_ts,
        w_Avar=w_Avar, w_Dvar=w_Dvar, w_Svar=w_Svar, w_Evar=w_Evar,
        g_Avar=g_Avar, g_Dvar=g_Dvar, g_Svar=g_Svar, g_Evar=g_Evar,
        ng_Avar=ng_Avar, ng_Dvar=ng_Dvar, ng_Svar=ng_Svar, ng_Evar=ng_Evar,
        g_Ats=Ybar, g_Dts=Dbar / 2, g_Sts=Sbar / 2,   # DSEvars.m L309-L311
        Stat=Stat,
    )


# ─── 2. DVARS with statistical inference ─────────────────────────────────────

def dvars_calc(Y, scale=1/100, trans_power=1/3,
               var_type='hIQR', mean_type='median', test_method='X2'):
    """
    DVARS with statistical inference. Faithful port of DVARSCalc.m.

    Reference: Afyouni S. & Nichols T.E., NeuroImage 172 (2018) 585-601.
    Source:    https://github.com/asoroosh/DVARS  (DVARSCalc.m)

    Parameters
    ----------
    Y : (I, T) array-like or NIfTI path.
    scale : float
        Intensity scale factor applied as Y *= scale (DVARSCalc.m 'scale' option
        with md=1). Default 1/100 matches mu_test_dvars.m usage.
    trans_power : float
        Cube-root transformation power (default 1/3). DVARSCalc.m 'TransPower'.
    var_type : 'hIQR' | 'IQR'
        Robust variance estimator (default 'hIQR'). DVARSCalc.m 'VarType'.
    mean_type : 'median' | 'sig2bar' | 'sig2median' | 'sigbar2' | 'xbar'
        Robust mean estimator (default 'median'). DVARSCalc.m 'MeanType'.
    test_method : 'X2' | 'Z'
        Test statistic (default 'X2'). DVARSCalc.m 'TestMethod'.

    Returns
    -------
    DVARS : (T-1,) array — classic DVARS = sqrt(mean(diff^2))
    Stat  : dict — {DVARS2, pvals, Zval, c, nu, E, S, Mu0, Avar,
                    SDVARS_X2, SDVARS_Z, DeltapDvar, dim, dim0, GrandMean,
                    GrandMean0, Config}
    """
    Y = _load_bold(Y)
    I0, T0 = Y.shape

    # Remove zero/NaN voxels — DVARSCalc.m L213-L219
    Y, _idx = _remove_bad_voxels(Y)
    I1 = Y.shape[0]

    mvY0 = Y.mean(axis=1)    # untouched grand mean

    # Intensity scaling — DVARSCalc.m L232-L235
    # 'scale' with md=1: Y = (Y/1) * scale = Y * scale
    Y = Y * scale

    # Centre — DVARSCalc.m L244-L246
    mvY = Y.mean(axis=1)
    Y   = Y - mvY[:, np.newaxis]

    # Classic DVARS — DVARSCalc.m L284-L285
    # diff(Y,1,2) in MATLAB = Y[:,1:] - Y[:,:-1]  →  np.diff(axis=1)
    DY    = np.diff(Y, n=1, axis=1)                           # (I1, T0-1)
    DVARS = np.sqrt((DY ** 2).sum(axis=0) / I1)               # (T0-1,)

    # Mean squared diff — DVARSCalc.m L305
    DVARS2 = (DY ** 2).mean(axis=0)                           # (T0-1,)

    # Robust std per voxel — DVARSCalc.m L306
    # IQRsd(DY')' : IQRsd applied across time (axis=1 of DY)
    Rob_S_D = (np.quantile(DY, 0.75, axis=1)
               - np.quantile(DY, 0.25, axis=1)) / 1.349       # (I1,)

    # Robust mean estimators — DVARSCalc.m L308-L310
    # Mn = [mean(Rob_S_D^2), median(Rob_S_D^2), median(DVARS2),
    #        mean(Rob_S_D)^2, mean(DVARS2)]
    Mn = np.array([
        float(np.mean(Rob_S_D ** 2)),    # sig2bar
        float(np.median(Rob_S_D ** 2)),  # sig2median
        float(np.median(DVARS2)),        # median  ← default (WhichExpVal=3)
        float(np.mean(Rob_S_D) ** 2),   # sigbar2
        float(np.mean(DVARS2)),          # xbar
    ])
    _mean_names = ['sig2bar', 'sig2median', 'median', 'sigbar2', 'xbar']
    WhichExpVal = _mean_names.index(mean_type)

    # Power transformation — DVARSCalc.m L312-L313
    dd  = trans_power
    Z   = DVARS2 ** dd
    M_Z = float(np.median(Z))

    # Robust variance estimators — DVARSCalc.m L315-L316
    # Va = [var(DVARS2), (1/dd * M_Z^(1/dd-1) * IQRsd(Z))^2,
    #                    (1/dd * M_Z^(1/dd-1) * H_IQRsd(Z))^2]
    _f  = 1.0 / dd * M_Z ** (1.0 / dd - 1)
    Va  = np.array([
        float(np.var(DVARS2)),                     # S2
        float((_f * _iqrsd(Z)) ** 2),              # IQRd
        float((_f * _h_iqrsd(Z)) ** 2),            # hIQRd  ← default (WhichVar=3)
    ])
    _var_names = ['S2', 'IQR', 'hIQR']
    WhichVar = _var_names.index(var_type)

    M_DV2 = Mn[WhichExpVal]
    S_DV2 = float(np.sqrt(Va[WhichVar]))

    # Statistical test — DVARSCalc.m L326-L351
    Zval = (DVARS2 - M_DV2) / S_DV2                           # (T0-1,)

    if test_method == 'X2':
        # X2stat = 2*m/s^2 * x;  X2df = 2*m^2/s^2
        # X2p = chi2cdf(X2stat, X2df, 'upper')
        c    = 2.0 * M_DV2 / S_DV2 ** 2 * DVARS2             # (T0-1,)
        nu   = 2.0 * M_DV2 ** 2 / S_DV2 ** 2                 # scalar
        Pval = _chi2.sf(c, nu)                                 # (T0-1,)

        # NDVARS_X2 = -norminv(Pval) — DVARSCalc.m L342
        with np.errstate(divide='ignore', invalid='ignore'):
            NDVARS_X2 = -_spnorm.ppf(Pval)
        # Replace infs with X2p0 — DVARSCalc.m L343-L346
        X2p0 = (c - nu) / np.sqrt(2.0 * nu)
        inf_mask = np.isinf(NDVARS_X2)
        NDVARS_X2[inf_mask] = X2p0[inf_mask]
        NDVARS_Z = np.full_like(DVARS2, np.nan)

    elif test_method == 'Z':
        # DVARSCalc.m L327-L334
        Pval     = 1.0 - _spnorm.cdf(Zval)
        c        = np.full_like(DVARS2, np.nan)
        nu       = np.nan
        NDVARS_X2 = np.full_like(DVARS2, np.nan)
        NDVARS_Z  = Zval

    else:
        raise ValueError(f"Unknown test_method '{test_method}': use 'X2' or 'Z'")

    # Mu0 and Avar — DVARSCalc.m L371-L372
    # Stat.Mu0 = mean(IQRsd(Y)) — IQRsd applied to Y (I1×T0) along voxel axis
    Mu0  = float(np.mean(
        (np.quantile(Y, 0.75, axis=0) - np.quantile(Y, 0.25, axis=0)) / 1.349
    ))
    Avar = float(np.mean(Y ** 2))    # grand mean Y^2 = A-var

    # DeltapDvar — DVARSCalc.m L379
    # (DVARS2 - median(DVARS2)) / (4 * Avar) * 100
    DeltapDvar = (DVARS2 - np.median(DVARS2)) / (4.0 * Avar) * 100

    Stat = dict(
        DVARS2=DVARS2,
        E=Mn, S=np.sqrt(Va),
        nu=nu, c=c,
        pvals=Pval,
        Zval=Zval,
        Mu0=Mu0, Avar=Avar,
        SDVARS_X2=NDVARS_X2,
        SDVARS_Z=NDVARS_Z,
        DeltapDvar=DeltapDvar,
        dim=(I1, T0), dim0=(I0, T0),
        GrandMean=float(mvY.mean()),
        GrandMean0=float(mvY0.mean()),
        Config=dict(
            TestMeth=test_method,
            PowerTrans=dd,
            WhichExpVal=_mean_names[WhichExpVal],
            WhichVar=_var_names[WhichVar],
        ),
    )

    return DVARS, Stat


# ─── 3. Five-panel QC figure ──────────────────────────────────────────────────

# Afyouni colour scheme
_C_D   = '#D62728'   # red   — D (fast variance)
_C_S   = '#1F77B4'   # blue  — S (slow variance)
_C_E   = '#7F7F7F'   # grey  — E (edge)
_C_gD  = '#F7B6B6'   # light red   — global D
_C_gS  = '#AEC7E8'   # light blue  — global S
_C_gE  = '#C7C7C7'   # light grey  — global E
_C_FD  = '#2CA02C'   # green — FD
_C_ROT = '#8C564B'   # brown — rotation
_C_TR  = '#BCBD22'   # olive — translation


def fmri_diag_plot(V_dse, dvars_stat, fd=None, abs_mov=None,
                   sig_idx=None, title=None, figsize=(5, 7),
                   fd_thresh=0.3, axes=None):
    """
    Five-panel fMRI diagnostic figure. Based on fMRIDiag_plot.m (Afyouni 2018)
    and mu_test_dvars.m usage pattern.

    Reference: Afyouni S. & Nichols T.E., NeuroImage 172 (2018) 585-601.

    Parameters
    ----------
    V_dse      : dict — from dse_decomposition()
    dvars_stat : dict — Stat sub-dict from dse_decomposition() or from dvars_calc()
    fd         : (T,) FD time series
    abs_mov    : (T, 2) — col 0: |Rot|×10 (rad×10), col 1: |Trans| (mm)
                 Matches MATLAB: [(FD_Stat.AbsRot)*10, FD_Stat.AbsTrans]
    sig_idx    : 0-indexed frame positions in DVARS space (length T-1) marking
                 significant timepoints (e.g. pvals < 0.05)
    title      : panel column title
    figsize    : (width_in, height_in) used only when axes=None
    fd_thresh  : horizontal reference line on FD panel
    axes       : optional (5,) array of pre-created Axes (for embedding in a
                 larger multi-column figure)

    Returns
    -------
    fig : matplotlib Figure
    """
    T  = len(V_dse['Avar_ts'])
    xA = np.arange(T)           # frame axis for A-var (length T)
    xD = np.arange(1, T)        # frame axis for D/S-var (length T-1), 1-indexed

    if axes is None:
        fig, axes = plt.subplots(
            5, 1, figsize=figsize, sharex=True,
            gridspec_kw={'height_ratios': [1.2, 1.2, 1.5, 1.5, 1.5],
                         'hspace': 0.45}
        )
    else:
        fig = axes[0].get_figure()

    # ── Panel 1: FD ───────────────────────────────────────────────────────────
    ax = axes[0]
    if fd is not None:
        ax.plot(xA, fd, color=_C_FD, lw=0.9, rasterized=True)
        ax.axhline(fd_thresh, color='black', lw=0.7, ls='--', alpha=0.6)
        _fd_nan = np.where(~np.isnan(fd) & (fd > fd_thresh))[0]
        if len(_fd_nan):
            ax.scatter(_fd_nan, fd[_fd_nan], color='red', s=6, zorder=4,
                       linewidths=0)
    ax.set_ylabel('FD (mm)', fontsize=8)
    ax.set_title('Framewise Displacement', fontsize=8, pad=2)

    # ── Panel 2: AbsMov ───────────────────────────────────────────────────────
    ax = axes[1]
    if abs_mov is not None:
        am = np.asarray(abs_mov)
        ax.plot(xA, am[:, 0], color=_C_ROT, lw=0.9, label='|Rot|×10',
                rasterized=True)
        ax.plot(xA, am[:, 1], color=_C_TR,  lw=0.9, label='|Trans|',
                rasterized=True)
        ax.legend(loc='upper right', fontsize=7, handlelength=1.2)
    ax.set_ylabel('Motion', fontsize=8)
    ax.set_title('AbsMov (|Rot|×10, |Trans|)', fontsize=8, pad=2)

    # ── Panel 3: Δ%D-var with significance markers ────────────────────────────
    ax = axes[2]
    dD = dvars_stat.get('DeltapDvar', V_dse['Stat']['DeltapDvar'])
    ax.plot(xD, dD, color=_C_D, lw=0.9, rasterized=True)
    ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.5)
    if sig_idx is not None and len(sig_idx):
        _si = np.asarray(sig_idx, dtype=int)
        _si = _si[_si < len(dD)]
        ax.scatter(xD[_si], dD[_si], color='black', s=10, zorder=5,
                   linewidths=0, label='p<0.05')
        ax.legend(loc='upper right', fontsize=7, handlelength=1.0)
    ax.set_ylabel('Δ%D-var', fontsize=8)
    ax.set_title('Standardised DVARS (Δ%D-var)', fontsize=8, pad=2)

    # ── Panel 4: Non-global D/S/E variance ────────────────────────────────────
    ax = axes[3]
    ax.plot(xD, V_dse['ng_Dvar_ts'], color=_C_D, lw=0.9,
            label='ng_Dvar', rasterized=True)
    ax.plot(xD, V_dse['ng_Svar_ts'], color=_C_S, lw=0.9,
            label='ng_Svar', rasterized=True)
    ax.scatter([xA[0], xA[-1]], V_dse['ng_Evar_ts'], color=_C_E,
               s=20, zorder=5, linewidths=0, label='ng_Evar')
    ax.set_ylabel('Var', fontsize=8)
    ax.set_title('Non-global D/S/E var', fontsize=8, pad=2)
    ax.legend(loc='upper right', fontsize=7, handlelength=1.2)

    # ── Panel 5: Global D/S/E variance ────────────────────────────────────────
    ax = axes[4]
    ax.plot(xD, V_dse['g_Dvar_ts'], color=_C_gD, lw=0.9,
            label='g_Dvar', rasterized=True)
    ax.plot(xD, V_dse['g_Svar_ts'], color=_C_gS, lw=0.9,
            label='g_Svar', rasterized=True)
    ax.scatter([xA[0], xA[-1]], V_dse['g_Evar_ts'], color=_C_gE,
               s=20, zorder=5, linewidths=0, label='g_Evar')
    ax.set_ylabel('Var', fontsize=8)
    ax.set_title('Global D/S/E var', fontsize=8, pad=2)
    ax.set_xlabel('Frame index', fontsize=8)
    ax.legend(loc='upper right', fontsize=7, handlelength=1.2)

    if title:
        fig.suptitle(title, fontsize=9, y=1.01)

    return fig
