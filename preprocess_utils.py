"""
Load preprocess_params.csv and apply forward/inverse transform, plus non-negativity clamping.
Also provides fit_transform_per_column / fit_transform_for_augmentation for in-memory fit (no run_preprocess_pipeline needed).

Usage:
    params = load_preprocess_params("preprocess_params.csv")
    X_prep = transform(X_raw, params, feat_cols)
    X_prep_clamped = clamp_for_nonneg(X_prep, params, feat_cols)
    X_raw = inverse_transform(X_prep_clamped, params, feat_cols)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer

# -----------------------------------------------------------------------------
# Optional variance filters (Python implementation)
# -----------------------------------------------------------------------------
# Keep this section at the top so it is easy to tweak before running.
DEFAULT_VAR_FILTER_METHOD = "none"  # "none" | "nzv" | "variance"
DEFAULT_VAR_THRESHOLD = 0.01        # used when method == "variance"
DEFAULT_NZV_FREQ_CUT = 95.0 / 5.0   # used when method == "nzv"
DEFAULT_NZV_UNIQUE_CUT = 10.0       # used when method == "nzv"
DEFAULT_SKIP_PREFIXES = ("groups",) # shared skip prefixes for all preprocessing


def _is_skipped_column(col_name, skip_columns=None, skip_prefixes=DEFAULT_SKIP_PREFIXES):
    if skip_columns and col_name in set(skip_columns):
        return True
    return any(col_name.startswith(prefix) for prefix in (skip_prefixes or ()))


def select_numeric_feature_columns(
    df,
    skip_columns=None,
    skip_prefixes=DEFAULT_SKIP_PREFIXES,
):
    """
    Select numeric feature columns only, skipping all group columns (e.g. groups, groups2).
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = [
        c for c in numeric_cols
        if not _is_skipped_column(c, skip_columns=skip_columns, skip_prefixes=skip_prefixes)
    ]
    return feat_cols


def select_passthrough_columns(
    df,
    skip_columns=None,
    skip_prefixes=DEFAULT_SKIP_PREFIXES,
):
    """
    Select columns that should bypass preprocessing and be copied back later.
    """
    passthrough_cols = [
        c for c in df.columns
        if _is_skipped_column(c, skip_columns=skip_columns, skip_prefixes=skip_prefixes)
    ]
    return passthrough_cols


def _near_zero_var_mask(
    X_df,
    freq_cut=DEFAULT_NZV_FREQ_CUT,
    unique_cut=DEFAULT_NZV_UNIQUE_CUT,
):
    """
    caret::nearZeroVar-like rule:
    flag if zero variance OR (percent_unique <= unique_cut and freq_ratio > freq_cut).
    """
    n_rows = max(len(X_df), 1)
    drop_mask = []
    for col in X_df.columns:
        series = X_df[col]
        counts = series.value_counts(dropna=False)
        n_unique = int(counts.shape[0])

        # zeroVar in caret terms (single unique value)
        if n_unique <= 1:
            drop_mask.append(True)
            continue

        # freqRatio = most common / second most common
        top = float(counts.iloc[0]) if counts.shape[0] >= 1 else 0.0
        second = float(counts.iloc[1]) if counts.shape[0] >= 2 else 0.0
        freq_ratio = np.inf if second == 0 else top / second

        percent_unique = 100.0 * n_unique / n_rows
        is_nzv = (percent_unique <= unique_cut) and (freq_ratio > freq_cut)
        drop_mask.append(bool(is_nzv))
    return np.array(drop_mask, dtype=bool)


def filter_feature_columns(
    df,
    filter_method=DEFAULT_VAR_FILTER_METHOD,
    variance_threshold=DEFAULT_VAR_THRESHOLD,
    nzv_freq_cut=DEFAULT_NZV_FREQ_CUT,
    nzv_unique_cut=DEFAULT_NZV_UNIQUE_CUT,
    skip_columns=None,
    skip_prefixes=DEFAULT_SKIP_PREFIXES,
):
    """
    Filter numeric feature columns using one of:
      - "none": no filtering
      - "variance": drop columns with variance < variance_threshold
      - "nzv": nearZeroVar-like filtering (caret-style rule)

    Returns:
      kept_cols: list[str]
      filtered_df: pd.DataFrame (kept feature columns only)
    """
    method = str(filter_method).lower()
    if method not in {"none", "variance", "nzv"}:
        raise ValueError("filter_method must be one of: 'none', 'variance', 'nzv'")

    feat_cols = select_numeric_feature_columns(
        df, skip_columns=skip_columns, skip_prefixes=skip_prefixes
    )
    X_feat = df[feat_cols].copy()

    if method == "none":
        kept_cols = feat_cols
    elif method == "variance":
        variances = X_feat.apply(
            lambda z: np.var(z[np.isfinite(z)], ddof=1) if np.isfinite(z).sum() >= 2 else 0.0,
            axis=0,
        )
        kept_cols = variances.index[variances >= variance_threshold].tolist()
    else:  # method == "nzv"
        nzv_drop_mask = _near_zero_var_mask(
            X_feat,
            freq_cut=nzv_freq_cut,
            unique_cut=nzv_unique_cut,
        )
        kept_cols = X_feat.columns[~nzv_drop_mask].tolist()

    if len(kept_cols) == 0:
        raise ValueError("All numeric feature columns were filtered out. Please relax filter settings.")

    return kept_cols, df[kept_cols].copy()


def _yeo_johnson_forward(x, lam):
    """Yeo-Johnson forward transform (matches sklearn)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    if np.abs(lam) < 1e-10:
        out[pos] = np.log(x[pos] + 1)
    else:
        out[pos] = (np.power(x[pos] + 1, lam) - 1) / lam
    if np.abs(lam - 2) < 1e-10:
        out[neg] = -np.log(-x[neg] + 1)
    else:
        out[neg] = -(np.power(-x[neg] + 1, 2 - lam) - 1) / (2 - lam)
    return out


def _yeo_johnson_inv(y, lam):
    """Inverse Yeo-Johnson. y is scalar or array, lam is scalar."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    mask_pos = y >= 0
    mask_neg = ~mask_pos
    if np.abs(lam) < 1e-10:
        out[mask_pos] = np.exp(y[mask_pos]) - 1
        out[mask_neg] = 1 - np.exp(-y[mask_neg])
    else:
        out[mask_pos] = np.where(
            np.abs(lam) < 1e-10,
            np.exp(y[mask_pos]) - 1,
            (y[mask_pos] * lam + 1) ** (1 / lam) - 1,
        )
        out[mask_neg] = np.where(
            np.abs(lam - 2) < 1e-10,
            1 - np.exp(-y[mask_neg]),
            1 - (-(2 - lam) * y[mask_neg] + 1) ** (1 / (2 - lam)),
        )
    return out


def _yeo_johnson_inv_simple(y, lam):
    """Inverse YJ using sklearn-style logic."""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    pos = y >= 0
    neg = ~pos
    # y >= 0: ((y*lam + 1)^(1/lam)) - 1  for lam != 0; exp(y)-1 for lam==0
    # y < 0:  1 - ((-(2-lam)*y + 1)^(1/(2-lam)))  for lam != 2; 1-exp(-y) for lam==2
    if np.abs(lam) < 1e-10:
        out[pos] = np.exp(y[pos]) - 1
    else:
        out[pos] = np.power(y[pos] * lam + 1, 1 / lam) - 1
    if np.abs(lam - 2) < 1e-10:
        out[neg] = 1 - np.exp(-y[neg])
    else:
        out[neg] = 1 - np.power(-(2 - lam) * y[neg] + 1, 1 / (2 - lam))
    return out


def load_preprocess_params(path):
    """Load preprocess_params.csv. Returns DataFrame with Feature as index for easy lookup."""
    df = pd.read_csv(path)
    return df


def _fit_yj(col):
    """Fit YJ on one column, return (lambda, yj_values)."""
    col = np.asarray(col, dtype=float).reshape(-1, 1)
    valid = np.isfinite(col)
    if not np.all(valid):
        col = np.where(valid, col, np.nanmean(col[valid]))
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(col)
    lam = float(pt.lambdas_[0])
    yj = pt.transform(col).ravel()
    return lam, yj


def fit_transform_per_column(X, columns, yj_mode, zscore, skewness_threshold=1.0):
    """
    Per column: optionally YJ (selected = only if skewness > threshold, all = always), optionally zscore.
    Always produce a params row for inverse; untransformed columns get preprocess_label "A".
    Returns (params_df, X_prep).
    params_df columns: Feature, preprocess_label, lambda, mu, sigma, nonneg_label, X_prep_min.
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    rows = []
    X_prep = np.empty_like(X)

    for j in range(n_features):
        col = X[:, j].copy()
        valid = np.isfinite(col)
        if not np.all(valid):
            col = np.where(valid, col, np.nanmean(col[valid]))
        name = columns[j]
        raw_nonneg = np.all(X[:, j] >= -1e-12)

        skew = float(stats.skew(col, nan_policy="omit")) if np.sum(valid) >= 3 else 0.0
        do_yj = (yj_mode == "all") or (yj_mode == "selected" and skew > skewness_threshold)

        if not do_yj and not zscore:
            X_prep[:, j] = col
            rows.append({
                "Feature": name,
                "preprocess_label": "A",
                "lambda": np.nan,
                "mu": 0.0,
                "sigma": 1.0,
                "nonneg_label": 0,
                "X_prep_min": np.nan,
            })
            continue

        if do_yj:
            lam, yj = _fit_yj(col)
            if zscore:
                mu = float(np.mean(yj))
                sigma = float(np.std(yj))
                if sigma < 1e-12:
                    sigma = 1.0
                X_prep[:, j] = (yj - mu) / sigma
                x_prep_min = (-mu / sigma) if (raw_nonneg and sigma != 0) else np.nan
                nonneg_label = 1 if raw_nonneg else 0
                rows.append({
                    "Feature": name,
                    "preprocess_label": "B",
                    "lambda": lam,
                    "mu": mu,
                    "sigma": sigma,
                    "nonneg_label": nonneg_label,
                    "X_prep_min": x_prep_min if raw_nonneg else np.nan,
                })
            else:
                X_prep[:, j] = yj
                x_prep_min = 0.0 if raw_nonneg else np.nan
                rows.append({
                    "Feature": name,
                    "preprocess_label": "B",
                    "lambda": lam,
                    "mu": 0.0,
                    "sigma": 1.0,
                    "nonneg_label": 1 if raw_nonneg else 0,
                    "X_prep_min": x_prep_min,
                })
        else:
            mu = float(np.mean(col))
            sigma = float(np.std(col))
            if sigma < 1e-12:
                sigma = 1.0
            X_prep[:, j] = (col - mu) / sigma
            x_prep_min = (-mu / sigma) if (raw_nonneg and sigma != 0) else np.nan
            rows.append({
                "Feature": name,
                "preprocess_label": "C",
                "lambda": np.nan,
                "mu": mu,
                "sigma": sigma,
                "nonneg_label": 1 if raw_nonneg else 0,
                "X_prep_min": x_prep_min if raw_nonneg else np.nan,
            })

    params_df = pd.DataFrame(rows)
    return params_df, X_prep


def fit_transform_for_augmentation(X_raw, feat_cols, yj_mode, zscore, skewness_threshold=1.0):
    """
    Fit YJ/zscore per column on raw data and return (X_prep, params_df) for use in augmentation.
    Does not read or write any file; params are in-memory for this run.
    yj_mode: False (no YJ), "selected" (YJ only when skewness > threshold), "all".
    zscore: True/False.
    Returns (X_prep, params_df). params_df has columns Feature, preprocess_label, lambda, mu, sigma, nonneg_label, X_prep_min.
    """
    params_df, X_prep = fit_transform_per_column(
        X_raw, feat_cols,
        yj_mode=yj_mode,
        zscore=zscore,
        skewness_threshold=skewness_threshold,
    )
    return X_prep, params_df


def clamp_for_nonneg(X_prep, params_df, feat_cols):
    """
    Clamp preprocessed values so that inverse-transformed data respects non-negativity.
    For features with nonneg_label=1 and finite X_prep_min: X_prep = max(X_prep, X_prep_min).
    """
    X = np.asarray(X_prep) if not isinstance(X_prep, np.ndarray) else X_prep.copy()
    if X.ndim == 1:
        X = X.reshape(1, -1)
    feat2idx = {f: i for i, f in enumerate(feat_cols)}
    for _, r in params_df.iterrows():
        if r["nonneg_label"] != 1:
            continue
        xmin = r["X_prep_min"]
        if pd.isna(xmin) or not np.isfinite(xmin):
            continue
        idx = feat2idx.get(r["Feature"])
        if idx is not None:
            X[:, idx] = np.maximum(X[:, idx], xmin)
    return X


def inverse_transform(X_prep, params_df, feat_cols):
    """
    Inverse transform from preprocessed space to raw space.
    X_prep: (n_samples, n_features) in column order feat_cols.
    """
    X = np.asarray(X_prep, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    out = np.empty_like(X)
    feat2idx = {f: i for i, f in enumerate(feat_cols)}
    for _, r in params_df.iterrows():
        idx = feat2idx.get(r["Feature"])
        if idx is None:
            continue
        lab = r["preprocess_label"]
        x = X[:, idx]
        if lab == "A":
            out[:, idx] = x
        elif lab == "C":
            mu, sigma = r["mu"], r["sigma"]
            out[:, idx] = x * sigma + mu
        else:  # B
            mu, sigma, lam = r["mu"], r["sigma"], r["lambda"]
            x_yj = x * sigma + mu
            out[:, idx] = _yeo_johnson_inv_simple(x_yj, lam)
    return out


def transform(X_raw, params_df, feat_cols):
    """Forward transform from raw to preprocessed space."""
    X = np.asarray(X_raw, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    out = np.empty_like(X)
    feat2idx = {f: i for i, f in enumerate(feat_cols)}
    for _, r in params_df.iterrows():
        idx = feat2idx.get(r["Feature"])
        if idx is None:
            continue
        lab = r["preprocess_label"]
        x = X[:, idx]
        if lab == "A":
            out[:, idx] = x
        elif lab == "C":
            mu, sigma = r["mu"], r["sigma"]
            out[:, idx] = (x - mu) / sigma
        else:  # B
            lam, mu, sigma = r["lambda"], r["mu"], r["sigma"]
            x_yj = _yeo_johnson_forward(x, lam)
            out[:, idx] = (x_yj - mu) / sigma
    return out
