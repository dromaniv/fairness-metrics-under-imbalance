"""Discrimination injection and detection-power benchmarking for fairness metrics.

Generates synthetic confusion matrices with a controlled true discrimination level δ
(TPR gap, FPR gap, or both), then measures each metric's ability to detect it via
Spearman ρ, detection power, false alarm rate, and ROC AUC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats

from metric_registry import COUNT_COLUMNS, compute_metrics
from synthetic_data import add_base_columns


def _group_sizes(n: int, ir: float, gr: float) -> tuple[int, int, int, int]:
    """Return (j_total, i_total, j_pos, i_pos).

    ir = overall positive rate (positives / n).
    gr = j-group fraction (j_total / n).
    Both groups share the same base positive rate (ir) before discrimination is injected.
    """
    j_total = max(1, round(gr * n))
    i_total = max(1, n - j_total)
    total_pos = max(0, min(n, round(ir * n)))
    j_pos = max(0, min(j_total, round(total_pos * gr)))
    i_pos = max(0, min(i_total, total_pos - j_pos))
    return j_total, i_total, j_pos, i_pos


def generate_discriminated_matrices(
    n: int,
    ir: float,
    gr: float,
    delta: float,
    disc_type: str = "tpr_gap",
    n_samples: int = 500,
    seed: int = 2137,
) -> pd.DataFrame:
    """Generate confusion matrices with controlled discrimination level δ.

    disc_type:
      "tpr_gap" — TPR_j − TPR_i = δ, FPR equalized between groups.
      "fpr_gap" — FPR_j − FPR_i = δ, TPR equalized between groups.
      "both"    — both TPR_j − TPR_i = δ and FPR_j − FPR_i = δ.

    Returns a DataFrame with COUNT_COLUMNS + "true_delta" column.
    """
    rng = np.random.default_rng(seed)
    j_total, i_total, j_pos, i_pos = _group_sizes(n, ir, gr)
    j_neg = j_total - j_pos
    i_neg = i_total - i_pos

    def _sample_gap(n_s: int, gap: float) -> tuple[np.ndarray, np.ndarray]:
        """Sample (rate_j, rate_i) such that rate_j − rate_i = gap."""
        lo = max(0.0, gap)
        hi = min(1.0, 1.0 + gap)
        if lo >= hi - 1e-9:
            rate_j = np.full(n_s, np.clip(lo, 0.0, 1.0))
        else:
            rate_j = rng.uniform(lo, hi, n_s)
        rate_i = np.clip(rate_j - gap, 0.0, 1.0)
        return rate_j, rate_i

    if disc_type == "tpr_gap":
        tpr_j, tpr_i = _sample_gap(n_samples, delta)
        base_fpr = rng.uniform(0.0, 1.0, n_samples)
        fpr_j, fpr_i = base_fpr.copy(), base_fpr.copy()
    elif disc_type == "fpr_gap":
        fpr_j, fpr_i = _sample_gap(n_samples, delta)
        base_tpr = rng.uniform(0.0, 1.0, n_samples)
        tpr_j, tpr_i = base_tpr.copy(), base_tpr.copy()
    else:  # "both"
        tpr_j, tpr_i = _sample_gap(n_samples, delta)
        fpr_j, fpr_i = _sample_gap(n_samples, delta)

    j_tp = np.clip(np.round(tpr_j * j_pos).astype(int), 0, j_pos)
    j_fn = j_pos - j_tp
    i_tp = np.clip(np.round(tpr_i * i_pos).astype(int), 0, i_pos)
    i_fn = i_pos - i_tp
    j_fp = np.clip(np.round(fpr_j * j_neg).astype(int), 0, j_neg)
    j_tn = j_neg - j_fp
    i_fp = np.clip(np.round(fpr_i * i_neg).astype(int), 0, i_neg)
    i_tn = i_neg - i_fp

    data = np.stack([i_tp, i_fp, i_tn, i_fn, j_tp, j_fp, j_tn, j_fn], axis=1)
    df = pd.DataFrame(data, columns=COUNT_COLUMNS)
    df["true_delta"] = float(delta)
    return df


def sweep_discrimination(
    n: int,
    ir: float,
    gr: float,
    delta_values: list[float],
    disc_type: str = "tpr_gap",
    n_per_delta: int = 300,
    metric_keys: list[str] | None = None,
    seed: int = 2137,
) -> pd.DataFrame:
    """Sweep δ over *delta_values*, compute base columns and fairness metrics."""
    frames = []
    for k, delta in enumerate(delta_values):
        chunk = generate_discriminated_matrices(n, ir, gr, delta, disc_type, n_per_delta, seed + k * 997)
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    df = add_base_columns(raw)
    if metric_keys:
        mdf = compute_metrics(df, metric_keys)
        for col in mdf.columns:
            df[col] = mdf[col].to_numpy(np.float64)
    return df


def benchmark_metrics(
    df: pd.DataFrame,
    metric_keys: list[str],
    threshold: float = 0.05,
    null_eps: float = 0.01,
) -> pd.DataFrame:
    """Compute Spearman ρ, detection power, and false alarm rate for each metric.

    Detection criterion: |metric value| > threshold.
    Null definition:     |true_delta| <= null_eps.
    """
    if "true_delta" not in df.columns:
        raise ValueError("DataFrame must contain a 'true_delta' column.")
    delta = df["true_delta"].to_numpy(np.float64)
    pos_mask = np.abs(delta) > null_eps
    null_mask = ~pos_mask

    rows = []
    for key in metric_keys:
        if key not in df.columns:
            continue
        vals = df[key].to_numpy(np.float64)
        abs_vals = np.abs(vals)
        valid = np.isfinite(vals)

        spearman_r = np.nan
        if valid.sum() >= 4:
            try:
                r, _ = scipy.stats.spearmanr(delta[valid], vals[valid])
                spearman_r = float(r)
            except Exception:
                pass

        pos_valid = pos_mask & valid
        null_valid = null_mask & valid
        detection_power = float((abs_vals[pos_valid] > threshold).mean()) if pos_valid.sum() > 0 else np.nan
        false_alarm = float((abs_vals[null_valid] > threshold).mean()) if null_valid.sum() > 0 else np.nan

        rows.append({
            "metric": key,
            "spearman_r": spearman_r,
            "detection_power": detection_power,
            "false_alarm_rate": false_alarm,
            "n_positive": int(pos_valid.sum()),
            "n_null": int(null_valid.sum()),
        })
    return pd.DataFrame(rows)


def compute_roc_data(
    df: pd.DataFrame,
    metric_keys: list[str],
    null_eps: float = 0.01,
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """Compute ROC (fpr_array, tpr_array, auc) for each metric.

    Positive label: |true_delta| > null_eps.
    Score:          |metric value| (NaN rows excluded).
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    delta = df["true_delta"].to_numpy(np.float64)
    labels = (np.abs(delta) > null_eps).astype(int)

    result: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for key in metric_keys:
        if key not in df.columns:
            continue
        scores = np.abs(df[key].to_numpy(np.float64))
        valid = np.isfinite(scores)
        lv, sv = labels[valid], scores[valid]
        if lv.sum() < 2 or (1 - lv).sum() < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(lv, sv)
            auc_val = float(roc_auc_score(lv, sv))
            result[key] = (fpr, tpr, auc_val)
        except Exception:
            pass
    return result


def sweep_detection_by_ir(
    n: int,
    gr: float,
    ir_values: list[float],
    delta: float,
    disc_type: str = "tpr_gap",
    metric_keys: list[str] | None = None,
    n_per_delta: int = 200,
    threshold: float = 0.05,
    null_eps: float = 0.01,
    seed: int = 2137,
) -> pd.DataFrame:
    """For each IR value sweep δ ∈ {−|δ|, 0, +|δ|} and compute benchmark metrics.

    Returns DataFrame: ir, metric, detection_power, false_alarm_rate, spearman_r, auc.
    """
    from sklearn.metrics import roc_auc_score

    mkeys = metric_keys or []
    abs_delta = abs(delta)
    delta_sweep = sorted({-abs_delta, 0.0, abs_delta})

    rows = []
    for k, ir in enumerate(ir_values):
        df = sweep_discrimination(n, ir, gr, delta_sweep, disc_type, n_per_delta, mkeys, seed + k * 10007)
        if df.empty:
            continue
        bench = benchmark_metrics(df, mkeys, threshold=threshold, null_eps=null_eps)

        lbl_delta = df["true_delta"].to_numpy(np.float64)
        labels = (np.abs(lbl_delta) > null_eps).astype(int)

        for _, row in bench.iterrows():
            auc_val = np.nan
            key = row["metric"]
            if key in df.columns:
                scores = np.abs(df[key].to_numpy(np.float64))
                valid = np.isfinite(scores)
                lv, sv = labels[valid], scores[valid]
                if lv.sum() >= 2 and (1 - lv).sum() >= 2:
                    try:
                        auc_val = float(roc_auc_score(lv, sv))
                    except Exception:
                        pass
            rows.append({
                "ir": ir,
                "metric": key,
                "detection_power": row["detection_power"],
                "false_alarm_rate": row["false_alarm_rate"],
                "spearman_r": row["spearman_r"],
                "auc": auc_val,
            })
    return pd.DataFrame(rows)

