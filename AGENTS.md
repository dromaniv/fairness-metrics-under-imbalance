# Agent Coding Guide

This file provides guidance to AI Agents when working with code in this repository.

## What This Project Is

A Streamlit app for exploring confusion-matrix-based fairness metrics under varying class imbalance (IR) and group representation (GR) conditions. Five pages: **Synthetic Study** (enumerate/sample all possible confusion matrices), **Adult Case Study** (UCI Adult dataset with controlled resampling), **Stereotypical Bias Study** (SR correlation analysis), **Fairness Benchmark** (discrimination injection and detection power), and **Metric Registry** (browseable metric catalogue).

## Commands

```bash
pip install -r requirements.txt   # install dependencies
streamlit run app.py              # run the app (only entry point)
```

No test suite, linter, or CI pipeline is configured.

## Architecture

All modules live at the repo root (no package structure). Imports are direct (`from metric_registry import ...`).

**Metric system (registry pattern):**
- `metric_registry.py` — `MetricSpec` dataclass, `register_metric()`, `compute_metric(s)`, `safe_divide()`, `odds_ratio_to_q()`
- `builtin_metrics.py` — all built-in fairness/ratio/performance metrics + `register_builtin_metrics()`
- `custom_metrics.py` — user-extensible metrics, auto-registered on import
- `metric_bounds.py` — feasible-range bounds for FRN normalization, `_BOUNDS_REGISTRY`

**Data & analysis:**
- `synthetic_data.py` — confusion-matrix generation (exact enumeration, Monte Carlo, pickle I/O), `add_base_columns()`
- `synthetic_analysis.py` — helpers for probability curves, heatmaps, ratio filtering
- `stereotypical_study.py` — SR-sweep helpers, stratified Spearman correlation with Fisher Z-transformation
- `adult_case_study.py` — Adult dataset loading, resampling, classifier evaluation pipeline

**Benchmark:**
- `fairness_benchmark.py` — discrimination injection, detection power, ROC analysis

**Presentation:**
- `plots.py` — all matplotlib figure builders (no computation logic here); `plot_case_line(absolute=bool)` handles both signed and absolute views
- `app.py` — Streamlit UI only (no domain logic); shared helpers: `_build_synthetic_dataset()`, `_filter_degenerate()`, `_render_data_table_tab()`

## Key Conventions

**Metric compute functions** have signature `def metric(df: pd.DataFrame) -> np.ndarray`. Return `np.nan` for undefined values. Use `safe_divide(num, denom)` for any division — never use `/` directly on denominators that could be zero.

**Sign convention:** all fairness differences are computed as **j - i** (group j is the target/protected group).

**Random seed:** default is `2137`, passed explicitly as `seed=` or `random_state=`.

**Streamlit page pattern:** sidebar holds data-source controls and action buttons. Analysis slice controls (fixed GR, metric selector, SR variant) go **inline in tabs**, not in the sidebar. Session state keys: `synthetic_df`, `synthetic_total`, `adult_fairness_results`, `adult_performance_results`, `stereo_df`, `stereo_label`.

**Plot conventions:** all figure builders live in `plots.py`, always call `fig.tight_layout()` before returning. Use `COLOURS` palette, `STYLE_OVERRIDES` for named metrics, `ratio_label()` for axis tick labels.

**Smoothable metrics:** CQA and CYA accept a `smoothing=bool` parameter. Controlled via `_SMOOTHABLE_METRICS` set and `apply_smoothing_override()` in `app.py`.

**FRN variants:** raw metric key maps to FRN key via `_FRN_KEY_MAP` in `app.py`. Adding a new FRN metric requires a bounds function in `metric_bounds.py`.

## Adding a New Metric

1. Write `my_metric(df: pd.DataFrame) -> np.ndarray` in `builtin_metrics.py` (or `custom_metrics.py`).
2. Append a `MetricSpec(...)` to the `specs` list inside `register_builtin_metrics()`.
3. It auto-appears in all dropdowns, plots, and analyses.

## SR Variants

Three stereotypical ratio columns: `stereotypical_ratio` (SR_p, positive predictions), `stereotypical_ratio_negative` (SR_n, negative predictions), `stereotypical_ratio_combined` (SR_c, geometric mean). Use `compute_sr_sensitivity_stratified()` (Fisher Z over (IR,GR) strata) — pooled correlation is confounded.

## Data

- `data/adult.data` — UCI Adult dataset (tracked in git)
- `data/Set(08,56).bin` — precomputed confusion matrices for n=56 (~4.4 GB, gitignored)