# Agent Coding Guide

## Tech Stack

Python 3.11 · Streamlit · pandas · numpy · matplotlib · scikit-learn · imbalanced-learn · scipy

---

## Module Map

| File | Responsibility |
|---|---|
| `metric_registry.py` | `MetricSpec` dataclass, `register_metric`, `list_metrics`, `compute_metric(s)` |
| `builtin_metrics.py` | All built-in metric compute functions + `register_builtin_metrics()` |
| `custom_metrics.py` | User-extensible metrics (optional import) |
| `metric_bounds.py` | Feasible-range bounds for FRN normalization |
| `synthetic_data.py` | Confusion-matrix generation (exact / Monte Carlo / pickle) + `add_base_columns` |
| `synthetic_analysis.py` | `probability_of_perfect_fairness`, `probability_of_nan`, `ensure_metric_columns` |
| `stereotypical_study.py` | SR-sweep helpers: `metric_means_by_sr_multi_ir`, `compute_sr_sensitivity`, `compute_sr_sensitivity_stratified` |
| `adult_case_study.py` | Adult dataset loading, sampler, experiment runner, `collect_adult_confusion_matrices` |
| `plots.py` | All matplotlib figure builders (no computation logic) |
| `app.py` | Streamlit UI only — imports from all above, no domain logic |

---

## Adding a New Metric (one-file edit)

1. Write a pure compute function `my_metric(df: pd.DataFrame) -> np.ndarray` in `builtin_metrics.py`.
2. Append a `MetricSpec(...)` to the `specs` list inside `register_builtin_metrics()`.
3. Done — it auto-appears in all dropdowns, plots, and analyses.

For a **custom** metric, do the same in `custom_metrics.py` (registered at module import time).

For an **FRN-wrapped** variant:
1. Add bounds function to `metric_bounds.py` and register it in `_BOUNDS_REGISTRY`.
2. Create the FRN function with `_frn_wrap(raw_fn, bounds_fn)` in `builtin_metrics.py`.
3. Add a `MetricSpec` with `category="fairness_frn"`.
4. Add the `raw_key → frn_key` pair to `_FRN_KEY_MAP` in `app.py`.

---

## Key Conventions

### Sign convention
All fairness difference metrics follow **j − i**. Group j is the target/protected group.

### Undefined values
Use `np.nan`. Never silently coerce. Use `safe_divide(num, denom)` from `metric_registry` for any rate computation.

### Random seeds
Pass `seed`/`random_state` explicitly everywhere. Default is `2137`.

### Typing
Typed function signatures throughout. Use `dataclass(frozen=True)` for config objects.

### Comments
Minimal and high-signal only. No decorative separators or boilerplate.

---

## Streamlit Page Pattern

Every page function follows this exact structure:

```python
def render_my_page() -> None:
    st.header("...")
    st.write("...")

    _prog_bar = st.empty()   # progress goes in main area, not sidebar
    _prog_cap = st.empty()

    with st.sidebar:
        st.subheader("...")
        # data-source controls + action button
        # store result in st.session_state["my_df"]

    df = st.session_state.get("my_df")
    if df is None:
        st.info("Build a dataset from the sidebar.")
        return

    # summary metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(...)

    tabs = st.tabs([...])

    with tabs[0]:
        col1, col2, col3 = st.columns([1, 1, 1])
        # inline controls per tab
        # plot + download button
```

**Never** put analysis slice controls (fixed GR, metric selector, SR variant, etc.) in the sidebar. Those go **inline** at the top of each tab using `st.columns`.

---

## Session State Keys

| Key | Contents |
|---|---|
| `synthetic_df` | DataFrame built on Synthetic Study page |
| `synthetic_total` | `n` used for `synthetic_df` |
| `adult_fairness_results` | Adult case study fairness results DataFrame |
| `adult_performance_results` | Adult case study performance results DataFrame |
| `stereo_df` | Stereotypical study working DataFrame (synthetic or adult) |
| `stereo_label` | Display label for `stereo_df` |

---

## Plot Conventions

- All figure builders live in `plots.py`. No matplotlib in `app.py`.
- Line colour palette: `COLOURS` list. Named-metric overrides: `STYLE_OVERRIDES`.
- Tick labels for ratios: use `ratio_label(value)` (returns fraction string where exact).
- Always call `fig.tight_layout()` before returning.
- PIL decompression bomb disabled at app startup (`_PIL_Image.MAX_IMAGE_PIXELS = None`).
- `figure_png_bytes(fig)` auto-caps DPI so downloads stay under 100 MP.

---

## Adding a New Study Page

1. Write analysis helpers in a new module (e.g. `my_study.py`).
2. Write figure builders in `plots.py`.
3. Write `render_my_page()` in `app.py`, following the page pattern above.
4. Import helpers at the top of `app.py`.
5. Add `"My study"` to the `st.radio("Workflow", ...)` options at the bottom of `app.py` and wire the `elif` branch.

---

## Smoothable Metrics (CQA / CYA)

Metrics in `_SMOOTHABLE_METRICS` in `app.py` accept a `smoothing` bool.
- `apply_smoothing_override(df, key, smoothing)` re-computes the column with the toggle value.
- `smoothing_toggle(key, widget_key)` renders the checkbox (returns `True` for non-smoothable metrics so call sites are uniform).

To add a new smoothable metric:
1. Implement `my_metric(df, smoothing=True)` in `custom_metrics.py`.
2. Add `"my_metric"` to `_SMOOTHABLE_METRICS` in `app.py`.
3. Handle it in `apply_smoothing_override`.

---

## Stereotypical Study — SR Variants

Three SR columns are computed in `add_base_columns` (via `builtin_metrics.py`) and available in every DataFrame built by that function:

| Column | Symbol | Definition |
|---|---|---|
| `stereotypical_ratio` | SR_p | j's share of **positive** predictions: (j_tp+j_fp) / total predicted positives |
| `stereotypical_ratio_negative` | SR_n | j's share of **negative** predictions: (j_tn+j_fn) / total predicted negatives |
| `stereotypical_ratio_combined` | SR_c | √(SR_p × SR_n) — geometric mean; = GR_j when both decisions are proportional |

All three are registered in `builtin_metrics.py` under `category="ratio"` so they can be computed on demand via `ensure_metric_columns`.

`SR_LABELS` and `SR_COLUMNS` dicts in `stereotypical_study.py` map column names ↔ display labels.

`resolve_ratio_column` in `synthetic_analysis.py` maps ratio type strings:
- `"sr"` → `"stereotypical_ratio"`
- `"sr_n"` → `"stereotypical_ratio_negative"`
- `"sr_c"` → `"stereotypical_ratio_combined"`

`X_LABELS` in `plots.py` covers all three types for axis labelling.

The **_SR_VARIANTS** dict inside `render_stereotypical_page` in `app.py` is the single source of truth for the UI picker:

```python
_SR_VARIANTS: dict[str, tuple[str, str]] = {
    "SR_p": ("stereotypical_ratio", "sr"),
    "SR_n": ("stereotypical_ratio_negative", "sr_n"),
    "SR_c": ("stereotypical_ratio_combined", "sr_c"),
}
```

Each tab in the Stereotypical Bias Study page has an inline `SR variant` radio that reads from `_SR_VARIANTS` to determine which column and ratio type to pass downstream.

---

## Stereotypical Study — SR Sensitivity Analysis

Two correlation methods are available in `stereotypical_study.py`:

### `compute_sr_sensitivity` (pooled)
Computes Spearman ρ over all rows (optional IR/GR filter). IR and GR are confounders — use only for exploratory analysis or when explicitly filtered to a fixed stratum.

### `compute_sr_sensitivity_stratified` (recommended, default in UI)
Groups by (IR, GR) strata, computes within-stratum Spearman ρ, then combines via **Fisher Z-transformation** (weight = n_stratum − 3). This controls for class imbalance and group size as confounders, preventing Simpson's paradox from distorting the ranking.

The UI defaults to stratified and runs it for **all three SR variants simultaneously**, displaying:
- Three side-by-side bar charts (one per variant).
- A combined |ρ| table for all metrics × all SR variants.

"Averaging LIES": pooled correlations are driven by the marginal joint distribution of IR, GR, and SR — not the within-condition SR effect. Stratified is the honest answer.

---

## Stereotypical Study — SR Resistance Rankings

From n=12 synthetic data (stratified Spearman, exact enumeration):

### vs SR_p (positive prediction share)
| Metric | |ρ| | verdict |
|---|---|---|
| Equalized Odds Difference | ~0.000 | most resistant |
| Accuracy Equality Difference | ~0.001 | most resistant |
| Negative Predictive Parity Difference | ~0.002 | resistant |
| Positive Predictive Parity Difference | ~0.002 | resistant |
| Equal Opportunity Difference | ~0.69 | sensitive |
| Predictive Equality Difference | ~0.69 | sensitive |
| Statistical Parity Difference | ~0.95 | very sensitive |
| Predictive Stereotype Score | ~1.00 | maximally sensitive (PSS ≡ SR_p − GR_j) |

### vs SR_n (negative prediction share)
| Metric | |ρ| | verdict |
|---|---|---|
| Equalized Odds Difference | ~0.000 | most resistant |
| Accuracy Equality Difference | ~0.001 | most resistant |
| Positive Predictive Parity Difference | ~0.002 | resistant |
| Negative Predictive Parity Difference | ~0.002 | resistant |
| Equal Opportunity Difference | ~0.69 | sensitive |
| Predictive Equality Difference | ~0.69 | sensitive |
| Statistical Parity Difference | ~0.96 | very sensitive |

### vs SR_c (combined, geometric mean)
| Metric | |ρ| | verdict |
|---|---|---|
| Negative Predictive Parity Difference | ~0.001 | most resistant |
| Positive Predictive Parity Difference | ~0.001 | most resistant |
| Accuracy Equality Difference | ~0.001 | most resistant |
| Equal Opportunity Difference | ~0.002 | resistant |
| Predictive Equality Difference | ~0.002 | resistant |
| Statistical Parity Difference | ~0.004 | near-resistant |
| Predictive Stereotype Score | ~0.22 | moderately sensitive |

**Key insight**: SR_c reveals that most difference metrics become near-resistant when both positive and negative prediction allocations are considered jointly. Equalized Odds Difference and Accuracy Equality Difference are the most consistently SR-resistant metrics across all three SR variants. PSS is maximally sensitive to SR_p by construction (PSS ≡ SR_p − GR_j).
