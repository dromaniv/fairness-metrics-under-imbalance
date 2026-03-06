# Copilot Instructions

## Repo layout
- `experiments/` — main Python package (metrics, generator, VST wrapper, Streamlit app)
- `experiments/app.py` — Streamlit dashboard; run with `streamlit run experiments/app.py`
- `experiments/metrics.py` — all fairness metric implementations
- `experiments/generate.py` — CM-pair generation (`generate_cm_pairs`) and `.bin` loading (`load_cm_pairs`)
- `experiments/wrapper.py` — Jeffreys-VST wrapper (`apply_jeffreys_df`, `apply_wrapper_df`, `wrap`)
- `experiments/plots.py` — Plotly figure factories (`plot_eod_vs_ir`, etc.)
- `sets_creation.py` — combinatorial simplex generator (do not modify)
- `utils.py` — vectorised IR/GR helpers (do not modify)
- `tests/test_sanity.py` — pytest suite; import from `experiments.*`

## Metrics
Keys in `METRIC_LABELS` (no "Difference" suffix in labels):
`accd`, `spd`, `eod`, `fprd`, `eqod`, `ppvd`, `npvd`, `caf_q_rms_ha`, `caf_q_rms_ha_abs`, `caf_eopp_q_ha`

- `eqod` = Equalized Odds = `max(|EOD|, |FPRD|)` ∈ [0, 1], unsigned.
- CAF metrics accept a `kappa` float (Haldane–Anscombe correction, default 0.5).
- Classic metrics accept `policy` ("nan" | "smooth_alpha") and `alpha`.
- All return `pd.Series` aligned to the input DataFrame index.

## Data model
Each row is one confusion-matrix pair: `i_tp i_fp i_tn i_fn j_tp j_fp j_tn j_fn` (int16) + `ir gr` (float64).
- `i` = protected group, `j` = unprotected group
- `ir` = (all positives) / n, `gr` = (all j) / n
- Row sum of the 8 CM columns = n (the generation budget)

`.bin` files are `pickle` dumps of `(m, 8)` int8 arrays from `sets_creation.py`. Load with `load_cm_pairs(path)`.
The app accepts a file-system path (no size limit); do not use `st.file_uploader` for `.bin` files.

## Wrapper (two stages, separate toggles)
- **Stage 1 — Jeffreys smoothing**: `apply_jeffreys_df(df, metric)` → smoothed difference r̃ⱼ − r̃ᵢ, no arcsin.
- **Stage 2 — VST**: `apply_wrapper_df(df, metric)` → full Jeffreys + arcsin transform ∈ [−1, 1].
- `WRAPPER_METRICS = ("eod", "fprd", "spd", "ppvd", "npvd", "accd")`. Not applicable to CAF.

## Constraints
- Max generation n = 32.
- Histogram bins and slice tolerance are auto-scaled from n via `_auto_hist_params(n)`.
- Perfect fairness = |value| ≤ 0 (exact zero), reported as a ratio in the header.
- Do not import from `fairness` or `fairness_wrapper` (old names); use `experiments`.
- Do not add new root-level Python scripts; put code in `experiments/` or `tests/`.
