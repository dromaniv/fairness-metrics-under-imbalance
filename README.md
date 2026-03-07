# Fairness Measures Explorer

A modular Streamlit recreation of the provided fairness-measure repository.

The app covers two workflows:

1. **Synthetic study**
   - generate the space of 8-cell confusion matrices exactly for moderate `n`
   - or sample uniformly from that space with Monte Carlo draws
   - compute any registered fairness metric
   - reproduce the paper-style histogram grids, perfect-fairness curves, undefined-value curves, and fairness-vs-performance heatmaps

2. **Adult case study**
   - load `adult.data`
   - resample subsets with controlled group ratio (GR) and imbalance ratio (IR)
   - run the same family of classifiers as in the original repository
   - reproduce the case-study line plots, absolute-value plots, NaN plots, grouped bars, and aggregated tables

## Project layout

- `app.py` - Streamlit UI
- `metric_registry.py` - registry and helpers for metrics
- `builtin_metrics.py` - built-in fairness, ratio, and performance metrics
- `custom_metrics.py` - edit this to add your own metrics
- `synthetic_data.py` - exact and sampled confusion-matrix generation
- `synthetic_analysis.py` - perfect-fairness / NaN / heatmap helpers
- `adult_case_study.py` - Adult dataset loading, preprocessing, resampling, and evaluation
- `plots.py` - Matplotlib plot builders used by the app
- `requirements.txt` - Python dependencies

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Adding a new metric

1. Open `custom_metrics.py`.
2. Copy the example metric skeleton.
3. Change the formula and metadata.
4. Save the file and reload Streamlit.

Custom metrics automatically appear in the Streamlit dropdowns if they are registered with:

```python
register_metric(MetricSpec(...))
```

## Notes on reproducibility

- The metric formulas follow the **repository implementation** sign convention: built-in fairness differences are computed as `j - i`.
- In the synthetic workflow, exact enumeration is practical only for moderate totals. The paper-scale case `n=56` contains **553,270,671** confusion matrices, so the app defaults to smaller exact runs or Monte Carlo sampling.
- The Adult case-study page includes a compatibility toggle for the ordered-education encoding. When enabled, it reproduces the broadcast behavior present in the provided script; when disabled, it uses the corrected per-row ordered encoding.
- The Adult dataset is **not bundled**. Either upload `adult.data` through the app or point the app to a local path such as `data/adult.data`.

## Suggested workflow

### Synthetic study

1. Start with exact enumeration for `n=24`.
2. Select one or more fairness metrics.
3. Inspect histogram grids for selected GR/IR panels.
4. Inspect perfect-fairness and undefined-value curves.
5. Compare fairness against `Accuracy` or `G-mean` with the heatmap.
6. Edit `custom_metrics.py` and repeat.

### Adult case study

1. Upload `adult.data`.
2. Keep the paper sweep ratios, or reduce them for quicker runs.
3. Choose a subset of classifiers if you want faster feedback.
4. Inspect mean/std or mean/SEM fairness bands.
5. Export raw CSV results for later analysis.
