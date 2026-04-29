# ECAF Figure Notes

## Files

| File | Description |
|---|---|
| `plot_ecaf_figure.py` | Standalone plotting script (run from repo root or this directory) |
| `ecaf_ridgeline_metrics.png` | Main figure, 300 dpi PNG |
| `ecaf_ridgeline_metrics.pdf` | Main figure, vector PDF (include in LaTeX) |
| `ecaf_ridgeline_metrics.svg` | Main figure, vector SVG |
| `ecaf_ridgeline_metrics_jr.png` | Alternative ridgeline (joyplot) style, 300 dpi PNG |
| `ecaf_ridgeline_metrics_jr.pdf` | Alternative ridgeline style, vector PDF |

---

## Data source

**Fully enumerated confusion matrices** generated on-the-fly using `_enumerate_matrices(n_total=30, ir, gr)`.

For each (IR, GR) condition, the function enumerates every valid 8-tuple `(i_tp, i_fp, i_tn, i_fn, j_tp, j_fp, j_tn, j_fn)` whose marginals satisfy the target imbalance ratio and group ratio exactly.

Row counts:
- IR=0.5, GR=0.5 → 47,328 matrices
- IR=0.1, GR=0.5 → 4,184 matrices
- IR=0.5, GR=0.1 → 4,184 matrices

---

## Experimental conditions

Three conditions per panel, all at n=30:

| Condition | IR | GR | Colour | Line |
|---|---|---|---|---|
| Balanced reference | 0.5 | 0.5 | Blue (#1A5276) | Solid |
| IR skew | 0.1 | 0.5 | Red (#9B2335) | Solid |
| GR skew | 0.5 | 0.1 | Green (#2E7D32) | Dashed |

IR was chosen as the primary skew axis because it produces the most visually unambiguous contrast: at IR=0.1, EOD and Equalized Odds exhibit both high NaN rates and dramatic distribution changes. The GR=0.1 condition exposes the secondary story — SPD disperses widely under group imbalance while Phi concentrates near 0 (stable).

---

## Metrics selected and why

### Standard metrics — row (a)

| Metric | Key | Behaviour |
|---|---|---|
| Statistical Parity Difference (SPD) | `statistical_parity_diff` | IR-stable (blue ≈ red), but GR-sensitive: green curve disperses widely under GR=0.1, showing unreliable readings. Included as the "partially good" classical baseline. |
| Equal Opportunity Difference (EOD) | `equal_opportunity_diff` | IR-degrades severely: TPR denominator collapses at IR=0.1, yielding ~40% undefined values and a discrete multi-spike distribution. |
| Equalized Odds Difference | `equalized_odds_diff` | Mixed degradation: max(|TPR_gap|, |FPR_gap|) inherits EOD's NaN and discretisation (TPR path) while FPR path remains near-continuous — produces a visually distinct mixed pattern. |

### Proposed metrics — row (b)

| Metric | Key | Behaviour |
|---|---|---|
| Fairness Phi (Φ) | `fairness_phi` | IR-stable (blue ≈ red) AND GR-stable: green curve concentrates tightly near 0 under GR=0.1 rather than dispersing — the key "hidden talent" distinguishing Phi from SPD. |
| Marginal Y Association | `marginal_y_association` | Confirms Phi's stability result holds for the Yule-Y family. |
| Conditional Y Association (CYA) | `conditional_y_association` | Zero NaN at all IR levels; Haldane–Anscombe smoothing preserves stability. |

---

## Design choices

- **No fill under curves** — overlapping transparent fills add visual clutter without information. Lines alone suffice with colour + linestyle encoding.
- **No annotations** — kept clean for author-supplied LaTeX annotations in the paper.
- **`sharex='col'`** — suppresses duplicate x-tick labels; only bottom row shows ticks.
- **No suptitle** — title added in LaTeX.
- **STIX font** — matches Computer Modern in LaTeX documents.
- **y-ceiling = 2.2× second-tallest peak** per panel — prevents a single spike from dominating while preserving shape detail.

---

## Alternatives considered and rejected

### SR as a 4th condition line
Tested filtering balanced matrices to SR_p=0.1 (biased classifier). Produces a single sharp spike far left on all difference metrics. Visually compelling but adds a separate conceptual story (classifier bias vs distributional stability) that complicates the figure's message. Removed.

### SR as a 4th column
Added stereotypical_ratio_combined as a separate panel. Since it is the same metric in both rows, it adds no comparative information. Removed.

### Filled density curves
The semi-transparent fills make overlapping curves hard to read and add no information beyond the line itself. Removed.

### FWHM bracket annotations
Added ↔ brackets over the GR=0.1 curves to label "dispersed" (SPD) vs "stable" (Phi). Removed at author request — annotations will be added in LaTeX.

### Shared y-scale per row (`sharey='row'`)
EOD's tall discrete spikes dominate the row scale, making SPD look flat. Per-panel scaling is better.

### Joyplot / stacked ridgelines within each panel
See `ecaf_ridgeline_metrics_jr.{png,pdf}` for this alternative. Each condition gets its own horizontal strip, making shape comparison cleaner but reducing direct overlay comparison. Author to decide.
