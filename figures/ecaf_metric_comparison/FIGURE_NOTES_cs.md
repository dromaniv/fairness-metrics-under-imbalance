# Figure Interpretation — Metric Distributions Under Imbalance (Case Study, Adult Dataset)

Each panel shows the distribution of a fairness metric computed from real classifier outputs on the UCI Adult dataset. Distributions are built from 6 classifiers × 50 stratified holdout splits (n=1,100 per split, 33% held out) under four conditions: balanced (IR=0.5, GR=0.5), class imbalance alone (IR=0.1, GR=0.5), group representation imbalance alone (IR=0.5, GR=0.1), and simultaneous extreme imbalance (IR=0.05, GR=0.05). Protected attribute is sex (Female vs Male); positive class is income >50K. Each density is a KDE over 300 per-condition values (6 × 50).

---

## Row (a) — Standard metrics

**Statistical Parity Difference (SPD)** has its tallest, narrowest peak under IR skew (orange, density ~12) — confirming that SPD is essentially blind to class prevalence. Under GR skew (teal) and double skew (pink), the distribution widens visibly: the same fair classifier now produces a 2–3× broader spread of SPD values purely because of how few minority-group members each split contains. SPD is reading group size, not unfairness.

**Equal Opportunity Difference (EOD)** is narrowly concentrated at 0 in the balanced condition (peak ~22) but loses 60–70% of its peak height under any skew. The double-skew distribution (pink) is broad, asymmetric, and shifted off zero, making it nearly uninterpretable. Even where EOD is defined on real classifiers, the noise floor inflates rapidly with imbalance.

**Equalized Odds Difference** is shifted entirely to the positive half-line (it is a max of absolute gaps). The balanced peak is ~11; the double-skew distribution (pink) flattens to ~3 and spreads to 0.6, again driven by the TPR component going noisy under low prevalence in the protected group.

---

## Row (b) — Proposed metrics

**Fairness Phi (Φ)** has the most stable distribution of any metric in this figure. All four density curves nearly overlap, peaking near 0 at density ~7, with no meaningful shape change under IR skew, GR skew, or both combined. **This is the headline result: Φ's measurement of (un)fairness is essentially invariant to the imbalance condition on real Adult data.** No undefined values arise.

**Marginal Y** retains its sharp balanced peak (~14) but loses peak height under skew (orange/teal ~5, pink ~2.5) and the double-skew density picks up a left tail extending to -0.5. Y is more imbalance-stable than EOD but less stable than Φ in this case study; both belong to the Yule family but condition differently.

**Conditional Y** is bounded to [0, 1] (it is a conditional |Y|). The balanced and IR-skew densities are tight near 0.1; the double-skew density (pink) shifts substantially to higher values (0.2–0.4) and broadens. Conditional Y is well-defined throughout but is *not* shift-invariant under joint extreme imbalance — it picks up an apparent association signal from the small denominators in the conditional cells.

---

## Key findings

1. **Fairness Φ is the clear winner on real data.** Across all four imbalance conditions, Φ's density is visually a single curve. Every other metric — including SPD, EOD, Equalized Odds, Marginal Y, and Conditional Y — shows visible peak-height loss, broadening, or location shift under skew.
2. **Φ beats SPD decisively.** SPD widens by ~2× under GR skew; Φ does not move. The "small-group-inflation" artefact that drives SPD is absent in Φ.
3. **Φ beats EOD/Equalized Odds.** EOD's peak collapses by 60–70% under any skew on real data, with location shift under double skew; Φ stays put. Φ has no undefined values; EOD does.
4. **Φ beats Marginal Y on imbalance stability** despite both being Yule-family metrics: Marginal Y loses peak height and develops a left tail under double skew, while Φ does not.
5. **Conditional Y trades stability for boundedness.** Its [0, 1] support is convenient, but it shifts noticeably under double skew — so it should be reported alongside Φ rather than instead of it.

---

## Comparison to the synthetic figure

The case-study distributions are tighter than the synthetic-enumeration distributions in every panel because real classifiers occupy a small, performance-concentrated region of confusion-matrix space rather than the full enumerable set. The bimodality seen for Φ and Y under GR skew at n=30 in the synthetic figure is absent here (n=1,100 dissolves the discretisation artefact). The headline ranking of metrics is preserved and sharpened: **Φ is the most imbalance-stable, EOD/Equalized Odds the least.**

---

## Bottom line

On real Adult-dataset classifiers, **Fairness Φ produces a single distributional fingerprint regardless of class or group imbalance, while every standard metric (SPD, EOD, Equalized Odds) visibly degrades under skew.** Marginal Y partially shares Φ's stability; Conditional Y is bounded but drifts under double skew. For fairness audits of imbalanced real data, Φ should be the primary metric, with the standard metrics retained only as supplements where defined.
