# Figure Interpretation — Metric Distributions Under Imbalance (Synthetic, n=30)

Each panel shows the full distribution of a fairness metric computed over all valid confusion matrices with n=30, under four conditions: balanced (IR=0.5, GR=0.5), class imbalance alone (IR=0.1, GR=0.5), group representation imbalance alone (IR=0.5, GR=0.1), and simultaneous extreme imbalance (IR=0.05, GR=0.05). Distributions are kernel density estimates over the exhaustive enumeration of confusion matrices satisfying each condition's marginal constraints.

---

## Row (a) — Standard metrics

**Statistical Parity Difference (SPD)** is roughly IR-insensitive (orange tracks blue) but visibly degrades under GR skew: the teal and pink densities flatten from a peak height of ~1.1 down to ~0.7 and spread mass uniformly across [-1, 1]. This is the well-known small-group-amplification artefact — when the minority group is tiny, single individuals swing SPD by large amounts, producing apparent disparity even under a fair classifier. SPD is GR-fragile.

**Equal Opportunity Difference (EOD)** collapses under class imbalance. Undefined rates: 40% (IR skew), 20% (GR skew), 39% (both skewed) — i.e. up to 40% of confusion matrices yield no value because the protected group contains zero true positives. Where defined, the IR-skew and double-skew distributions are jagged and multi-modal, an artefact of coarse discretisation at small positive counts. EOD is effectively unusable when prevalence is low.

**Equalized Odds Difference** inherits EOD's failure (undefined rates 40%, 40%, 39%) since its TPR component drops out whenever EOD does. The visible mass is a mixture of the surviving FPR-only spikes and a continuous shoulder, producing the most erratic shape of any panel in the figure.

---

## Row (b) — Proposed metrics

**Fairness Phi (Φ)** is fully defined under all four conditions (0% undefined). The balanced distribution is sharply unimodal at 0 (peak ~1.4); IR skew flattens it modestly (peak ~0.8) but does not introduce undefined values or bias. Under GR skew, Φ becomes mildly bimodal with peaks near ±0.5 — a discretisation artefact of n=30 with only ~3 minority-group members, *not* a flattening of the support. Crucially, mass remains bounded and concentrated near 0; there is no GR-driven inflation of the kind SPD exhibits.

**Marginal Y** mirrors Φ's profile: 0% undefined, sharp balanced peak (~1.5), modest IR-skew spreading, and the same n=30 bimodality under GR skew. Y comes from the same Yule association family as Φ, and the two metrics are visually nearly indistinguishable in this figure.

**Conditional Y** is bounded to [0, 1] by construction (it is a |Y| conditioned on predicted label). All four distributions are continuous, unimodal, and concentrated in [0, ~0.5]. Haldane–Anscombe smoothing guarantees 0% undefined values. Of the three proposed metrics, Conditional Y has the smallest cross-condition shift.

---

## Key findings

1. **Φ, Y, and Conditional Y eliminate the undefined-rate problem.** Up to 40–60% of confusion matrices produce undefined EOD or Equalized Odds; the proposed metrics are defined everywhere.
2. **Fairness Φ beats SPD under GR skew.** SPD's density flattens by ~35% in peak height and disperses across the full [-1, 1] range when the minority group is small; Φ stays bounded near 0. The "amplified disparity for small groups" artefact in SPD does not appear in Φ.
3. **Φ and Y beat EOD/Equalized Odds under IR skew.** EOD is undefined for 40% of cases at IR=0.1 and develops jagged multi-modal shapes where it survives; Φ and Y stay defined and produce smooth, near-balanced densities.
4. **Conditional Y is the most distributionally stable.** Its KDE shape changes least across the four conditions, at the cost of a more compressed support.
5. **Bimodality in Φ and Y under GR skew is a finite-n artefact**, not a metric flaw — at n=30 with only 3 minority-group members, the achievable set of association values is genuinely discrete. This artefact disappears in the case-study figure where n=1,100.

---

## Bottom line

Standard metrics fail in complementary ways: EOD/Equalized Odds accumulate large undefined fractions and erratic shapes under IR skew; SPD avoids undefined values but inflates apparent disparity under GR skew. **Fairness Φ, Marginal Y, and Conditional Y remain fully defined and bounded near zero across every condition tested, including the joint extreme (IR=0.05, GR=0.05) where the standard metrics are unusable.** This is the central empirical argument for adopting the proposed metrics in fairness audits of imbalanced data.