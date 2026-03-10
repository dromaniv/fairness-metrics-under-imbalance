from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st

import builtin_metrics  # noqa: F401 - registers built-in metrics
try:
    import custom_metrics  # noqa: F401 - optional user-defined metrics
except Exception as exc:  # pragma: no cover - Streamlit feedback only
    CUSTOM_METRIC_IMPORT_ERROR = str(exc)
else:
    CUSTOM_METRIC_IMPORT_ERROR = None

from adult_case_study import (
    CLASSIFIERS,
    aggregate_case_results,
    evaluate_case_study,
    load_adult_dataset,
    paper_ratio_sweep,
)
from metric_registry import list_metrics, compute_metrics, COUNT_COLUMNS
from plots import (
    plot_case_grouped_bar_by_classifier,
    plot_case_grouped_bar_by_metric,
    plot_case_line,
    plot_case_line_abs,
    plot_case_line_all,
    plot_case_nan,
    plot_histogram_grid,
    plot_metric_vs_performance_heatmap,
    plot_probability_lines,
    ratio_label,
)
from synthetic_analysis import (
    probability_of_nan,
    probability_of_perfect_fairness,
)
from synthetic_data import (
    add_base_columns,
    count_confusion_matrices,
    dump_confusion_matrices_to_pickle,
    generate_exact_confusion_matrices,
    load_confusion_matrices_from_pickle,
    paper_ratio_defaults,
    ratio_values,
    sample_uniform_confusion_matrices,
)


st.set_page_config(page_title="Fairness Measures Explorer", layout="wide", page_icon="⚖️")


def nearest_available_ratios(total: int, targets: list[float]) -> list[float]:
    available = ratio_values(total)
    chosen: list[float] = []
    for target in targets:
        best = min(available, key=lambda value: abs(value - target))
        if best not in chosen:
            chosen.append(best)
    return chosen


_SMOOTHABLE_METRICS = {"conditional_q_association", "conditional_y_association"}


def apply_smoothing_override(df: pd.DataFrame, metric_key: str, smoothing: bool) -> pd.DataFrame:
    if metric_key not in _SMOOTHABLE_METRICS:
        return df
    from custom_metrics import conditional_q_association, conditional_y_association
    fn = conditional_q_association if metric_key == "conditional_q_association" else conditional_y_association
    out = df.copy()
    out[metric_key] = fn(out, smoothing=smoothing)
    return out


def smoothing_toggle(metric_key: str, widget_key: str) -> bool:
    """Render a smoothing checkbox when metric_key is smoothable; return the current value."""
    if metric_key in _SMOOTHABLE_METRICS:
        return st.checkbox(
            "Haldane-Anscombe smoothing (+0.5)",
            value=True,
            key=widget_key,
            help="Adds 0.5 to each cell of the per-stratum 2×2 table before computing the odds ratio. "
                 "When enabled, CQA is always defined (even at IR=0 or IR=1).",
        )
    return True  # irrelevant for other metrics, value unused


# Mapping from raw metric key → its FRN-wrapped sibling.
_FRN_KEY_MAP: dict[str, str] = {
    "accuracy_equality_difference":          "frn_accd",
    "statistical_parity_difference":         "frn_spd",
    "equal_opportunity_difference":          "frn_eod",
    "predictive_equality_difference":        "frn_fprd",
    "positive_predictive_parity_difference": "frn_ppvd",
    "negative_predictive_parity_difference": "frn_npvd",
}


def resolve_frn_key(key: str, use_frn: bool) -> str:
    """Return the FRN variant of *key* when *use_frn* is True and a variant exists."""
    if use_frn and key in _FRN_KEY_MAP:
        return _FRN_KEY_MAP[key]
    return key


def frn_toggle(metric_key: str, widget_key: str) -> bool:
    """Render an FRN checkbox when *metric_key* has an FRN variant; return the current value."""
    if metric_key in _FRN_KEY_MAP:
        frn_key = _FRN_KEY_MAP[metric_key]
        try:
            frn_label = list_metrics("fairness_frn")
            frn_label = next(s.label for s in frn_label if s.key == frn_key)
        except StopIteration:
            frn_label = frn_key
        return st.checkbox(
            "Apply Feasible-Range Normalization",
            value=False,
            key=widget_key,
            help=(
                "Scales the metric by the maximum (or minimum) value it could reach given the "
                "current confusion-matrix marginals, so the result is always in [−1, 1].\n\n"
                f"Shows: **{frn_label}**"
            ),
        )
    return False


def apply_frn_to_keys(keys: list[str], use_frn: bool) -> list[str]:
    """Replace each key in *keys* with its FRN variant when *use_frn* is True."""
    return [resolve_frn_key(k, use_frn) for k in keys]


def _valid_fairness_keys(keys: list[str]) -> list[str]:
    """Filter out any keys no longer present in the registry (e.g. from stale session state).
    Accepts both 'fairness' and 'fairness_frn' category keys.
    """
    registered = {spec.key for spec in list_metrics("fairness")} | \
                 {spec.key for spec in list_metrics("fairness_frn")}
    return [k for k in keys if k in registered]


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def figure_png_bytes(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer.read()


def fairness_metric_specs():
    return list_metrics("fairness")


def performance_metric_specs():
    return list_metrics("performance")


def metric_selector(label: str, category: str, default_keys: list[str] | None = None):
    specs = list_metrics(category)
    options = [spec.key for spec in specs]
    label_map = {spec.key: spec.label for spec in specs}
    default = [k for k in (default_keys or options) if k in options]
    return st.multiselect(label, options=options, default=default, format_func=lambda key: label_map[key])


def render_metric_registry_page() -> None:
    st.header("Metric registry")
    if CUSTOM_METRIC_IMPORT_ERROR:
        st.warning(f"custom_metrics.py failed to import: {CUSTOM_METRIC_IMPORT_ERROR}")

    category_labels = {
        "fairness":     "Fairness metrics",
        "fairness_frn": "Feasible-Range Normalized metrics",
        "performance":  "Performance metrics",
        "component":    "Component metrics",
        "ratio":        "Ratio metrics",
    }

    all_cats = list(dict.fromkeys(s.category for s in list_metrics()))
    for cat in all_cats:
        specs = list_metrics(cat)
        st.subheader(category_labels.get(cat, cat))
        for spec in specs:
            with st.expander(spec.label):
                st.latex(spec.formula)
                st.caption(spec.description)


def render_synthetic_page() -> None:
    st.header("Synthetic study")
    st.write(
        "Generate or sample the full space of 8-cell confusion matrices, compute fairness metrics, and reproduce the paper's synthetic plots."
    )

    with st.sidebar:
        st.subheader("Synthetic dataset")
        synth_mode = st.radio(
            "Dataset source",
            options=["Exact enumeration", "Monte Carlo sample", "Load pickle"],
            key="synthetic_mode",
        )
        total = st.number_input("n (total samples)", min_value=1, value=24, step=1, key="synthetic_total_input")
        estimated_rows = count_confusion_matrices(int(total))
        st.caption(f"All possible confusion matrices: {estimated_rows:,}")
        if estimated_rows > 20_000_000:
            st.warning("Exact generation at this n is likely too heavy for an interactive session. Use sampling or reduce n.")

        max_exact_rows = st.number_input(
            "Exact-generation row cap",
            min_value=1000,
            value=20_000_000,
            step=1000,
            key="synthetic_exact_cap",
        )
        monte_carlo_draws = st.number_input(
            "Monte Carlo draws",
            min_value=100,
            value=200_000,
            step=100,
            key="synthetic_draws",
        )
        seed = st.number_input("Random seed", min_value=0, value=2137, step=1, key="synthetic_seed")
        pickle_path = st.text_input(
            "Pickle path (Set(08,n).bin or saved by this app)",
            value="",
            key="synthetic_pickle_path",
        )

        if st.button("Build synthetic dataset", type="primary"):
            try:
                if synth_mode == "Exact enumeration":
                    df = add_base_columns(generate_exact_confusion_matrices(int(total), max_rows=int(max_exact_rows)))
                    built_total = int(total)
                elif synth_mode == "Monte Carlo sample":
                    df = add_base_columns(sample_uniform_confusion_matrices(int(total), int(monte_carlo_draws), seed=int(seed)))
                    built_total = int(total)
                else:
                    if not pickle_path.strip():
                        raise ValueError("Enter a path to a pickle file first.")
                    df = add_base_columns(load_confusion_matrices_from_pickle(pickle_path.strip()))
                    built_total = int(df[["i_tp", "i_fp", "i_tn", "i_fn", "j_tp", "j_fp", "j_tn", "j_fn"]].iloc[0].sum())
                st.session_state["synthetic_df"] = df
                st.session_state["synthetic_total"] = built_total
                st.success("Synthetic dataset ready.")
            except Exception as exc:
                st.error(str(exc))

    synthetic_df = st.session_state.get("synthetic_df")
    synthetic_total = int(st.session_state.get("synthetic_total", total))
    if synthetic_df is None:
        st.info("Build a synthetic dataset from the sidebar to begin.")
        return

    st.subheader("Dataset summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(synthetic_df):,}")
    c2.metric("n", synthetic_total)
    c3.metric("Unique IR values", synthetic_df["imbalance_ratio"].nunique())
    c4.metric("Unique GR values", synthetic_df["group_ratio_i"].nunique())

    with st.expander("Download current synthetic dataset"):
        st.download_button(
            "Download pickle",
            data=dump_confusion_matrices_to_pickle(synthetic_df),
            file_name=f"Set(08,{synthetic_total})_streamlit.pkl",
            mime="application/octet-stream",
        )
        st.download_button(
            "Download CSV",
            data=dataframe_csv_bytes(synthetic_df),
            file_name=f"synthetic_confusion_matrices_n{synthetic_total}.csv",
            mime="text/csv",
        )

    tabs = st.tabs(["Histogram grids", "Perfect fairness / NaN", "Fairness vs performance", "Data table"])
    fairness_specs = fairness_metric_specs()
    fairness_label_map = {spec.key: spec.label for spec in fairness_specs}

    with tabs[0]:
        st.subheader("Histogram grids")
        col1, col2, col3 = st.columns([1, 1, 1])
        metric_key = col1.selectbox(
            "Metric",
            options=[spec.key for spec in fairness_specs],
            format_func=lambda key: fairness_label_map[key],
            key="hist_metric_key",
        )
        group_ratio_basis = col2.radio(
            "Group ratio basis",
            options=["i", "j"],
            format_func=lambda value: f"{value}-group / total",
            horizontal=True,
            key="hist_gr_basis",
        )
        bins = int(col3.number_input("Histogram bins", min_value=5, value=109, step=1, key="hist_bins"))

        default_grid = nearest_available_ratios(synthetic_total, paper_ratio_defaults(synthetic_total))
        available_gr = sorted(ratio_values(synthetic_total))
        available_ir = sorted(ratio_values(synthetic_total))

        selected_gr = sorted(st.multiselect(
            "GR panel values",
            options=available_gr,
            default=default_grid,
            format_func=ratio_label,
            key="hist_selected_gr",
        ))
        selected_ir = sorted(st.multiselect(
            "IR panel values",
            options=available_ir,
            default=sorted(default_grid),
            format_func=ratio_label,
            key="hist_selected_ir",
        ))
        show_nan_bar = st.checkbox("Show separate undefined-value bar", value=True, key="hist_nan_bar")
        hist_smoothing = smoothing_toggle(metric_key, "hist_smoothing")
        hist_frn = frn_toggle(metric_key, "hist_frn")
        active_metric_key = resolve_frn_key(metric_key, hist_frn)

        if selected_gr and selected_ir:
            hist_df = apply_smoothing_override(synthetic_df, active_metric_key, hist_smoothing)
            fig = plot_histogram_grid(
                hist_df,
                active_metric_key,
                fairness_label_map.get(active_metric_key, fairness_label_map[metric_key]),
                selected_gr,
                selected_ir,
                bins=bins,
                group_ratio_basis=group_ratio_basis,
                show_nan_bar=show_nan_bar,
            )
            st.pyplot(fig, use_container_width=True)
            st.download_button(
                "Download histogram grid (PNG)",
                data=figure_png_bytes(fig),
                file_name=f"histogram_grid_{active_metric_key}.png",
                mime="image/png",
            )
        else:
            st.info("Pick at least one GR and one IR value.")

    with tabs[1]:
        st.subheader("Probability of perfect fairness and undefined values")
        col1, col2, col3, col4 = st.columns([1.4, 1, 1, 1])
        selected_metric_keys = col1.multiselect(
            "Metrics",
            options=[spec.key for spec in fairness_specs],
            default=[spec.key for spec in fairness_specs],
            format_func=lambda key: fairness_label_map[key],
            key="ppf_metric_keys",
        )
        ratio_type = col2.radio("Sweep", options=["ir", "gr"], horizontal=True, key="ppf_ratio_type")
        epsilon = float(col3.number_input("Epsilon for near-perfect fairness", min_value=0.0, value=0.0, step=0.001))
        ppf_basis = col4.radio(
            "GR basis",
            options=["i", "j"],
            horizontal=True,
            key="ppf_gr_basis",
        )
        ppf_smoothing = st.checkbox(
            "Haldane-Anscombe smoothing for Conditional Q Association",
            value=True,
            key="ppf_smoothing",
            help="Applies +0.5 smoothing when computing Conditional Q Association.",
        ) if any(k in _SMOOTHABLE_METRICS for k in selected_metric_keys) else True

        ppf_frn = st.checkbox(
            "Apply Feasible-Range Normalization to supported metrics",
            value=False,
            key="ppf_frn",
            help=(
                "Replaces each metric that has a feasible-range normalized variant with that version. "
                "Supported: " + ", ".join(
                    s.label for s in list_metrics("fairness") if s.key in _FRN_KEY_MAP
                )
            ),
        ) if any(k in _FRN_KEY_MAP for k in selected_metric_keys) else False

        active_ppf_keys = apply_frn_to_keys(selected_metric_keys, ppf_frn)
        # rebuild label map to include any FRN keys that replaced raw ones
        all_ppf_label_map = {**fairness_label_map, **{spec.key: spec.label for spec in list_metrics("fairness_frn")}}

        if selected_metric_keys:
            ppf_work_df = synthetic_df
            for k in active_ppf_keys:
                ppf_work_df = apply_smoothing_override(ppf_work_df, k, ppf_smoothing)
            ppf_df = probability_of_perfect_fairness(
                ppf_work_df,
                active_ppf_keys,
                ratio_type,
                epsilon=epsilon,
                group_ratio_basis=ppf_basis,
            )
            nan_df = probability_of_nan(
                ppf_work_df,
                active_ppf_keys,
                ratio_type,
                group_ratio_basis=ppf_basis,
            )
            fig1 = plot_probability_lines(
                ppf_df,
                active_ppf_keys,
                all_ppf_label_map,
                ratio_type,
                title="Probability of perfect fairness",
                y_label="Probability of perfect fairness",
                group_ratio_basis=ppf_basis,
                y_max=1.0 if ratio_type == "ir" else None,
            )
            fig2 = plot_probability_lines(
                nan_df,
                active_ppf_keys,
                all_ppf_label_map,
                ratio_type,
                title="Probability of undefined values",
                y_label="Probability of undefined metric value",
                group_ratio_basis=ppf_basis,
                y_max=1.0,
            )
            left, right = st.columns(2)
            left.pyplot(fig1, use_container_width=True)
            right.pyplot(fig2, use_container_width=True)
            st.download_button(
                "Download perfect-fairness CSV",
                data=dataframe_csv_bytes(ppf_df),
                file_name=f"perfect_fairness_{ratio_type}.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download undefined-value CSV",
                data=dataframe_csv_bytes(nan_df),
                file_name=f"undefined_probability_{ratio_type}.csv",
                mime="text/csv",
            )

    with tabs[2]:
        st.subheader("Fairness vs predictive performance")
        col1, col2, col3 = st.columns([1, 1, 1])
        fairness_key = col1.selectbox(
            "Fairness metric",
            options=[spec.key for spec in fairness_specs],
            format_func=lambda key: fairness_label_map[key],
            key="heatmap_fairness_key",
        )
        performance_specs = performance_metric_specs()
        performance_label_map = {spec.key: spec.label for spec in performance_specs}
        performance_key = col2.selectbox(
            "Performance measure",
            options=[spec.key for spec in performance_specs],
            format_func=lambda key: performance_label_map[key],
            key="heatmap_performance_key",
        )
        heat_bins = int(col3.number_input("Heatmap bins", min_value=10, value=100, step=10, key="heatmap_bins"))
        heat_smoothing = smoothing_toggle(fairness_key, "heatmap_smoothing")
        heat_frn = frn_toggle(fairness_key, "heatmap_frn")
        active_fairness_key = resolve_frn_key(fairness_key, heat_frn)
        heat_df = apply_smoothing_override(synthetic_df, active_fairness_key, heat_smoothing)
        fig = plot_metric_vs_performance_heatmap(
            heat_df,
            active_fairness_key,
            fairness_label_map.get(active_fairness_key, fairness_label_map[fairness_key]),
            performance_key,
            performance_label_map[performance_key],
            bins=heat_bins,
        )
        st.pyplot(fig, use_container_width=True)
        st.download_button(
            "Download heatmap (PNG)",
            data=figure_png_bytes(fig),
            file_name=f"heatmap_{active_fairness_key}_vs_{performance_key}.png",
            mime="image/png",
        )

    with tabs[3]:
        st.subheader("Synthetic data table")
        all_preview_specs = list_metrics("fairness")
        all_preview_keys = [s.key for s in all_preview_specs]
        all_preview_labels = {s.key: s.label for s in all_preview_specs}
        preview_metric_keys = st.multiselect(
            "Additional metric columns",
            options=all_preview_keys,
            default=[],
            format_func=lambda key: all_preview_labels.get(key, key),
            key="synthetic_preview_metrics",
        )
        hide_degenerate = st.checkbox(
            "Hide rows where either group is empty",
            value=True,
            key="synthetic_hide_degenerate",
            help="Removes rows where i_total=0 or j_total=0. These produce NaN for most metrics.",
        )
        work_df = synthetic_df.copy()
        if hide_degenerate:
            i_total = work_df["i_tp"] + work_df["i_fp"] + work_df["i_tn"] + work_df["i_fn"]
            j_total = work_df["j_tp"] + work_df["j_fp"] + work_df["j_tn"] + work_df["j_fn"]
            work_df = work_df[(i_total > 0) & (j_total > 0)].reset_index(drop=True)
        else:
            work_df = work_df.reset_index(drop=True)

        perf_computed = compute_metrics(work_df, ["accuracy", "g_mean"]).reset_index(drop=True)
        base_df = work_df[COUNT_COLUMNS].reset_index(drop=True)
        if preview_metric_keys:
            extra_computed = compute_metrics(work_df, preview_metric_keys).reset_index(drop=True)
            display_df = pd.concat([base_df, perf_computed, extra_computed], axis=1)
        else:
            display_df = pd.concat([base_df, perf_computed], axis=1)
        st.caption(f"Showing {min(1000, len(display_df)):,} of {len(display_df):,} rows.")
        st.dataframe(display_df.head(1000), use_container_width=True)


def render_case_study_page() -> None:
    st.header("Adult case study")
    st.write(
        "Run the controlled Adult/Census Income experiment with varying imbalance ratio (IR) and group ratio (GR), mirroring the case study in the paper."
    )

    # Progress placeholders live in the main area so they are visible while the sidebar is open.
    _progress_bar = st.empty()
    _progress_caption = st.empty()

    with st.sidebar:
        st.subheader("Adult data source")
        default_path = "data/adult.data"
        adult_source_mode = st.radio("Adult dataset source", options=["Local path", "Upload file"], key="adult_source_mode")
        uploaded = None
        path_value = default_path
        if adult_source_mode == "Upload file":
            uploaded = st.file_uploader("adult.data", type=["data", "csv", "txt"], key="adult_upload")
        else:
            path_value = st.text_input("Path to adult.data", value=default_path, key="adult_path")

        st.subheader("Experiment controls")
        sweep_ratios = st.multiselect(
            "Ratios to sweep",
            options=paper_ratio_sweep(),
            default=paper_ratio_sweep(),
            format_func=lambda value: f"{value:.2f}",
            key="adult_ratio_values",
        )
        fixed_ratio = st.number_input("Fixed ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="adult_fixed_ratio")
        sample_size = st.number_input("Subset size", min_value=100, value=1100, step=100, key="adult_sample_size")
        holdout_splits = st.number_input("Holdout repetitions", min_value=1, value=50, step=1, key="adult_holdout_splits")
        test_size = st.slider("Test size", min_value=0.1, max_value=0.9, value=0.33, step=0.01, key="adult_test_size")
        random_state = st.number_input("Random seed", min_value=0, value=2137, step=1, key="adult_random_state")
        selected_classifiers = st.multiselect(
            "Classifiers",
            options=list(CLASSIFIERS.keys()),
            default=list(CLASSIFIERS.keys()),
            key="adult_classifiers",
        )
        selected_fairness_metrics = metric_selector(
            "Fairness metrics",
            "fairness",
            default_keys=[spec.key for spec in fairness_metric_specs()],
        )
        case_frn = st.checkbox(
            "Apply Feasible-Range Normalization to supported metrics",
            value=False,
            key="adult_frn",
            help=(
                "Replaces each selected metric that has a feasible-range normalized variant with its "
                "normalized version before running the experiment."
            ),
        )

        if st.button("Run Adult case study", type="primary"):
            try:
                if adult_source_mode == "Upload file":
                    if uploaded is None:
                        raise ValueError("Upload the Adult dataset first.")
                    adult_df = load_adult_dataset(uploaded.getvalue())
                else:
                    adult_df = load_adult_dataset(path_value)

                # Guard against stale widget state holding keys removed from the registry.
                validated_metrics = _valid_fairness_keys(
                    apply_frn_to_keys(selected_fairness_metrics, case_frn)
                )
                if not validated_metrics:
                    raise ValueError("No valid fairness metrics selected.")

                _progress_bar.progress(0.0, text="Starting…")

                def _progress(frac: float, msg: str) -> None:
                    _progress_bar.progress(min(frac, 1.0), text=msg)
                    _progress_caption.caption(msg)

                fairness_results, performance_results = evaluate_case_study(
                    adult_df,
                    ratio_values=sweep_ratios,
                    fixed_ratio=fixed_ratio,
                    sample_size=int(sample_size),
                    holdout_splits=int(holdout_splits),
                    test_size=float(test_size),
                    classifier_names=selected_classifiers,
                    fairness_metric_keys=validated_metrics,
                    random_state=int(random_state),
                    progress_callback=_progress,
                )
                _progress_bar.empty()
                _progress_caption.empty()
                st.session_state["adult_fairness_results"] = fairness_results
                st.session_state["adult_performance_results"] = performance_results
                st.success("Adult case study finished.")
            except Exception as exc:
                st.error(str(exc))

    fairness_results = st.session_state.get("adult_fairness_results")
    performance_results = st.session_state.get("adult_performance_results")
    if fairness_results is None or performance_results is None:
        st.info("Provide the Adult dataset in the sidebar and run the experiment.")
        return

    all_fairness_specs = list_metrics("fairness") + list_metrics("fairness_frn")
    fairness_label_map = {spec.key: spec.label for spec in all_fairness_specs}

    c1, c2, c3 = st.columns(3)
    c1.metric("Fairness rows", f"{len(fairness_results):,}")
    c2.metric("Performance rows", f"{len(performance_results):,}")
    c3.metric("Classifiers", len(pd.unique(fairness_results["clf"])))

    tabs = st.tabs(["Fairness lines", "NaN probability", "All metrics", "Tables", "Grouped bars", "Raw results"])

    with tabs[0]:
        col1, col2, col3 = st.columns([1, 1, 1])
        line_metric_key = col1.selectbox(
            "Metric",
            options=list(pd.unique(fairness_results["metric"])),
            format_func=lambda key: fairness_label_map.get(key, key),
            key="adult_line_metric",
        )
        ratio_type = col2.radio("Sweep", options=["ir", "gr"], horizontal=True, key="adult_line_ratio")
        fill = col3.radio("Band", options=["std", "err"], horizontal=True, key="adult_line_fill")
        fig1 = plot_case_line(
            fairness_results,
            line_metric_key,
            fairness_label_map.get(line_metric_key, line_metric_key),
            ratio_type,
            fill=fill,
        )
        fig2 = plot_case_line_abs(
            fairness_results,
            line_metric_key,
            fairness_label_map.get(line_metric_key, line_metric_key),
            ratio_type,
            fill=fill,
        )
        left, right = st.columns(2)
        left.pyplot(fig1, use_container_width=True)
        right.pyplot(fig2, use_container_width=True)

    with tabs[1]:
        ratio_type = st.radio("NaN sweep", options=["ir", "gr"], horizontal=True, key="adult_nan_ratio")
        metric_keys = list(pd.unique(fairness_results["metric"]))
        fig = plot_case_nan(fairness_results, metric_keys, fairness_label_map, ratio_type)
        st.pyplot(fig, use_container_width=True)

    with tabs[2]:
        ratio_type = st.radio("Combined sweep", options=["ir", "gr"], horizontal=True, key="adult_all_ratio")
        fill = st.radio("Combined band", options=["std", "err"], horizontal=True, key="adult_all_fill")
        metric_keys = list(pd.unique(fairness_results["metric"]))
        fig = plot_case_line_all(fairness_results, metric_keys, fairness_label_map, ratio_type, fill=fill)
        st.pyplot(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Aggregated tables")
        fairness_agg = aggregate_case_results(fairness_results)
        performance_agg = aggregate_case_results(performance_results)
        st.write("Fairness metrics")
        st.dataframe(fairness_agg, use_container_width=True, hide_index=True)
        st.write("Classifier performance")
        st.dataframe(performance_agg, use_container_width=True, hide_index=True)
        st.download_button(
            "Download fairness aggregation CSV",
            data=dataframe_csv_bytes(fairness_agg),
            file_name="adult_fairness_aggregation.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download performance aggregation CSV",
            data=dataframe_csv_bytes(performance_agg),
            file_name="adult_performance_aggregation.csv",
            mime="text/csv",
        )

    with tabs[4]:
        st.subheader("Single-slice grouped bar charts")
        available_gr = sorted(pd.unique(fairness_results["gr"]))
        available_ir = sorted(pd.unique(fairness_results["ir"]))
        col1, col2 = st.columns(2)
        selected_gr = col1.selectbox("GR", options=available_gr, format_func=lambda value: f"{value:.2f}", key="adult_bar_gr")
        selected_ir = col2.selectbox("IR", options=available_ir, format_func=lambda value: f"{value:.2f}", key="adult_bar_ir")
        subset = fairness_results[
            (fairness_results["gr"] == selected_gr) & (fairness_results["ir"] == selected_ir)
        ]
        if subset.empty:
            st.info("No rows for the selected GR/IR combination.")
        else:
            fairness_map: dict[str, dict[str, float]] = {}
            for clf_name, group in subset.groupby("clf"):
                fairness_map[clf_name] = {
                    fairness_label_map.get(mk, mk): float(group[group["metric"] == mk]["value"].mean())
                    for mk in pd.unique(subset["metric"])
                }
            title = f"Fairness metrics for classifiers; GR = {selected_gr:.2f}, IR = {selected_ir:.2f}"
            fig1 = plot_case_grouped_bar_by_metric(fairness_map, title=title)
            fig2 = plot_case_grouped_bar_by_classifier(fairness_map, title=title)
            left, right = st.columns(2)
            left.pyplot(fig1, use_container_width=True)
            right.pyplot(fig2, use_container_width=True)

    with tabs[5]:
        st.subheader("Raw results")
        st.write("Fairness results")
        st.dataframe(fairness_results.head(1000), use_container_width=True)
        st.write("Performance results")
        st.dataframe(performance_results.head(1000), use_container_width=True)
        st.download_button(
            "Download raw fairness CSV",
            data=dataframe_csv_bytes(fairness_results),
            file_name="adult_fairness_results.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download raw performance CSV",
            data=dataframe_csv_bytes(performance_results),
            file_name="adult_performance_results.csv",
            mime="text/csv",
        )


st.title("Fairness Measures Explorer")
st.caption(
    "Interactive Streamlit recreation of the synthetic analyses and Adult case study from the provided fairness-measure repository."
)

page = st.radio("Workflow", options=["Synthetic study", "Adult case study", "Metric registry"], horizontal=True)

if page == "Synthetic study":
    render_synthetic_page()
elif page == "Adult case study":
    render_case_study_page()
else:
    render_metric_registry_page()
