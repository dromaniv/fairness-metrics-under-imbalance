"""Adult dataset case study reproduced as reusable functions for the Streamlit app."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from imblearn.metrics import geometric_mean_score
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, recall_score
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from metric_registry import compute_metrics


warnings.filterwarnings("ignore")


FEATURES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

ADULT_COLUMNS = FEATURES + ["income"]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

EDUCATION_ORDER = [
    "Preschool",
    "1st-4th",
    "5th-6th",
    "7th-8th",
    "9th",
    "10th",
    "11th",
    "12th",
    "HS-grad",
    "Some-college",
    "Assoc-acdm",
    "Assoc-voc",
    "Bachelors",
    "Masters",
    "Prof-school",
    "Doctorate",
]


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    builder: Callable[[int], ClassifierMixin]


CLASSIFIERS: dict[str, ClassifierSpec] = {
    "Random Forest": ClassifierSpec(
        name="Random Forest",
        builder=lambda random_state: RandomForestClassifier(random_state=random_state),
    ),
    "Decision Tree": ClassifierSpec(
        name="Decision Tree",
        builder=lambda random_state: DecisionTreeClassifier(random_state=random_state),
    ),
    "Gaussian NB": ClassifierSpec(
        name="Gaussian NB",
        builder=lambda random_state: GaussianNB(),
    ),
    "Logistic Regression": ClassifierSpec(
        name="Logistic Regression",
        builder=lambda random_state: LogisticRegression(),
    ),
    "KNN": ClassifierSpec(
        name="KNN",
        builder=lambda random_state: KNeighborsClassifier(),
    ),
    "MLP": ClassifierSpec(
        name="MLP",
        builder=lambda random_state: MLPClassifier(random_state=random_state),
    ),
}


PERFORMANCE_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc_score": lambda y_true, y_pred: float(roc_auc_score(y_true, y_pred)),
    "geometric_mean_score": lambda y_true, y_pred: float(geometric_mean_score(y_true, y_pred, labels=[0, 1])),
    "recall_score": lambda y_true, y_pred: float(recall_score(y_true, y_pred, labels=[0, 1])),
    "f1_score": lambda y_true, y_pred: float(f1_score(y_true, y_pred, labels=[0, 1])),
}


def paper_ratio_sweep() -> list[float]:
    return [0.01, 0.02, 0.05] + [round(x, 2) for x in np.arange(0.1, 1.0, 0.1)] + [0.95, 0.98, 0.99]



def load_adult_dataset(source: str | Path | bytes | bytearray) -> pd.DataFrame:
    """Load the Adult dataset from a path or raw uploaded bytes."""

    if isinstance(source, (str, Path)):
        df = pd.read_csv(
            source,
            sep=r",\s*",
            engine="python",
            na_values=["?", " ?"],
            header=None,
            names=ADULT_COLUMNS,
        )
    else:
        from io import BytesIO

        df = pd.read_csv(
            BytesIO(bytes(source)),
            sep=r",\s*",
            engine="python",
            na_values=["?", " ?"],
            header=None,
            names=ADULT_COLUMNS,
        )
    return df



def _four_way_counts(sample_size: int, gr: float, ir: float) -> dict[tuple[str, str], int]:
    """Allocate integer cell counts while preserving the requested total size."""

    ideals = {
        ("protected", "negative"): sample_size * gr * (1.0 - ir),
        ("protected", "positive"): sample_size * gr * ir,
        ("unprotected", "negative"): sample_size * (1.0 - gr) * (1.0 - ir),
        ("unprotected", "positive"): sample_size * (1.0 - gr) * ir,
    }
    counts = {key: int(np.floor(value)) for key, value in ideals.items()}
    remainder = sample_size - sum(counts.values())
    order = sorted(ideals.keys(), key=lambda key: (ideals[key] - counts[key]), reverse=True)
    for idx in range(remainder):
        counts[order[idx % len(order)]] += 1
    return counts



def sample_adult_subset(
    df: pd.DataFrame,
    *,
    sample_size: int,
    gr: float,
    ir: float,
    protected_col: str = "sex",
    protected_value: str = "Female",
    target_col: str = "income",
    positive_label: str = ">50K",
    random_state: int = 2137,
) -> pd.DataFrame:
    """Sample a subset with approximately the requested protected/class proportions."""

    counts = _four_way_counts(sample_size, gr, ir)
    negative_label = next(value for value in sorted(df[target_col].dropna().unique()) if value != positive_label)
    pieces: list[pd.DataFrame] = []
    for (group_name, class_name), n_rows in counts.items():
        if n_rows == 0:
            continue
        group_value = protected_value if group_name == "protected" else None
        target_value = positive_label if class_name == "positive" else negative_label
        if group_name == "protected":
            pool = df[(df[protected_col] == protected_value) & (df[target_col] == target_value)]
        else:
            pool = df[(df[protected_col] != protected_value) & (df[target_col] == target_value)]
        if len(pool) < n_rows:
            raise ValueError(
                f"Not enough rows for cell {(group_name, class_name)}. Requested {n_rows}, available {len(pool)}."
            )
        pieces.append(pool.sample(n=n_rows, random_state=random_state))
        _ = group_value  # keep the logic explicit for readability
    sampled = pd.concat(pieces, axis=0).reset_index(drop=True)
    return sampled



def preprocess_adult(
    dataset: pd.DataFrame,
    *,
    target_col: str = "income",
    protected_col: str = "sex",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Encode the Adult dataset for classifier training."""

    feature_columns = [column for column in dataset.columns if column != target_col and column != "fnlwgt"]
    X_all = dataset[feature_columns].copy()
    y_all = LabelEncoder().fit_transform(dataset[target_col])
    protected_values = dataset[protected_col].to_numpy(copy=True)

    categorical = [column for column in CATEGORICAL_FEATURES if column in X_all.columns]
    numeric = [column for column in X_all.columns if column not in categorical]

    data_encoder = OrdinalEncoder()
    X_categorical = data_encoder.fit_transform(X_all[categorical])

    if "education" in categorical:
        education_idx = categorical.index("education")
        education_encoder = OrdinalEncoder(
            categories=[EDUCATION_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        )
        X_categorical[:, education_idx] = education_encoder.fit_transform(X_all[["education"]]).reshape(-1)

    X_numeric = np.concatenate([X_all[numeric].to_numpy(dtype=np.float64), X_categorical], axis=1)
    feature_order = numeric + categorical
    return X_numeric, y_all, protected_values, feature_order



def confusion_row_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_values: np.ndarray,
    *,
    protected_value: str,
    positive_class: int = 1,
) -> dict[str, int]:
    """Return one row of i/j confusion-matrix counts.

    ``i`` is the protected group and ``j`` is the complementary group so that the
    built-in fairness metrics keep the sign convention ``j - i``.
    """

    negative_class = 1 - positive_class
    protected_mask = protected_values == protected_value
    unprotected_mask = ~protected_mask

    def _counts(mask: np.ndarray) -> tuple[int, int, int, int]:
        yt = y_true[mask]
        yp = y_pred[mask]
        tp = int(np.sum((yt == positive_class) & (yp == positive_class)))
        fp = int(np.sum((yt == negative_class) & (yp == positive_class)))
        tn = int(np.sum((yt == negative_class) & (yp == negative_class)))
        fn = int(np.sum((yt == positive_class) & (yp == negative_class)))
        return tp, fp, tn, fn

    i_tp, i_fp, i_tn, i_fn = _counts(protected_mask)
    j_tp, j_fp, j_tn, j_fn = _counts(unprotected_mask)
    return {
        "i_tp": i_tp,
        "i_fp": i_fp,
        "i_tn": i_tn,
        "i_fn": i_fn,
        "j_tp": j_tp,
        "j_fp": j_fp,
        "j_tn": j_tn,
        "j_fn": j_fn,
    }



def evaluate_case_study(
    adult_df: pd.DataFrame,
    *,
    ratio_values: Iterable[float] | None = None,
    fixed_ratio: float = 0.5,
    sample_size: int = 1100,
    holdout_splits: int = 50,
    test_size: float = 0.33,
    classifier_names: Iterable[str] | None = None,
    fairness_metric_keys: Iterable[str] | None = None,
    protected_col: str = "sex",
    protected_value: str = "Female",
    target_col: str = "income",
    positive_label: str = ">50K",
    random_state: int = 2137,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the controlled Adult case study.

    Returns
    -------
    fairness_results:
        DataFrame with columns [gr, ir, clf, metric, value].
    performance_results:
        DataFrame with columns [gr, ir, clf, metric, value].
    """

    if ratio_values is None:
        ratio_values = paper_ratio_sweep()
    if classifier_names is None:
        classifier_names = list(CLASSIFIERS.keys())
    if fairness_metric_keys is None:
        raise ValueError("At least one fairness metric key must be supplied.")

    classifier_names = list(classifier_names)
    fairness_metric_keys = list(fairness_metric_keys)

    ratios = [(fixed_ratio, float(ir)) for ir in ratio_values] + [(float(gr), fixed_ratio) for gr in ratio_values]
    holdout = ShuffleSplit(n_splits=holdout_splits, test_size=test_size, random_state=random_state)

    fairness_rows: list[list[object]] = []
    performance_rows: list[list[object]] = []

    total_steps = len(ratios)
    for step_idx, (gr, ir) in enumerate(ratios):
        if progress_callback is not None:
            progress_callback(step_idx / total_steps, f"GR={gr:.2f} IR={ir:.2f}  ({step_idx + 1}/{total_steps})")
        subset = sample_adult_subset(
            adult_df,
            sample_size=sample_size,
            gr=gr,
            ir=ir,
            protected_col=protected_col,
            protected_value=protected_value,
            target_col=target_col,
            positive_label=positive_label,
            random_state=random_state,
        )
        X_all, y_all, protected_values, _ = preprocess_adult(
            subset,
            target_col=target_col,
            protected_col=protected_col,
        )

        for train_idx, test_idx in holdout.split(X_all):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]
            protected_test = protected_values[test_idx]

            for classifier_name in classifier_names:
                spec = CLASSIFIERS[classifier_name]
                estimator = spec.builder(random_state)
                pipe = make_pipeline(KNNImputer(), StandardScaler(), estimator)
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                confusion_row = confusion_row_from_predictions(
                    y_test,
                    y_pred,
                    protected_test,
                    protected_value=protected_value,
                    positive_class=1,
                )
                fairness_frame = compute_metrics(pd.DataFrame([confusion_row]), fairness_metric_keys)
                for metric_key in fairness_metric_keys:
                    fairness_rows.append([gr, ir, classifier_name, metric_key, float(fairness_frame.iloc[0][metric_key])])

                for metric_name, metric_func in PERFORMANCE_METRICS.items():
                    try:
                        value = metric_func(y_test, y_pred)
                    except Exception:
                        value = np.nan
                    performance_rows.append([gr, ir, classifier_name, metric_name, value])

    fairness_results = pd.DataFrame(fairness_rows, columns=["gr", "ir", "clf", "metric", "value"])
    performance_results = pd.DataFrame(performance_rows, columns=["gr", "ir", "clf", "metric", "value"])
    if progress_callback is not None:
        progress_callback(1.0, "Done")
    return fairness_results, performance_results



def aggregate_case_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean and std for table display and CSV export."""

    grouped = (
        results_df.groupby(["metric", "gr", "ir", "clf"], dropna=False)["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["metric", "ir", "gr", "clf"])
    )
    return grouped



def collect_adult_confusion_matrices(
    adult_df: pd.DataFrame,
    *,
    ratio_values: Iterable[float] | None = None,
    fixed_ratio: float = 0.5,
    sample_size: int = 1100,
    holdout_splits: int = 20,
    test_size: float = 0.33,
    classifier_names: Iterable[str] | None = None,
    protected_col: str = "sex",
    protected_value: str = "Female",
    target_col: str = "income",
    positive_label: str = ">50K",
    random_state: int = 2137,
    progress_callback: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """Collect raw confusion-matrix rows from real classifiers on the Adult dataset.

    Returns a DataFrame with the 8 confusion-matrix count columns plus
    ``gr``, ``ir``, and ``clf``.  Pass the result to ``add_base_columns``
    to get imbalance_ratio, stereotypical_ratio, etc.
    """
    if ratio_values is None:
        ratio_values = paper_ratio_sweep()
    if classifier_names is None:
        classifier_names = list(CLASSIFIERS.keys())

    classifier_names = list(classifier_names)
    ratios = (
        [(fixed_ratio, float(ir)) for ir in ratio_values]
        + [(float(gr), fixed_ratio) for gr in ratio_values]
    )
    holdout = ShuffleSplit(n_splits=holdout_splits, test_size=test_size, random_state=random_state)

    rows: list[dict] = []
    total = len(ratios)
    for step, (gr, ir) in enumerate(ratios):
        if progress_callback is not None:
            progress_callback(step / total, f"GR={gr:.2f} IR={ir:.2f}  ({step + 1}/{total})")
        try:
            subset = sample_adult_subset(
                adult_df,
                sample_size=sample_size,
                gr=gr,
                ir=ir,
                protected_col=protected_col,
                protected_value=protected_value,
                target_col=target_col,
                positive_label=positive_label,
                random_state=random_state,
            )
        except ValueError:
            continue
        X_all, y_all, protected_values, _ = preprocess_adult(
            subset, target_col=target_col, protected_col=protected_col,
        )
        for train_idx, test_idx in holdout.split(X_all):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]
            protected_test = protected_values[test_idx]
            for clf_name in classifier_names:
                spec = CLASSIFIERS[clf_name]
                try:
                    pipe = make_pipeline(KNNImputer(), StandardScaler(), spec.builder(random_state))
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)
                except Exception:
                    continue
                conf = confusion_row_from_predictions(
                    y_test, y_pred, protected_test,
                    protected_value=protected_value, positive_class=1,
                )
                conf["gr"] = gr
                conf["ir"] = ir
                conf["clf"] = clf_name
                rows.append(conf)

    if progress_callback is not None:
        progress_callback(1.0, "Done")
    return pd.DataFrame(rows)
