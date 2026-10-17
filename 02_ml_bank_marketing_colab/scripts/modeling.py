from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from scripts.data_setup import DEFAULT_DATA_DIR, MODEL_FEATURES

TARGET = "y"
CATEGORICAL_FEATURES = {"job", "education", "housing", "contact", "month"}
FEATURE_SETS = {
    "basic": ("age", "job", "education"),
    "standard": MODEL_FEATURES,
    "degraded": MODEL_FEATURES,
}
MODEL_NAMES = ("logistic_regression", "decision_tree", "random_forest")
DEFAULT_TRAIN_SIZES = (100, 300, 800, 1500)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a fully numeric table, ready to fit any sklearn model directly."""
    prepared = data.copy()
    if "had_previous_contact" not in prepared.columns:
        pdays = pd.to_numeric(prepared["pdays"], errors="coerce")
        prepared["had_previous_contact"] = (
            pdays.notna() & pdays.ne(999)
        ).astype("int64")
        prepared["pdays"] = pdays.mask(pdays.eq(999))

    prepared = prepared.loc[:, [*MODEL_FEATURES, TARGET]].copy()
    # Brak pdays oznacza brak wcześniejszego kontaktu; -1 to umowna wartość "nigdy".
    prepared["pdays"] = prepared["pdays"].fillna(-1)
    # Kolumny tekstowe zamieniamy na kolumny 0/1 (one-hot encoding).
    categorical = [column for column in MODEL_FEATURES if column in CATEGORICAL_FEATURES]
    return pd.get_dummies(prepared, columns=categorical, dtype="int64")


def make_fixed_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the shared stratified train/test split used by every experiment."""
    if TARGET not in data.columns:
        raise ValueError("Brakuje kolumny wynikowej 'y'.")
    if "duration" in data.columns:
        raise ValueError("'duration' musi być wykluczone: jest niedostępne przed rozmową.")
    if not 0 < test_size < 1:
        raise ValueError("test_size musi być pomiędzy 0 i 1.")

    valid = data.loc[data[TARGET].isin(["no", "yes"])].copy()
    features = [column for column in valid.columns if column != TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        valid[features],
        valid[TARGET],
        test_size=test_size,
        random_state=random_state,
        stratify=valid[TARGET],
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def _nested_training_order(y_train: pd.Series, random_state: int) -> np.ndarray:
    """Return one class-balanced random order whose prefixes are nested samples."""
    labels = y_train.to_numpy()
    classes = sorted(pd.unique(labels).tolist())
    proportions = {label: float(np.mean(labels == label)) for label in classes}
    rng = np.random.default_rng(random_state)
    queues = {
        label: rng.permutation(np.flatnonzero(labels == label)).tolist()
        for label in classes
    }
    positions = {label: 0 for label in classes}
    observed = {label: 0 for label in classes}
    order = []

    for sample_size in range(1, len(labels) + 1):
        available = [label for label in classes if positions[label] < len(queues[label])]
        next_label = max(
            available,
            key=lambda label: sample_size * proportions[label] - observed[label],
        )
        order.append(queues[next_label][positions[next_label]])
        positions[next_label] += 1
        observed[next_label] += 1

    return np.asarray(order, dtype=int)


def _apply_training_degradation(
    X: pd.DataFrame,
    rate: float,
    random_state: int,
) -> pd.DataFrame:
    """Hide a fraction of training feature cells; the test set is never changed."""
    if not 0 <= rate < 1:
        raise ValueError("degradation_rate musi być w przedziale [0, 1).")
    if rate == 0 or X.empty:
        return X.copy()

    result = X.copy()
    rng = np.random.default_rng(random_state)
    missing_mask = rng.random(result.shape) < rate
    for index, column in enumerate(result.columns):
        result.loc[missing_mask[:, index], column] = np.nan
    return result


def _estimator(model_name: str, random_state: int):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=8,
            random_state=random_state,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Nieznany model '{model_name}'. Wybierz: {MODEL_NAMES}")


def build_pipeline(
    model_name: str,
    feature_columns: tuple[str, ...] | list[str],
    random_state: int = 42,
) -> Pipeline:
    """Build a leakage-safe preprocessing and model pipeline."""
    feature_columns = list(feature_columns)
    # Kolumny spoza CATEGORICAL_FEATURES (także 0/1 z preprocess_data) traktujemy jako liczbowe.
    numeric = [column for column in feature_columns if column not in CATEGORICAL_FEATURES]
    categorical = [column for column in feature_columns if column in CATEGORICAL_FEATURES]

    transformers = []
    if numeric:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if model_name == "logistic_regression":
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", _estimator(model_name, random_state)),
    ])


def _score(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "recall_yes": float(recall_score(y_test, predictions, pos_label="yes", zero_division=0)),
    }


def always_no_baseline(
    data: pd.DataFrame,
    test_size: float = 0.2,
    split_seed: int = 42,
) -> dict[str, float | int | str]:
    """Score a constant 'no' prediction on the fixed stratified test set."""
    _, X_test, _, y_test = make_fixed_split(data, test_size, split_seed)
    predictions = np.repeat("no", len(y_test))
    return {
        "model": "zawsze brak lokaty",
        "test_rows": len(X_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "recall_yes": float(recall_score(y_test, predictions, pos_label="yes", zero_division=0)),
    }


def run_experiments(
    data: pd.DataFrame,
    model_name: str,
    train_sizes: tuple[int, ...] = DEFAULT_TRAIN_SIZES,
    feature_sets: tuple[str, ...] = ("basic", "standard", "degraded"),
    repeats: int = 3,
    degradation_rate: float = 0.15,
    test_size: float = 0.2,
    split_seed: int = 42,
) -> pd.DataFrame:
    """Evaluate nested training samples on one immutable test set."""
    if repeats < 1:
        raise ValueError("repeats musi wynosić co najmniej 1.")
    if not train_sizes or any(size < 1 for size in train_sizes):
        raise ValueError("train_sizes musi zawierać dodatnie liczby.")
    if tuple(sorted(set(train_sizes))) != train_sizes:
        raise ValueError("train_sizes podaj rosnąco, bez duplikatów.")
    unknown_sets = set(feature_sets) - set(FEATURE_SETS)
    if unknown_sets:
        raise ValueError(f"Nieznane zestawy cech: {sorted(unknown_sets)}")

    X_train, X_test, y_train, y_test = make_fixed_split(data, test_size, split_seed)
    if max(train_sizes) > len(X_train):
        raise ValueError(
            f"Największy trening ma {max(train_sizes)} rekordów, "
            f"ale po podziale dostępnych jest {len(X_train)}."
        )

    rows = []
    for feature_set in feature_sets:
        feature_columns = list(FEATURE_SETS[feature_set])
        absent = set(feature_columns) - set(X_train.columns)
        if absent:
            raise ValueError(f"Brakuje cech {sorted(absent)} w danych wejściowych.")

        for repeat in range(repeats):
            repeat_seed = split_seed + 1009 * repeat
            order = _nested_training_order(y_train, repeat_seed)
            for train_size in train_sizes:
                positions = order[:train_size]
                X_subset = X_train.iloc[positions][feature_columns].copy()
                y_subset = y_train.iloc[positions]
                if feature_set == "degraded":
                    X_subset = _apply_training_degradation(
                        X_subset,
                        degradation_rate,
                        random_state=repeat_seed + train_size,
                    )

                model = build_pipeline(model_name, feature_columns, random_state=repeat_seed)
                model.fit(X_subset, y_subset)
                scores = _score(model, X_test[feature_columns], y_test)
                rows.append({
                    "model": model_name,
                    "feature_set": feature_set,
                    "train_size": train_size,
                    "repeat": repeat + 1,
                    "yes_in_train": int(y_subset.eq("yes").sum()),
                    **scores,
                })

    return pd.DataFrame(rows)


def summarize_experiments(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize average scores and variation across repeated train samples."""
    if results.empty:
        return results.copy()
    summary = results.groupby(
        ["model", "feature_set", "train_size"],
        as_index=False,
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        recall_yes_mean=("recall_yes", "mean"),
        recall_yes_std=("recall_yes", "std"),
    )
    return summary.fillna(0).round(3)


def plot_metric(summary: pd.DataFrame, metric: str = "recall_yes") -> None:
    """Plot mean score by training size, with run-to-run variation bars."""
    if metric not in {"accuracy", "recall_yes"}:
        raise ValueError("metric wybierz jako 'accuracy' albo 'recall_yes'.")
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    if mean_column not in summary:
        raise ValueError("Uruchom najpierw eksperymenty i summarize_experiments().")

    _, axis = plt.subplots(figsize=(8, 4.5))
    for feature_set, group in summary.groupby("feature_set", sort=False):
        group = group.sort_values("train_size")
        axis.errorbar(
            group["train_size"],
            group[mean_column],
            yerr=group[std_column],
            marker="o",
            capsize=3,
            label=feature_set,
        )
    axis.set_title(f"{metric}: wpływ wielkości i jakości danych")
    axis.set_xlabel("Liczba rekordów treningowych")
    axis.set_ylabel(metric)
    axis.legend(title="Zestaw danych")
    axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def fit_explanation_model(
    data: pd.DataFrame,
    model_name: str,
    feature_set: str = "standard",
    train_size: int = 1500,
    split_seed: int = 42,
    degradation_rate: float = 0.15,
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Fit one reproducible model on the shared split for interpretation."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Nieznany zestaw cech '{feature_set}'.")
    X_train, X_test, y_train, y_test = make_fixed_split(data, random_state=split_seed)
    if train_size > len(X_train):
        raise ValueError(f"Dostępnych jest tylko {len(X_train)} wierszy treningowych.")

    feature_columns = list(FEATURE_SETS[feature_set])
    order = _nested_training_order(y_train, split_seed)
    positions = order[:train_size]
    X_subset = X_train.iloc[positions][feature_columns].copy()
    y_subset = y_train.iloc[positions]
    if feature_set == "degraded":
        X_subset = _apply_training_degradation(X_subset, degradation_rate, split_seed + train_size)

    model = build_pipeline(model_name, feature_columns, random_state=split_seed)
    model.fit(X_subset, y_subset)
    return model, X_test[feature_columns], y_test




def drop_features(X: pd.DataFrame, features: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Drop original features from preprocessed data, including all of their 0/1 columns."""
    unknown = set(features) - set(MODEL_FEATURES)
    if unknown:
        raise ValueError(f"Nieznane cechy: {sorted(unknown)}. Wybierz z: {list(MODEL_FEATURES)}")
    to_drop = [
        column for column in X.columns
        if any(column == feature or column.startswith(f"{feature}_") for feature in features)
    ]
    return X.drop(columns=to_drop)


def run_learning_curve(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_sizes: tuple[int, ...] = (100, 300, 1000, 3000, 10000),
    repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train copies of a model on random training samples of growing size, always on the same test set."""
    from sklearn.base import clone
    from sklearn.metrics import balanced_accuracy_score

    sizes = [size for size in train_sizes if size < len(X_train)] + [len(X_train)]
    rng = np.random.default_rng(random_state)
    rows = []
    for size in sizes:
        # Cały zbiór treningowy jest zawsze ten sam, więc wystarczy jeden trening.
        for repeat in range(repeats if size < len(X_train) else 1):
            positions = rng.choice(len(X_train), size=size, replace=False)
            fitted = clone(model).fit(X_train.iloc[positions], y_train.iloc[positions])
            predictions = fitted.predict(X_test)
            rows.append({
                "train_size": size,
                "repeat": repeat + 1,
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            })
    return pd.DataFrame(rows)


def plot_learning_curve(results: pd.DataFrame) -> pd.DataFrame:
    """Plot mean accuracy and balanced accuracy with the min-max band across repeated trainings."""
    summary = results.groupby("train_size")[["accuracy", "balanced_accuracy"]].agg(["mean", "min", "max"])
    series = {
        "accuracy": ("Accuracy", "#2a78d6"),
        "balanced_accuracy": ("Balanced accuracy", "#eb6834"),
    }

    _, axis = plt.subplots(figsize=(9, 5))
    sizes = summary.index.to_numpy()
    for metric, (label, color) in series.items():
        axis.fill_between(sizes, summary[(metric, "min")], summary[(metric, "max")], color=color, alpha=0.18, linewidth=0)
        axis.plot(sizes, summary[(metric, "mean")], color=color, linewidth=2, marker="o", markersize=6, label=label)
        axis.annotate(
            f"{summary[(metric, 'mean')].iloc[-1]:.1%}",
            (sizes[-1], summary[(metric, "mean")].iloc[-1]),
            xytext=(8, 0), textcoords="offset points", va="center", color="#52514e",
        )
    axis.axhline(0.5, color="#52514e", linewidth=1, linestyle="--")
    axis.text(sizes[0], 0.5, " balanced accuracy baseline'u = 50%", va="bottom", color="#52514e", fontsize=9)

    axis.set_xscale("log")
    axis.set_xticks(sizes, [f"{size:,}".replace(",", " ") for size in sizes])
    axis.minorticks_off()
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.set_xlabel("Liczba rekordów treningowych (skala logarytmiczna)")
    axis.set_ylabel("Wynik na zbiorze testowym")
    axis.set_title("Więcej danych treningowych → lepszy i stabilniejszy model\n(linia = średnia, pas = najgorszy i najlepszy trening)")
    axis.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.show()

    table = summary.copy()
    table.columns = [f"{metric}_{stat}" for metric, stat in table.columns]
    return table.round(3)
