from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from scripts.data_setup import MODEL_FEATURES

TARGET = "y"
CATEGORICAL_FEATURES = {"job", "education", "housing", "contact", "month"}


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


def _pl(value: float, fmt: str = ".2f") -> str:
    """Format a number with a Polish decimal comma."""
    return format(value, fmt).replace(".", ",")


def _style_axis(axis) -> None:
    axis.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def plot_regression_predictions(linear_predictions, logistic_probabilities) -> None:
    """Compare linear-regression outputs (any number) with logistic-regression probabilities (0-1)."""
    linear_predictions = np.asarray(linear_predictions)
    logistic_probabilities = np.asarray(logistic_probabilities)
    outside = np.mean((linear_predictions < 0) | (linear_predictions > 1))
    low = min(-0.2, linear_predictions.min() - 0.05)
    high = max(1.2, linear_predictions.max() + 0.05)
    bins = np.linspace(low, high, 50)

    _, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    panels = (
        (axes[0], linear_predictions, f"Regresja liniowa: {_pl(outside, '.1%')} wyników poza przedziałem 0–1"),
        (axes[1], logistic_probabilities, "Regresja logistyczna: wszystkie wyniki w przedziale 0–1"),
    )
    for axis, values, title in panels:
        axis.axvspan(low, 0, color="#52514e", alpha=0.12, linewidth=0)
        axis.axvspan(1, high, color="#52514e", alpha=0.12, linewidth=0)
        axis.hist(values, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.5)
        axis.set_title(title)
        axis.set_xlabel("Przewidywana „szansa” na lokatę")
        _style_axis(axis)
    axes[0].set_ylabel("Liczba klientów testowych")
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.figtext(0.5, 0.01, "Szare pola: wartości poniżej 0 i powyżej 1 – niemożliwe jako prawdopodobieństwo", ha="center", color="#52514e")
    plt.show()


def threshold_metrics(y_true: pd.Series, probabilities, min_predicted_yes: int = 30) -> pd.DataFrame:
    """Score yes/no predictions for each threshold; precision is hidden when too few clients get yes."""
    is_yes = np.asarray(y_true) == "yes"
    thresholds = np.round(np.arange(0.02, 0.96, 0.01), 2)
    said_yes = np.asarray(probabilities)[None, :] >= thresholds[:, None]
    true_pos = (said_yes & is_yes).sum(axis=1)
    true_neg = (~said_yes & ~is_yes).sum(axis=1)
    predicted_yes = said_yes.sum(axis=1)
    recall = true_pos / is_yes.sum()
    specificity = true_neg / (~is_yes).sum()
    precision = np.where(predicted_yes >= min_predicted_yes, true_pos / np.maximum(predicted_yes, 1), np.nan)
    return pd.DataFrame({
        "próg": thresholds,
        "accuracy": (true_pos + true_neg) / len(is_yes),
        "balanced_accuracy": (recall + specificity) / 2,
        "recall_yes": recall,
        "precision_yes": precision,
    })


def plot_threshold_metrics(y_true: pd.Series, probabilities) -> None:
    """Plot how accuracy, balanced accuracy, recall and precision change with the threshold."""
    metrics = threshold_metrics(y_true, probabilities)
    series = {
        "accuracy": ("Accuracy", "#2a78d6"),
        "balanced_accuracy": ("Balanced accuracy", "#eb6834"),
        "recall_yes": ("Recall (ilu chętnych znaleźliśmy)", "#1baf7a"),
        "precision_yes": ("Precision (ilu z wskazanych to chętni)", "#eda100"),
    }

    _, axis = plt.subplots(figsize=(10, 5))
    for column, (label, color) in series.items():
        axis.plot(metrics["próg"], metrics[column], color=color, linewidth=2, label=label)
    axis.axvline(0.5, color="#52514e", linewidth=1, linestyle="--")
    axis.text(0.505, 0.02, "domyślny próg 0,5", color="#52514e", fontsize=9)
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.set_xlabel("Próg decyzyjny: od jakiej szansy model mówi „yes”")
    axis.set_ylabel("Wynik na zbiorze testowym")
    axis.set_title("Jak próg decyzyjny zmienia wyniki modelu")
    _style_axis(axis)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    plt.tight_layout()
    plt.show()


def plot_profit_by_threshold(
    y_true: pd.Series,
    probabilities,
    call_cost: float,
    deposit_profit: float,
) -> pd.DataFrame:
    """Plot bank profit for each threshold: every call costs money, every deposit earns money."""
    is_yes = np.asarray(y_true) == "yes"
    probabilities = np.asarray(probabilities)
    thresholds = np.round(np.arange(0.01, 0.96, 0.01), 2)

    def outcome(called: np.ndarray) -> dict[str, float]:
        calls, deposits = int(called.sum()), int((called & is_yes).sum())
        return {"telefony": calls, "lokaty": deposits, "zysk [zł]": deposits * deposit_profit - calls * call_cost}

    profits = np.array([outcome(probabilities >= threshold)["zysk [zł]"] for threshold in thresholds])
    best = int(np.argmax(profits))
    call_everyone = outcome(np.ones_like(is_yes))["zysk [zł]"]

    _, axis = plt.subplots(figsize=(10, 5))
    axis.plot(thresholds, profits, color="#2a78d6", linewidth=2, label="zysk przy danym progu")
    axis.axhline(call_everyone, color="#eb6834", linewidth=1.5, linestyle="--", label="dzwonimy do wszystkich")
    axis.axhline(0, color="#52514e", linewidth=0.8)
    axis.axvline(0.5, color="#52514e", linewidth=1, linestyle=":", label="domyślny próg 0,5")
    axis.plot(thresholds[best], profits[best], "o", color="#0b0b0b", markersize=8)
    axis.annotate(
        f"najlepszy próg {_pl(thresholds[best])}\nzysk {profits[best]:,.0f} zł".replace(",", " "),
        (thresholds[best], profits[best]),
        xytext=(25, -10),
        textcoords="offset points",
        va="top",
    )
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:,.0f} zł".replace(",", " ")))
    axis.set_xlabel("Próg decyzyjny: dzwonimy, gdy szansa na lokatę jest co najmniej taka")
    axis.set_ylabel("Zysk banku na klientach testowych")
    axis.set_title(f"Telefon kosztuje {call_cost:g} zł, lokata przynosi {deposit_profit:g} zł")
    _style_axis(axis)
    axis.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(
        [
            outcome(np.ones_like(is_yes)),
            outcome(probabilities >= 0.5),
            outcome(probabilities >= thresholds[best]),
        ],
        index=["dzwonimy do wszystkich", "próg 0,5", f"najlepszy próg {_pl(thresholds[best])}"],
    )
