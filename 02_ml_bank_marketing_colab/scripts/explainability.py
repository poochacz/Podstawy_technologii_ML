from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree


class _NoPreprocessing:
    """Stand-in preprocessor for models trained directly on already numeric data."""

    def __init__(self, feature_names):
        self._feature_names = list(feature_names)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self._feature_names].astype(float)

    def get_feature_names_out(self):
        return self._feature_names


def _as_array(transformed) -> np.ndarray:
    return np.asarray(transformed, dtype=float)


def _parts(model):
    return _NoPreprocessing(model.feature_names_in_), model


def plot_logistic_coefficients(model: LogisticRegression, top_n: int = 12) -> pd.DataFrame:
    """Plot the strongest logistic-regression coefficients for class yes."""
    if not isinstance(model, LogisticRegression):
        raise TypeError("Ta wizualizacja wymaga Logistic Regression.")

    # coef_ dotyczy klasy model.classes_[1], czyli "yes"
    result = pd.DataFrame({"cecha": model.feature_names_in_, "współczynnik": model.coef_[0]})
    result = result.reindex(result["współczynnik"].abs().sort_values().tail(top_n).index)
    colors = ["#2a78d6" if value < 0 else "#eb6834" for value in result["współczynnik"]]

    _, axis = plt.subplots(figsize=(9, max(4, 0.38 * len(result))))
    axis.barh(result["cecha"], result["współczynnik"], color=colors)
    axis.axvline(0, color="#52514e", linewidth=0.8)
    axis.set_title(f"Regresja logistyczna: {top_n} cech o największym wpływie na szansę lokaty")
    axis.set_xlabel("← zmniejsza szansę na „yes”          zwiększa szansę na „yes” →")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    plt.tight_layout()
    plt.show()
    return result.sort_values("współczynnik", key=lambda values: values.abs(), ascending=False).reset_index(drop=True).round(3)


def _plot_tree_estimator(
    classifier: DecisionTreeClassifier,
    feature_names: list[str],
    max_depth: int,
    title: str,
    class_names: list[str] | None = None,
) -> None:
    _, axis = plt.subplots(figsize=(20, 10))
    plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=class_names or [str(value) for value in classifier.classes_],
        max_depth=max_depth,
        filled=True,
        rounded=True,
        proportion=True,
        impurity=False,
        ax=axis,
        fontsize=8,
    )
    axis.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_decision_tree(model: DecisionTreeClassifier, max_depth: int = 3) -> None:
    """Plot the fitted decision tree, limiting depth for readability."""
    preprocessor, classifier = _parts(model)
    if not isinstance(classifier, DecisionTreeClassifier):
        raise TypeError("Ta wizualizacja wymaga Decision Tree.")
    _plot_tree_estimator(
        classifier,
        list(preprocessor.get_feature_names_out()),
        max_depth,
        "Decision Tree: przykładowe reguły przewidywania",
    )


def plot_random_forest_tree(
    model: RandomForestClassifier,
    tree_index: int = 0,
    max_depth: int = 3,
) -> None:
    """Plot one shallow tree from a fitted random forest."""
    preprocessor, classifier = _parts(model)
    if not isinstance(classifier, RandomForestClassifier):
        raise TypeError("Ta wizualizacja wymaga Random Forest.")
    if not 0 <= tree_index < len(classifier.estimators_):
        raise ValueError(f"tree_index musi być w zakresie 0–{len(classifier.estimators_) - 1}.")
    _plot_tree_estimator(
        classifier.estimators_[tree_index],
        list(preprocessor.get_feature_names_out()),
        max_depth,
        f"Random Forest: przykładowe drzewo nr {tree_index}",
        class_names=[str(value) for value in classifier.classes_],
    )


def forest_first_questions(model: RandomForestClassifier) -> pd.DataFrame:
    """Count which feature each tree of a fitted random forest asks about first."""
    preprocessor, classifier = _parts(model)
    if not isinstance(classifier, RandomForestClassifier):
        raise TypeError("Ta tabela wymaga Random Forest.")
    feature_names = preprocessor.get_feature_names_out()
    roots = [feature_names[tree.tree_.feature[0]] for tree in classifier.estimators_ if tree.tree_.feature[0] >= 0]
    counts = pd.Series(roots).value_counts()
    return pd.DataFrame({"pierwsze pytanie drzewa": counts.index, "liczba drzew": counts.values})


def plot_forest_votes(model: RandomForestClassifier, row: pd.Series | pd.DataFrame) -> None:
    """Plot how sure each tree of a fitted random forest is that one client answers yes."""
    preprocessor, classifier = _parts(model)
    if not isinstance(classifier, RandomForestClassifier):
        raise TypeError("Głosowanie wymaga Random Forest.")
    row_frame = row.to_frame().T if isinstance(row, pd.Series) else row.copy()
    transformed = preprocessor.transform(row_frame)
    yes_index = list(classifier.classes_).index("yes")
    # Pojedyncze drzewa lasu znają klasy jako 0/1 (w kolejności classifier.classes_) i nie znają nazw kolumn.
    values = _as_array(transformed)
    tree_yes = np.sort([tree.predict_proba(values)[0, yes_index] for tree in classifier.estimators_])
    forest_yes = float(classifier.predict_proba(transformed)[0, yes_index])
    prediction = classifier.predict(transformed)[0]
    votes_yes = int((tree_yes > 0.5).sum())

    _, axis = plt.subplots(figsize=(10, 4.5))
    positions = np.arange(1, len(tree_yes) + 1)
    says_yes = tree_yes > 0.5
    axis.bar(positions[~says_yes], tree_yes[~says_yes], width=0.8, color="#2a78d6", label="drzewo mówi „nie”")
    axis.bar(positions[says_yes], tree_yes[says_yes], width=0.8, color="#eb6834", label="drzewo mówi „tak”")
    axis.axhline(0.5, color="#52514e", linewidth=1, linestyle="--", label="próg 0,5")
    axis.axhline(forest_yes, color="#0b0b0b", linewidth=2, label=f"średnia pewność lasu = {forest_yes:.2f}")
    axis.set_ylim(0, 1)
    axis.set_xlim(0, len(tree_yes) + 1)
    axis.set_xlabel("Drzewa w lesie (posortowane od najmniej do najbardziej pewnych „tak”)")
    axis.set_ylabel("Pewność drzewa, że „tak”")
    axis.set_title(
        f"{votes_yes} z {len(tree_yes)} drzew mówi „tak”; średnia pewność {forest_yes:.2f} "
        f"→ las odpowiada „{'tak' if prediction == 'yes' else 'nie'}”"
    )
    axis.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.show()
