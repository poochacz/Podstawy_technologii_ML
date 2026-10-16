from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


def _distribution(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna("<NaN>").value_counts(normalize=True)


def compare_category(sample: pd.DataFrame, reference: pd.DataFrame, col: str, top_n: int = 15) -> pd.DataFrame:
    s = _distribution(sample, col)
    r = _distribution(reference, col)
    categories = r.head(top_n).index.union(s.head(top_n).index)
    result = pd.DataFrame({
        "próbka_2000": s.reindex(categories, fill_value=0),
        "pełny_zbiór": r.reindex(categories, fill_value=0),
    })
    result["różnica_pp"] = (result["próbka_2000"] - result["pełny_zbiór"]) * 100
    result = result.sort_values("różnica_pp", key=lambda x: x.abs(), ascending=False)
    display(result.round(4))

    plot = result[["próbka_2000", "pełny_zbiór"]].copy() * 100
    plt.figure(figsize=(8, 4.2))
    plot.sort_values("pełny_zbiór").plot(kind="barh", ax=plt.gca())
    plt.title(f"Reprezentatywność: {col}")
    plt.xlabel("Udział [%]")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()
    return result


def compare_age(sample: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    def bands(df):
        age = pd.to_numeric(df["age"], errors="coerce")
        return pd.cut(
            age,
            bins=[0, 29, 39, 49, 59, 69, 120],
            labels=["<30", "30–39", "40–49", "50–59", "60–69", "70+"],
        )
    s = bands(sample).value_counts(normalize=True, sort=False)
    r = bands(reference).value_counts(normalize=True, sort=False)
    result = pd.DataFrame({"próbka_2000": s, "pełny_zbiór": r})
    result["różnica_pp"] = (result["próbka_2000"] - result["pełny_zbiór"]) * 100
    display(result.round(4))

    plt.figure(figsize=(7, 3.8))
    (result[["próbka_2000", "pełny_zbiór"]] * 100).plot(kind="bar", ax=plt.gca())
    plt.title("Struktura wieku: próbka vs pełny zbiór")
    plt.xlabel("Grupa wieku")
    plt.ylabel("Udział [%]")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return result


def compare_target(sample: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    s = sample["y"].value_counts(normalize=True)
    r = reference["y"].value_counts(normalize=True)
    result = pd.DataFrame({"próbka_2000": s, "pełny_zbiór": r})
    result["różnica_pp"] = (result["próbka_2000"] - result["pełny_zbiór"]) * 100
    display(result.round(4))
    return result


def largest_gaps(sample: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["month", "contact", "job"]:
        s = _distribution(sample, col)
        r = _distribution(reference, col)
        for category in s.index.union(r.index):
            rows.append({
                "kolumna": col,
                "kategoria": category,
                "próbka": s.get(category, 0),
                "pełny_zbiór": r.get(category, 0),
                "różnica_pp": (s.get(category, 0) - r.get(category, 0)) * 100,
            })
    result = pd.DataFrame(rows)
    result["abs"] = result["różnica_pp"].abs()
    result = result.sort_values("abs", ascending=False).drop(columns="abs").head(15)
    display(result.round(4))
    return result
