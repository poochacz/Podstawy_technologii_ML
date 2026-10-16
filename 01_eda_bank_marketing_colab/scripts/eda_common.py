from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


NUMERIC_COLUMNS = ["age", "duration", "campaign", "pdays"]
CATEGORICAL_COLUMNS = ["job", "education", "housing", "contact", "month", "y"]


def dataset_overview(df: pd.DataFrame) -> None:
    print(f"Liczba rekordów: {len(df):,}")
    print(f"Liczba kolumn: {df.shape[1]}")
    print("\nPierwsze rekordy:")
    display(df.head(8))

    summary = pd.DataFrame({
        "typ": df.dtypes.astype(str),
        "braki_NaN": df.isna().sum(),
        "unikalne_wartosci": df.nunique(dropna=True),
    })
    print("\nPodstawowe informacje o kolumnach:")
    display(summary)


def numeric_summary(df: pd.DataFrame) -> None:
    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    display(numeric.describe().T)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        plt.figure(figsize=(7, 3.8))
        values.plot(kind="hist", bins=30)
        plt.title(f"Rozkład: {col}")
        plt.xlabel(col)
        plt.ylabel("Liczba obserwacji")
        plt.tight_layout()
        plt.show()


def plot_categorical_distributions(df: pd.DataFrame, top_n: int = 15) -> None:
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        counts = df[col].fillna("<NaN>").value_counts().head(top_n)
        plt.figure(figsize=(7, 3.8))
        counts.sort_values().plot(kind="barh")
        plt.title(f"Najczęstsze wartości: {col}")
        plt.xlabel("Liczba obserwacji")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()


def target_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["y"].value_counts(dropna=False)
    result = pd.DataFrame({
        "liczba": counts,
        "udział": (counts / len(df)).round(4),
    })
    display(result)

    plt.figure(figsize=(5, 3.6))
    counts.plot(kind="bar")
    plt.title("Rozkład klasy docelowej y")
    plt.xlabel("Czy klient założył lokatę?")
    plt.ylabel("Liczba obserwacji")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return result
