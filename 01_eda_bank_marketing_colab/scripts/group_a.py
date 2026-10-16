from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


NUMERIC_RULES = {
    "age": lambda s: (s < 0) | (s > 110),
    "duration": lambda s: s < 0,
    "campaign": lambda s: s < 1,
    "pdays": lambda s: s < 0,
}


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        nan = int(df[col].isna().sum())
        unknown = int(df[col].astype(str).str.lower().eq("unknown").sum())
        rows.append({
            "kolumna": col,
            "NaN": nan,
            "wartość_'unknown'": unknown,
            "razem_podejrzane": nan + unknown,
        })
    result = pd.DataFrame(rows).sort_values("razem_podejrzane", ascending=False)
    display(result)
    return result


def invalid_values_report(df: pd.DataFrame) -> pd.DataFrame:
    findings = []
    for col, rule in NUMERIC_RULES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        mask = rule(s) & s.notna()
        for idx, value in df.loc[mask, col].items():
            findings.append({"wiersz": idx, "kolumna": col, "wartość": value})
    result = pd.DataFrame(findings)
    if result.empty:
        print("Nie znaleziono wartości łamiących proste reguły domenowe.")
    else:
        display(result)
    return result


def extreme_values_report(df: pd.DataFrame) -> None:
    for col in ["age", "duration", "campaign", "pdays"]:
        if col not in df.columns:
            continue
        print(f"\n### {col}")
        s = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        print(f"Reguła IQR sugeruje wartości odstające poza zakresem: {low:.2f} – {high:.2f}")
        print("Najmniejsze wartości:")
        display(df[[col]].sort_values(col).head(8).T)
        print("Największe wartości:")
        display(df[[col]].sort_values(col, ascending=False).head(8).T)


def show_pdays_distribution(df: pd.DataFrame) -> None:
    counts = df["pdays"].value_counts(dropna=False).head(12)
    display(counts.to_frame("liczba"))

    plt.figure(figsize=(7, 3.8))
    df["pdays"].dropna().clip(upper=1000).plot(kind="hist", bins=30)
    plt.title("Rozkład pdays")
    plt.xlabel("pdays")
    plt.ylabel("Liczba obserwacji")
    plt.tight_layout()
    plt.show()


def categorical_report(df: pd.DataFrame) -> None:
    for col in ["job", "education", "housing", "contact", "month"]:
        print(f"\n### {col}")
        result = pd.DataFrame({
            "liczba": df[col].fillna("<NaN>").value_counts(),
            "udział": df[col].fillna("<NaN>").value_counts(normalize=True).round(4),
        })
        display(result)


def compare_age_imputation(df: pd.DataFrame) -> pd.DataFrame:
    age = pd.to_numeric(df["age"], errors="coerce")
    valid = age.where((age >= 0) & (age <= 110))

    variants = {
        "oryginał_bez_braków": valid.dropna(),
        "średnia": valid.fillna(valid.mean()),
        "mediana": valid.fillna(valid.median()),
    }
    rows = []
    for name, series in variants.items():
        rows.append({
            "wariant": name,
            "n": int(series.notna().sum()),
            "średnia": float(series.mean()),
            "mediana": float(series.median()),
            "odchylenie_std": float(series.std()),
        })
    result = pd.DataFrame(rows).round(2)
    display(result)
    return result
