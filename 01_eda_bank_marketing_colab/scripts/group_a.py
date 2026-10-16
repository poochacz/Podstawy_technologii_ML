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


def missing_values_analysis(df: pd.DataFrame, target: str = "y", examples: int = 4):
    """
    Analiza różnych sposobów reprezentowania brakujących informacji:
    - NaN: techniczny brak wartości
    - 'unknown': jawna informacja, że wartość jest nieznana
    - known: znana wartość

    Dla każdej kolumny pokazuje:
    1. liczbę NaN i 'unknown',
    2. przykładowe rekordy,
    3. udział targetu y='yes' w każdej grupie.
    """

    summary = []

    for col in df.columns:
        if col == target:
            continue

        nan_mask = df[col].isna()

        if pd.api.types.is_object_dtype(df[col]):
            unknown_mask = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("unknown")
                & ~nan_mask
            )
        else:
            unknown_mask = pd.Series(False, index=df.index)

        suspicious = nan_mask.sum() + unknown_mask.sum()

        if suspicious == 0:
            continue

        summary.append({
            "kolumna": col,
            "NaN": int(nan_mask.sum()),
            "unknown": int(unknown_mask.sum()),
            "łącznie": int(suspicious),
            "udział_podejrzanych": round(suspicious / len(df), 3)
        })

    summary_df = pd.DataFrame(summary)

    print("=== 1. Gdzie występują braki lub wartości 'unknown'? ===")
    if summary_df.empty:
        print("Nie znaleziono braków ani wartości 'unknown'.")
        return

    display(
        summary_df.sort_values("łącznie", ascending=False)
    )

    for col in summary_df["kolumna"]:

        print("\n" + "=" * 70)
        print(f"KOLUMNA: {col}")
        print("=" * 70)

        nan_mask = df[col].isna()

        if pd.api.types.is_object_dtype(df[col]):
            unknown_mask = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("unknown")
                & ~nan_mask
            )
        else:
            unknown_mask = pd.Series(False, index=df.index)

        known_mask = ~(nan_mask | unknown_mask)

        # ---------------------------------------------------------
        # Przykładowe rekordy
        # ---------------------------------------------------------

        if nan_mask.any():
            print("\nPrzykładowe rekordy z NaN:")
            display(
                df.loc[nan_mask]
                .head(examples)
            )

        if unknown_mask.any():
            print("\nPrzykładowe rekordy z 'unknown':")
            display(
                df.loc[unknown_mask]
                .head(examples)
            )

        # ---------------------------------------------------------
        # Czy rodzaj braku wiąże się z targetem?
        # ---------------------------------------------------------

        groups = []

        for name, mask in [
            ("NaN", nan_mask),
            ("unknown", unknown_mask),
            ("znana wartość", known_mask),
        ]:
            subset = df.loc[mask]

            if len(subset) == 0:
                continue

            yes_rate = (
                subset[target]
                .eq("yes")
                .mean()
            )

            groups.append({
                "status": name,
                "liczba_rekordów": len(subset),
                "udział_y_yes": round(yes_rate, 3)
            })

        print("\nCzy sposób braku informacji wiąże się z targetem?")
        display(pd.DataFrame(groups))

        print("\n" + "=" * 70)
        print("PYTANIA DO DYSKUSJI")
        print("=" * 70)

        print("""
                1. Czy NaN i 'unknown' powstały z tego samego powodu?

                2. Czy 'unknown' może być informacją samą w sobie?
                Np. czy klienci, dla których bank nie zna wykształcenia,
                zachowują się inaczej niż pozostali?

                3. Czy powinniśmy:
                - usunąć takie rekordy,
                - uzupełnić wartość,
                - czy pozostawić 'unknown' jako osobną kategorię?

                4. Czy sposób powstawania braków może powodować bias?
                """)


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
