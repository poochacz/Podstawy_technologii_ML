from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display


NUMERIC_RULES = {
    "age": lambda s: (s < 0) | (s > 110),
    "duration": lambda s: s < 0,
    "campaign": lambda s: s < 1,
    "pdays": lambda s: s < 0,
}


def _unknown_mask(series: pd.Series) -> pd.Series:
    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return pd.Series(False, index=series.index)

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .eq("unknown")
        .fillna(False)
    )


def missing_values_analysis(df: pd.DataFrame) -> None:
    """Show NaN and 'unknown' counts for every column and affected rows."""
    unknown_masks = pd.DataFrame(
        {column: _unknown_mask(df[column]) for column in df.columns},
        index=df.index,
    )
    nan_masks = df.isna()

    summary = pd.DataFrame({
        "zmienna": df.columns,
        "NaN (wiersze)": nan_masks.sum().to_numpy(),
        "'unknown' (wiersze)": unknown_masks.sum().to_numpy(),
    })
    summary["łącznie w zmiennej"] = (
        summary["NaN (wiersze)"] + summary["'unknown' (wiersze)"]
    )

    rows_with_nan = nan_masks.any(axis=1)
    rows_with_unknown = unknown_masks.any(axis=1)
    rows_with_either = rows_with_nan | rows_with_unknown

    print("Liczba wierszy z co najmniej jednym NaN:", int(rows_with_nan.sum()))
    print("Liczba wierszy z co najmniej jednym 'unknown':", int(rows_with_unknown.sum()))
    print("Liczba wierszy z NaN lub 'unknown' (bez podwójnego liczenia):", int(rows_with_either.sum()))
    display(summary)


def missing_values_for_column(df: pd.DataFrame, column: str) -> None:
    """Show counts of NaN and 'unknown' in one selected column."""
    if column not in df.columns:
        raise ValueError(
            f"Nie znaleziono zmiennej '{column}'. Dostępne zmienne: {list(df.columns)}"
        )

    nan_mask = df[column].isna()
    unknown_mask = _unknown_mask(df[column])
    nan_count = int(nan_mask.sum())
    unknown_count = int(unknown_mask.sum())
    row_count = len(df)
    result = pd.DataFrame({
        "typ wartości": ["NaN", "'unknown'"],
        "liczba wierszy": [nan_count, unknown_count],
        "udział wierszy (%)": [
            round(100 * nan_count / row_count, 1) if row_count else 0.0,
            round(100 * unknown_count / row_count, 1) if row_count else 0.0,
        ],
    })

    print(f"Braki informacji w zmiennej '{column}' (na {row_count} wierszy):")
    display(result)

    nan_sample_count = min(5, nan_count)
    unknown_sample_count = min(5, unknown_count)
    remaining = 10 - nan_sample_count - unknown_sample_count
    nan_remaining = nan_count - nan_sample_count
    unknown_remaining = unknown_count - unknown_sample_count

    if nan_remaining >= unknown_remaining:
        extra_nan = min(remaining, nan_remaining)
        nan_sample_count += extra_nan
        remaining -= extra_nan
        unknown_sample_count += min(remaining, unknown_remaining)
    else:
        extra_unknown = min(remaining, unknown_remaining)
        unknown_sample_count += extra_unknown
        remaining -= extra_unknown
        nan_sample_count += min(remaining, nan_remaining)

    if nan_sample_count + unknown_sample_count == 0:
        print(f"W zmiennej '{column}' nie ma wierszy z NaN ani 'unknown'.")
        return

    nan_examples = df.loc[nan_mask].sample(n=nan_sample_count, random_state=42).copy()
    nan_examples.insert(0, "typ braku", "NaN")
    nan_examples.insert(0, "wiersz", nan_examples.index)

    unknown_examples = df.loc[unknown_mask].sample(
        n=unknown_sample_count,
        random_state=42,
    ).copy()
    unknown_examples.insert(0, "typ braku", "Unknown")
    unknown_examples.insert(0, "wiersz", unknown_examples.index)

    examples = pd.concat([nan_examples, unknown_examples]).sample(
        frac=1,
        random_state=42,
    )
    print("Przykładowe wiersze z brakami w wybranej zmiennej:")
    display(examples)


def anomaly_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """List impossible values and rare numeric extremes for human review."""
    invalid_reasons = {
        "age": "Wiek powinien mieścić się w zakresie 0–110 lat.",
        "duration": "Czas rozmowy nie może być ujemny.",
        "campaign": "Liczba kontaktów w kampanii powinna wynosić co najmniej 1.",
        "pdays": "Liczba dni od poprzedniego kontaktu nie może być ujemna.",
    }
    findings = []

    for column, rule in NUMERIC_RULES.items():
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        invalid_mask = rule(values) & values.notna()

        for row, value in values.loc[invalid_mask].items():
            findings.append({
                "wiersz": row,
                "zmienna": column,
                "wartość": value,
                "sygnał": "wartość niemożliwa",
                "dlaczego": invalid_reasons[column],
            })

        typical_values = values.loc[~invalid_mask & values.notna()]
        if column == "pdays":
            typical_values = typical_values.loc[typical_values.ne(999)]
        if typical_values.empty:
            continue

        low, high = typical_values.quantile([0.01, 0.99])
        extreme_mask = (
            values.lt(low) | values.gt(high)
        ) & values.notna() & ~invalid_mask
        if column == "pdays":
            extreme_mask &= values.ne(999)

        for row, value in values.loc[extreme_mask].items():
            findings.append({
                "wiersz": row,
                "zmienna": column,
                "wartość": value,
                "sygnał": "rzadka skrajna wartość",
                "dlaczego": (
                    f"Wartość leży poza zakresem około 98% poprawnych "
                    f"obserwacji ({low:.1f}–{high:.1f}); może być prawidłowa."
                ),
            })

    columns = ["wiersz", "zmienna", "wartość", "sygnał", "dlaczego"]
    result = pd.DataFrame(findings, columns=columns)
    if result.empty:
        print("Nie znaleziono wartości do omówienia.")
    else:
        result = result.sort_values(["zmienna", "wiersz"]).reset_index(drop=True)
    return result


def sample_anomaly_candidates(
    candidates: pd.DataFrame,
    sample_size: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Select a reproducible sample of candidates across all variables."""
    if sample_size < 0:
        raise ValueError("sample_size nie może być ujemne.")
    if len(candidates) <= sample_size:
        return candidates.copy().reset_index(drop=True)

    return (
        candidates.sample(n=sample_size, random_state=random_state)
        .sort_values(["zmienna", "wiersz"])
        .reset_index(drop=True)
    )


def display_anomaly_candidates(
    candidates: pd.DataFrame,
    include_row: bool = True,
) -> None:
    """Display candidate values without revealing their classifications."""
    if candidates.empty:
        print("Nie znaleziono kandydatów dla tej zmiennej.")
        return

    columns = ["wiersz", "zmienna", "wartość"] if include_row else ["zmienna", "wartość"]
    table = candidates[columns].to_html(
        index=False,
        escape=True,
        classes="anomaly-candidates",
    )
    style = """
    <style>
    table.anomaly-candidates { width: 100%; table-layout: fixed; }
    table.anomaly-candidates th, table.anomaly-candidates td {
        white-space: normal; overflow-wrap: anywhere; vertical-align: top;
    }
    table.anomaly-candidates th:nth-child(1), table.anomaly-candidates td:nth-child(1) { width: 12%; }
    table.anomaly-candidates th:nth-child(2), table.anomaly-candidates td:nth-child(2) { width: 30%; }
    table.anomaly-candidates th:nth-child(3), table.anomaly-candidates td:nth-child(3) { width: 58%; }
    </style>
    """
    display(HTML(style + table))


def display_anomaly_rows(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    """Show all source columns for a small set of candidate rows."""
    if candidates.empty:
        print("Brak przykładowych wierszy do wyświetlenia.")
        return

    rows = []
    for candidate in candidates.itertuples(index=False):
        if candidate.wiersz not in df.index:
            raise ValueError(f"Nie znaleziono wiersza {candidate.wiersz} w danych.")
        row = {
            "wiersz": candidate.wiersz,
            "zmienna_odstająca": candidate.zmienna,
            "wartość_odstająca": candidate.wartość,
        }
        row.update(df.loc[candidate.wiersz].to_dict())
        rows.append(row)

    print("Pełne dane dla wybranych przykładów (maksymalnie 10 wierszy):")
    display(pd.DataFrame(rows))


def save_anomaly_decisions(
    candidates: pd.DataFrame,
    decisions: dict[tuple[int, str], str],
    reviewer: str,
    output_path: str | Path,
) -> Path:
    """Save one reviewer's keep/remove decisions for anomaly candidates."""
    if not reviewer.strip():
        raise ValueError("Podaj identyfikator osoby lub grupy oceniającej.")

    allowed_decisions = {"zostaw", "usuń"}
    invalid_decisions = set(decisions.values()) - allowed_decisions
    if invalid_decisions:
        raise ValueError(
            f"Decyzje muszą mieć wartość 'zostaw' albo 'usuń': {invalid_decisions}"
        )

    candidate_keys = set(zip(candidates["wiersz"], candidates["zmienna"]))
    unknown_keys = set(decisions) - candidate_keys
    if unknown_keys:
        raise ValueError(f"Decyzje dotyczą przypadków spoza tabeli: {unknown_keys}")

    result = candidates.copy()
    result["oceniający"] = reviewer.strip()
    result["decyzja"] = "nieocenione"
    for (row, column), decision in decisions.items():
        mask = result["wiersz"].eq(row) & result["zmienna"].eq(column)
        result.loc[mask, "decyzja"] = decision

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Zapisano decyzje: {path}")
    return path


def show_pdays_distribution(df: pd.DataFrame) -> None:
    counts = df["pdays"].value_counts(dropna=False).head(12)
    display(counts.to_frame("liczba"))

    numeric_pdays = pd.to_numeric(df["pdays"], errors="coerce")
    plt.figure(figsize=(7, 3.8))
    numeric_pdays.dropna().clip(upper=1000).plot(kind="hist", bins=30)
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


def categorical_target_report(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("housing", "contact"),
    target: str = "y",
    positive: str = "yes",
) -> pd.DataFrame:
    """Compare term-deposit rates across values of categorical variables."""
    required = [*columns, target]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Brakuje wymaganych kolumn: {missing}")

    rows = []
    for column in columns:
        grouped_data = pd.DataFrame({
            "kategoria": df[column].astype("object").where(df[column].notna(), "brak danych"),
            "wynik": df[target],
        })
        for category, group in grouped_data.groupby("kategoria", sort=False):
            target_known = int(group["wynik"].notna().sum())
            positive_count = int(group["wynik"].eq(positive).sum())
            rows.append({
                "zmienna": column,
                "wartość kategorii": category,
                "liczba klientów": len(group),
                "liczba założonych lokat": positive_count,
                "odsetek klientów z lokatą (%)": (
                    round(100 * positive_count / target_known, 1)
                    if target_known
                    else float("nan")
                ),
            })

    result = pd.DataFrame(rows).sort_values(
        ["zmienna", "odsetek klientów z lokatą (%)"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)
    print(
        "Porównanie odsetka klientów, którzy założyli lokatę (y='yes'), "
        "w obrębie każdej kategorii:"
    )
    display(result)
    return result


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


def age_target_analysis(df: pd.DataFrame, target: str = "y") -> pd.DataFrame:
    """Compare term-deposit outcomes for missing ages and ages 35–45."""
    required = {"age", target}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Brakuje wymaganych kolumn: {sorted(missing_columns)}")

    age = pd.to_numeric(df["age"], errors="coerce")
    valid_age = age.where(age.between(0, 110))
    target_binary = df[target].map({"no": 0, "yes": 1})
    correlation_data = pd.DataFrame({
        "wiek": valid_age,
        "lokata_yes": target_binary,
    }).dropna()

    correlation = (
        correlation_data["wiek"].corr(correlation_data["lokata_yes"])
        if len(correlation_data) >= 2
        else float("nan")
    )
    print("Wynik y='yes' oznacza założenie lokaty terminowej.")
    if pd.notna(correlation):
        print(
            "Korelacja wieku z założeniem lokaty "
            f"(Pearson r, wiek 0–110, n={len(correlation_data)}): {correlation:.3f}"
        )
    else:
        print("Nie można obliczyć korelacji wieku z założeniem lokaty dla tych danych.")
    print("Wartość r opisuje współwystępowanie, nie dowodzi związku przyczynowego.")

    groups = {
        "Brakujący wiek": age.isna(),
        "Wiek 35–45 lat": age.between(35, 45, inclusive="both"),
    }
    rows = []
    for group_name, group_mask in groups.items():
        target_values = df.loc[group_mask, target]
        known_target = target_values.isin(["yes", "no"])
        yes_count = int(target_values.eq("yes").sum())
        no_count = int(target_values.eq("no").sum())
        known_count = yes_count + no_count
        rows.append({
            "grupa klientów": group_name,
            "liczba klientów": int(group_mask.sum()),
            "wynik znany (yes/no)": known_count,
            "założyli lokatę (yes)": yes_count,
            "nie założyli lokaty (no)": no_count,
            "odsetek lokat (%)": (
                round(100 * yes_count / known_count, 1)
                if known_count
                else float("nan")
            ),
        })

    result = pd.DataFrame(rows)
    print("Decyzje dotyczące lokaty: brak wieku porównany z osobami w wieku 35–45 lat.")
    display(result)
    return result
