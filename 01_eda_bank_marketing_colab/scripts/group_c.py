from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


def _distribution(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].fillna("<NaN>").value_counts(normalize=True)


def compare_column_coverage(
    source_columns: list[str],
    workshop_columns: list[str],
) -> pd.DataFrame:
    """Show which source columns were retained in the workshop dataset."""
    workshop_set = set(workshop_columns)
    result = pd.DataFrame({"kolumna źródłowa": source_columns})
    result["w zestawie warsztatowym"] = result["kolumna źródłowa"].isin(workshop_set)
    result["status"] = result["w zestawie warsztatowym"].map({True: "uwzględniona", False: "pominięta"})
    result = result.sort_values("w zestawie warsztatowym").reset_index(drop=True)
    print(
        f"Zachowano {int(result['w zestawie warsztatowym'].sum())} z "
        f"{len(result)} kolumn źródłowych."
    )
    display(result)
    return result


def categories_absent_from_sample(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    columns: tuple[str, ...] = ("month", "contact", "job", "education", "housing"),
) -> pd.DataFrame:
    """List reference categories that have no observations in the sample."""
    rows = []
    for column in columns:
        if column not in sample.columns or column not in reference.columns:
            continue

        sample_values = sample[column].astype("string").fillna("brak danych")
        reference_values = reference[column].astype("string").fillna("brak danych")
        sample_categories = set(sample_values.unique())
        counts = reference_values.value_counts()

        for category, count in counts.items():
            if category not in sample_categories:
                rows.append({
                    "zmienna": column,
                    "kategoria nieobecna w próbce": category,
                    "liczba w referencji": int(count),
                    "udział w referencji (%)": round(100 * count / len(reference), 2),
                })

    result = pd.DataFrame(rows, columns=[
        "zmienna",
        "kategoria nieobecna w próbce",
        "liczba w referencji",
        "udział w referencji (%)",
    ])
    if result.empty:
        print("Każda sprawdzana kategoria z referencji występuje w próbce.")
    else:
        result = result.sort_values(
            "udział w referencji (%)",
            ascending=False,
        ).reset_index(drop=True)
        display(result)
    return result


def simulate_category_exclusion(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    column: str,
    category: str,
) -> pd.DataFrame:
    """Show how removing one category from the sample changes its outcomes."""
    required = {column, "y"}
    missing_sample = required - set(sample.columns)
    missing_reference = required - set(reference.columns)
    if missing_sample or missing_reference:
        raise ValueError(
            f"Brak kolumn: próbka={sorted(missing_sample)}, "
            f"referencja={sorted(missing_reference)}"
        )

    category_mask = sample[column].astype("string").eq(category).fillna(False)
    excluded_count = int(category_mask.sum())
    if excluded_count == 0:
        observed = sorted(sample[column].dropna().astype(str).unique())
        raise ValueError(
            f"Kategoria '{category}' nie występuje w próbce. Dostępne: {observed}"
        )

    sample_after = sample.loc[~category_mask]
    rows = []
    for name, frame, removed in (
        ("Próbka przed wykluczeniem", sample, 0),
        (f"Próbka bez {column} = {category}", sample_after, excluded_count),
        ("Pełna referencja", reference, 0),
    ):
        valid_target = frame["y"].isin(["no", "yes"])
        outcomes = frame.loc[valid_target, "y"]
        no_count = int(outcomes.eq("no").sum())
        yes_count = int(outcomes.eq("yes").sum())
        target_count = no_count + yes_count
        yes_rate = 100 * yes_count / target_count if target_count else float("nan")
        rows.append({
            "wariant": name,
            "liczba klientów": len(frame),
            "usunięte rekordy": removed,
            "brak lokaty": no_count,
            "lokata": yes_count,
            "odsetek lokat (%)": round(yes_rate, 1) if pd.notna(yes_rate) else float("nan"),
        })

    result = pd.DataFrame(rows)
    reference_rate = result.loc[result["wariant"] == "Pełna referencja", "odsetek lokat (%)"].iloc[0]
    result["różnica do referencji (p.p.)"] = (
        result["odsetek lokat (%)"] - reference_rate
    ).round(1)
    print(
        f"Symulacja wykluczenia kategorii '{category}' z próbki. "
        "Pełna referencja pozostaje bez zmian."
    )
    display(result)
    return result


def compare_category_target_rates(
    sample: pd.DataFrame,
    reference: pd.DataFrame,
    columns: tuple[str, ...] = ("month", "contact", "job"),
) -> pd.DataFrame:
    """Compare within-category term-deposit rates between sample and reference."""
    output_columns = [
        "zmienna",
        "kategoria",
        "klienci w próbce",
        "lokaty w próbce",
        "odsetek lokat w próbce (%)",
        "klienci w referencji",
        "lokaty w referencji",
        "odsetek lokat w referencji (%)",
        "różnica (p.p.)",
    ]
    rows = []

    for column in columns:
        if column not in sample.columns or column not in reference.columns:
            continue

        summaries = {}
        for name, frame in (("sample", sample), ("reference", reference)):
            valid = frame.loc[frame["y"].isin(["no", "yes"]), [column, "y"]].copy()
            valid["kategoria"] = valid[column].astype("string").fillna("brak danych")
            summary = valid.groupby("kategoria")["y"].agg(
                klienci="size",
                lokaty=lambda values: int(values.eq("yes").sum()),
            )
            summary["odsetek"] = 100 * summary["lokaty"] / summary["klienci"]
            summaries[name] = summary

        categories = summaries["sample"].index.union(summaries["reference"].index)
        for category in categories:
            sample_row = summaries["sample"].reindex([category]).iloc[0]
            reference_row = summaries["reference"].reindex([category]).iloc[0]
            sample_rate = sample_row["odsetek"]
            reference_rate = reference_row["odsetek"]
            rows.append({
                "zmienna": column,
                "kategoria": category,
                "klienci w próbce": int(sample_row["klienci"]) if pd.notna(sample_row["klienci"]) else 0,
                "lokaty w próbce": int(sample_row["lokaty"]) if pd.notna(sample_row["lokaty"]) else 0,
                "odsetek lokat w próbce (%)": round(sample_rate, 1) if pd.notna(sample_rate) else float("nan"),
                "klienci w referencji": int(reference_row["klienci"]) if pd.notna(reference_row["klienci"]) else 0,
                "lokaty w referencji": int(reference_row["lokaty"]) if pd.notna(reference_row["lokaty"]) else 0,
                "odsetek lokat w referencji (%)": round(reference_rate, 1) if pd.notna(reference_rate) else float("nan"),
                "różnica (p.p.)": round(sample_rate - reference_rate, 1)
                if pd.notna(sample_rate) and pd.notna(reference_rate)
                else float("nan"),
            })

    result = pd.DataFrame(rows, columns=output_columns)
    if not result.empty:
        result["_bezwzględna różnica"] = result["różnica (p.p.)"].abs()
        result = result.sort_values(
            ["_bezwzględna różnica", "zmienna"],
            ascending=[False, True],
            na_position="last",
        ).drop(columns="_bezwzględna różnica").reset_index(drop=True)
        display(result.round(2))
    return result


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
