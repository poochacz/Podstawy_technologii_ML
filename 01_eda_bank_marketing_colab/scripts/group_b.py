from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


NUMERIC_FEATURES = {"age", "duration", "campaign", "pdays"}


def target_balance(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["y"].value_counts()
    result = pd.DataFrame({
        "liczba": counts,
        "udział": (counts / len(df)).round(4),
    })
    display(result)

    plt.figure(figsize=(5, 3.6))
    counts.plot(kind="bar")
    plt.title("Balans klas")
    plt.xlabel("y")
    plt.ylabel("Liczba obserwacji")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return result


def naive_baseline(df: pd.DataFrame) -> dict:
    majority = df["y"].value_counts().idxmax()
    accuracy = float((df["y"] == majority).mean())
    result = {"zawsze_przewiduj": majority, "accuracy": accuracy}
    print(f"Najczęstsza klasa: {majority}")
    print(f"Accuracy modelu, który zawsze przewiduje '{majority}': {accuracy:.1%}")
    return result


def feature_differences_by_target(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric and categorical feature differences by target class."""
    if "y" not in df.columns:
        raise ValueError("Brakuje kolumny targetu 'y'.")

    target_mask = df["y"].isin(["no", "yes"])
    target = df.loc[target_mask, "y"]
    rows = []

    for feature in df.columns:
        if feature == "y":
            continue

        if feature in NUMERIC_FEATURES:
            values = pd.to_numeric(df.loc[target_mask, feature], errors="coerce")
            group_no = values.loc[target.eq("no")].dropna()
            group_yes = values.loc[target.eq("yes")].dropna()
            if group_no.empty or group_yes.empty:
                continue

            mean_no = float(group_no.mean())
            mean_yes = float(group_yes.mean())
            rows.append({
                "cecha": feature,
                "typ": "liczbowa",
                "porównanie": "średnia wartość",
                "wynik dla no": round(mean_no, 2),
                "wynik dla yes": round(mean_yes, 2),
                "różnica (yes - no)": round(mean_yes - mean_no, 2),
                "liczba no / yes": f"{len(group_no)} / {len(group_yes)}",
            })
            continue

        categories = (
            df.loc[target_mask, feature]
            .astype("string")
            .fillna("brak danych")
        )
        shares = pd.crosstab(categories, target, normalize="columns")
        shares = shares.reindex(columns=["no", "yes"], fill_value=0)
        if shares.empty:
            continue

        differences = shares["yes"].sub(shares["no"]).fillna(0)
        category = differences.abs().idxmax()
        share_no = float(shares.loc[category, "no"] * 100)
        share_yes = float(shares.loc[category, "yes"] * 100)
        rows.append({
            "cecha": feature,
            "typ": "kategoryczna",
            "porównanie": f"udział kategorii '{category}' (%)",
            "wynik dla no": round(share_no, 1),
            "wynik dla yes": round(share_yes, 1),
            "różnica (yes - no)": round(share_yes - share_no, 1),
            "liczba no / yes": (
                f"{int(target.eq('no').sum())} / {int(target.eq('yes').sum())}"
            ),
        })

    return pd.DataFrame(rows)


def categorical_outcome_rates(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("job", "education", "housing", "contact", "month"),
) -> pd.DataFrame:
    """Show yes/no counts and deposit rate within each category."""
    if "y" not in df.columns:
        raise ValueError("Brakuje kolumny targetu 'y'.")

    valid_target = df["y"].isin(["no", "yes"])
    target = df.loc[valid_target, "y"]
    if target.empty:
        raise ValueError("Brak prawidłowych wartości 'no' lub 'yes' w kolumnie y.")

    overall_yes_rate = 100 * target.eq("yes").mean()
    rows = []
    for column in columns:
        if column not in df.columns:
            continue

        values = df.loc[valid_target, column].astype("object")
        values = values.where(values.notna(), "brak danych")
        grouped = pd.DataFrame({"kategoria": values, "wynik": target})

        for category, group in grouped.groupby("kategoria", sort=False):
            yes_count = int(group["wynik"].eq("yes").sum())
            no_count = int(group["wynik"].eq("no").sum())
            group_size = yes_count + no_count
            yes_rate = 100 * yes_count / group_size if group_size else float("nan")
            rows.append({
                "zmienna": column,
                "kategoria": category,
                "liczba klientów": group_size,
                "brak lokaty": no_count,
                "lokata": yes_count,
                "odsetek lokat w kategorii (%)": round(yes_rate, 1),
                "odsetek lokat w całej próbce (%)": round(overall_yes_rate, 1),
                "różnica względem próbki (p.p.)": round(yes_rate - overall_yes_rate, 1),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["zmienna", "odsetek lokat w kategorii (%)"],
            ascending=[True, False],
        ).reset_index(drop=True)
    return result


def plot_feature_by_target(df: pd.DataFrame, feature: str) -> None:
    """Plot one selected feature split by target class."""
    if feature not in df.columns:
        raise ValueError(f"Nie znaleziono cechy '{feature}'. Dostępne: {list(df.columns)}")
    if feature == "y":
        raise ValueError("Wybierz cechę, nie kolumnę wynikową 'y'.")
    if "y" not in df.columns:
        raise ValueError("Brakuje kolumny targetu 'y'.")

    valid_target = df["y"].isin(["no", "yes"])
    if feature in NUMERIC_FEATURES:
        values = pd.to_numeric(df.loc[valid_target, feature], errors="coerce")
        labels = df.loc[valid_target, "y"]
        group_no = values.loc[labels.eq("no")].dropna()
        group_yes = values.loc[labels.eq("yes")].dropna()
        if group_no.empty or group_yes.empty:
            print(f"Za mało danych do narysowania cechy '{feature}'.")
            return

        plt.figure(figsize=(7, 4))
        plt.boxplot([group_no, group_yes], tick_labels=["no", "yes"], showfliers=False)
        plt.ylabel(feature)
    else:
        categories = (
            df.loc[valid_target, feature]
            .astype("string")
            .fillna("brak danych")
        )
        labels = df.loc[valid_target, "y"]
        shares = pd.crosstab(categories, labels, normalize="columns")
        shares = shares.reindex(columns=["no", "yes"], fill_value=0) * 100
        if shares.empty:
            print(f"Brak kategorii do pokazania dla cechy '{feature}'.")
            return

        figure_height = max(4, 0.35 * len(shares))
        shares.plot(kind="barh", figsize=(8, figure_height))
        plt.xlabel("Udział w klasie (%)")
        plt.ylabel(feature)
        plt.legend(title="Założono lokatę (y)")

    plt.title(f"Cecha '{feature}' w podziale na wynik lokaty")
    if feature in NUMERIC_FEATURES:
        plt.xlabel("Wynik lokaty (y)")
    plt.tight_layout()
    plt.show()


def duration_by_target(df: pd.DataFrame) -> None:
    yes = pd.to_numeric(df.loc[df["y"] == "yes", "duration"], errors="coerce").dropna()
    no = pd.to_numeric(df.loc[df["y"] == "no", "duration"], errors="coerce").dropna()

    print("duration – podsumowanie:")
    display(pd.DataFrame({
        "yes": yes.describe(percentiles=[0.25, 0.5, 0.75]),
        "no": no.describe(percentiles=[0.25, 0.5, 0.75]),
    }))

    plt.figure(figsize=(7, 3.8))
    plt.boxplot([no, yes], tick_labels=["no", "yes"], showfliers=False)
    plt.title("Czas rozmowy (duration) a wynik kampanii")
    plt.xlabel("y")
    plt.ylabel("duration [s]")
    plt.tight_layout()
    plt.show()


def feature_context() -> pd.DataFrame:
    descriptions = [
        ("age", "wiek klienta", "znane przed rozmową"),
        ("job", "rodzaj zatrudnienia", "znane przed rozmową"),
        ("education", "poziom wykształcenia", "znane przed rozmową"),
        ("housing", "kredyt mieszkaniowy: yes/no/unknown", "znane przed rozmową"),
        ("contact", "rodzaj kanału kontaktu", "znane przed / przy rozpoczęciu kontaktu"),
        ("month", "miesiąc ostatniego kontaktu", "znane w momencie kontaktu"),
        ("duration", "czas trwania ostatniej rozmowy telefonicznej", "znane dopiero po rozmowie"),
        ("campaign", "liczba kontaktów w bieżącej kampanii", "zależy od momentu predykcji"),
        ("pdays", "dni od poprzedniego kontaktu; 999 oznacza brak wcześniejszego kontaktu", "historyczne"),
    ]
    result = pd.DataFrame(descriptions, columns=["cecha", "znaczenie", "dostępność"])
    display(result)
    return result
