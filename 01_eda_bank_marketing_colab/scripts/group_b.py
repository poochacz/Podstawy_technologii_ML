from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


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


def numeric_features_by_target(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["age", "duration", "campaign", "pdays"]:
        s = pd.to_numeric(df[col], errors="coerce")
        temp = pd.DataFrame({"x": s, "y": df["y"]}).dropna()
        stats = temp.groupby("y")["x"].agg(["count", "mean", "median"])
        if {"yes", "no"}.issubset(stats.index):
            rows.append({
                "cecha": col,
                "średnia_yes": stats.loc["yes", "mean"],
                "średnia_no": stats.loc["no", "mean"],
                "mediana_yes": stats.loc["yes", "median"],
                "mediana_no": stats.loc["no", "median"],
                "różnica_średnich": stats.loc["yes", "mean"] - stats.loc["no", "mean"],
            })
    result = pd.DataFrame(rows).sort_values(
        "różnica_średnich", key=lambda s: s.abs(), ascending=False
    ).round(2)
    display(result)
    return result


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
