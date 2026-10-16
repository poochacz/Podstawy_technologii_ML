from __future__ import annotations

from pathlib import Path
import json
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd


UCI_PRIMARY = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
UCI_SECONDARY = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

SELECTED_COLUMNS = [
    "age",
    "job",
    "education",
    "housing",
    "contact",
    "month",
    "duration",
    "campaign",
    "pdays",
    "y",
]

SOURCE_FILE = "bank-additional/bank-additional-full.csv"
WORKSHOP_FILE = "bank_workshop_2000.csv"
REFERENCE_FILE = "bank_reference_selected.csv"
META_FILE = "workshop_metadata.json"


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 workshop-ml/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def download_bank_marketing(data_dir: str | Path = "data", force: bool = False) -> pd.DataFrame:
    """Download the official UCI Bank Marketing archive and return bank-additional-full.csv."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    extracted = data_dir / SOURCE_FILE
    if extracted.exists() and not force:
        return pd.read_csv(extracted, sep=";")

    archive = data_dir / "bank-additional.zip"
    errors = []

    for url in (UCI_PRIMARY, UCI_SECONDARY):
        try:
            _download(url, archive)
            with zipfile.ZipFile(archive, "r") as zf:
                names = zf.namelist()
                candidate = None
                for name in names:
                    if name.endswith("bank-additional-full.csv"):
                        candidate = name
                        break
                if candidate is None:
                    raise FileNotFoundError("Archiwum nie zawiera bank-additional-full.csv")

                target_dir = data_dir / "bank-additional"
                target_dir.mkdir(exist_ok=True)
                with zf.open(candidate) as src, (target_dir / "bank-additional-full.csv").open("wb") as dst:
                    shutil.copyfileobj(src, dst)

            return pd.read_csv(target_dir / "bank-additional-full.csv", sep=";")
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Nie udało się pobrać danych UCI. Sprawdź dostęp do Internetu.\n"
        + "\n".join(errors)
    )


def _weighted_sample(reference: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    """
    Create a deliberately imperfect workshop sample.
    The bias is subtle enough to require EDA:
      - May is overrepresented,
      - rare spring/autumn months are underrepresented,
      - cellular contacts are slightly overrepresented,
      - clients >60 are slightly underrepresented.
    """
    w = pd.Series(1.0, index=reference.index)

    w *= np.where(reference["month"].eq("may"), 2.15, 1.0)
    w *= np.where(reference["month"].isin(["mar", "sep", "oct", "dec"]), 0.55, 1.0)
    w *= np.where(reference["contact"].eq("cellular"), 1.20, 0.82)
    w *= np.where(reference["age"].gt(60), 0.62, 1.0)

    return reference.sample(
        n=min(n, len(reference)),
        replace=False,
        weights=w,
        random_state=random_state,
    ).copy()


def _inject_training_issues(sample: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Inject a small number of controlled quality issues for the EDA exercise."""
    rng = np.random.default_rng(random_state)
    df = sample.copy().reset_index(drop=True)

    # Actual NaN values in different data types.
    missing_plan = {
        "age": 18,
        "education": 24,
        "housing": 14,
        "month": 8,
        "duration": 8,
        "campaign": 8,
        "pdays": 8,
    }
    reserved_indices = {}
    for col, count in missing_plan.items():
        idx = rng.choice(df.index, size=count, replace=False)
        df.loc[idx, col] = np.nan
        reserved_indices[col] = set(idx)

    unknown_plan = {
        "month": 8,
        "duration": 4,
        "campaign": 4,
        "pdays": 4,
    }
    for col, count in unknown_plan.items():
        available = df.index[~df.index.isin(reserved_indices[col])]
        idx = rng.choice(available, size=count, replace=False)
        df[col] = df[col].astype("object")
        df.loc[idx, col] = "unknown"
        reserved_indices[col].update(idx)

    # Clearly impossible / invalid numeric values.
    invalid_values = {
        "age": [160, -4],
        "campaign": [-2],
        "duration": [-30],
        "pdays": [-5],
    }
    used = set()
    for col, values in invalid_values.items():
        for value in values:
            excluded = used | reserved_indices.get(col, set())
            candidates = [i for i in df.index if i not in excluded]
            i = int(rng.choice(candidates))
            used.add(i)
            df.loc[i, col] = value

    return df


def prepare_workshop_data(
    data_dir: str | Path = "data",
    n: int = 2000,
    random_state: int = 42,
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download UCI data, keep 10 columns, create one biased 2000-row sample,
    and inject controlled EDA issues.
    """
    data_dir = Path(data_dir)
    full = download_bank_marketing(data_dir=data_dir, force=force_download)

    missing = [c for c in SELECTED_COLUMNS if c not in full.columns]
    if missing:
        raise ValueError(f"Brakuje oczekiwanych kolumn: {missing}")

    reference = full[SELECTED_COLUMNS].copy()
    sample = _weighted_sample(reference, n=n, random_state=random_state)
    sample = _inject_training_issues(sample, random_state=random_state + 100)

    reference.to_csv(data_dir / REFERENCE_FILE, index=False)
    sample.to_csv(data_dir / WORKSHOP_FILE, index=False)

    metadata = {
        "source": "UCI Machine Learning Repository – Bank Marketing",
        "source_url": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "source_file": "bank-additional-full.csv",
        "reference_rows": int(len(reference)),
        "workshop_rows": int(len(sample)),
        "selected_columns": SELECTED_COLUMNS,
        "source_columns": list(full.columns),
        "excluded_columns": [column for column in full.columns if column not in SELECTED_COLUMNS],
        "target": "y",
        "random_state": random_state,
    }
    (data_dir / META_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return sample, reference


def load_workshop_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    workshop = data_dir / WORKSHOP_FILE
    reference = data_dir / REFERENCE_FILE
    if not workshop.exists() or not reference.exists():
        raise FileNotFoundError(
            "Brakuje przygotowanych danych. Najpierw uruchom notebook 00_EDA_wspolne.ipynb."
        )
    return pd.read_csv(workshop), pd.read_csv(reference)
