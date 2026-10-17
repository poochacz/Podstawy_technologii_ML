from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
UCI_URLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
    "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
)
SOURCE_CSV = Path("raw") / "bank-additional" / "bank-additional-full.csv"
BUNDLED_CSV = PROJECT_DIR / "source_data" / "bank-additional-full.csv"
SAMPLE_CSV = "bank_modeling_{size}.csv"
METADATA_JSON = "modeling_metadata_{size}.json"
SOURCE_COLUMNS = (
    "age",
    "job",
    "education",
    "housing",
    "contact",
    "month",
    "campaign",
    "pdays",
    "y",
)
MODEL_FEATURES = (
    "age",
    "job",
    "education",
    "housing",
    "contact",
    "month",
    "campaign",
    "pdays",
    "had_previous_contact",
)


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 ml-manager-workshop/1.0"},
    )
    with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def download_source(data_dir: str | Path = DEFAULT_DATA_DIR, force: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Download the UCI archive once and return its full table and column names."""
    data_dir = Path(data_dir)
    source_path = data_dir / SOURCE_CSV
    # Kopia z repozytorium: notatniki działają nawet wtedy, gdy UCI jest niedostępne.
    for local_path in (source_path, BUNDLED_CSV):
        if local_path.exists() and not force:
            frame = pd.read_csv(local_path, sep=";")
            return frame, frame.columns.tolist()

    source_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "raw" / "bank-additional.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    errors = []

    for url in UCI_URLS:
        try:
            _download(url, archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                member = next(
                    (name for name in archive.namelist() if name.endswith("bank-additional-full.csv")),
                    None,
                )
                if member is None:
                    raise FileNotFoundError("Archiwum nie zawiera bank-additional-full.csv")
                with archive.open(member) as source, source_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            frame = pd.read_csv(source_path, sep=";")
            return frame, frame.columns.tolist()
        except Exception as error:
            errors.append(f"{url}: {error}")

    raise RuntimeError("Nie udało się pobrać danych UCI. Sprawdź połączenie internetowe.\n" + "\n".join(errors))


def prepare_model_data(
    n: int | None = 2000,
    random_state: int = 42,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create/cache a stratified, model-ready sample; n=None keeps every record. duration is excluded."""
    if n is not None and n < 20:
        raise ValueError("Próbka musi zawierać co najmniej 20 rekordów.")

    data_dir = Path(data_dir)
    size_label = "full" if n is None else str(n)
    sample_path = data_dir / SAMPLE_CSV.format(size=size_label)
    metadata_path = data_dir / METADATA_JSON.format(size=size_label)
    if not force and sample_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("random_state") == random_state:
            cached = pd.read_csv(sample_path)
            expected_rows = metadata.get("sample_rows") if n is None else n
            if len(cached) == expected_rows and set(MODEL_FEATURES).issubset(cached.columns) and "duration" not in cached.columns:
                return cached, metadata

    source, source_columns = download_source(data_dir=data_dir, force=force)
    missing_columns = [column for column in SOURCE_COLUMNS if column not in source.columns]
    if missing_columns:
        raise ValueError(f"W pliku UCI brakuje wymaganych kolumn: {missing_columns}")

    selected = source.loc[:, SOURCE_COLUMNS].dropna(subset=["y"]).copy()
    selected["had_previous_contact"] = selected["pdays"].ne(999).astype("int64")
    selected.loc[selected["pdays"].eq(999), "pdays"] = np.nan
    selected = selected.loc[:, [*MODEL_FEATURES, "y"]]
    if n is None:
        sample = selected
    elif n > len(selected):
        raise ValueError(f"Za mało rekordów w źródle: dostępne {len(selected)}, żądane {n}.")
    elif n == len(selected):
        sample = selected
    else:
        sample, _ = train_test_split(
            selected,
            train_size=n,
            random_state=random_state,
            stratify=selected["y"],
        )
    sample = sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)

    metadata: dict[str, object] = {
        "source": "UCI Machine Learning Repository – Bank Marketing",
        "source_url": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "source_rows": int(len(source)),
        "source_columns": source_columns,
        "retained_source_columns": list(SOURCE_COLUMNS),
        "excluded_source_columns": [column for column in source_columns if column not in SOURCE_COLUMNS],
        "model_features": list(MODEL_FEATURES),
        "target": "y",
        "sample_rows": int(len(sample)),
        "random_state": random_state,
        "duration_excluded": True,
        "pdays_999_encoded_as": "had_previous_contact indicator; pdays set to missing before imputation",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return sample, metadata
