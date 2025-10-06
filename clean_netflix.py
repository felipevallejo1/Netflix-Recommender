import os
import re
from typing import List

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "netflix")
MOVIES_PATH = os.path.join(DATA_DIR, "movies_on_netflix.csv")
SERIES_PATH = os.path.join(DATA_DIR, "tv_series_on_netflix.csv")

OUTPUT_CSV = os.path.join(DATA_DIR, "cleaned_netflix.csv")
OUTPUT_PARQUET = os.path.join(DATA_DIR, "cleaned_netflix.parquet")


ALLOWED_MATURITY_RATINGS = {

    "PG", "PG-13", "R", "NC-17",
    "TV-PG", "TV-14", "TV-MA",
}


def _load_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    expected = [
        "type", "title", "description", "listed_in", "rating", "duration",
        "release_year", "season_count",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = None
    
    for col in ["type", "title", "description", "listed_in", "rating", "duration"]:
        df[col] = df[col].fillna("").astype(str)
    
    for col in ["release_year", "season_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_duration_minutes(text: str) -> float:
    if not text:
        return float("nan")
    
    m = re.search(r"(\d+)", text)
    if not m:
        return float("nan")
    value = int(m.group(1))
    
    return float(value)


def clean_and_save() -> pd.DataFrame:
    movies = _load_dataframe(MOVIES_PATH)
    series = _load_dataframe(SERIES_PATH)

    df = pd.concat([movies, series], ignore_index=True, sort=False)
    if df.empty:
        raise FileNotFoundError(
            f"No se encontraron CSVs en {DATA_DIR}. Asegúrate de tener los archivos de Netflix."
        )

    df = _normalize_columns(df)

    
    df = df[df["type"].isin(["Movie", "TV Show"])]

    
    df["description_len"] = df["description"].str.len()
    df = df[df["description_len"] >= 120]

    df = df[df["rating"].isin(ALLOWED_MATURITY_RATINGS)]

    
    df["genres_list"] = (
        df["listed_in"].apply(lambda s: [g.strip() for g in s.split(",") if g.strip()])
    )

    df["duration_minutes"] = df.apply(
        lambda r: _parse_duration_minutes(r["duration"]) if r["type"] == "Movie" else float("nan"),
        axis=1,
    )
    df["seasons"] = df.apply(
        lambda r: int(r["season_count"]) if (r["type"] == "TV Show" and pd.notna(r["season_count"])) else float("nan"),
        axis=1,
    )

    df = df.drop_duplicates(subset=["title", "type"], keep="first")

    keep_cols = [
        "type", "title", "description", "listed_in", "rating", "release_year",
        "duration_minutes", "seasons", "genres_list",
    ]
    df_out = df[keep_cols].reset_index(drop=True)

    
    os.makedirs(DATA_DIR, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    try:
        df_out.to_parquet(OUTPUT_PARQUET, index=False)
    except Exception:
        
        pass

    return df_out


if __name__ == "__main__":
    cleaned = clean_and_save()
    print(f"Guardado dataset limpio con {len(cleaned)} filas en: {OUTPUT_CSV}")


