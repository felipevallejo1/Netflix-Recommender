import ast
import os
import time
from typing import List, Tuple

import gradio as gr
import numpy as np
import pandas as pd

from clean_netflix import clean_and_save, DATA_DIR

# LangChain / OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
CLEANED_CSV = os.path.join(DATA_DIR, "cleaned_netflix.csv")


def _read_openai_key() -> str:
    # Reads key from api_key.txt located at project root
    root = os.path.dirname(__file__)
    key_path = os.path.join(root, "api_key.txt")
    if not os.path.exists(key_path):
        raise FileNotFoundError("No se encontró api_key.txt")
    with open(key_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    # Allow formats like: key = "sk-..." or just the key
    if "sk-" in content:
        # extract quoted value if present
        import re

        m = re.search(r"sk-[a-zA-Z0-9_\-]+[a-zA-Z0-9_\-]", content)
        if m:
            return m.group(0)
    return content


def _ensure_clean_dataset() -> pd.DataFrame:
    if not os.path.exists(CLEANED_CSV):
        df = clean_and_save()
    else:
        df = pd.read_csv(CLEANED_CSV)
    # Convert genres_list back to list
    if "genres_list" in df.columns and isinstance(df.loc[0, "genres_list"], str):
        try:
            df["genres_list"] = df["genres_list"].apply(
                lambda s: ast.literal_eval(s) if isinstance(s, str) else s
            )
        except Exception:
            df["genres_list"] = df["listed_in"].fillna("").apply(
                lambda s: [g.strip() for g in s.split(",") if g.strip()]
            )
    return df


def _build_or_load_index(df: pd.DataFrame, openai_api_key: str) -> Tuple[FAISS, List[dict]]:
    # We will create documents as combined text strings and keep metadata list aligned by index
    docs: List[str] = []
    metadatas: List[dict] = []
    for _, row in df.iterrows():
        genres = ", ".join(row.get("genres_list", []) or [])
        text = (
            f"Title: {row['title']}\n"
            f"Type: {row['type']}\n"
            f"Year: {int(row['release_year']) if not pd.isna(row['release_year']) else ''}\n"
            f"Genres: {genres}\n"
            f"Description: {row['description']}"
        )
        docs.append(text)
        metadatas.append({
            "title": row["title"],
            "type": row["type"],
            "year": int(row["release_year"]) if not pd.isna(row["release_year"]) else None,
            "genres": row.get("genres_list", []) or [],
            "rating": row.get("rating", None),
            "description": row["description"],
        })

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    if os.path.exists(INDEX_DIR):
        try:
            vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            return vs, metadatas
        except Exception:
           
            pass

    vs = FAISS.from_texts(docs, embedding=embeddings, metadatas=metadatas)
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    return vs, metadatas


def _filter_indices_by(df: pd.DataFrame, type_filter: str, genres: List[str]) -> List[int]:
    mask = pd.Series([True] * len(df))
    if type_filter and type_filter != "All":
        mask &= df["type"] == type_filter
    if genres:
        mask &= df["genres_list"].apply(lambda gl: all(g in (gl or []) for g in genres))
    return list(np.flatnonzero(mask.to_numpy()))


def create_app() -> None:
    df = _ensure_clean_dataset()
    all_genres = sorted({g for gl in df["genres_list"].tolist() for g in (gl or [])})
    api_key = _read_openai_key()
    vectorstore, _ = _build_or_load_index(df, api_key)

    def recommend(query: str, type_filter: str, genres: List[str], k: int):
        if not query:
            return pd.DataFrame(columns=["Title", "Type", "Year", "Genres", "Rating", "Description"])  # empty

        
        results = vectorstore.similarity_search_with_score(query, k=100)

        
        rows = []
        for doc, score in results:
            meta = doc.metadata
            rows.append((meta["title"], meta["type"], meta["year"], ", ".join(meta["genres"]), meta.get("rating"), meta["description"], score))

        res_df = pd.DataFrame(rows, columns=["Title", "Type", "Year", "Genres", "Rating", "Description", "score"])

        
        if type_filter != "All":
            res_df = res_df[res_df["Type"] == type_filter]
        if genres:
            for g in genres:
                res_df = res_df[res_df["Genres"].str.contains(fr"\b{g}\b", na=False)]

        res_df = res_df.sort_values("score", ascending=True).head(k)
        res_df = res_df.drop(columns=["score"]).reset_index(drop=True)
        return res_df

    with gr.Blocks(title="Recomendador Netflix", theme=gr.themes.Soft()) as demo:
        gr.Markdown("**Recomendador semántico de películas y series en Netflix**")
        with gr.Row():
            query = gr.Textbox(label="Descripción de lo que querés ver", placeholder="recomendame una serie o película de asesinos en serie", lines=2)
        with gr.Row():
            type_dd = gr.Dropdown(choices=["All", "Movie", "TV Show"], value="All", label="Tipo")
            genre_ms = gr.Dropdown(choices=all_genres, value=[], label="Género(s)", multiselect=True)
            k_slider = gr.Slider(1, 20, value=12, step=1, label="Resultados")
            btn = gr.Button("Buscar recomendaciones", variant="primary")
        out = gr.Dataframe(headers=["Title", "Type", "Year", "Genres", "Rating", "Description"], interactive=False)

        btn.click(fn=recommend, inputs=[query, type_dd, genre_ms, k_slider], outputs=out)

    demo.launch(server_name="0.0.0.0", server_port=4000, show_error=True)


if __name__ == "__main__":
    create_app()


