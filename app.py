import streamlit as st
from pathlib import Path
from src.preprocess import load_data
from src.recommender import SteamRecommender

st.set_page_config(page_title="Steam Recommender", layout="wide")

DATA_PATH = Path("data/dataset.xlsx")
MODEL_DIR = Path("models")


@st.cache_resource
def load_reco():
    df = load_data(DATA_PATH)
    reco = SteamRecommender()
    reco.load(MODEL_DIR, df)
    return reco


reco = load_reco()

st.title("🎮 Steam Game Recommender")

title = st.text_input("Digite o título de referência:")
cols = st.columns(2)
top_n = cols[0].slider("Top‑N", 5, 20, 10)
if cols[1].button("Recomendar"):
    if not title:
        st.warning("Digite um título.")
    else:
        result = reco.recommend(title, top_n)
        if result is None:
            st.warning("Título não encontrado.")
        else:
            st.dataframe(result, use_container_width=True)
