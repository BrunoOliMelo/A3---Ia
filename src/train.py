from pathlib import Path
from preprocess import load_data
from recommender import SteamRecommender

DATA_PATH = Path("data/dataset.xlsx")
MODEL_DIR = Path("models")

print("Carregando dados…")
df = load_data(DATA_PATH)

print("Treinando modelo…")
reco = SteamRecommender()
reco.fit(df)
reco.save(MODEL_DIR)

print("Modelos salvos em", MODEL_DIR.resolve())
