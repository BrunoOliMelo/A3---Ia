Steam Game Recommender – A3 UC IA

# 1. Criar / ativar ambiente Python 3.11
python -m venv .venv311
.venv311\Scripts\Activate.ps1

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Treinar modelos (uma vez)
python src\train.py

# 4. Iniciar interface
streamlit run app.py

