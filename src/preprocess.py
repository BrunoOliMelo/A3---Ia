from pathlib import Path
import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Lê o arquivo Excel de metadados da Steam e devolve um DataFrame.
    """
    df = pd.read_excel(path)
    return df
