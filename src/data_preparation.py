
import matplotlib.pyplot as plt


from pathlib import Path
import pandas as pd

def load_raw_data(filename="PSP_Jan_Feb_2019.xlsx"):
    """
    Lädt Rohdaten aus data/raw/.
    """
    data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / filename
    if not data_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {data_path}")

    return pd.read_excel(data_path)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Liest die Rohdaten ein und gibt ein DataFrame zurück.
    """
    df = pd.read_excel(filepath)
    return df


def explore_raw_data(df: pd.DataFrame):
    """
    Gibt grundlegende Strukturinformationen über den Datensatz aus.
    """
    print("📌 Datensatz Form (Zeilen, Spalten):")
    print(df.shape)
    print("\n📌 Info:")
    print(df.info())
    print("\n📌 Erste Zeilen:")
    print(df.head())

    print("\n📌 Eindeutige Werte pro Spalte:")
    print(df.nunique())

    print("\n📌 Fehlende Werte pro Spalte:")
    print(df.isna().sum())


if __name__ == "__main__":
    # Pfad anpassen – Rohdaten liegen später in data/raw/
    filepath = Path("data/raw/PSP_Jan_Feb_2019.xlsx")

    df = load_raw_data(filepath)
    explore_raw_data(df)
