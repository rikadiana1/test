import pandas as pd


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt überflüssige 'Unnamed'-Spalten, die beim Excel-Export entstehen.
    """
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vereinheitlicht Datentypen für spätere Verarbeitung.
    """
    df["tmsp"] = pd.to_datetime(df["tmsp"], errors="coerce")
    df["country"] = df["country"].astype("category")
    df["PSP"] = df["PSP"].astype("category")
    df["card"] = df["card"].astype("category")
    df["success"] = df["success"].astype(int)
    df["3D_secured"] = df["3D_secured"].astype(int)
    return df


def create_attempt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erzeugt Wiederholungsversuch-Features:
    - attempt_id (Minute × Land × Betrag)
    - attempt_no (Versuchsnummer)
    - is_retry (1 = Wiederholungsversuch)
    """
    df["tmsp_min"] = df["tmsp"].dt.floor("T")

    df["attempt_id"] = (
        df["tmsp_min"].astype(str)
        + "|" + df["country"].astype(str)
        + "|" + df["amount"].astype(str)
    )

    df["attempt_no"] = df.groupby("attempt_id").cumcount() + 1
    df["is_retry"] = (df["attempt_no"] > 1).astype(int)
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erzeugt zeitbezogene Features:
    - hour (0–23)
    - dow (Wochentag 0–6)
    """
    df["hour"] = df["tmsp"].dt.hour
    df["dow"] = df["tmsp"].dt.dayofweek
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Führt alle Feature-Engineering-Schritte aus.
    """
    df = clean_columns(df)
    df = cast_types(df)
    df = create_attempt_features(df)
    df = create_time_features(df)
    return df


if __name__ == "__main__":
    # Testlauf (nur ausführen, wenn Datei direkt gestartet wird)
    from pathlib import Path
    from data_preparation import load_raw_data

    raw_path = Path("data/raw/PSP_Jan_Feb_2019.xlsx")

    df = load_raw_data(raw_path)
    df = create_all_features(df)

    print(df.head(3))
    print(df.dtypes)
