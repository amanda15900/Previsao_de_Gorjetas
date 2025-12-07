import os
import pandas as pd
import numpy as np
import seaborn as sns


def load_raw_dataset():
    """Carrega o dataset direto do seaborn."""
    df = sns.load_dataset("tips")
    return df


def save_raw_csv(df, path="data/raw/tips_raw.csv"):
    """Salva o dataset bruto em CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Raw dataset salvo em: {path}")


def clean_dataset(df):
    """Realiza limpeza inicial do dataset."""
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def encode_dataset(df):
    """Aplica one-hot encoding nas colunas categóricas."""
    cat_cols = ['sex', 'smoker', 'day', 'time']
    df_encoded = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)

    # Reordena colunas: target e principais primeiro
    cols = df_encoded.columns.tolist()
    ordered_cols = ['total_bill', 'tip', 'size'] + [
        c for c in cols if c not in ('total_bill', 'tip', 'size')
    ]

    df_encoded = df_encoded[ordered_cols]
    return df_encoded


def save_processed_dataset(df, path="data/processed/tips_clean.csv"):
    """Salva CSV processado."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Processed dataset salvo em: {path}")


def save_numpy_arrays(df, folder="data/processed"):
    """Salva X e y como arquivos numpy."""
    os.makedirs(folder, exist_ok=True)

    X = df.drop(columns=["tip"]).values
    y = df["tip"].values

    np.save(os.path.join(folder, "features.npy"), X)
    np.save(os.path.join(folder, "target.npy"), y)

    print("features.npy e target.npy salvos com sucesso.")


def run_etl():
    """Pipeline completo de ETL."""
    df = load_raw_dataset()
    save_raw_csv(df)

    df = clean_dataset(df)
    df = encode_dataset(df)

    save_processed_dataset(df)
    save_numpy_arrays(df)

    return df
