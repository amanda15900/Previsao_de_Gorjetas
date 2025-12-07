import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_processed_data(path="data/processed/tips_clean.csv"):
    return pd.read_csv(path)


def correlation_heatmap(df):
    """Gera heatmap de correlação."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="vlag", fmt=".2f")
    plt.title("Matriz de Correlação")
    plt.show()


def scatter_total_bill_vs_tip(df):
    """Scatterplot entre total_bill e tip."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df["total_bill"], y=df["tip"])
    plt.title("Total Bill vs Tip")
    plt.xlabel("Total Bill ($)")
    plt.ylabel("Tip ($)")
    plt.grid(True)
    plt.show()


def histograms(df):
    """Histogramas das principais variáveis."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    sns.histplot(df["total_bill"], bins=20, kde=True, ax=ax[0])
    ax[0].set_title("Distribuição total_bill")

    sns.histplot(df["tip"], bins=20, kde=True, ax=ax[1])
    ax[1].set_title("Distribuição tip")

    sns.countplot(x=df["size"], ax=ax[2])
    ax[2].set_title("Distribuição da variável size")

    plt.tight_layout()
    plt.show()
