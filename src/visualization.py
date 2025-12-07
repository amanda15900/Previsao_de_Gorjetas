# src/visualization.py
import os
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = "reports/figures"

def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)

def save_scatter_total_bill_vs_tip(df, fname="total_vs_tip.png"):
    ensure_fig_dir()
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df['total_bill'], y=df['tip'])
    plt.title("Total bill vs Tip")
    plt.xlabel("Total bill ($)")
    plt.ylabel("Tip ($)")
    plt.grid(True)
    path = f"{FIG_DIR}/{fname}"
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    return path

def save_correlation_heatmap(df, fname="heatmap_correlation.png"):
    ensure_fig_dir()
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap="vlag", fmt=".2f", center=0)
    plt.title("Matriz de correlação")
    path = f"{FIG_DIR}/{fname}"
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    return path

def save_residuals_hist(residuals, fname="residuals_hist.png"):
    ensure_fig_dir()
    plt.figure(figsize=(8,4))
    sns.histplot(residuals, bins=20, kde=True)
    plt.title("Distribuição dos resíduos")
    plt.axvline(0, color='k', ls='--')
    path = f"{FIG_DIR}/{fname}"
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    return path
