import pandas as pd
import numpy as np

def print_separator():
    print("-" * 60)


def preview_dataframe(df, lines=5):
    """Mostra algumas linhas do dataframe."""
    print_separator()
    print(df.head(lines))
    print_separator()


def describe_dataframe(df):
    """Mostra descrição estatística."""
    print_separator()
    print(df.describe())
    print_separator()
