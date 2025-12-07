"""
MAIN PIPELINE DO PROJETO
------------------------
Este script executa todo o fluxo:
1. ETL (carregar, limpar, encoding, salvar dados)
2. EDA (opcional)
3. Modelagem (treinar, avaliar, salvar modelo)
"""

import os
from src import etl, eda, model


def run_etl_pipeline():
    print("\n======================")
    print(" ETL PIPELINE INICIADO")
    print("======================")

    df_processed = etl.run_etl()

    print("\nETL finalizado com sucesso.")
    print(f"Dataset processado contém {df_processed.shape[0]} linhas e {df_processed.shape[1]} colunas.")

    return df_processed


def run_model_pipeline():
    print("\n============================")
    print(" TREINAMENTO DO MODELO")
    print("============================")

    # Carregar dados processados
    X, y = model.load_data()

    # Treinar modelo
    trained_model, X_test, y_test = model.train_model(X, y)

    print("\nModelo treinado com sucesso!")

    # Avaliar modelo
    r2, rmse, y_pred = model.evaluate_model(trained_model, X_test, y_test)

    print("\n============================")
    print(" RESULTADOS DO MODELO")
    print("============================")
    print(f"R² (R-Quadrado): {r2:.4f}")
    print(f"RMSE (Erro Médio): {rmse:.4f}")

    # Salvar modelo
    model.save_model(trained_model)

    return trained_model


def run_eda_pipeline(df):
    print("\n=======================")
    print(" INICIANDO EDA (visuais)")
    print("=======================")

    eda.scatter_total_bill_vs_tip(df)
    eda.correlation_heatmap(df)
    eda.histograms(df)

    print("\nEDA concluída com sucesso!")


def menu():
    print("""
=========================================
          PROJETO DE GORJETAS
=========================================

Escolha uma opção:

1 - Executar ETL completo
2 - Executar EDA (análise exploratória)
3 - Treinar e avaliar modelo
4 - Executar TUDO (ETL + EDA + Modelagem)
0 - Sair
""")

    return input("Digite sua opção: ").strip()


if __name__ == "__main__":
    while True:
        opcao = menu()

        if opcao == "1":
            df = run_etl_pipeline()

        elif opcao == "2":
            try:
                df = eda.load_processed_data()
                run_eda_pipeline(df)
            except:
                print("\nERRO: Execute o ETL antes da EDA.")

        elif opcao == "3":
            run_model_pipeline()

        elif opcao == "4":
            df = run_etl_pipeline()
            run_eda_pipeline(df)
            run_model_pipeline()

        elif opcao == "0":
            print("\nEncerrando o programa... Até mais!")
            break

        else:
            print("\nOpção inválida. Tente novamente.")
