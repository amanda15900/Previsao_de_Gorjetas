# main.py
"""
MAIN PIPELINE DO PROJETO (versão com logging e validação cruzada)
"""

import logging
import sys
from src import etl, eda, model
from src import visualization

# logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_etl_pipeline():
    logger.info("ETL pipeline iniciado")
    df_processed = etl.run_etl()
    logger.info(f"ETL finalizado - shape: {df_processed.shape}")
    return df_processed

def run_eda_pipeline(df, save_figures=False):
    logger.info("Executando EDA (gráficos)")
    # exibe gráficos (funções do src/eda.py)
    eda.scatter_total_bill_vs_tip(df)
    eda.correlation_heatmap(df)
    eda.histograms(df)

    if save_figures:
        # salvar figuras via src/visualization.py
        p1 = visualization.save_scatter_total_bill_vs_tip(df)
        p2 = visualization.save_correlation_heatmap(df)
        logger.info(f"Figuras salvas em: {p1}, {p2}")

    logger.info("EDA concluída")

def run_model_pipeline(do_cross_val=False, cv=5):
    logger.info("Treinamento do modelo")
    X, y = model.load_data()

    # baseline
    b_r2, b_rmse = model.baseline_evaluate(X, y)
    logger.info(f"Baseline (mean) R2: {b_r2:.4f} | RMSE: {b_rmse:.4f}")

    # treinar o modelo
    trained_model, X_test, y_test, X_train, y_train = model.train_model(X, y)
    r2, rmse, y_pred = model.evaluate_model(trained_model, X_test, y_test)

    logger.info(f"Modelo Linear - Test R2: {r2:.4f} | RMSE: {rmse:.4f}")

    # salvar modelo
    model.save_model(trained_model)

    if do_cross_val:
        logger.info(f"Executando cross-validation (cv={cv})")
        cv_results = model.cross_validate_model(trained_model, X, y, cv=cv)
        logger.info(f"CrossVal R2 mean: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
        logger.info(f"CrossVal RMSE mean: {cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")

    return trained_model, (X_test, y_test, y_pred)

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
5 - Treinar + validação cruzada (k-fold)
0 - Sair
""")
    return input("Digite sua opção: ").strip()

if __name__ == "__main__":
    while True:
        opcao = menu()

        if opcao == "1":
            run_etl_pipeline()

        elif opcao == "2":
            try:
                df = eda.load_processed_data()
                run_eda_pipeline(df, save_figures=True)
            except Exception as e:
                logger.error("Erro na EDA: verifique se o ETL foi executado.")
                logger.exception(e)

        elif opcao == "3":
            try:
                run_model_pipeline(do_cross_val=False)
            except Exception as e:
                logger.error("Erro no treinamento do modelo.")
                logger.exception(e)

        elif opcao == "4":
            try:
                df = run_etl_pipeline()
                run_eda_pipeline(df, save_figures=True)
                run_model_pipeline(do_cross_val=False)
            except Exception as e:
                logger.error("Erro na execução completa.")
                logger.exception(e)

        elif opcao == "5":
            try:
                run_etl_pipeline()
                # rodada com CV
                run_model_pipeline(do_cross_val=True, cv=5)
            except Exception as e:
                logger.error("Erro na execução com validação cruzada.")
                logger.exception(e)

        elif opcao == "0":
            logger.info("Encerrando o programa... Até mais!")
            break

        else:
            print("\nOpção inválida. Tente novamente.")
