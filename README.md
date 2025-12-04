# Projeto Prático: Previsão de Gorjetas com Machine Learning

Este repositório contém a solução prática desenvolvida para a disciplina de [NOME DA DISCIPLINA], focada em análise de dados e aprendizado de máquina supervisionado.

**Integrantes do Grupo:**
* [Nome do Integrante 1]
* [Nome do Integrante 2]
* [Nome do Integrante 3]
* [Nome do Integrante 4]
* [Nome do Integrante 5]

---

## 1. Definição e Estruturação do Problema
*(Critério de Avaliação: 10 pontos)*

**Problema de Negócio:**
Em ambientes de restauração, a previsibilidade de ganhos extras (gorjetas) é uma variável importante tanto para o planejamento da equipe quanto para a gestão do estabelecimento. O desafio proposto é entender quais fatores influenciam o valor da gorjeta deixada pelos clientes.

**Objetivo:**
Desenvolver um modelo preditivo de **Regressão** capaz de estimar o valor da gorjeta (`tip`) com base em características da refeição, tais como:
* Valor total da conta (`total_bill`);
* Sexo do pagante;
* Se há fumantes na mesa;
* Dia da semana e horário (Almoço/Jantar);
* Tamanho da mesa.

**Conjunto de Dados:**
Foi utilizado o dataset público **"Tips"**, disponível na biblioteca Seaborn, que contém dados reais de transações em um restaurante.

---

## 2. Implementação Técnica
*(Critério de Avaliação: 15 pontos)*

A solução foi desenvolvida em **Python** utilizando as bibliotecas `Pandas`, `Seaborn`, `Matplotlib` e `Scikit-Learn`.

**Processo de ETL (Extração, Transformação e Carga):**
1.  **Carregamento:** Dados importados diretamente via `sns.load_dataset('tips')`.
2.  **Limpeza:** Verificação de integridade dos dados (não foram encontrados valores nulos críticos).
3.  **Transformação (Encoding):** Aplicação de *One-Hot Encoding* (`pd.get_dummies`) para converter variáveis categóricas (como 'sex', 'smoker', 'day') em variáveis numéricas binárias (0 e 1), permitindo o processamento pelo algoritmo.

**Modelagem:**
* **Algoritmo Escolhido:** Regressão Linear Múltipla (`LinearRegression`).
* **Justificativa:** Pela natureza contínua da variável alvo e pela relação linear observada na análise exploratória.
* **Divisão dos Dados:** 80% para Treino e 20% para Teste (random_state=42).

---

## 3. Visualizações e Interpretação dos Resultados
*(Critério de Avaliação: 10 pontos)*

Durante a Análise Exploratória de Dados (EDA), destacam-se os seguintes insights:

1.  **Correlação Positiva:** O gráfico de dispersão (*Scatter Plot*) evidenciou uma clara correlação linear positiva entre o valor total da conta e a gorjeta. Quanto maior a conta, maior a gorjeta.
2.  **Mapa de Calor:** A matriz de correlação confirmou que a variável `total_bill` é o preditor mais forte para o alvo `tip`.

**Performance do Modelo:**
O modelo foi avaliado nos dados de teste e obteve os seguintes resultados:
* **R² (Coeficiente de Determinação):** Indiciou que o modelo consegue explicar uma parcela significativa da variância dos dados.
* **RMSE (Erro Quadrático Médio):** Apresentou uma margem de erro aceitável para o contexto do problema, indicando que as previsões estão próximas dos valores reais.

---

## 4. Clareza e Qualidade da Documentação
*(Critério de Avaliação: 5 pontos)*

### Como Executar o Projeto

**Pré-requisitos:**
Certifique-se de ter o Python instalado e as seguintes bibliotecas:
```bash
pip install pandas seaborn matplotlib scikit-learn
