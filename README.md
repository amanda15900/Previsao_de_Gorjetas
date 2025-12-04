# Projeto Prático: Previsão de Gorjetas com Machine Learning

Este repositório contém a solução prática desenvolvida para a disciplina de Mineração de Dados, focada em análise de dados e aprendizado de máquina supervisionado.

**Integrantes do Grupo:**
* Amanda Rodrigues Agelune
* Thalles Silva
* Henrique Nazario

---

### Contextualização
Em serviços de alimentação e hospitalidade, a previsibilidade de receitas variáveis é crucial tanto para a gestão do estabelecimento quanto para o planejamento financeiro dos colaboradores. As gorjetas representam uma parte significativa da remuneração em muitos países.

### Objetivo do Projeto
O objetivo principal é desenvolver um **Modelo Preditivo Supervisionado (Regressão)** capaz de estimar o valor da gorjeta (`tip`) com base em variáveis observáveis no momento do serviço.

### Descrição dos Dados (Dataset)
Utilizamos o conjunto de dados público **Tips**, que contém registros de consumo em um restaurante. As variáveis explicativas (features) utilizadas foram:
* `total_bill`: Valor total da conta (em dólares).
* `sex`: Gênero do pagante da conta.
* `smoker`: Presença de fumantes na mesa.
* `day`: Dia da semana (Quinta a Domingo).
* `time`: Horário da refeição (Almoço ou Jantar).
* `size`: Quantidade de pessoas na mesa.

---

## 2. ⚙️ Implementação Técnica (ETL e Modelagem)
*(Critério de Avaliação: 15 pontos)*

A solução foi desenvolvida inteiramente em **Python**, utilizando o ambiente **Google Colab**. Abaixo, detalhamos o pipeline de dados construído:

### A. Bibliotecas Utilizadas
* **Pandas:** Manipulação e estruturação dos dados tabulares.
* **Seaborn & Matplotlib:** Criação de gráficos para análise exploratória.
* **Scikit-Learn:** Construção do modelo de machine learning e métricas de avaliação.

### B. Processo de ETL (Extração, Transformação e Carga)
1.  **Ingestão:** Carregamento automatizado via `sns.load_dataset('tips')`.
2.  **Verificação de Qualidade:** Análise de valores nulos (missing values) e integridade dos tipos de dados. O dataset apresentou-se limpo, não exigindo imputação de dados.
3.  **Encoding (Transformação Categórica):**
    * Como algoritmos de regressão matemática não processam texto, aplicamos a técnica de **One-Hot Encoding** (via `pd.get_dummies`).
    * Variáveis como `sex` e `smoker` foram convertidas em vetores binários (0 e 1).

### C. Estratégia de Modelagem
* **Algoritmo:** Regressão Linear Múltipla (`LinearRegression`).
* **Justificativa:** A análise preliminar indicou uma forte relação linear entre a conta e a gorjeta, tornando este algoritmo eficiente e de alta interpretabilidade.
* **Separação de Dados:**
    * **Treino (80%):** Para o algoritmo aprender os padrões.
    * **Teste (20%):** Dados inéditos para validar a performance real do modelo.

---

## 3. 📈 Visualizações e Interpretação dos Resultados
*(Critério de Avaliação: 10 pontos)*

### Análise Exploratória (EDA)
Durante a fase de exploração, geramos visualizações que trouxeram os seguintes insights:
1.  **Correlação Positiva Forte:** O gráfico de dispersão (*Scatter Plot*) entre `total_bill` e `tip` evidenciou que, conforme o valor da conta aumenta, o valor da gorjeta tende a aumentar proporcionalmente.
2.  **Mapa de Calor (Heatmap):** A matriz de correlação confirmou matematicamente que a variável `total_bill` possui o maior coeficiente de correlação com o alvo `tip`, sendo o preditor mais importante.

### Performance do Modelo
Após o treinamento, o modelo foi submetido aos dados de teste, obtendo as seguintes métricas:

| Métrica | Valor Obtido | Interpretação |
| :--- | :--- | :--- |
| **R² (R-Quadrado)** | **~0.44** | O modelo consegue explicar cerca de 44% da variância das gorjetas baseando-se nas variáveis fornecidas. |
| **RMSE (Erro Médio)** | **~$1.00** | Em média, o modelo erra o valor da gorjeta em aproximadamente 1 dólar para mais ou para menos. |

**Análise Crítica:** O resultado é satisfatório considerando que o ato de dar gorjeta possui um componente subjetivo (humano) que não pode ser totalmente capturado apenas pelos dados da conta.

---
### Como Executar o Projeto

O arquivo principal é o notebook `.ipynb`. Basta abri-lo no Google Colab ou Jupyter e executar todas as células.
