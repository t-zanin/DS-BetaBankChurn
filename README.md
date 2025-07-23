#### (Portuguese version below)

# Beta Bank Churn Prediction

## Project Overview

This project was developed for Beta Bank to predict whether a customer is likely to leave the bank soon, aiming to reduce customer churn, which is more costly than retaining existing customers. Using data on past customer behavior and contract terminations, a machine learning model was built to predict churn, targeting an F1-score of at least 0.59 on the test set. The AUC-ROC metric was calculated for comparison with the F1-score. Class imbalance correction techniques were applied, and the final model was optimized to meet the project’s criteria.

## Objectives

- Load and prepare data from the `Churn.csv` file.
- Investigate class balance and train an initial model without addressing imbalance.
- Improve model quality using at least two techniques to correct class imbalance.
- Evaluate the model using F1-score and AUC-ROC on the test set.
- Conduct the final test with the best model, achieving an F1-score ≥ 0.59.
- Document all steps and findings, keeping the code clean and structured.

## Data Description

The dataset is contained in the file:

- **Churn.csv**: Data on Beta Bank customers.

The file includes the following columns:
- `RowNumber`: Index of data rows.
- `CustomerId`: Unique customer identifier.
- `Surname`: Surname.
- `CreditScore`: Credit score.
- `Geography`: Country of residence.
- `Gender`: Gender.
- `Age`: Age.
- `Tenure`: Period of fixed deposit maturation (years).
- `Balance`: Account balance.
- `NumOfProducts`: Number of bank products used.
- `HasCrCard`: Customer has a credit card (1 - yes; 0 - no).
- `IsActiveMember`: Active customer (1 - yes; 0 - no).
- `EstimatedSalary`: Estimated salary.
- `Exited`: Customer churned (1 - yes; 0 - no, target variable).

### Business Conditions
- **Primary Objective**: Predict churn (`Exited`) with an F1-score ≥ 0.59 on the test set.
- **Metrics**: Evaluate F1-score and AUC-ROC, comparing the results.
- **Class Imbalance**: Address the identified imbalance (20.37% churned vs. 79.63% remained).
- **Correction Techniques**: Use at least two approaches for imbalance (e.g., `class_weight`, undersampling).

*Note*: The data is synthetic and does not include additional contract details.

## Data Preprocessing

- **Data Types**:
  - The `Churn.csv` file was loaded using Pandas. The columns `CreditScore`, `Age`, `Balance`, and `EstimatedSalary` are continuous numeric; `Tenure`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, and `Exited` are discrete numeric; `Geography` and `Gender` are categorical.
  - The `Tenure` column was in float format but was converted to int after handling missing values.

- **Missing Values**:
  - Identified 909 missing values in `Tenure`, interpreted as customers with less than 1 year at the bank. These were replaced with 0.
  - No other columns had missing values.

- **Data Quality**:
  - No duplicates or inconsistencies were found.
  - The columns `RowNumber`, `CustomerId`, and `Surname` were removed as they were irrelevant for prediction.
  - Categorical variables (`Geography` and `Gender`) were encoded:
    - `Geography`: One-Hot Encoding (e.g., France, Spain, Germany), with `drop_first=True` to avoid multicollinearity.
    - `Gender`: Label Encoding (0 for Female, 1 for Male).
  - Continuous numeric variables (`CreditScore`, `Age`, `Balance`, `EstimatedSalary`) were normalized using `StandardScaler` to standardize scales.

## Analysis Steps

1. **Data Splitting**:
   - The data was split into 60% for training, 20% for validation, and 20% for testing, using `train_test_split` with `random_state=12345` and `stratify=y` to maintain class proportions.
   - The `train_test_split` function was used to modularize the process.

2. **Class Balance Analysis**:
   - The target variable `Exited` showed imbalance: 79.63% (class 0, remained) and 20.37% (class 1, churned), visualized with a bar plot.
   - The `value_counts` function and Seaborn visualization were used for analysis.

3. **Initial Model Without Imbalance Correction**:
   - Trained a Logistic Regression model (`LogisticRegression`, `random_state=12345`, `solver='liblinear'`) on the training set.
   - Results on the test set:
     - F1-score: 0.3004 (low recall for class 1: 0.20).
     - AUC-ROC: 0.7767 (higher than F1, as it considers probabilities).
     - Classification report: Precision 0.83, recall 0.96, F1 0.89 for class 0; precision 0.59, recall 0.20, F1 0.30 for class 1.

4. **Class Imbalance Correction**:
   - **Technique 1: class_weight='balanced'**:
     - Applied to Logistic Regression to adjust class weights.
     - Results on the validation set: F1-score of 0.5091, AUC-ROC of 0.7918.
   - **Technique 2: Undersampling**:
     - Reduced the majority class (0) to the size of the minority class (1) using random sampling (`random_state=12345`).
     - Trained Logistic Regression with balanced data.
     - Results on the validation set: F1-score of 0.5105, AUC-ROC of 0.7908.
     - Initial test with undersampling: F1-score of 0.5043, AUC-ROC of 0.7793.

5. **Optimization with Threshold Adjustment**:
   - Tested thresholds from 0.3 to 0.7 on the undersampling model (Logistic Regression).
   - Best threshold (0.6) yielded an F1-score of 0.5163 on the validation set.
   - Test results: F1-score of 0.5121, AUC-ROC of 0.7793 (below the 0.59 target).

6. **Random Forest Model with Undersampling**:
   - Replaced Logistic Regression with Random Forest (`n_estimators=100`, `random_state=12345`) using undersampled data.
   - Results on the validation set (threshold 0.5): F1-score of 0.6008.
   - Threshold adjustment (0.6) yielded an F1-score of 0.6129 on the validation set.
   - Final test: F1-score of 0.6165, AUC-ROC of 0.8559, with precision of 0.58 and recall of 0.65 for class 1.

## Key Findings

- **Initial Model Performance**:
  - Logistic Regression without correction had a low F1-score (0.3004) due to class imbalance, with weak recall for class 1 (0.20). The AUC-ROC (0.7767) was higher, indicating good class separation based on probabilities.

- **Imbalance Correction**:
  - Both techniques (`class_weight='balanced'` and undersampling) improved the F1-score (~0.51 on validation) but did not initially meet the 0.59 target on the test set.
  - Random Forest with undersampling and an optimized threshold (0.6) achieved an F1-score of 0.6165 on the test set, surpassing the target.

- **F1-score vs. AUC-ROC Comparison**:
  - The AUC-ROC (0.8559) was consistently higher than the F1-score (0.6165), as it evaluates probability rankings, while the F1-score depends on a fixed threshold.
  - Random Forest improved both metrics compared to Logistic Regression.

## Conclusions and Recommendations

- **Recommended Model**:
  - The Random Forest model with undersampling and a threshold of 0.6 is the best choice, achieving an F1-score of 0.6165 and AUC-ROC of 0.8559 on the test set, surpassing the 0.59 target.
  - The model is effective for predicting churn, with strong recall (0.65) for class 1 (churning customers).

- **Justification**:
  - Random Forest better handled the imbalance and captured complex data patterns.
  - The threshold adjustment (0.6) optimized the balance between precision and recall, maximizing the F1-score.
  - The high AUC-ROC confirms the model’s ability to distinguish between churning and non-churning customers.

- **Business Impact**:
  - Churn prediction enables Beta Bank to identify at-risk customers and offer incentives (e.g., promotions, special plans) for retention, reducing costs compared to acquiring new customers.
  - It is recommended to deploy the model in production, monitoring performance and adjusting the threshold as needed.

## Tools and Technologies

- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn (LogisticRegression, RandomForestClassifier, train_test_split, StandardScaler, LabelEncoder, f1_score, roc_auc_score), Seaborn, Matplotlib
- **Environment**: Jupyter Notebook

## Project Structure

- `notebooks/`: Jupyter Notebook with preprocessing, class analysis, model training, optimization, and final testing.
- `data/`: File `Churn.csv` (not included in the repository due to size).
- `README.md`: Project overview and documentation.

____________________________________
____________________________________
____________________________________
____________________________________

# Previsão de Churn do Beta Bank

## Visão Geral do Projeto

Este projeto foi desenvolvido para o Beta Bank, com o objetivo de prever se um cliente deixará o banco em breve, ajudando a reduzir a perda de clientes, que é mais custosa do que manter os existentes. Utilizando dados sobre o comportamento passado dos clientes e rescisões de contratos, foi construído um modelo de machine learning para prever churn, visando alcançar um F1-score de pelo menos 0,59 no conjunto de teste. A métrica AUC-ROC foi calculada para comparação com o F1-score. Foram aplicadas técnicas de correção de desequilíbrio de classes, e o modelo final foi otimizado para atender aos critérios do projeto.

## Objetivos

- Carregar e preparar os dados do arquivo `Churn.csv`.
- Investigar o equilíbrio das classes e treinar um modelo inicial sem correção de desequilíbrio.
- Melhorar a qualidade do modelo usando pelo menos duas técnicas para corrigir o desequilíbrio de classes.
- Avaliar o modelo com F1-score e AUC-ROC no conjunto de teste.
- Realizar o teste final com o melhor modelo, alcançando um F1-score ≥ 0,59.
- Documentar todas as etapas e descobertas, mantendo o código limpo e estruturado.

## Descrição dos Dados

O conjunto de dados está no arquivo:

- **Churn.csv**: Dados sobre clientes do Beta Bank.

O arquivo contém as seguintes colunas:
- `RowNumber`: Índice das linhas de dados.
- `CustomerId`: Identificador único do cliente.
- `Surname`: Sobrenome.
- `CreditScore`: Pontuação de crédito.
- `Geography`: País de residência.
- `Gender`: Gênero.
- `Age`: Idade.
- `Tenure`: Período de maturação do depósito fixo (anos).
- `Balance`: Saldo da conta.
- `NumOfProducts`: Número de produtos bancários usados.
- `HasCrCard`: Cliente possui cartão de crédito (1 - sim; 0 - não).
- `IsActiveMember`: Cliente ativo (1 - sim; 0 - não).
- `EstimatedSalary`: Salário estimado.
- `Exited`: Cliente saiu (1 - sim; 0 - não, variável alvo).

### Condições do Negócio
- **Objetivo principal**: Prever churn (`Exited`) com F1-score ≥ 0,59 no conjunto de teste.
- **Métricas**: Avaliar F1-score e AUC-ROC, comparando os resultados.
- **Desequilíbrio de classes**: Corrigir o desequilíbrio identificado (20,37% saíram vs. 79,63% permaneceram).
- **Técnicas de correção**: Usar pelo menos duas abordagens para desequilíbrio (ex.: `class_weight`, undersampling).

*Nota*: Os dados são sintéticos e não incluem detalhes adicionais de contratos.

## Pré-processamento dos Dados

- **Tipos de Dados**:
  - O arquivo `Churn.csv` foi carregado com Pandas. As colunas `CreditScore`, `Age`, `Balance`, `EstimatedSalary` são numéricas contínuas; `Tenure`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `Exited` são numéricas discretas; `Geography` e `Gender` são categóricas.
  - A coluna `Tenure` estava em formato float, mas foi convertida para int após tratar valores ausentes.

- **Valores Ausentes**:
  - Identificados 909 valores ausentes em `Tenure`, interpretados como clientes com menos de 1 ano no banco. Foram substituídos por 0.
  - Nenhuma outra coluna apresentou valores ausentes.

- **Qualidade dos Dados**:
  - Não foram encontradas duplicatas ou inconsistências.
  - As colunas `RowNumber`, `CustomerId` e `Surname` foram removidas por serem irrelevantes para a previsão.
  - Variáveis categóricas (`Geography` e `Gender`) foram codificadas:
    - `Geography`: One-Hot Encoding (ex.: França, Espanha, Alemanha), com `drop_first=True` para evitar multicolinearidade.
    - `Gender`: Label Encoding (0 para Female, 1 para Male).
  - Variáveis numéricas contínuas (`CreditScore`, `Age`, `Balance`, `EstimatedSalary`) foram normalizadas com `StandardScaler` para padronizar as escalas.

## Etapas da Análise

1. **Divisão dos Dados**:
   - Os dados foram divididos em 60% para treinamento, 20% para validação e 20% para teste, usando `train_test_split` com `random_state=12345` e `stratify=y` para manter proporções das classes.
   - Criada a função `train_test_split` para modularizar o processo.

2. **Análise do Equilíbrio de Classes**:
   - A variável alvo `Exited` apresentou desequilíbrio: 79,63% (classe 0, permaneceram) e 20,37% (classe 1, saíram), visualizado com gráfico de barras.
   - Função `value_counts` e visualização com Seaborn foram usadas para análise.

3. **Modelo Inicial sem Correção de Desequilíbrio**:
   - Treinada uma Regressão Logística (`LogisticRegression`, `random_state=12345`, `solver='liblinear'`) no conjunto de treinamento.
   - Resultados no conjunto de teste:
     - F1-score: 0,3004 (baixo recall para classe 1: 0,20).
     - AUC-ROC: 0,7767 (melhor que F1, por considerar probabilidades).
     - Relatório de classificação: Precisão 0,83, recall 0,96, F1 0,89 para classe 0; precisão 0,59, recall 0,20, F1 0,30 para classe 1.

4. **Correção do Desequilíbrio de Classes**:
   - **Técnica 1: class_weight='balanced'**:
     - Aplicado na Regressão Logística para ajustar pesos das classes.
     - Resultados no conjunto de validação: F1-score de 0,5091, AUC-ROC de 0,7918.
   - **Técnica 2: Undersampling**:
     - Reduzida a classe majoritária (0) ao tamanho da classe minoritária (1) com amostragem aleatória (`random_state=12345`).
     - Treinada Regressão Logística com dados balanceados.
     - Resultados no conjunto de validação: F1-score de 0,5105, AUC-ROC de 0,7908.
   - Teste inicial com undersampling no conjunto de teste: F1-score de 0,5043, AUC-ROC de 0,7793.

5. **Otimização com Ajuste de Limiar**:
   - Testados limiares de 0,3 a 0,7 no modelo de undersampling (Regressão Logística).
   - Melhor limiar (0,6) gerou F1-score de 0,5163 na validação.
   - Resultados no teste: F1-score de 0,5121, AUC-ROC de 0,7793 (abaixo do alvo de 0,59).

6. **Modelo Random Forest com Undersampling**:
   - Substituída a Regressão Logística por Random Forest (`n_estimators=100`, `random_state=12345`) com dados balanceados por undersampling.
   - Resultados no conjunto de validação (limiar 0,5): F1-score de 0,6008.
   - Ajuste de limiar (0,6) gerou F1-score de 0,6129 na validação.
   - Teste final: F1-score de 0,6165, AUC-ROC de 0,8559, com precisão de 0,58 e recall de 0,65 para classe 1.

## Principais Descobertas

- **Desempenho do Modelo Inicial**:
  - A Regressão Logística sem correção teve baixo F1-score (0,3004) devido ao desequilíbrio de classes, com recall fraco para a classe 1 (0,20). O AUC-ROC (0,7767) foi superior, indicando boa separação das classes por probabilidades.

- **Correção do Desequilíbrio**:
  - Ambas as técnicas (`class_weight='balanced'` e undersampling) melhoraram o F1-score (~0,51 na validação), mas não atingiram 0,59 no teste inicial.
  - O Random Forest com undersampling e limiar otimizado (0,6) alcançou F1-score de 0,6165 no teste, superando o objetivo.

- **Comparação F1-score e AUC-ROC**:
  - O AUC-ROC (0,8559) foi consistentemente superior ao F1-score (0,6165), pois avalia o ranking de probabilidades, enquanto o F1-score depende de um limiar fixo.
  - O Random Forest melhorou ambas as métricas em relação à Regressão Logística.

## Conclusões e Recomendações

- **Modelo Recomendado**:
  - O modelo Random Forest com undersampling e limiar de 0,6 é a melhor escolha, com F1-score de 0,6165 e AUC-ROC de 0,8559 no conjunto de teste, superando o objetivo de 0,59.
  - O modelo é eficaz para prever churn, com bom recall (0,65) para a classe 1 (clientes que saem).

- **Justificativa**:
  - O Random Forest lidou melhor com o desequilíbrio e capturou padrões complexos nos dados.
  - O ajuste de limiar (0,6) otimizou o equilíbrio entre precisão e recall, maximizando o F1-score.
  - O AUC-ROC alto confirma a capacidade do modelo de distinguir clientes que saem dos que permanecem.

- **Impacto no Negócio**:
  - A previsão de churn permite ao Beta Bank identificar clientes em risco e oferecer incentivos (ex.: promoções, planos especiais) para retenção, reduzindo custos comparado à prospecção de novos clientes.
  - Recomenda-se implementar o modelo em produção, monitorando o desempenho e ajustando o limiar conforme necessário.

## Ferramentas e Tecnologias

- **Linguagem**: Python
- **Bibliotecas**: Pandas, NumPy, Scikit-learn (LogisticRegression, RandomForestClassifier, train_test_split, StandardScaler, LabelEncoder, f1_score, roc_auc_score), Seaborn, Matplotlib
- **Ambiente**: Jupyter Notebook

## Estrutura do Projeto

- `notebooks/`: Notebook Jupyter com pré-processamento, análise de classes, treinamento de modelos, otimização e teste final.
- `data/`: Arquivo `Churn.csv` (não incluído no repositório devido ao tamanho).
- `README.md`: Visão geral e documentação do projeto.