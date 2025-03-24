# Kobe Bryant Shot Prediction: End-to-End MLOps Project

[Versão em Português abaixo]

## About the Project

This project implements a complete Machine Learning Operations (MLOps) pipeline to predict whether Kobe Bryant's shots (nicknamed "Black Mamba") will result in a basket or a miss. We use two modeling approaches: regression and classification. The project follows the Microsoft Team Data Science Process (TDSP) structure, ensuring organization and reproducibility.

## Objective

Develop, train, and deploy machine learning models capable of predicting the outcome of Kobe Bryant's shots, using historical data from his NBA career. The project aims to demonstrate a complete MLOps workflow, from data preparation to model monitoring in production.

## Tools and Technologies

- Python: Main programming language
- MLflow: Experiment tracking, model registry, and deployment
- PyCaret: Machine learning automation for model training
- Scikit-learn: Machine learning algorithms
- Streamlit: Monitoring dashboard
- Git: Version control

## Project Structure (TDSP)

The project follows the Microsoft TDSP structure:

Code/ - All source code

DataPrep/ - Data preparation code
Model/ - Training and evaluation code
Operationalization/ - Deployment and monitoring code

Data/ - Data directories

Raw/ - Raw data
Processed/ - Processed data
Modeling/ - Data for modeling

Docs/ - Documentation

Project/ - Project documents
DataReport/ - Data reports and analysis
Model/ - Model documentation

README.md - This file
requirements.txt - Project dependencies

## Implemented Pipelines

1. PreparacaoDados: Pipeline for data preparation and cleaning
   - Removal of missing values
   - Selection of relevant features
   - Division into training and testing sets

2. Treinamento: Pipeline for training and evaluating models
   - Logistic Regression with PyCaret/sklearn
   - Decision Tree with PyCaret/sklearn
   - Evaluation with metrics (log loss, F1 score)
   - Selection of the best model

3. PipelineAplicacao: Production application pipeline
   - Loading of trained model
   - Application on production data
   - Performance monitoring

## Installation and Setup

# Clone the repository
git clone https://github.com/luisabrasildematos/kobe-shot-prediction-mlops.git
cd kobe-shot-prediction-mlops

# Install dependencies
pip install -r requirements.txt

# Download the data (datasets in .parquet format)
# Place in Data/Raw/

## Project Usage

### Data Processing
python Code/DataPrep/processamento_dados.py

### Model Training
python Code/Model/treinamento.py

### Running the Application Pipeline
python Code/Operationalization/aplicacao.py

### Starting the Monitoring Dashboard
streamlit run Code/Operationalization/dashboard.py

## Monitoring and Maintenance

The project includes strategies for:
- Monitoring model health in production
- Reactive and predictive retraining strategies
- Streamlit dashboard for metrics visualization

----------------------------
VERSAO EM PORTUGUES
----------------------------

# Previsão de Arremessos do Kobe Bryant: Projeto End-to-End de MLOps

## Sobre o Projeto

Este projeto implementa um pipeline completo de Machine Learning Operations (MLOps) para prever se os arremessos do Kobe Bryant (apelidado de "Black Mamba") resultarão em cesta ou erro. Utilizamos duas abordagens de modelagem: regressão e classificação. O projeto segue a estrutura do Microsoft Team Data Science Process (TDSP), garantindo organização e reprodutibilidade.

## Objetivo

Desenvolver, treinar e implantar modelos de machine learning capazes de prever o resultado dos arremessos do Kobe Bryant, utilizando dados históricos de sua carreira na NBA. O projeto visa demonstrar um fluxo de trabalho de MLOps completo, desde a preparação dos dados até o monitoramento do modelo em produção.

## Ferramentas e Tecnologias

- Python: Linguagem de programação principal
- MLflow: Rastreamento de experimentos, registro de modelos e implantação
- PyCaret: Automação de machine learning para treinamento de modelos
- Scikit-learn: Algoritmos de machine learning
- Streamlit: Dashboard para monitoramento
- Git: Controle de versão

## Estrutura do Projeto (TDSP)

O projeto segue a estrutura Microsoft TDSP:

Code/ - Todo o código fonte

DataPrep/ - Código para preparação de dados
Model/ - Código para treinamento e avaliação
Operationalization/ - Código para implantação e monitoramento

Data/ - Diretórios de dados

Raw/ - Dados brutos
Processed/ - Dados processados
Modeling/ - Dados para modelagem

Docs/ - Documentação

Project/ - Documentos do projeto
DataReport/ - Relatórios e análises de dados
Model/ - Documentação dos modelos

README.md - Este arquivo
requirements.txt - Dependências do projet

## Pipelines Implementados

1. PreparacaoDados: Pipeline que prepara e limpa os dados
   - Remoção de valores ausentes
   - Seleção de features relevantes
   - Divisão em conjuntos de treino e teste

3. Treinamento: Pipeline para treinar e avaliar modelos
   - Regressão Logística com PyCaret/sklearn
   - Árvore de Decisão com PyCaret/sklearn
   - Avaliação com métricas (log loss, F1 score)
   - Seleção do melhor modelo

4. PipelineAplicacao: Pipeline de aplicação em produção
   - Carregamento do modelo treinado
   - Aplicação em dados de produção
   - Monitoramento de performance

## Instalação e Configuração

# Clonar o repositório
git clone https://github.com/luisabrasildematos/kobe-shot-prediction-mlops.git
cd kobe-shot-prediction-mlops

# Instalar dependências
pip install -r requirements.txt

# Baixar os dados (datasets em .parquet)
# Colocar em Data/Raw/

## Uso do Projeto

### Processamento de Dados
python Code/DataPrep/processamento_dados.py

### Treinamento do Modelo
python Code/Model/treinamento.py

### Execução do Pipeline de Aplicação
python Code/Operationalization/aplicacao.py

### Iniciar o Dashboard de Monitoramento
streamlit run Code/Operationalization/dashboard.py

## Monitoramento e Manutenção

O projeto inclui estratégias para:
- Monitoramento da saúde do modelo em produção
- Estratégias reativas e preditivas de retreinamento
- Dashboard Streamlit para visualização de métricas
