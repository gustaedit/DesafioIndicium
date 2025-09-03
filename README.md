# Desafio Cientista de Dados: Análise Preditiva de Filmes do IMDB

## 1. Visão Geral do Projeto

Este projeto foi desenvolvido como a minha submissão para o desafio de Cientista de Dados da Indicium. O objetivo foi realizar uma análise completa de um conjunto de dados de filmes do IMDB para orientar o estúdio "PProductions" nas suas próximas produções.

O trabalho abrange desde a limpeza e análise exploratória dos dados até a construção de um modelo de Machine Learning para prever a nota de um filme no IMDB.

## 2. Estrutura do Repositório

-   `desafio_indicium_imdb.csv`: O conjunto de dados original.
-   `analise_filmes.py`: Script Python com todo o fluxo de trabalho, incluindo análise, modelagem e geração de resultados.
-   `requirements.txt`: Arquivo com as dependências do projeto para fácil instalação.
-   `imdb_rating_predictor.pkl`: O modelo de Machine Learning treinado e pronto para uso.
-   `/imagens/`: Pasta contendo os gráficos gerados pela análise exploratória.

## 3. Como Instalar e Executar o Projeto

### Pré-requisitos

-   Python 3.7+

### Passos para Execução

1.  **Clonar o repositório:**
    ```bash
    git clone <URL-DO-REPOSITORIO>
    cd <NOME-DO-REPOSITORIO>
    ```

2.  **Criar e ativar um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  
    ```

3.  **Instalar as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Executar o script principal:**
    ```bash
    python analise_filmes.py
    ```

    Após a execução, os resultados serão impressos no terminal, e os arquivos de imagem (gráficos) e o modelo (`.pkl`) serão salvos no diretório.

## 4. Análises e Respostas

As respostas detalhadas para as perguntas do desafio estão documentadas no relatório principal (Jupyter Notebook ou PDF), mas um resumo das minhas conclusões está abaixo.

### Análise Exploratória (EDA)

-   **Hipótese 1: A popularidade impulsiona o faturamento.** O mapa de calor de correlação mostrou uma forte correlação positiva (0.61) entre `No_of_Votes` e `Gross`. Isso sugere que filmes que geram mais engajamento e discussão (mais votos) tendem a ter um desempenho financeiro significativamente melhor.
-   **Hipótese 2: Filmes de drama são a aposta mais segura para aclamação.** O gênero "Drama" é o mais frequente no dataset, indicando que filmes deste gênero compõem a maior parte das obras de alta avaliação. Isso pode significar que são mais propensos a receber reconhecimento crítico e do público.
-   **Hipótese 3: A duração ideal de um filme aclamado é de cerca de 2 horas.** A distribuição da variável `Runtime` mostrou uma concentração de filmes em torno de 120-130 minutos, sugerindo um "ponto ideal" de duração para filmes bem-sucedidos.

### Respostas às Perguntas

-   **Qual filme eu recomendaria para uma pessoa que não conheço?**
    Recomendaria **"The Shawshank Redemption"**. É o filme com a maior nota no IMDB (9.3), aclamado universalmente, e pertence ao gênero Drama, que possui um apelo amplo e acessível.

-   **Quais são os principais fatores relacionados com alta expectativa de faturamento?**
    Os principais fatores são:
    1.  **Popularidade (`No_of_Votes`):** O fator mais forte. Um alto número de votos indica engajamento e, consequentemente, maior bilheteira.
    2.  **Qualidade Percebida (`IMDB_Rating` e `Meta_score`):** Boas avaliações geram boca-a-boca positivo.
    3.  **Gênero:** Filmes de Ação e Aventura historicamente têm alto potencial de bilheteira.

-   **Quais insights podem ser tirados com a coluna Overview?**
    A coluna `Overview` é uma mina de dados para Processamento de Linguagem Natural (PLN). A partir dela, **sim, é possível inferir o gênero do filme**. Usando técnicas como TF-IDF, podemos transformar as sinopses em vetores numéricos e treinar um modelo de classificação de texto para prever os gêneros com base nas palavras e temas presentes no resumo.

### Previsão da Nota do IMDB

-   **Tipo de Problema:** É um problema de **Regressão**, pois o alvo (`IMDB_Rating`) é um valor contínuo.
-   **Variáveis Utilizadas:** Utilizei `Released_Year`, `Runtime`, `Meta_score`, `No_of_Votes`, e `Gross` como variáveis numéricas. Para as categóricas (`Genre`, `Certificate`, `Director`, `Star1`), apliquei One-Hot Encoding nas categorias mais frequentes para evitar dimensionalidade excessiva.
-   **Modelo Escolhido:** Optei pelo **Random Forest Regressor**.
    -   **Prós:** Robusto, lida bem com relações não-lineares e misturas de tipos de dados, e fornece a importância das features.
    -   **Contras:** É um modelo de "caixa-preta", sendo menos interpretável que um modelo linear.
-   **Métrica de Performance:** Escolhi o **Erro Quadrático Médio (MSE)** porque ele penaliza erros maiores de forma mais significativa, o que é crucial para evitar previsões muito distantes do valor real.

-   **Qual seria a nota do IMDB para o filme de exemplo?**
    Para um filme com as características de "The Shawshank Redemption", o meu modelo previu uma nota de **8.78**.
