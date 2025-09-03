import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# --- 1. Carregamento e Limpeza dos Dados ---
print("Iniciando o processo...")
df = pd.read_csv('desafio_indicium_imdb.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(df.columns[0], axis=1)

print("Realizando limpeza e pré-processamento...")
df['Released_Year'] = df['Released_Year'].replace('PG', '1979')
df['Released_Year'] = pd.to_numeric(df['Released_Year'])
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(int)
df['Gross'] = df['Gross'].str.replace(',', '').astype(float)

df['Certificate'].fillna('Not Rated', inplace=True)
df['Meta_score'].fillna(df['Meta_score'].median(), inplace=True)
df['Gross'].fillna(df['Gross'].median(), inplace=True)
print("Limpeza concluída.")

# --- 2. Análise Exploratória de Dados (EDA) ---
print("Gerando visualizações da Análise Exploratória...")
sns.set_style("whitegrid")

# Criar o diretório para salvar as imagens, se ele não existir
output_dir = "imagens"
os.makedirs(output_dir, exist_ok=True)


plt.figure(figsize=(15, 10))
plt.suptitle('Análise Exploratória das Variáveis Numéricas', fontsize=16)
plt.subplot(2, 2, 1)
sns.histplot(df['IMDB_Rating'], bins=20, kde=True).set_title('Distribuição da Nota IMDB')
plt.subplot(2, 2, 2)
sns.histplot(df['Meta_score'], bins=20, kde=True).set_title('Distribuição do Meta Score')
plt.subplot(2, 2, 3)
sns.histplot(df['Gross'], bins=20, kde=True).set_title('Distribuição do Faturamento')
plt.subplot(2, 2, 4)
sns.histplot(df['No_of_Votes'], bins=20, kde=True).set_title('Distribuição do Número de Votos')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, "distribuicoes_numericas.png"))

genre_counts = df['Genre'].str.split(', ').explode().value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(x=genre_counts.head(15).values, y=genre_counts.head(15).index, palette='viridis').set_title('Top 15 Géneros de Filmes')
plt.xlabel('Número de Filmes')
plt.ylabel('Género')
plt.savefig(os.path.join(output_dir, "top_generos.png"))

plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=np.number)
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f").set_title('Mapa de Calor de Correlação')
plt.savefig(os.path.join(output_dir, "mapa_correlacao.png"))
print(f"Visualizações da EDA salvas na pasta '{output_dir}'.")

# --- 3. Preparação para Modelagem ---
print("Preparando dados para modelagem...")
genres = df['Genre'].str.get_dummies(sep=', ')
df = pd.concat([df, genres], axis=1)

top_certificates = df['Certificate'].value_counts().nlargest(5).index
for cert in top_certificates:
    df[f'Cert_{cert}'] = np.where(df['Certificate'] == cert, 1, 0)

top_directors = df['Director'].value_counts().nlargest(10).index
for director in top_directors:
    df[f'Dir_{director}'] = np.where(df['Director'] == director, 1, 0)

top_stars = df['Star1'].value_counts().nlargest(10).index
for star in top_stars:
    df[f'Star_{star}'] = np.where(df['Star1'] == star, 1, 0)

features = ['Released_Year', 'Runtime', 'Meta_score', 'No_of_Votes', 'Gross'] + \
           list(genres.columns) + \
           [f'Cert_{cert}' for cert in top_certificates] + \
           [f'Dir_{director}' for director in top_directors] + \
           [f'Star_{star}' for star in top_stars]

X = df[features]
y = df['IMDB_Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dados preparados.")

# --- 4. Treinamento e Avaliação do Modelo ---
print("Treinando o modelo RandomForestRegressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Resultados da Avaliação do Modelo ---")
print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# --- 5. Previsão para um Filme Específico ---
print("\nRealizando previsão para 'The Shawshank Redemption'...")
new_movie_data = {
    'Released_Year': [1994], 'Runtime': [142], 'Meta_score': [80.0],
    'No_of_Votes': [2343110], 'Gross': [28341469.0], 'Genre': ['Drama'],
    'Certificate': ['A'], 'Director': ['Frank Darabont'], 'Star1': ['Tim Robbins']
}
new_movie_df = pd.DataFrame(new_movie_data)

for genre in genres.columns:
    new_movie_df[genre] = 1 if genre in new_movie_df['Genre'][0] else 0
for cert in top_certificates:
    new_movie_df[f'Cert_{cert}'] = 1 if cert == new_movie_df['Certificate'][0] else 0
for director in top_directors:
    new_movie_df[f'Dir_{director}'] = 1 if director == new_movie_df['Director'][0] else 0
for star in top_stars:
    new_movie_df[f'Star_{star}'] = 1 if star == new_movie_df['Star1'][0] else 0

new_movie_df = new_movie_df[features]
predicted_rating = rf_model.predict(new_movie_df)

print(f"\nA nota prevista para o filme é: {predicted_rating[0]:.2f}")

# --- 6. Salvar o Modelo ---
model_filename = 'imdb_rating_predictor.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"\nModelo salvo com sucesso como '{model_filename}'")
print("\nProcesso finalizado.")
