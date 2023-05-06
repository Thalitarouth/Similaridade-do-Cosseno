# Similaridade-do-Cosseno
Desafio de Algebra Linear
```{Python}

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

# carregar dados do arquivo CSV
dados = pd.read_excel('excel_perfume.xlsx')

# selecionar as colunas relevantes para comparação
dados_selecionados = dados[['Name', 'Notes']]

# transformar os dados em um formato de texto unificado para vetorização
dados_texto = dados_selecionados.apply(lambda x: ' '.join(x.astype(str)), axis=1)

# criar o vetorizador TF-IDF
vetorizador = TfidfVectorizer()

# vetorizar as descrições dos produtos
vetores = vetorizador.fit_transform(dados_texto)

# exemplo das nota perfume fornecido pelo usuário
nota_perfume = input('Digite a nota de perfume que mais te agrada: ').split()

# transformar o exemplo em um vetor numérico
nota_perfume_vetor = vetorizador.transform([' '.join(map(str, nota_perfume + [''] * (len(dados_selecionados.columns) - len(nota_perfume))))])

# calcular a similaridade do cosseno entre a nota_perfume fornecido pelo usuário e outra nota_perfume
similaridades = cosine_similarity(nota_perfume_vetor, vetores)

# obter os índices da nota_perfume ordenada por ordem decrescente de similaridade
indices_similares = similaridades.argsort()[0][::-1]

# selecionar os 10 perfumes com notas mais similares,
top_similares = []
for i in indices_similares:
    if dados.loc[i, 'Name'] != nota_perfume:
        top_similares.append(i)
    if len(top_similares) >= 10:
        break

# adicionar as colunas de similaridade do cosseno e ângulo do cosseno
dados_selecionados['Cosine Similarity'] = similaridades[0]
dados_selecionados['Cosine Angle'] = [math.degrees(math.acos(similarity)) for similarity in similaridades[0]]

# imprimir os 10 perfumes mais similares com as colunas adicionais
print("A nota_perfume mais similar a", ' '.join(map(str, nota_perfume)), "são:")
display(dados_selecionados.loc[top_similares, ['Name', 'Notes', 'Cosine Similarity', 'Cosine Angle']])
``` 

