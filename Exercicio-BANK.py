import pandas as pd
from collections import Counter
from sklearn import preprocessing

dados = pd.read_csv('bank.csv', sep=';')

def identificarColunasCategoricas(dados):
    colunas_categoricas = []
    colunas_numericas = []
    cont_col = 0
    for coluns in dados.columns:
        if isinstance(dados.iloc[0, cont_col], str):
            colunas_categoricas.append(coluns)
        else:
            colunas_numericas.append(coluns)
        cont_col+=1

    return colunas_categoricas, colunas_numericas

col_cat, col_nun = identificarColunasCategoricas(dados)

# Selecionar colunas categoricas e aplicar o get_dummies para criar as colunas
dados_categorias = dados[col_cat]
cont_job = Counter(dados_categorias)

dados_categorias_norm = pd.get_dummies(dados_categorias, prefix=dados_categorias.columns)

# Dados para realizar a normalização dos dados númericos
dados_numericos = dados[col_nun]

normalizador = preprocessing.MinMaxScaler()
dados_normalizados = normalizador.fit_transform(dados_numericos)

# Juntar todos os dados
dados_finais = pd.DataFrame(dados_normalizados, columns = dados_numericos.columns)
dados_finais = dados_finais.join(dados_categorias_norm, how='left')

dados_finais.to_csv('dados_normalizados_final.csv', index=False, sep = ';')
