## Atividade
# 1 - Em um arquivo / Programa: Abrir dados, obter o modelo normalizador e salvar em disco
# 2 - Em outro arquivo / programa: Abrir o modelo normalizador, simular novas instancias e, normaliza-las

import pandas as pd
import sklearn
from sklearn import preprocessing
from pickle import dump, load

dados = pd.read_csv('BostonHousing.csv', sep=';')
# print(dados.head)

# Normalizar usando a média e o desvio padrão
dados.normalizados_dp = (dados - dados.mean() / dados.std())
# print(dados.normalizados_dp)

# Normalizar atraves da utilização de maximo e minimo
dados.normalizados_minmax = (dados - dados.min()) / (dados.max() - dados.min())
# print(dados.normalizados_minmax)

# Instânciar o normalizador Scaler
normalizador = preprocessing.MinMaxScaler()

dados.normalizados_scaler = normalizador.fit_transform(dados)
# print(dados.normalizados_scaler)

dados_finais = pd.DataFrame(dados.normalizados_scaler, columns=dados.columns)
print(dados_finais)

# agregar os dados categóricos normalizados ao objeto dados finais
# Como não possuimos dados categoricos, não será necessário realizar
# a ação de agregar os dados categoricos normalizados no data frame
# dados_finais = dados_finais.join(dados_finais.normalizados,how='left')

dados_finais.to_csv('modelo_dados_normalizados.csv', index=False, sep=';')

dados.modelo_nomalizador = normalizador.fit(dados)
dump(dados.modelo_nomalizador, open('MinMaxScaler_model.pkl', 'wb'))
