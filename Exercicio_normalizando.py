# NORMALIZAÇÃO DE DADOS
# altura 1.70
# peso 85

# objetivos
# 1. normalizar os dados
# 2. salvar modelo de normalização
# 3. discussão sobre estratégias de código

import pandas as pd
import sklearn
from sklearn import preprocessing

dados = pd.read_csv('housing.data.csv', sep=';')
print(dados.head())

#1. obter o vetor de valores numéricos
#1.1. segmentar as colunas que são numéricas para normalização quantitiva

#1.2 obter o vetor numerico a partir dos dados que são numéricos

#2. Normalizar
#2.1. Utilizando o método Min Max manualmente
# z = (x - min(DADOS) / max(DADOS) - min(DADOS))

dados.normalizados = (dados - dados.min())/(dados.max()-dados.min())
print(dados.normalizados)

#2.2 Utilizando a media manual
# Z = (x-media(atributo)/desvio padrao do atributo)

dados.normalizados_media = (dados - dados.mean())/ (dados.std())

print(dados.normalizados_media)

# 3. Utilizando o MinMaxScaler()
#3.1 Criar um objeto nomarlizador
normalizador = preprocessing.MinMaxScaler()

#o metodo cria o modelode de normalização com base nos dados
dados.modelo_normalizador = normalizador.fit(dados)

#metodo fit_transform() normaliza os dados retornando um objeto do tipo nparray
dados.normalizados_scaler = normalizador.fit_transform(dados)
print(dados.normalizados_scaler)
print(normalizador.fit_transform(dados))

# normalizar dados Categoricos
# print('#######')

# incorporar dados normalizados em um dataframe
# dados_categorias.normalizados = pd.get_dummies(dados_categorias)

print(type(dados.normalizados_scaler))

dados_finais = pd.DataFrame(dados.normalizados_scaler, columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])
print(dados_finais)

dados_finais.to_csv('housing_data_normalizados_final.csv', index=False, sep=';')

