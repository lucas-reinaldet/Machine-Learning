"""
1. NORMALIZAR
2. SALVAR MODELO DE NORMALIZAÇÃO
3. DISCUSSÃO SOBRE ESTRATÉGIAS DE CÓDIGO
"""

import pandas as pd
import sklearn
from sklearn import preprocessing

dados = pd.read_csv("dados_normalizar.csv", sep = ";")
#print(dados.head())

#1. obter o vetor de valores núméricos
#1.1 Segmentar as colunas que são numéricas, para normalização quantititativa
dados_num = dados.drop(columns = ['sexo'])
dados_categorias = dados['sexo']
#print(dados_num)
#print(dados_categorias)

#1.2 Obter o vetor numérico a partir dos dados que são numéricos
#"volta qui"
dados_num.X = dados_num.values
#print(dados_num.X)

#2.Normalizar
#2.1 Utilizando o método Min Max manualmente
# Z = (x - min(DADOS) / (max(DADOS)-min(DADOS)))

dados_num.normalizados_minmax = (dados_num-dados_num.min()/(dados_num.max()-dados_num.min()))
#print(dados_num.normalizados_minmax)

#2.2 Utilizando a média manual
# Z = (x - media(atributo)/desvio padrão do atributo)

dados_num.normalizados_media = (dados_num-dados_num.mean())/dados_num.std()
#print(dados_num.normalizados_media)

#altura (cm) 189 ==> peso1 (-1 : 1)
#Peso (kg) 75,00 ==> peso2  (-1 : 1)


#3. Utilizando o MinMaxScaler()
#3.1 Criar um objeto normalizador
normalizador = preprocessing.MinMaxScaler()

dados_num.modelo_normalizador = normalizador.fit(dados_num)# o método fit() cria o modelo normalizado
dados_num.normalizados_scaler = normalizador.fit_transform(dados_num)# o método fit_transform() normalizado
# print("Normalizados")
# print(dados_num.modelo_normalizador)
# print(dados_num.normalizados_scaler)


#NORMALIZAR DADOS CATEGÓRICOS
dados_categorias.normalizados = pd.get_dummies(dados_categorias)
#print('########################')
#print('Atributo sexo normalizado')
#print(dados_categorias.normalizados)

#INCORPORAR OS SEGMENTOS NORMALIZADOS EM UM SÓ DATAFRAME
print(type(dados_num.normalizados_scaler))

#Converter o NPARRAY, com os dados numéricos e normalizados em um Data Frame
dados_finais = pd.DataFrame(dados_num.normalizados_scaler, columns = ['Idade', 'Altura', 'Peso'])
print(dados_finais)

#agregar os dados categóricos normalizados ao objeto dados finais
dados_finais = dados_finais.join(dados_categorias.normalizados, how='left')
print(dados_finais)

#Salvar os dados em arquivo csv
dados_finais.to_csv('dados_normalizados_final.csv', index=False, sep=';')

print('#' * 30)

#normalizar os arquivos
# housing.data.csv (descrição dos dados em housing.names.txt

# vote.csv

