# Atividade
# 1.1 Normalizar os dados do arquivo BreastCancer
# 1.1.1 A Coluna deg_malig é numerica, as demais são categoricas
# 2. Obter modelo (RandoForests)
# 2.1 Gerar relatório de acurácia (Classification report)
# 3. Salvar o modelo em disco
# Prazo: 20 Minutos

from sklearn import preprocessing
import numpy
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dados = pd.read_csv('breast-cancer.csv', sep=',')

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
dados_categorias = dados[col_cat[:-1]]
cont_job = Counter(dados_categorias)

dados_categorias_norm = pd.get_dummies(dados_categorias, prefix=dados_categorias.columns)

# Dados para realizar a normalização dos dados númericos
dados_numericos = dados[col_nun]

normalizador = preprocessing.MinMaxScaler()
dados_normalizados = normalizador.fit_transform(dados_numericos)

# Juntar todos os dados
dados_finais = pd.DataFrame(dados_normalizados, columns = dados_numericos.columns)
dados_finais = dados_finais.join(dados_categorias_norm, how='left')

dados_finais.to_csv('breast_cancer_dados_normalizados_final.csv', index=False, sep = ';')

# Criar um objeto com o Smote de Base
oversample = SMOTE()
X, Y = oversample.fit_resample(dados_finais, dados[col_cat[-1]])

# O SMOTE recebe as partes originais e rtorna equivalentes partes balanceadas

cont_Y = Counter(Y)
print('Frequencia de classes após balanceamento', cont_Y)

print('Linhas de Dados mão balanceados', dados.count()+1)
print('Linhas de Dados balanceados', X.count()+1)

X_train, X_test, Y_train, Y_teste = train_test_split(X, Y, test_size=0.3)

rf_breast_cancer = RandomForestClassifier().fit(X_train, Y_train)

y_previsto = rf_breast_cancer.predict(X_test)

print('\nClasses previstas com o X_test')
print(y_previsto)

cv_results = cross_validate(rf_breast_cancer, X_train, Y_train, cv=10)

print('\n', cv_results.keys())
print(cv_results['test_score'])

print('\nScore médio após cross validation')
print(numpy.mean(cv_results['test_score']))

matriz_contigencia = confusion_matrix(Y_teste, y_previsto)

print('\nMatriz de contigência')
print(matriz_contigencia)

nomes_classes = rf_breast_cancer.classes_

# print('\nNomes das Classes')
# print(nomes_classes)

class_report = classification_report(Y_teste, y_previsto, target_names=rf_breast_cancer.classes_)
print(class_report)
# Salvar modelo obtido em disco
from pickle import dump

dump(rf_breast_cancer, open('breast_cancer_forest_model.pkl', 'wb'))
