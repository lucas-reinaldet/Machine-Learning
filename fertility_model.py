import numpy
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dados = pd.read_csv('fertility_Diagnosis.txt', sep=',')

# Segmento da Classe
dados.atributos = dados.drop(columns=['Output'])
dados.classe = dados['Output']
cont_dados = Counter(dados.classe)
print('Frequencia de classes', cont_dados)

#Balancear os dados

# Criar um objeto com o Smote de Base
oversample = SMOTE()
X, Y = oversample.fit_resample(dados.atributos, dados.classe)

# O SMOTE recebe as partes originais e rtorna equivalentes partes balanceadas

cont_Y = Counter(Y)
print('Frequencia de classes após balanceamento', cont_Y)

print('Linhas de Dados mão balanceados', dados.count()+1)
print('Linhas de Dados balanceados', X.count()+1)

# normalizar os dados

# Segmentar em dados para aprendizado e dados para testes
# Conjunto de atributos para aprendizado
# Conjunto de Classes para Aprendizados, paralelo ao conjunto de atributos

# Conjunto de Atributos para Testes
# Conjunto de classes para testes

X_train, X_test, Y_train, Y_teste = train_test_split(X, Y, test_size=0.3)

rf_fertility = RandomForestClassifier().fit(X_train, Y_train)

# Pré teste do modelo
# Passar o X_test para o modelo obtido, retornando para um vetor
y_previsto = rf_fertility.predict(X_test)

print('\nClasses previstas com o X_test')
print(y_previsto)

# Para verificar a acurácia do modelo obtido comparar y_previsto com o Y_test
from sklearn import metrics

#verificar a acurácia do modelo obtido
print('\nAcurácia global', metrics.accuracy_score(Y_teste, y_previsto))

#validação usando Cross
# O cross validate requer: modelo obtido, atributos para teste X e Y, e numero de segmentos
cv_results = cross_validate(rf_fertility, X_train, Y_train, cv=10)

print('\n', cv_results.keys())
print(cv_results['test_score'])

# Calculaa média
print('\nScore médio após cross validation')
print(numpy.mean(cv_results['test_score']))

# Matriz de contigência

#Passa para a confusion matrix:
## Classes reservadas para testes (Y_test)
## Classes obtidas após o teste (Y_previsto)

matriz_contigencia = confusion_matrix(Y_teste, y_previsto)

print('\nMatriz de contigência')
print(matriz_contigencia)

# Classiication report
from sklearn.metrics import classification_report

# Parametro: Y_test, y_previsto, nomes das classes
nomes_classes = rf_fertility.classes_

# print('\nNomes das Classes')
# print(nomes_classes)

class_report = classification_report(Y_teste, y_previsto, target_names=rf_fertility.classes_)
print(class_report)
# Salvar modelo obtido em disco
from pickle import dump

dump(rf_fertility, open('fertility_randon_forest_model.pkl', 'wb'))
