
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import euclidean, cdist

#objeto para os dados
iris = pd.read_csv('iris.csv') #, usecols=[0,1,2,3]

#Dados para plotagem
x0=iris.iloc[:,0]
x1=iris.iloc[:,1]
x2=iris.iloc[:,2]
x3=iris.iloc[:,3]


iris.y = iris.iloc[:,4]#isolar a coluna classe
iris.x = iris.drop(columns=['class'],axis=1) #isolar as colunas independentes

fig,ax =plt.subplots()

#Agrupar os objetos em um lista de vetores
X = np.array(list(zip(x0,x1,x2,x3))).reshape(len(x0),4)


KmeansModel = KMeans(n_clusters=2).fit(X)
print('########################')
print(KmeansModel.cluster_centers_)
print(KmeansModel.inertia_)
print('########################')

#Calcular as ditorções
distorcoes=[]

K = range(1,11) # Criando uma lista de valores (de 1 a 10 clusters para avaliar o ganho)
for k in K:
    # obtenção provisória de um modelo de clusters com k grupos
    KmeansModel = KMeans(n_clusters=k).fit(X)

    distorcoes.append(
        sum(
            np.min(
                cdist(X, KmeansModel.cluster_centers_, 'euclidean')
                , axis=1)) / X.shape[0])

print(distorcoes)

# Setup e impressão do gráfico Elbow
ax.plot(K, distorcoes)
ax.set(xlabel='Clusters', ylabel='Distorcao',
       title='Método Elbow')

plt.savefig('elbow_dist.png')
plt.show()
