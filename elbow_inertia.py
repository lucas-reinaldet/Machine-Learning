
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

#isolar a coluna classe
iris.y = iris.iloc[:,4]

# isolar as colunas independentes
iris.x = iris.drop(columns=['class'],axis=1)

fig,ax =plt.subplots()

#Agrupar os objetos em um lista de vetores
X = np.array(list(zip(x0,x1,x2,x3))).reshape(len(x0),4)


KmeansModel = KMeans(n_clusters=2).fit(X)
print('########################')
# Imprimir os centroides
print(KmeansModel.cluster_centers_)
# Imprimir Inercia do  modelo obtido de clusters
print(KmeansModel.inertia_)
print('########################')

#Calcular as ditorções

inercias = []
K = range(1, 11)

for k in K:
    Kmeans = KMeans(n_clusters= k).fit(X)
    inercias.append(Kmeans.inertia_)


# Setup e impressão do gráfico Elbow
ax.plot(K, inercias)
ax.set(xlabel='Clusters', ylabel='Inercias',
       title='Método Elbow')

plt.savefig('elbow_dist.png')
plt.show()
