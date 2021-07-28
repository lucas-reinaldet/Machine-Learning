import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import euclidean, cdist


fert = pd.read_csv('fertility_Diagnosis.txt')

#dados para  plotagem
x0=fert.iloc[:,0]
x1=fert.iloc[:,1]
x2=fert.iloc[:,2]
x3=fert.iloc[:,3]
x4=fert.iloc[:,4]
x5=fert.iloc[:,5]
x6=fert.iloc[:,6]
x7=fert.iloc[:,7]
x8=fert.iloc[:,8]
x9=fert.iloc[:,9]

# print(fert.head())
fert.y = fert.iloc[:,9] #isolar a coluna output
fert.x = fert.drop(columns=['Output'], axis=1)

print(fert.y.head())
print(fert.x.head())

# Agrupar os objetos em um data frame
Xfert = np.array(list(zip(x0, x1, x2, x3, x4, x5, x6, x7, x8))).reshape(len(x0), 9)

KmeansModel = KMeans(n_clusters=2).fit(Xfert)
# print('########################')
# print(KmeansModel.cluster_centers_) #imprimir os centroides
# print(KmeansModel.inertia_) #imprimir a inércia do modelo obtido de clusters
# print('########################')

# #Calculo da variação/Distorção
distorcoes=[]
inercias=[]
fig, ax = plt.subplots()
Kdist = range(1,11) # Criando uma lista de valores (de 1 a 10 clusters para avaliar ganho)

for k in Kdist:
    #obtenção provisório de um  modelo de clusters com k grupos
    kmeansModel = KMeans(n_clusters=k).fit(Xfert)
    distorcoes.append(
    sum(
        np.min(
        cdist(Xfert, kmeansModel.cluster_centers_, 'euclidean')
        , axis=1)) / Xfert.shape[0])
    inercias.append(kmeansModel.inertia_)


# # Setup e impressão do gráfico Elbow
# # Setup e impressão do gráfico Elbow
color = 'tab:red'
ax.plot(Kdist, distorcoes, color='tab:red')
ax.set_xlabel('Clusters')
ax.set_ylabel('Distorções')
ax2 = ax.twinx()
ax2.plot(Kdist, inercias, color='tab:blue')
ax2.set_ylabel('Inércias')
# ax.set(xlabel="Clusters", title="Método Elbow")
plt.show()
