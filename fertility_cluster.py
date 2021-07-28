import pandas as pd
import math
from sklearn.cluster import KMeans


# Função retirada de https://jtemporal.com/kmeans-and-elbow-method/
def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = len(wcss), wcss[-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)

    return distances.index(max(distances)) + 2


dados = pd.read_csv('fertility_Diagnosis.txt', sep=',')
# print(dados.head())
X = dados.values

# Determinar a matriz de inercias WCSS
inercias = []

K = range(1,101)
for k in K:
    KmeansModel = KMeans(n_clusters=k, random_state=1).fit(X)
    inercias.append(KmeansModel.inertia_)

K = optimal_number_of_clusters(inercias)
print(K)
