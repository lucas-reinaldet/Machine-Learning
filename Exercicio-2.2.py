from pickle import dump, load
import numpy as np
import random
import pandas as pd

modelo_normalizador = load(open('MinMaxScaler_model.pkl', 'rb'))

novos_dados = []

for i in range(1, 100):
    novos = []
    novos.append(random.random())
    novos.append(random.randint(0,100))
    novos.append(random.randint(0, 30) + random.random())
    novos.append(random.randint(0,1))
    novos.append(random.random())
    novos.append(random.randint(0, 10) + random.random())
    novos.append(random.randint(0, 120) + random.random())
    novos.append(random.randint(0, 80) + random.random())
    novos.append(random.randint(0, 10))
    novos.append(random.randint(200, 500))
    novos.append(random.randint(200, 500) + random.random())
    novos.append(random.randint(0, 25) + random.random())
    novos.append(random.randint(2, 50) + random.random())
    novos_dados.append(novos)

novos_dados = np.array(novos_dados)
novos_dados = pd.DataFrame(novos_dados, columns=['idade','altura','Peso'])