
import pandas as pd
from pickle import  load
import numpy as np

#Abrir o modelo
rf_fertility = load(open('fertility_randon_forest_model.pkl', 'rb'))

#nova instancia
novo_paciente = np.array([[-0.33, 0.5, 1, 1, 0, -1, 0.8, 0, 0.88]])

print('Diagnostico indicado para o novo paciente: ', rf_fertility.predict(novo_paciente))