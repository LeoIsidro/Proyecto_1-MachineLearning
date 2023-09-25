import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import pandas as pd
from  skimage.io import imread, imshow
from knn import KNN
import pywt
import pywt.data
import os
from sklearn.neighbors import KNeighborsClassifier

def Get_Feacture(picture, cortes):
  LL = picture
  for i in range(cortes):
     LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
  return LL.flatten().tolist()

def cargar_dataset():

    carpeta_imagenes = './imagenes_1/'

    # Se ordena los archivos segun el nombre asignado
    archivos = os.listdir(carpeta_imagenes)

    clases = []
    vectores_caracteristicos = []


    for archivo in archivos:
        # Verifica si el archivo es una imagen (puedes agregar más extensiones si es necesario)
        if archivo.endswith(('.png')):
            # Construye la ruta completa del archivo
            ruta_completa = os.path.join(carpeta_imagenes, archivo)
            imagen = imread(ruta_completa)

            # Se añade la clase correspondiente al vector clases 
            if int(archivo[1]) < 1:
                clases.append(int(archivo[2]))
            else:
                clases.append(int(archivo[1:3]))

            # Se añade el vector caracteristico de cada imagen
            vectores_caracteristicos.append(Get_Feacture(imagen, 2))

    return vectores_caracteristicos,clases

def acuary(y_prueba,y_correct):
    correctos= np.sum(y_prueba == y_correct)
    return (correctos/len(y_correct))*100      

  

# Definimos el dataset

x,y= cargar_dataset()

X=np.array(x)[:175]
y=np.array(y)[:175]

# Define el número de folds (por ejemplo, k=5)
k = 5

# Crea un objeto KFold para dividir los datos en k grupos
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Variables para almacenar las métricas de rendimiento
accuracies = []

# Realiza k iteraciones
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Crea y entrena un modelo de k-NN
    model = KNeighborsClassifier(n_neighbors=5)  # Configura el número de vecinos (k)
    model.fit(X_train, y_train)
    
    # Evalúa el modelo en el conjunto de prueba
    accuracy = model.score(X_test, y_test)
    
    # Almacena la precisión en la lista de accuracies
    accuracies.append(accuracy)

# Calcula la precisión promedio y otras métricas de rendimiento
average_accuracy = np.mean(accuracies)
print(f'Precisión promedio: {average_accuracy:.2f}')