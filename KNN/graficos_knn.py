import numpy as np
import pandas as pd
from  skimage.io import imread, imshow
from knn import KNN
import pywt
import pywt.data
import os
import matplotlib.pyplot as plt

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

x=np.array(x)
y=np.array(y)


# Mezclamos los datos aleatoriamente
indices = np.arange(len(x))
np.random.shuffle(indices)
X = x[indices]
Y = y[indices]


total_samples = len(X)
train = int(0.7 * total_samples)
validation = int(0.3 * total_samples)


X_train = X[:train]
y_train = Y[:train]

X_test = X[train:train + validation]
y_test = Y[train:train + validation]

# Lista para almacenar las precisión en función de k
accuracies = []

# Prueba diferentes valores de k
k_values = range(1, 50)  # Prueba desde k=1 hasta k=20

for k in k_values:
    # Crea y entrena un modelo de k-NN
    clasificador = KNN(k)
    clasificador.aprendizaje(X_train.T, y_train)
    
    # Evalúa el modelo en el conjunto de prueba y almacena la precisión
    y_pred = clasificador.clasificacion(X_test.T)
    accuracy = acuary(y_pred,y_test)
    accuracies.append(accuracy)

# Crea una gráfica de precisión vs. k
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Precisión')
plt.title('Precisión vs. Número de Vecinos (k) para k-NN')
plt.grid(True)
plt.show()
