import numpy as np
import pandas as pd
from  skimage.io import imread, imshow
from pathlib import Path
import pywt
import pywt.data
from PIL import Image
import os

def Get_Feacture(picture, cortes):
  LL = picture
  for i in range(cortes):
     LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
  return LL.flatten().tolist()

carpeta_imagenes = './imagenes_1/'

# Lista todos los archivos en la carpeta y ordénalos
archivos = sorted(os.listdir(carpeta_imagenes))

# Inicializa listas para almacenar datos
clases = []
vectores_caracteristicos = []

# Itera a través de los archivos en la carpeta en el orden deseado
for archivo in archivos:
    # Verifica si el archivo es una imagen (puedes agregar más extensiones si es necesario)
    if archivo.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        # Construye la ruta completa del archivo
        ruta_completa = os.path.join(carpeta_imagenes, archivo)
        imagen = imread(ruta_completa)

        # Realiza operaciones en la imagen aquí, por ejemplo, mostrarla o procesarla

        # Añade la clase según tu lógica (ajusta esto según tus necesidades)
        if int(archivo[1]) < 1:
            clases.append(int(archivo[2]))
        else:
            clases.append(int(archivo[1:3]))

        # Añade los vectores característicos (ajusta esto según tus necesidades)
        vectores_caracteristicos.append(Get_Feacture(imagen, 2))
  

X=np.array(vectores_caracteristicos)
y=np.array(clases)

from knn import KNN
import matplotlib.pyplot as plt

X_train = X[10:150].T
y_train = y[10:150]   # 0:175

"""

for i in range(X_train.shape[1]):
    if y_train[i]==1:
        marcar='v'
        color='red'
    else:
        marcar='o'
        color='blue'
    plt.scatter(x=X_train[1,i],y=X_train[7,i],c=color,s=100,marker=marcar)

plt.show()
"""
# iniciar KNN
clasificador = KNN(k=10)
clasificador.aprendizaje(X_train,y_train) # fase de aprendizaje

# nuevos datos a clasificar con KNN
x1=X[0:5].T
x2=X[160:165].T
X_test = np.concatenate((x1,x2),axis=1)
clasificar = clasificador.clasificacion(X_test)
print('clases de los puntos y(n): ',clasificar)