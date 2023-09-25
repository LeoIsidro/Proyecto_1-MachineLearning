from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from  skimage.io import imread, imshow
import pywt
import pywt.data
import os

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

X=np.array(x)
y=np.array(y)

# Dividir el conjunto de datos en entrenamiento (70%), validación (15%) y prueba (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Crear un modelo de regresión logística multiclase
model = LogisticRegression(max_iter=1000)

# Entrenar el modelo en el conjunto de entrenamiento
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de validación
y_valid_pred = model.predict(X_valid)

# Calcular la precisión en el conjunto de validación
accuracy_valid = accuracy_score(y_valid, y_valid_pred)
print("Precisión en el conjunto de validación:", accuracy_valid)

# Hacer predicciones en el conjunto de prueba
y_test_pred = model.predict(X_test)

# Calcular la precisión en el conjunto de prueba
accuracy_test = accuracy_score(y_test, y_test_pred)
print("Precisión en el conjunto de prueba:", accuracy_test)
