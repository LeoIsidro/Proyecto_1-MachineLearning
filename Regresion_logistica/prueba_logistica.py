from regresion_logistica import Regresion
import numpy as np
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

x=np.array(x)
y=np.array(y)

# Definimos los tamaños de los conjuntos (70% entrenamiento, 15% validación, 15% prueba)
total_samples = len(x)
train = int(0.7 * total_samples)
validation = int(0.15 * total_samples)

# Dividimoss los datos en conjuntos de entrenamiento, validación y prueba
X_train = x[:train]
y_train = y[:train]

X_val = x[train:train + validation]
y_val = y[train:train + validation]

X_test = x[train + validation:]
y_test = y[train + validation:]

# Crear y entrenar el modelo
modelo = Regresion(1000,0.000001)
modelo.train(X_train, y_train)

# Validacion
y_pred_val = modelo.predict(X_val)
precision_val=acuary(y_val,y_pred_val)

print(f"Precisión en el conjunto de validación: {precision_val:.2f}")

# Predicción
y_pred_test = modelo.predict(X_test)
precision_test=acuary(y_test,y_pred_test)

print(f"Precisión en el conjunto de testeo: {precision_test:.2f}")