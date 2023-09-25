import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Supongamos que tienes tus datos de entrenamiento X_train y etiquetas y_train

class MultiClassSVM:
    def __init__(self, C=1, learning_rate=0.0000001, num_iterations=300):
        self.C = C
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.classifiers = {}

        for class_label in self.classes:
            binary_y = np.where(y == class_label, 1, -1)
            classifier = self.train_binary_svm(X, binary_y)
            self.classifiers[class_label] = classifier

    def train_binary_svm(self, X, y):
        num_samples, num_features = X.shape
        weights = np.zeros(num_features)
        bias = 0

        for _ in range(self.num_iterations):
            for i in range(num_samples):
                condition = y[i] * (np.dot(X[i], weights) - bias) >= 1
                if condition:
                    weights -= self.learning_rate * (2 * self.C * weights)
                else:
                    weights -= self.learning_rate * (2 * self.C * weights - np.dot(X[i], y[i]))
                    bias -= self.learning_rate * y[i]

        return (weights, bias)

    def predict(self, X):
        predictions = []

        for i in range(X.shape[0]):
            scores = {}
            for class_label, classifier in self.classifiers.items():
                weights, bias = classifier
                score = np.dot(X[i], weights) - bias
                scores[class_label] = score

            predicted_class = max(scores, key=scores.get)
            predictions.append(predicted_class)

        return np.array(predictions)

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

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Ejemplo de uso
svm_multiclass = MultiClassSVM()
svm_multiclass.fit(X_train, y_train)
y_pred = svm_multiclass.predict(X_test)
# Calcula la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print("Precisión:", accuracy)

# Las predicciones para cada clase están en el arreglo "predictions"
