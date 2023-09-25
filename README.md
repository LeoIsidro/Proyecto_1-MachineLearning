# Proyecto_1-MachineLearning


## Clasificación KNN

El algoritmo de clasificación de k vecinos más cercanos, también conocido como KNN o k-NN, es un clasificador de aprendizaje supervisado no paramétrico, que utiliza la proximidad para hacer clasificaciones o predicciones sobre la agrupación de un punto de datos individual.

<p align="center">
 <img width="702" alt="image" src="https://github.com/LeoIsidro/Proyecto_1-MachineLearning/assets/90939274/a517c354-34fa-440c-8c2f-87ae78b359f2">
</p>

## ¿Como hallaremos la distancia a los K vecinos?

Para determinar qué puntos de datos están más cerca de un punto de consulta determinado, será necesario calcular la distancia entre el punto de consulta y los otros puntos de datos. Estas métricas de distancia ayudan a formar límites de decisión, que dividen los puntos de consulta en diferentes regiones.

En esta implementacion de KNN utilizaremos la distancia euclidiana, la cual esta tiene la siguiente formula.
<p align="center">
 <img width="655" alt="image" src="https://github.com/LeoIsidro/Proyecto_1-MachineLearning/assets/90939274/758eedcf-ffb5-47db-ae11-120664eb6cb2">
</p>

## ¿Como definimos el valor de K?

Definir k puede puede ser complicado de determinar ya que diferentes valores pueden llevar a un ajuste excesivo(overfiting) o insuficiente(underfiting). Los valores más bajos de k pueden tener una varianza alta, pero un sesgo bajo, y los valores más grandes de k pueden generar un sesgo alto y una varianza más baja. La elección de k dependerá en gran medida de los datos de entrada, ya que los datos con más valores atípicos o ruido probablemente funcionarán mejor con valores más altos de k. En general, se recomienda tener un número impar para k para evitar empates en la clasificación, y las tácticas de validación cruzada(K-fold cross validation) pueden ayudar a elegir el valor de k óptimo para el conjunto de datos.

## Ventajas
- Fácil de implementar: Dada la simplicidad y precisión del algoritmo.
- Se adapta fácilmente: A medida que se agregan nuevas muestras de entrenamiento, el algoritmo se ajusta para tener en cuenta cualquier dato nuevo, ya que todos los datos de entrenamiento se almacenan en la memoria.
- Pocos hiperparámetros: KNN solo requiere un valor k y una métrica de distancia, que es baja en comparación con otros algoritmos de machine learning.

## Desventajas
- La maldición de la dimensionalidad: El algoritmo KNN tiende a ser víctima de la maldición de la dimensionalidad, lo que significa que no funciona bien con entradas de datos de alta dimensión.
- No escala bien: Dado que KNN es un algoritmo perezoso, ocupa más memoria y almacenamiento de datos en comparación con otros clasificadores. Esto puede ser costoso desde una perspectiva de tiempo y dinero. Más memoria y almacenamiento aumentarán los gastos comerciales y más datos pueden tardar más en procesarse.

## Experimentacion
En la parte de experimentacion para el modelo de KNN, hemos graficado la precision del metodo para un rango de K.
<p align="center">
 <img width="1002" alt="image" src="https://github.com/LeoIsidro/Proyecto_1-MachineLearning/assets/90939274/9bd13e37-41ff-423d-94ec-fa68e5f7edf0">
</p>
Como podemos observar en la imagen la mejor precision del modelo ocurre con un k=20, por otro lado la peor precision ocurre cuando se tiene un k=6 o k=7, suponemos que esto ocurre debido a que es un valor muy pequeño de k lo que ocaciona un overfiting.
