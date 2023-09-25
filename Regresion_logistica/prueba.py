import numpy as np

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Entrenamiento de la regresión logística
def train(X, y, learning_rate, num_iterations):
    num_samples, num_features = X.shape
    num_classes = len(np.unique(y))
    
    # Inicializamos los parámetros
    theta = np.zeros((num_classes, num_features))
    
    for i in range(num_iterations):
        for c in range(num_classes):
            # Creamos un vector de etiquetas para la clase actual
            y_class = (y == c).astype(int)
            
            # Calculamos la función hipótesis
            z = np.dot(X, theta[c])
            h = sigmoid(z)
            
            # Calculamos el error
            error = h - y_class
            
            # Actualizamos los parámetros
            gradient = np.dot(X.T, error) / num_samples
            theta[c] -= learning_rate * gradient
    
    return theta

# Predicción
def predict(X, theta):
    num_classes = theta.shape[0]
    num_samples = X.shape[0]
    y_pred = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        class_probs = [sigmoid(np.dot(X[i], theta[c])) for c in range(num_classes)]
        y_pred[i] = np.argmax(class_probs)
    
    return y_pred

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo
    X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])
    y = np.array([0, 1, 2, 0, 2])
    
    # Entrenamiento
    learning_rate = 0.1
    num_iterations = 100000
    theta = train(X, y, learning_rate, num_iterations)
    
    # Predicción
    X_test = np.array([[1, 2], [4, 5], [6, 7]])
    y_pred = predict(X_test, theta)
    
    print("Predicciones:", y_pred)
