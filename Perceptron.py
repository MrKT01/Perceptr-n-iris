# Importaciones de iberia # Experimento de redes (Perceptrón)
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import  load_iris 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

# cargando el dataset de iris
iris = load_iris()
X =  iris.data[:, :2] # Usamos el Largo y el ancho del Sépalo 
y = (iris.target== 0).astype(int) # Clase Setosa vs resto

# Normalizamos los datos 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividimos el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Creamos el Perceptrón
clf = Perceptron(eta0=0.1, random_state=42)
clases = np.array([0, 1])
xs = np.linspace(-3, 3, 100)
plt.ion()
fig, ax = plt.subplots()
for epoca in range(1, 31):
    clf.partial_fit(X_train, y_train, classes=clases)
    w0, w1 = clf.coef_[0]
    b = clf.intercept_[0]
    ax.clear()
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm")
    if w1 != 0:
       ax.plot(xs, -(w0 * xs + b) / w1, "k")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(f"Epoca {epoca} | precision: {clf.score(X_test, y_test):.2f}")
plt.ioff()
plt.show()    

# Definimos  el nuevo punto sepal length 0,5 cm and sepal width 1.4 cm
new_data_point = np.array([[5.0, 3.5]])

# Escalamos para estar igual que el set de entrenamiento
new_data_point_scaled = scaler.transform(new_data_point)

# Predice  la clase con la red neuronal 
predicted_class = clf.predict(new_data_point_scaled)

predicted_label = 'Setosa' if predicted_class[0] == 1 else 'Not Setosa'
print(f"Para el dato sepal length 5.0 and sepal width 3.5 (normalized: {new_data_point_scaled[0]}")
print(f"La clase es: {predicted_class[0]} ({predicted_label})")