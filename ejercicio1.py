import pandas as pd
import numpy as np

ruta_datos = 'https://raw.githubusercontent.com/samy1-2-3/Examen-245-2/main/Iris.csv'
datos_iris = pd.read_csv(ruta_datos)

datos_iris['tipo'] = datos_iris['species'].astype('category').cat.codes
X = datos_iris.drop(columns=['species', 'tipo']).values
Y = datos_iris['tipo'].values

X = (X - X.mean(axis=0)) / X.std(axis=0)

Y2= np.zeros((Y.size, Y.max() + 1))
Y2[np.arange(Y.size), Y] = 1

entradas = X.shape[1]
salidas = Y2.shape[1]
p1 = np.random.randn(entradas, 5)
p2 = np.random.randn(5, salidas)
s1 = np.zeros((1, 5))
s2 = np.zeros((1, salidas))

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def derivada_sigmoide(z):
    return z * (1 - z)

def perdida_entropia(prediccion, objetivo):
    return -np.mean(objetivo * np.log(prediccion))

t_apren = 0.4
iteraciones = 10000

for epoch in range(iteraciones + 1):

    c1 = np.dot(X, p1) + s1
    act1 = sigmoide(c1)
    c2 = np.dot(act1, p2) + s2
    act2 = sigmoide(c2)

    error = act2 - Y2
    perdida = perdida_entropia(act2, Y2)

    de2 = error * derivada_sigmoide(act2)
    gradiente_1 = np.dot(de2, p2.T)
    de1 = gradiente_1 * derivada_sigmoide(act1)

    p2 -= t_apren * np.dot(act1.T, de2)
    s2 -= t_apren * np.sum(de2, axis=0, keepdims=True)
    p1 -= t_apren * np.dot(X.T, de1)
    s1 -= t_apren * np.sum(de1, axis=0, keepdims=True)

c1 = np.dot(X, p1) + s1
act1 = sigmoide(c1)
c2 = np.dot(act1, p2) + s2
act2 = sigmoide(c2)
predicciones = np.argmax(act2, axis=1)
precision = np.mean(predicciones == Y)

print(f'Precision: {precision}')
