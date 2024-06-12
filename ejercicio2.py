import pandas as pd
import numpy as np

ruta_datos = 'https://raw.githubusercontent.com/samy1-2-3/Examen-245-2/main/Iris.csv'
datos_iris = pd.read_csv(ruta_datos)

datos_iris['tipo'] = datos_iris['species'].astype('category').cat.codes
X = datos_iris.drop(columns=['species', 'tipo']).values
Y = datos_iris['tipo'].values

X = (X - X.mean(axis=0)) / X.std(axis=0)

Y2 = np.zeros((Y.size, Y.max() + 1))
Y2[np.arange(Y.size), Y] = 1

entradas = X.shape[1]
oc1 = 5
oc2 = 5
salidas = Y2.shape[1]
p1 = np.random.randn(entradas, oc1)
p2 = np.random.randn(oc1, oc2)
p3 = np.random.randn(oc2, salidas)
s1 = np.zeros((1, oc1))
s2 = np.zeros((1, oc2))
s3 = np.zeros((1, salidas))

def escalon(z):
    return np.where(z >= 0, 1, 0)

t_apren = 0.2
iteraciones = 10000

for epoch in range(iteraciones + 1):
    c1 = np.dot(X, p1) + s1
    act1 = escalon(c1)
    c2 = np.dot(act1, p2) + s2
    act2 = escalon(c2)
    c3 = np.dot(act2, p3) + s3
    act3 = escalon(c3)

    error = act3 - Y2

    de3 = error
    gra2 = np.dot(de3, p3.T)
    de2 = gra2
    gra1 = np.dot(de2, p2.T)
    de1 = gra1

    p3 -= t_apren * np.dot(act2.T, de3)
    s3 -= t_apren * np.sum(de3, axis=0, keepdims=True)
    p2 -= t_apren * np.dot(act1.T, de2)
    s2 -= t_apren * np.sum(de2, axis=0, keepdims=True)
    p1 -= t_apren * np.dot(X.T, de1)
    s1 -= t_apren * np.sum(de1, axis=0, keepdims=True)

c1 = np.dot(X, p1) + s1
act1 = escalon(c1)
c2 = np.dot(act1, p2) + s2
act2 = escalon(c2)
capa_3 = np.dot(act2, p3) + s3
act3 = escalon(capa_3)
predicciones = np.argmax(act3, axis=1)
precision = np.mean(predicciones == Y)

print(f'Precision: {precision}')
