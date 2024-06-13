import numpy as np
import random

distancia = np.array([
    [0, 2, 4, 3, 6],
    [2, 0, 4, 3, 3],
    [4, 4, 0, 7, 3],
    [3, 3, 7, 0, 3],
    [6, 3, 3, 3, 0]
])

def eval_dis(individual):
    s = 0
    for i in range(len(individual)):
        s += distancia[individual[i], individual[(i + 1) % len(individual)]]
    return s

def eval_mochila(pesos, combinacion):
    return sum(pesos[i] for i in combinacion)

def mochila(limite, pesos, objetos):
    mejor_combinacion = None
    mejor_peso = 0
    for i in range(2, len(objetos) + 1):
        for _ in range(20):
            combinacion = random.sample(objetos, i)
            peso_total = eval_mochila(pesos, combinacion)
            if peso_total <= limite and peso_total > mejor_peso:
                mejor_combinacion = combinacion
                mejor_peso = peso_total
    return mejor_combinacion, mejor_peso

if __name__ == "__main__":
    solucion = [1, 2, 3, 4, 0]
    print(f"Distancia : {eval_dis(solucion)}\n")

    peso_mochila = 15
    objetos = [0, 1, 2, 3, 4]
    pesos = [12, 1, 1, 3, 4]
    
    mejor_combinacion, mejor_peso = mochila(peso_mochila, pesos, objetos)
    if mejor_combinacion:
        print(f"Mejor combinacion: {mejor_combinacion}, Peso total: {mejor_peso}")
    else:
        print("No se encontró una combinación válida")
