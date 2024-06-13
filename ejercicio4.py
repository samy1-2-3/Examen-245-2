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

def mochila_simulado(limite, pesos, objetos, temp_inicial=100, temp_final=1, alpha=0.9, iteraciones=100):

    mejor_combinacion = random.sample(objetos, len(objetos))
    mejor_peso = eval_mochila(pesos, mejor_combinacion)
    temp = temp_inicial

    def obtener_vecino(combinacion):
        vecino = combinacion[:]
        i, j = random.sample(range(len(combinacion)), 2)
        vecino[i], vecino[j] = vecino[j], vecino[i]
        return vecino
    
    while temp > temp_final:
        for _ in range(iteraciones):
            nueva_combinacion = obtener_vecino(mejor_combinacion)
            nuevo_peso = eval_mochila(pesos, nueva_combinacion)

            if nuevo_peso <= limite and (nuevo_peso > mejor_peso or random.uniform(0, 1) < np.exp((mejor_peso - nuevo_peso) / temp)):
                mejor_combinacion = nueva_combinacion
                mejor_peso = nuevo_peso

        temp *= alpha
    
    return mejor_combinacion, mejor_peso

if __name__ == "__main__":
    solucion = [1, 2, 3, 4, 0]
    print(f"Distancia total de la solucion: {eval_dis(solucion)}\n")

    peso_mochila = 15
    objetos = [0, 1, 2, 3, 4]
    pesos = [12, 1, 1, 3, 4]
    
    mejor_combinacion, mejor_peso = mochila_simulado(peso_mochila, pesos, objetos)
    if mejor_combinacion:
        print(f"Mejor combinacion (simulado): {mejor_combinacion}, Peso total: {mejor_peso}")
    else:
        print("No se encontro una combinacion aceptable")
