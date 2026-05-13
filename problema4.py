import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# 1. Cargar la matriz completa de 20 disparos
try:
    datos = np.load("picos_f_k.npy")
except FileNotFoundError:
    print("Error: Ejecuta primero el Problema 3 para generar los datos.")
    exit()

print(f"Datos cargados: {len(datos)} picos detectados.")

# Extraemos las columnas para graficar
disparos = datos[:, 0]  # Eje Z (Tiempo cronológico de los martillazos)
f_idx = datos[:, 1]     # Frecuencia
k_idx = datos[:, 2]     # Número de Onda

plt.figure(figsize=(10, 8))

# Graficamos toda la nube. Usamos 'viridis' para ver cómo avanza el disparo
scatter = plt.scatter(k_idx, f_idx, c=disparos, cmap='viridis', s=100, zorder=5, edgecolors='k')
plt.colorbar(scatter, label='Índice de Disparo (0 a 19)')

# 2. Bucle de Kuhn-Munkres para seguir la trayectoria paso a paso
print("Calculando asignaciones óptimas de trayectoria...")
for i in range(len(datos) - 1):
    picos_t1 = np.atleast_2d(datos[i, 1:])
    picos_t2 = np.atleast_2d(datos[i+1, 1:])
    
    matriz_costos = cdist(picos_t1, picos_t2)
    fila_ind, col_ind = linear_sum_assignment(matriz_costos)
    
    # Dibujar la línea de conexión
    for r, c in zip(fila_ind, col_ind):
        # Truco para que la leyenda no se repita 19 veces
        etiqueta = 'Trayectoria (Kuhn-Munkres)' if i == 0 else ""
        plt.plot([picos_t1[r, 1], picos_t2[c, 1]], 
                 [picos_t1[r, 0], picos_t2[c, 0]], 'r--', alpha=0.7, zorder=1, label=etiqueta)

plt.title('Seguimiento Automático de Curva en 20 Disparos (Grafos)')
plt.xlabel('Eje k (Número de Onda - Índice)')
plt.ylabel('Eje f (Frecuencia - Índice)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print("Grafo completado. Si ves puntos que saltan mucho, es la variabilidad del terreno.")