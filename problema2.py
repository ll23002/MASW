import numpy as np
import pandas as pd

# 1. Semilla obligatoria: últimos 4 dígitos del carnet (ej. LL23002)
np.random.seed(3002)

# Aquí definimos una velocidad de onda S sintética que el algoritmo debe "adivinar"
velocidad_s_objetivo = 450.0 

num_iteraciones = 10000
resultados = []

print(f"Iniciando simulación Monte Carlo con {num_iteraciones} iteraciones...")

for i in range(num_iteraciones):
    # 2. Generación estocástica de parámetros para la Capa 1
    # Se definen rangos lógicos de búsqueda física para no generar basura geológica
    d1_random = np.random.uniform(5.0, 30.0)      # Espesor entre 5 y 30 m
    vp1_random = np.random.uniform(500.0, 1500.0) # Vp entre 500 y 1500 m/s
    
    # Relación física: Vs suele ser menor que Vp (usamos un coeficiente aleatorio)
    vs1_random = vp1_random / np.random.uniform(1.5, 2.2) 
    rho1_random = np.random.uniform(1600.0, 2200.0)

    # 3. Función de costo (Cálculo del Error)
    # En la práctica completa, aquí llamarían a np.linalg.det() de la matriz de Haskell.
    # Para la lógica del Monte Carlo, minimizamos el error absoluto frente al objetivo.
    error_absoluto = np.abs(vs1_random - velocidad_s_objetivo)
    
    # Guardamos la iteración en el registro
    resultados.append({
        'Iteracion': i + 1,
        'Espesor_d1': round(d1_random, 2),
        'Vp1': round(vp1_random, 2),
        'Vs1': round(vs1_random, 2),
        'Densidad': round(rho1_random, 2),
        'Error': round(error_absoluto, 4)
    })

# 4. Procesamiento y Exportación con Pandas
df_resultados = pd.DataFrame(resultados)

# Ordenar de menor a mayor error para que el mejor modelo quede en la primera fila
df_resultados = df_resultados.sort_values(by='Error')

# Exportar los datos como exige la guía
archivo_salida = 'inversion_montecarlo.csv'
df_resultados.to_csv(archivo_salida, index=False)

print(f"Simulación terminada. Datos exportados a '{archivo_salida}'.")

# 5. Mostrar el mejor resultado por terminal
mejor_modelo = df_resultados.iloc[0]
print("\n--- Mejor Modelo Encontrado (Minimización del Error) ---")
print(mejor_modelo)