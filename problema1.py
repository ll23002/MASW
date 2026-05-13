import numpy as np
from numpy.lib.scimath import sqrt 
import matplotlib.pyplot as plt

# ==========================================
# PERFIL ESTRATIGRÁFICO (Modelo Correcto 1 Capa + Semiespacio)
# ==========================================
# Capa 1 (Estrato superficial - Finito)
d1 = 15.0       # Espesor (m)
vp1 = 800.0     # Vel. Onda P (m/s)
vs1 = 400.0     # Vel. Onda S (m/s)
rho1 = 1800.0   # Densidad (kg/m^3)

# Capa 2 (Semiespacio Infinito - Solo necesitamos sus propiedades)
vp2 = 2000.0    # Vel. Onda P (m/s)
vs2 = 1200.0    # Vel. Onda S (m/s)
rho2 = 2500.0   # Densidad (kg/m^3)

def matriz_transferencia_capa(f, v_fase, d, vp, vs, rho):
    """
    Matriz de transferencia A (4x4) de Haskell-Thomson para la capa 1.
    """
    omega = 2 * np.pi * f
    k = omega / v_fase
    ra = sqrt((v_fase / vp)**2 - 1.0)
    rb = sqrt((v_fase / vs)**2 - 1.0)
    gamma = 2.0 * (vs / v_fase)**2
    
    P, Q = k * d * ra, k * d * rb
    cosP, sinP = np.cos(P), np.sin(P)
    cosQ, sinQ = np.cos(Q), np.sin(Q)
    
    A = np.zeros((4, 4), dtype=complex)
    
    A[0, 0] = gamma * cosP + (1 - gamma) * cosQ
    A[0, 1] = 1j * (((1 - gamma) / ra) * sinP + gamma * rb * sinQ)
    A[0, 2] = - (1 / (rho * omega**2)) * (cosP - cosQ)
    A[0, 3] = 1j * (1 / (rho * omega**2)) * ((1 / ra) * sinP + rb * sinQ)
    
    A[1, 0] = -1j * (gamma * ra * sinP + ((1 - gamma) / rb) * sinQ)
    A[1, 1] = (1 - gamma) * cosP + gamma * cosQ
    A[1, 2] = 1j * (1 / (rho * omega**2)) * (ra * sinP + (1 / rb) * sinQ)
    A[1, 3] = A[0, 2]
    
    A[2, 0] = rho * omega**2 * gamma * (1 - gamma) * (cosP - cosQ)
    A[2, 1] = 1j * rho * omega**2 * (((1 - gamma)**2 / ra) * sinP + gamma**2 * rb * sinQ)
    A[2, 2] = A[1, 1] 
    A[2, 3] = A[0, 1] 
    
    A[3, 0] = -1j * rho * omega**2 * (gamma**2 * ra * sinP + ((1 - gamma)**2 / rb) * sinQ)
    A[3, 1] = A[2, 0] 
    A[3, 2] = A[1, 0] 
    A[3, 3] = A[0, 0] 
    
    return A

def condicion_frontera_semiespacio(v_fase, vp, vs, rho):
    """
    Matriz inversa E (2x4) para el semiespacio (Capa 2).
    Aísla únicamente las ondas ascendentes para obligarlas a ser cero.
    """
    ra = sqrt((v_fase / vp)**2 - 1.0)
    rb = sqrt((v_fase / vs)**2 - 1.0)
    gamma = 2.0 * (vs / v_fase)**2
    rho_c2 = rho * v_fase**2
    
    E_inv = np.zeros((2, 4), dtype=complex)
    
    # Condición para anular onda P ascendente
    E_inv[0, 0] = -gamma
    E_inv[0, 1] = 1j * (1.0 - gamma) / ra
    E_inv[0, 2] = 1.0 / rho_c2
    E_inv[0, 3] = -1j / (rho_c2 * ra)
    
    # Condición para anular onda S ascendente
    E_inv[1, 0] = 1j * (1.0 - gamma) / rb
    E_inv[1, 1] = gamma
    E_inv[1, 2] = 1j / (rho_c2 * rb)
    E_inv[1, 3] = 1.0 / rho_c2
    
    return E_inv

# ==========================================
# ESPACIO DE BÚSQUEDA Y EJECUCIÓN
# ==========================================
f_vec = np.linspace(1, 50, 100)
v_vec = np.linspace(300, 1500, 100) 

F, V = np.meshgrid(f_vec, v_vec)
Determinante = np.zeros_like(F, dtype=float)

print("Calculando modelo real de capa sobre semiespacio infinito...")

for i in range(F.shape[0]):
    for j in range(F.shape[1]):
        f_actual = F[i, j]
        v_actual = V[i, j]
        
        # 1. Matriz de la Capa 1
        A1 = matriz_transferencia_capa(f_actual, v_actual, d1, vp1, vs1, rho1)
        
        # 2. Matriz de frontera del Semiespacio
        E_inv = condicion_frontera_semiespacio(v_actual, vp2, vs2, rho2)
        
        # 3. Sistema Global: J (2x4) = E_inv (2x4) @ A1 (4x4)
        J = E_inv @ A1
        
        # 4. Condición de superficie libre: Esfuerzos son cero.
        # Solo sobreviven las columnas de desplazamientos U y W (índices 0 y 1).
        # Obtenemos la submatriz 2x2.
        submatriz = J[:, 0:2]
        
        # 5. Calculamos el determinante
        Determinante[i, j] = np.abs(np.linalg.det(submatriz))

# ==========================================
# SALIDA GRÁFICA EXIGIDA
# ==========================================
plt.figure(figsize=(10, 6))
plt.pcolormesh(F, V, np.log10(Determinante + 1e-10), shading='auto', cmap='inferno')
plt.colorbar(label='Log10(|Determinante|)')
plt.title('Dispersión de Rayleigh: 1 Capa sobre Semiespacio Infinito')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Velocidad de Fase (m/s)')
plt.show()