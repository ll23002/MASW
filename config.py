"""
config.py - Constantes globales del pipeline MASW Wavefield Inversion
======================================================================
Centraliza todos los parámetros de configuración del sistema para que
sean importados por cada etapa sin duplicación.
"""

import sys, os, warnings

warnings.filterwarnings("ignore")

# --- Rutas ---
CPS_BIN = "/home/alexander/CPS/PROGRAMS.330/bin/"
BEL1D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pyBEL1D_src")
if BEL1D_PATH not in sys.path:
    sys.path.insert(0, BEL1D_PATH)

# --- Geometría de adquisición ---
DX = 2.0          # Separación entre geófonos [m]
N_TRACES = 24     # Número de trazas (actualizado dinámicamente en main.py)

# --- Parámetros de señal ---
F_MIN = 0.1       # Frecuencia mínima del filtro pasabanda [Hz]
F_MAX = 30.0      # Frecuencia máxima del filtro pasabanda [Hz]
DT = 0.002        # Intervalo de muestreo [s] → 500 Hz
N_SAMPLES = 8192  # Muestras por traza (8192 * 0.002s = 16.38s)

# --- Modelo ---
N_LAYER = 5       # Número de capas (4 finitas + 1 semiespacio)
