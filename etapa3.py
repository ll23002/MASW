"""
Etapa 3 - Definición del Prior (Espacio de Búsqueda)
=====================================================
Define los límites físicos del espacio de búsqueda usando distribuciones uniformes
para 5 capas (4 finitas + 1 semiespacio) y configura el objeto BEL1D.MODELSET
integrando la función de forward de la Etapa 2.
"""

import sys, os
import numpy as np
from scipy import stats

BEL1D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pyBEL1D_src")
if BEL1D_PATH not in sys.path:
    sys.path.insert(0, BEL1D_PATH)

from pyBEL1D import BEL1D

from config import DT, N_SAMPLES, N_TRACES
from etapa2 import cps_forward_wavefield


def construir_prior(n_layer):
    """Construye las distribuciones a priori del espacio de búsqueda.

    Define distribuciones uniformes para los 6 parámetros físicos (H, Vs, Vp, Rho, Qp, Qs)
    de cada capa del modelo 1D. La última capa (semiespacio) no tiene espesor.

    Args:
        n_layer (int): Número total de capas (incluye el semiespacio).

    Returns:
        tuple: (ListPrior, NamesFU, Mins, Maxs)
            - ListPrior (list): Lista de distribuciones scipy.stats.uniform.
            - NamesFU (list): Nombres de cada parámetro con unidades.
            - Mins (list): Valores mínimos de cada parámetro.
            - Maxs (list): Valores máximos de cada parámetro.
    """

    # H_min, H_max, Vs_min, Vs_max, Vp_min, Vp_max, Rho_min, Rho_max, Qp_min, Qp_max, Qs_min, Qs_max
    prior_matrix = np.array([
        [0.0005, 0.003, 0.100, 0.300, 0.300, 0.600, 1.4, 1.6, 20, 100, 5, 30],      # Capa 1 (superficial)
        [0.002,  0.010, 0.150, 0.350, 0.400, 0.800, 1.6, 1.8, 20, 100, 5, 40],      # Capa 2
        [0.005,  0.015, 0.300, 0.550, 0.800, 1.200, 1.8, 2.0, 30, 150, 10, 60],     # Capa 3
        [0.010,  0.020, 0.500, 0.900, 1.200, 1.800, 2.0, 2.2, 50, 200, 20, 100],    # Capa 4
        [0.000,  0.000, 0.800, 1.500, 1.800, 3.000, 2.2, 2.5, 80, 300, 30, 150],    # Semiespacio (halfspace, H=0)
    ])

    NamesFull = ["Thickness", "Vs", "Vp", "Rho", "Qp", "Qs"]
    Units = [" [km]", " [km/s]", " [km/s]", " [g/cc]", "", ""]

    ListPrior = []
    NamesFU = []
    Mins = []
    Maxs = []

    for j in range(6):
        for i in range(n_layer):
            if (i == n_layer - 1) and (j == 0): continue
            cmin = prior_matrix[i, j*2]
            cmax = prior_matrix[i, j*2+1]
            ListPrior.append(stats.uniform(loc=cmin, scale=cmax - cmin))
            Mins.append(cmin)
            Maxs.append(cmax)
            NamesFU.append(f"{NamesFull[j]} {i+1}{Units[j]}")

    return ListPrior, NamesFU, Mins, Maxs


def construir_modelset(n_layer):
    """Configura el objeto BEL1D.MODELSET con el prior y la función de forward.

    Integra la función de forward de la Etapa 2 (cps_forward_wavefield) con el
    ecosistema de pyBEL1D, definiendo el espacio de parámetros y la función de
    restricción condicional.

    Args:
        n_layer (int): Número total de capas (incluye el semiespacio).

    Returns:
        BEL1D.MODELSET: Objeto MODELSET configurado y listo para PREBEL.
    """

    ListPrior, NamesFU, Mins, Maxs = construir_prior(n_layer)

    def cond(model):
        """Valida que los parámetros del modelo se encuentren dentro de los límites especificados.

        Esta función comprueba que todos los parámetros del modelo 1D (espesores de capas,
        velocidades de onda, densidad y factores de calidad) se encuentren dentro de los
        rangos permitidos definidos por las matrices Mins y Maxs. Se utiliza como función
        de restricción condicional durante la inversión probabilística en el framework BEL1D.

        Args:
            model (numpy.ndarray): Array 1D aplanado que contiene los parámetros del modelo
                en el orden [h_1, ..., h_N-1, Vs_1, ..., Vs_N, Vp_1, ..., Vp_N,
                rho_1, ..., rho_N, Qp_1, ..., Qp_N, Qs_1, ..., Qs_N], donde N es el número
                de capas estratificadas.

        Returns:
            bool: Retorna True si TODOS los parámetros del modelo se encuentran dentro de
                los límites [Mins[i], Maxs[i]] para cada parámetro i. Retorna False si
                al menos un parámetro viola sus límites de restricción.
        """
        return (np.logical_and(np.greater_equal(model, Mins), np.less_equal(model, Maxs))).all()

    NamesFull = ["Thickness", "Vs", "Vp", "Rho", "Qp", "Qs"]
    paramNames = {"NamesFU": NamesFU, "NamesSU": NamesFU, "NamesS": NamesFU,
                  "NamesGlobal": NamesFull, "NamesGlobalS": NamesFull,
                  "DataUnits": "Amplitude", "DataName": "Wavefield", "DataAxis": "Time [s]"}

    Timing = np.linspace(0, N_SAMPLES * DT, N_SAMPLES * N_TRACES)

    ModelSet = BEL1D.MODELSET(prior=ListPrior, cond=cond, method="Wavefield",
                              forwardFun={"Fun": cps_forward_wavefield, "Axis": Timing},
                              paramNames=paramNames, nbLayer=n_layer, logTransform=[False, False])

    return ModelSet
