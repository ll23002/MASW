import os
import numpy as np
import matplotlib.pyplot as plt

from pathos import multiprocessing as mp, pools as pp

from config import N_SAMPLES, N_TRACES
from etapa1 import procesar_un_campo_real
from pyBEL1D import BEL1D


def correr_prebel(model_set, n_models=500):
    """Genera modelos sintéticos previos (prior) usando PREBEL con multiprocesamiento.

    Args:
        model_set (BEL1D.MODELSET): Objeto MODELSET configurado (Etapa 3).
        n_models (int): Número de modelos sintéticos a generar (mínimo 500).

    Returns:
        BEL1D.PREBEL: Objeto Prebel con los modelos y forwards generados.
    """

    pool = pp.ProcessPool(mp.cpu_count())
    Prebel = BEL1D.PREBEL(model_set, nbModels=n_models)
    Prebel.run(Parallelization=[True, pool], verbose=True)
    pool.terminate()

    print(f"\n[INFO] Modelos sintéticos (Prior) generados: {Prebel.MODELS.shape[0]}")
    return Prebel


def calcular_rmse_y_mejor_modelo(Prebel, dataset_real):
    """Calcula el RMSE entre el dataset real y cada sismograma sintético del prior.

    Obtiene la matriz de sismogramas sintéticos de Prebel.FORWARD, calcula el
    Error Cuadrático Medio (RMSE) con respecto al dataset real y determina cuál
    modelo tiene el menor error.

    Args:
        Prebel (BEL1D.PREBEL): Objeto Prebel con modelos y forwards generados.
        dataset_real (numpy.ndarray): Vector plano (N_TRACES * N_SAMPLES,) del
            campo de ondas real procesado (Etapa 1).

    Returns:
        tuple: (mejor_modelo, mejor_forward_matrix)
            - mejor_modelo (numpy.ndarray): Parámetros del modelo con menor RMSE.
            - mejor_forward_matrix (numpy.ndarray): Sismograma sintético del mejor
              modelo con forma (N_TRACES, N_SAMPLES).
    """
    forward_matrix = Prebel.FORWARD  # shape: (n_models, N_TRACES * N_SAMPLES)

    rmse_vals = np.sqrt(np.mean((forward_matrix - dataset_real) ** 2, axis=1))
    idx_mejor = np.argmin(rmse_vals)

    mejor_modelo = Prebel.MODELS[idx_mejor]
    mejor_forward_flat = forward_matrix[idx_mejor]
    mejor_forward_matrix = mejor_forward_flat.reshape(N_TRACES, N_SAMPLES)

    print(f"\n[ETAPA 4] Mejor modelo (menor RMSE = {rmse_vals[idx_mejor]:.6f}):")
    print(mejor_modelo)

    return mejor_modelo, mejor_forward_matrix


def correr_postbel_y_graficas(Prebel, dataset_real, archivo_label, idx, n_layer):
    """Ejecuta POSTBEL sobre un disparo y genera gráficas CCA/Posterior para el primero.

    Args:
        Prebel (BEL1D.PREBEL): Objeto Prebel con modelos y forwards previos.
        dataset_real (numpy.ndarray): Vector plano del campo de ondas real.
        archivo_label (str): Nombre del archivo para mensajes de log.
        idx (int): Índice del disparo actual (0 = primer disparo).
        n_layer (int): Número de capas del modelo.

    Returns:
        tuple: (vs, h)
            - vs (numpy.ndarray): Velocidades Vs del modelo medio posterior (N_LAYER,).
            - h (numpy.ndarray): Espesores del modelo medio posterior (N_LAYER - 1,).
    """
    from pyBEL1D import BEL1D

    print(f"\n[INFO] Procesando disparo: {archivo_label} ({idx+1})...")

    Postbel = BEL1D.POSTBEL(Prebel)
    Postbel.run(Dataset=dataset_real, nbSamples=500, NoiseModel=None)

    mean_model = np.mean(Postbel.SAMPLES, axis=0)
    nLayer_model = (len(mean_model) + 1) // 6
    h = mean_model[0:nLayer_model-1]
    vs = mean_model[nLayer_model-1:2*nLayer_model-1]

    if idx == 0:
        print("Generando gráficas CCA, Posterior y Wiggle para el primer disparo...")
        try:
            Postbel.ShowDataset()
            fig_cca = plt.gcf()
            fig_cca.savefig("etapa4_cca.png", dpi=150)
            plt.close(fig_cca)

            Postbel.ShowPost()
            fig_post = plt.gcf()
            fig_post.savefig("etapa4_posterior.png", dpi=150)
            plt.close(fig_post)
            print("Archivos 'etapa4_cca.png' y 'etapa4_posterior.png' guardados.")
        except Exception as e:
            print(f"[WARNING] Falló la generación de gráficas CCA/Posterior de la Etapa 4: {e}")

    return vs, h


def ejecutar_inversion_todos_disparos(model_set, Prebel, archivos, n_traces, n_layer):
    """Itera sobre todos los archivos SG2 ejecutando POSTBEL y acumulando el perfil 2D.

    Args:
        model_set (BEL1D.MODELSET): Objeto MODELSET configurado.
        Prebel (BEL1D.PREBEL): Objeto Prebel con modelos y forwards previos.
        archivos (list): Lista de rutas a archivos SG2.
        n_traces (int): Número de trazas detectadas.
        n_layer (int): Número de capas del modelo.

    Returns:
        tuple: (perfil_2d, modelos_profundidad)
            - perfil_2d (list): Lista de arrays Vs por disparo.
            - modelos_profundidad (list): Lista de arrays de espesores por disparo.
    """
    perfil_2d = []
    modelos_profundidad = []

    print("\n[ETAPA 4] Iniciando iteración POSTBEL sobre los archivos (Perfil 2D)...")
    for idx, archivo in enumerate(archivos):
        dataset_real = procesar_un_campo_real(archivo, n_traces)

        vs, h = correr_postbel_y_graficas(
            Prebel, dataset_real,
            os.path.basename(archivo),
            idx, n_layer
        )

        perfil_2d.append(vs)
        modelos_profundidad.append(h)

    return perfil_2d, modelos_profundidad
