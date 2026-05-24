"""
Etapa 1 - Preprocesamiento del Campo de Ondas Real
===================================================
Lee archivos crudos de campo en formato SG2, aplica filtro pasabanda (0.1-30 Hz),
remuestrea a la frecuencia objetivo y retorna un vector plano (N_TRACES * N_SAMPLES,).
"""

import numpy as np
from obspy import read
import glob

from config import F_MIN, F_MAX, DT, N_SAMPLES


def procesar_un_campo_real(archivo_sg2, n_traces):
    """Procesa un archivo de datos sísmicos reales en formato SG2.

    Esta función lee un archivo de datos sísmicos reales en formato SG2, aplica filtrado
    pasa-banda, remuestreo y normalización para preparar los datos para su inversión.
    Los datos procesados se extraen hasta el número de trazas especificado (n_traces)
    y se normalizan por su amplitud máxima.

    El flujo de procesamiento incluye:
    1. Lectura del archivo SG2 usando ObsPy
    2. Aplicación de filtro pasa-banda [F_MIN, F_MAX] Hz
    3. Remuestreo a la tasa de muestreo DT especificada
    4. Truncamiento/padding a exactamente N_SAMPLES muestras por traza
    5. Normalización de cada traza por su amplitud máxima absoluta

    Args:
        archivo_sg2 (str): Ruta del archivo SG2 a procesar. Debe ser un archivo
            válido en formato SG2.
        n_traces (int): Número de trazas (geófonos) a procesar.

    Returns:
        numpy.ndarray: Array 1D aplanado de forma (n_traces * N_SAMPLES,) que contiene
            los datos sísmicos procesados y normalizados del componente vertical Z.
            Cada traza está normalizada por su amplitud máxima absoluta.

    Raises:
        FileNotFoundError: Si el archivo SG2 no existe o no es accesible.
        Exception: Cualquier excepción de lectura de datos o procesamiento con ObsPy
            será propagada.
    """

    st = read(archivo_sg2)
    st.filter("bandpass", freqmin=F_MIN, freqmax=F_MAX, corners=4, zerophase=True)
    st.resample(1.0 / DT)

    traces = []
    for tr in st[:n_traces]:
        d = tr.data
        if len(d) > N_SAMPLES: d = d[:N_SAMPLES]
        elif len(d) < N_SAMPLES: d = np.pad(d, (0, N_SAMPLES - len(d)))

        max_val = np.max(np.abs(d))
        if max_val > 0: d = d / max_val
        traces.append(d)

    return np.array(traces).flatten()


def detectar_n_trazas(archivos):
    """Detecta dinámicamente el número de trazas (geófonos) usando el primer archivo.

    Args:
        archivos (list): Lista de rutas a archivos SG2.

    Returns:
        int: Número de trazas detectadas en el primer archivo.
    """
    st_test = read(archivos[0])
    n_traces = len(st_test)
    print(f"[INFO] Se detectaron dinámicamente {n_traces} trazas (geófonos) en los archivos .sg2")
    return n_traces


def cargar_archivos_sg2(directorio="datos_sg2"):
    """Busca y retorna los archivos SG2 disponibles en el directorio indicado.

    Args:
        directorio (str): Directorio donde se encuentran los archivos .sg2.

    Returns:
        list: Lista ordenada de rutas a archivos .sg2.

    Raises:
        FileNotFoundError: Si no se encuentran archivos en el directorio.
    """
    archivos = sorted(glob.glob(f"{directorio}/*.sg2"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos en {directorio}/")
    return archivos
