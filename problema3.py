import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from scipy.fft import fft2, fftshift
import glob
import os


def procesar_datos_fk(ruta_archivos="datos_sg2/*.sg2", dx=2.0, f_max=150.0):
    """
    Procesa archivos SEG-2, aplica FFT 2D y extrae los picos de energía
    en el espacio Frecuencia–Número de Onda (F–K).

    Parameters
    ----------
    ruta_archivos : str, optional
        Patrón glob para localizar los archivos .sg2 (por defecto "datos_sg2/*.sg2").
    dx : float, optional
        Espaciado entre geófonos en metros (por defecto 2.0).
    f_max : float, optional
        Frecuencia máxima de análisis en Hz (por defecto 150.0).

    Returns
    -------
    dict
        Diccionario con las siguientes claves:
        - 'picos_indices' : numpy.ndarray, shape (N, 3)
            Índices de los picos detectados por archivo: (indice_archivo, fila_f, col_k)
            donde fila_f es el índice de frecuencia (fila dentro de la submatriz analizada)
            y col_k es el índice de número de onda en la mitad positiva.
        - 'frecuencias_hz' : numpy.ndarray, shape (N,)
            Frecuencias reales de cada pico en Hz (mapeadas desde 0 → f_max).
        - 'k_values' : numpy.ndarray, shape (N,)
            Números de onda reales de cada pico en 1/m (valores de la mitad positiva del eje k).
        - 'fs' : float
            Tasa de muestreo (Hz) tomada del primer archivo procesado exitosamente.
        - 'dx' : float
            Espaciado entre geófonos (m), igual al parámetro de entrada.
        - 'filas_t' : int
            Número de muestras en tiempo (filas) del primer archivo procesado.
        - 'columnas_e' : int
            Número de canales (geófonos) del primer archivo procesado.
        - 'filas_150hz' : int
            Número de filas de la FFT que corresponden al rango de frecuencia 0 → f_max.
        - 'f_max' : float
            Frecuencia máxima usada para el mapeo (Hz).
        - 'espectro_control' : numpy.ndarray
            Espectro (energía) usado para la gráfica de control. Es la submatriz
            de energía (|FFT|^2) correspondiente al primer pico válido. Shape ≈
            (filas_150hz, n_k_pos).
        - 'fila_control' : int or None
            Índice de fila del pico detectado dentro de 'espectro_control' (o None si no hay).
        - 'col_control' : int or None
            Índice de columna del pico detectado dentro de 'espectro_control' (o None si no hay).
        - 'archivos' : list of str
            Lista de rutas de archivos .sg2 procesadas (ordenadas).

    Notes
    -----
    - El código considera sólo la mitad positiva del eje de números de onda (k > 0)
      y las frecuencias desde 0 hasta f_max.
    - La conversión de índices a Hz y a 1/m se realiza mapeando la submatriz
      utilizada (de tamaño 'filas_150hz' en frecuencia y n_k_pos en k) al rango físico.
    """

    archivos = sorted(glob.glob(ruta_archivos))

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos .sg2 en: {ruta_archivos}"
        )

    resultados_maximos = []
    espectro_control = None
    fila_control = col_control = None
    filas_t = columnas_e = filas_150hz = fs = None

    print(f"[P3] {len(archivos)} registros SEG-2 encontrados. Procesando...")

    for i, archivo in enumerate(archivos):
        try:
            st = read(archivo, format="SEG2")
            # Matriz (Tiempo x Canales)
            matriz_sismica = np.array([tr.data for tr in st]).T
            fs_actual = st[0].stats.sampling_rate

            filas_t_actual, columnas_e_actual = matriz_sismica.shape

            filas_max = int(filas_t_actual * (f_max / fs_actual))

            #Calcula la transformada de Fourier 2D
            espectro_fk = fftshift(fft2(matriz_sismica))
            energia_fk = np.abs(espectro_fk) ** 2

            #DC está en el centro de la matriz
            centro_f = filas_t_actual // 2
            centro_k = columnas_e_actual // 2

            energia_util = energia_fk[centro_f: centro_f + filas_max, centro_k:]

            idx_max = np.unravel_index(np.argmax(energia_util), energia_util.shape)

            if filas_t is None:
                filas_t = filas_t_actual
                columnas_e = columnas_e_actual
                filas_150hz = filas_max
                fs = fs_actual

            if idx_max[1] == 0:
                continue

            resultados_maximos.append((i, idx_max[0], idx_max[1]))

            if len(resultados_maximos) == 1:
                espectro_control = energia_util
                fila_control = idx_max[0]
                col_control = idx_max[1]

        except Exception as e:
            print(f"[P3] Saltando {os.path.basename(archivo)}: {e}")

    if not resultados_maximos:
        raise RuntimeError("No se procesó ningún archivo SEG-2 exitosamente.")

    picos_indices = np.array(resultados_maximos)

    frecuencias_hz = picos_indices[:, 1] * (f_max / filas_150hz)


    k_nyquist = 1.0 / (2.0 * dx)
    n_k_pos = columnas_e - columnas_e // 2
    k_axis_pos = np.linspace(0, k_nyquist, n_k_pos)
    k_values = k_axis_pos[picos_indices[:, 2].astype(int)]

    print(f"[P3] Listo. {len(picos_indices)} picos extraídos.")
    print(f"     Rango f: {frecuencias_hz.min():.1f} – {frecuencias_hz.max():.1f} Hz")
    print(f"     Rango k: {k_values.min():.4f} – {k_values.max():.4f} 1/m")

    return {
        "picos_indices": picos_indices,
        "frecuencias_hz": frecuencias_hz,
        "k_values": k_values,
        "fs": fs,
        "dx": dx,
        "filas_t": filas_t,
        "columnas_e": columnas_e,
        "filas_150hz": filas_150hz,
        "f_max": f_max,
        "espectro_control": espectro_control,
        "fila_control": fila_control,
        "col_control": col_control,
        "archivos": archivos,
    }


if __name__ == "__main__":
    resultado = procesar_datos_fk()

    np.save("picos_f_k.npy", resultado["picos_indices"])
    print("Nube de puntos guardada")

    dx = resultado["dx"]
    f_max = resultado["f_max"]
    filas_150hz = resultado["filas_150hz"]
    columnas_e = resultado["columnas_e"]
    espectro_control = resultado["espectro_control"]
    fila_c = resultado["fila_control"]
    col_c = resultado["col_control"]
    archivos = resultado["archivos"]

    k_limit = 1.0 / (2.0 * dx)

    plt.figure(figsize=(10, 8))
    plt.imshow(
        np.log10(espectro_control + 1e-10),
        extent=[0, k_limit, 0, f_max],
        aspect="auto",
        cmap="jet",
        origin="lower",
    )

    n_k_pos = columnas_e - columnas_e // 2
    k_axis = np.linspace(0, k_limit, n_k_pos)
    f_axis = np.linspace(0, f_max, filas_150hz)
    plt.plot(
        k_axis[col_c], f_axis[fila_c],
        "rx", markersize=15, markeredgewidth=3, label="Pico Detectado",
    )
    plt.colorbar(label="Log10(Energía)")
    plt.title(
        f"Análisis F-K — {os.path.basename(archivos[0])}"
    )
    plt.xlabel("Número de Onda k (1/m)")
    plt.ylabel("Frecuencia f (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.show()