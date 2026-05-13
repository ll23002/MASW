import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from scipy.fft import fft2, fftshift
import glob
import os

# ==========================================
# FUNCIÓN PRINCIPAL EXPORTABLE
# ==========================================

def procesar_datos_fk(ruta_archivos="datos_sg2/*.sg2", dx=2.0, f_max=150.0):
    """
    Procesa archivos SEG-2, aplica FFT 2D y extrae los picos de energía
    en el espacio Frecuencia-Número de Onda (F-K).

    Parámetros
    ----------
    ruta_archivos : str
        Glob pattern para los archivos .sg2
    dx : float
        Espaciado entre geófonos en metros.
    f_max : float
        Frecuencia máxima de análisis en Hz.

    Retorna
    -------
    dict con:
        'picos_indices' : np.ndarray shape (N, 3)  [indice_archivo, fila_f, col_k]
        'frecuencias_hz': np.ndarray shape (N,)    frecuencias reales de cada pico (Hz)
        'k_values'      : np.ndarray shape (N,)    números de onda reales de cada pico (1/m)
        'fs'            : float                    tasa de muestreo (Hz)
        'dx'            : float                    espaciado entre geófonos (m)
        'filas_t'       : int                      muestras de tiempo del primer archivo
        'columnas_e'    : int                      número de canales (geófonos)
        'filas_150hz'   : int                      filas FFT correspondientes a f_max
        'espectro_control': np.ndarray             espectro del primer disparo (para gráfica)
        'fila_control'  : int                      fila del pico en espectro_control
        'col_control'   : int                      columna del pico en espectro_control
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

            # FFT 2D + shift
            espectro_fk = fftshift(fft2(matriz_sismica))
            energia_fk = np.abs(espectro_fk) ** 2

            # Tras fftshift: DC está en (filas_t//2, columnas_e//2)
            centro_f = filas_t_actual // 2
            centro_k = columnas_e_actual // 2

            # Solo frecuencias positivas (0 → f_max) y k positivas (forward)
            energia_util = energia_fk[
                centro_f: centro_f + filas_max,
                centro_k:                        # mitad derecha = k > 0
            ]

            idx_max = np.unravel_index(np.argmax(energia_util), energia_util.shape)

            # Siempre inicializar parámetros físicos en el primer archivo leído
            if filas_t is None:
                filas_t = filas_t_actual
                columnas_e = columnas_e_actual
                filas_150hz = filas_max
                fs = fs_actual

            # Saltar picos en col_k=0 (artefacto DC de la FFT)
            if idx_max[1] == 0:
                continue

            # col_k guardado como índice dentro de la mitad positiva
            resultados_maximos.append((i, idx_max[0], idx_max[1]))

            # Actualizar espectro de control al primer pico válido (no DC)
            if len(resultados_maximos) == 1:
                espectro_control = energia_util
                fila_control = idx_max[0]
                col_control = idx_max[1]

        except Exception as e:
            print(f"[P3] Saltando {os.path.basename(archivo)}: {e}")

    if not resultados_maximos:
        raise RuntimeError("No se procesó ningún archivo SEG-2 exitosamente.")

    picos_indices = np.array(resultados_maximos)  # (N, 3)

    # ── Convertir índices matriciales a unidades físicas ──────────────────────
    # Eje de frecuencias: 0 → f_max Hz, mapeado en filas_150hz bins
    frecuencias_hz = picos_indices[:, 1] * (f_max / filas_150hz)

    # Eje de números de onda: solo la mitad positiva (k > 0)
    # El número de columnas en la mitad positiva es columnas_e - columnas_e//2
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


# ==========================================
# EJECUCIÓN STANDALONE (python problema3.py)
# ==========================================
if __name__ == "__main__":
    resultado = procesar_datos_fk()

    # Guardar para uso externo
    np.save("picos_f_k.npy", resultado["picos_indices"])
    print(f"[P3] Nube de puntos guardada en 'picos_f_k.npy' "
          f"({len(resultado['picos_indices'])} picos).")

    # ── Gráfica de control (primer disparo) ───────────────────────────────────
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
    # Marcar pico detectado
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