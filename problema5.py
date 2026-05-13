import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# ==========================================
# FUNCIÓN AUXILIAR DE TENSORES
# ==========================================

def preparar_tensores(x, y, z):
    """
    Convierte coordenadas cartesianas (x, y, z) a cilíndricas (r, theta, z).
    Retorna el tensor 3D (N, 1, 3) y la matriz aplanada (N, 3).
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    cilindricas = np.column_stack((r, np.degrees(theta), z))
    tensor_3d = cilindricas.reshape(cilindricas.shape[0], 1, 3)
    return tensor_3d, cilindricas


# ==========================================
# FUNCIÓN PRINCIPAL EXPORTABLE
# ==========================================

def ejecutar_clustering(datos_crudos, f_max=150.0, dx=2.0,
                        filas_150hz=None, columnas_e=None,
                        eps=15, min_samples=3):
    """
    Aplica DBSCAN sobre los picos F-K, extrae el Clúster 0 (el válido)
    y calcula la curva de dispersión observada v_c = f / k.

    Parámetros
    ----------
    datos_crudos : np.ndarray shape (N, 3)
        Array [indice_archivo, fila_f, col_k] — salida de problema3.
    f_max : float
        Frecuencia máxima de análisis (Hz). Default 150.
    dx : float
        Espaciado entre geófonos (m). Default 2.0.
    filas_150hz : int, opcional
        Número de filas FFT correspondientes a f_max.
        Si se omite, se estima de datos_crudos.
    columnas_e : int, opcional
        Número de canales (geófonos).
        Si se omite, se estima de datos_crudos.
    eps : float
        Radio de vecindad para DBSCAN (en espacio cilíndrico normalizado).
    min_samples : int
        Mínimo de muestras por clúster en DBSCAN.

    Retorna
    -------
    dict con:
        'frecuencias_hz'    : array f (Hz) del Clúster 0
        'k_values'          : array k (1/m) del Clúster 0
        'velocidades_fase'  : array v_c = f/k (m/s) de cada punto del Clúster 0
        'curva_dispersion'  : dict {'f': array, 'vc': array}  — un punto por frecuencia
        'etiquetas'         : todas las etiquetas DBSCAN
        'datos_ml'          : coordenadas cilíndricas (N, 3)
        'frecuencias_raw'   : frecuencias (Hz) de todos los picos
        'k_raw'             : k (1/m) de todos los picos
    """
    # ── Mapear índices → unidades físicas ────────────────────────────────────
    if filas_150hz is None:
        # Estimación conservadora: máximo valor del índice de fila + 1
        filas_150hz = int(datos_crudos[:, 1].max()) + 1
    if columnas_e is None:
        columnas_e = int(datos_crudos[:, 2].max()) + 1

    # Frecuencias reales (Hz)
    frecuencias_raw = datos_crudos[:, 1] * (f_max / filas_150hz)

    # Números de onda reales (1/m) — solo la mitad positiva (k > 0)
    # El índice col_k apunta dentro de la mitad positiva del espectro shiftado
    k_nyquist = 1.0 / (2.0 * dx)
    n_k_pos = columnas_e - columnas_e // 2
    k_axis_pos = np.linspace(0, k_nyquist, n_k_pos)
    k_raw = k_axis_pos[datos_crudos[:, 2].astype(int)]

    # ── Coordenadas para clustering ───────────────────────────────────────────
    # Usamos (col_k, fila_f, indice) en coordenadas cilíndricas
    X = datos_crudos[:, 2]   # col_k  (índice entero)
    Y = datos_crudos[:, 1]   # fila_f (índice entero)
    Z = datos_crudos[:, 0]   # índice de archivo

    tensor_3d, datos_ml = preparar_tensores(X, Y, Z)

    # ── DBSCAN ───────────────────────────────────────────────────────────────
    print(f"[P5] Ejecutando DBSCAN (eps={eps}, min_samples={min_samples})...")
    modelo = DBSCAN(eps=eps, min_samples=min_samples)
    etiquetas = modelo.fit_predict(datos_ml)

    clusters_unicos = np.unique(etiquetas)
    print(f"[P5] Clústeres encontrados: {clusters_unicos}")

    # ── Seleccionar el clúster físicamente válido ─────────────────────────────
    # Criterio: el clúster cuya mediana de Vc = f/k esté en rango sísmico
    VC_MIN, VC_MAX = 50.0, 1500.0
    candidatos = [c for c in clusters_unicos if c != -1]

    if not candidatos:
        raise RuntimeError(
            "[P5] DBSCAN clasificó todo como ruido. Ajusta eps o min_samples."
        )

    cluster_elegido = None
    for c in sorted(candidatos):
        mask = (etiquetas == c)
        f_c = frecuencias_raw[mask]
        k_c = k_raw[mask]
        k_pos = k_c[k_c > 1e-6]
        f_pos = f_c[k_c > 1e-6]
        if len(k_pos) == 0:
            continue
        vc_mediana = np.median(f_pos / k_pos)
        print(f"[P5]   Clúster {c}: {mask.sum()} pts, "
              f"Vc mediana = {vc_mediana:.0f} m/s")
        if VC_MIN <= vc_mediana <= VC_MAX and cluster_elegido is None:
            cluster_elegido = c

    if cluster_elegido is None:
        # Fallback: clúster con más puntos k>0
        conteos = {}
        for c in candidatos:
            mask = (etiquetas == c)
            conteos[c] = np.sum(k_raw[mask] > 1e-6)
        cluster_elegido = max(conteos, key=conteos.get)
        print(f"[P5] Fallback: usando clúster {cluster_elegido}.")

    print(f"[P5] → Clúster válido seleccionado: {cluster_elegido}")
    mascara_cluster = (etiquetas == cluster_elegido)

    frecuencias_cluster = frecuencias_raw[mascara_cluster]
    k_cluster = k_raw[mascara_cluster]

    # Filtrar k ≤ 0
    validos_k = k_cluster > 1e-6
    frecuencias_cluster = frecuencias_cluster[validos_k]
    k_cluster = k_cluster[validos_k]

    if len(k_cluster) == 0:
        raise RuntimeError(
            "[P5] El clúster elegido no tiene puntos con k > 0. "
            "Revisa la extracción F-K del Problema 3."
        )

    velocidades_fase = frecuencias_cluster / k_cluster

    print(f"[P5] Clúster 0: {len(frecuencias_cluster)} puntos válidos (k>0).")
    print(f"     Rango f: {frecuencias_cluster.min():.1f} – "
          f"{frecuencias_cluster.max():.1f} Hz")
    print(f"     Rango Vc: {velocidades_fase.min():.0f} – "
          f"{velocidades_fase.max():.0f} m/s")

    # ── Curva de dispersión: promediar k por bin de frecuencia ────────────────
    # Redondear frecuencias a 1 Hz de resolución para agrupar
    bins_f = np.round(frecuencias_cluster).astype(int)
    frecuencias_unicas = np.unique(bins_f)

    vc_promedio = []
    f_promedio = []
    for f_bin in frecuencias_unicas:
        mascara_f = (bins_f == f_bin)
        k_medio = np.mean(k_cluster[mascara_f])
        if k_medio > 0:
            vc_promedio.append(f_bin / k_medio)
            f_promedio.append(f_bin)

    f_promedio = np.array(f_promedio, dtype=float)
    vc_promedio = np.array(vc_promedio, dtype=float)

    # Ordenar por frecuencia ascendente
    orden = np.argsort(f_promedio)
    f_promedio = f_promedio[orden]
    vc_promedio = vc_promedio[orden]

    print(f"[P5] Curva de dispersión: {len(f_promedio)} puntos "
          f"(f={f_promedio[0]:.0f}–{f_promedio[-1]:.0f} Hz, "
          f"Vc={vc_promedio.min():.0f}–{vc_promedio.max():.0f} m/s).")

    return {
        "frecuencias_hz": frecuencias_cluster,
        "k_values": k_cluster,
        "velocidades_fase": velocidades_fase,
        "curva_dispersion": {"f": f_promedio, "vc": vc_promedio},
        "etiquetas": etiquetas,
        "datos_ml": datos_ml,
        "frecuencias_raw": frecuencias_raw,
        "k_raw": k_raw,
    }


# ==========================================
# EJECUCIÓN STANDALONE (python problema5.py)
# ==========================================
if __name__ == "__main__":
    print("[P5] Cargando datos del Problema 3...")
    try:
        datos_crudos = np.load("picos_f_k.npy")
    except FileNotFoundError:
        print("[P5] Error: No se encontró 'picos_f_k.npy'. Ejecuta problema3.py primero.")
        raise SystemExit(1)

    resultado = ejecutar_clustering(datos_crudos)

    etiquetas = resultado["etiquetas"]
    datos_ml = resultado["datos_ml"]
    f_curva = resultado["curva_dispersion"]["f"]
    vc_curva = resultado["curva_dispersion"]["vc"]

    # ── Figura 1: Clustering 3D ───────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 5))

    ax3d = fig.add_subplot(121, projection="3d")
    scatter = ax3d.scatter(
        datos_ml[:, 0], datos_ml[:, 1], datos_ml[:, 2],
        c=etiquetas, cmap="plasma", marker="o", s=80, alpha=0.85, edgecolors="k",
    )
    ax3d.set_title("DBSCAN — Coordenadas Cilíndricas")
    ax3d.set_xlabel("Radio (r)")
    ax3d.set_ylabel("Ángulo (θ°)")
    ax3d.set_zlabel("Índice Disparo")
    try:
        cbar = plt.colorbar(scatter, ax=ax3d, pad=0.1, shrink=0.6)
        cbar.set_label("Clúster ID")
    except Exception:
        pass

    # ── Figura 2: Curva de dispersión observada ───────────────────────────────
    ax2d = fig.add_subplot(122)
    ax2d.scatter(
        resultado["frecuencias_hz"], resultado["velocidades_fase"],
        c="steelblue", s=30, alpha=0.5, label="Puntos Clúster 0",
    )
    ax2d.plot(f_curva, vc_curva, "r-o", markersize=4,
              linewidth=2, label="Curva promedio (v_c = f/k)")
    ax2d.set_xlabel("Frecuencia (Hz)")
    ax2d.set_ylabel("Velocidad de Fase (m/s)")
    ax2d.set_title("Curva de Dispersión Observada — Comalapa")
    ax2d.legend()
    ax2d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("[P5] Proceso de Machine Learning finalizado exitosamente.")