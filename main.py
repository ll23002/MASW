import sys, os, warnings, tempfile, subprocess
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
import glob
from scipy import stats

warnings.filterwarnings("ignore")

CPS_BIN = "/home/alexander/CPS/PROGRAMS.330/bin/"
BEL1D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pyBEL1D_src")
if BEL1D_PATH not in sys.path:
    sys.path.insert(0, BEL1D_PATH)

from pyBEL1D import BEL1D
from pathos import multiprocessing as mp, pools as pp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DX = 2.0
F_MIN = 0.1
F_MAX = 30.0
DT = 0.002
N_SAMPLES = 8192  # 8192 muestras * 0.002s = 16.38s, 1/16.38 = 0.061s
N_TRACES = 24

def cps_forward_wavefield(model):
    """Simula el campo de ondas sísmico para un modelo 1D estratificado usando CPS.

    Esta función genera sismogramas sintéticos ejecutando los programas de modelado
    directo de CPS (Computer Programs in Seismology). Toma un modelo 1D parametrizado
    y calcula el movimiento del terreno vertical (componente Z) en ubicaciones de
    receptores especificadas utilizando métodos espectrales.

    El parámetro model se analiza internamente para extraer propiedades específicas
    de cada capa: espesor (h), velocidad de ondas S (Vs), velocidad de ondas P (Vp),
    densidad (rho), factor de calidad de ondas P (Qp) y factor de calidad de ondas S (Qs).

    El flujo de trabajo incluye:
    1. Creación del archivo de modelo de velocidad en formato CPS
    2. Ejecución de cálculos de desplazamiento (sdisp96)
    3. Cálculo de sismogramas sintéticos (sregn96)
    4. Generación de sismogramas de componente vertical (spulse96)
    5. Conversión de salida a formato SAC (f96tosac)
    6. Filtrado y normalización de las trazas

    Args:
        model (numpy.ndarray): Array aplanado de forma (6*N_CAPA - 1,) que contiene
            los parámetros del modelo en el orden:
            [h_1, ..., h_N-1, Vs_1, ..., Vs_N, Vp_1, ..., Vp_N,
             rho_1, ..., rho_N, Qp_1, ..., Qp_N, Qs_1, ..., Qs_N]
            donde N es el número de capas (se infiere automáticamente de la longitud).
            Unidades: h [km], Vs [km/s], Vp [km/s], rho [g/cc].
            La última capa (semiespacio) tiene espesor infinito (h=0.0).

    Returns:
        numpy.ndarray: Array 1D aplanado de forma (N_TRACES * N_SAMPLES,) que contiene
            el movimiento del terreno vertical normalizado (componente Z) para todas
            las trazas. Cada traza está filtrada pasa-banda y normalizada en amplitud.
            Retorna un array de ruido aleatorio si la generación de archivos SAC falla.

    Raises:
        No se lanzan excepciones explícitas. Las fallas en las llamadas a subprocesos
        o en la generación de archivos resultan en una salida de ruido aleatorio como
        alternativa.
    """
    
    nLayer = (len(model) + 1) // 6
    h = model[0:nLayer-1]
    vs = model[nLayer-1:2*nLayer-1]
    vp = model[2*nLayer-1:3*nLayer-1]
    rho = model[3*nLayer-1:4*nLayer-1]
    qp = model[4*nLayer-1:5*nLayer-1]
    qs = model[5*nLayer-1:6*nLayer-1]

    h_full = np.append(h, 0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod_file = os.path.join(tmpdir, "model.mod")
        with open(mod_file, "w") as f:
            f.write("MODEL.01\nwavefield\nISOTROPIC\nKGS\nFLAT EARTH\n1-D\nCONSTANT VELOCITY\n")
            f.write("LINE08\nLINE09\nLINE10\nLINE11\n")
            f.write("H(KM) VP(KM/S) VS(KM/S) RHO(GM/CC) QP QS ETAP ETAS FREFP FREFS\n")
            for i in range(nLayer):
                f.write(f"{h_full[i]:.4f} {vp[i]:.4f} {vs[i]:.4f} {rho[i]:.4f} {qp[i]:.1f} {qs[i]:.1f} 0 0 1 1\n")
        
        dfile = os.path.join(tmpdir, "dfile")
        with open(dfile, "w") as f:
            for i in range(1, N_TRACES + 1):
                dist = (i * DX) / 1000.0
                f.write(f"{dist} {DT} {N_SAMPLES} 0.0 0.0\n")
        
        subprocess.run([os.path.join(CPS_BIN, "sprep96"), "-M", "model.mod", "-d", "dfile", "-R", "-FMIN", str(F_MIN), "-FMAX", str(F_MAX)], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([os.path.join(CPS_BIN, "sdisp96")], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([os.path.join(CPS_BIN, "sregn96")], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        pulse_out = os.path.join(tmpdir, "pulse.out")
        with open(pulse_out, "w") as f:
            subprocess.run([os.path.join(CPS_BIN, "spulse96"), "-d", "dfile", "-V", "-p", "-l", "2"], cwd=tmpdir, stdout=f, stderr=subprocess.DEVNULL)
        
        subprocess.run([os.path.join(CPS_BIN, "f96tosac"), "-B", "pulse.out"], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        z_files = sorted(glob.glob(os.path.join(tmpdir, "*ZVF.sac")))

        if not z_files or len(z_files) != N_TRACES:
            return np.random.rand(N_TRACES * N_SAMPLES) * 1e9
            
        traces = []
        for zf in z_files:
            tr = read(zf)[0]
            try:
                tr.filter("bandpass", freqmin=F_MIN, freqmax=F_MAX, corners=4, zerophase=True)
            except Exception:
                pass

            d = tr.data
            d = np.nan_to_num(d, nan=0.0)
            if len(d) > N_SAMPLES: d = d[:N_SAMPLES]
            elif len(d) < N_SAMPLES: d = np.pad(d, (0, N_SAMPLES - len(d)))

            max_val = np.max(np.abs(d))
            if max_val > 0: d = d / max_val
            traces.append(d)
            
        return np.array(traces).flatten()

def procesar_un_campo_real(archivo_sg2):
    """Procesa un archivo de datos sísmicos reales en formato SG2.

    Esta función lee un archivo de datos sísmicos reales en formato SG2, aplica filtrado
    pasa-banda, remuestreo y normalización para preparar los datos para su inversión.
    Los datos procesados se extraen hasta el número de trazas especificado (N_TRACES)
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

    Returns:
        numpy.ndarray: Array 1D aplanado de forma (N_TRACES * N_SAMPLES,) que contiene
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
    for tr in st[:N_TRACES]:
        d = tr.data
        if len(d) > N_SAMPLES: d = d[:N_SAMPLES]
        elif len(d) < N_SAMPLES: d = np.pad(d, (0, N_SAMPLES - len(d)))

        max_val = np.max(np.abs(d))
        if max_val > 0: d = d / max_val
        traces.append(d)
        
    return np.array(traces).flatten()

def ejecutar_inversion_wavefield():
    global N_TRACES
    print("[ETAPA 1] Leyendo metadata de datos reales...")
    archivos = sorted(glob.glob("datos_sg2/*.sg2"))
    if not archivos:
        raise FileNotFoundError("No se encontraron archivos en datos_sg2/")
    
    # Detectar el número de trazas dinámicamente usando el primer archivo
    st_test = read(archivos[0])
    N_TRACES = len(st_test)
    print(f"[INFO] Se detectaron dinámicamente {N_TRACES} trazas (geófonos) en los archivos .sg2")

    # H_min, H_max, Vs_min, Vs_max, Vp_min, Vp_max, Rho_min, Rho_max, Qp_min, Qp_max, Qs_min, Qs_max
    prior_matrix = np.array([
        [0.0005, 0.003, 0.100, 0.300, 0.300, 0.600, 1.4, 1.6, 20, 100, 5, 30],      # Capa 1 (superficial)
        [0.002,  0.010, 0.150, 0.350, 0.400, 0.800, 1.6, 1.8, 20, 100, 5, 40],      # Capa 2
        [0.005,  0.015, 0.300, 0.550, 0.800, 1.200, 1.8, 2.0, 30, 150, 10, 60],     # Capa 3
        [0.010,  0.020, 0.500, 0.900, 1.200, 1.800, 2.0, 2.2, 50, 200, 20, 100],    # Capa 4
        [0.000,  0.000, 0.800, 1.500, 1.800, 3.000, 2.2, 2.5, 80, 300, 30, 150],    # Semiespacio (halfspace, H=0)
    ])
    N_LAYER = len(prior_matrix)
    
    ListPrior = []
    NamesFull = ["Thickness", "Vs", "Vp", "Rho", "Qp", "Qs"]
    Units = [" [km]", " [km/s]", " [km/s]", " [g/cc]", "", ""]
    NamesFU = []
    Mins = []
    Maxs = []

    for j in range(6):
        for i in range(N_LAYER):
            if (i == N_LAYER - 1) and (j == 0): continue
            cmin = prior_matrix[i, j*2]
            cmax = prior_matrix[i, j*2+1]
            ListPrior.append(stats.uniform(loc=cmin, scale=cmax - cmin))
            Mins.append(cmin)
            Maxs.append(cmax)
            NamesFU.append(f"{NamesFull[j]} {i+1}{Units[j]}")

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

    paramNames = {"NamesFU": NamesFU, "NamesSU": NamesFU, "NamesS": NamesFU, 
                  "NamesGlobal": NamesFull, "NamesGlobalS": NamesFull, 
                  "DataUnits": "Amplitude", "DataName": "Wavefield", "DataAxis": "Time [s]"}

    Timing = np.linspace(0, N_SAMPLES * DT, N_SAMPLES * N_TRACES)

    print("\n[ETAPA 2] Configurando BEL1D MODELSET...")
    ModelSet = BEL1D.MODELSET(prior=ListPrior, cond=cond, method="Wavefield",
                              forwardFun={"Fun": cps_forward_wavefield, "Axis": Timing}, 
                              paramNames=paramNames, nbLayer=N_LAYER, logTransform=[False, False])

    print("\n[ETAPA 3] Corriendo PREBEL (Simulaciones Iniciales Prior)...")
    N_MODELS = 500
    pool = pp.ProcessPool(mp.cpu_count())
    Prebel = BEL1D.PREBEL(ModelSet, nbModels=N_MODELS)
    Prebel.run(Parallelization=[True, pool], verbose=True)
    pool.terminate()

    print(f"\n[INFO] Modelos sintéticos (Prior) generados: {Prebel.MODELS.shape[0]}")

    perfil_2d = []
    modelos_profundidad = []

    print("\n[ETAPA 4] Iniciando iteración POSTBEL sobre los archivos (Perfil 2D)...")
    for idx, archivo in enumerate(archivos):
        print(f"\n[INFO] Procesando disparo: {os.path.basename(archivo)} ({idx+1}/{len(archivos)})...")
        Dataset = procesar_un_campo_real(archivo)

        Postbel = BEL1D.POSTBEL(Prebel)
        Postbel.run(Dataset=Dataset, nbSamples=500, NoiseModel=None) 

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
                
            try:
                fig_comp, axes = plt.subplots(1, 3, figsize=(18, 8))
                fig_comp.patch.set_facecolor("#0d1117")
                for ax in axes: ax.set_facecolor("#0d1117")
                
                dist_grid = np.arange(1, N_TRACES + 1) * DX
                time_grid = np.arange(N_SAMPLES) * DT
                datos_reales = Dataset.reshape(N_TRACES, N_SAMPLES)
                mejor_forward = cps_forward_wavefield(mean_model).reshape(N_TRACES, N_SAMPLES)
                
                # a) Real
                ax = axes[0]
                for i in range(N_TRACES):
                    ax.plot(datos_reales[i, :] + dist_grid[i], time_grid, color="#00e5ff", lw=1)
                    ax.fill_betweenx(time_grid, dist_grid[i], datos_reales[i, :] + dist_grid[i], where=(datos_reales[i, :]>0), color="#00e5ff", alpha=0.5)
                ax.invert_yaxis()
                ax.set_ylim(1.5, 0)
                ax.set_title("Datos Reales (.sg2)\n(Wiggle Plot)", color="white")
                ax.set_ylabel("Tiempo (s)", color="white")
                ax.set_xlabel("Distancia (m)", color="white")
                
                # b) Sintético
                ax = axes[1]
                for i in range(N_TRACES):
                    ax.plot(mejor_forward[i, :] + dist_grid[i], time_grid, color="#76ff03", lw=1)
                    ax.fill_betweenx(time_grid, dist_grid[i], mejor_forward[i, :] + dist_grid[i], where=(mejor_forward[i, :]>0), color="#76ff03", alpha=0.5)
                ax.invert_yaxis()
                ax.set_ylim(1.5, 0)
                ax.set_title("Sintético: Mean Model Posterior", color="white")
                ax.set_xlabel("Distancia (m)", color="white")
                
                # c) Efecto Q
                model_elastic = mean_model.copy()
                model_elastic[4*nLayer_model-1:6*nLayer_model-1] = 5000
                model_anelastic = mean_model.copy()
                for i in range(nLayer_model-1):
                    model_anelastic[5*nLayer_model-1 + i] = 5
                
                fwd_el = cps_forward_wavefield(model_elastic).reshape(N_TRACES, N_SAMPLES)
                fwd_anel = cps_forward_wavefield(model_anelastic).reshape(N_TRACES, N_SAMPLES)
                
                ax = axes[2]
                for i in range(N_TRACES):
                    ax.plot(fwd_el[i, :] + dist_grid[i], time_grid, color="white", lw=1.5, label="Elástico (Q=5000)" if i==0 else "")
                    ax.plot(fwd_anel[i, :] + dist_grid[i], time_grid, color="#ff3d00", lw=1.5, ls="--", label="Anelástico (Qs=5)" if i==0 else "")
                ax.invert_yaxis()
                ax.set_ylim(1.5, 0)
                ax.set_title("Efecto de Q (Elástico vs Anelástico)", color="white")
                ax.set_xlabel("Distancia (m)", color="white")
                ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9, framealpha=0.8)
                
                for ax in axes:
                    ax.tick_params(colors="white")
                    ax.spines[:].set_color("#444")
                    ax.grid(True, color="#333", lw=0.3, alpha=0.4)
                    
                fig_comp.tight_layout()
                fig_comp.savefig("wavefields_comparativa.png", dpi=150, facecolor=fig_comp.get_facecolor())
                plt.close(fig_comp)
                print("-> Archivo 'wavefields_comparativa.png' guardado.")
            except Exception as e:
                print(f"[WARNING] Falló la comparativa wiggle plot: {e}")
        
        perfil_2d.append(vs)
        modelos_profundidad.append(h)

    print("\n[ETAPA 5] Generando Perfil 2D Final...")
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(1, len(archivos) + 1)

    mean_h = np.mean(modelos_profundidad, axis=0)
    z_interfaces = np.zeros(N_LAYER)
    for i in range(N_LAYER - 1):
        z_interfaces[i+1] = z_interfaces[i] + mean_h[i]
        
    z_grid = list(z_interfaces)
    z_grid.append(z_interfaces[-1] + 0.020)
    z_grid = np.array(z_grid) * 1000
    
    V = np.array(perfil_2d).T
    
    X, Z = np.meshgrid(np.append(x, x[-1]+1) - 0.5, z_grid)
    
    mesh = ax.pcolormesh(X, Z, V * 1000, cmap="jet", shading="flat")
    ax.invert_yaxis()
    ax.set_title("Perfil 2D de Onda de Corte (Vs) - MASW Wavefield Inversion")
    ax.set_ylabel("Profundidad (m)")
    ax.set_xlabel("Número de Disparo (Shot)")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Velocidad Vs (m/s)")
    
    fig.tight_layout()
    out_fig = "perfil_2d_Vs.png"
    fig.savefig(out_fig, dpi=150)
    print(f"\n[LISTO] Figura guardada en: {out_fig}")
    plt.show()

if __name__ == "__main__":
    ejecutar_inversion_wavefield()
