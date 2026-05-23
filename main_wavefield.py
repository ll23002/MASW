import sys, os, warnings, tempfile, subprocess
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
import glob

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
DT = 0.002 # 500Hz
N_SAMPLES = 8192  # 8192 * 0.002s = 16.384s, df = 1/T = 1/16.384 = 0.061 Hz

def cps_forward_wavefield(model):
    """Calcula el campo de ondas sintético a partir de un modelo 1-D.

    La función escribe archivos temporales para CPS, ejecuta los binarios
    `sprep96`, `sdisp96`, `sregn96`, `spulse96` y `f96tosac`, y luego lee los
    trazos SAC generados para construir un vector aplanado de amplitud
    normalizada.

    El vector `model` debe contener los parámetros del subsuelo en este orden:
    espesores, velocidades de onda S (`Vs`), velocidades de onda P (`Vp`),
    densidades (`rho`), factores de calidad de P (`Qp`) y factores de calidad
    de S (`Qs`). Para `nLayer` capas, la longitud esperada es `6 * nLayer - 1`
    porque la capa inferior corresponde a un semiespacio y no tiene espesor.

    Args:
        model: Vector unidimensional con los parámetros del modelo en el orden
            descrito arriba. Las unidades esperadas son kilómetros para
            espesores, km/s para velocidades, g/cc para densidad y valores
            adimensionales para `Qp` y `Qs`.

    Returns:
        numpy.ndarray: Vector unidimensional de longitud `24 * N_SAMPLES` con
        las 24 trazas horizontales aplanadas y normalizadas por su amplitud
        máxima absoluta.

    Notes:
        - Si la generación de archivos SAC falla o no se obtienen las 24
          trazas esperadas, la función devuelve un vector aleatorio grande para
          evitar varianza cero y problemas numéricos posteriores.
        - La función depende de la ruta definida en `CPS_BIN` y de la
          disponibilidad de los ejecutables de CPS en ese directorio.

    """
    nLayer = (len(model) + 1) // 6
    h = model[0:nLayer-1]
    vs = model[nLayer-1:2*nLayer-1]
    vp = model[2*nLayer-1:3*nLayer-1]
    rho = model[3*nLayer-1:4*nLayer-1]
    qp = model[4*nLayer-1:5*nLayer-1]
    qs = model[5*nLayer-1:6*nLayer-1]

    # Halfspace
    h_full = np.append(h, 0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        mod_file = os.path.join(tmpdir, "model.mod")
        with open(mod_file, "w") as f:
            f.write("MODEL.01\nwavefield\nISOTROPIC\nKGS\nFLAT EARTH\n1-D\nCONSTANT VELOCITY\n")
            f.write("LINE08\nLINE09\nLINE10\nLINE11\n")
            f.write("H(KM) VP(KM/S) VS(KM/S) RHO(GM/CC) QP QS ETAP ETAS FREFP FREFS\n")
            for i in range(nLayer):
                f.write(f"{h_full[i]:.4f} {vp[i]:.4f} {vs[i]:.4f} {rho[i]:.4f} {qp[i]:.1f} {qs[i]:.1f} 0 0 1 1\n")
        
        # Por que se asume 24 geofonos en todo el código?
        dfile = os.path.join(tmpdir, "dfile")
        with open(dfile, "w") as f:
            for i in range(1, 25):
                dist = (i * DX) / 1000.0 #km
                f.write(f"{dist} {DT} {N_SAMPLES} 0.0 0.0\n")
        
        # CPS binaries (Modal summation for Rayleigh waves)
        subprocess.run([os.path.join(CPS_BIN, "sprep96"), "-M", "model.mod", "-d", "dfile", "-R", "-FMIN", str(F_MIN), "-FMAX", str(F_MAX)], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([os.path.join(CPS_BIN, "sdisp96")], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([os.path.join(CPS_BIN, "sregn96")], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        pulse_out = os.path.join(tmpdir, "pulse.out")
        with open(pulse_out, "w") as f:
            subprocess.run([os.path.join(CPS_BIN, "spulse96"), "-d", "dfile", "-V", "-p", "-l", "2"], cwd=tmpdir, stdout=f, stderr=subprocess.DEVNULL)
        
        subprocess.run([os.path.join(CPS_BIN, "f96tosac"), "-B", "pulse.out"], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


        z_files = sorted(glob.glob(os.path.join(tmpdir, "*ZVF.sac")))

        if not z_files or len(z_files) != 24:
            return np.random.rand(24 * N_SAMPLES) * 1e9
            
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

            # Normalizar la traza para comparar forma de onda
            max_val = np.max(np.abs(d))
            if max_val > 0: d = d / max_val
            traces.append(d)
            
        return np.array(traces).flatten()



def procesar_campo_real(ruta_archivos):
    """
    Lee los sg2, hace downsampling y recorta a N_SAMPLES muestras.
    Retorna vector aplanado de 24 x N_SAMPLES.
    """
    archivos = sorted(glob.glob(ruta_archivos))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos en {ruta_archivos}")
    
    # Tomaremos el primer archivo para el ejemplo
    st = read(archivos[0])
    st.filter("bandpass", freqmin=F_MIN, freqmax=F_MAX, corners=4, zerophase=True)
    st.resample(1.0 / DT)
    
    traces = []
    for tr in st[:24]:
        d = tr.data
        if len(d) > N_SAMPLES: d = d[:N_SAMPLES]
        elif len(d) < N_SAMPLES: d = np.pad(d, (0, N_SAMPLES - len(d)))
        
        # Mute antes de la llegada de onda? Para simpleza solo normalizamos
        max_val = np.max(np.abs(d))
        if max_val > 0: d = d / max_val
        traces.append(d)
        
    return np.array(traces).flatten()


def ejecutar_inversion_wavefield():
    print("[ETAPA 1] Leyendo datos reales (campo de ondas)...")
    Dataset = procesar_campo_real("datos_sg2/*.sg2")
    
    print("[INFO] Vector de datos reales (aplanado):", Dataset.shape)

    # 5 capas (4 finitas + 1 semiespacio)
    # RANGOS (prior): h(km), Vs(km/s), Vp(km/s), Rho(g/cc), Qp, Qs
    # Ahora Vp, Qp y Qs están libres.
    prior_matrix = np.array([
        #  h_min, h_max, Vs_min, Vs_max, Vp_min, Vp_max, Rho_min, Rho_max, Qp_min, Qp_max, Qs_min, Qs_max
        [0.0005, 0.003, 0.100, 0.300, 0.300, 0.600, 1.4, 1.6, 20, 100, 5, 30],
        [0.002,  0.010, 0.150, 0.350, 0.400, 0.800, 1.6, 1.8, 20, 100, 5, 40],
        [0.005,  0.015, 0.300, 0.550, 0.800, 1.200, 1.8, 2.0, 30, 150, 10, 60],
        [0.010,  0.020, 0.500, 0.900, 1.200, 1.800, 2.0, 2.2, 50, 200, 20, 100],
        [0.000,  0.000, 0.800, 1.500, 1.800, 3.000, 2.2, 2.5, 80, 300, 30, 150],
    ])
    N_LAYER = len(prior_matrix)
    
    from scipy import stats
    ListPrior = []
    NamesFull = ["Thickness", "Vs", "Vp", "Rho", "Qp", "Qs"]
    Units = [" [km]", " [km/s]", " [km/s]", " [g/cc]", "", ""]
    NamesFU = []
    Mins = []
    Maxs = []

    for j in range(6):
        for i in range(N_LAYER):
            if (i == N_LAYER - 1) and (j == 0): continue # h del semiespacio es infinito
            cmin = prior_matrix[i, j*2]
            cmax = prior_matrix[i, j*2+1]
            ListPrior.append(stats.uniform(loc=cmin, scale=cmax - cmin))
            Mins.append(cmin)
            Maxs.append(cmax)
            NamesFU.append(f"{NamesFull[j]} {i+1}{Units[j]}")

    def cond(model):
        return (np.logical_and(np.greater_equal(model, Mins), np.less_equal(model, Maxs))).all()

    # Como DataName y DataAxis podemos usar Time
    paramNames = {"NamesFU": NamesFU, "NamesSU": NamesFU, "NamesS": NamesFU, 
                  "NamesGlobal": NamesFull, "NamesGlobalS": NamesFull, 
                  "DataUnits": "Amplitude", "DataName": "Wavefield", "DataAxis": "Time [s]"}

    # Time vector (solo referencial)
    Timing = np.linspace(0, N_SAMPLES * DT, N_SAMPLES * 24)

    print("[ETAPA 2] Configurando BEL1D MODELSET...")
    ModelSet = BEL1D.MODELSET(prior=ListPrior, cond=cond, method="Wavefield", 
                              forwardFun={"Fun": cps_forward_wavefield, "Axis": Timing}, 
                              paramNames=paramNames, nbLayer=N_LAYER, logTransform=[False, False])
                              
    print("[ETAPA 3] Corriendo BEL1D (Simulaciones Iniciales)...")
    N_MODELS = 500  # Reducido para que termine pronto
    pool = pp.ProcessPool(mp.cpu_count())
    Prebel = BEL1D.PREBEL(ModelSet, nbModels=N_MODELS)
    Prebel.run(Parallelization=[True, pool], verbose=True)
    pool.terminate()

    samples = Prebel.MODELS
    sampDC = Prebel.FORWARD

    print(f"\n[INFO] Modelos sintéticos generados: {samples.shape[0]}")
    
    # RMSE calculations
    # Error: RMSE between real Dataset and synthetic
    # sampDC has shape (N_MODELS, 24*N_SAMPLES)
    rmse = np.sqrt(np.nanmean(((Dataset - sampDC)) ** 2, axis=1))
    finite_mask = np.isfinite(rmse)
    rmse_ok = rmse[finite_mask]
    samp_ok = samples[finite_mask]
    sampDC_ok = sampDC[finite_mask]
    
    p_threshold = np.percentile(rmse_ok, 10)
    top_mask = rmse_ok <= p_threshold
    rmse_top = rmse_ok[top_mask]
    samp_top = samp_ok[top_mask]
    sampDC_top = sampDC_ok[top_mask]
    
    idx_mejor = np.argmin(rmse_top)
    mejor_modelo = samp_top[idx_mejor]
    mejor_forward = sampDC_top[idx_mejor].reshape(24, N_SAMPLES)
    datos_reales = Dataset.reshape(24, N_SAMPLES)
    
    print(f"[INFO] RMSE Mejor modelo: {rmse_top[idx_mejor]:.4f}")
    
    print("[ETAPA 4] Generando Gráficas CCA (Mreyen & Eppinger)...")
    # Genera los CCA de pyBEL1D, esto toma Dataset internamente
    # Para la inversión completa con Dataset:
    # Postbel = BEL1D.POSTBEL(Prebel)
    # Postbel.run(Dataset=Dataset)
    # fig_cca = Postbel.ShowDataset() ...
    
    # Para no complicar con la clase Postbel y el ruido, mostraremos simplemente la sensibilidad 
    # visualizando el Wiggle plot y las simulaciones con Q variable
    
    print("[ETAPA 5] Simulaciones Anelásticas comparativas y Wiggle Plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes: ax.set_facecolor("#0d1117")
    
    dist_grid = np.arange(1, 25) * DX
    time_grid = np.arange(N_SAMPLES) * DT
    
    # a) Real
    ax = axes[0]
    for i in range(24):
        ax.plot(datos_reales[i, :] + dist_grid[i], time_grid, color="#00e5ff", lw=1)
        ax.fill_betweenx(time_grid, dist_grid[i], datos_reales[i, :] + dist_grid[i], where=(datos_reales[i, :]>0), color="#00e5ff", alpha=0.5)
    ax.invert_yaxis()
    ax.set_title("Datos Reales (.sg2)\n(Wiggle Plot)", color="white")
    ax.set_ylabel("Tiempo (s)", color="white")
    ax.set_xlabel("Distancia (m)", color="white")
    
    # b) Sintético Ganador
    ax = axes[1]
    for i in range(24):
        ax.plot(mejor_forward[i, :] + dist_grid[i], time_grid, color="#76ff03", lw=1)
        ax.fill_betweenx(time_grid, dist_grid[i], mejor_forward[i, :] + dist_grid[i], where=(mejor_forward[i, :]>0), color="#76ff03", alpha=0.5)
    ax.invert_yaxis()
    ax.set_title("Sintético: Mejor Modelo posterior", color="white")
    ax.set_xlabel("Distancia (m)", color="white")
    
    # c) Superposición Q elástico vs anelástico
    # Creamos dos modelos de prueba basados en el mejor
    model_elastic = mejor_modelo.copy()
    model_elastic[4*N_LAYER-1:6*N_LAYER-1] = 5000  # Q muy alto = elástico
    
    model_anelastic = mejor_modelo.copy()
    # Forzar un Qs bajo (ej 5) en los estratos superiores
    for i in range(N_LAYER-1):
        model_anelastic[5*N_LAYER-1 + i] = 5  # Qs bajo
        
    fwd_el = cps_forward_wavefield(model_elastic).reshape(24, N_SAMPLES)
    fwd_anel = cps_forward_wavefield(model_anelastic).reshape(24, N_SAMPLES)
    
    ax = axes[2]
    for i in range(24):
        ax.plot(fwd_el[i, :] + dist_grid[i], time_grid, color="white", lw=1.5, label="Elástico (Q=5000)" if i==0 else "")
        ax.plot(fwd_anel[i, :] + dist_grid[i], time_grid, color="#ff3d00", lw=1.5, ls="--", label="Anelástico (Qs=5)" if i==0 else "")
    ax.invert_yaxis()
    ax.set_title("Efecto de Q (Elástico vs Anelástico)", color="white")
    ax.set_xlabel("Distancia (m)", color="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9, framealpha=0.8)
    
    for ax in axes:
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.grid(True, color="#333", lw=0.3, alpha=0.4)
        
    plt.tight_layout()
    out_fig = "wavefields_comparativa.png"
    plt.savefig(out_fig, dpi=150, facecolor=fig.get_facecolor())
    print(f"\n[LISTO] Figura guardada en: {out_fig}")
    plt.show()

if __name__ == "__main__":
    ejecutar_inversion_wavefield()
