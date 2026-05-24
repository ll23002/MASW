"""
Etapa 2 - Forward Modeling Automático con CPS
=============================================
Simula la propagación de ondas en la tierra dados los parámetros del subsuelo.
Recibe un vector de propiedades del suelo y retorna el sismograma sintético como
vector plano (N_TRACES * N_SAMPLES,).
"""

import os, tempfile, subprocess
import numpy as np
import glob
from obspy import read

from config import CPS_BIN, DX, F_MIN, F_MAX, DT, N_SAMPLES, N_TRACES


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
