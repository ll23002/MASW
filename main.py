import sys, os, warnings
import config  # noqa: F401

warnings.filterwarnings("ignore")

BEL1D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pyBEL1D_src")
if BEL1D_PATH not in sys.path:
    sys.path.insert(0, BEL1D_PATH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from etapa1 import cargar_archivos_sg2, detectar_n_trazas, procesar_un_campo_real
from etapa3 import construir_modelset
from etapa4 import (
    correr_prebel,
    calcular_rmse_y_mejor_modelo,
    correr_postbel_y_graficas,
)
from etapa5 import wiggle_plot_comparativa, generar_perfil_2d
import config as cfg


def ejecutar_inversion_wavefield():
    # ETAPA 1 - Preprocesamiento del Campo de Ondas Real
    print("[ETAPA 1] Leyendo metadata de datos reales...")
    archivos = cargar_archivos_sg2("datos_sg2")

    cfg.N_TRACES = detectar_n_trazas(archivos)

    # Actualizar N_TRACES en los módulos que lo usan
    import etapa2, etapa4, etapa5
    etapa2.N_TRACES = cfg.N_TRACES
    etapa4.N_TRACES = cfg.N_TRACES
    etapa5.N_TRACES = cfg.N_TRACES

    # ETAPA 2 - ETAPA 3 - Definición del Prior y configuración de MODELSET
    print("\n[ETAPA 2] Configurando BEL1D MODELSET...")
    ModelSet = construir_modelset(cfg.N_LAYER)

    # ETAPA 4 - Inversión Estocástica: PREBEL + POSTBEL + RMSE
    print("\n[ETAPA 3] Corriendo PREBEL (Simulaciones Iniciales Prior)...")
    Prebel = correr_prebel(ModelSet, n_models=500)

    # Calcular RMSE del primer disparo vs prior para reportar el mejor modelo
    dataset_primer_disparo = procesar_un_campo_real(archivos[0], cfg.N_TRACES)
    mejor_modelo, mejor_forward = calcular_rmse_y_mejor_modelo(Prebel, dataset_primer_disparo)

    print("\n[ETAPA 4] Iniciando iteración POSTBEL sobre los archivos (Perfil 2D)...")
    perfil_2d = []
    modelos_profundidad = []

    for idx, archivo in enumerate(archivos):
        dataset_real = procesar_un_campo_real(archivo, cfg.N_TRACES)
        vs, h = correr_postbel_y_graficas(
            Prebel, dataset_real,
            os.path.basename(archivo),
            idx, cfg.N_LAYER
        )
        perfil_2d.append(vs)
        modelos_profundidad.append(h)

    # ETAPA 5 - Visualización: Wiggle Plots y Atenuación + Perfil 2D
    print("\n[ETAPA 5] Generando visualizaciones...")

    try:
        datos_reales_2d = dataset_primer_disparo.reshape(cfg.N_TRACES, cfg.N_SAMPLES)
        nLayer_model = (len(mejor_modelo) + 1) // 6
        wiggle_plot_comparativa(datos_reales_2d, mejor_forward, mejor_modelo, nLayer_model)
    except Exception as e:
        print(f"[WARNING] Falló la comparativa wiggle plot: {e}")

    generar_perfil_2d(perfil_2d, modelos_profundidad, len(archivos), cfg.N_LAYER)


if __name__ == "__main__":
    ejecutar_inversion_wavefield()
