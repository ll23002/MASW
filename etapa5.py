import numpy as np
import matplotlib.pyplot as plt

from config import DT, N_SAMPLES, N_TRACES, DX
from etapa2 import cps_forward_wavefield


def wiggle_plot_comparativa(datos_reales, mejor_forward, mean_model, n_layer):
    """Genera la figura comparativa de 3 paneles con wiggle plots.

    Panel 1: Datos reales (campo de ondas observado).
    Panel 2: Mejor sismograma sintético del Prebel/Postbel.
    Panel 3: Superposición elástico (Q=5000) vs anelástico (Qs=5) para evaluar atenuación.

    Args:
        datos_reales (numpy.ndarray): Array (N_TRACES, N_SAMPLES) del campo real.
        mejor_forward (numpy.ndarray): Array (N_TRACES, N_SAMPLES) del mejor sintético.
        mean_model (numpy.ndarray): Vector de parámetros del modelo medio posterior.
        n_layer (int): Número de capas del modelo.

    Returns:
        matplotlib.figure.Figure: Figura generada (ya guardada en disco).
    """
    nLayer_model = n_layer

    dist_grid = np.arange(1, N_TRACES + 1) * DX
    time_grid = np.arange(N_SAMPLES) * DT

    fig_comp, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig_comp.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")

    # Panel 1: Real
    ax = axes[0]
    for i in range(N_TRACES):
        ax.plot(datos_reales[i, :] + dist_grid[i], time_grid, color="#00e5ff", lw=1)
        ax.fill_betweenx(time_grid, dist_grid[i], datos_reales[i, :] + dist_grid[i],
                         where=(datos_reales[i, :] > 0), color="#00e5ff", alpha=0.5)
    ax.invert_yaxis()
    ax.set_ylim(1.5, 0)
    ax.set_title("Datos Reales (.sg2)\n(Wiggle Plot)", color="white")
    ax.set_ylabel("Tiempo (s)", color="white")
    ax.set_xlabel("Distancia (m)", color="white")

    # Panel 2: Sintético
    ax = axes[1]
    for i in range(N_TRACES):
        ax.plot(mejor_forward[i, :] + dist_grid[i], time_grid, color="#76ff03", lw=1)
        ax.fill_betweenx(time_grid, dist_grid[i], mejor_forward[i, :] + dist_grid[i],
                         where=(mejor_forward[i, :] > 0), color="#76ff03", alpha=0.5)
    ax.invert_yaxis()
    ax.set_ylim(1.5, 0)
    ax.set_title("Sintético: Mean Model Posterior", color="white")
    ax.set_xlabel("Distancia (m)", color="white")

    # Panel 3: Efecto Q (Elástico vs Anelástico)
    model_elastic = mean_model.copy()
    model_elastic[4*nLayer_model-1:6*nLayer_model-1] = 5000

    model_anelastic = mean_model.copy()
    for i in range(nLayer_model-1):
        model_anelastic[5*nLayer_model-1 + i] = 5

    fwd_el = cps_forward_wavefield(model_elastic).reshape(N_TRACES, N_SAMPLES)
    fwd_anel = cps_forward_wavefield(model_anelastic).reshape(N_TRACES, N_SAMPLES)

    ax = axes[2]
    for i in range(N_TRACES):
        ax.plot(fwd_el[i, :] + dist_grid[i], time_grid, color="white", lw=1.5,
                label="Elástico (Q=5000)" if i == 0 else "")
        ax.plot(fwd_anel[i, :] + dist_grid[i], time_grid, color="#ff3d00", lw=1.5, ls="--",
                label="Anelástico (Qs=5)" if i == 0 else "")
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
    print("Archivo 'wavefields_comparativa.png' guardado.")

    return fig_comp


def generar_perfil_2d(perfil_2d, modelos_profundidad, n_archivos, n_layer):
    """Genera y guarda el perfil 2D de Vs a lo largo del tendido sísmico.

    Args:
        perfil_2d (list): Lista de arrays Vs por disparo (shape de cada uno: (N_LAYER,)).
        modelos_profundidad (list): Lista de arrays de espesores por disparo.
        n_archivos (int): Número total de disparos (archivos SG2).
        n_layer (int): Número de capas del modelo.
    """
    print("\n[ETAPA 5] Generando Perfil 2D Final...")
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(1, n_archivos + 1)

    mean_h = np.mean(modelos_profundidad, axis=0)
    z_interfaces = np.zeros(n_layer)
    for i in range(n_layer - 1):
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
