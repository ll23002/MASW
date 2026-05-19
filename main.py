import sys, os, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d
warnings.filterwarnings("ignore")

BEL1D_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pyBEL1D_src")
if BEL1D_PATH not in sys.path:
    sys.path.insert(0, BEL1D_PATH)

from pyBEL1D import BEL1D
from pathos import multiprocessing as mp, pools as pp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problema3 import procesar_datos_fk
from problema5 import ejecutar_clustering


print("\n[ETAPA 1] Procesando registros SEG-2...")
res3 = procesar_datos_fk(ruta_archivos="datos_sg2/*.sg2", dx=2.0, f_max=150.0)

print("\n[ETAPA 2] Clustering DBSCAN → curva observada...")
res5 = ejecutar_clustering(
    datos_crudos=res3["picos_indices"],
    f_max=res3["f_max"], dx=res3["dx"],
    filas_150hz=res3["filas_150hz"], columnas_e=res3["columnas_e"],
    eps=15, min_samples=3,
)

f_raw  = res5["curva_dispersion"]["f"]   # Hz
vc_raw = res5["curva_dispersion"]["vc"]  # m/s



# Descarta el ruido del clustering DBSCAN que no cumplen con el comportamiento geofísico esperado
tol_mono = 30.0
idx_s = np.argsort(f_raw)
f_s   = f_raw[idx_s]
vc_s  = vc_raw[idx_s]
keep = [0]
for i in range(1, len(f_s)):
    if vc_s[i] <= vc_s[keep[-1]] + tol_mono:
        keep.append(i)
f_obs  = f_s[keep]
vc_obs = vc_s[keep]
print(f"\n[INFO] Curva observada: {len(f_raw)} pts brutos → {len(f_obs)} pts físicamente válidos")
print(f"       f = {f_obs[0]:.0f}–{f_obs[-1]:.0f} Hz  |  Vc = {vc_obs.min():.0f}–{vc_obs.max():.0f} m/s")


vc_km = vc_obs / 1000.0


N_INTERP = 20
f_interp  = np.linspace(f_obs[0], f_obs[-1], N_INTERP)
interp_fn = interp1d(f_obs, vc_km, kind='linear',
                     bounds_error=False, fill_value=(vc_km[0], vc_km[-1]))
Dataset   = interp_fn(f_interp)
Frequency = f_interp

print(f"\n[INFO] Curva interpolada a {N_INTERP} puntos para BEL1D PCA")
print(f"f = {Frequency[0]:.1f}–{Frequency[-1]:.1f} Hz")
print(f"Vc = {Dataset.min()*1000:.0f}–{Dataset.max()*1000:.0f} m/s")

NoiseModel = np.array([
    (0.10 * vc * 1000.0 + 20.0 / f) / 1000.0
    for vc, f in zip(Dataset, Frequency)
])
print(f"σ = {NoiseModel.min()*1000:.1f}–{NoiseModel.max()*1000:.1f} m/s")

#  Capa | Descripción                    | h (m)    | Vs (m/s)
#  -----+--------------------------------+----------+----------
#    1  | Suelo orgánico / coluvión      | 0.5 – 3  | 100 – 300
#    2  | Tierra Blanca Joven suelta     | 2   – 10 | 150 – 350
#    3  | TBJ compacta / piroclásticos   | 5   – 15 | 300 – 550
#    4  | Tobas muy compactadas          | 10  – 20 | 500 – 900
#    5  | Basamento / Fm. Bálsamo (∞)   | —        | 800 – 1500
prior = np.array([
    [0.0005, 0.003,  0.100, 0.300],   # C1
    [0.002,  0.010,  0.150, 0.350],   # C2
    [0.005,  0.015,  0.300, 0.550],   # C3
    [0.010,  0.020,  0.500, 0.900],   # C4
    [0.000,  0.000,  0.800, 1.500],   # C5 semiespacio
])
N_LAYER = len(prior)

# Vp y densidades fijas por capa
# El factor 1.87 viene de la relación elástica para convertir Vs a Vp usando un coeficiente de Poisson (\nu=0.3)
VP_FIXED  = np.array([1.87, 1.87, 1.87, 1.87, 1.87]) * prior[:, 2:4].mean(axis=1)
# El investigador proporciona estos datos
RHO_FIXED = np.array([1.5, 1.7, 1.9, 2.1, 2.4])  # g/cm³ = T/m³


print("\n[ETAPA 3] Construyendo MODELSET.DCVs para BEL1D...")

ModelSet = BEL1D.MODELSET.DCVs(
    prior=prior,
    Frequency=Frequency,
    VpFixed=VP_FIXED,
    RhoFixed=RHO_FIXED,
)
print(f"[INFO] MODELSET construido: {N_LAYER} capas, {N_INTERP} frecuencias")


print("\n[ETAPA 4] Ejecutando BEL1D (Generación de Modelos)...")

N_MODELS = 3000
pool = pp.ProcessPool(mp.cpu_count())

Prebel = BEL1D.PREBEL(ModelSet, nbModels=N_MODELS)
Prebel.run(Parallelization=[True, pool], verbose=True)
pool.terminate()

samples = Prebel.MODELS # Matriz de modelos sintéticos
sampDC  = Prebel.FORWARD # Matriz de respuestas forward para cada modelo

print(f"\n[INFO] Modelos sintéticos generados: {samples.shape[0]}")

# Calculamos Misfit para todos los modelos
rmse = np.sqrt(np.nanmean(((Dataset - sampDC) / Dataset) ** 2, axis=1))
finite_mask = np.isfinite(rmse)
rmse_ok  = rmse[finite_mask]
samp_ok  = samples[finite_mask]
sampDC_ok = sampDC[finite_mask]

# Se selecciona el 10% con menor RMSE
p_threshold = np.percentile(rmse_ok, 10)
top_mask = rmse_ok <= p_threshold
rmse_top = rmse_ok[top_mask]
samp_top = samp_ok[top_mask]
sampDC_top = sampDC_ok[top_mask]

print(f"[INFO] Seleccionados {len(rmse_top)} mejores modelos (Top 10%)")
print(f"[INFO] RMSE: min={rmse_top.min():.4f}  med={np.median(rmse_top):.4f}  max={rmse_top.max():.4f}")

Z_MAX   = 0.050   # km
dz      = 0.00025
z_grid  = np.arange(0, Z_MAX + dz, dz)

vs_min_plot = prior[:, 2].min()   # 0.1 km/s
vs_max_plot = prior[:, 3].max()   # 1.5 km/s
N_VS_BINS   = 200
vs_bins     = np.linspace(vs_min_plot, vs_max_plot, N_VS_BINS + 1)
vs_ctrs     = 0.5 * (vs_bins[:-1] + vs_bins[1:])

pesos = 1.0 / (rmse_ok + 1e-9)
pesos /= pesos.max()

densidad = np.zeros((len(z_grid), N_VS_BINS), dtype=float)

# Contruye una matriz que representa la probabilidad/densidad de que cada velocidad aparezca a cada profundidad
for ii in range(len(samp_top)):
    h_arr  = samp_top[ii, 0:N_LAYER - 1]   # km (4 espesores)
    vs_arr = samp_top[ii, N_LAYER - 1:]     # km/s (5 Vs)
    w = pesos[ii]
    p = 0.0
    for ci in range(N_LAYER):
        p_end = (p + h_arr[ci]) if ci < N_LAYER - 1 else Z_MAX + 1
        mask_z = (z_grid >= p) & (z_grid < p_end)
        bi = np.searchsorted(vs_bins, vs_arr[ci]) - 1
        bi = np.clip(bi, 0, N_VS_BINS - 1)
        densidad[mask_z, bi] += w
        p = p_end if ci < N_LAYER - 1 else p

# vs_mediana: arreglo con la velocidad mediana para cada profundidad
vs_mediana = np.zeros(len(z_grid))
for zi in range(len(z_grid)):
    vals = []
    for ii in range(len(samp_top)):
        p = 0.0
        h_arr  = samp_top[ii, 0:N_LAYER - 1]
        vs_arr = samp_top[ii, N_LAYER - 1:]
        for ci in range(N_LAYER):
            p_end = (p + h_arr[ci]) if ci < N_LAYER - 1 else Z_MAX + 1
            if p <= z_grid[zi] < p_end:
                vals.append(vs_arr[ci]); break
            p = p_end if ci < N_LAYER - 1 else p
    vs_mediana[zi] = np.median(vals) if vals else np.nan

# Vector escalonado vs_z_mejor con la Vs del mejor modelo en función de la profundidad
idx_mejor = np.argmin(rmse_top)
h_mejor   = samp_top[idx_mejor, 0:N_LAYER - 1]
vs_mejor  = samp_top[idx_mejor, N_LAYER - 1:]
vs_z_mejor = np.zeros(len(z_grid))
p = 0.0
for ci in range(N_LAYER):
    p_end = (p + h_mejor[ci]) if ci < N_LAYER - 1 else Z_MAX + 1
    vs_z_mejor[(z_grid >= p) & (z_grid < p_end)] = vs_mejor[ci]
    p = p_end if ci < N_LAYER - 1 else p

# FIGURA
fig, axes = plt.subplots(1, 2, figsize=(16, 10),
                         gridspec_kw={"width_ratios": [2, 1]})
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#0d1117")

# Panel izquierdo
ax_h = axes[0]
dens_pos = densidad.copy()
dens_pos[dens_pos <= 0] = np.nan
im = ax_h.pcolormesh(
    vs_ctrs * 1000, z_grid * 1000, dens_pos,
    norm=mcolors.LogNorm(
        vmin=np.nanpercentile(dens_pos[np.isfinite(dens_pos)], 5),
        vmax=np.nanmax(dens_pos)),
    cmap="inferno", shading="auto",
)
ax_h.plot(vs_mediana * 1000, z_grid * 1000,
          color="#00e5ff", lw=2.5, label="Mediana Vs", zorder=5)
ax_h.plot(vs_z_mejor * 1000, z_grid * 1000,
          color="#76ff03", lw=2, ls="--",
          label=f"Mejor modelo (RMSE={rmse_top[idx_mejor]:.4f})", zorder=6)
p_km = 0.0
for ci in range(N_LAYER - 1):
    p_km += h_mejor[ci]
    if p_km * 1000 <= Z_MAX * 1000:
        ax_h.axhline(p_km * 1000, color="white", lw=0.7, ls=":", alpha=0.5)
cb = plt.colorbar(im, ax=ax_h, pad=0.02)
cb.set_label("Densidad de modelos (peso 1/RMSE)", color="white", fontsize=10)
plt.setp(cb.ax.get_yticklabels(), color="white")
ax_h.set_xlim(vs_min_plot * 1000, vs_max_plot * 1000)
ax_h.set_ylim(Z_MAX * 1000, 0)
ax_h.set_xlabel("Velocidad Onda S, Vs (m/s)", color="white", fontsize=12)
ax_h.set_ylabel("Profundidad (m)", color="white", fontsize=12)
ax_h.set_title(
    "Mapa de Calor Vs — Modelos BEL1D Top 10%\nSitio Comalapa · 5 capas geológicas",
    color="white", fontsize=13, fontweight="bold")
ax_h.tick_params(colors="white")
ax_h.spines[:].set_color("#444")
ax_h.legend(loc="lower right", facecolor="#1a1a2e",
            labelcolor="white", fontsize=9, framealpha=0.8)
ax_h.grid(True, color="#333", lw=0.3, alpha=0.4)

# Panel derecho
ax_d = axes[1]
norm_m  = mcolors.Normalize(vmin=rmse_top.min(),
                             vmax=rmse_top.max())
cmap_d  = plt.cm.plasma_r
sort_idx = np.argsort(rmse_top)[::-1]

for ii in sort_idx:
    dc = sampDC_top[ii]
    if np.any(np.isnan(dc)):
        continue
    ax_d.plot(dc * 1000, Frequency,
              color=cmap_d(norm_m(rmse_top[ii])), alpha=0.30, lw=0.7)

# Curva observada original
ax_d.scatter(vc_raw / 1.0, f_raw, color="#ff6e40", s=40, zorder=9,
             label="Puntos brutos (DBSCAN)", alpha=0.6)
# Curva filtrada
ax_d.plot(vc_obs, f_obs, "o-", color="#00e5ff", ms=7, lw=2.5, zorder=10,
          label="Curva observada filtrada")
# Banda ±1σ
ax_d.fill_betweenx(Frequency,
                   (Dataset - NoiseModel) * 1000,
                   (Dataset + NoiseModel) * 1000,
                   alpha=0.20, color="#00e5ff", label="±1σ ruido")

sm = ScalarMappable(cmap=cmap_d, norm=norm_m)
sm.set_array([])
cb2 = plt.colorbar(sm, ax=ax_d, pad=0.04)
cb2.set_label("RMSE relativo", color="white", fontsize=9)
plt.setp(cb2.ax.get_yticklabels(), color="white")
ax_d.set_xlabel("Velocidad de Fase Vc (m/s)", color="white", fontsize=11)
ax_d.set_ylabel("Frecuencia (Hz)", color="white", fontsize=11)
ax_d.set_title("Curvas de Dispersión\nModelos Sintéticos vs Observada",
               color="white", fontsize=12, fontweight="bold")
ax_d.tick_params(colors="white")
ax_d.spines[:].set_color("#444")
ax_d.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, loc="upper right")
ax_d.grid(True, color="#333", lw=0.4, alpha=0.6)

plt.tight_layout(pad=2.0)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapa_calor_comalapa.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"\n[LISTO] Figura guardada: {out_path}")

nombres = ["Suelo orgánico", "TBJ suelta", "TBJ compacta",
           "Tobas compactadas", "Basamento (∞)"]
print("\n" + "=" * 60)
print("  RESUMEN — MEJOR MODELO (BEL1D Top 10%)")
print("=" * 60)
print(f"  RMSE relativo: {rmse_top[idx_mejor]:.5f}")
for ci in range(N_LAYER):
    h_str = f"{h_mejor[ci]*1000:.1f} m" if ci < N_LAYER - 1 else "∞"
    print(f"  C{ci+1} {nombres[ci]:22s}  Vs={vs_mejor[ci]*1000:.0f} m/s  h={h_str}")

print("\n  Estadísticas de la Familia Top 10% (media ± std):")
for ci in range(N_LAYER - 1):
    h_samp = samp_top[:, ci] * 1000
    print(f"  C{ci+1} h:  {h_samp.mean():.1f} ± {h_samp.std():.1f} m")
for ci in range(N_LAYER):
    vs_samp = samp_top[:, N_LAYER - 1 + ci] * 1000
    print(f"  C{ci+1} Vs: {vs_samp.mean():.0f} ± {vs_samp.std():.0f} m/s")

p_m, travel = 0.0, 0.0
for ci in range(N_LAYER - 1):
    hi = h_mejor[ci] * 1000
    seg = min(p_m + hi, 30.0) - min(p_m, 30.0)
    if seg > 0: travel += seg / (vs_mejor[ci] * 1000)
    p_m += hi
remain = 30.0 - min(p_m, 30.0)
if remain > 0: travel += remain / (vs_mejor[-1] * 1000)
vs30 = 30.0 / travel if travel > 0 else vs_mejor[-1] * 1000
print(f"\n  Vs30 estimado: {vs30:.0f} m/s")
print("=" * 60)
