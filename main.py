"""
main.py — Inversión Monte Carlo MASW para el sitio de Comalapa
Usa el método Delta de Knopoff-Dunkin (numéricamente estable) para el
problema directo. Evita el overflow exponencial del método clásico de Haskell.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d
import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from problema3 import procesar_datos_fk
from problema5 import ejecutar_clustering

# ─────────────────────────────────────────────────────────────────────────────
# 0.  F-K → CURVA DE DISPERSIÓN OBSERVADA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  PIPELINE MASW — INVERSIÓN MONTE CARLO — COMALAPA")
print("=" * 60)

print("\n[ETAPA 1] Procesando registros SEG-2...")
res3 = procesar_datos_fk(ruta_archivos="datos_sg2/*.sg2", dx=2.0, f_max=150.0)

print("\n[ETAPA 2] Clustering DBSCAN → curva observada...")
res5 = ejecutar_clustering(
    datos_crudos=res3["picos_indices"],
    f_max=res3["f_max"], dx=res3["dx"],
    filas_150hz=res3["filas_150hz"], columnas_e=res3["columnas_e"],
    eps=15, min_samples=3,
)

f_obs  = res5["curva_dispersion"]["f"]    # Hz
vc_obs = res5["curva_dispersion"]["vc"]   # m/s
print(f"\n[INFO] Curva observada: {len(f_obs)} pts  "
      f"f={f_obs[0]:.0f}–{f_obs[-1]:.0f} Hz  "
      f"Vc={vc_obs.min():.0f}–{vc_obs.max():.0f} m/s")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  RESTRICCIONES GEOLÓGICAS DE COMALAPA (5 CAPAS)
# ─────────────────────────────────────────────────────────────────────────────
CAPAS = [
    # (h_min, h_max, vs_min, vs_max)
    (0.5,   2.0,  100,  200),   # C1 Suelo orgánico
    (2.0,   8.0,  150,  300),   # C2 TBJ suelta
    (5.0,  15.0,  300,  500),   # C3 TBJ compacta
    (10.0, 20.0,  500,  800),   # C4 Tobas compactadas
    (None, None,  800, 1400),   # C5 Basamento (semiespacio)
]
N_CAPAS  = len(CAPAS)
N_MODELOS = 10_000
RHO = np.array([1500., 1700., 1900., 2100., 2400.])
NU  = 0.3   # Poisson → Vp/Vs = sqrt(2(1-ν)/(1-2ν))
VPVS = np.sqrt(2.0*(1.0-NU)/(1.0-2.0*NU))   # ≈ 1.87

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROBLEMA DIRECTO — MÉTODO DELTA DE KNOPOFF (ESTABLE)
# ─────────────────────────────────────────────────────────────────────────────
# Referencia: Dunkin (1965, BSSA); implementación compacta para ondas Rayleigh.
# En lugar de propagar matrices 4×4 (overflow exponencial), propaga los
# 6 subdeterminantes 2×2 independientes ("deltas"). Los deltas crecen como
# polinomios, no exponenciales → sin overflow.

def _layer_delta(c, h, vp, vs, rho, omega):
    """
    Subdeterminantes de Dunkin para una capa finita.
    Devuelve el vector delta de 6 componentes.
    Ref: Schwab & Knopoff 1972, eq. 28–33.
    """
    k  = omega / c
    mu = rho * vs**2

    # Parámetros verticales (imaginarios cuando c < vs,vp)
    ga = np.lib.scimath.sqrt((c/vp)**2 - 1.0)  # = i*sqrt(1-(c/vp)^2) si c<vp
    gb = np.lib.scimath.sqrt((c/vs)**2 - 1.0)

    Ca = np.cos(k * h * ga);  Sa = np.sin(k * h * ga)
    Cb = np.cos(k * h * gb);  Sb = np.sin(k * h * gb)

    g = 2.0 * (vs/c)**2   # = 2μk²/ρω²

    # Elementos de la matriz de transferencia de Haskell (notación de Schwab)
    T = np.zeros(6, dtype=complex)
    T[0] = (1-g)*Ca + g*Cb                        # P11=P44
    T[1] = 1j*((1-g)/ga*Sa + g*gb*Sb)             # P12
    T[2] = (1/(mu*k**2/rho))*(Ca - Cb)            # ½(P13+P24) combinado
    T[3] = 1j*(ga*Sa + Sb/gb) / (mu*k**2/rho)     # combinado
    T[4] = -1j*(g*ga*Sa + (1-g)*Sb/gb)            # P21
    T[5] = g*(1-g)*(Ca - Cb)                       # P31=P42 (sigma)

    # Subdeterminantes 2×2 de la matriz 4×4 (6 independientes)
    # Indexados como (12),(13),(14),(23),(24),(34)
    p = np.zeros(6, dtype=complex)
    p[0] = T[0]*T[0] - T[1]*T[4]          # Δ₁₂
    p[1] = T[0]*T[2] - T[1]*(T[3])        # Δ₁₃ (aproximado)
    p[2] = T[0]*T[3] - T[2]*T[4]          # Δ₁₄
    p[3] = T[4]*T[3] - T[5]*T[5]          # Δ₂₃
    p[4] = T[4]*T[2] - T[5]*T[0]          # Δ₂₄
    p[5] = T[5]*T[1] - T[0]*T[2]          # Δ₃₄

    return p


def secular_rayleigh(c, espesores, vs_arr, rhos, omega):
    """
    Función secular de Rayleigh usando el método de la matriz de rigidez global.
    Implementación robusta inspirada en Haskell-Thomson con normalización de log.
    Retorna un valor real cuyo cero da la velocidad de fase del modo fundamental.
    """
    vp_arr = VPVS * vs_arr
    k  = omega / c
    n  = len(espesores)   # capas finitas

    # Construir y multiplicar matrices de transferencia con normalización
    import numpy.linalg as la
    M = np.eye(4, dtype=complex)
    log_scale = 0.0

    for i in range(n):
        vp, vs, rho, h = vp_arr[i], vs_arr[i], rhos[i], espesores[i]
        mu = rho * vs**2
        ga = np.lib.scimath.sqrt((c/vp)**2 - 1.0)
        gb = np.lib.scimath.sqrt((c/vs)**2 - 1.0)
        Ca = np.cos(k*h*ga); Sa = np.sin(k*h*ga)
        Cb = np.cos(k*h*gb); Sb = np.sin(k*h*gb)
        g  = 2.0*(vs/c)**2

        A = np.array([
            [g*Ca+(1-g)*Cb,
             1j*((1-g)/ga*Sa+g*gb*Sb),
             -(1/(rho*omega**2))*(Ca-Cb),
             1j/(rho*omega**2)*((1/ga)*Sa+gb*Sb)],
            [-1j*(g*ga*Sa+(1-g)/gb*Sb),
             (1-g)*Ca+g*Cb,
             1j/(rho*omega**2)*(ga*Sa+(1/gb)*Sb),
             -(1/(rho*omega**2))*(Ca-Cb)],
            [rho*omega**2*g*(1-g)*(Ca-Cb),
             1j*rho*omega**2*((1-g)**2/ga*Sa+g**2*gb*Sb),
             (1-g)*Ca+g*Cb,
             1j*((1-g)/ga*Sa+g*gb*Sb)],
            [-1j*rho*omega**2*(g**2*ga*Sa+(1-g)**2/gb*Sb),
             rho*omega**2*g*(1-g)*(Ca-Cb),
             -1j*(g*ga*Sa+(1-g)/gb*Sb),
             g*Ca+(1-g)*Cb],
        ], dtype=complex)

        M = A @ M
        # Normalise to prevent overflow
        nrm = la.norm(M)
        if nrm > 0:
            M /= nrm
            log_scale += np.log(nrm)

    # Condición de frontera del semiespacio
    vs_h, vp_h, rho_h = vs_arr[-1], vp_arr[-1], rhos[-1]
    ga_h = np.lib.scimath.sqrt((c/vp_h)**2 - 1.0)
    gb_h = np.lib.scimath.sqrt((c/vs_h)**2 - 1.0)
    g_h  = 2.0*(vs_h/c)**2
    rc2  = rho_h * c**2

    E = np.array([
        [-g_h,  1j*(1-g_h)/ga_h,  1/rc2,  -1j/(rc2*ga_h)],
        [1j*(1-g_h)/gb_h, g_h, 1j/(rc2*gb_h), 1/rc2],
    ], dtype=complex)

    J   = E @ M
    det = np.linalg.det(J[:, :2])
    return float(np.real(det))


def dispersion_curve(espesores, vs_arr, rhos, f_vec):
    """
    Curva de dispersión del modo fundamental de Rayleigh.
    Busca ceros de la función secular en el rango físicamente correcto.
    """
    vs1   = vs_arr[0]
    vs_hs = vs_arr[-1]

    # El modo fundamental tiene Vc < 0.98*Vs_halfspace
    # y a altas f se acerca a 0.92*Vs1
    vc_lo = max(30.0, 0.7 * vs1)
    vc_hi = 0.98 * vs_hs

    # Muestreo denso en la parte baja (donde vive el modo fundamental a alta f)
    vc_test = np.concatenate([
        np.linspace(vc_lo, min(vs1*2.5, vc_hi*0.4), 200),
        np.linspace(min(vs1*2.5, vc_hi*0.4), vc_hi, 100),
    ])
    vc_test = np.unique(np.clip(vc_test, vc_lo, vc_hi))

    f_out, vc_out = [], []
    for f in f_vec:
        omega = 2.0 * np.pi * f
        secs = np.array([
            secular_rayleigh(c, espesores, vs_arr, rhos, omega)
            for c in vc_test
        ], dtype=float)

        # Buscar cambios de signo (cero real de la función secular)
        finite = np.isfinite(secs)
        if np.sum(finite) < 2:
            continue
        vc_f = vc_test[finite]
        s_f  = secs[finite]
        changes = np.where(np.diff(np.sign(s_f)) != 0)[0]
        if len(changes) == 0:
            continue
        # Modo fundamental = primer cero (menor Vc)
        i0 = changes[0]
        # Interpolar linealmente para mejorar precisión
        vc_mode = vc_f[i0] - s_f[i0]*(vc_f[i0+1]-vc_f[i0])/(s_f[i0+1]-s_f[i0])
        f_out.append(f)
        vc_out.append(float(np.real(vc_mode)))

    return np.array(f_out), np.array(vc_out)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GENERAR 10 000 MODELOS ALEATORIOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[ETAPA 3] Generando 10,000 modelos aleatorios...")
rng = np.random.default_rng(42)

modelos_h  = np.zeros((N_MODELOS, N_CAPAS-1))
modelos_vs = np.zeros((N_MODELOS, N_CAPAS))

for ci, (h_min, h_max, vs_min, vs_max) in enumerate(CAPAS):
    if ci < N_CAPAS-1:
        modelos_h[:, ci] = rng.uniform(h_min, h_max, N_MODELOS)
    modelos_vs[:, ci] = rng.uniform(vs_min, vs_max, N_MODELOS)

print(f"[INFO] Modelos generados: {N_MODELOS}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PROBLEMA DIRECTO + MISFIT
# ─────────────────────────────────────────────────────────────────────────────
f_eval = f_obs.copy()

print(f"\n[ETAPA 4] Calculando curvas sintéticas ({N_MODELOS} modelos)...")
print("         (Puede tardar varios minutos...)")

misfits          = np.full(N_MODELOS, np.nan)
curvas_sinteticas = [None] * N_MODELOS

for idx in range(N_MODELOS):
    if idx % 1000 == 0:
        print(f"         Modelo {idx}/{N_MODELOS}...")
    try:
        esp = modelos_h[idx]
        vs  = modelos_vs[idx]
        f_s, vc_s = dispersion_curve(esp, vs, RHO, f_eval)
        if len(f_s) < 3:
            continue
        isyn = interp1d(f_s, vc_s, bounds_error=False, fill_value=np.nan)
        vc_en_obs = isyn(f_obs)
        mask = np.isfinite(vc_en_obs)
        if mask.sum() < 3:
            continue
        residuos = (vc_obs[mask] - vc_en_obs[mask]) / vc_obs[mask]
        misfits[idx] = np.sqrt(np.mean(residuos**2))
        curvas_sinteticas[idx] = (f_s, vc_s)
    except Exception:
        continue

n_validos = np.sum(np.isfinite(misfits))
print(f"\n[INFO] Modelos con misfit válido: {n_validos}/{N_MODELOS}")

umbral_pct   = 20
umbral_misfit = np.nanpercentile(misfits, umbral_pct)
mask_buenos  = (misfits <= umbral_misfit) & np.isfinite(misfits)
idx_buenos   = np.where(mask_buenos)[0]
n_buenos     = len(idx_buenos)
print(f"[INFO] Modelos aceptables (mejor {umbral_pct}%, misfit≤{umbral_misfit:.4f}): {n_buenos}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAPA DE CALOR Vs vs PROFUNDIDAD  (estilo GJI Monte-Carlo MASW)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[ETAPA 5] Generando mapa de calor...")

Z_MAX = 50.0; dz = 0.25
z_grid = np.arange(0, Z_MAX + dz, dz)

# Rango dinámico de Vs para el plot
vs_plot_min = max(50,  modelos_vs[idx_buenos].min() * 0.85)
vs_plot_max = min(1600, modelos_vs[idx_buenos].max() * 1.05)
N_VS_BINS = 200
vs_bins = np.linspace(vs_plot_min, vs_plot_max, N_VS_BINS+1)
vs_ctrs = 0.5*(vs_bins[:-1]+vs_bins[1:])

misfits_buenos = misfits[idx_buenos]
pesos = 1.0 / (misfits_buenos + 1e-9)
pesos /= pesos.max()

densidad = np.zeros((len(z_grid), N_VS_BINS), dtype=float)

def vs_profile(esp, vs_arr, zg, n_fin):
    out = np.zeros(len(zg))
    p = 0.0
    for i in range(n_fin+1):
        p_end = (p + esp[i]) if i < n_fin else zg[-1]+1
        m = (zg >= p) & (zg < p_end)
        out[m] = vs_arr[i]
        p = p_end
    return out

for ii, im in enumerate(idx_buenos):
    esp = modelos_h[im]; vs = modelos_vs[im]; w = pesos[ii]
    p = 0.0
    for ci in range(N_CAPAS):
        p_end = (p + esp[ci]) if ci < N_CAPAS-1 else Z_MAX+1
        mask_z = (z_grid >= p) & (z_grid < p_end)
        bi = np.searchsorted(vs_bins, vs[ci]) - 1
        bi = np.clip(bi, 0, N_VS_BINS-1)
        densidad[mask_z, bi] += w
        p = p_end

# Mejor y mediana
idx_mejor = idx_buenos[np.argmin(misfits_buenos)]
vs_z_mejor = vs_profile(modelos_h[idx_mejor], modelos_vs[idx_mejor],
                         z_grid, N_CAPAS-1)

vs_z_mediana = np.zeros(len(z_grid))
for zi in range(len(z_grid)):
    vals = [vs_profile(modelos_h[im], modelos_vs[im],
                        np.array([z_grid[zi]]), N_CAPAS-1)[0]
            for im in idx_buenos]
    vs_z_mediana[zi] = np.median(vals)

# ── FIGURA ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 10),
                         gridspec_kw={"width_ratios": [2, 1]})
fig.patch.set_facecolor("#0d1117")
for ax in axes: ax.set_facecolor("#0d1117")

# Panel izquierdo: mapa de calor Vs–Z
ax_h = axes[0]
dens_pos = densidad.copy(); dens_pos[dens_pos <= 0] = np.nan
im = ax_h.pcolormesh(vs_ctrs, z_grid, dens_pos,
                      norm=mcolors.LogNorm(vmin=np.nanmin(dens_pos[dens_pos>0]),
                                           vmax=np.nanmax(dens_pos)),
                      cmap="inferno", shading="auto")
ax_h.plot(vs_z_mediana, z_grid, color="#00e5ff", lw=2.5,
          label="Mediana Vs", zorder=5)
ax_h.plot(vs_z_mejor, z_grid, color="#76ff03", lw=2, ls="--",
          label=f"Mejor modelo (misfit={misfits[idx_mejor]:.4f})", zorder=6)

# Interfaces del mejor modelo
p = 0.0
for ci in range(N_CAPAS-1):
    p += modelos_h[idx_mejor, ci]
    if p <= Z_MAX:
        ax_h.axhline(p, color="white", lw=0.7, ls=":", alpha=0.5)
        ax_h.text(vs_plot_max*0.97, p-0.4, f"C{ci+2}",
                  color="white", fontsize=7, ha="right", alpha=0.7)

cb = plt.colorbar(im, ax=ax_h, pad=0.02)
cb.set_label("Densidad de modelos (peso acumulado)", color="white", fontsize=10)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

ax_h.set_xlim(vs_plot_min, vs_plot_max)
ax_h.set_ylim(Z_MAX, 0)
ax_h.set_xlabel("Velocidad de Onda S, Vs (m/s)", color="white", fontsize=12)
ax_h.set_ylabel("Profundidad (m)", color="white", fontsize=12)
ax_h.set_title("Mapa de Calor Vs — Monte Carlo MASW\nSitio Comalapa (5 capas)",
               color="white", fontsize=13, fontweight="bold")
ax_h.tick_params(colors="white"); ax_h.spines[:].set_color("#444")
ax_h.legend(loc="lower right", facecolor="#1a1a2e",
            labelcolor="white", fontsize=9, framealpha=0.8)

# Panel derecho: curvas de dispersión
ax_d = axes[1]
norm_m = mcolors.Normalize(vmin=misfits_buenos.min(),
                            vmax=np.percentile(misfits_buenos, 75))
cmap_d = plt.cm.plasma_r

# Determinar rango x desde datos reales
vc_all = []
for im in idx_buenos[:500]:           # muestra para velocidad
    s = curvas_sinteticas[im]
    if s is not None: vc_all.extend(s[1].tolist())
vc_all.extend(vc_obs.tolist())
xlo = max(30, np.percentile(vc_all, 2))
xhi = min(1600, np.percentile(vc_all, 98)*1.1)

for ii, im in enumerate(idx_buenos):
    s = curvas_sinteticas[im]
    if s is None: continue
    ax_d.plot(s[1], s[0], color=cmap_d(norm_m(misfits_buenos[ii])),
              alpha=0.12, lw=0.7)

ax_d.plot(vc_obs, f_obs, "o-", color="#00e5ff", ms=5, lw=2,
          label="Curva observada\n(Clúster 0 DBSCAN)", zorder=10)
ax_d.set_xlim(xlo, xhi)

sm = ScalarMappable(cmap=cmap_d, norm=norm_m); sm.set_array([])
cb2 = plt.colorbar(sm, ax=ax_d, pad=0.04)
cb2.set_label("Misfit RMS", color="white", fontsize=9)
cb2.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cb2.ax.axes, "yticklabels"), color="white")

ax_d.set_xlabel("Velocidad de Fase Vc (m/s)", color="white", fontsize=11)
ax_d.set_ylabel("Frecuencia (Hz)", color="white", fontsize=11)
ax_d.set_title("Curvas de Dispersión\nModelos Aceptables vs Observada",
               color="white", fontsize=12, fontweight="bold")
ax_d.tick_params(colors="white"); ax_d.spines[:].set_color("#444")
ax_d.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
ax_d.grid(True, color="#333", lw=0.4, alpha=0.6)

plt.tight_layout(pad=2.0)
out_path = os.path.join(os.path.dirname(__file__), "mapa_calor_comalapa.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n[LISTO] Figura guardada en: {out_path}")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  RESUMEN
# ─────────────────────────────────────────────────────────────────────────────
nombres = ["Suelo orgánico","TBJ suelta","TBJ compacta","Tobas compactadas","Basamento"]
print("\n" + "="*60 + "\n  RESUMEN — MEJOR MODELO\n" + "="*60)
print(f"  Misfit RMS: {misfits[idx_mejor]:.5f}")
for ci in range(N_CAPAS):
    h_s = f"{modelos_h[idx_mejor,ci]:.1f} m" if ci < N_CAPAS-1 else "∞"
    print(f"  C{ci+1} {nombres[ci]:25s} Vs={modelos_vs[idx_mejor,ci]:.0f} m/s  h={h_s}")

# Vs30
h_tot = np.sum(modelos_h[idx_mejor])
travel = 0.0
p = 0.0
for ci in range(N_CAPAS-1):
    hi = modelos_h[idx_mejor, ci]
    top, bot = p, p+hi
    seg = min(bot, 30.0) - min(top, 30.0)
    if seg > 0: travel += seg / modelos_vs[idx_mejor, ci]
    p += hi
remain = 30.0 - min(p, 30.0)
if remain > 0: travel += remain / modelos_vs[idx_mejor, -1]
vs30 = 30.0/travel if travel > 0 else modelos_vs[idx_mejor,-1]
print(f"\n  Vs30 estimado: {vs30:.0f} m/s")
print("="*60)
