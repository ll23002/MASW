# Pipeline MASW — Inversión Monte Carlo · Sitio Comalapa

> Análisis multicanal de ondas superficiales (MASW) para caracterización
> sísmica. El pipeline extrae la curva
> de dispersión de ondas Rayleigh desde registros SEG-2, aplica clustering
> para filtrar ruido, y realiza inversión estocástica Monte Carlo con el
> método de matriz de transferencia estabilizado (Knopoff-Dunkin).

---

## 1. Fundamento teórico

### 1.1 Ondas de Rayleigh y dispersión

Las ondas de Rayleigh son ondas sísmicas superficiales que se propagan a lo
largo de la interfaz libre de la Tierra. En un semiespacio homogéneo viajan a
≈ 0.92 · Vs. En un medio estratificado exhiben **dispersión**: la velocidad
de fase `Vc(f)` varía con la frecuencia porque cada frecuencia "siente" una
profundidad diferente.

```
Frecuencia alta (λ corta) → penetra poco  → Vc ≈ Vs superficial
Frecuencia baja  (λ larga) → penetra más  → Vc → Vs del basamento
```

Esta relación `Vc(f)` es la **curva de dispersión** y es el dato central de
todo análisis MASW.

### 1.2 Transformada F-K (Frecuencia – Número de onda)

El registro sísmico multicanal se convierte al dominio frecuencia-número de
onda mediante la **FFT 2D**:

```
u(x, t)  →  FFT2D  →  U(k, f)
```

La energía se concentra a lo largo de la curva `f = Vc(f) · k`. Localizar
los máximos de energía para cada frecuencia equivale a leer la curva de
dispersión. El espaciado entre geófonos `dx` determina el número de onda de
Nyquist:

```
k_Nyquist = 1 / (2 · dx)   [1/m]
```

### 1.3 Forward Modeling (pysurf96)

Para calcular la curva de dispersión **sintética** de un modelo de capas se
usa el modelo directo **surf96** (Computer Programs in Seismology, Herrmann).
Se accede mediante el wrapper `pysurf96`, el cual computa las velocidades de
fase del modo fundamental de Rayleigh usando matrices de transferencia optimizadas.
Esta implementación es mucho más robusta y estable numéricamente que la clásica
matriz de Haskell-Thomson a altas frecuencias, evitando problemas de overflow
exponencial.

### 1.4 Bayesian Evidential Learning (BEL1D)

El pipeline original de Monte Carlo fue reemplazado por el framework
**Bayesian Evidential Learning 1D (BEL1D)** (Mreyen et al., 2023, GJI).
BEL1D es un método probabilístico que evita la inversión exhaustiva (MCMC)
mediante aprendizaje estadístico:

1. **PREBEL:** Genera un conjunto de modelos sintéticos a partir del
   *prior geológico* (capas de Comalapa) y computa sus respuestas forward.
2. **Reducción de Dimensionalidad:** Usa Análisis de Componentes Principales
   (PCA) para reducir la complejidad tanto del espacio de datos (curvas)
   como de los modelos.
3. **Correlación Canónica (CCA):** Encuentra un subespacio donde las
   respuestas sintéticas y los parámetros del modelo tienen correlación
   máxima.
4. **Estimación Bayesiana:** En este proyecto, utilizamos una variante directa:
   generamos miles de modelos de la distribución *prior* con el forward model
   optimizado y extraemos directamente el **Top 10%** de los que mejor se
   ajustan a los datos de campo, construyendo empíricamente la familia
   de perfiles probabilísticos del subsuelo.

El **mapa de calor Vs vs profundidad** (estilo GJI) muestra la densidad de
modelos aceptables ponderada por 1/misfit. Las zonas más brillantes indican
mayor concentración de modelos compatibles con los datos → mayor certeza
en ese rango de Vs.

---

## 3. Descripción de archivos

| Archivo | Rol | Entrada | Salida |
|---|---|---|---|
| `problema3.py` | Procesamiento F-K | Archivos `.sg2` | Dict con picos y parámetros físicos |
| `problema5.py` | Clustering DBSCAN | Picos F-K (índices) | Curva de dispersión observada |
| `main.py` | Inversión BEL1D | Curva observada | Mapa de calor + resumen |
| `_pyBEL1D_src/` | Librería pyBEL1D | Modelo y Priors | Espacio de muestreo |
| `datos_sg2/` | Registros sísmicos brutos | — | — |

---

## 4. Flujo paso a paso

### Etapa 1 — `problema3.py`: Extracción F-K

```
Para cada archivo SEG-2 (disparo):
  1. Leer con ObsPy → matriz (tiempo × canales)
  2. Aplicar FFT 2D → dominio (k, f)
  3. fftshift → centrar DC en (filas//2, columnas//2)
  4. Recortar: solo frecuencias positivas (0 → f_max Hz)
               solo números de onda positivos (0 → k_Nyquist)
  5. Buscar argmax de energía → (fila_f, col_k)
  6. Descartar col_k = 0 (artefacto DC)
  7. Guardar (índice_archivo, fila_f, col_k)

Al final:
  - Convertir fila_f → f [Hz]:  f = fila_f × (f_max / filas_150hz)
  - Convertir col_k  → k [1/m]: k = col_k × (k_Nyquist / n_k_positivo)
```

### Etapa 2 — `problema5.py`: Clustering y curva observada

```
1. Mapear índices matriciales a (f, k) en unidades físicas
2. Definir coordenadas para clustering:
   X = col_k, Y = fila_f, Z = índice_disparo
3. Convertir a coordenadas cilíndricas: (r, θ, Z)
4. DBSCAN sobre (r, θ, Z) → etiquetas de clúster
5. Seleccionar el clúster cuya mediana Vc = f/k esté en (50, 1500) m/s
6. Calcular Vc = f/k para cada punto del clúster válido
7. Promediar k por bin de frecuencia → un punto (f, Vc) por Hz
8. Ordenar ascendente → curva de dispersión observada final
```

### Etapa 3 — `main.py`: Pre-procesamiento de datos e Interpolación

```
Para adaptar el problema al framework de BEL1D:
  1. Filtrado físico: se eliminan puntos atípicos de la curva 
     (modos superiores) validando una dispersión decreciente monotónica.
  2. Interpolación: La curva resultante se interpola a una grilla uniforme
     de 20 puntos de frecuencia para asegurar suficientes dimensiones 
     estadísticas para el Análisis de Componentes Principales (PCA).
```

### Etapa 4 — `main.py`: BEL1D PREBEL y Modelo Directo (pysurf96)

```
1. Construcción de pyBEL1D.MODELSET.DCVs con las capas de Comalapa.
2. Ejecución de PREBEL paralelizada con Pathos:
   - Extrae 3000 modelos del espacio de variables aleatorias del prior.
   - Pysurf96 evalúa cada modelo generando su curva de dispersión Vc(f).
```

### Etapa 5 — `main.py`: Misfit y Selección Bayesiana Directa

```
Para cada modelo sintético computado por PREBEL:
  1. misfit = sqrt(mean(((Vc_obs - Vc_syn) / Vc_obs)²))
  2. Guardar misfit relativo de todos los 3000 modelos.

Se selecciona el 10% (Top 10%) de modelos que mejor ajusten la curva
observada empíricamente. Este subconjunto representa las zonas de alta
probabilidad posterior del modelo bayesiano.
Peso de cada modelo aceptable: w = 1 / (misfit + ε)
```

### Etapa 6 — `main.py`: Mapa de calor

```
Para cada modelo aceptable (peso w):
  - Construir perfil Vs(z) escalonado (un valor por capa)
  - Para cada profundidad z: acumular w en el bin (z, Vs) del mapa

Calcular mediana de Vs por profundidad
Calcular perfil del mejor modelo (misfit mínimo)
Graficar: pcolormesh(Vs_bins, z_grid, densidad, norm=LogNorm)
```

---

## 5. Funciones principales

### `problema3.py` — `procesar_datos_fk(ruta_archivos, dx, f_max)`

**Qué hace:** Lee todos los archivos SEG-2, aplica FFT 2D a cada disparo y
extrae el pico de energía en el dominio F-K positivo.

**Parámetros:**
- `ruta_archivos`: glob pattern hacia los `.sg2`
- `dx`: espaciado entre geófonos en metros (default 2.0)
- `f_max`: frecuencia máxima de análisis en Hz (default 150.0)

**Retorna:** diccionario con:
- `picos_indices`: array (N, 3) con `[idx_archivo, fila_f, col_k]`
- `frecuencias_hz`: frecuencias reales de cada pico (Hz)
- `k_values`: números de onda reales de cada pico (1/m)
- `fs`, `dx`, `filas_t`, `columnas_e`, `filas_150hz`: parámetros de grilla FFT

---

### `problema5.py` — `ejecutar_clustering(datos_crudos, f_max, dx, filas_150hz, columnas_e, eps, min_samples)`

**Qué hace:** Convierte índices matriciales a unidades físicas, aplica DBSCAN
en coordenadas cilíndricas, selecciona el clúster sísmicamente válido y
construye la curva de dispersión promedio `Vc(f)`.

**Parámetros:**
- `datos_crudos`: array (N, 3) de `procesar_datos_fk`
- `eps`, `min_samples`: hiperparámetros de DBSCAN

**Selección del clúster válido:** Se itera sobre todos los clústeres no-ruido
y se elige el primero cuya mediana de `Vc = f/k` esté en el rango sísmico
(50–1500 m/s).

**Retorna:** diccionario con:
- `curva_dispersion`: dict `{f: array Hz, vc: array m/s}`
- `frecuencias_hz`, `k_values`, `velocidades_fase`: datos crudos del clúster
- `etiquetas`, `datos_ml`: para visualización 3D

---

## 6. Librerías científicas usadas

### `numpy` — Álgebra lineal y aritmética vectorizada

Usada en prácticamente todo el código.

| Uso concreto | Función |
|---|---|
| FFT 2D sobre registros sísmicos | `np.fft.fft2` / `scipy.fft.fft2` |
| Centrar espectro (DC al centro) | `np.fft.fftshift` |
| Producto de matrices 4×4 de Haskell | `@` operator (matmul) |
| Norma de Frobenius para estabilización | `np.linalg.norm(M)` |
| Determinante de la submatriz secular | `np.linalg.det(J[:,0:2])` |
| Raíces cuadradas complejas | `np.lib.scimath.sqrt` |
| Muestreo aleatorio uniforme | `np.random.default_rng().uniform` |
| Percentil de misfits | `np.nanpercentile` |
| Cambios de signo (detección de ceros) | `np.diff(np.sign(secular))` |

### `scipy.fft` — Transformada rápida de Fourier

`fft2(matriz_sismica)` convierte la señal del dominio tiempo-espacio
`(t, x)` al dominio frecuencia-número de onda `(f, k)`. Es más rápida que
`numpy.fft.fft2` para matrices grandes gracias a algoritmos de factorización
de primos (Cooley-Tukey).

`fftshift` reordena el resultado para que la componente DC (frecuencia cero,
número de onda cero) quede en el centro del array en lugar de en la esquina.

### `scipy.interpolate.interp1d` — Interpolación 1D

Se usa en dos lugares:

1. **Evaluar la curva sintética en las frecuencias observadas:** cada modelo
   tiene su propia grilla de frecuencias; `interp1d` permite evaluarla en el
   mismo vector `f_obs` para que el misfit sea comparable.

2. **Extrapolación de la curva:** `fill_value='extrapolate'` o valores de
   borde para no penalizar modelos que cubren un rango de frecuencias
   ligeramente diferente.

### `obspy` — Lectura de datos sísmicos SEG-2

`obspy.read(archivo, format="SEG2")` parsea el formato binario SEG-2 (estándar
de sismógrafos de exploración superficial) y retorna un objeto `Stream` con
trazas sísmicas. Cada traza contiene el vector de muestras y los metadatos
del encabezado (tasa de muestreo, número de canales, etc.).

### `pysurf96` — Forward Model optimizado

Implementación vectorizada y compilada del clásico programa `surf96`
(Hermann). Genera las curvas de dispersión teóricas del modo fundamental 
usando rutinas en FORTRAN y C, completamente libres de overflow a altas 
frecuencias. Funciona con matrices de transferencia.

### `pyBEL1D` y `scikit-learn` — Inversión Bayesiana

El núcleo de la inversión se realiza con pyBEL1D, la cual se apoya 
fuertemente en PCA y Clustering de `scikit-learn` para aprender y 
proyectar variables geofísicas sin el costo computacional masivo del 
tradicional Monte Carlo. En nuestro caso aplicamos una aproximación a la
posterior truncando los resultados directos de la etapa inicial de aprendizaje.

### `matplotlib` — Visualización científica

| Componente | Uso |
|---|---|
| `plt.pcolormesh` | Mapa de calor Vs vs profundidad |
| `mcolors.LogNorm` | Escala logarítmica para la densidad (rangos muy amplios) |
| `ScalarMappable` | Barra de color para el misfit en el panel de dispersión |
| `plt.colorbar` | Leyendas de escala en ambos paneles |
| `ax.plot` | Perfil de mediana y mejor modelo superpuestos |
| `fig.savefig` | Exportar a PNG de alta resolución (180 dpi) |

---

## 7. Modelo geológico de Comalapa

El sitio de San Pedro Nonualco / Comalapa (El Salvador) presenta la siguiente
estratigrafía típica de la zona volcánica:

| Capa | Descripción geológica | Espesor (m) | Vs (m/s) |
|:---:|---|:---:|:---:|
| 1 | Suelo superficial orgánico / coluvión | 0.5 – 2 | 100 – 200 |
| 2 | Tierra Blanca Joven (TBJ) suelta | 2 – 8 | 150 – 300 |
| 3 | TBJ compacta / depósitos piroclásticos | 5 – 15 | 300 – 500 |
| 4 | Tobas muy compactadas | 10 – 20 | 500 – 800 |
| 5 | Basamento rocoso / Formación Bálsamo (∞) | — | > 800 |

La **Tierra Blanca Joven** es el depósito de tefra de la erupción del Ilopango
(~431 d.C.) que domina la estratigrafía somera de gran parte de El Salvador
central. Su alta variabilidad de compactación justifica los amplios rangos de
Vs de las capas 2 y 3.

**Vs30** (velocidad media equivalente de los 30 m superiores): el resultado del
mejor modelo obtenido es ≈ 310–420 m/s, clasificación sísmica **Clase C o D**
según NEHRP/ASCE 7.

---


### Parámetros ajustables en `main.py`

| Variable | Ubicación | Descripción |
|---|---|---|
| `N_MODELS` | línea ~143 | Número de modelos a evaluar en BEL1D (3000) |
| `p_threshold` | línea ~178 | Percentil top bayesiano aceptable (Top 10%) |
| `prior` | línea ~117 | Restricciones geológicas a priori (km y km/s) |
| `RHO_FIXED` | línea ~126 | Densidades fijadas por capa (g/cm³) |
| `VP_FIXED` | línea ~124 | Vp derivado por Poisson de las medias de Vs |
| `Z_MAX` | línea ~187 | Profundidad máxima del mapa de calor (m) |

### Parámetros ajustables en `problema5.py`

| Parámetro | Descripción |
|---|---|
| `eps=15` | Radio DBSCAN en espacio cilíndrico (aumentar si pocos clústeres) |
| `min_samples=3` | Mínimo de disparos coherentes para un clúster |

---

## 9. Salidas

### `mapa_calor_comalapa.png`

Figura de dos paneles:

**Panel izquierdo — Mapa de calor Vs vs Profundidad:**
- Eje X: Velocidad Vs (m/s), escalado al rango real de los modelos aceptables
- Eje Y: Profundidad (m), creciente hacia abajo
- Color: densidad de modelos ponderada por `1/misfit` (escala logarítmica)
- Línea cyan: mediana de Vs por profundidad (estimación central)
- Línea verde punteada: mejor modelo individual (misfit mínimo)
- Líneas blancas punteadas: interfaces del mejor modelo

**Panel derecho — Curvas de Dispersión:**
- Cada línea: curva sintética de un modelo aceptable, coloreada por misfit
- Línea cyan con puntos: curva observada extraída de los datos reales
- Eje X auto-escalado al rango de datos (no fijo en 0–1500 m/s)


---

## 10. Referencias

- **Thomson, W.T. (1950).** Transmission of elastic waves through a stratified
  solid medium. *J. Appl. Phys.*, 21, 89–93.

- **Haskell, N.A. (1953).** The dispersion of surface waves on multilayered
  media. *Bull. Seismol. Soc. Am.*, 43(1), 17–34.

- **Knopoff, L. (1964).** A matrix method for elastic wave problems.
  *Bull. Seismol. Soc. Am.*, 54(1), 431–438.

- **Dunkin, J.W. (1965).** Computation of modal solutions in wavenumber and
  their use in the reduction of the matrix methods.
  *Bull. Seismol. Soc. Am.*, 55(2), 335–358.

- **Park, C.B., Miller, R.D., & Xia, J. (1999).** Multichannel analysis of
  surface waves. *Geophysics*, 64(3), 800–808.

- **Schwab, F. & Knopoff, L. (1972).** Fast surface wave and free mode
  computations. *Methods in Computational Physics*, 11, 87–180.

- **Sambridge, M. & Mosegaard, K. (2002).** Monte Carlo methods in geophysical
  inverse problems. *Rev. Geophys.*, 40(3), 1009.

- **Garofalo, F. et al. (2016).** InterPACIFIC project: Comparison of
  invasive and non-invasive methods for seismic site characterization.
  *Soil Dyn. Earthq. Eng.*, 82, 222–240.

- **Boiero, D. & Socco, L.V. (2025).** *[Artículo de referencia GJI 244(2)]*
  Inversión conjunta de ondas superficiales y ondas de cuerpo para
  caracterización sísmica de sitios. *Geophys. J. Int.*, 244(2), ggaf498.


