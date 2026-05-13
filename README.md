# Pipeline MASW — Inversión Monte Carlo · Sitio Comalapa

> Análisis multicanal de ondas superficiales (MASW) para caracterización
> sísmica del sitio de Comalapa, El Salvador. El pipeline extrae la curva
> de dispersión de ondas Rayleigh desde registros SEG-2, aplica clustering
> para filtrar ruido, y realiza inversión estocástica Monte Carlo con el
> método de matriz de transferencia estabilizado (Knopoff-Dunkin).

---

## Tabla de contenidos

1. [Fundamento teórico](#1-fundamento-teórico)
2. [Arquitectura del pipeline](#2-arquitectura-del-pipeline)
3. [Descripción de archivos](#3-descripción-de-archivos)
4. [Flujo paso a paso](#4-flujo-paso-a-paso)
5. [Funciones principales](#5-funciones-principales)
6. [Librerías científicas usadas](#6-librerías-científicas-usadas)
7. [Modelo geológico de Comalapa](#7-modelo-geológico-de-comalapa)
8. [Cómo ejecutar](#8-cómo-ejecutar)
9. [Salidas](#9-salidas)
10. [Referencias](#10-referencias)

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

### 1.3 Método de la Matriz de Transferencia (Haskell-Thomson)

Para calcular la curva de dispersión **sintética** de un modelo de capas se
usa el método de la **matriz de transferencia de Haskell** (Thomson 1950,
Haskell 1953). Cada capa finita `i` se describe por una matriz 4×4 `A_i` que
relaciona los vectores de esfuerzo-desplazamiento en la parte superior e
inferior de la capa.

Para `N` capas sobre un semiespacio infinito:

```
Sistema global:  J = E_inv · (A_N · A_{N-1} · ... · A_1)
Condición libre: det( J[:, 0:2] ) = 0   ← función secular
```

Los ceros de la función secular dan las velocidades de fase del modo
fundamental de Rayleigh para cada frecuencia.

### 1.4 Problema numérico: overflow exponencial

La matriz de capa `A_i` contiene términos `cosh(k·h·η)` y `sinh(k·h·η)`,
donde `η` es imaginario puro cuando `Vc < Vs`. En ese caso:

```
cosh(i·k·h·|η|) = cos(k·h·|η|)   → acotado ✓
```

Pero si `Vc ≈ Vs` o hay muchas capas a alta frecuencia, la acumulación
numérica en el producto de matrices 4×4 produce **overflow exponencial**
(valores ~10⁶–10¹²), generando mínimos espurios de la función secular a
velocidades erróneas (800–1400 m/s en lugar de 100–300 m/s).

**Solución:** normalizar la matriz acumulada por su norma de Frobenius después
de cada multiplicación de capa. Esto estabiliza el cálculo sin perder la
información de cambio de signo de la función secular.

```python
M = A_i @ M
nrm = np.linalg.norm(M)
if nrm > 0:
    M /= nrm   # la escala se elimina, el SIGNO se preserva
```

### 1.5 Detección del modo fundamental por cambio de signo

En lugar de buscar el mínimo de `|det(J)|` (sensible a artefactos), se evalúa
la parte real de `det(J)` en una grilla densa de velocidades y se localiza el
**primer cambio de signo** (menor Vc). Ese cruce corresponde al modo
fundamental de Rayleigh. La velocidad exacta se obtiene por interpolación
lineal:

```
Vc_modo = Vc[i] - sec[i] · (Vc[i+1] - Vc[i]) / (sec[i+1] - sec[i])
```

### 1.6 Inversión Monte Carlo

La inversión estocástica sigue el esquema de **búsqueda aleatoria con
función de costo** (Sambridge & Mosegaard 2002):

```
1. Generar N modelos aleatorios dentro de restricciones a priori
2. Para cada modelo → calcular curva de dispersión sintética Vc_syn(f)
3. Calcular misfit RMS relativo con la curva observada Vc_obs(f):

        misfit = sqrt( mean( ((Vc_obs - Vc_syn) / Vc_obs)^2 ) )

4. Seleccionar el p-percentil de mejores modelos
5. Visualizar la distribución de Vs como mapa de calor
```

El **mapa de calor Vs vs profundidad** (estilo GJI) muestra la densidad de
modelos aceptables ponderada por 1/misfit. Las zonas más brillantes indican
mayor concentración de modelos compatibles con los datos → mayor certeza
en ese rango de Vs.

---

## 2. Arquitectura del pipeline

```
datos_sg2/*.sg2
      │
      ▼
┌─────────────────┐
│  problema3.py   │  FFT 2D sobre registros SEG-2
│  procesar_datos │  Extrae picos de energía en espacio F-K
│  _fk()          │  Salida: índices (archivo, fila_f, col_k)
└────────┬────────┘
         │  picos_indices + parámetros físicos (fs, dx, filas, columnas)
         ▼
┌─────────────────┐
│  problema5.py   │  Convierte índices → (f [Hz], k [1/m])
│  ejecutar_       │  DBSCAN en coord. cilíndricas para filtrar ruido
│  clustering()   │  Selecciona clúster físicamente válido (Vc = f/k)
└────────┬────────┘
         │  curva_dispersion: {f, vc}  ← datos observados
         ▼
┌─────────────────┐
│    main.py      │  1. Define restricciones geológicas (5 capas)
│                 │  2. Genera 10,000 modelos aleatorios
│  Problema       │  3. Calcula curva sintética (Haskell-Thomson estable)
│  Directo +      │  4. Compara con curva observada (misfit RMS)
│  Inversión MC   │  5. Filtra 20% mejores modelos
└────────┬────────┘
         │
         ▼
  mapa_calor_comalapa.png
  (Vs vs profundidad + curvas de dispersión)
```

---

## 3. Descripción de archivos

| Archivo | Rol | Entrada | Salida |
|---|---|---|---|
| `problema3.py` | Procesamiento F-K | Archivos `.sg2` | Dict con picos y parámetros físicos |
| `problema5.py` | Clustering DBSCAN | Picos F-K (índices) | Curva de dispersión observada |
| `main.py` | Inversión Monte Carlo | Curva observada | Mapa de calor + resumen |
| `problema1.py` | Referencia: 1 capa + semiespacio | — | Figura de dispersión |
| `problema2.py` | Preprocesamiento auxiliar | — | — |
| `problema4.py` | Análisis auxiliar | — | — |
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

### Etapa 3 — `main.py`: Generación de modelos

```
Para cada una de las 5 capas:
  - Muestreo uniforme de espesor h en [h_min, h_max]
  - Muestreo uniforme de Vs en [vs_min, vs_max]
→ 10,000 modelos de 9 parámetros cada uno
   (4 espesores + 5 velocidades Vs)
```

### Etapa 4 — `main.py`: Problema directo

```
Para cada modelo (esp, vs, rho):
  Para cada frecuencia f en la curva observada:
    1. Definir grilla de velocidades Vc en (0.7·Vs1, 0.98·Vs_hs)
    2. Para cada Vc en la grilla:
       a. k = ω/Vc
       b. Para cada capa finita i:
          - Calcular ga = sqrt((Vc/Vp)²-1), gb = sqrt((Vc/Vs)²-1)
          - Construir matriz de Haskell 4×4
          - M = A_i @ M; normalizar M
       c. Aplicar condición de frontera del semiespacio → J (2×4)
       d. secular = Re(det(J[:,0:2]))
    3. Buscar primer cambio de signo en secular(Vc) → Vc_modo
    4. Registrar (f, Vc_modo)
```

### Etapa 5 — `main.py`: Misfit y selección

```
Para cada modelo con curva sintética (f_syn, Vc_syn):
  1. Interpolar Vc_syn en las frecuencias de f_obs
  2. misfit = sqrt(mean(((Vc_obs - Vc_syn) / Vc_obs)²))
  3. Guardar misfit

Seleccionar el 20% de modelos con menor misfit (umbral percentil 20)
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

### `main.py` — `secular_rayleigh(c, espesores, vs_arr, rhos, omega)`

**Qué hace:** Evalúa la función secular de Rayleigh en una velocidad de fase
`c` dada. Construye el producto de matrices de Haskell con normalización por
norma de Frobenius y aplica la condición de frontera del semiespacio.

**Por qué la normalización:** Cuando `Vc < Vs` (caso típico del modo
fundamental), los términos `ga` y `gb` son imaginarios puros, lo que hace que
`cosh(k·h·|ga|)` crezca exponencialmente con `k·h`. En modelos de 4–5 capas
a frecuencias de 30–50 Hz esto produce overflow. La normalización mantiene la
magnitud de `M` en O(1) mientras conserva su dirección (y por tanto el signo
del determinante).

**Retorna:** `float` — parte real del determinante de la submatriz 2×2.
Un cero de esta función es una velocidad de fase del modo fundamental.

---

### `main.py` — `dispersion_curve(espesores, vs_arr, rhos, f_vec)`

**Qué hace:** Para cada frecuencia en `f_vec`, evalúa `secular_rayleigh` en
una grilla de velocidades y detecta el **primer cambio de signo** (modo
fundamental). Usa interpolación lineal para precisión subgrilla.

**Grilla de búsqueda:** densa en el rango bajo `(0.7·Vs1, 2.5·Vs1)` y más
espaciada hasta `0.98·Vs_halfspace`. Esto garantiza resolución donde vive el
modo fundamental a alta frecuencia.

**Retorna:** `(f_out, vc_out)` — arrays con la curva de dispersión sintética.

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

### `sklearn.cluster.DBSCAN` — Clustering de densidad

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) agrupa
puntos cercanos sin requerir especificar el número de clústeres a priori.
Puntos aislados se etiquetan como ruido (`-1`).

**Por qué se usa aquí:** Los picos F-K de múltiples disparos forman nubes
en el espacio (f, k). Disparos "buenos" generan picos coherentes agrupados
(un clúster denso = la curva de dispersión real). Disparos ruidosos o con
artefactos producen puntos dispersos (ruido DBSCAN). El clustering permite
separar automáticamente la señal del ruido sin umbral manual.

**Parámetros críticos:**
- `eps=15`: radio de vecindad en el espacio cilíndrico normalizado
- `min_samples=3`: mínimo de puntos para formar un clúster

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

## 8. Cómo ejecutar

### Requisitos

```bash
# Desde el directorio del proyecto, con el venv activo:
pip install numpy scipy matplotlib obspy scikit-learn
```

### Ejecución

```bash
# Pipeline completo (≈ 20 min con 10,000 modelos)
python main.py

# Solo extracción F-K (con gráfica de control)
python problema3.py

# Solo clustering y curva de dispersión (requiere picos_f_k.npy)
python problema5.py
```

### Parámetros ajustables en `main.py`

| Variable | Ubicación | Descripción |
|---|---|---|
| `N_MODELOS` | línea ~76 | Número de modelos Monte Carlo (10,000) |
| `umbral_pct` | línea ~243 | Percentil de modelos aceptables (20%) |
| `CAPAS` | líneas 46–52 | Restricciones geológicas a priori |
| `RHO` | línea ~78 | Densidades por capa (kg/m³) |
| `NU` | línea ~79 | Razón de Poisson → ratio Vp/Vs |
| `Z_MAX` | línea ~259 | Profundidad máxima del mapa de calor (m) |

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

### Resumen en consola

```
==================================================
  RESUMEN — MEJOR MODELO
==================================================
  Misfit RMS: 0.08800
  C1 Suelo orgánico          Vs=XXX m/s  h=X.X m
  C2 TBJ suelta              Vs=XXX m/s  h=X.X m
  C3 TBJ compacta            Vs=XXX m/s  h=XX.X m
  C4 Tobas compactadas       Vs=XXX m/s  h=XX.X m
  C5 Basamento               Vs=XXX m/s  h=∞

  Vs30 estimado: XXX m/s
==================================================
```

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

---

*Pipeline desarrollado para el análisis sísmico del sitio de Comalapa,
El Salvador. Implementación Python con método de matriz de transferencia
estabilizado por normalización de Frobenius.*
