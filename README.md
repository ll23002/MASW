## Instrucciones de Ejecución

### 1. Construir la Imagen de Docker

```bash
docker build -t masw-inversion .
```

### 2. Ejecutar el Contenedor

```bash
docker run --rm -v $(pwd):/app masw-inversion
```