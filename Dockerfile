FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema y compiladores necesarios para compilar CPS y pysurf96
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libx11-dev \
    libncurses-dev \
    git \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip e instalar dependencias básicas de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    obspy \
    pathos \
    scikit-learn \
    dill

# Instalar pysurf96
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/hadrienmichel/pysurf96

COPY CPS /opt/CPS

RUN cd /opt/CPS/PROGRAMS.330 && \
    ./Setup LINUX6440 && \
    ./C

ENV CPS_BIN=/opt/CPS/PROGRAMS.330/bin

WORKDIR /app

COPY . /app

CMD ["python", "main.py"]