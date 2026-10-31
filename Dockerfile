# Imagen de Airflow 3 con las dependencias del proyecto. Debe coincidir con la versión del docker-compose.yaml oficial.
ARG AIRFLOW_VERSION=3.3.2
FROM apache/airflow:${AIRFLOW_VERSION}

USER root
# LightGBM necesita la librería OpenMP del sistema
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt
