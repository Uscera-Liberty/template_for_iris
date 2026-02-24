# Базовый образ Airflow
# Базовый образ Airflow
FROM apache/airflow:2.9.3

# Ставим пакеты сразу от airflow пользователя
USER airflow

# Обновляем pip и ставим зависимости
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --timeout 120 --retries 8 \
       mlflow==2.15.1 \
       boto3 \
       apache-airflow-providers-docker

# Создаём рабочую директорию для проекта
WORKDIR /opt/iris

# Копируем локальные скрипты и данные
COPY --chown=airflow:airflow ./scripts ./scripts
COPY --chown=airflow:airflow ./data ./data

