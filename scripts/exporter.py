import os
import time
import requests
from prometheus_client import start_http_server, Gauge, Counter
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
AIRFLOW_URI = os.getenv('AIRFLOW_URI', 'http://localhost:8080')
AIRFLOW_USER = os.getenv('AIRFLOW_USER', 'airflow')
AIRFLOW_PASS = os.getenv('AIRFLOW_PASS', 'airflow')
MINIO_URI = os.getenv('MINIO_URI', 'http://localhost:9000')
MINIO_USER = os.getenv('MINIO_USER', 'mlops')
MINIO_PASS = os.getenv('MINIO_PASS', 'quZK9X8PfXfrlpi')
SCRAPE_INTERVAL = int(os.getenv('SCRAPE_INTERVAL', '30'))


# MLFLOW METRICS
mlflow_runs_total = Gauge(
    'mlflow_runs_total',
    'Total number of runs per experiment',
    ['experiment_name', 'status']
)
mlflow_registered_models_total = Gauge(
    'mlflow_registered_models_total',
    'Total number of registered models'
)
mlflow_model_versions_total = Gauge(
    'mlflow_model_versions_total',
    'Total number of model versions per model and stage',
    ['model_name', 'stage']
)
mlflow_experiment_total = Gauge(
    'mlflow_experiments_total',
    'Total number of experiments'
)
mlflow_latest_run_duration = Gauge(
    'mlflow_latest_run_duration_seconds',
    'Duration of latest run per experiment',
    ['experiment_name']
)
mlflow_latest_accuracy = Gauge(
    'mlflow_latest_run_accuracy',
    'Accuracy of latest run per experiment',
    ['experiment_name']
)
#AIRFLOW_METRICS
airflow_dag_status = Gauge(
    'airflow_dag_run_status',
    'DAG run status counts',
    ['dag_id', 'status']
)
airflow_task_duration = Gauge(
    'airflow_task_duration_seconds',
    'Task duration in seconds',
    ['dag_id', 'task_id']
)
airflow_active_dags = Gauge(
    'airflow_active_dags_total',
    'Number of active DAGs'
)
airflow_failed_tasks = Gauge(
    'airflow_failed_tasks_total',
    'Number of failed tasks in last 24h',
    ['dag_id']
)
#MINIO
minio_buckets_total = Gauge(
    'minio_buckets_total',
    'Total number of buckets'
)
minio_objects_total = Gauge(
    'minio_objects_total',
    'Total number of objects per bucket',
    ['bucket']
)
minio_bucket_size_bytes = Gauge(
    'minio_bucket_size_bytes',
    'Total size of bucket in bytes',
    ['bucket']
)
minio_up = Gauge(
    'minio_up',
    'MinIO is reachable'
)
def collect_mlflow_metrics():
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()

        # Эксперименты
        experiments = client.search_experiments()
        mlflow_experiment_total.set(len(experiments))

        for exp in experiments:
            exp_name = exp.name
            if exp_name == 'Default':
                continue

            # Runs по статусам
            for status in ['FINISHED', 'FAILED', 'RUNNING']:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"status = '{status}'"
                )
                mlflow_runs_total.labels(
                    experiment_name=exp_name,
                    status=status
                ).set(len(runs))

            # Последний run — duration и accuracy
            latest_runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=['start_time DESC'],
                max_results=1
            )
            if latest_runs:
                run = latest_runs[0]
                if run.info.end_time and run.info.start_time:
                    duration = (run.info.end_time - run.info.start_time) / 1000
                    mlflow_latest_run_duration.labels(
                        experiment_name=exp_name
                    ).set(duration)
                accuracy = run.data.metrics.get('accuracy')
                if accuracy is not None:
                    mlflow_latest_accuracy.labels(
                        experiment_name=exp_name
                    ).set(accuracy)

        # Зарегистрированные модели
        registered_models = client.search_registered_models()
        mlflow_registered_models_total.set(len(registered_models))

        for model in registered_models:
            for stage in ['None', 'Staging', 'Production', 'Archived']:
                versions = client.get_latest_versions(model.name, stages=[stage])
                mlflow_model_versions_total.labels(
                    model_name=model.name,
                    stage=stage
                ).set(len(versions))

        print(f"MLflow metrics collected: {len(experiments)} experiments, {len(registered_models)} models")

    except Exception as e:
        print(f"Error collecting MLflow metrics: {e}")


def collect_airflow_metrics():
    try:
        auth = (AIRFLOW_USER, AIRFLOW_PASS)
        headers = {'Content-Type': 'application/json'}
        dags_resp = requests.get(
            f"{AIRFLOW_URI}/api/v1/dags",
            auth=auth,
            headers=headers,
            timeout=10
        )
        if dags_resp.status_code == 200:
            dags = dags_resp.json().get('dags', [])
            active = [d for d in dags if not d.get('is_paused')]
            airflow_active_dags.set(len(active))
            for dag in dags:
                dag_id = dag['dag_id']

                for status in ['success', 'failed', 'running']:
                    runs_resp = requests.get(
                        f"{AIRFLOW_URI}/api/v1/dags/{dag_id}/dagRuns",
                        auth=auth,
                        headers=headers,
                        params={'state': status, 'limit': 100},
                        timeout=10
                    )
                    if runs_resp.status_code == 200:
                        count = runs_resp.json().get('total_entries', 0)
                        airflow_dag_status.labels(
                            dag_id=dag_id,
                            status=status
                        ).set(count)
                runs_resp = requests.get(
                    f"{AIRFLOW_URI}/api/v1/dags/{dag_id}/dagRuns",
                    auth=auth,
                    headers=headers,
                    params={'state': 'success', 'limit': 1, 'order_by': '-start_date'},
                    timeout=10
                )
                if runs_resp.status_code == 200:
                    runs = runs_resp.json().get('dag_runs', [])
                    if runs:
                        run_id = runs[0]['dag_run_id']
                        tasks_resp = requests.get(
                            f"{AIRFLOW_URI}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances",
                            auth=auth,
                            headers=headers,
                            timeout=10
                        )
                        if tasks_resp.status_code == 200:
                            tasks = tasks_resp.json().get('task_instances', [])
                            for task in tasks:
                                duration = task.get('duration')
                                if duration:
                                    airflow_task_duration.labels(
                                        dag_id=dag_id,
                                        task_id=task['task_id']
                                    ).set(duration)
                failed_resp = requests.get(
                    f"{AIRFLOW_URI}/api/v1/dags/{dag_id}/dagRuns",
                    auth=auth,
                    headers=headers,
                    params={'state': 'failed', 'limit': 100},
                    timeout=10
                )
                if failed_resp.status_code == 200:
                    failed_count = failed_resp.json().get('total_entries', 0)
                    airflow_failed_tasks.labels(dag_id=dag_id).set(failed_count)

        print(f"Airflow metrics collected: {len(dags)} DAGs")

    except Exception as e:
        print(f"Error collecting Airflow metrics: {e}")

def collect_minio_metrics():
    try:
        # Проверяем доступность
        health = requests.get(
            f"{MINIO_URI}/minio/health/live",
            timeout=5
        )
        minio_up.set(1 if health.status_code == 200 else 0)

        # Используем MinIO API для статистики бакетов
        from xml.etree import ElementTree as ET
        import hmac
        import hashlib
        import datetime
        import boto3
        from botocore.client import Config

        s3 = boto3.client(
            's3',
            endpoint_url=MINIO_URI,
            aws_access_key_id=MINIO_USER,
            aws_secret_access_key=MINIO_PASS,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )

        buckets = s3.list_buckets().get('Buckets', [])
        minio_buckets_total.set(len(buckets))

        for bucket in buckets:
            bucket_name = bucket['Name']
            total_size = 0
            total_objects = 0

            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name):
                objects = page.get('Contents', [])
                total_objects += len(objects)
                total_size += sum(obj['Size'] for obj in objects)

            minio_objects_total.labels(bucket=bucket_name).set(total_objects)
            minio_bucket_size_bytes.labels(bucket=bucket_name).set(total_size)

        print(f"MinIO metrics collected: {len(buckets)} buckets")

    except Exception as e:
        print(f"Error collecting MinIO metrics: {e}")

if __name__ == '__main__':
    print(f"Starting exporter on port 9101")
    print(f"MLflow: {MLFLOW_URI}")
    print(f"Airflow: {AIRFLOW_URI}")
    print(f"MinIO: {MINIO_URI}")
    print(f"Scrape interval: {SCRAPE_INTERVAL}s")

    start_http_server(9101)

    while True:
        collect_mlflow_metrics()
        collect_airflow_metrics()
        collect_minio_metrics()
        time.sleep(SCRAPE_INTERVAL)