from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

DEFAULT_ARGS = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,  # Количество повторных попыток при ошибке
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=30),
}

# ВАЖНО: Твой внешний MLflow сервер
MLFLOW_URI = "http://mlflow:5000"

# Образ для обучения (соберём локально)
TRAINING_IMAGE = "iris-train:local"

# Environment variables для всех задач
ENV_VARS = {
    'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
    'MLFLOW_S3_ENDPOINT_URL':'http://minio:9000',
    'AWS_ACCESS_KEY_ID': 'mlops',
    'AWS_SECRET_ACCESS_KEY': 'quZK9X8PfXfrlpi',
    'PYTHONUNBUFFERED': '1'
}

# ============================================
# СОЗДАНИЕ DAG
# ============================================

dag = DAG(
    dag_id='iris_ml_pipeline',
    default_args=DEFAULT_ARGS,
    description='ml-pipeline',
    schedule_interval=None,  # Ручной запуск
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Не запускать для прошлых дат
    tags=['ml', 'iris', 'production', 'local'],
)

# ============================================
# ЗАДАЧА 1: ОБУЧЕНИЕ МОДЕЛИ
# ============================================

train_model_task = DockerOperator(
    task_id='train_model',
    image=TRAINING_IMAGE,
    api_version='auto',
    auto_remove=True,  # Удалять контейнер после выполнения
    
    
    # Команда для запуска обучения
    command='python /opt/iris/scripts/train.py',
    
    # Environment variables
    environment=ENV_VARS,
    
    # Docker настройки
    docker_url='unix://var/run/docker.sock',
    network_mode='iris-project_airflow-net',
    mount_tmp_dir=False,
    
    # Таймаут выполнения
    execution_timeout=timedelta(minutes=15),
    
    dag=dag,
)

# ============================================
# ЗАДАЧА 2: ВАЛИДАЦИЯ МОДЕЛИ
# ============================================

validate_model_task = DockerOperator(
    task_id='validate_model',
    image=TRAINING_IMAGE,
    api_version='auto',
    auto_remove=True,
    
    # Запускаем Python код inline для валидации
    command=[
        'python', '-c',
        f"""
import mlflow
import sys

print('=' * 60)
print('📊 ВАЛИДАЦИЯ МОДЕЛИ')
print('=' * 60)

mlflow_uri = '{MLFLOW_URI}'
print(f'Подключение к MLflow: {{mlflow_uri}}')
mlflow.set_tracking_uri(mlflow_uri)

try:
    # Ищем последний run в эксперименте
    experiment = mlflow.get_experiment_by_name('iris-classification')
    if not experiment:
        print('❌ Эксперимент iris-classification не найден!')
        sys.exit(1)

    print(f'✅ Эксперимент найден: {{experiment.experiment_id}}')

    # Получаем последний run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['start_time DESC'],
        max_results=1
    )

    if len(runs) == 0:
        print('❌ Нет runs в эксперименте!')
        sys.exit(1)

    # Извлекаем метрики
    run_id = runs.iloc[0]['run_id']
    accuracy = runs.iloc[0]['metrics.accuracy']

    print(f'\\n📊 Результаты последнего обучения:')
    print(f'   Run ID: {{run_id}}')
    print(f'   Accuracy: {{accuracy:.4f}} ({{accuracy*100:.2f}}%)')

    # Валидация по минимальному порогу
    MIN_ACCURACY = 0.85

    print(f'\\n✅ Проверка порога: accuracy >= {{MIN_ACCURACY}}')

    if accuracy >= MIN_ACCURACY:
        print(f'✅ PASSED: Модель прошла валидацию!')
        print(f'   Модель готова к регистрации и деплою')
        sys.exit(0)
    else:
        print(f'❌ FAILED: Модель НЕ прошла валидацию')
        print(f'   Accuracy {{accuracy:.4f}} < {{MIN_ACCURACY}}')
        print(f'   Деплой отменён!')
        sys.exit(1)

except Exception as e:
    print(f'❌ Ошибка валидации: {{e}}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    ],
    
    environment=ENV_VARS,
    docker_url='unix://var/run/docker.sock',
    network_mode='iris-project_airflow-net',
    mount_tmp_dir=False,
    
    dag=dag,
)

# ============================================
# ЗАДАЧА 3: РЕГИСТРАЦИЯ В MODEL REGISTRY
# ============================================

register_model_task = DockerOperator(
    task_id='register_model',
    image=TRAINING_IMAGE,
    api_version='auto',
    auto_remove=True,
    
    command=[
        'python', '-c',
        f"""
import mlflow
from mlflow.tracking import MlflowClient
import sys

print('=' * 60)
print('📝 РЕГИСТРАЦИЯ МОДЕЛИ В MODEL REGISTRY')
print('=' * 60)

mlflow_uri = '{MLFLOW_URI}'
mlflow.set_tracking_uri(mlflow_uri)
client = MlflowClient()

try:
    # Получаем последний run
    experiment = mlflow.get_experiment_by_name('iris-classification')
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=['start_time DESC'],
        max_results=1
    )

    run_id = runs.iloc[0]['run_id']
    accuracy = runs.iloc[0]['metrics.accuracy']
    model_uri = f'runs:/{{run_id}}/model'

    print(f'\\n📦 Информация о модели:')
    print(f'   Run ID: {{run_id}}')
    print(f'   Model URI: {{model_uri}}')
    print(f'   Accuracy: {{accuracy:.4f}}')

    # Имя модели в Registry
    model_name = 'iris-classifier'

    # Регистрируем новую версию
    print(f'\\n🔄 Создание новой версии модели: {{model_name}}')
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description=f'Iris classifier with accuracy {{accuracy:.4f}}'
    )
    
    version_number = model_version.version
    print(f'✅ Модель зарегистрирована как версия {{version_number}}')
    
    # Переводим в Production stage
    print(f'\\n🚀 Перевод версии {{version_number}} в Production...')
    client.transition_model_version_stage(
        name=model_name,
        version=version_number,
        stage='Production',
        archive_existing_versions=True
    )
    
    print(f'✅ Версия {{version_number}} переведена в Production!')
    print(f'\\n🎉 Регистрация успешно завершена!')
    sys.exit(0)
    
except Exception as e:
    print(f'❌ Ошибка регистрации: {{e}}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    ],
    
    environment=ENV_VARS,
    docker_url='unix://var/run/docker.sock',
    network_mode='iris-project_airflow-net',
    mount_tmp_dir=False,
    
    dag=dag,
)

reload_model_task = DockerOperator(
    task_id='reload_model_server',
    image='curlimages/curl:latest',
    api_version='auto',
    auto_remove=True,
    command=[
        'curl', '-X', 'POST',
        'http://iris-model-server:8000/reload-model',
        '--retry', '3',
        '--retry-delay', '2',
        '-v'
    ],
    docker_url='unix://var/run/docker.sock',
    network_mode='iris-project_airflow-net',
    mount_tmp_dir=False,
    dag=dag,
)

# ============================================
# ЗАДАЧА 4: УСПЕШНОЕ ЗАВЕРШЕНИЕ (ИНФОРМАЦИЯ)
# ============================================

def print_success_message(**context):
    logging.info("=" * 60)
    logging.info("🎉 ML ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН!")
    logging.info("=" * 60)
    logging.info("\n✅ Выполнены все этапы:")
    logging.info("   1. ✅ Обучение модели")
    logging.info("   2. ✅ Валидация качества")
    logging.info("   3. ✅ Регистрация в Model Registry")
    logging.info(f"\n🔗 Результаты доступны в MLflow: {MLFLOW_URI}")
    logging.info("=" * 60)

success_task = PythonOperator(
    task_id='pipeline_success',
    python_callable=print_success_message,
    dag=dag,
)

# ============================================
# ОПРЕДЕЛЯЕМ ПОРЯДОК ВЫПОЛНЕНИЯ
# ============================================

train_model_task >> validate_model_task >> register_model_task >> success_task >> reload_model_task
