from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.operators.python import PythonOperator
from kubernetes.client import models as k8s
from datetime import datetime, timedelta
import logging

# ============================================
# КОНФИГУРАЦИЯ DAG
# ============================================

# Базовые параметры для всех задач
DEFAULT_ARGS = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,  # Количество повторных попыток при ошибке
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=30),
}

TRAINING_IMAGE = "192.168.1.100:5000/iris-train:1.0"
K8S_NAMESPACE = "airflow"
MLFLOW_URI = "http://mlflow-server.mlflow.svc.cluster.local:5000"

# Ресурсы для обучения
TRAINING_RESOURCES = k8s.V1ResourceRequirements(
    requests={
        'memory': '2Gi',
        'cpu': '1'
    },
    limits={
        'memory': '4Gi',
        'cpu': '2'
    }
)

# ============================================
# СОЗДАНИЕ DAG
# ============================================

dag = DAG(
    dag_id='iris_ml_pipeline',
    default_args=DEFAULT_ARGS,
    description='Полный ML пайплайн iris-classifier',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Не запускать для прошлых дат
    tags=['ml', 'iris', 'production', 'coursework'],
)

train_model_task = KubernetesPodOperator(
    task_id='train_model',
    name='iris-train-model',
    namespace=K8S_NAMESPACE,
    image=TRAINING_IMAGE,
    
    # Команда для запуска
    cmds=["python"],
    arguments=["train.py"],
    
    # Environment variables
    env_vars={
        'MLFLOW_TRACKING_URI': MLFLOW_URI,
        'PYTHONUNBUFFERED': '1',  # Для логов в реальном времени
    },
    
    # Ресурсы CPU/Memory
    container_resources=TRAINING_RESOURCES,
    
    # Политики
    is_delete_operator_pod=True,  # Удалять Pod после выполнения
    get_logs=True,  # Собирать логи в Airflow
    log_events_on_failure=True,  # Логировать события при ошибках
    
    # Таймаут выполнения
    execution_timeout=timedelta(minutes=15),
    
    # ImagePullPolicy - Always чтобы всегда подтягивать свежий образ
    image_pull_policy='Always',
    
    dag=dag,
)

# ============================================
# ЗАДАЧА 2: ВАЛИДАЦИЯ МОДЕЛИ
# ============================================

validate_model_task = KubernetesPodOperator(
    task_id='validate_model',
    name='iris-validate-model',
    namespace=K8S_NAMESPACE,
    image=TRAINING_IMAGE,
    
    # Запускаем Python код inline для валидации
    cmds=["python"],
    arguments=["-c", """
import mlflow
import sys

print('ВАЛИДАЦИЯ МОДЕЛИ')
print('='*60)

mlflow_uri = '""" + MLFLOW_URI + """'
print(f'Подключение к MLflow: {mlflow_uri}')
mlflow.set_tracking_uri(mlflow_uri)

# Ищем последний run в эксперименте
experiment = mlflow.get_experiment_by_name('iris-classification')
if not experiment:
    print('❌ Эксперимент iris-classification не найден!')
    sys.exit(1)

print(f'✅ Эксперимент найден: {experiment.experiment_id}')

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
print(f'   Run ID: {run_id}')
print(f'   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')

# Валидация по минимальному порогу
MIN_ACCURACY = 0.85

print(f'\\n✅ Проверка порога: accuracy >= {MIN_ACCURACY}')

if accuracy >= MIN_ACCURACY:
    print(f'✅ PASSED: Модель прошла валидацию!')
    print(f'   Модель готова к регистрации и деплою')
    sys.exit(0)
else:
    print(f'❌ FAILED: Модель НЕ прошла валидацию')
    print(f'   Accuracy {accuracy:.4f} < {MIN_ACCURACY}')
    print(f'   Деплой отменён!')
    sys.exit(1)
    """],
    
    env_vars={
        'MLFLOW_TRACKING_URI': MLFLOW_URI,
        'PYTHONUNBUFFERED': '1',
    },
    
    is_delete_operator_pod=True,
    get_logs=True,
    image_pull_policy='Always',
    
    dag=dag,
)

register_model_task = KubernetesPodOperator(
    task_id='register_model',
    name='iris-register-model',
    namespace=K8S_NAMESPACE,
    image=TRAINING_IMAGE,
    
    cmds=["python"],
    arguments=["-c", """
import mlflow
from mlflow.tracking import MlflowClient
import sys

print('='*60)
print('📝 РЕГИСТРАЦИЯ МОДЕЛИ В MODEL REGISTRY')
print('='*60)

# Подключаемся
mlflow_uri = '""" + MLFLOW_URI + """'
mlflow.set_tracking_uri(mlflow_uri)
client = MlflowClient()

# Получаем последний run
experiment = mlflow.get_experiment_by_name('iris-classification')
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=['start_time DESC'],
    max_results=1
)

run_id = runs.iloc[0]['run_id']
accuracy = runs.iloc[0]['metrics.accuracy']
model_uri = f'runs:/{run_id}/model'

print(f'\\n📦 Информация о модели:')
print(f'   Run ID: {run_id}')
print(f'   Model URI: {model_uri}')
print(f'   Accuracy: {accuracy:.4f}')

# Имя модели в Registry
model_name = 'iris-classifier'

try:
    # Регистрируем новую версию
    print(f'\\n🔄 Создание новой версии модели: {model_name}')
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description=f'Iris classifier with accuracy {accuracy:.4f}'
    )
    
    version_number = model_version.version
    print(f'✅ Модель зарегистрирована как версия {version_number}')
    
    # Переводим в Production stage
    print(f'\\n🚀 Перевод версии {version_number} в Production...')
    client.transition_model_version_stage(
        name=model_name,
        version=version_number,
        stage='Production',
        archive_existing_versions=True
    )
    
    print(f'✅ Версия {version_number} переведена в Production!')
    print(f'\\n🎉 Регистрация успешно завершена!')
    
except Exception as e:
    print(f'❌ Ошибка регистрации: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
    """],
    
    env_vars={
        'MLFLOW_TRACKING_URI': MLFLOW_URI,
        'PYTHONUNBUFFERED': '1',
    },
    
    is_delete_operator_pod=True,
    get_logs=True,
    image_pull_policy='Always',
    
    dag=dag,
)

# ============================================
# ЗАДАЧА 4: УСПЕШНОЕ ЗАВЕРШЕНИЕ (ИНФОРМАЦИЯ)
# ============================================

def print_success_message(**context):
    logging.info("="*60)
    logging.info("ML ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЁН!")
    logging.info("="*60)
    logging.info("\n✅ Выполнены все этапы:")
    logging.info("   1. ✅ Обучение модели")
    logging.info("   2. ✅ Валидация качества")
    logging.info("   3. ✅ Регистрация в Model Registry")
    logging.info("="*60)

success_task = PythonOperator(
    task_id='pipeline_success',
    python_callable=print_success_message,
    dag=dag,
)

train_model_task >> validate_model_task >> register_model_task >> success_task