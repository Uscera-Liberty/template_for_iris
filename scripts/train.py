import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

# MLflow tracking URI - адрес сервера MLflow
# ВАЖНО: Замени на свой IP из команды: kubectl get svc -n mlflow
MLFLOW_TRACKING_URI = os.getenv(
    'MLFLOW_TRACKING_URI', 
    'http://192.168.10.12:30055/'
)

# Параметры модели Random Forest
MODEL_PARAMS = {
    'n_estimators': 100,      # Количество деревьев в лесу
    'max_depth': 5,           # Максимальная глубина дерева
    'min_samples_split': 2,   # Минимум образцов для разделения узла
    'min_samples_leaf': 1,    # Минимум образцов в листе
    'random_state': 42        # Для воспроизводимости результатов
}

# Размер тестовой выборки (20% данных)
TEST_SIZE = 0.2

# Минимальная точность для прохождения валидации
MIN_ACCURACY = 0.85

def load_data():
    """
    Загружает датасет Iris из sklearn.
    
    Датасет содержит:
    - 150 образцов (50 каждого класса)
    - 4 признака (длина/ширина чашелистика и лепестка)
    - 3 класса цветов
    
    Returns:
        tuple: (X, y, feature_names, target_names)
    """
    print("\n" + "="*60)
    print("📦 ЗАГРУЗКА ДАННЫХ")
    print("="*60)
    
    # Загружаем встроенный датасет
    iris = load_iris()
    
    # Извлекаем данные
    X = iris.data                    # Признаки (150 x 4)
    y = iris.target                  # Целевые метки (150,)
    feature_names = iris.feature_names  # Названия признаков
    target_names = iris.target_names    # Названия классов
    
    # Выводим информацию
    print(f"✅ Датасет Iris успешно загружен!")
    print(f"\n📊 Информация о данных:")
    print(f"   • Всего образцов: {len(X)}")
    print(f"   • Количество признаков: {X.shape[1]}")
    print(f"   • Классы: {', '.join(target_names)}")
    print(f"\n🔍 Признаки:")
    for i, name in enumerate(feature_names, 1):
        print(f"   {i}. {name}")
    
    # Показываем распределение классов
    print(f"\n📈 Распределение классов:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(target_names, counts):
        print(f"   • {cls}: {count} образцов")
    
    return X, y, feature_names, target_names

def split_data(X, y):
    """
    Разделяет данные на обучающую и тестовую выборки.
    
    Используем stratify для сохранения пропорций классов.
    
    Args:
        X: Признаки
        y: Целевые метки
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*60)
    print("✂️  РАЗДЕЛЕНИЕ ДАННЫХ")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y  # Сохраняем пропорции классов
    )
    
    print(f"✅ Данные успешно разделены!")
    print(f"\n📊 Размеры выборок:")
    print(f"   • Обучающая: {len(X_train)} образцов ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"   • Тестовая:  {len(X_test)} образцов ({TEST_SIZE*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Обучает модель Random Forest Classifier.
    
    Args:
        X_train: Обучающие признаки
        y_train: Обучающие метки
        
    Returns:
        model: Обученная модель
    """
    print("\n" + "="*60)
    print("🎯 ОБУЧЕНИЕ МОДЕЛИ")
    print("="*60)
    
    print(f"\n⚙️  Параметры модели:")
    for param, value in MODEL_PARAMS.items():
        print(f"   • {param}: {value}")
    
    # Создаём модель
    model = RandomForestClassifier(**MODEL_PARAMS)
    
    # Обучаем
    print(f"\n🔄 Обучение Random Forest...")
    model.fit(X_train, y_train)
    
    print(f"✅ Модель успешно обучена!")
    
    return model

def evaluate_model(model, X_test, y_test, target_names):
    """
    Оценивает качество модели на тестовой выборке.
    
    Args:
        model: Обученная модель
        X_test: Тестовые признаки
        y_test: Тестовые метки
        target_names: Названия классов
        
    Returns:
        dict: Словарь с метриками
    """
    print("\n" + "="*60)
    print("📈 ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
    print("="*60)
    
    # Делаем предсказания
    y_pred = model.predict(X_test)
    
    # Вычисляем accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Детальный отчёт по классам
    print(f"\n📊 Детальный отчёт по классам:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    print(f"📋 Матрица ошибок:")
    print(cm)
    
    # Собираем метрики
    metrics = {
        'accuracy': accuracy,
        'predictions': y_pred,
        'confusion_matrix': cm
    }
    
    return metrics

def log_to_mlflow(model, metrics, params, feature_names, X_train, y_pred):
    """
    Логирует модель и метрики в MLflow.
    
    Args:
        model: Обученная модель
        metrics: Словарь с метриками
        params: Параметры модели
        feature_names: Названия признаков
        X_train: Обучающие данные (для signature)
        y_pred: Предсказания (для signature)
        
    Returns:
        str: Run ID в MLflow
    """
    print("\n" + "="*60)
    print("📝 ЛОГИРОВАНИЕ В MLFLOW")
    print("="*60)
    
    # Подключаемся к MLflow
    print(f"\n🔗 Подключение к MLflow: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("iris-classification")
    
    # Создаём уникальное имя run
    run_name = f"iris-rf-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Начинаем run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"✅ Run создан: {run_id}")
        
        # Логируем параметры
        print(f"\n📌 Логирование параметров...")
        mlflow.log_params(params)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("dataset", "iris")
        
        # Логируем метрики
        print(f"📊 Логирование метрик...")
        mlflow.log_metric("accuracy", metrics['accuracy'])
        
        # Логируем важность признаков
        print(f"🎯 Логирование важности признаков...")
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n   Топ признаков:")
        for _, row in feature_importance.iterrows():
            print(f"   • {row['feature']}: {row['importance']:.4f}")
        
        print(f"\n💾 Сохранение модели...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="iris-classifier",
            signature=mlflow.models.infer_signature(X_train, y_pred)
        )
        
        print(f"✅ Модель сохранена в MLflow!")
        
        return run_id

def main():
    print("\n" + "="*70)
    print("🚀 ЗАПУСК ОБУЧЕНИЯ ML-МОДЕЛИ")
    print("="*70)
    print(f"⏰ Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔗 MLflow URI: {MLFLOW_TRACKING_URI}")
    
    try:
        # 1. Загружаем данные
        X, y, feature_names, target_names = load_data()
        
        # 2. Разделяем на train/test
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 3. Обучаем модель
        model = train_model(X_train, y_train)
        
        # 4. Оценка качества
        metrics = evaluate_model(model, X_test, y_test, target_names)
        
        # 5. Логируем в MLflow
        run_id = log_to_mlflow(
            model, metrics, MODEL_PARAMS, 
            feature_names, X_train, metrics['predictions']
        )
        
        print("\n" + "="*70)
        print("🎉 ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print("="*70)
        print(f"\n📊 Результаты:")
        print(f"   • Run ID: {run_id}")
        print(f"   • Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   • Модель: iris-classifier")
        print(f"   • Эксперимент: iris-classification")
        
        print(f"\n🔗 Просмотр результатов:")
        print(f"   MLflow UI: {MLFLOW_TRACKING_URI}")
        
        # Проверка валидации
        print(f"\n✅ ВАЛИДАЦИЯ:")
        if metrics['accuracy'] >= MIN_ACCURACY:
            print(f"   ✅ PASSED - Модель готова к деплою")
            print(f"   Accuracy {metrics['accuracy']:.4f} >= {MIN_ACCURACY}")
            return 0
        else:
            print(f"   FAILED - Модель не прошла валидацию")
            print(f"   Accuracy {metrics['accuracy']:.4f} < {MIN_ACCURACY}")
            return 1
            
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    print("\n" + "="*70)
    sys.exit(exit_code)