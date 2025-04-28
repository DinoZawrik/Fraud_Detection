import joblib
import os
import pandas as pd

def save_pipeline(pipeline, file_path):
    """Сохраняет пайплайн в файл."""
    try:
        joblib.dump(pipeline, file_path)
        print(f"Пайплайн успешно сохранен в: {file_path}")
    except Exception as e:
        print(f"Ошибка при сохранении пайплайна в {file_path}: {e}")

def load_pipeline(file_path):
    """Загружает пайплайн из файла."""
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл модели не найден по пути {file_path}")
        return None
    try:
        pipeline = joblib.load(file_path)
        print(f"Пайплайн успешно загружен из: {file_path}")
        return pipeline
    except Exception as e:
        print(f"Ошибка при загрузке пайплайна из {file_path}: {e}")
        return None

def load_data(file_path):
    """Загружает данные из CSV."""
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл данных не найден по пути {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"Данные успешно загружены из: {file_path}")
        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных из {file_path}: {e}")
        return None