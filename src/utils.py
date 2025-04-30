# src/utils.py
import joblib
import os
import pandas as pd
import traceback # <--- Добавь этот импорт

def save_joblib(data, file_path):
    """Универсальная функция для сохранения объекта с помощью joblib."""
    try:
        joblib.dump(data, file_path)
        print(f"Объект успешно сохранен в: {file_path}")
    except Exception as e:
        print(f"Ошибка при сохранении объекта в {file_path}: {e}")

def load_joblib(file_path):
    """Универсальная функция для загрузки объекта с помощью joblib."""
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл не найден по пути {file_path}")
        return None
    try:
        print(f"--> Попытка загрузки: {file_path}") # Добавлено для отладки
        data = joblib.load(file_path)
        print(f"--> Объект успешно загружен из: {file_path}") # Добавлено для отладки
        return data
    except Exception as e:
        # ---> ВЫВОДИМ ПОЛНУЮ ОШИБКУ <---
        print(f"!!! Ошибка при загрузке объекта из {file_path}: {e}")
        print("!!! Полный Traceback: ")
        traceback.print_exc() # Печатаем полный traceback в консоль/логи
        # ---> КОНЕЦ ИЗМЕНЕНИЙ <---
        return None

# Переименуем для ясности
save_pipeline = save_joblib
load_pipeline = load_joblib

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