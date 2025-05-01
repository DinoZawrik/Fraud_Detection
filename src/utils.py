"""
Вспомогательные утилиты для проекта.

Содержит функции для сохранения и загрузки объектов (моделей, метрик и т.д.)
с использованием joblib, а также для загрузки данных из CSV.
"""

import joblib
import os
import pandas as pd
import traceback


def save_joblib(data, file_path):
    """Сохраняет объект в файл с использованием joblib."""
    try:
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(data, file_path)
        print(f"Объект успешно сохранен: {file_path}")
    except Exception as e:
        print(f"Ошибка при сохранении объекта в {file_path}:")
        traceback.print_exc()  # Выводим traceback для диагностики


def load_joblib(file_path):
    """Загружает объект из файла с использованием joblib."""
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл не найден по пути {file_path}")
        return None
    try:
        print(f"Загрузка объекта: {file_path}")
        data = joblib.load(file_path)
        print(f"Объект успешно загружен: {file_path}")
        return data
    except Exception as e:
        print(f"Ошибка при загрузке объекта из {file_path}:")
        traceback.print_exc()  # Выводим traceback для диагностики
        return None


# Псевдонимы для большей ясности при использовании для пайплайнов
save_pipeline = save_joblib
load_pipeline = load_joblib


def load_data(file_path):
    """Загружает данные из CSV файла в pandas DataFrame."""
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл данных не найден по пути {file_path}")
        return None
    try:
        print(f"Загрузка данных: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Данные успешно загружены: {file_path} (строк: {len(df)})")
        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных из {file_path}:")
        traceback.print_exc()  # Выводим traceback для диагностики
        return None
