"""
Вспомогательные утилиты для проекта.

Содержит функции для сохранения и загрузки объектов (моделей, метрик и т.д.)
с использованием joblib, а также для загрузки данных из CSV.
"""

import joblib
import os
import pandas as pd
import traceback
import requests

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


from io import StringIO # Для чтения CSV из текста

def load_data(file_path_or_url):
    """Загружает данные из CSV по локальному пути или URL."""
    if file_path_or_url.startswith('http'):
        try:
            print(f"Загрузка данных по URL: {file_path_or_url}")
            response = requests.get(file_path_or_url)
            response.raise_for_status() # Проверка на ошибки HTTP
            csv_data = StringIO(response.text) # Читаем текстовый ответ как файл
            df = pd.read_csv(csv_data)
            print("Данные успешно загружены по URL.")
            return df
        except Exception as e:
            print(f"Ошибка при загрузке данных по URL {file_path_or_url}: {e}")
            return None
    else: # Локальный путь
        if not os.path.exists(file_path_or_url):
            print(f"Ошибка: Файл данных не найден по пути {file_path_or_url}")
            return None
        try:
            df = pd.read_csv(file_path_or_url)
            print(f"Данные успешно загружены из: {file_path_or_url}")
            return df
        except Exception as e:
            print(f"Ошибка при загрузке данных из {file_path_or_url}: {e}")
            return None
