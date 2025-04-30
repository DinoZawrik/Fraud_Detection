# --- Базовый образ Python ---
    FROM python:3.10-slim

    # --- Установка рабочей директории ---
    WORKDIR /app
    
    # --- Копирование файла зависимостей ---
    COPY requirements.txt ./
    
    # --- !!! УСТАНОВКА СИСТЕМНОЙ ЗАВИСИМОСТИ libgomp1 !!! ---
    RUN apt-get update && \
        apt-get install -y libgomp1 && \
        rm -rf /var/lib/apt/lists/*
    # --------------------------------------------------------
    
    # --- Установка зависимостей Python ---
    RUN pip install --upgrade pip && \
        pip install --no-cache-dir --default-timeout=100 -r requirements.txt
    
    # --- Копирование всего кода проекта ---
    COPY . .
    
    # --- Открытие порта ---
    EXPOSE 8501
    
    # --- Команда запуска ---
    CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]