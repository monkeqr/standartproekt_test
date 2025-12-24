ML Recognition Service (Regno Picker)
Промышленный микросервис для интеллектуального выбора и валидации государственных регистрационных знаков (ГРЗ), основанный на модели CatBoost.

Описание проекта
Сервис предназначен для высоконагруженной обработки данных с камер дорожного движения. Он выполняет классификацию и оценку вероятности корректности распознанного номера, обеспечивая интеграцию ML-модели в продуктивную среду через REST API.

Целевой показатель производительности: 500 запросов в секунду (RPS).

Архитектура и логика работы
Схема обработки данных
Фрагмент кода

graph TD
    A[Клиентский запрос JSON Batch] --> B[FastAPI: Валидация Pydantic]
    B --> C[Feature Engineering: Подготовка признаков]
    C --> D[Singleton Model: Инференс CatBoost]
    D --> E[Формирование ответа JSON]
    
    subgraph "Feature Engineering"
    C1[Регулярные выражения: Тип номера]
    C2[Парсинг временных меток: Окно времени]
    C3[Агрегация Scores: Статистика уверенности]
    C1 & C2 & C3 --> C
    end
Ключевые проектные решения
Singleton Pattern для модели: Загрузка весов модели (micromodel.cbm) производится однократно при инициализации сервиса. Это исключает задержки на дисковый ввод-вывод при обработке запросов.

Batch Processing: Реализована возможность обработки массива объектов в одном запросе, что минимизирует накладные расходы на сетевой стек HTTP.

Конкурентность: Использование ASGI-сервера (Uvicorn) и воркеров Gunicorn позволяет эффективно утилизировать многоядерные процессоры (включая архитектуру Apple M4).

Raw Strings в Regex: Все регулярные выражения для классификации номеров исправлены с использованием raw-строк для предотвращения SyntaxWarning в Python 3.12+.

Запуск в различных окружениях
1. Универсальный запуск через Docker (Рекомендуется)
Подходит для любой ОС, где установлен Docker.

Bash

# Сборка образа
docker build -t regno-service .

# Запуск контейнера
docker run -p 8000:8000 regno-service
2. macOS (Apple Silicon M1/M2/M3/M4)
Запуск в нативном режиме для максимальной производительности.

Bash

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
3. Linux (Ubuntu/Debian)
Использование Gunicorn для обеспечения стабильности при 500 RPS.

Bash

# Установка зависимостей
pip install -r requirements.txt

# Запуск с расчетом воркеров: (2 * кол-во ядер) + 1
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
4. Windows
PowerShell

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
API Документация
После запуска сервис предоставляет интерактивную документацию:

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

Пример POST запроса (/predict_batch)
JSON

[
  {
    "regno_recognize": "А939НО196",
    "afts_regno_ai": "А939НО190",
    "recognition_accuracy": 6.4,
    "afts_regno_ai_score": 0.8689,
    "afts_regno_ai_char_scores": "[0.9, 0.9, 0.9]",
    "afts_regno_ai_length_scores": "[0.1]",
    "camera_type": "Стационарная",
    "camera_class": "Астра-Трафик",
    "time_check": "2021-08-01 09:02:59",
    "direction": 0
  }
]
Тестирование производительности
Для эмуляции нагрузки в 500 RPS рекомендуется использовать инструмент hey:

Bash

hey -z 30s -m POST -H "Content-Type: application/json" -d @payload.json http://localhost:8000/predict_batch
Структура проекта
main.py — веб-интерфейс API.

logic.py — ядро обработки данных и инференса.

schemas.py — схемы валидации данных.

Dockerfile — конфигурация контейнеризации.

micromodel.cbm — бинарный файл модели.
