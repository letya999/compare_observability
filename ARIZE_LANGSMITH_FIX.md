# Arize & LangSmith Integration - Deployment Checklist

## Что было исправлено (основано на официальной документации 2026)

### 1. **Arize AX** - Полная переделка ✅
**Проблема:** Использовалась устаревшая интеграция через `phoenix.otel` и `PHOENIX_*` переменные.

**Решение:** Переписан на актуальный метод 2026 года:
- Пакет: `arize-otel` (не `arize-phoenix-otel`)
- Функция: `arize.otel.register(space_id, api_key, project_name)`
- Переменные окружения:
  ```bash
  ARIZE_SPACE_ID=***
  ARIZE_API_KEY=***
  ARIZE_PROJECT_NAME=compare-observability
  ```

**Источник:** https://arize.com/docs/ax/integrations/python-agent-frameworks/langchain/langchain-tracing

---

### 2. **LangSmith** - Исправлены flush и atexit ✅
**Проблема:** В Streamlit приложении `shutdown()` не вызывался, трейсы не успевали отправляться.

**Решение:**
- Добавлен `atexit.register(self.shutdown)` в `ObservabilityManager.__init__`
- Теперь при завершении программы автоматически вызывается `force_flush()` для всех провайдеров
- LangSmith получает глобальный `Client()` и использует `client.flush()` в shutdown

**Источник:** https://docs.langchain.com/langsmith/observability-quickstart

---

## Инструкции по запуску

### Шаг 1: Установите зависимости
```bash
pip install --upgrade arize-otel langsmith openinference-instrumentation-openai
# или
pip install -e .
```

### Шаг 2: Проверьте `.env`
Файл `.env` уже обновлён правильными переменными:
```bash
# Arize AX (2026)
ARIZE_SPACE_ID=***
ARIZE_API_KEY=***
ARIZE_PROJECT_NAME=compare-observability

# LangSmith
LANGCHAIN_API_KEY=***
LANGCHAIN_PROJECT=compare_observability
LANGSMITH_TRACING=true

```

### Шаг 3: Запустите Streamlit приложение
```bash
streamlit run app.py
```

### Шаг 4: Проверьте дашборды
- **Arize AX**: https://arize.com/spaces/U3BhY2U6Mzc1MjQ6NXRFZA==/tracing
- **LangSmith**: https://smith.langchain.com/projects/compare_observability

---

## Технические детали

### Arize: Почему не работало
1. **Конфликт провайдеров:** Opik и старый Phoenix код оба пытались захватить глобальный `TracerProvider`
2. **Отсутствие Space ID:** Старая версия использовала `PHOENIX_COLLECTOR_ENDPOINT`, новая требует `ARIZE_SPACE_ID`
3. **Неправильный пакет:** `arize-phoenix-otel` устарел в 2026, актуальный — `arize-otel`

### LangSmith: Почему не работало
1. **Streamlit особенность:** Нет встроенного `on_exit` хука
2. **Асинхронная отправка:** LangSmith отправляет трейсы в фоне, нужен explicit flush

### Решение: `atexit`
Добавлен автоматический вызов `shutdown()` при выходе из программы:
```python
import atexit
atexit.register(self.shutdown)
```

Это гарантирует, что все провайдеры (LangSmith, Arize, Braintrust, Opik) успевают отправить данные перед завершением.

---

## Проверка

После запуска приложения:
1. Задайте вопрос в UI
2. Подождите 2-3 секунды после получения ответа
3. Проверьте дашборды Arize и LangSmith

**Все должно работать.**
