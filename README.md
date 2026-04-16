## Что это за проект

Это автономный веб-агент на `Playwright + LangGraph + LLM`, который решает пользовательскую задачу пошагово:
- декомпозирует цель на подзадачи,
- выполняет по одному действию за шаг,
- проверяет прогресс критиком,
- умеет восстанавливаться после сбоев, циклов и «ухода не туда».

Ключевой фокус решения — не просто «кликать», а **контролируемо доводить задачу до результата** с прозрачными логами и диагностикой.

## Ключевые решения

- **Двухролевая модель управления:** `Critic` отвечает за план/контроль, `Executor` — за точечные действия.
- **Устойчивость к нестабильному UI:** рестарты подзадач, детект повторов, реплан при застревании.
- **Безопасность операций:** потенциально рискованные клики блокируются через отдельный путь `dangerous_action`.
- **Мультимодальная проверка:** критик оценивает и текстовое состояние страницы, и скриншот текущего шага.
- **Навигационная память:** агент ведет карту пройденного пути (`navigation_trace`) и использует ее для возврата в правильный контекст.
- **Наблюдаемость “из коробки”:** подробные `jsonl`-логи, таймлайн и итоговая статистика по времени/инструментам.

## Быстрый старт

```bash
python main.py
```

Перед запуском настройте переменные окружения (см. раздел ниже).

## Архитектура

- `main.py` — точка входа.
- `src/agent_core.py` — оркестрация рантайма (LangGraph state machine, `Critic`, `Executor`).
- `src/entry.py` — браузерное окружение (single-tab guard, оверлеи элементов, скриншот + текстовое состояние).
- `src/agent_tools.py` — инструменты, доступные исполнителю (`navigate`, `click`, `type_text`, `press_enter`, `dangerous_action`).
- `src/prompts.py` — все системные/пользовательские промпты для planner/replan/critic/executor/extract.
- `src/run_logger.py` — отдельный логгер рантайма и итоговой статистики.

## Модель выполнения

1. `Critic` строит план из 4-8 подзадач.
2. `Executor` на каждом шаге выбирает **ровно один** инструмент.
3. После действия снимается новое состояние страницы (скрин + структурированный текст).
4. `Critic` оценивает текущую подзадачу (`done/progress/stuck/offtrack`).
5. При проблемах рантайм перезапускает подзадачу или делает реплан.
6. На выходе формируется `SUCCESS` или `BLOCKED` с подробным отчетом.

## Граф узлов (LangGraph)

Основной поток:

`init -> subtask_setup -> guard -> snapshot -> executor_decide -> executor_execute -> critic_evaluate`

Ветвления:
- `critic_evaluate -> guard` (продолжить подзадачу),
- `critic_evaluate -> subtask_setup` (следующая подзадача),
- `critic_evaluate -> replan` (перестроить план),
- `critic_evaluate -> finalize` (успех/блокировка),
- `guard -> finalize` (лимиты/жесткий стоп).

## Логирование и аналитика

Для каждого запуска создается директория в `artifacts/runs/<run_id>/`:
- `events.jsonl` — поток всех событий рантайма,
- `timeline.md` — человекочитаемая хронология.

В финальном событии `run_finished` дополнительно пишутся:
- **полный маршрут агента** (`agent_route`),
- **карта контрольных точек** (`navigation_trace`),
- **время выполнения** (`elapsed_seconds`),
- **статистика инструментов** (`tool_call_counts`).

Это позволяет быстро анализировать стабильность, узкие места и стоимость шага.

## Переменные окружения

Обязательные:
- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL` (по умолчанию `https://openrouter.ai/api/v1`)
- `AGENT_MODEL` (по умолчанию `openai/gpt-4o`)

Лимиты рантайма:
- `AGENT_MAX_TOTAL_STEPS`
- `AGENT_MAX_STEPS_PER_SUBTASK`
- `AGENT_MAX_RESTARTS_PER_SUBTASK`
- `AGENT_LOOP_REPEAT_THRESHOLD`
- `AGENT_LANGGRAPH_RECURSION_LIMIT` (по умолчанию `200`)
- `AGENT_MAX_SNAPSHOT_FAIL_STREAK` (по умолчанию `5`)
- `AGENT_MAX_REPLANS`

Тюнинг снапшотов (`src/entry.py`):
- `AGENT_SNAPSHOT_LOAD_WAIT_MS`
- `AGENT_SNAPSHOT_MUTATION_WAIT_MS`
- `AGENT_SNAPSHOT_MUTATION_IDLE_MS`
- `AGENT_SNAPSHOT_FALLBACK_SETTLE_MS`

Рекомендуемый baseline:

```env
AGENT_MAX_TOTAL_STEPS=45
AGENT_MAX_STEPS_PER_SUBTASK=10
AGENT_MAX_RESTARTS_PER_SUBTASK=2
AGENT_LOOP_REPEAT_THRESHOLD=3
AGENT_LANGGRAPH_RECURSION_LIMIT=200
AGENT_MAX_SNAPSHOT_FAIL_STREAK=5
AGENT_MAX_REPLANS=2
```