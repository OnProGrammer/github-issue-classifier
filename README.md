# ML4 - Классификация GitHub Issues

Классификация issues из репозитория **kubernetes/kubernetes** на 4 класса: `BUG`, `FEATURE`, `DOCS`, `SUPPORT`

Модель - **Bi-LSTM** с BPE-токенизацией (`youtokentome`)

## Структура проекта

```
├── parser.py                       # CLI entrypoint для скрапера
├── scraper/
│   ├── config.py                   # YAML-конфиг + dataclass
│   ├── issue_scraper.py            # Оркестратор pipeline сбора данных
│   ├── net/
│   │   ├── proxy_pool.py           # Пул прокси с ротацией и ban-листом
│   │   ├── http_client.py          # HTTP-клиент: retry, backoff, rate limiter
│   │   └── logging_config.py       # Structured logging (text / JSON)
│   ├── parsing/
│   │   └── github_parser.py        # Парсинг HTML страниц GitHub
│   └── storage/
│       ├── sqlite_storage.py       # SQLite storage с upsert
│       └── checkpoint.py           # Checkpoint/resume
├── scraper_config.example.yaml
├── dataset.py                      # Построение CSV-датасета из SQLite
├── train.py                        # Обучение и оценка модели
├── Отчет.ipynb
├── data/
│   ├── issues_bug.db
│   ├── issues_feature.db
│   ├── issues_docs.db
│   └── issues_support.db
├── out/
│   ├── issues.csv
│   ├── bpe_issues.model
│   └── issues_ml_4_model.pt
└── graphs/
    ├── training_stats.png
    └── confusion_matrix.png
```

## Scraper

### Компоненты

| Модуль | Что делает |
|--------|-----------|
| `ProxyPool` | Round-robin ротация прокси, автобан при серии ошибок с эскалацией (5м -> 10м -> ...), cooldown, health-статистика |
| `HttpClient` | Retry с exponential backoff + jitter, rate limiting (рандомная задержка), ротация User-Agent / Accept-Language |
| `IssueStorage` | SQLite с upsert по `(repo, issue_id)`, context manager |
| `ScrapeProgress` | JSON-checkpoint по `(target, label, page)`, продолжение после Ctrl+C |
| `IssueScraper` | Graceful shutdown по SIGINT/SIGTERM, периодический checkpoint, логирование прогресса |

### Конфигурация

Три уровня (приоритет: CLI > YAML > defaults):

```bash
# Сгенерировать шаблон конфига
python parser.py --dump-config scraper_config.yaml

# Запуск с конфигом
python parser.py --config scraper_config.yaml

# Переопределение через CLI
python parser.py --config scraper_config.yaml --targets BUG FEATURE --num-pages 20 --log-level DEBUG
```

### Прокси

```bash
# Через CLI
python parser.py --proxies "http://user:pass@host1:port" "http://user:pass@host2:port"

# Через файл (один URL на строку, # для комментариев)
python parser.py --proxy-file proxies.txt
```

### Checkpoint / Resume

Прогресс сохраняется в `data/checkpoint.json`, при повторном запуске уже обработанные страницы пропускаются:

```bash
python parser.py --config scraper_config.yaml          # продолжит с места остановки
python parser.py --config scraper_config.yaml --fresh   # начать заново
```

### Логирование

```bash
python parser.py --log-level INFO                      # human-readable (дефолт)
python parser.py --log-json --log-level DEBUG           # JSON для ELK / Grafana Loki
```

Пример JSON-лога:
```json
{"ts": "2025-01-15T12:00:00+00:00", "level": "INFO", "logger": "scraper.issue_scraper", "msg": "saved issue #12345 [BUG], total: 42"}
```

## Запуск полного пайплайна

```bash
# 1. Сбор данных
python parser.py --config scraper_config.yaml

# 2. Построение датасета
python dataset.py --data-dir data --output out/issues.csv --train-fraction 0.8 --seed 42

# 3. Обучение модели
python train.py --csv out/issues.csv --epochs 6 --batch-size 64 --lr 5e-4
```

## Данные

Для каждого target-класса issues фильтруются по лейблам `kind/*`:
- BUG: `kind/bug`, `kind/regression`, `kind/failing-test`, `kind/flake`
- FEATURE: `kind/feature`, `kind/api-change`, `kind/deprecation`
- DOCS: `kind/documentation`
- SUPPORT: `kind/support`

Текстовое поле: `[TITLE] заголовок [BODY] тело`

Токенизация - BPE (`youtokentome`), словарь 16000 токенов

Сплит 80/20, seed=42

## Модель

```
Embedding(vocab_size, 128, padding_idx=0)
    |
Bi-LSTM(128 -> 256, 1 layer) -> concat hidden [512]
    |
Dropout(0.3) -> Linear(512, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 4)
```

CrossEntropyLoss, Adam lr=5e-4, batch 64, 6 эпох

## Результаты

Test loss ~0.85, test accuracy ~0.65

### Графики обучения

![Training curves](graphs/training_stats.png)

### Матрица ошибок

![Confusion matrix](graphs/confusion_matrix.png)

- BUG и FEATURE классифицируются хорошо (95 верных из ~130)
- DOCS - 52 верных, часть путается с BUG и FEATURE
- SUPPORT - наибольшая ошибка, мало примеров + похожая семантика

## Зависимости

- Python 3.10+
- `requests`, `beautifulsoup4`, `pyyaml`
- `pandas`, `numpy`
- `torch`
- `youtokentome`
- `scikit-learn`, `matplotlib`
