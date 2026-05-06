# AlphaZero Chess

AlphaZero-style шахматный движок на PyTorch. Архитектура сети совпадает с C++-версией в `chess-ai-cpp`: ResNet 8×224 каналов (~7.9M параметров).

---

## Установка

```bash
cd alpha_zero
pip install -r requirements.txt
```

Требует Python 3.10+ и PyTorch. Для GPU — CUDA-совместимую видеокарту и соответствующий torch (см. [pytorch.org](https://pytorch.org/get-started)).

---

## Запуск обучения

Все команды запускаются из папки `alpha_zero/`.

### С нуля

```bash
python main.py train
```

Дефолтные параметры: 20 итераций, 8 партий/итерацию, 64 симуляции MCTS, batch 64, lr 1e-3.

### С чекпоинта (продолжить обучение)

```bash
python main.py train --resume checkpoints/latest.pt
```

Архитектура сети восстанавливается автоматически из чекпоинта — флаги `--channels`/`--res-blocks` игнорируются при `--resume`.

### Ограничить по времени

```bash
python main.py train --time-limit-minutes 60
```

Останавливается после ближайшего сохранённого чекпоинта по истечении времени.

---

## Куда что сохраняется

| Что | Путь по умолчанию | Описание |
|-----|-------------------|----------|
| Чекпоинты | `checkpoints/` | `<run_id>_alpha_zero_chess_iter_XXXX.pt` + `latest.pt` |
| Партии (PGN) | `games/` | Первые `--record-games` партий каждой итерации |
| Метрики (CSV) | `metrics/` | Лосс, W/B/D, accuracy (если задан eval engine) |
| TensorBoard | `runs/` | По одному подкаталогу `<run_id>/` на каждый запуск |

Все пути переопределяются флагами `--checkpoint-dir`, `--records-dir`, `--metrics-dir`, `--tensorboard-dir`.

---

## TensorBoard

```bash
# из папки alpha_zero/
tensorboard --logdir runs
```

Открыть в браузере: [http://localhost:6006](http://localhost:6006)

Что логируется:
- **loss/total, loss/policy, loss/value** — каждый шаг обучения (глобальный step)
- **lr** — learning rate каждый шаг
- **selfplay/white_wins, black_wins, draws** — по итерациям
- **selfplay/positions** — позиций собрано за итерацию
- **selfplay/buffer_size** — размер replay buffer
- **selfplay/speed_pos_per_s** — скорость сбора позиций
- **train/avg_loss, avg_policy_loss, avg_value_loss** — средний лосс за итерацию
- **time/iteration_seconds** — время итерации
- **audit/accuracy, audit/cp_loss** — точность ходов (только если задан `--eval-engine`)
- Граф модели — записывается один раз при старте

Если TensorBoard не нужен:
```bash
python main.py train --no-tensorboard
```

---

## Ключевые флаги обучения

### Архитектура сети

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--channels` | 224 | Ширина сети (каналы ResNet) |
| `--res-blocks` | 8 | Количество residual-блоков |

```bash
# Быстрая маленькая сеть для отладки
python main.py train --channels 64 --res-blocks 4
```

### Самоигра

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--iterations` | 20 | Кол-во итераций train-loop |
| `--games-per-iteration` | 8 | Партий self-play за итерацию |
| `--simulations` | 64 | Симуляций MCTS на ход |
| `--max-moves` | 400 | Макс. полуходов в партии (~200 ходов) |
| `--buffer-size` | 50000 | Размер replay buffer (позиций) |
| `--mcts-batch-size` | 8 | Листьев за один forward pass в MCTS (больше = лучше GPU-утилизация) |

### Обучение

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--epochs` | 4 | Эпох на итерацию |
| `--batch-size` | 64 | Размер батча |
| `--learning-rate` | 0.001 | Learning rate (Adam) |

### Настройка наград

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--draw-score` | 0.45 | Ценность ничьей (0=плохо, 1=хорошо) |
| `--capture-reward-scale` | 0.01 | Бонус за взятие фигуры |
| `--capture-reward-cap` | 0.2 | Макс. бонус за взятие |
| `--allow-immediate-draws` | выкл. | Разрешить немедленные ничьи в self-play |

### Запись партий

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--record-games` | 2 | Сколько партий записывать в PGN за итерацию |
| `--records-dir` | `games/` | Куда писать PGN |

---

## Пример типичного запуска

```bash
# Полноценное обучение на GPU, 100 итераций, 16 партий/итерацию
python main.py train \
  --iterations 100 \
  --games-per-iteration 16 \
  --simulations 200 \
  --epochs 6 \
  --batch-size 256 \
  --buffer-size 100000
```

GPU определяется автоматически. При старте выводится:
```
device     | cuda: yes | name: NVIDIA GeForce RTX 3080 (sm_86)
model      | channels: 224 | res_blocks: 8 | params: 7,889,415
tensorboard| dir: runs/20240501_123456
```

---

## Игра против модели

### Текстовый режим (терминал)

```bash
python main.py play --checkpoint checkpoints/latest.pt --color white
```

### GUI (tkinter)

```bash
python main.py gui --checkpoint checkpoints/latest.pt --color white
```

Флаг `--color` — за какой цвет играет человек (`white` или `black`).  
`--simulations` — кол-во симуляций MCTS на ход модели (по умолчанию 128).

---

## Просмотр записанных партий

```bash
# Открыть последнюю записанную партию
python main.py pgn-viewer --latest

# Открыть конкретный файл
python main.py pgn-viewer --pgn games/my_game.pgn

# Авто-обновление раз в 5 секунд (следить за обучением)
python main.py pgn-viewer --latest --poll-seconds 5
```

Управление: кнопки **Back / Forward** для прокрутки ходов, **Reload latest** — обновить на последнюю партию.

---

## Структура проекта

```
alpha_zero/
├── main.py               # точка входа: train / play / gui / pgn-viewer
├── requirements.txt
├── model/
│   └── net.py            # AlphaZeroNet (ResNet 8×224)
├── mcts/
│   └── mcts.py           # Monte Carlo Tree Search
├── training/
│   ├── self_play.py      # генерация партий
│   ├── train.py          # один шаг обучения
│   └── buffer.py         # replay buffer
├── env/
│   └── chess_env.py      # обёртка над python-chess (encode_state и пр.)
├── utils/
│   └── fast_chess.py     # numba-ускорение
├── checkpoints/          # сохранённые веса (создаётся автоматически)
├── games/                # PGN партий (создаётся автоматически)
├── metrics/              # CSV с метриками (создаётся автоматически)
└── runs/                 # TensorBoard логи (создаётся автоматически)
```

---

## Оценка качества через Stockfish (опционально)

```bash
python main.py train \
  --eval-engine /path/to/stockfish \
  --eval-time 0.05 \
  --eval-games 1
```

Для каждой оцениваемой партии считаются `avg_accuracy` и `avg_cp_loss` (centipawn loss). Результаты пишутся в CSV и TensorBoard (`audit/` группа).

---

## Чекпоинт: что внутри

Каждый `.pt` файл содержит:

```python
{
    "model_state_dict": ...,      # веса модели
    "optimizer_state_dict": ...,  # состояние Adam
    "model_config": {             # архитектура (для автозагрузки)
        "input_channels": 19,
        "num_actions": 4672,
        "board_size": 8,
        "channels": 224,
        "num_res_blocks": 8,
    },
    "iteration": 42,
    "replay_buffer_size": 12800,
    "saved_at": "2024-05-01T12:34:56+00:00",
    "metadata": { ... },
}
```

При `--resume`, `play` или `gui` архитектура восстанавливается из `model_config` — не нужно передавать `--channels`/`--res-blocks` вручную.
