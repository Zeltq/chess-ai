# AlphaZero Chess

AlphaZero-style шахматный движок на PyTorch. Сеть — ResNet с настраиваемым размером (по умолчанию 224×8 ≈ 6.5M параметров). Self-play, MCTS с виртуальной потерей, FPU и tree-reuse, GPU-инференс с батчингом из нескольких процессов одновременно.

---

## Установка

```bash
cd alpha_zero
pip install -r requirements.txt
```

Требуется Python 3.10+. На Ampere и новее (RTX 30/40, A100, H100) автоматически включаются `bfloat16` autocast и `channels_last` память — отдельно ничего настраивать не нужно. На старых GPU откат на `float16` с GradScaler.

Все команды запускаются из папки `alpha_zero/`.

---

## Быстрый старт

Рекомендую для серьёзного прогона на одной GPU (~RTX 3060 и выше):

```bash
python main.py train \
  --iterations 200 \
  --games-per-iteration 24 \
  --simulations 200 \
  --max-moves 400 \
  --parallel-games 8 \
  --server-batch-size 256 \
  --mcts-batch-size 16 \
  --channels 128 \
  --res-blocks 8 \
  --buffer-size 50000 \
  --epochs 1 \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --draw-score 0.5 \
  --capture-reward-scale 0.005 \
  --temperature 1.0 \
  --temperature-drop-move 30 \
  --fpu-reduction 0.25 \
  --record-games 2 \
  --seed 42 \
  --time-limit-minutes 1500
```

После прогрева (5–10 секунд на старт воркеров в первой итерации) ожидать **60–80 позиций/сек** суммарно. Полная итерация ~1.5 мин. Чекпойнт после каждой итерации в `checkpoints/latest.pt`.

Если хочешь просто проверить что всё работает — короткая команда:

```bash
python main.py train --iterations 2 --games-per-iteration 4 \
  --simulations 32 --parallel-games 4 --channels 64 --res-blocks 2
```

---

## Обучение

### С нуля

См. **Быстрый старт** выше. Минимально:

```bash
python main.py train
```

Дефолты приоритезируют корректность, не скорость. Для реальных прогонов задай хотя бы `--parallel-games 8 --simulations 100`.

### Продолжить с чекпойнта

```bash
python main.py train --resume checkpoints/latest.pt
```

Архитектура сети, текущая итерация, состояние оптимизатора и LR-шедулера восстанавливаются автоматически. Флаги `--channels`/`--res-blocks` при `--resume` игнорируются — берутся из чекпойнта.

Чтобы сбросить LR-шедулер (например, перезапустить в режим "плато" с другим расписанием):

```bash
python main.py train --resume checkpoints/latest.pt --reset-lr \
  --learning-rate 5e-4 --lr-milestones 50 100
```

### Ограничить по времени

```bash
python main.py train --time-limit-minutes 240
```

Останавливается **после** ближайшего сохранённого чекпойнта по истечении времени. Безопасно прерывает.

### Воспроизводимость

```bash
python main.py train --seed 42
```

Cеет numpy/random/torch (включая CUDA seed). Полной побитовой воспроизводимости нет — CUDA-ядра non-deterministic — но run-to-run сравнения становятся осмысленными.

---

## Игра против модели

### Текстовый режим (UCI ввод в терминале)

```bash
python main.py play --checkpoint checkpoints/latest.pt --color white
```

Ходы вводятся в формате UCI: `e2e4`, `g1f3` и т.д.

### GUI (tkinter)

```bash
python main.py gui --checkpoint checkpoints/latest.pt --color white
```

Кликаешь свою фигуру → подсвечиваются легальные ходы зелёным → кликаешь поле назначения. Промоушен пешки автоматически в ферзя.

| Флаг | Значение | Описание |
|------|----------|----------|
| `--checkpoint` | обязательно | Путь к `.pt` файлу |
| `--color` | `white` | За какой цвет играет человек (`white` / `black`) |
| `--simulations` | 128 | MCTS sims на ход модели. Больше = сильнее, но дольше думает |

Сила игры приблизительно: 50 sims — слабовато, 200 — комфортный уровень, 800+ — серьёзная задержка.

---

## Просмотр записанных партий

```bash
# последняя сохранённая
python main.py pgn-viewer --latest

# конкретный файл
python main.py pgn-viewer --pgn games/20260508_iter_0042_game_01.pgn

# авто-обновление каждые 5 сек (следить за обучением)
python main.py pgn-viewer --latest --poll-seconds 5
```

Кнопки **Back / Forward** прокручивают ходы. **Reload latest** — переключиться на самую свежую партию.

---

## TensorBoard

```bash
# из папки alpha_zero/
tensorboard --logdir runs
```

Открыть: [http://localhost:6006](http://localhost:6006)

### Что мониторить

**Скорость / утилизация GPU:**
- `inference/avg_batch_size` — средний размер батча на сервере. Целевое ≥80 при `parallel-games 8`. Если 30 — увеличь `--mcts-batch-size`.
- `inference/max_batch_size` — пиковый. Должен достигать `--server-batch-size`.
- `inference/positions_per_iter` — позиций инферилось за итерацию.
- `selfplay/speed_pos_per_s` — общий throughput (включает train).

**Качество MCTS:**
- `mcts/avg_reused_visits` — сколько визитов перенесено через tree-reuse. Со временем растёт. Целевое ~30–50% от `--simulations`.

**Обучение:**
- `loss/total`, `loss/policy`, `loss/value` — каждый шаг.
- `train/avg_loss`, `train/avg_policy_loss`, `train/avg_value_loss` — средние за итерацию.
- `lr` — learning rate.

**Self-play:**
- `selfplay/white_wins`, `black_wins`, `draws` — по итерациям. Пока модель слабая будет много ничьих по 50-move rule.
- `selfplay/positions`, `selfplay/buffer_size`.

**Аудит (если включён `--eval-engine`):**
- `audit/accuracy`, `audit/cp_loss` — по сравнению с внешним движком.

Отключить TensorBoard:
```bash
python main.py train --no-tensorboard
```

---

## Архитектура: как выглядит параллелизм

При `--parallel-games > 1` поднимается:

```
            Main process (GPU)
            ┌─────────────────────────────┐
            │ Model + MPInferenceServer   │◄──┐
            │ (drains request queue,      │   │ numpy arrays through
            │  batches, runs forward,     │   │ multiprocessing.Queue
            │  fans out responses)        │   │
            └─────────────┬───────────────┘   │
                          │ response queues   │
                          ▼                   │
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │ worker 0 │ │ worker 1 │ │ worker N │
            │  (own    │ │  (own    │ │  (own    │
            │  GIL,    │ │  GIL,    │ │  GIL,    │
            │  MCTS,   │ │  MCTS,   │ │  MCTS,   │
            │  encode) │ │  encode) │ │  encode) │
            └──────────┘ └──────────┘ └──────────┘
```

Каждая партия в своём процессе → нет GIL-конкуренции. Модель в GPU только одна — ферст forward batch собирается из запросов всех воркеров одновременно. Веса обновляются in-place в главном процессе → воркеры видят свежие веса автоматически (без broadcast).

При `--parallel-games 1` многопоточности нет, всё в одном процессе через `LocalEvaluator`.

---

## Полный справочник флагов

### Архитектура сети

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--channels` | 224 | Ширина ResNet (каналы) |
| `--res-blocks` | 8 | Количество residual-блоков |

```bash
# маленькая быстрая сеть для отладки
python main.py train --channels 64 --res-blocks 4
```

### Self-play

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--iterations` | 20 | Кол-во итераций (self-play + train) |
| `--games-per-iteration` | 8 | Партий self-play за итерацию |
| `--simulations` | 64 | MCTS sims на ход |
| `--max-moves` | 400 | Макс. полуходов в партии (~200 ходов) |
| `--parallel-games` | 1 | Сколько партий в параллель (отдельные процессы) |
| `--mcts-batch-size` | 8 | Листьев на один forward в MCTS |
| `--server-batch-size` | 256 | Макс. батч у inference-сервера |
| `--server-max-wait-ms` | 2.0 | Окно ожидания на батчинг до flush |
| `--temperature` | 1.0 | Температура сэмплирования по visit-counts |
| `--temperature-drop-move` | 30 | После какого полухода температура → 0 (greedy) |
| `--fpu-reduction` | 0.25 | First-Play Urgency: Q непосещённого = parent.value − fpu_reduction |
| `--c-puct-base` | None | Включить log-c_puct (см. AZ paper). Полезно при sims ≥ 400 |
| `--c-puct-init` | 1.25 | Init term для log-c_puct |
| `--record-games` | None (все) | Сколько PGN записывать на итерацию. Поставь 1–2 чтобы не забивать диск |

### Сигнал и shaping

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--draw-score` | 0.45 | Ценность ничьей (0=строго плохо, 0.5=нейтрально, 1=строго хорошо). AZ paper: 0.5 |
| `--capture-reward-scale` | 0.01 | Бонус за взятие. Помогает на ранних итерациях, потом мешает — поставь `0.005` или `0` |
| `--capture-reward-cap` | 0.2 | Макс. бонус |
| `--capture-reward-ramp-iterations` | iterations//3 | За сколько итераций линейно сбросить shaping в 0 |

### Replay и обучение

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--buffer-size` | 50000 | Размер replay buffer (позиций) |
| `--mirror-augment` | выкл. | 50% шанс отдать зеркальный sample. ~2× данных, цена — небольшой distribution shift на позициях с рокировкой |
| `--epochs` | 4 | Эпох на итерацию (обычно достаточно 1) |
| `--batch-size` | 64 | Размер батча обучения |
| `--learning-rate` | 1e-3 | Learning rate (Adam) |
| `--weight-decay` | 1e-4 | L2 регуляризация |
| `--lr-milestones` | 50%/75% от iterations | Итерации, на которых LR умножается на factor |
| `--lr-decay-factor` | 0.1 | Множитель LR на milestone |

### Resume / воспроизводимость

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--resume` | — | Путь к `.pt` файлу для продолжения |
| `--reset-lr` | выкл. | Игнорировать LR-шедулер из чекпойнта, начать заново с `--learning-rate` |
| `--seed` | None | Seed для numpy / random / torch |

### Аудит через UCI-движок

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--eval-engine` | — | Путь к UCI-движку (например, Stockfish) |
| `--eval-time` | 0.05 | Сек на ход для движка |
| `--eval-games` | 1 | Сколько партий аудитить за итерацию |

```bash
python main.py train \
  --eval-engine /usr/games/stockfish \
  --eval-time 0.05 \
  --eval-games 1
```

Считает `avg_accuracy` и `avg_cp_loss` по логике chess.com. Пишет в CSV/TB.

### Пути / выходы

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--checkpoint-dir` | `checkpoints/` | Куда писать `.pt` |
| `--records-dir` | `games/` | Куда писать PGN |
| `--metrics-dir` | `metrics/` | Куда писать CSV |
| `--metrics-file` | автоген | Конкретный CSV (переопределяет `--metrics-dir`) |
| `--tensorboard-dir` | `runs/` | Где TB-логи |
| `--no-tensorboard` | — | Не запускать TB writer |
| `--no-save-buffer` | — | Не сохранять `buffer.npz` рядом с чекпойнтом |
| `--time-limit-minutes` | None | Стоп после следующего чекпойнта по времени |

---

## Куда что сохраняется

| Что | Путь по умолчанию | Описание |
|-----|-------------------|----------|
| Чекпойнты | `checkpoints/` | `<run_id>_alpha_zero_chess_iter_XXXX.pt` + `latest.pt` |
| Replay buffer | `checkpoints/buffer.npz` | Восстанавливается при `--resume` если рядом с чекпойнтом |
| Партии (PGN) | `games/` | Сколько задано в `--record-games` |
| Метрики (CSV) | `metrics/` | Лосс, W/B/D, accuracy |
| TensorBoard | `runs/<run_id>/` | По одному подкаталогу на запуск |

---

## Тесты

Из `alpha_zero/`:

```bash
python -m pytest tests/ -q
```

Покрывают:
- инвариантность кодирования (move↔action round-trip, mirror involution),
- регрессии MCTS (visit accounting, virtual-loss balance, tree-reuse, mate-in-1).

---

## Бенчмарк

Замеры hot-paths без полного прогона:

```bash
# по умолчанию: encode + forward + selfplay
python bench.py --channels 128 --res-blocks 8

# только пропускная способность self-play
python bench.py --section selfplay --channels 128 --res-blocks 8 \
                --parallel 8 --sims 100 --max-moves 60
```

---

## Структура проекта

```
alpha_zero/
├── main.py                  # точка входа: train / play / gui / pgn-viewer
├── bench.py                 # микробенчмарки
├── requirements.txt
├── env/
│   └── chess_env.py         # python-chess обёртка, encode_state (bitboards)
├── model/
│   └── net.py               # AlphaZeroNet (ResNet)
├── mcts/
│   ├── mcts.py              # MCTS с virtual-loss, tree-reuse, FPU
│   ├── node.py              # Node с параллельными массивами children
│   └── inference.py         # LocalEvaluator + MPInferenceServer/Client
├── training/
│   ├── self_play.py         # генерация партий
│   ├── train.py             # шаг обучения
│   ├── buffer.py            # replay buffer (+ mirror augment)
│   └── augment.py           # mirror state/policy
├── utils/
│   └── fast_chess.py        # numba: action encoding, нормализация
├── tests/                   # pytest
├── checkpoints/             # сохранённые веса (создаётся автоматически)
├── games/                   # PGN партий
├── metrics/                 # CSV
└── runs/                    # TensorBoard логи
```

---

## Чекпойнт: что внутри

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "model_config": {
        "input_channels": 19,
        "num_actions": 4672,
        "board_size": 8,
        "channels": 128,
        "num_res_blocks": 8,
    },
    "iteration": 42,
    "replay_buffer_size": 12800,
    "saved_at": "2026-05-08T22:34:56+00:00",
    "metadata": {
        "run_id": "...", "lr_milestones": [...],
        "engine_eval": {"average_accuracy": ..., "average_cp_loss": ...},
        ...
    },
}
```

`--resume`, `play`, `gui` восстанавливают архитектуру из `model_config` — `--channels`/`--res-blocks` указывать не нужно.

---

## Типичный рабочий процесс

1. Запустить **Быстрый старт**.
2. В отдельном терминале:
   ```bash
   tensorboard --logdir runs
   ```
3. Через 5–10 итераций:
   ```bash
   python main.py pgn-viewer --latest --poll-seconds 5
   ```
   — смотреть свежие партии прямо в процессе.
4. Когда модель заметно подучилась (loss/policy падает ниже ~3.0, начали появляться победы):
   ```bash
   python main.py gui --checkpoint checkpoints/latest.pt --simulations 200
   ```
   — поиграть против неё.
5. Если прервался — `--resume checkpoints/latest.pt`.
