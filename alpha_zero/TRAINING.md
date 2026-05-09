# Training playbook

Конфигурации обучения с обоснованиями. Не общий справочник по флагам
(он в [`README.md`](../README.md)) — а **проверенные/рекомендованные комбинации**
с расчётами времени и компромиссами.

---

## Config A: Baseline 128×8 + 200 sims (доказал работоспособность)

Проверено: до iter ~50 модель учится осмысленно, начинает ловить мат-в-1,
забирает свободные фигуры. Достигает **~1200–1500 chess.com** при ~10–15 ч
тренировки. Дальше упирается в потолок архитектуры.

```bash
python main.py train \
  --iterations 200 \
  --games-per-iteration 24 \
  --simulations 200 \
  --max-moves 300 \
  --parallel-games 8 \
  --server-batch-size 256 \
  --mcts-batch-size 32 \
  --channels 128 \
  --res-blocks 8 \
  --buffer-size 50000 \
  --epochs 1 \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --draw-score 0.4 \
  --capture-reward-scale 0.02 \
  --capture-reward-cap 0.3 \
  --capture-reward-ramp-iterations 120 \
  --adjudicate-material 5 \
  --adjudicate-min-move 40 \
  --adjudicate-consecutive 4 \
  --temperature 1.0 \
  --temperature-drop-move 30 \
  --fpu-reduction 0.25 \
  --record-games 2 \
  --seed 42 \
  --time-limit-minutes 1500
```

**Параметры модели:** ~3.0M.
**Throughput:** ~70 pos/s суммарно (RTX 3060).
**Per iter:** ~30 сек self-play + ~3 сек train = ~33 сек.
**200 итераций:** ~110 минут (1.8 ч).

---

## Config B: Scaled-up 224×8 + 300 sims (~7.9M параметров)

Цель: пробить потолок 128×8 и дойти до **~1700–2000 chess.com**.

Архитектура совпадает с C++ референс-реализацией в репозитории.

```bash
rm -rf checkpoints/ games/ metrics/ runs/

python main.py train \
  --iterations 800 \
  --games-per-iteration 24 \
  --simulations 300 \
  --max-moves 300 \
  --parallel-games 8 \
  --server-batch-size 256 \
  --server-max-wait-ms 2 \
  --mcts-batch-size 32 \
  --channels 224 \
  --res-blocks 8 \
  --buffer-size 80000 \
  --epochs 1 \
  --batch-size 256 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --lr-milestones 400 600 \
  --lr-decay-factor 0.1 \
  --draw-score 0.4 \
  --capture-reward-scale 0.02 \
  --capture-reward-cap 0.3 \
  --capture-reward-ramp-iterations 400 \
  --adjudicate-material 5 \
  --adjudicate-min-move 40 \
  --adjudicate-consecutive 4 \
  --temperature 1.0 \
  --temperature-drop-move 30 \
  --fpu-reduction 0.25 \
  --record-games 2 \
  --seed 42 \
  --time-limit-minutes 1500
```

**Параметры модели:** ~7.9M (в 2.6× больше Config A).
**Throughput:** ~30–40 pos/s суммарно.
**Per iter:** ~50 сек self-play + ~5 сек train = ~55 сек.
**800 итераций:** ~12 часов.

---

## Что отличается от Config A и почему

| Флаг | A (128×8) | B (224×8) | Почему |
|---|---|---|---|
| `--channels` | 128 | **224** | Больше capacity — потолок ELO выше. Forward в ~1.5× медленнее, но на batch=128+ GPU не голодает. |
| `--res-blocks` | 8 | 8 | Глубина та же — широкая сеть для шахмат эффективнее глубокой при равных параметрах. |
| `--simulations` | 200 | **300** | На обученной сети больше sims линейно повышают качество policy targets. 300 — sweet spot для домашних AZ-репликаций (Lc0 small models). |
| `--iterations` | 200 | **800** | Большая модель учится дольше (больше параметров → больше нужно градиентных шагов). |
| `--lr-milestones` | дефолт (50%/75%) | **400 600** | Декей на 50% и 75% от 800. До iter 400 LR полный 1e-3. |
| `--buffer-size` | 50000 | **80000** | Больше партий — больше данных в буфере → больше разнообразия для большой сети. |
| `--capture-reward-ramp-iterations` | 120 | **400** | Растянули ramp в 3.3× — на 800 итераций shaping должен помогать дольше (1/2 от total). |
| `--server-max-wait-ms` | (default 2) | **2** | Явно укажи — на большом model + sims критично, чтобы воркеры не простаивали. |

**Что НЕ меняем:**
- `--draw-score 0.4`, adjudication, temperature, fpu — те же. Это правила игры, не зависят от архитектуры.
- `--mcts-batch-size 32` — тот же. С 8 воркерами это даёт server avg batch ~100–150 (хорошая утилизация).
- `--epochs 1`, `--batch-size 256` — золотой стандарт AZ. Меняй только если знаешь зачем.

---

## Memory check (RTX 3060 12 GB)

Для 224×8 с batch 256 во время train:
- Параметры: 7.9M × 4 байта (FP32) = 32 MB
- Параметры в bfloat16 для forward: 16 MB
- Adam state (momentum + variance): 2 × 32 MB = 64 MB
- Активации batch=256, 224 каналов, 8×8 = 256 × 224 × 64 × 8 ≈ 30 MB на блок × 16 (forward+backward) = 480 MB
- Inference server при peak batch=256: те же ~480 MB активаций

Итого peak: **~600–800 MB**. На 3060 (12 GB) комфортно.

---

## Что мониторить в TensorBoard на Config B

Контрольные точки:

| Iter | loss/policy | loss/value | W/B/D на 24 партии | Что проверить |
|---|---|---|---|---|
| 1 | ~5.5 | ~0.6 | 0/0/24 (пока шум) | server batch ≥80 |
| 30 | ~3.5 | ~0.3 | 3/2/19 | **GUI: модель должна забирать свободного ферзя** |
| 80 | ~2.8 | ~0.2 | 6/5/13 | **GUI: ставит мат-в-1 надёжно** |
| 200 | ~2.4 | ~0.15 | 8/8/8 | **GUI: разумный дебют (1.e4 e5 2.Nf3 Nc6 ...)** |
| 400 | ~2.1 | ~0.12 | 10/9/5 | LR падает до 1e-4. Свободные пешки не зевает |
| 600 | ~1.9 | ~0.10 | 11/10/3 | LR падает до 1e-5. Финальная фаза |
| 800 | ~1.7 | ~0.08 | 12/11/1 | Готово. Ожидаемый уровень: 1700–2000 chess.com |

Если на каком-то этапе **loss подскочил вверх** и не возвращается за 10 итераций
— значит что-то пошло не так. Откатывайся на чекпойнт N−10 и думай что
изменилось.

---

## Чего НЕЛЬЗЯ менять во время Config B

Это правило универсальное (записано в README), но повторю:

| Безопасно поменять на лету | НЕЛЬЗЯ без `--reset-lr` |
|---|---|
| `--parallel-games`, `--mcts-batch-size`, `--server-*` | **`--simulations`** (классическая ловушка) |
| `--buffer-size`, `--max-moves`, `--record-games` | `--temperature*`, `--draw-score`, `--capture-reward-*` |
| `--epochs`, `--time-limit-minutes` | `--c-puct*`, `--fpu-reduction` |
| `--adjudicate-*` (умеренно) | `--batch-size` (Adam state по-разному ведёт себя на разных batch) |

**При изменении параметра из правой колонки** обязательно:
```bash
... --resume checkpoints/latest.pt --reset-lr --learning-rate 3e-4 ...
```
Пониженный LR (3e-4 вместо 1e-3) даст Adam'у мягко перестроиться.

---

## Если 224×8 не дотянет до 1800

Запасной план — ещё больше:

**Config C: 256×10 (~13M параметров, ~24h обучения):**
```bash
... --channels 256 --res-blocks 10 --simulations 400 ...
... --iterations 1200 --lr-milestones 600 900 ...
... --buffer-size 100000 --capture-reward-ramp-iterations 600 ...
... --time-limit-minutes 1700 ...
```

Расчёт: 256×10 = ~14M параметров. Forward ~12-15ms на batch=128. 400 sims × 8
воркеров × 24 games × 80 ходов / 25 pos/s ≈ 75 сек self-play на iter. 1200
iter ≈ 25 часов. Ожидаемый уровень: **2000+ chess.com**.

**ВАЖНО:** между Config B и Config C нельзя сделать `--resume` — архитектура
не переносится. Запускай C **с нуля** или после полного Config B как
независимый эксперимент.

---

## Запуск с полной чистоты

```bash
cd /mnt/d/Codes/chess-ai/alpha_zero
source ../.venv/bin/activate
rm -rf checkpoints/ games/ metrics/ runs/
# вставь команду Config B (или A, или C) сюда
```

В отдельном терминале:
```bash
tensorboard --logdir runs
# открой http://localhost:6006
```

Через 2 итерации (~2 минуты после старта) проверь TB:
- `inference/avg_batch_size` ≥ 80 → MP-сервер работает
- `loss/policy` начинает падать
- `selfplay/draws` пока ~24/24 (норм для iter 1-3)

Через 30 итераций — обязательно тест в GUI на свободного ферзя:
```bash
python main.py gui --checkpoint checkpoints/latest.pt --color black --simulations 400
```
