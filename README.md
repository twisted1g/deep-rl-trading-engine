# deeprl-trading-engine

Инференс RL-агентов, обученных в [DeepRLTradingResearch](https://github.com/twisted1g/DeepRLTradingResearch),
на Binance Futures USDT-M (testnet/live).

Поддерживает все 4 алгоритма из research-репа (A2C / PPO / DQN / Dueling DQN)
и оба state space (`baseline` 6D и `lstm` 64D).

## Структура

```
external/DeepRLTradingResearch/   # git submodule с research-кодом
src/
  research_bridge.py    # bootstrap sys.path к submodule
  observation.py        # building obs через настоящие env-классы research-репа
  model_loader.py       # SB3 model.zip + vecnorm.pkl, с поддержкой Dueling DQN
  exchange.py           # обёртка над python-binance (фьючерсы)
  trader.py             # 1h bar-close loop, forced exits, % equity sizing
run.py
config.yaml
```

## Установка

```bash
git clone --recurse-submodules <repo>
# или, если уже склонировано без submodule:
git submodule update --init --recursive

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # вставь testnet-ключи: https://testnet.binancefuture.com
```

## Конфигурация (`config.yaml`)

Главное — указать **директорию эксперимента** из research-репа:

```yaml
model:
  experiment_dir: ../DeepRLTradingResearch/experiments/a2c-baseline/20260517-153458-f0cb0e
  algo: a2c            # a2c | ppo | dqn | dueling_dqn
  state_space: baseline  # baseline | lstm
```

Эта папка должна содержать `model.zip` (SB3) и `vecnorm.pkl` (VecNormalize stats).
Оба сохраняются автоматически тренировочными скриптами research-репа.

Для `state_space: lstm` также указать путь к чекпоинту энкодера
(он gitignored в research-репо, поэтому submodule его не подтянет — указывай
путь к своему рабочему чекауту):

```yaml
model:
  lstm_checkpoint_path: ../DeepRLTradingResearch/src/encoders/lstm_encoder.pt
```

Параметры env (`feature_window`, `lstm_window_size`, …) и торговли
(`max_holding_time`, `max_drawdown_threshold`) **должны совпадать** с обучением.
Дефолты в `config.yaml` соответствуют дефолтам в research-репо.

## Запуск

```bash
python run.py --once     # один step сразу, для проверки
python run.py            # бесконечный цикл: ждёт bar-close → решает → исполняет
```

## Семантика действий и позиций

Согласовано с `MyTradingEnv.ACTION_TO_POSITION` из research-репо:
- `0` → flat (закрыть)
- `1` → long
- `2` → short

Сайзинг — 100% доступного USDT (`equity_fraction: 1.0`), как в обучении.
На testnet можно снизить до 0.1–0.5, чтобы не упереться в лимиты.

## Forced exits (зеркало training-среды)

Агент обучался в среде, которая принудительно закрывала позицию при:
- `holding_time >= 72` (72 часовых бара)
- `unrealized_drawdown >= 8%`

Эти же правила применяются в live-режиме — иначе поведение в инференсе
систематически расходится с тренировкой. Параметры в `config.yaml`,
`trading.max_holding_time` / `trading.max_drawdown_threshold`.

## Тайминг

Цикл просыпается, ждёт закрытия 1h бара (+ `bar_close_grace_seconds` буфера),
вытаскивает свежие свечи (отбрасывая текущую незакрытую), строит observation,
зовёт `model.predict(...)` и исполняет переход позиций одним market-ордером.

## Переход на mainnet

В `config.yaml`: `exchange.testnet: false`, mainnet-ключи в `.env`.
Сначала **обязательно** прогнать на testnet и убедиться, что переходы
позиций и размеры ордеров соответствуют ожиданиям.
