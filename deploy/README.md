# Deploy

## Docker compose

```bash
git submodule update --init --recursive
cp .env.example .env  # заполнить Binance + Telegram ключи
docker compose build
docker compose up -d
docker compose logs -f
```

State (SQLite + логи) лежит в `./state/` — переживает рестарт.

## systemd (опционально, для bare-metal сервера)

```bash
sudo cp deploy/deeprl-engine.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now deeprl-engine
journalctl -u deeprl-engine -f
```

Подразумевается, что репо лежит в `/opt/deeprl-trading-engine` (или поправьте `WorkingDirectory` в unit-файле).

## Обновление

```bash
git pull
git submodule update --recursive
docker compose build
docker compose up -d
```

## Telegram

1. Создайте бота через @BotFather → токен в `.env` (`TELEGRAM_BOT_TOKEN`).
2. Напишите боту `/start`, узнайте свой chat id (например через @userinfobot) → `.env` (`TELEGRAM_CHAT_ID`).
3. `telegram.enabled: true` и `telegram.commands_enabled: true` в `config.yaml`.

Команды бота: `/status /pnl /trades [N] /pause /resume /close`.
