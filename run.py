from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

from typing import Optional

import yaml
from dotenv import load_dotenv

from src.exchange import from_env
from src.journal import TradeJournal
from src.model_loader import load_model_and_vecnorm
from src.notifier import TelegramNotifier
from src.observation import ObservationBuilder
from src.trader import Trader, TraderConfig


def setup_logging(log_dir: Optional[str] = "state/logs") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        from logging.handlers import RotatingFileHandler
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(
            Path(log_dir) / "engine.log",
            maxBytes=10 * 1024 * 1024, backupCount=5,
        ))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once", action="store_true",
                        help="Один шаг сразу, без ожидания нового бара")
    args = parser.parse_args()

    setup_logging()
    load_dotenv()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    state_space = cfg["model"]["state_space"]
    obs_builder = ObservationBuilder(
        state_space=state_space,
        feature_window=cfg["env"]["feature_window"],
        lstm_window_size=cfg["env"].get("lstm_window_size", 128),
        lstm_hidden_size=cfg["env"].get("lstm_hidden_size", 64),
        lstm_layers=cfg["env"].get("lstm_layers", 2),
        lstm_checkpoint_path=cfg["model"].get("lstm_checkpoint_path"),
        lstm_device=cfg["model"].get("lstm_device", "cpu"),
    )

    obs_dim = obs_builder.obs_dim

    exp_dir = Path(cfg["model"]["experiment_dir"]).expanduser().resolve()
    model, vec_env = load_model_and_vecnorm(
        algo=cfg["model"]["algo"],
        model_path=exp_dir / "model.zip",
        vecnorm_path=exp_dir / "vecnorm.pkl",
        obs_dim=obs_dim,
    )

    ex = from_env(
        symbol=cfg["exchange"]["symbol"],
        testnet=cfg["exchange"]["testnet"],
        leverage=cfg["exchange"]["leverage"],
        margin_type=cfg["exchange"]["margin_type"],
    )

    tcfg = TraderConfig(
        symbol=cfg["exchange"]["symbol"],
        interval=cfg["exchange"]["interval"],
        equity_fraction=cfg["trading"]["equity_fraction"],
        leverage=cfg["exchange"]["leverage"],
        max_holding_time=cfg["trading"]["max_holding_time"],
        max_drawdown_threshold=cfg["trading"]["max_drawdown_threshold"],
        poll_seconds=cfg["trading"]["poll_seconds"],
        bar_close_grace_seconds=cfg["trading"]["bar_close_grace_seconds"],
    )

    tg_cfg = cfg.get("telegram", {}) or {}
    notifier = TelegramNotifier.from_env(
        enabled=tg_cfg.get("enabled", False),
        chat_id=tg_cfg.get("chat_id"),
    )

    persistence_cfg = cfg.get("persistence", {}) or {}
    journal = TradeJournal(persistence_cfg.get("db_path", "state/trades.db"))

    trader = Trader(ex, obs_builder, model, vec_env, tcfg,
                    notifier=notifier, journal=journal)
    notifier.notify_startup(
        f"{tcfg.symbol} {tcfg.interval} | algo={cfg['model']['algo']} "
        f"state={cfg['model']['state_space']} testnet={cfg['exchange']['testnet']}"
    )

    if (
        tg_cfg.get("commands_enabled", False)
        and tg_cfg.get("enabled", False)
        and os.environ.get("TELEGRAM_BOT_TOKEN")
        and (tg_cfg.get("chat_id") or os.environ.get("TELEGRAM_CHAT_ID"))
        and not args.once
    ):
        from src.tg_bot import CommandBot
        bot = CommandBot(
            token=os.environ["TELEGRAM_BOT_TOKEN"],
            chat_id=int(tg_cfg.get("chat_id") or os.environ["TELEGRAM_CHAT_ID"]),
            trader=trader,
            journal=journal,
        )
        bot.start()

    log = logging.getLogger(__name__)

    def _on_sigterm(signum, _frame):
        log.warning("Received signal %d — graceful shutdown", signum)
        if cfg["trading"].get("close_on_shutdown", False):
            try:
                trader.force_close()
            except Exception:
                log.exception("close_on_shutdown failed")
        notifier.notify_shutdown(f"(signal {signum})")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    if args.once:
        trader.step()
    else:
        trader.run_forever()


if __name__ == "__main__":
    main()
