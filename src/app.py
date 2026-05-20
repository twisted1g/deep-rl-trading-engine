"""Engine assembly and startup from a config dict."""
from __future__ import annotations

import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .core.observation import ObservationBuilder
from .core.trader import Trader, TraderConfig
from .exchange.wallet import from_env
from .model.loader import load_model_and_vecnorm
from .notifications.notifier import TelegramNotifier
from .persistence.journal import TradeJournal

log = logging.getLogger(__name__)


def setup_logging(log_dir: Optional[str] = "state/logs") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(
            Path(log_dir) / "engine.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        ))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def build_and_run(cfg: dict, once: bool = False) -> None:
    """Assemble all components from *cfg* and start the trading engine."""
    obs_builder = ObservationBuilder(
        state_space=cfg["model"]["state_space"],
        feature_window=cfg["env"]["feature_window"],
        lstm_window_size=cfg["env"].get("lstm_window_size", 128),
        lstm_hidden_size=cfg["env"].get("lstm_hidden_size", 64),
        lstm_layers=cfg["env"].get("lstm_layers", 2),
        lstm_checkpoint_path=cfg["model"].get("lstm_checkpoint_path"),
        lstm_device=cfg["model"].get("lstm_device", "cpu"),
    )

    exp_dir = Path(cfg["model"]["experiment_dir"]).expanduser().resolve()
    model, vec_env = load_model_and_vecnorm(
        algo=cfg["model"]["algo"],
        model_path=exp_dir / "model.zip",
        vecnorm_path=exp_dir / "vecnorm.pkl",
        obs_dim=obs_builder.obs_dim,
    )

    ex = from_env(
        symbol=cfg["exchange"]["symbol"],
        testnet=cfg["exchange"]["testnet"],
        leverage=cfg["exchange"]["leverage"],
        margin_type=cfg["exchange"]["margin_type"],
    )

    trader_cfg = TraderConfig(
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

    trader = Trader(ex, obs_builder, model, vec_env, trader_cfg,
                    notifier=notifier, journal=journal)

    notifier.notify_startup(
        f"{trader_cfg.symbol} {trader_cfg.interval} | algo={cfg['model']['algo']} "
        f"state={cfg['model']['state_space']} testnet={cfg['exchange']['testnet']}"
    )

    if (
        not once
        and tg_cfg.get("commands_enabled", False)
        and tg_cfg.get("enabled", False)
        and os.environ.get("TELEGRAM_BOT_TOKEN")
        and (tg_cfg.get("chat_id") or os.environ.get("TELEGRAM_CHAT_ID"))
    ):
        from .notifications.bot import CommandBot
        bot = CommandBot(
            token=os.environ["TELEGRAM_BOT_TOKEN"],
            chat_id=int(tg_cfg.get("chat_id") or os.environ["TELEGRAM_CHAT_ID"]),
            trader=trader,
            journal=journal,
        )
        bot.start()

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

    if once:
        trader.step()
    else:
        trader.run_forever()
