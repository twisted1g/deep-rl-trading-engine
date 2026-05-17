from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.exchange import from_env
from src.model_loader import load_model_and_vecnorm
from src.observation import ObservationBuilder
from src.trader import Trader, TraderConfig


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
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

    obs_dim = (
        6 if state_space == "baseline" else cfg["env"].get("lstm_hidden_size", 64)
    )

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

    trader = Trader(ex, obs_builder, model, vec_env, tcfg)
    if args.once:
        trader.step()
    else:
        trader.run_forever()


if __name__ == "__main__":
    main()
