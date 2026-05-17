from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.agent import SB3Agent
from src.exchange import from_env
from src.trader import Trader, TraderConfig


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once", action="store_true", help="Сделать один шаг и выйти")
    args = parser.parse_args()

    setup_logging()
    load_dotenv()
    cfg = load_config(args.config)

    ex = from_env(
        symbol=cfg["exchange"]["symbol"],
        testnet=cfg["exchange"]["testnet"],
        leverage=cfg["exchange"]["leverage"],
        margin_type=cfg["exchange"]["margin_type"],
    )

    agent = SB3Agent(
        model_path=cfg["model"]["path"],
        deterministic=cfg["model"]["deterministic"],
    )

    tcfg = TraderConfig(
        symbol=cfg["exchange"]["symbol"],
        interval=cfg["exchange"]["interval"],
        window_size=cfg["trading"]["window_size"],
        poll_seconds=cfg["trading"]["poll_seconds"],
        order_quantity=cfg["trading"]["order_quantity"],
        action_space=cfg["trading"]["action_space"],
        max_position=cfg["trading"]["max_position"],
        feature_indicators=cfg["features"]["indicators"],
        feature_normalize=cfg["features"]["normalize"],
        use_ohlcv=cfg["features"]["ohlcv"],
    )

    trader = Trader(ex, agent, tcfg)

    if args.once:
        trader.step()
    else:
        trader.run_forever()


if __name__ == "__main__":
    main()
