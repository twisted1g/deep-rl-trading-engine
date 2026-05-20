"""Entry point — parse CLI args and hand off to the engine."""
from __future__ import annotations

import argparse

import yaml
from dotenv import load_dotenv

from src.app import build_and_run, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once", action="store_true",
                        help="Run a single step immediately without waiting for a new bar")
    args = parser.parse_args()

    setup_logging()
    load_dotenv()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_and_run(cfg, once=args.once)


if __name__ == "__main__":
    main()
