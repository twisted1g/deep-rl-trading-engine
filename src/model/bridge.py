"""Bootstrap vendor/ research code into sys.path.

vendor/ (env, agents, encoders) is added as a top-level import path.
Import this module before any vendor code is used; it fails early if
the directory is missing or a required module can't be loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[2] / "vendor"

if not _VENDOR.exists():
    raise RuntimeError(
        f"vendor/ not found at {_VENDOR}. "
        "Research code must be copied into ./vendor/."
    )

if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from env.trading_env_baseline import MyTradingEnv  # noqa: E402,F401
from env.trading_env_lstm import MyTradingEnvLSTM  # noqa: E402,F401
from agents.dueling_dqn_policy import DuelingDQNPolicy  # noqa: E402,F401
from encoders.lstm_pretrain import load_lstm_encoder  # noqa: E402,F401

RESEARCH_SRC = _VENDOR
