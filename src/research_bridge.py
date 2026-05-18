"""Bootstrap import path к vendored research-коду.

Завендоренные модули лежат в `vendor/` (env, agents, encoders) и доступны
как top-level пакеты после вставки `vendor/` в sys.path.
"""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[1] / "vendor"

if not _VENDOR.exists():
    raise RuntimeError(
        f"vendor/ not found at {_VENDOR}. "
        "Файлы research-кода должны быть скопированы в ./vendor/."
    )

if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

# Eagerly import, чтобы упасть рано если что-то не так.
from env.trading_env_baseline import MyTradingEnv  # noqa: E402,F401
from env.trading_env_lstm import MyTradingEnvLSTM  # noqa: E402,F401
from agents.dueling_dqn_policy import DuelingDQNPolicy  # noqa: E402,F401
from encoders.lstm_pretrain import load_lstm_encoder  # noqa: E402,F401

RESEARCH_SRC = _VENDOR
