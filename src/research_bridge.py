"""Bootstrap import path к git submodule с research-репозиторием.

Делает доступными модули из external/DeepRLTradingResearch/src как top-level:
    from env.trading_env_baseline import MyTradingEnv
    from env.trading_env_lstm import MyTradingEnvLSTM
    from agents.dueling_dqn_policy import DuelingDQNPolicy
    from encoders.lstm_pretrain import load_lstm_encoder

Импортировать ОБЯЗАТЕЛЬНО до первого обращения к этим модулям.
"""
from __future__ import annotations

import sys
from pathlib import Path

_RESEARCH_SRC = (
    Path(__file__).resolve().parents[1] / "external" / "DeepRLTradingResearch" / "src"
)

if not _RESEARCH_SRC.exists():
    raise RuntimeError(
        f"Research submodule not found at {_RESEARCH_SRC}. "
        "Run: git submodule update --init --recursive"
    )

if str(_RESEARCH_SRC) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_SRC))

# Eagerly import, чтобы упасть рано если что-то не так с submodule.
from env.trading_env_baseline import MyTradingEnv  # noqa: E402,F401
from env.trading_env_lstm import MyTradingEnvLSTM  # noqa: E402,F401
from agents.dueling_dqn_policy import DuelingDQNPolicy  # noqa: E402,F401
from encoders.lstm_pretrain import load_lstm_encoder  # noqa: E402,F401

RESEARCH_SRC = _RESEARCH_SRC
