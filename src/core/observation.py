"""Build observations for live inference — mirrors the research environment exactly."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from ..model import bridge  # noqa: F401 — bootstrap vendor sys.path
from env.trading_env_baseline import MyTradingEnv
from env.trading_env_lstm import MyTradingEnvLSTM


StateSpace = Literal["baseline", "lstm"]


class ObservationBuilder:
    def __init__(
        self,
        state_space: StateSpace,
        feature_window: int = 20,
        lstm_window_size: int = 128,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 2,
        lstm_checkpoint_path: Optional[str] = None,
        lstm_device: str = "cpu",
    ):
        self.state_space = state_space
        self.feature_window = int(feature_window)
        self.lstm_window_size = int(lstm_window_size)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.lstm_layers = int(lstm_layers)
        self.lstm_device = lstm_device

        self._lstm_encoder = None
        self._lstm_layernorm = None
        if state_space == "lstm":
            if lstm_checkpoint_path is None:
                raise ValueError("lstm_checkpoint_path required for state_space=lstm")
            from encoders.lstm_pretrain import load_lstm_encoder

            ckpt = Path(lstm_checkpoint_path).expanduser().resolve()
            if not ckpt.exists():
                raise FileNotFoundError(f"LSTM checkpoint not found: {ckpt}")
            self._lstm_encoder, self._lstm_layernorm = load_lstm_encoder(
                str(ckpt), device=lstm_device
            )

    @property
    def obs_dim(self) -> int:
        if self.state_space == "baseline":
            return 9
        return self.lstm_hidden_size + 4

    def min_history(self) -> int:
        if self.state_space == "baseline":
            return self.feature_window + 2
        return self.lstm_window_size + self.feature_window + 2

    def build(self, df: pd.DataFrame, position: int) -> np.ndarray:
        """df: OHLCV candles, last row = most recent closed bar."""
        if "close" not in df.columns or "volume" not in df.columns:
            raise ValueError("df must contain 'close' and 'volume' columns")
        if len(df) < self.min_history():
            raise RuntimeError(
                f"Not enough history: have {len(df)}, need {self.min_history()}"
            )

        df_reset = df.reset_index(drop=True)

        if self.state_space == "baseline":
            env = MyTradingEnv(df=df_reset, feature_window=self.feature_window)
            env.current_step = len(df_reset) - 1
            env.position = int(position)
            return env._get_observation().astype(np.float32)

        env = MyTradingEnvLSTM(
            df=df_reset,
            feature_window=self.feature_window,
            lstm_window_size=self.lstm_window_size,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_layers=self.lstm_layers,
            lstm_encoder=self._lstm_encoder,
            lstm_device=self.lstm_device,
        )
        if self._lstm_layernorm is not None:
            env.lstm_layernorm = self._lstm_layernorm
        env.current_step = len(df_reset) - 1
        env.position = int(position)
        return env._get_observation().astype(np.float32)
