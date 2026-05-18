from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from .trading_env_baseline import MyTradingEnv


class MyTradingEnvLSTM(MyTradingEnv):
    """Same trading mechanics as ``MyTradingEnv``, but the observation is the
    last hidden state of a pre-trained LSTM consuming a sliding window of
    ``(log_return, rolling_vol, volume_norm, position)`` features.

    Logging / history semantics (append-only across resets) are inherited
    from the base class — see its docstring.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1000.0,
        commission: float = 0.0001,
        slippage: float = 0.0005,
        max_holding_time: int = 72,
        max_drawdown_threshold: float = 0.08,
        max_steps: Optional[int] = None,
        feature_window: int = 20,
        inactivity_penalty: float = 0.0,
        lstm_window_size: int = 128,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 2,
        lstm_encoder: Optional[torch.nn.LSTM] = None,
        lstm_checkpoint_path: Optional[str] = None,
        lstm_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(
            df=df,
            initial_balance=initial_balance,
            commission=commission,
            slippage=slippage,
            max_holding_time=max_holding_time,
            max_drawdown_threshold=max_drawdown_threshold,
            max_steps=max_steps,
            feature_window=feature_window,
            inactivity_penalty=inactivity_penalty,
            **kwargs,
        )

        self.lstm_window_size = int(lstm_window_size)
        self.lstm_hidden_size = int(lstm_hidden_size)
        self.lstm_layers = int(lstm_layers)
        self.lstm_device = str(lstm_device)

        if lstm_encoder is None and lstm_checkpoint_path is not None:
            from encoders.lstm_pretrain import load_lstm_encoder

            checkpoint_path = Path(lstm_checkpoint_path)
            if not checkpoint_path.is_absolute():
                project_root = Path(__file__).resolve().parents[2]
                checkpoint_path = project_root / checkpoint_path

            self.lstm_encoder, self.lstm_layernorm = load_lstm_encoder(
                str(checkpoint_path),
                device=self.lstm_device,
            )
        elif lstm_encoder is None:
            self.lstm_encoder = torch.nn.LSTM(
                input_size=3,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_layers,
                batch_first=True,
            )
            self.lstm_layernorm = torch.nn.LayerNorm(self.lstm_hidden_size)
        else:
            self.lstm_encoder = lstm_encoder
            self.lstm_layernorm = torch.nn.LayerNorm(self.lstm_hidden_size)

        self.lstm_encoder.to(self.lstm_device)
        self.lstm_encoder.eval()
        self.lstm_layernorm.to(self.lstm_device)
        self.lstm_layernorm.eval()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lstm_hidden_size + 4,),
            dtype=np.float32,
        )

        self._market_features = self._precompute_market_features()

    def _precompute_market_features(self) -> np.ndarray:
        """Compute (N, 3) matrix [log_return, rolling_vol, volume_norm] once.

        These are the 3 channels the LSTM encoder is pre-trained on.
        Position and P&L context are passed to the policy MLP separately
        (concatenated to the LSTM hidden state in `_get_observation`),
        not fed into the LSTM, to avoid the train/inference distribution
        shift that arises when pretrain sees position=0 but RL sees ±1.
        """
        close = self.df["close"].astype(float).to_numpy()
        volume = self.df["volume"].astype(float).to_numpy()
        n = close.shape[0]

        log_return = np.zeros(n, dtype=np.float32)
        prev = close[:-1]
        curr = close[1:]
        valid = prev > 0
        log_return[1:][valid] = np.log(curr[valid] / prev[valid]).astype(np.float32)

        rolling_vol = np.zeros(n, dtype=np.float32)
        volume_norm = np.zeros(n, dtype=np.float32)
        fw = int(self.feature_window)
        for i in range(n):
            r_start = max(1, i - fw + 1)
            r_window = log_return[r_start : i + 1]
            rolling_vol[i] = float(np.std(r_window)) if r_window.size > 1 else 0.0

            v_start = max(0, i - fw + 1)
            v_window = volume[v_start : i + 1]
            v_mean = float(v_window.mean()) if v_window.size > 0 else 0.0
            volume_norm[i] = float(v_window[-1] / v_mean) if v_mean > 0 else 0.0

        return np.stack([log_return, rolling_vol, volume_norm], axis=1).astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        end = int(self.current_step)
        start = end - self.lstm_window_size + 1
        pad_len = max(0, -start)
        start = max(0, start)

        market = self._market_features[start : end + 1]
        if pad_len > 0:
            market = np.concatenate(
                [np.zeros((pad_len, 3), dtype=np.float32), market], axis=0
            )
        if market.shape[0] > self.lstm_window_size:
            market = market[-self.lstm_window_size :]

        seq = torch.as_tensor(
            market, dtype=torch.float32, device=self.lstm_device
        ).unsqueeze(0)
        with torch.no_grad():
            _, (h_n, _) = self.lstm_encoder(seq)
            hidden = self.lstm_layernorm(h_n[-1])
        hidden_np = hidden[0].detach().cpu().numpy().astype(np.float32)

        current_price = float(self.df.iloc[self.current_step]["close"])
        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl_pct = (
                (current_price - self.entry_price) / self.entry_price * float(self.position)
            )
            bars_since_entry_n = (self.current_step - self.entry_step) / 100.0
            position_signed_pnl = float(self.position) * float(np.sign(unrealized_pnl_pct))
        else:
            unrealized_pnl_pct = 0.0
            bars_since_entry_n = 0.0
            position_signed_pnl = 0.0

        extra = np.array(
            [
                float(self.position),
                unrealized_pnl_pct,
                bars_since_entry_n,
                position_signed_pnl,
            ],
            dtype=np.float32,
        )
        return np.concatenate([hidden_np, extra], axis=0)

    def _pick_start_index(self) -> int:
        if self.max_steps is None:
            start_max = len(self.df) - 1
        else:
            start_max = len(self.df) - self.max_steps

        min_start = max(1, self.lstm_window_size)
        fixed_start = getattr(self, "fixed_start_index", None)
        if fixed_start is not None:
            start_index = int(fixed_start)
            return max(min_start, min(start_index, start_max - 1))
        return int(self.np_random.integers(min_start, start_max))
