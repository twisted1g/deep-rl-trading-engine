from __future__ import annotations

from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from gymnasium import Env, spaces


class MyTradingEnv(Env):
    """Discrete long/short/flat trading environment.

    Action mapping: 0 → flat, 1 → long, 2 → short (see ``ACTION_TO_POSITION``).
    Observation is a 6-dim feature vector (log_return, rolling vol, volume
    z-score, price z-score, EMA-diff, current position).

    Logging buffers (``step_history``, ``trade_history``, ``portfolio_history``)
    are **append-only across resets** — they are cleared only by
    :meth:`clear_history`. This is important because Stable-Baselines3
    ``VecEnv`` auto-resets the env after a ``done`` step, which would
    otherwise wipe the just-completed episode's data.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    ACTION_TO_POSITION: ClassVar[Mapping[int, int]] = {0: 0, 1: 1, 2: -1}

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
        turnover_penalty: float = 0.0012,
        commission_warmup_fraction: float = 0.3,
        total_train_steps: Optional[int] = None,
        **kwargs,
    ):
        self.initial_balance = float(initial_balance)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.max_holding_time = int(max_holding_time)
        self.max_drawdown_threshold = float(max_drawdown_threshold)
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.feature_window = int(feature_window)
        self.inactivity_penalty = float(inactivity_penalty)
        self.turnover_penalty = float(turnover_penalty)
        self.commission_warmup_fraction = float(commission_warmup_fraction)
        self.total_train_steps = int(total_train_steps) if total_train_steps is not None else None
        self._global_step_counter = 0

        self.df = df.copy().reset_index(drop=True)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self.current_step = None
        self.position = 0
        self.prev_position = 0
        self.units = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        self.cash = 0.0
        self.portfolio_value = 0.0
        self.position_value = 0.0
        self.current_holding_time = 0
        self.max_drawdown = 0.0
        self._steps_elapsed = 0
        self.last_exit_reason = None
        self.prev_portfolio_value = self.initial_balance
        self.episode_id = 0
        self.trade_history = []
        self.portfolio_history = []
        self.step_history = []

    def _get_observation(self) -> np.ndarray:
        return self._get_feature_vector_at(self.current_step)

    def _get_feature_vector_at(self, step_index: int) -> np.ndarray:
        if "close" not in self.df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        if "volume" not in self.df.columns:
            raise ValueError("DataFrame must contain 'volume' column")

        current_price = float(self.df.iloc[step_index]["close"])
        if step_index > 0:
            prev_price = float(self.df.iloc[step_index - 1]["close"])
            log_return = (
                float(np.log(current_price / prev_price)) if prev_price > 0 else 0.0
            )
        else:
            log_return = 0.0

        start = max(1, step_index - self.feature_window + 1)
        log_returns = []
        for i in range(start, step_index + 1):
            prev_price = float(self.df.iloc[i - 1]["close"])
            curr_price = float(self.df.iloc[i]["close"])
            if prev_price > 0 and curr_price > 0:
                log_returns.append(float(np.log(curr_price / prev_price)))
            else:
                log_returns.append(0.0)

        rolling_volatility = float(np.std(log_returns)) if len(log_returns) > 1 else 0.0

        window_start = max(0, step_index - self.feature_window + 1)
        close_window = self.df.iloc[window_start : step_index + 1]["close"].astype(float)
        volume_window = self.df.iloc[window_start : step_index + 1]["volume"].astype(float)

        price_mean = float(close_window.mean()) if len(close_window) > 0 else 0.0
        price_std = float(close_window.std(ddof=0)) if len(close_window) > 1 else 0.0
        price_zscore = (
            float((current_price - price_mean) / (price_std + 1e-12)) if price_std > 0 else 0.0
        )

        volume_mean = float(volume_window.mean()) if len(volume_window) > 0 else 0.0
        volume_std = float(volume_window.std(ddof=0)) if len(volume_window) > 1 else 0.0
        volume_zscore = (
            float((volume_window.iloc[-1] - volume_mean) / (volume_std + 1e-12))
            if volume_std > 0
            else 0.0
        )

        ema_fast_span = max(2, self.feature_window // 4)
        ema_slow_span = max(3, self.feature_window)
        ema_fast = float(close_window.ewm(span=ema_fast_span, adjust=False).mean().iloc[-1])
        ema_slow = float(close_window.ewm(span=ema_slow_span, adjust=False).mean().iloc[-1])
        ema_diff = float((ema_fast - ema_slow) / (abs(ema_slow) + 1e-12))

        if self.position != 0 and self.entry_price > 0:
            unrealized_pnl_pct = (
                (current_price - self.entry_price) / self.entry_price * float(self.position)
            )
            bars_since_entry_n = (step_index - self.entry_step) / 100.0
            position_signed_pnl = float(self.position) * float(np.sign(unrealized_pnl_pct))
        else:
            unrealized_pnl_pct = 0.0
            bars_since_entry_n = 0.0
            position_signed_pnl = 0.0

        return np.array(
            [
                log_return,
                rolling_volatility,
                volume_zscore,
                price_zscore,
                ema_diff,
                float(self.position),
                unrealized_pnl_pct,
                bars_since_entry_n,
                position_signed_pnl,
            ],
            dtype=np.float32,
        )

    def _calculate_reward(self, done: bool) -> float:
        if self.prev_portfolio_value <= 0.0:
            return 0.0
        log_ret = float(np.log(self.portfolio_value / self.prev_portfolio_value))
        turnover = abs(int(self.position) - int(self.prev_position))
        return log_ret - self.turnover_penalty * turnover

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action)

        self.prev_portfolio_value = self.portfolio_value
        self.prev_position = self.position
        current_price = float(self.df.iloc[self.current_step]["close"])
        self.last_exit_reason = None
        target_position = self.ACTION_TO_POSITION[int(action)]

        if self.total_train_steps and self.commission_warmup_fraction > 0:
            progress = self._global_step_counter / float(self.total_train_steps)
            warmup = self.commission_warmup_fraction
            if progress < warmup:
                scale = 0.0
            else:
                scale = min(1.0, (progress - warmup) / max(1e-9, 1.0 - warmup))
            commission_eff = self.commission * scale
            slippage_eff = self.slippage * scale
        else:
            commission_eff = self.commission
            slippage_eff = self.slippage
        self._global_step_counter += 1

        orig_c, orig_s = self.commission, self.slippage
        self.commission, self.slippage = commission_eff, slippage_eff

        def open_long() -> None:
            price_with_slip = current_price * (1.0 + self.slippage)
            invest_amount = self.portfolio_value
            commission_fee = invest_amount * self.commission
            units = (invest_amount - commission_fee) / price_with_slip

            self.units = float(units)
            self.entry_price = price_with_slip
            self.entry_step = int(self.current_step)
            self.position = 1
            self.current_holding_time = 0
            self.max_drawdown = 0.0
            self.cash = max(
                0.0, self.portfolio_value - units * price_with_slip - commission_fee
            )
            self.position_value = units * current_price
            self.portfolio_value = self.cash + self.position_value

        def open_short() -> None:
            price_with_slip = current_price * (1.0 - self.slippage)
            invest_amount = self.portfolio_value
            commission_fee = invest_amount * self.commission
            units = (invest_amount - commission_fee) / price_with_slip

            self.units = float(units)
            self.entry_price = price_with_slip
            self.entry_step = int(self.current_step)
            self.position = -1
            self.current_holding_time = 0
            self.max_drawdown = 0.0
            proceeds = units * price_with_slip
            self.cash = max(0.0, self.portfolio_value + proceeds - commission_fee)
            self.position_value = -units * current_price
            self.portfolio_value = self.cash + self.position_value

        def close_position(exit_reason: str) -> None:
            self.last_exit_reason = exit_reason

            if self.position == 1:
                exit_price = current_price * (1.0 - self.slippage)
                exit_value = self.units * exit_price
                commission_fee = exit_value * self.commission
                pnl = exit_value - (self.units * self.entry_price) - commission_fee
                self.cash += exit_value - commission_fee
            else:
                exit_price = current_price * (1.0 + self.slippage)
                exit_value = self.units * exit_price
                commission_fee = exit_value * self.commission
                pnl = (self.units * self.entry_price) - exit_value - commission_fee
                self.cash -= exit_value + commission_fee

            self.position_value = 0.0
            self.portfolio_value = self.cash

            self.trade_history.append(
                {
                    "episode": int(self.episode_id),
                    "entry_price": float(self.entry_price),
                    "exit_price": float(exit_price),
                    "pnl": float(pnl),
                    "units": float(self.units),
                    "holding_time": int(self.current_holding_time),
                    "max_drawdown": float(self.max_drawdown),
                    "exit_reason": self.last_exit_reason,
                }
            )

            self.units = 0.0
            self.entry_price = 0.0
            self.entry_step = 0
            self.position = 0
            self.current_holding_time = 0
            self.max_drawdown = 0.0

        if self.position == 0:
            if target_position == 1:
                open_long()
            elif target_position == -1:
                open_short()
        else:
            self.current_holding_time += 1
            if self.position == 1:
                self.position_value = self.units * current_price
                unrealized_pnl = self.position_value - (self.units * self.entry_price)
            else:
                self.position_value = -self.units * current_price
                unrealized_pnl = (self.units * self.entry_price) - (
                    self.units * current_price
                )
            self.portfolio_value = self.cash + self.position_value

            current_drawdown = (
                -unrealized_pnl / (self.units * self.entry_price + 1e-12)
                if unrealized_pnl < 0
                else 0.0
            )
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

            forced_close = (
                self.current_holding_time >= self.max_holding_time
                or current_drawdown >= self.max_drawdown_threshold
            )

            if forced_close:
                reason = (
                    "time"
                    if self.current_holding_time >= self.max_holding_time
                    else "drawdown"
                )
                close_position(reason)
                target_position = 0
            elif target_position != self.position:
                close_position("agent")
                if target_position == 1:
                    open_long()
                elif target_position == -1:
                    open_short()

        self.commission, self.slippage = orig_c, orig_s

        self.current_step += 1
        self._steps_elapsed += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = self.max_steps is not None and self._steps_elapsed >= self.max_steps

        done = terminated or truncated
        reward = self._calculate_reward(done)

        obs = self._get_observation()
        info = {
            "portfolio_value": float(self.portfolio_value),
            "position": int(self.position),
            "holding_time": int(self.current_holding_time),
            "current_price": float(current_price),
            "n_trades": len(self.trade_history),
            "last_exit_reason": self.last_exit_reason,
        }

        self.portfolio_history.append(float(self.portfolio_value))
        self._log_step(
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        return obs, reward, terminated, truncated, info

    def _pick_start_index(self) -> int:
        if self.max_steps is None:
            start_max = len(self.df) - 1
        else:
            start_max = len(self.df) - self.max_steps

        min_start = max(1, self.feature_window)
        fixed_start = getattr(self, "fixed_start_index", None)
        if fixed_start is not None:
            start_index = int(fixed_start)
            return max(min_start, min(start_index, start_max - 1))
        return int(self.np_random.integers(min_start, start_max))

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        # Initializes / seeds self.np_random via gymnasium.Env machinery
        # without touching the global numpy RNG.
        super().reset(seed=seed)

        self.current_step = self._pick_start_index()

        self.position = 0
        self.prev_position = 0
        self.units = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        self.cash = float(self.initial_balance)
        self.portfolio_value = float(self.initial_balance)
        self.position_value = 0.0
        self.current_holding_time = 0
        self.max_drawdown = 0.0
        self._steps_elapsed = 0
        self.prev_portfolio_value = float(self.initial_balance)
        self.last_exit_reason = None
        self.episode_id += 1

        # NOTE: trade_history / portfolio_history / step_history are NOT cleared
        # here — they accumulate across episodes and resets. Clear them via
        # `clear_history()` only. This is what makes the env robust to
        # VecEnv auto-reset on `done`.

        obs = self._get_observation()
        return obs, {}

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            step = self.current_step
            total = len(self.df) - 1
            if self.position == 1:
                pos_label = "Long"
            elif self.position == -1:
                pos_label = "Short"
            else:
                pos_label = "Flat"
            print(
                f"Step: {step}/{total} | Portfolio: ${self.portfolio_value:,.2f} | "
                f"Position: {pos_label} | "
                f"Hold time: {self.current_holding_time} | Trades: {len(self.trade_history)}"
            )

    def _log_step(
        self,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> None:
        df_index = int(self.current_step - 1)
        row = {
            "episode": int(self.episode_id),
            "step": int(self._steps_elapsed),
            "df_index": df_index,
            "action": int(action),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "portfolio_value": float(self.portfolio_value),
            "cash": float(self.cash),
            "position": int(self.position),
            "units": float(self.units),
            "entry_price": float(self.entry_price),
            "position_value": float(self.position_value),
            "holding_time": int(self.current_holding_time),
            "current_price": float(info.get("current_price", np.nan)),
            "n_trades": int(info.get("n_trades", len(self.trade_history))),
            "last_exit_reason": info.get("last_exit_reason"),
        }

        if self.entry_price > 0 and self.position != 0:
            if self.position == 1:
                row["unrealized_pnl"] = float(
                    self.position_value - (self.units * self.entry_price)
                )
            else:
                row["unrealized_pnl"] = float(
                    (self.units * self.entry_price)
                    - (self.units * row["current_price"])
                )
        else:
            row["unrealized_pnl"] = 0.0

        if "timestamp" in self.df.columns:
            row["timestamp"] = self.df.iloc[df_index]["timestamp"]
        elif "date" in self.df.columns:
            row["date"] = self.df.iloc[df_index]["date"]

        self.step_history.append(row)

    def get_steps_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.step_history)

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_history)

    def clear_history(self) -> None:
        self.step_history = []
        self.trade_history = []
        self.portfolio_history = []
        self.episode_id = 0
