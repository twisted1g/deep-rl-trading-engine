"""Live-trader loop: 1h bar-close → obs → predict → execute."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .exchange import BinanceFutures
from .model_loader import predict_action
from .observation import ObservationBuilder

log = logging.getLogger(__name__)

# Mapping из research-env (см. trading_env_baseline.py:26):
ACTION_TO_POSITION = {0: 0, 1: 1, 2: -1}


@dataclass
class TraderConfig:
    symbol: str
    interval: str = "1h"
    # equity sizing
    equity_fraction: float = 1.0       # 100% balance, как в обучении
    leverage: int = 1
    # forced exits (как в env)
    max_holding_time: int = 72
    max_drawdown_threshold: float = 0.08
    # poll
    poll_seconds: int = 30
    bar_close_grace_seconds: int = 5


@dataclass
class PositionState:
    """Виртуальное состояние позиции, согласованное с семантикой env."""
    position: int = 0                  # -1 short, 0 flat, 1 long
    entry_price: float = 0.0
    entry_bar_time: Optional[pd.Timestamp] = None
    holding_time: int = 0              # в барах
    max_drawdown: float = 0.0
    last_processed_bar_time: Optional[pd.Timestamp] = None

    def reset_flat(self) -> None:
        self.position = 0
        self.entry_price = 0.0
        self.entry_bar_time = None
        self.holding_time = 0
        self.max_drawdown = 0.0


class Trader:
    def __init__(
        self,
        ex: BinanceFutures,
        obs_builder: ObservationBuilder,
        model,
        vec_env,
        cfg: TraderConfig,
    ):
        self.ex = ex
        self.obs_builder = obs_builder
        self.model = model
        self.vec_env = vec_env
        self.cfg = cfg
        self.state = PositionState()
        self._sync_position_from_exchange()

    # ---------- public ----------

    def run_forever(self) -> None:
        log.info(
            "Trader started: %s %s, equity_fraction=%.2f, leverage=%d",
            self.cfg.symbol, self.cfg.interval,
            self.cfg.equity_fraction, self.cfg.leverage,
        )
        while True:
            try:
                self._wait_for_new_bar()
                self.step()
            except KeyboardInterrupt:
                log.info("Stopped by user")
                return
            except Exception:
                log.exception("step failed")
                time.sleep(self.cfg.poll_seconds)

    def step(self) -> None:
        klines = self.ex.fetch_klines(
            self.cfg.interval, limit=max(self.obs_builder.min_history() + 10, 256)
        )
        # последняя свеча от Binance может быть НЕзакрытой — отбросим её,
        # если её open_time == текущему интервалу.
        klines = self._drop_open_bar(klines)
        latest_bar_time = klines.index[-1]
        latest_close = float(klines["close"].iloc[-1])

        # Если бар уже обработан — выходим.
        if self.state.last_processed_bar_time == latest_bar_time:
            return

        # --- 1. обновить holding_time / max_drawdown по последнему бару
        self._update_holding_state(latest_close)

        # --- 2. forced exits
        forced = self._check_forced_exit()
        if forced:
            log.info("Forced exit (%s) at %.2f", forced, latest_close)
            self._close_position()
            self.state.last_processed_bar_time = latest_bar_time
            return

        # --- 3. observation + predict
        obs = self.obs_builder.build(klines, position=self.state.position)
        action = predict_action(self.model, self.vec_env, obs)
        target = ACTION_TO_POSITION[action]
        log.info(
            "bar=%s close=%.2f pos=%+d hold=%d dd=%.4f action=%d target=%+d",
            latest_bar_time, latest_close, self.state.position,
            self.state.holding_time, self.state.max_drawdown, action, target,
        )

        # --- 4. исполнить переход позиций
        self._transition_to(target, latest_close, latest_bar_time)

        self.state.last_processed_bar_time = latest_bar_time

    # ---------- timing ----------

    def _wait_for_new_bar(self) -> None:
        """Спим, опрашивая, пока не появится новая закрытая свеча."""
        while True:
            klines = self.ex.fetch_klines(self.cfg.interval, limit=3)
            klines = self._drop_open_bar(klines)
            latest = klines.index[-1]
            if self.state.last_processed_bar_time is None:
                # первый запуск — сразу обрабатываем
                return
            if latest > self.state.last_processed_bar_time:
                # дать рынку миг устаканиться
                time.sleep(self.cfg.bar_close_grace_seconds)
                return
            time.sleep(self.cfg.poll_seconds)

    @staticmethod
    def _drop_open_bar(klines: pd.DataFrame) -> pd.DataFrame:
        """Последний kline от Binance может быть текущим, незакрытым баром."""
        if len(klines) < 2:
            return klines
        # Определяем интервал по разнице последних двух open_time.
        interval = klines.index[-1] - klines.index[-2]
        now = pd.Timestamp.now(tz=timezone.utc)
        if now < klines.index[-1] + interval:
            return klines.iloc[:-1]
        return klines

    # ---------- state ----------

    def _update_holding_state(self, current_price: float) -> None:
        if self.state.position == 0:
            return
        self.state.holding_time += 1
        if self.state.position == 1:
            dd = max(0.0, (self.state.entry_price - current_price) / self.state.entry_price)
        else:
            dd = max(0.0, (current_price - self.state.entry_price) / self.state.entry_price)
        if dd > self.state.max_drawdown:
            self.state.max_drawdown = dd

    def _check_forced_exit(self) -> Optional[str]:
        if self.state.position == 0:
            return None
        if self.state.holding_time >= self.cfg.max_holding_time:
            return "time"
        if self.state.max_drawdown >= self.cfg.max_drawdown_threshold:
            return "drawdown"
        return None

    def _sync_position_from_exchange(self) -> None:
        pos = self.ex.position()
        if abs(pos.amount) < 1e-12:
            self.state.position = 0
        elif pos.amount > 0:
            self.state.position = 1
            self.state.entry_price = pos.entry_price or self.ex.last_price()
        else:
            self.state.position = -1
            self.state.entry_price = pos.entry_price or self.ex.last_price()
        log.info(
            "Synced from exchange: pos=%+d entry=%.2f (binance amt=%.6f)",
            self.state.position, self.state.entry_price, pos.amount,
        )

    # ---------- execution ----------

    def _transition_to(
        self, target: int, price: float, bar_time: pd.Timestamp
    ) -> None:
        cur = self.state.position
        if target == cur:
            return

        # Сначала закрыть, потом открыть в обратную сторону.
        if cur != 0:
            self._close_position()
        if target == 1:
            self._open_position(side="BUY", price=price, bar_time=bar_time, sign=1)
        elif target == -1:
            self._open_position(side="SELL", price=price, bar_time=bar_time, sign=-1)

    def _open_position(self, side: str, price: float, bar_time: pd.Timestamp, sign: int) -> None:
        balance = self.ex.balance_usdt()
        notional = balance * self.cfg.equity_fraction * self.cfg.leverage
        raw_qty = notional / price
        qty = self.ex.round_quantity(raw_qty)
        if qty <= 0:
            log.warning(
                "Computed quantity <= 0 (balance=%.4f price=%.2f raw=%.8f)",
                balance, price, raw_qty,
            )
            return
        self.ex.market_order(side, qty)
        # Подтянем фактическую позицию и entry_price.
        pos = self.ex.position()
        self.state.position = sign
        self.state.entry_price = pos.entry_price or price
        self.state.entry_bar_time = bar_time
        self.state.holding_time = 0
        self.state.max_drawdown = 0.0

    def _close_position(self) -> None:
        self.ex.close_position()
        self.state.reset_flat()
