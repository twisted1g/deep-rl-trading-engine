"""Live trading loop: 1h bar-close → observation → predict → execute."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import timezone
from typing import Optional

import pandas as pd

from ..exchange.client import HyperliquidFutures
from ..model.loader import predict_action
from ..notifications.notifier import TelegramNotifier
from ..persistence.journal import TradeJournal
from .observation import ObservationBuilder

log = logging.getLogger(__name__)

# Maps research-env action integers (trading_env_baseline.py:26) to position signs.
ACTION_TO_POSITION = {0: 0, 1: 1, 2: -1}
SIGN_TO_NAME = {1: "long", -1: "short", 0: "flat"}


@dataclass
class TraderConfig:
    symbol: str
    interval: str = "1h"
    equity_fraction: float = 1.0
    leverage: int = 1
    max_holding_time: int = 72
    max_drawdown_threshold: float = 0.08
    poll_seconds: int = 30
    bar_close_grace_seconds: int = 5


@dataclass
class PositionState:
    """Virtual position state mirroring research-env semantics."""
    position: int = 0          # -1 short, 0 flat, 1 long
    entry_price: float = 0.0
    entry_bar_time: Optional[pd.Timestamp] = None
    holding_time: int = 0      # in bars
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
        ex: HyperliquidFutures,
        obs_builder: ObservationBuilder,
        model,
        vec_env,
        cfg: TraderConfig,
        notifier: Optional[TelegramNotifier] = None,
        journal: Optional[TradeJournal] = None,
    ):
        self.ex = ex
        self.obs_builder = obs_builder
        self.model = model
        self.vec_env = vec_env
        self.cfg = cfg
        self.notifier = notifier or TelegramNotifier(token=None, chat_id=None, enabled=False)
        self.journal = journal
        self.state = PositionState()
        self._lock = threading.RLock()
        self.paused = False
        self._open_trade_id: Optional[int] = None
        self._sync_position_from_exchange()
        self._restore_state()

    def _restore_state(self) -> None:
        if self.journal is None:
            return
        snap = self.journal.load_state()
        if snap is None:
            return
        self.paused = snap.paused
        if snap.last_processed_bar_time:
            try:
                self.state.last_processed_bar_time = pd.Timestamp(
                    snap.last_processed_bar_time
                )
            except Exception:
                log.warning("could not parse stored bar time: %s",
                            snap.last_processed_bar_time)
        if self.state.position != 0:
            row = self.journal.last_open_trade()
            if row is not None:
                self._open_trade_id = int(row["id"])
        log.info("Restored state: paused=%s last_bar=%s open_trade_id=%s",
                 self.paused, self.state.last_processed_bar_time, self._open_trade_id)

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
                self.notifier.notify_shutdown("(KeyboardInterrupt)")
                return
            except Exception as exc:
                log.exception("step failed")
                self.notifier.notify_error(exc)
                time.sleep(self.cfg.poll_seconds)

    def step(self) -> None:
        with self._lock:
            self._resync_state_from_exchange()

            klines = self.ex.fetch_klines(
                self.cfg.interval, limit=max(self.obs_builder.min_history() + 10, 256)
            )
            klines = self._drop_open_bar(klines)
            latest_bar_time = klines.index[-1]
            latest_close = float(klines["close"].iloc[-1])

            if self.state.last_processed_bar_time == latest_bar_time:
                return

            self._update_holding_state(latest_close)

            forced = self._check_forced_exit()
            if forced:
                log.info("Forced exit (%s) at %.2f", forced, latest_close)
                self.notifier.notify_forced_exit(forced, latest_close)
                self._close_position(exit_price=latest_close, reason=forced)
                self.state.last_processed_bar_time = latest_bar_time
                self._persist_state()
                return

            if self.paused:
                log.info("bar=%s close=%.2f [paused — skip predict]",
                         latest_bar_time, latest_close)
                self.state.last_processed_bar_time = latest_bar_time
                self._persist_state()
                return

            obs = self.obs_builder.build(klines, position=self.state.position)
            action = predict_action(self.model, self.vec_env, obs)
            target = ACTION_TO_POSITION[action]
            log.info(
                "bar=%s close=%.2f pos=%+d hold=%d dd=%.4f action=%d target=%+d",
                latest_bar_time, latest_close, self.state.position,
                self.state.holding_time, self.state.max_drawdown, action, target,
            )

            self._transition_to(target, latest_close, latest_bar_time)
            self.state.last_processed_bar_time = latest_bar_time
            self._persist_state()

    def set_paused(self, value: bool) -> None:
        with self._lock:
            self.paused = bool(value)
            self._persist_state()

    def status_snapshot(self) -> dict:
        with self._lock:
            return {
                "position": SIGN_TO_NAME[self.state.position],
                "entry_price": self.state.entry_price,
                "holding_time": self.state.holding_time,
                "max_drawdown": self.state.max_drawdown,
                "last_bar": str(self.state.last_processed_bar_time)
                if self.state.last_processed_bar_time is not None else None,
                "paused": self.paused,
                "symbol": self.cfg.symbol,
                "interval": self.cfg.interval,
            }

    def force_close(self) -> Optional[float]:
        with self._lock:
            if self.state.position == 0:
                return None
            price = self.ex.last_price()
            self._close_position(exit_price=price, reason="manual")
            self._persist_state()
            return price

    def force_resync(self) -> dict:
        with self._lock:
            changed = self._resync_state_from_exchange()
            self._persist_state()
            return {
                "changed": changed,
                "position": SIGN_TO_NAME[self.state.position],
                "entry": self.state.entry_price,
            }

    def test_open(self, side: str, fraction: float = 0.05) -> dict:
        """Open a position bypassing the model — for smoke-testing before deploy."""
        assert side in ("BUY", "SELL")
        with self._lock:
            self._resync_state_from_exchange()
            if self.state.position != 0:
                return {"error": f"position already open: {SIGN_TO_NAME[self.state.position]}"}
            balance = self.ex.balance_usdt()
            price = self.ex.last_price()
            notional = balance * fraction * self.cfg.leverage
            raw_qty = notional / price
            qty = self.ex.round_quantity(raw_qty)
            if qty <= 0:
                return {"error": f"qty <= 0 (balance={balance:.2f} price={price:.2f})"}
            sign = 1 if side == "BUY" else -1
            bar_time = self.state.last_processed_bar_time or pd.Timestamp.now(tz=timezone.utc)
            self._open_position(side=side, price=price, bar_time=bar_time, sign=sign)
            return {
                "ok": True,
                "side": SIGN_TO_NAME[sign],
                "qty": qty,
                "price": self.state.entry_price,
                "balance_before": balance,
            }

    # ---------- timing ----------

    def _wait_for_new_bar(self) -> None:
        while True:
            klines = self.ex.fetch_klines(self.cfg.interval, limit=3)
            klines = self._drop_open_bar(klines)
            latest = klines.index[-1]
            if self.state.last_processed_bar_time is None:
                return
            if latest > self.state.last_processed_bar_time:
                time.sleep(self.cfg.bar_close_grace_seconds)
                return
            time.sleep(self.cfg.poll_seconds)

    @staticmethod
    def _drop_open_bar(klines: pd.DataFrame) -> pd.DataFrame:
        if len(klines) < 2:
            return klines
        interval = klines.index[-1] - klines.index[-2]
        now = pd.Timestamp.now(tz=timezone.utc)
        if now < klines.index[-1] + interval:
            return klines.iloc[:-1]
        return klines

    # ---------- position state ----------

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
        try:
            pos = self.ex.position()
        except Exception as exc:
            log.warning("position sync failed (%s) — starting in flat-state", exc)
            self.notifier.notify_error(exc)
            self.state.position = 0
            return
        if abs(pos.amount) < 1e-12:
            self.state.position = 0
        elif pos.amount > 0:
            self.state.position = 1
            try:
                self.state.entry_price = pos.entry_price or self.ex.last_price()
            except Exception:
                self.state.entry_price = pos.entry_price or 0.0
        else:
            self.state.position = -1
            try:
                self.state.entry_price = pos.entry_price or self.ex.last_price()
            except Exception:
                self.state.entry_price = pos.entry_price or 0.0
        log.info(
            "Synced from exchange: pos=%+d entry=%.2f (amount=%.6f)",
            self.state.position, self.state.entry_price, pos.amount,
        )

    def _resync_state_from_exchange(self) -> bool:
        """Check exchange position vs internal state; apply any manual changes.

        Called before each step() to pick up trades placed via the exchange UI.
        Returns True if a mismatch was detected and applied.
        """
        try:
            pos = self.ex.position()
        except Exception as exc:
            log.warning("periodic position sync failed: %s", exc)
            return False
        exchange_sign = 0
        if abs(pos.amount) > 1e-12:
            exchange_sign = 1 if pos.amount > 0 else -1
        if exchange_sign == self.state.position:
            if exchange_sign != 0 and pos.entry_price > 0:
                self.state.entry_price = pos.entry_price
            return False
        log.warning(
            "Position mismatch — resyncing: state=%+d exchange=%+d (amount=%.6f entry=%.2f)",
            self.state.position, exchange_sign, pos.amount, pos.entry_price,
        )
        if self._open_trade_id is not None and self.journal is not None:
            try:
                self.journal.close_trade(
                    trade_id=self._open_trade_id,
                    exit_price=self.ex.last_price(),
                    pnl_pct=0.0,
                    exit_reason="manual_resync",
                )
            except Exception:
                log.exception("could not close stale journal trade during resync")
            self._open_trade_id = None
        if exchange_sign == 0:
            self.state.reset_flat()
        else:
            self.state.position = exchange_sign
            self.state.entry_price = pos.entry_price or self.ex.last_price()
            self.state.entry_bar_time = None
            self.state.holding_time = 0
            self.state.max_drawdown = 0.0
        self.notifier.notify_error(
            RuntimeError(
                f"Position resynced from exchange → {SIGN_TO_NAME[exchange_sign]} "
                f"@ {self.state.entry_price:.2f}"
            )
        )
        return True

    # ---------- execution ----------

    def _transition_to(
        self, target: int, price: float, bar_time: pd.Timestamp
    ) -> None:
        cur = self.state.position
        if target == cur:
            return
        if cur != 0:
            self._close_position(exit_price=price, reason="model")
        if target == 1:
            self._open_position(side="BUY", price=price, bar_time=bar_time, sign=1)
        elif target == -1:
            self._open_position(side="SELL", price=price, bar_time=bar_time, sign=-1)

    def _open_position(
        self, side: str, price: float, bar_time: pd.Timestamp, sign: int
    ) -> None:
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
        pos = self.ex.position()
        self.state.position = sign
        self.state.entry_price = pos.entry_price or price
        self.state.entry_bar_time = bar_time
        self.state.holding_time = 0
        self.state.max_drawdown = 0.0
        log.info("Opened %s @ %.2f qty=%.6f", SIGN_TO_NAME[sign], self.state.entry_price, qty)
        self.notifier.notify_open(SIGN_TO_NAME[sign], self.state.entry_price, qty)
        if self.journal is not None:
            self._open_trade_id = self.journal.open_trade(
                side=SIGN_TO_NAME[sign],
                qty=qty,
                entry_price=self.state.entry_price,
                bar_time=str(bar_time),
            )

    def _close_position(self, exit_price: float, reason: str = "model") -> None:
        if self.state.position == 0:
            return
        side = SIGN_TO_NAME[self.state.position]
        entry = self.state.entry_price
        pnl_pct = 0.0
        if entry > 0:
            pnl_pct = (exit_price - entry) / entry * self.state.position
        self.ex.close_position()
        log.info("Closed %s entry=%.2f exit=%.2f pnl=%+.2f%% (%s)",
                 side, entry, exit_price, pnl_pct * 100, reason)
        self.notifier.notify_close(side, entry, exit_price, pnl_pct, reason)
        if self.journal is not None and self._open_trade_id is not None:
            self.journal.close_trade(
                trade_id=self._open_trade_id,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                exit_reason=reason,
            )
            self._open_trade_id = None
        self.state.reset_flat()

    def _persist_state(self) -> None:
        if self.journal is None:
            return
        try:
            self.journal.save_state(
                last_processed_bar_time=(
                    str(self.state.last_processed_bar_time)
                    if self.state.last_processed_bar_time is not None else None
                ),
                paused=self.paused,
            )
        except Exception:
            log.exception("journal.save_state failed")
