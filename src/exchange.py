from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from binance.client import Client
from binance.enums import (
    FUTURE_ORDER_TYPE_MARKET,
    SIDE_BUY,
    SIDE_SELL,
)

log = logging.getLogger(__name__)

FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"


@dataclass
class Position:
    symbol: str
    amount: float
    entry_price: float
    unrealized_pnl: float


class BinanceFutures:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        testnet: bool = True,
        leverage: int = 1,
        margin_type: str = "ISOLATED",
    ):
        self.symbol = symbol.upper()
        self.client = Client(api_key, api_secret, testnet=testnet)
        if testnet:
            self.client.FUTURES_URL = FUTURES_TESTNET_URL + "/fapi"

        self._set_leverage(leverage)
        self._set_margin_type(margin_type)


    def fetch_klines(self, interval: str, limit: int = 500) -> pd.DataFrame:
        raw = self.client.futures_klines(
            symbol=self.symbol, interval=interval, limit=limit
        )
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_base_vol", "taker_quote_vol", "ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time")
        return df[["open", "high", "low", "close", "volume"]]

    def last_price(self) -> float:
        t = self.client.futures_symbol_ticker(symbol=self.symbol)
        return float(t["price"])


    def position(self) -> Position:
        info = self.client.futures_position_information(symbol=self.symbol)
        p = info[0]
        return Position(
            symbol=self.symbol,
            amount=float(p["positionAmt"]),
            entry_price=float(p["entryPrice"]),
            unrealized_pnl=float(p["unRealizedProfit"]),
        )

    def balance_usdt(self) -> float:
        for b in self.client.futures_account_balance():
            if b["asset"] == "USDT":
                return float(b["balance"])
        return 0.0

    def quantity_step(self) -> float:
        info = self.client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == self.symbol:
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        return float(f["stepSize"])
        return 0.001

    def round_quantity(self, qty: float) -> float:
        step = self.quantity_step()
        if step <= 0:
            return qty
        # floor к ближайшему кратному step
        n = int(qty / step)
        return round(n * step, 8)


    def market_order(self, side: str, quantity: float, reduce_only: bool = False) -> dict:
        assert side in ("BUY", "SELL")
        params = dict(
            symbol=self.symbol,
            side=SIDE_BUY if side == "BUY" else SIDE_SELL,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=round(quantity, 6),
        )
        if reduce_only:
            params["reduceOnly"] = "true"
        log.info("Sending market order: %s", params)
        return self.client.futures_create_order(**params)

    def close_position(self) -> Optional[dict]:
        pos = self.position()
        if abs(pos.amount) < 1e-12:
            return None
        side = "SELL" if pos.amount > 0 else "BUY"
        return self.market_order(side, abs(pos.amount), reduce_only=True)


    def _set_leverage(self, leverage: int) -> None:
        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=leverage)
        except Exception as e:
            log.warning("set leverage failed: %s", e)

    def _set_margin_type(self, margin_type: str) -> None:
        try:
            self.client.futures_change_margin_type(
                symbol=self.symbol, marginType=margin_type
            )
        except Exception as e:
            if "-4046" not in str(e):
                log.warning("set margin type failed: %s", e)


def from_env(symbol: str, testnet: bool, leverage: int, margin_type: str) -> BinanceFutures:
    key = os.environ.get("BINANCE_API_KEY")
    sec = os.environ.get("BINANCE_API_SECRET")
    if not key or not sec:
        raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET не заданы (.env)")
    return BinanceFutures(key, sec, symbol, testnet, leverage, margin_type)
