"""Hyperliquid testnet/mainnet client.

Без KYC и гео-блоков: подписи делает EVM-кошелёк (eth_account). Интерфейс
совместим с тем, что использует Trader — Position, fetch_klines, last_price,
position, balance_usdt, quantity_step, round_quantity, market_order,
close_position.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants

from .retry import with_retry

log = logging.getLogger(__name__)


# Hyperliquid interval strings: 1m,3m,5m,15m,30m,1h,2h,4h,8h,12h,1d,3d,1w,1M
# Совпадает с тем, что в нашем config.yaml.

# Сколько свечей назад грузить за один запрос (по умолчанию SDK тянет диапазон [start, end]).
# Этого достаточно для lstm_window_size=128 + feature_window=20 + запас.


@dataclass
class Position:
    symbol: str
    amount: float
    entry_price: float
    unrealized_pnl: float


def _coin_from_symbol(symbol: str) -> str:
    """BTCUSDT → BTC, ETH-USDT → ETH, BTC → BTC."""
    s = symbol.upper().replace("-", "")
    for quote in ("USDT", "USDC", "USD"):
        if s.endswith(quote) and len(s) > len(quote):
            return s[: -len(quote)]
    return s


def _interval_to_minutes(interval: str) -> int:
    s = interval.strip().lower()
    mul = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    unit = s[-1]
    if unit not in mul:
        raise ValueError(f"unknown interval: {interval}")
    return int(s[:-1]) * mul[unit]


class HyperliquidFutures:
    def __init__(
        self,
        private_key: str,
        account_address: str,
        symbol: str,
        testnet: bool = True,
        leverage: int = 1,
        margin_type: str = "CROSS",
    ):
        self.coin = _coin_from_symbol(symbol)
        self.symbol = self.coin  # для совместимости с trader.cfg.symbol логированием
        self._wallet = Account.from_key(private_key)
        self._account_address = account_address or self._wallet.address
        base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        self.info = Info(base_url=base_url, skip_ws=True)
        self.exchange = Exchange(
            self._wallet, base_url=base_url, account_address=self._account_address
        )
        self._sz_decimals: Optional[int] = None
        log.info(
            "Hyperliquid client: coin=%s testnet=%s wallet=%s account=%s",
            self.coin, testnet, self._wallet.address, self._account_address,
        )
        self._set_leverage(leverage, margin_type)

    # ---------- market data ----------

    @with_retry()
    def fetch_klines(self, interval: str, limit: int = 500) -> pd.DataFrame:
        minutes = _interval_to_minutes(interval)
        end_ms = int(time.time() * 1000)
        # +1 свеча, чтобы наверняка попасть в текущий открытый бар (его потом отбросит trader).
        start_ms = end_ms - (limit + 1) * minutes * 60_000
        rows = self.info.candles_snapshot(
            name=self.coin, interval=interval, startTime=start_ms, endTime=end_ms,
        )
        if not rows:
            raise RuntimeError(f"no klines returned for {self.coin} {interval}")
        df = pd.DataFrame(rows)
        # Поля: t (open ms), T (close ms), o, c, h, l, v, n
        df["open_time"] = pd.to_datetime(df["t"].astype("int64"), unit="ms", utc=True)
        for c in ("o", "h", "l", "c", "v"):
            df[c] = df[c].astype(float)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                                 "c": "close", "v": "volume"})
        df = df.sort_values("open_time").set_index("open_time")
        return df[["open", "high", "low", "close", "volume"]]

    @with_retry()
    def last_price(self) -> float:
        mids = self.info.all_mids()
        if self.coin not in mids:
            raise RuntimeError(f"mid not found for {self.coin}")
        return float(mids[self.coin])

    # ---------- account ----------

    @with_retry()
    def position(self) -> Position:
        state = self.info.user_state(self._account_address)
        for ap in state.get("assetPositions", []):
            p = ap.get("position", {})
            if p.get("coin") == self.coin:
                szi = float(p.get("szi") or 0.0)
                return Position(
                    symbol=self.coin,
                    amount=szi,
                    entry_price=float(p.get("entryPx") or 0.0),
                    unrealized_pnl=float(p.get("unrealizedPnl") or 0.0),
                )
        return Position(self.coin, 0.0, 0.0, 0.0)

    @with_retry()
    def balance_usdt(self) -> float:
        """Доступный USDC. В unified account Hyperliquid spot и perp — единое
        пространство collateral'а: в flat-state marginSummary.accountValue=0,
        но spot USDC доступен для открытия позиций. Берём максимум: perp
        accountValue (если позиция открыта) или spot USDC (если flat)."""
        perp_value = 0.0
        try:
            state = self.info.user_state(self._account_address)
            ms = state.get("marginSummary") or {}
            perp_value = float(ms.get("accountValue") or 0.0)
        except Exception as e:
            log.warning("perp user_state failed: %s", e)
        if perp_value > 0:
            return perp_value
        # Unified account fallback: spot USDC.
        try:
            sp = self.info.spot_user_state(self._account_address)
            for b in sp.get("balances", []):
                if b.get("coin") == "USDC":
                    return float(b.get("total") or 0.0)
        except Exception as e:
            log.warning("spot_user_state failed: %s", e)
        return 0.0

    # ---------- precision ----------

    def _load_sz_decimals(self) -> int:
        meta = self.info.meta()
        for entry in meta.get("universe", []):
            if entry.get("name") == self.coin:
                return int(entry.get("szDecimals") or 4)
        return 4

    def quantity_step(self) -> float:
        if self._sz_decimals is None:
            try:
                self._sz_decimals = self._load_sz_decimals()
            except Exception as e:
                log.warning("sz_decimals fallback (%s)", e)
                self._sz_decimals = 4
        return 10 ** (-self._sz_decimals)

    def round_quantity(self, qty: float) -> float:
        step = self.quantity_step()
        if step <= 0:
            return qty
        n = int(qty / step)
        return round(n * step, 8)

    # ---------- orders ----------

    @with_retry(max_attempts=3)
    def market_order(self, side: str, quantity: float, reduce_only: bool = False) -> dict:
        assert side in ("BUY", "SELL")
        is_buy = side == "BUY"
        sz = float(quantity)
        log.info("Hyperliquid market_open: coin=%s is_buy=%s sz=%s reduce_only=%s",
                 self.coin, is_buy, sz, reduce_only)
        if reduce_only:
            # market_close сам определяет сторону по текущей позиции и закрывает её;
            # это надёжнее чем market_open с reduceOnly, если есть округление.
            return self.exchange.market_close(self.coin)
        return self.exchange.market_open(
            name=self.coin, is_buy=is_buy, sz=sz, slippage=0.05,
        )

    def close_position(self) -> Optional[dict]:
        pos = self.position()
        if abs(pos.amount) < 1e-12:
            return None
        return self.exchange.market_close(self.coin)

    # ---------- setup ----------

    def _set_leverage(self, leverage: int, margin_type: str) -> None:
        is_cross = margin_type.upper() != "ISOLATED"
        try:
            self.exchange.update_leverage(leverage, self.coin, is_cross)
        except Exception as e:
            log.warning("update_leverage failed: %s", e)


def _resolve_private_key(state_dir: Path) -> tuple[str, str]:
    """Возвращает (private_key_hex_with_0x, account_address).

    Приоритет:
      1. ENV HL_WALLET_PRIVATE_KEY (+ опционально HL_ACCOUNT_ADDRESS).
      2. state/wallet.key — заранее созданный или сохранённый при прошлом запуске.
      3. Сгенерировать новый, сохранить в state/wallet.key, выйти с сообщением,
         что нужно пополнить через faucet.
    """
    key = os.environ.get("HL_WALLET_PRIVATE_KEY", "").strip()
    if key:
        if not key.startswith("0x"):
            key = "0x" + key
        addr = os.environ.get("HL_ACCOUNT_ADDRESS", "").strip()
        if not addr:
            addr = Account.from_key(key).address
        return key, addr

    state_dir.mkdir(parents=True, exist_ok=True)
    wallet_file = state_dir / "wallet.key"
    if wallet_file.exists():
        key = wallet_file.read_text().strip()
        if not key.startswith("0x"):
            key = "0x" + key
        addr = Account.from_key(key).address
        return key, addr

    # generate new
    acct = Account.create()
    wallet_file.write_text(acct.key.hex())
    try:
        wallet_file.chmod(0o600)
    except Exception:
        pass
    log.warning(
        "Hyperliquid wallet сгенерирован и сохранён в %s\n"
        "    Address: %s\n"
        "    1) Подключитесь к https://app.hyperliquid-testnet.xyz/ этим адресом\n"
        "    2) Portfolio → Claim Mock USDC (1000 USDC)\n"
        "    3) Перезапустите контейнер. До этого balance = 0 и ордера не пойдут.",
        wallet_file, acct.address,
    )
    return acct.key.hex(), acct.address


def from_env(
    symbol: str, testnet: bool, leverage: int, margin_type: str,
    state_dir: str | Path = "state",
) -> HyperliquidFutures:
    key, addr = _resolve_private_key(Path(state_dir))
    return HyperliquidFutures(
        private_key=key, account_address=addr,
        symbol=symbol, testnet=testnet,
        leverage=leverage, margin_type=margin_type,
    )
