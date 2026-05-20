"""Wallet key resolution for Hyperliquid: env var → file → generate."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from eth_account import Account

from .client import HyperliquidFutures

log = logging.getLogger(__name__)


def resolve_private_key(state_dir: Path) -> tuple[str, str]:
    """Return (private_key_hex_with_0x, account_address).

    Priority:
      1. HL_WALLET_PRIVATE_KEY env var (+ optional HL_ACCOUNT_ADDRESS).
      2. state/wallet.key — pre-created or saved on a previous run.
      3. Generate new key, save to state/wallet.key, exit with faucet instructions.
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

    acct = Account.create()
    wallet_file.write_text(acct.key.hex())
    try:
        wallet_file.chmod(0o600)
    except Exception:
        pass
    log.warning(
        "New Hyperliquid wallet generated and saved to %s\n"
        "    Address: %s\n"
        "    1) Connect to https://app.hyperliquid-testnet.xyz/ with this address\n"
        "    2) Portfolio → Claim Mock USDC (1000 USDC)\n"
        "    3) Restart the container. Until then balance = 0.",
        wallet_file, acct.address,
    )
    return acct.key.hex(), acct.address


def from_env(
    symbol: str,
    testnet: bool,
    leverage: int,
    margin_type: str,
    state_dir: str | Path = "state",
) -> HyperliquidFutures:
    key, addr = resolve_private_key(Path(state_dir))
    return HyperliquidFutures(
        private_key=key,
        account_address=addr,
        symbol=symbol,
        testnet=testnet,
        leverage=leverage,
        margin_type=margin_type,
    )
