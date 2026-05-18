"""Telegram push-уведомления о торговых событиях.

Реализация — синхронный POST на Bot API через httpx. Никакого asyncio в
торговом цикле; ошибки отправки только логируются.
"""
from __future__ import annotations

import html
import logging
import os
import time
from typing import Optional

import httpx

log = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    def __init__(
        self,
        token: Optional[str],
        chat_id: Optional[int | str],
        enabled: bool = True,
        timeout: float = 5.0,
    ):
        self.enabled = bool(enabled and token and chat_id)
        self.token = token
        self.chat_id = chat_id
        self.timeout = timeout
        self._last_error_ts: float = 0.0
        self._last_error_text: str = ""

    @classmethod
    def from_env(cls, enabled: bool, chat_id: Optional[int | str] = None) -> "TelegramNotifier":
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        cid = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        if enabled and (not token or not cid):
            log.warning("Telegram enabled, но TELEGRAM_BOT_TOKEN/CHAT_ID не заданы — отключаю")
            enabled = False
        return cls(token=token, chat_id=cid, enabled=enabled)

    # ---------- public events ----------

    def notify_startup(self, summary: str) -> None:
        self._send(f"🚀 <b>engine started</b>\n{html.escape(summary)}")

    def notify_shutdown(self, reason: str = "") -> None:
        self._send(f"🛑 <b>engine stopped</b> {html.escape(reason)}".strip())

    def notify_open(self, side: str, price: float, qty: float) -> None:
        emoji = "🟢" if side == "long" else "🔴"
        self._send(
            f"{emoji} <b>OPEN {html.escape(side.upper())}</b>\n"
            f"price: <code>{price:.2f}</code>\nqty: <code>{qty:.6f}</code>"
        )

    def notify_close(
        self,
        side: str,
        entry: float,
        exit_price: float,
        pnl_pct: float,
        reason: str = "model",
    ) -> None:
        emoji = "✅" if pnl_pct >= 0 else "🔻"
        self._send(
            f"{emoji} <b>CLOSE {html.escape(side.upper())}</b> ({html.escape(reason)})\n"
            f"entry: <code>{entry:.2f}</code> → exit: <code>{exit_price:.2f}</code>\n"
            f"pnl: <code>{pnl_pct*100:+.2f}%</code>"
        )

    def notify_forced_exit(self, reason: str, price: float) -> None:
        self._send(f"⚠️ <b>forced exit ({html.escape(reason)})</b> at <code>{price:.2f}</code>")

    def notify_error(self, exc: BaseException, cooldown: float = 300.0) -> None:
        text = f"{type(exc).__name__}: {exc}"
        now = time.time()
        if text == self._last_error_text and now - self._last_error_ts < cooldown:
            return
        self._last_error_text = text
        self._last_error_ts = now
        self._send(f"❗ <b>engine error</b>\n<pre>{html.escape(text)}</pre>")

    # ---------- transport ----------

    def _send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            httpx.post(
                _API.format(token=self.token),
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=self.timeout,
            )
        except Exception:
            log.exception("telegram send failed")
