"""Telegram-бот команд: /status /pnl /trades /pause /resume /close.

Запускается в отдельном потоке-демоне со своим asyncio loop. Авторизованы
только сообщения от заданного chat_id.
"""
from __future__ import annotations

import asyncio
import html
import logging
import threading
from typing import TYPE_CHECKING, Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

if TYPE_CHECKING:
    from .journal import TradeJournal
    from .trader import Trader

log = logging.getLogger(__name__)


class CommandBot:
    def __init__(
        self,
        token: str,
        chat_id: int,
        trader: "Trader",
        journal: Optional["TradeJournal"] = None,
    ):
        self.token = token
        self.chat_id = int(chat_id)
        self.trader = trader
        self.journal = journal
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._app: Optional[Application] = None

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="tg-cmd-bot", daemon=True
        )
        self._thread.start()
        log.info("CommandBot thread started")

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception:
            log.exception("CommandBot crashed")

    async def _serve(self) -> None:
        app = Application.builder().token(self.token).build()
        app.add_handler(CommandHandler("start", self._cmd_status))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("pnl", self._cmd_pnl))
        app.add_handler(CommandHandler("trades", self._cmd_trades))
        app.add_handler(CommandHandler("pause", self._cmd_pause))
        app.add_handler(CommandHandler("resume", self._cmd_resume))
        app.add_handler(CommandHandler("close", self._cmd_close))
        app.add_handler(CommandHandler("sync", self._cmd_sync))
        app.add_handler(CommandHandler("testlong", self._cmd_testlong))
        app.add_handler(CommandHandler("testshort", self._cmd_testshort))
        app.add_handler(CommandHandler("help", self._cmd_help))
        self._app = app
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        # Держим loop живым.
        while True:
            await asyncio.sleep(3600)

    # ---------- handlers ----------

    def _authorized(self, update: Update) -> bool:
        cid = update.effective_chat.id if update.effective_chat else None
        if cid != self.chat_id:
            log.warning("Ignored TG msg from unauthorized chat_id=%s", cid)
            return False
        return True

    async def _cmd_help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        await update.message.reply_text(
            "/status — текущая позиция\n"
            "/pnl — статистика по сделкам\n"
            "/trades [N] — последние N сделок (default 10)\n"
            "/pause — выключить вход в новые позиции\n"
            "/resume — включить обратно\n"
            "/close — закрыть текущую позицию по рынку\n"
            "/sync — подтянуть позицию с биржи (если открывали вручную)\n"
            "/testlong [frac] — открыть тестовый long (default 5% баланса)\n"
            "/testshort [frac] — открыть тестовый short"
        )

    async def _cmd_status(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        s = self.trader.status_snapshot()
        text = (
            f"<b>{html.escape(s['symbol'])} {html.escape(s['interval'])}</b>\n"
            f"position: <code>{html.escape(s['position'])}</code>\n"
            f"entry: <code>{s['entry_price']:.2f}</code>\n"
            f"holding_time: <code>{s['holding_time']}</code> bars\n"
            f"max_drawdown: <code>{s['max_drawdown']*100:.2f}%</code>\n"
            f"last_bar: <code>{html.escape(str(s['last_bar']))}</code>\n"
            f"paused: <code>{s['paused']}</code>"
        )
        await update.message.reply_html(text)

    async def _cmd_pnl(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        if self.journal is None:
            await update.message.reply_text("journal не настроен")
            return
        st = self.journal.stats()
        await update.message.reply_html(
            f"<b>PnL</b>\n"
            f"trades: <code>{st['n_trades']}</code>\n"
            f"wins: <code>{st['wins']}</code> ({st['win_rate']*100:.1f}%)\n"
            f"total: <code>{st['total_pnl_pct']*100:+.2f}%</code>\n"
            f"avg: <code>{st['avg_pnl_pct']*100:+.2f}%</code>"
        )

    async def _cmd_trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        if self.journal is None:
            await update.message.reply_text("journal не настроен")
            return
        n = 10
        if ctx.args:
            try:
                n = max(1, min(50, int(ctx.args[0])))
            except ValueError:
                pass
        rows = self.journal.recent(n)
        if not rows:
            await update.message.reply_text("сделок пока нет")
            return
        lines = []
        for r in rows:
            pnl = (r["pnl_pct"] or 0.0) * 100
            mark = "✅" if pnl >= 0 else "🔻"
            lines.append(
                f"{mark} <code>#{r['id']}</code> {html.escape(r['side'])} "
                f"{r['entry_price']:.2f}→{r['exit_price']:.2f} "
                f"<code>{pnl:+.2f}%</code> ({html.escape(r['exit_reason'] or '')})"
            )
        await update.message.reply_html("\n".join(lines))

    async def _cmd_pause(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        self.trader.set_paused(True)
        await update.message.reply_text("⏸ paused (forced exits всё ещё работают)")

    async def _cmd_resume(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        self.trader.set_paused(False)
        await update.message.reply_text("▶ resumed")

    async def _cmd_close(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        loop = asyncio.get_event_loop()
        price = await loop.run_in_executor(None, self.trader.force_close)
        if price is None:
            await update.message.reply_text("позиции нет")
        else:
            await update.message.reply_text(f"closed @ {price:.2f}")

    async def _cmd_testlong(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await self._cmd_testopen(update, ctx, side="BUY")

    async def _cmd_testshort(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await self._cmd_testopen(update, ctx, side="SELL")

    async def _cmd_testopen(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                            side: str) -> None:
        if not self._authorized(update):
            return
        frac = 0.05
        if ctx.args:
            try:
                frac = max(0.001, min(1.0, float(ctx.args[0])))
            except ValueError:
                pass
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(
            None, lambda: self.trader.test_open(side, frac)
        )
        if "error" in res:
            await update.message.reply_text(f"❌ {res['error']}")
            return
        await update.message.reply_html(
            f"✅ test {res['side']} opened\n"
            f"qty: <code>{res['qty']:.6f}</code>\n"
            f"entry: <code>{res['price']:.2f}</code>\n"
            f"balance: <code>{res['balance_before']:.2f}</code> USDC × {frac:.0%}"
        )

    async def _cmd_sync(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, self.trader.force_resync)
        mark = "🔄 resynced" if res["changed"] else "✓ already in sync"
        await update.message.reply_html(
            f"{mark}\nposition: <code>{res['position']}</code>\n"
            f"entry: <code>{res['entry']:.2f}</code>"
        )
