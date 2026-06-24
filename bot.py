"""
AI Telegram Bot — тарифы, файлы, фото, стриминг ответов.
Стек: Python 3.10+, aiogram 3, Groq API + DeepSeek V4 API (OpenAI-совместимый), SQLite.

Что нового по сравнению с первой версией:
  • Премиум работает на DeepSeek V4 (гибрид: глобальные ключи с ротацией + ключ юзера).
  • Стриминг ответа: бот «печатает» текст вживую, редактируя сообщение.
  • Markdown → Telegram HTML (жирный, курсив, код-блоки).
  • Длинные ответы разбиваются на части (лимит Telegram 4096 символов).
  • Аккуратные карточки меню/статуса/тарифов + полоса лимита для free.
  • Меню «Настройки» с выбором модели и вставкой своего DeepSeek-ключа.
  • PDF читается через pypdf, путь к БД вынесен в конфиг (для Railway volume).
"""

import asyncio
import logging
import sqlite3
import io
import json
import time
import html
import re
import itertools
import base64
from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import (
    Message, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton,
    LabeledPrice, PreCheckoutQuery
)
import httpx

from config import (
    BOT_TOKEN, GROQ_API_KEY, BOT_USERNAME, ADMIN_ID,
    FREE_REQUESTS_PER_DAY, SUBSCRIPTION_DAYS, REFERRAL_BONUS_DAYS,
    DEEPSEEK_KEYS, DEEPSEEK_PREMIUM_MODEL, DEEPSEEK_FAST_MODEL,
    DEEPSEEK_BASE_URL, DB_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_URL = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"

# Модель для анализа фото (vision). DeepSeek в этой сборке отвечает за текст,
# а картинки на платных тарифах идут через vision-модель Groq.
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# Резервная текстовая модель, если DeepSeek-ключей нет вообще.
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"

TG_LIMIT = 4096          # жёсткий лимит Telegram
SOFT_LIMIT = 3800        # порог, после которого начинаем новое сообщение
EDIT_THROTTLE = 1.4      # не редактировать сообщение чаще, чем раз в N секунд

# ─── Тарифы ──────────────────────────────────────────────────────────────────
PLANS = {
    "free": {
        "name": "🆓 Бесплатный",
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "vision": False, "files": False, "reasoning": False,
        "price_stars": 0,
        "desc": "3 запроса в день · только текст",
    },
    "basic": {
        "name": "⚡ Базовый",
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "vision": False, "files": True, "reasoning": False,
        "price_stars": 150,
        "desc": "Безлимит · текст + файлы",
    },
    "standard": {
        "name": "🔥 Стандарт",
        "provider": "groq",
        "model": GROQ_VISION_MODEL,
        "vision": True, "files": True, "reasoning": False,
        "price_stars": 250,
        "desc": "Безлимит · файлы + фото",
    },
    "premium": {
        "name": "👑 Премиум",
        "provider": "deepseek",
        "model": DEEPSEEK_PREMIUM_MODEL,
        "vision": True, "files": True, "reasoning": True,
        "price_stars": 450,
        "desc": "DeepSeek V4 · reasoning · всё включено",
    },
}

USER_COLUMNS = [
    "user_id", "username", "plan", "sub_until", "req_today", "req_date",
    "referred_by", "referral_count", "ds_key", "ds_model",
]

# ─── Ротация глобальных ключей DeepSeek ──────────────────────────────────────
_rr = {"i": 0}


def ordered_global_keys() -> list[str]:
    """Глобальные ключи, начиная с очередного (round-robin)."""
    keys = list(DEEPSEEK_KEYS)
    if not keys:
        return []
    i = _rr["i"] % len(keys)
    _rr["i"] = (_rr["i"] + 1) % len(keys)
    return keys[i:] + keys[:i]


class ApiError(Exception):
    def __init__(self, status: int, body: str = ""):
        self.status = status
        self.body = body
        super().__init__(f"API {status}: {body[:200]}")


# ─── База данных ─────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id        INTEGER PRIMARY KEY,
            username       TEXT,
            plan           TEXT DEFAULT 'free',
            sub_until      TEXT,
            req_today      INTEGER DEFAULT 0,
            req_date       TEXT,
            referred_by    INTEGER DEFAULT NULL,
            referral_count INTEGER DEFAULT 0
        );
    """)
    # Миграция: добавляем новые колонки, если их ещё нет.
    existing = {row[1] for row in con.execute("PRAGMA table_info(users)").fetchall()}
    if "ds_key" not in existing:
        con.execute("ALTER TABLE users ADD COLUMN ds_key TEXT")
    if "ds_model" not in existing:
        con.execute("ALTER TABLE users ADD COLUMN ds_model TEXT")
    con.commit()
    con.close()


def get_user(user_id: int) -> dict | None:
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        f"SELECT {', '.join(USER_COLUMNS)} FROM users WHERE user_id=?",
        (user_id,)
    ).fetchone()
    con.close()
    if not row:
        return None
    return dict(zip(USER_COLUMNS, row))


def upsert_user(user_id: int, username: str, referred_by: int = None) -> bool:
    con = sqlite3.connect(DB_PATH)
    is_new = con.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone() is None
    con.execute(
        "INSERT OR IGNORE INTO users (user_id, username, plan, req_today, req_date, referred_by) "
        "VALUES (?,?,'free',0,?,?)",
        (user_id, username, str(datetime.now().date()), referred_by)
    )
    con.execute("UPDATE users SET username=? WHERE user_id=?", (username, user_id))
    con.commit()
    con.close()
    return is_new


def set_user_field(user_id: int, field: str, value):
    if field not in ("ds_key", "ds_model"):
        raise ValueError("forbidden field")
    con = sqlite3.connect(DB_PATH)
    con.execute(f"UPDATE users SET {field}=? WHERE user_id=?", (value, user_id))
    con.commit()
    con.close()


def is_subscribed(user: dict) -> bool:
    if not user or not user.get("sub_until") or user.get("plan") == "free":
        return False
    return datetime.fromisoformat(user["sub_until"]) > datetime.now()


def check_and_inc_free(user_id: int) -> bool:
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT req_today, req_date FROM users WHERE user_id=?", (user_id,)).fetchone()
    today = str(datetime.now().date())
    if not row:
        con.close()
        return False
    req_today, req_date = row
    if req_date != today:
        req_today = 0
    if req_today >= FREE_REQUESTS_PER_DAY:
        con.close()
        return False
    con.execute("UPDATE users SET req_today=?, req_date=? WHERE user_id=?", (req_today + 1, today, user_id))
    con.commit()
    con.close()
    return True


def free_used_today(user: dict) -> int:
    today = str(datetime.now().date())
    return user["req_today"] if user and user.get("req_date") == today else 0


def activate_plan(user_id: int, plan: str, days: int) -> str:
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT sub_until, plan FROM users WHERE user_id=?", (user_id,)).fetchone()
    if row and row[0] and row[1] == plan and datetime.fromisoformat(row[0]) > datetime.now():
        until = (datetime.fromisoformat(row[0]) + timedelta(days=days)).isoformat()
    else:
        until = (datetime.now() + timedelta(days=days)).isoformat()
    con.execute("UPDATE users SET sub_until=?, plan=? WHERE user_id=?", (until, plan, user_id))
    con.commit()
    con.close()
    return until


def add_referral_bonus(referrer_id: int) -> str:
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE users SET referral_count = referral_count + 1 WHERE user_id=?", (referrer_id,))
    con.commit()
    con.close()
    user = get_user(referrer_id)
    plan = user["plan"] if user and user["plan"] != "free" else "basic"
    return activate_plan(referrer_id, plan, REFERRAL_BONUS_DAYS)


def get_stats() -> dict:
    con = sqlite3.connect(DB_PATH)
    total = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    now = datetime.now().isoformat()
    subs = con.execute("SELECT COUNT(*) FROM users WHERE sub_until > ? AND plan != 'free'", (now,)).fetchone()[0]
    by_plan = {}
    for plan in ["basic", "standard", "premium"]:
        by_plan[plan] = con.execute(
            "SELECT COUNT(*) FROM users WHERE plan=? AND sub_until > ?", (plan, now)
        ).fetchone()[0]
    con.close()
    return {"total": total, "subs": subs, "by_plan": by_plan}


def is_admin(user_id: int) -> bool:
    return user_id == ADMIN_ID


def get_user_plan(user: dict) -> dict:
    if not user:
        return PLANS["free"]
    if is_subscribed(user):
        return PLANS.get(user["plan"], PLANS["free"])
    return PLANS["free"]


# ─── Форматирование текста ───────────────────────────────────────────────────
def md_to_html(text: str) -> str:
    """Аккуратно конвертирует Markdown от модели в безопасный Telegram HTML."""
    if not text:
        return ""
    blocks: list[str] = []

    def keep(s: str) -> str:
        blocks.append(s)
        return f"\uffff{len(blocks) - 1}\uffff"

    # fenced ```code```
    text = re.sub(
        r"```[ \t]*[\w+\-]*\n?(.*?)```",
        lambda m: keep("<pre><code>" + html.escape(m.group(1)) + "</code></pre>"),
        text, flags=re.DOTALL,
    )
    # inline `code`
    text = re.sub(
        r"`([^`\n]+?)`",
        lambda m: keep("<code>" + html.escape(m.group(1)) + "</code>"),
        text,
    )
    # экранируем всё остальное
    text = html.escape(text)
    # заголовки -> жирный
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*(.+?)\s*#*$", r"<b>\1</b>", text)
    # **жирный** / __жирный__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\w)__(.+?)__(?!\w)", r"<b>\1</b>", text)
    # *курсив* / _курсив_  (только если рядом не пробел и не буква/звёздочка)
    text = re.sub(r"(?<![\*\w])\*(?=\S)(.+?)(?<=\S)\*(?![\*\w])", r"<i>\1</i>", text)
    text = re.sub(r"(?<![_\w])_(?=\S)(.+?)(?<=\S)_(?![_\w])", r"<i>\1</i>", text)
    # ссылки [text](url)
    text = re.sub(r"\[([^\]]+?)\]\((https?://[^\s)]+)\)", r'<a href="\2">\1</a>', text)
    # маркеры списка
    text = re.sub(r"(?m)^\s*[-*]\s+", "• ", text)
    # возвращаем код на место
    text = re.sub(r"\uffff(\d+)\uffff", lambda m: blocks[int(m.group(1))], text)
    return text


def split_text(text: str, limit: int = TG_LIMIT - 200) -> list[str]:
    """Разбивает длинный текст на части по границам строк."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            parts.append(line[:limit])
            line = line[limit:]
        if len(cur) + len(line) + 1 > limit:
            if cur:
                parts.append(cur)
            cur = line
        else:
            cur = (cur + "\n" + line) if cur else line
    if cur:
        parts.append(cur)
    return parts


async def safe_answer(message: Message, text: str, **kwargs):
    """Отправляет ответ с HTML, при ошибке парсинга — обычным текстом, и режет на части."""
    chunks = split_text(text)
    for chunk in chunks:
        try:
            await message.answer(md_to_html(chunk), parse_mode="HTML", **kwargs)
        except TelegramBadRequest:
            await message.answer(chunk, **kwargs)


def usage_bar(used: int, total: int, n: int = 5) -> str:
    filled = min(n, round((used / total) * n)) if total else 0
    return "▓" * filled + "░" * (n - filled) + f"  {used}/{total}"


# ─── Стриминг ответа в одно (или несколько) сообщений ────────────────────────
class TgStreamer:
    """Печатает ответ модели вживую, редактируя сообщение с троттлингом."""

    def __init__(self, placeholder: Message):
        self.current = placeholder
        self.buffer = ""          # текст текущего сообщения
        self.full = ""            # весь ответ целиком
        self.last_edit = 0.0
        self.last_render = None
        self.dirty = False

    @property
    def has_output(self) -> bool:
        return bool(self.full)

    async def push(self, delta: str):
        self.full += delta
        if len(self.buffer) + len(delta) > SOFT_LIMIT:
            await self._flush(force=True)
            self.current = await self.current.answer("…")
            self.buffer = ""
            self.last_render = None
        self.buffer += delta
        self.dirty = True
        now = time.monotonic()
        if now - self.last_edit >= EDIT_THROTTLE:
            await self._flush()

    async def _flush(self, force: bool = False):
        if not self.dirty or not self.buffer.strip():
            return
        rendered = md_to_html(self.buffer)
        if rendered == self.last_render:
            self.dirty = False
            return
        try:
            await self.current.edit_text(rendered, parse_mode="HTML")
            self.last_render = rendered
        except TelegramRetryAfter as e:
            await asyncio.sleep(e.retry_after + 0.5)
        except TelegramBadRequest:
            try:
                await self.current.edit_text(self.buffer)
                self.last_render = None
            except TelegramBadRequest:
                pass
        self.last_edit = time.monotonic()
        self.dirty = False

    async def finish(self):
        self.dirty = True
        await self._flush(force=True)

    async def fail(self, text: str):
        try:
            await self.current.edit_text(text)
        except TelegramBadRequest:
            await self.current.answer(text)


# ─── Запросы к моделям (OpenAI-совместимый стриминг) ─────────────────────────
async def stream_completion(provider: str, model: str, messages: list,
                            on_delta, api_key: str = None, reasoning: bool = False) -> str:
    if provider == "groq":
        url, key = GROQ_URL, GROQ_API_KEY
    else:
        url, key = DEEPSEEK_URL, api_key

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": 2048, "stream": True}
    if provider == "deepseek" and reasoning:
        payload["reasoning_effort"] = "high"

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=20.0)) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", "ignore")
                raise ApiError(resp.status_code, body)
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {})
                    piece = delta.get("content")  # reasoning_content намеренно не показываем
                    if piece:
                        await on_delta(piece)
                except Exception:
                    continue
    return ""


def resolve_text_engine(user: dict, plan: dict, admin: bool):
    """Возвращает (provider, model, api_key, reasoning) для текстового ответа."""
    wants_deepseek = admin or plan["provider"] == "deepseek"
    if wants_deepseek:
        if user and user.get("ds_key"):
            ds_model = user.get("ds_model") or DEEPSEEK_PREMIUM_MODEL
            return "deepseek", ds_model, [user["ds_key"]], True
        keys = ordered_global_keys()
        if keys:
            ds_model = (user.get("ds_model") if user else None) or DEEPSEEK_PREMIUM_MODEL
            return "deepseek", ds_model, keys, True
        # ключей нет вообще — отвечаем на резервной модели Groq
        return "groq", GROQ_FALLBACK_MODEL, None, False
    return "groq", plan["model"], None, False


async def run_text(msg: Message, messages: list, user: dict, plan: dict, admin: bool):
    provider, model, keys, reasoning = resolve_text_engine(user, plan, admin)
    placeholder = await msg.answer("🧠 Думаю…" if reasoning else "✍️ Печатаю…")
    streamer = TgStreamer(placeholder)

    candidates = keys if provider == "deepseek" else [None]
    last_err = None
    for key in candidates:
        try:
            await stream_completion(provider, model, messages, streamer.push,
                                    api_key=key, reasoning=reasoning)
            await streamer.finish()
            return
        except (ApiError, httpx.HTTPError) as e:
            last_err = e
            if streamer.has_output:   # уже что-то напечатали — не повторяем с другим ключом
                await streamer.finish()
                return
            continue

    log.error(f"AI error: {last_err}")
    if isinstance(last_err, ApiError) and last_err.status in (401, 402):
        await streamer.fail("⚠️ Ключ DeepSeek недействителен или закончился баланс. "
                            "Проверь ключ в ⚙️ Настройках или обратись к админу.")
    elif isinstance(last_err, ApiError) and last_err.status == 429:
        await streamer.fail("⏳ Сейчас слишком много запросов к DeepSeek (лимит ключа). "
                            "Подожди минуту и попробуй снова.")
    else:
        await streamer.fail("⚠️ Не получилось получить ответ от AI. Попробуй ещё раз чуть позже.")


# ─── Vision (анализ фото) ────────────────────────────────────────────────────
async def run_vision(msg: Message, caption: str, image_b64: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": caption or "Опиши, что на изображении"},
        ],
    }]
    placeholder = await msg.answer("🖼 Смотрю на фото…")
    streamer = TgStreamer(placeholder)
    try:
        await stream_completion("groq", GROQ_VISION_MODEL, messages, streamer.push)
        await streamer.finish()
    except (ApiError, httpx.HTTPError) as e:
        log.error(f"Vision error: {e}")
        await streamer.fail("⚠️ Не удалось разобрать изображение. Попробуй другое фото.")


# ─── Извлечение текста из файла ──────────────────────────────────────────────
async def extract_text_from_file(msg: Message, bot: Bot) -> str:
    doc = msg.document
    file = await bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, buf)
    content = buf.getvalue()

    if doc.mime_type == "text/plain" or (doc.file_name or "").lower().endswith(".txt"):
        try:
            return content.decode("utf-8")[:8000]
        except UnicodeDecodeError:
            return content.decode("latin-1", "ignore")[:8000]

    if doc.mime_type == "application/pdf" or (doc.file_name or "").lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            pages = [(p.extract_text() or "") for p in reader.pages[:20]]
            result = "\n".join(pages).strip()
            if len(result) > 30:
                return result[:8000]
        except Exception as e:
            log.error(f"PDF error: {e}")
        return "⚠️ Не удалось извлечь текст из PDF (возможно, это скан без текстового слоя)."

    return f"⚠️ Тип файла «{doc.mime_type}» не поддерживается. Поддерживаются: TXT, PDF."


# ─── Клавиатуры ──────────────────────────────────────────────────────────────
def kb_main(user_id: int) -> InlineKeyboardMarkup:
    buttons = []
    if is_admin(user_id):
        buttons.append([InlineKeyboardButton(text="👑 Админ-панель", callback_data="admin_panel")])
    buttons.extend([
        [InlineKeyboardButton(text="💎 Тарифы и подписка", callback_data="plans")],
        [InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings")],
        [InlineKeyboardButton(text="👥 Реферальная программа", callback_data="referral")],
        [InlineKeyboardButton(text="📊 Мой статус", callback_data="status")],
    ])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def kb_plans() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⚡ Базовый — 150 ⭐/мес", callback_data="buy_basic")],
        [InlineKeyboardButton(text="🔥 Стандарт — 250 ⭐/мес", callback_data="buy_standard")],
        [InlineKeyboardButton(text="👑 Премиум — 450 ⭐/мес", callback_data="buy_premium")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="back_main")],
    ])


def kb_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Назад", callback_data="back_main")]
    ])


def kb_admin() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Статистика", callback_data="admin_stats")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="back_main")],
    ])


def kb_settings(user: dict, admin: bool) -> InlineKeyboardMarkup:
    rows = []
    can_deepseek = admin or (is_subscribed(user) and user.get("plan") == "premium")
    if can_deepseek:
        cur_model = (user.get("ds_model") or DEEPSEEK_PREMIUM_MODEL) if user else DEEPSEEK_PREMIUM_MODEL
        pro_mark = "✅" if cur_model == DEEPSEEK_PREMIUM_MODEL else "▫️"
        fast_mark = "✅" if cur_model == DEEPSEEK_FAST_MODEL else "▫️"
        rows.append([InlineKeyboardButton(text=f"{pro_mark} Pro (умнее)", callback_data="model_pro"),
                     InlineKeyboardButton(text=f"{fast_mark} Flash (быстрее)", callback_data="model_fast")])
        if user and user.get("ds_key"):
            rows.append([InlineKeyboardButton(text="🔑 Заменить мой ключ", callback_data="set_ds_key")])
            rows.append([InlineKeyboardButton(text="🗑 Удалить мой ключ", callback_data="del_ds_key")])
        else:
            rows.append([InlineKeyboardButton(text="🔑 Вставить свой DeepSeek-ключ", callback_data="set_ds_key")])
    rows.append([InlineKeyboardButton(text="◀️ Назад", callback_data="back_main")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ─── FSM состояния ───────────────────────────────────────────────────────────
class SettingsSG(StatesGroup):
    waiting_key = State()


# ─── Тексты-карточки ─────────────────────────────────────────────────────────
def welcome_text(name: str) -> str:
    return (
        f"👋 Привет, <b>{html.escape(name)}</b>!\n\n"
        f"Я — AI-ассистент с выбором модели 🤖\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🆓 <b>Бесплатно</b> — 3 запроса/день\n"
        f"⚡ <b>Базовый</b> — безлимит + файлы\n"
        f"🔥 <b>Стандарт</b> — + анализ фото\n"
        f"👑 <b>Премиум</b> — DeepSeek V4, reasoning, всё включено\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Просто напиши мне что-нибудь — отвечу 👇"
    )


dp = Dispatcher(storage=MemoryStorage())


# ─── /start ──────────────────────────────────────────────────────────────────
@dp.message(CommandStart())
async def cmd_start(msg: Message, state: FSMContext):
    await state.clear()
    user_id = msg.from_user.id
    username = msg.from_user.username or ""
    referred_by = None
    args = msg.text.split()
    if len(args) > 1 and args[1].startswith("ref_"):
        try:
            ref_id = int(args[1].split("_")[1])
            if ref_id != user_id:
                referred_by = ref_id
        except (ValueError, IndexError):
            pass

    is_new = upsert_user(user_id, username, referred_by)

    if is_new and referred_by:
        if get_user(referred_by):
            until = add_referral_bonus(referred_by)
            until_str = datetime.fromisoformat(until).strftime("%d.%m.%Y")
            try:
                await msg.bot.send_message(
                    referred_by,
                    f"🎉 По твоей ссылке пришёл новый пользователь!\n"
                    f"✅ Начислено <b>+{REFERRAL_BONUS_DAYS} дней</b> подписки (до {until_str})",
                    parse_mode="HTML",
                )
            except Exception:
                pass

    await msg.answer(welcome_text(msg.from_user.first_name), parse_mode="HTML",
                     reply_markup=kb_main(user_id))


# ─── Тарифы ──────────────────────────────────────────────────────────────────
@dp.callback_query(F.data == "plans")
async def cb_plans(cb: CallbackQuery):
    await cb.message.edit_text(
        "💎 <b>Тарифы</b>\n"
        "━━━━━━━━━━━━━━━\n"
        "⚡ <b>Базовый</b> — 150 ⭐/мес\n"
        "  Llama 3.3 70B · безлимит · файлы\n\n"
        "🔥 <b>Стандарт</b> — 250 ⭐/мес\n"
        "  + анализ фото\n\n"
        "👑 <b>Премиум</b> — 450 ⭐/мес\n"
        "  DeepSeek V4 (reasoning) · топ-качество\n"
        "  можно подключить свой DeepSeek-ключ\n"
        "━━━━━━━━━━━━━━━\n"
        "Оплата — звёздами Telegram ⭐",
        parse_mode="HTML", reply_markup=kb_plans(),
    )
    await cb.answer()


@dp.callback_query(F.data.startswith("buy_"))
async def cb_buy(cb: CallbackQuery, bot: Bot):
    plan_key = cb.data.replace("buy_", "")
    plan = PLANS.get(plan_key)
    if not plan or plan_key == "free":
        await cb.answer("❌ Тариф не найден", show_alert=True)
        return
    await bot.send_invoice(
        chat_id=cb.from_user.id,
        title=f"{plan['name']} — AI подписка",
        description=plan["desc"],
        payload=f"plan_{plan_key}",
        currency="XTR",
        prices=[LabeledPrice(label=plan["name"], amount=plan["price_stars"])],
    )
    await cb.answer()


@dp.pre_checkout_query()
async def pre_checkout(query: PreCheckoutQuery, bot: Bot):
    await bot.answer_pre_checkout_query(query.id, ok=True)


@dp.message(F.successful_payment)
async def payment_success(msg: Message):
    payload = msg.successful_payment.invoice_payload
    plan_key = payload.replace("plan_", "")
    plan = PLANS.get(plan_key, PLANS["basic"])
    until = activate_plan(msg.from_user.id, plan_key, SUBSCRIPTION_DAYS)
    until_str = datetime.fromisoformat(until).strftime("%d.%m.%Y")
    await msg.answer(
        f"🎉 <b>Тариф {plan['name']} активирован!</b>\n"
        f"Действует до {until_str}\n\n"
        f"Модель: <b>{plan['model']}</b>\n"
        f"{plan['desc']}",
        parse_mode="HTML", reply_markup=kb_main(msg.from_user.id),
    )


# ─── Статус ──────────────────────────────────────────────────────────────────
@dp.callback_query(F.data == "status")
async def cb_status(cb: CallbackQuery):
    user_id = cb.from_user.id
    user = get_user(user_id)
    if not user:
        upsert_user(user_id, cb.from_user.username or "")
        user = get_user(user_id)

    plan = get_user_plan(user)
    if is_admin(user_id):
        status = "👑 <b>Администратор</b>\n♾️ Безлимит · DeepSeek V4"
    elif is_subscribed(user):
        until = datetime.fromisoformat(user["sub_until"]).strftime("%d.%m.%Y")
        status = f"{plan['name']}\nДо: <b>{until}</b>\nМодель: <code>{plan['model']}</code>"
    else:
        used = free_used_today(user)
        status = (f"🆓 <b>Бесплатный план</b>\n"
                  f"Лимит на сегодня:\n<code>{usage_bar(used, FREE_REQUESTS_PER_DAY)}</code>")

    await cb.message.edit_text(
        f"📊 <b>Твой статус</b>\n━━━━━━━━━━━━━━━\n{status}\n━━━━━━━━━━━━━━━\n"
        f"👥 Приглашено друзей: <b>{user['referral_count']}</b>",
        parse_mode="HTML", reply_markup=kb_main(user_id),
    )
    await cb.answer()


# ─── Настройки ───────────────────────────────────────────────────────────────
@dp.callback_query(F.data == "settings")
async def cb_settings(cb: CallbackQuery):
    user_id = cb.from_user.id
    user = get_user(user_id)
    if not user:
        upsert_user(user_id, cb.from_user.username or "")
        user = get_user(user_id)
    admin = is_admin(user_id)
    can_deepseek = admin or (is_subscribed(user) and user.get("plan") == "premium")

    if not can_deepseek:
        await cb.message.edit_text(
            "⚙️ <b>Настройки</b>\n━━━━━━━━━━━━━━━\n"
            "Выбор модели и подключение своего DeepSeek-ключа доступны "
            "на тарифе 👑 <b>Премиум</b>.",
            parse_mode="HTML", reply_markup=kb_back(),
        )
        await cb.answer()
        return

    cur_model = (user.get("ds_model") or DEEPSEEK_PREMIUM_MODEL)
    key_state = "свой ключ подключён 🔑" if user.get("ds_key") else "используются ключи бота"
    await cb.message.edit_text(
        "⚙️ <b>Настройки (Премиум)</b>\n━━━━━━━━━━━━━━━\n"
        f"Текущая модель: <code>{cur_model}</code>\n"
        f"Ключ: {key_state}\n━━━━━━━━━━━━━━━\n"
        "• <b>Pro</b> — максимально умная, но медленнее (reasoning)\n"
        "• <b>Flash</b> — быстрее и легче по лимитам\n\n"
        "Можешь вставить свой DeepSeek-ключ — тогда запросы пойдут через него.",
        parse_mode="HTML", reply_markup=kb_settings(user, admin),
    )
    await cb.answer()


@dp.callback_query(F.data == "model_pro")
async def cb_model_pro(cb: CallbackQuery):
    set_user_field(cb.from_user.id, "ds_model", DEEPSEEK_PREMIUM_MODEL)
    await cb.answer("Модель: Pro (умнее) ✅")
    await cb_settings(cb)


@dp.callback_query(F.data == "model_fast")
async def cb_model_fast(cb: CallbackQuery):
    set_user_field(cb.from_user.id, "ds_model", DEEPSEEK_FAST_MODEL)
    await cb.answer("Модель: Flash (быстрее) ✅")
    await cb_settings(cb)


@dp.callback_query(F.data == "del_ds_key")
async def cb_del_key(cb: CallbackQuery):
    set_user_field(cb.from_user.id, "ds_key", None)
    await cb.answer("Ключ удалён, вернулись на ключи бота ✅")
    await cb_settings(cb)


@dp.callback_query(F.data == "set_ds_key")
async def cb_set_key(cb: CallbackQuery, state: FSMContext):
    user = get_user(cb.from_user.id)
    admin = is_admin(cb.from_user.id)
    can_deepseek = admin or (is_subscribed(user) and user.get("plan") == "premium")
    if not can_deepseek:
        await cb.answer("Доступно на Премиуме", show_alert=True)
        return
    await state.set_state(SettingsSG.waiting_key)
    await cb.message.answer(
        "🔑 Пришли свой ключ DeepSeek одним сообщением (начинается с <code>sk-</code>).\n"
        "Получить можно на platform.deepseek.com.\n\n"
        "Чтобы отменить — отправь /cancel",
        parse_mode="HTML",
    )
    await cb.answer()


@dp.message(Command("cancel"))
async def cmd_cancel(msg: Message, state: FSMContext):
    if await state.get_state():
        await state.clear()
        await msg.answer("Отменено.", reply_markup=kb_main(msg.from_user.id))


@dp.message(StateFilter(SettingsSG.waiting_key), F.text)
async def receive_ds_key(msg: Message, state: FSMContext):
    key = msg.text.strip()
    await state.clear()
    # пробуем удалить сообщение с ключом, чтобы оно не висело в истории
    try:
        await msg.delete()
    except TelegramBadRequest:
        pass
    if not key.startswith("sk-") or len(key) < 12:
        await msg.answer("❌ Это не похоже на ключ DeepSeek (должен начинаться с sk-). Попробуй ещё раз через ⚙️ Настройки.")
        return
    set_user_field(msg.from_user.id, "ds_key", key)
    await msg.answer("✅ Ключ сохранён. Теперь Премиум-запросы идут через твой ключ.\n"
                     "(Сообщение с ключом я удалил из чата.)",
                     reply_markup=kb_main(msg.from_user.id))


# ─── Реферальная программа / навигация ───────────────────────────────────────
@dp.callback_query(F.data == "referral")
async def cb_referral(cb: CallbackQuery):
    user = get_user(cb.from_user.id) or {"referral_count": 0}
    ref_link = f"https://t.me/{BOT_USERNAME}?start=ref_{cb.from_user.id}"
    await cb.message.edit_text(
        f"👥 <b>Реферальная программа</b>\n━━━━━━━━━━━━━━━\n"
        f"За каждого друга: <b>+{REFERRAL_BONUS_DAYS} дней</b> подписки\n"
        f"Приглашено: <b>{user['referral_count']} чел.</b>\n━━━━━━━━━━━━━━━\n"
        f"Твоя ссылка:\n<code>{ref_link}</code>",
        parse_mode="HTML", reply_markup=kb_back(),
    )
    await cb.answer()


@dp.callback_query(F.data == "back_main")
async def cb_back(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await cb.message.edit_text("🏠 Главное меню", reply_markup=kb_main(cb.from_user.id))
    await cb.answer()


# ─── Админка ─────────────────────────────────────────────────────────────────
@dp.callback_query(F.data == "admin_panel")
async def cb_admin_panel(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        await cb.answer("⛔ Нет доступа", show_alert=True)
        return
    stats = get_stats()
    keys_note = f"{len(DEEPSEEK_KEYS)} шт." if DEEPSEEK_KEYS else "не заданы ⚠️"
    await cb.message.edit_text(
        f"👑 <b>Админ-панель</b>\n━━━━━━━━━━━━━━━\n"
        f"👥 Всего пользователей: <b>{stats['total']}</b>\n"
        f"💎 Активных подписок: <b>{stats['subs']}</b>\n\n"
        f"⚡ Базовых: <b>{stats['by_plan']['basic']}</b>\n"
        f"🔥 Стандарт: <b>{stats['by_plan']['standard']}</b>\n"
        f"👑 Премиум: <b>{stats['by_plan']['premium']}</b>\n\n"
        f"🔑 Глобальных DeepSeek-ключей: <b>{keys_note}</b>\n━━━━━━━━━━━━━━━\n"
        f"Выдать подписку:\n<code>/give USER_ID план дней</code>\n"
        f"Планы: basic, standard, premium\n"
        f"Пример: <code>/give 123456789 premium 30</code>",
        parse_mode="HTML", reply_markup=kb_admin(),
    )
    await cb.answer()


@dp.callback_query(F.data == "admin_stats")
async def cb_admin_stats(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        await cb.answer("⛔ Нет доступа", show_alert=True)
        return
    stats = get_stats()
    await cb.answer(f"👥 Пользователей: {stats['total']}\n💎 Подписок: {stats['subs']}", show_alert=True)


@dp.message(Command("give"))
async def cmd_give(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    args = msg.text.split()
    if len(args) != 4:
        await msg.answer("Использование: /give [user_id] [план] [дней]\nПример: /give 123456789 premium 30")
        return
    try:
        target_id, plan_key, days = int(args[1]), args[2], int(args[3])
    except ValueError:
        await msg.answer("❌ Неверный формат.")
        return
    if plan_key not in PLANS or plan_key == "free":
        await msg.answer("❌ Неверный план. Доступны: basic, standard, premium")
        return
    if not get_user(target_id):
        await msg.answer(f"❌ Пользователь {target_id} не найден.")
        return
    until = activate_plan(target_id, plan_key, days)
    until_str = datetime.fromisoformat(until).strftime("%d.%m.%Y")
    plan = PLANS[plan_key]
    await msg.answer(f"✅ Пользователю {target_id} выдан тариф {plan['name']} до {until_str}")
    try:
        await msg.bot.send_message(target_id, f"🎁 Тебе выдан тариф {plan['name']} до {until_str}!")
    except Exception:
        pass


# ─── Контент: фото / документы / текст ───────────────────────────────────────
@dp.message(F.photo)
async def handle_photo(msg: Message, bot: Bot):
    user_id = msg.from_user.id
    upsert_user(user_id, msg.from_user.username or "")
    user = get_user(user_id)
    admin = is_admin(user_id)
    plan = PLANS["premium"] if admin else get_user_plan(user)

    if not admin and not is_subscribed(user):
        await msg.answer("📸 Анализ фото доступен на платных тарифах.\n\nОформи подписку 👇",
                         reply_markup=kb_plans())
        return
    if not admin and not plan["vision"]:
        await msg.answer(f"📸 Фото недоступно на тарифе {plan['name']}.\nНужен 🔥 Стандарт или 👑 Премиум.",
                         reply_markup=kb_plans())
        return

    await bot.send_chat_action(msg.chat.id, "typing")
    photo = msg.photo[-1]
    file = await bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, buf)
    image_b64 = base64.b64encode(buf.getvalue()).decode()
    await run_vision(msg, msg.caption or "Опиши, что на изображении", image_b64)


@dp.message(F.document)
async def handle_document(msg: Message, bot: Bot):
    user_id = msg.from_user.id
    upsert_user(user_id, msg.from_user.username or "")
    user = get_user(user_id)
    admin = is_admin(user_id)
    plan = PLANS["premium"] if admin else get_user_plan(user)

    if not admin and not is_subscribed(user):
        await msg.answer("📄 Работа с файлами доступна на платных тарифах.\n\nОформи подписку 👇",
                         reply_markup=kb_plans())
        return
    if not admin and not plan["files"]:
        await msg.answer(f"📄 Файлы недоступны на тарифе {plan['name']}.", reply_markup=kb_plans())
        return

    await bot.send_chat_action(msg.chat.id, "typing")
    text = await extract_text_from_file(msg, bot)
    if text.startswith("⚠️"):
        await msg.answer(text)
        return

    caption = msg.caption or "Проанализируй этот текст и кратко изложи суть."
    messages = [{"role": "user", "content": f"{caption}\n\n---\n{text}"}]
    await run_text(msg, messages, user, plan, admin)


@dp.message(StateFilter(None), F.text & ~F.text.startswith("/"))
async def handle_message(msg: Message):
    user_id = msg.from_user.id
    upsert_user(user_id, msg.from_user.username or "")
    user = get_user(user_id)
    admin = is_admin(user_id)

    if not admin:
        plan = get_user_plan(user)
        if not is_subscribed(user):
            if not check_and_inc_free(user_id):
                await msg.answer(
                    f"⛔ Бесплатный лимит исчерпан ({FREE_REQUESTS_PER_DAY} запроса/день).\n\n"
                    f"Выбери тариф для продолжения 👇",
                    reply_markup=kb_plans(),
                )
                return
    else:
        plan = PLANS["premium"]

    await msg.bot.send_chat_action(msg.chat.id, "typing")
    messages = [{"role": "user", "content": msg.text}]
    await run_text(msg, messages, user, plan, admin)


# ─── Запуск ──────────────────────────────────────────────────────────────────
async def main():
    init_db()
    if not DEEPSEEK_KEYS:
        log.warning("DEEPSEEK_KEYS не заданы — Премиум будет отвечать на резервной модели Groq.")
    bot = Bot(token=BOT_TOKEN)
    log.info("Бот запущен ✅")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
