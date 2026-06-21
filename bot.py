"""
AI Telegram Bot с тарифами, файлами и фото
Стек: Python 3.10+, aiogram 3, Groq API, SQLite
"""

import asyncio
import logging
import sqlite3
import io
import base64
from datetime import datetime, timedelta

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton,
    LabeledPrice, PreCheckoutQuery
)
from aiogram.fsm.storage.memory import MemoryStorage
import httpx

from config import (
    BOT_TOKEN, GROQ_API_KEY, BOT_USERNAME, ADMIN_ID,
    FREE_REQUESTS_PER_DAY, SUBSCRIPTION_DAYS, REFERRAL_BONUS_DAYS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Тарифы ──────────────────────────────────────────────────────────────────
PLANS = {
    "free": {
        "name": "🆓 Бесплатный",
        "model": "llama-3.1-8b-instant",
        "vision": False,
        "files": False,
        "price_stars": 0,
        "desc": "3 запроса/день, только текст"
    },
    "basic": {
        "name": "⚡ Базовый",
        "model": "llama-3.3-70b-versatile",
        "vision": False,
        "files": True,
        "price_stars": 150,
        "desc": "Безлимит, текст + файлы"
    },
    "standard": {
        "name": "🔥 Стандарт",
        "model": "llama-3.2-11b-vision-preview",
        "vision": True,
        "files": True,
        "price_stars": 250,
        "desc": "Безлимит, текст + файлы + фото"
    },
    "premium": {
        "name": "👑 Премиум",
        "model": "llama-3.2-90b-vision-preview",
        "vision": True,
        "files": True,
        "price_stars": 450,
        "desc": "Безлимит, всё + максимум умности"
    }
}

# ─── База данных ─────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect("users.db")
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
    con.commit()
    con.close()

def get_user(user_id: int) -> dict | None:
    con = sqlite3.connect("users.db")
    row = con.execute(
        "SELECT user_id, username, plan, sub_until, req_today, req_date, referred_by, referral_count FROM users WHERE user_id=?",
        (user_id,)
    ).fetchone()
    con.close()
    if not row:
        return None
    return dict(zip(["user_id", "username", "plan", "sub_until", "req_today", "req_date", "referred_by", "referral_count"], row))

def upsert_user(user_id: int, username: str, referred_by: int = None):
    con = sqlite3.connect("users.db")
    is_new = con.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone() is None
    con.execute(
        "INSERT OR IGNORE INTO users (user_id, username, plan, req_today, req_date, referred_by) VALUES (?,?,'free',0,?,?)",
        (user_id, username, str(datetime.now().date()), referred_by)
    )
    con.execute("UPDATE users SET username=? WHERE user_id=?", (username, user_id))
    con.commit()
    con.close()
    return is_new

def is_subscribed(user: dict) -> bool:
    if not user or not user["sub_until"] or user["plan"] == "free":
        return False
    return datetime.fromisoformat(user["sub_until"]) > datetime.now()

def check_and_inc_free(user_id: int) -> bool:
    con = sqlite3.connect("users.db")
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

def activate_plan(user_id: int, plan: str, days: int) -> str:
    con = sqlite3.connect("users.db")
    user = con.execute("SELECT sub_until, plan FROM users WHERE user_id=?", (user_id,)).fetchone()
    if user and user[0] and user[1] == plan and datetime.fromisoformat(user[0]) > datetime.now():
        until = (datetime.fromisoformat(user[0]) + timedelta(days=days)).isoformat()
    else:
        until = (datetime.now() + timedelta(days=days)).isoformat()
    con.execute("UPDATE users SET sub_until=?, plan=? WHERE user_id=?", (until, plan, user_id))
    con.commit()
    con.close()
    return until

def add_referral_bonus(referrer_id: int) -> str:
    con = sqlite3.connect("users.db")
    con.execute("UPDATE users SET referral_count = referral_count + 1 WHERE user_id=?", (referrer_id,))
    con.commit()
    con.close()
    user = get_user(referrer_id)
    plan = user["plan"] if user and user["plan"] != "free" else "basic"
    return activate_plan(referrer_id, plan, REFERRAL_BONUS_DAYS)

def get_stats() -> dict:
    con = sqlite3.connect("users.db")
    total = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    subs = con.execute("SELECT COUNT(*) FROM users WHERE sub_until > ? AND plan != 'free'", (datetime.now().isoformat(),)).fetchone()[0]
    by_plan = {}
    for plan in ["basic", "standard", "premium"]:
        count = con.execute("SELECT COUNT(*) FROM users WHERE plan=? AND sub_until > ?", (plan, datetime.now().isoformat())).fetchone()[0]
        by_plan[plan] = count
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

# ─── Клавиатуры ──────────────────────────────────────────────────────────────
def kb_main(user_id: int) -> InlineKeyboardMarkup:
    buttons = []
    if is_admin(user_id):
        buttons.append([InlineKeyboardButton(text="👑 Админ-панель", callback_data="admin_panel")])
    buttons.extend([
        [InlineKeyboardButton(text="💎 Тарифы и подписка", callback_data="plans")],
        [InlineKeyboardButton(text="👥 Реферальная программа", callback_data="referral")],
        [InlineKeyboardButton(text="📊 Мой статус", callback_data="status")],
    ])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def kb_plans() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⚡ Базовый — 150 Stars/мес", callback_data="buy_basic")],
        [InlineKeyboardButton(text="🔥 Стандарт — 250 Stars/мес", callback_data="buy_standard")],
        [InlineKeyboardButton(text="👑 Премиум — 450 Stars/мес", callback_data="buy_premium")],
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

# ─── AI запросы ──────────────────────────────────────────────────────────────
async def ask_ai(text: str, model: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": model, "max_tokens": 1024, "messages": [{"role": "user", "content": text}]}
            )
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"Groq error: {e}")
        return "⚠️ Ошибка при обращении к AI. Попробуй чуть позже."

async def ask_ai_vision(text: str, image_b64: str, model: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "max_tokens": 1024,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                            {"type": "text", "text": text or "Опиши что на изображении"}
                        ]
                    }]
                }
            )
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"Groq vision error: {e}")
        return "⚠️ Ошибка при анализе изображения."

async def extract_text_from_file(msg: Message, bot: Bot) -> str:
    """Извлекает текст из документа (PDF, TXT и др.)"""
    doc = msg.document
    file = await bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, buf)
    buf.seek(0)
    content = buf.read()

    # TXT файлы
    if doc.mime_type == "text/plain":
        try:
            return content.decode("utf-8")[:4000]
        except:
            return content.decode("latin-1")[:4000]

    # PDF файлы
    if doc.mime_type == "application/pdf":
        try:
            import re
            text = content.decode("latin-1", errors="ignore")
            # Простое извлечение текста из PDF
            parts = re.findall(r'BT(.*?)ET', text, re.DOTALL)
            extracted = []
            for part in parts:
                words = re.findall(r'\((.*?)\)', part)
                extracted.extend(words)
            result = " ".join(extracted)[:4000]
            if len(result) > 50:
                return result
        except:
            pass
        return "⚠️ Не удалось извлечь текст из PDF. Попробуй скопировать текст вручную."

    return f"⚠️ Тип файла {doc.mime_type} не поддерживается. Поддерживаются: TXT, PDF."

# ─── Хендлеры ────────────────────────────────────────────────────────────────
dp = Dispatcher(storage=MemoryStorage())

@dp.message(CommandStart())
async def cmd_start(msg: Message):
    user_id = msg.from_user.id
    username = msg.from_user.username or ""
    referred_by = None
    args = msg.text.split()
    if len(args) > 1 and args[1].startswith("ref_"):
        try:
            ref_id = int(args[1].split("_")[1])
            if ref_id != user_id:
                referred_by = ref_id
        except:
            pass

    is_new = upsert_user(user_id, username, referred_by)

    if is_new and referred_by:
        referrer = get_user(referred_by)
        if referrer:
            until = add_referral_bonus(referred_by)
            until_str = datetime.fromisoformat(until).strftime("%d.%m.%Y")
            try:
                await msg.bot.send_message(
                    referred_by,
                    f"🎉 По твоей ссылке пришёл новый пользователь!\n"
                    f"✅ Тебе начислено <b>+{REFERRAL_BONUS_DAYS} дней</b> подписки (до {until_str})",
                    parse_mode="HTML"
                )
            except:
                pass

    await msg.answer(
        f"👋 Привет, <b>{msg.from_user.first_name}</b>!\n\n"
        f"Я — AI-ассистент с выбором модели 🤖\n\n"
        f"🆓 <b>Бесплатно:</b> 3 запроса/день (Llama 8B)\n"
        f"⚡ <b>Базовый:</b> безлимит + файлы (Llama 70B)\n"
        f"🔥 <b>Стандарт:</b> безлимит + файлы + фото (Llama 11B Vision)\n"
        f"👑 <b>Премиум:</b> максимум умности + всё (Llama 90B Vision)\n\n"
        f"Просто напиши мне что-нибудь — я отвечу!",
        parse_mode="HTML",
        reply_markup=kb_main(user_id)
    )

@dp.callback_query(F.data == "plans")
async def cb_plans(cb: CallbackQuery):
    await cb.message.edit_text(
        f"💎 <b>Выбери тариф</b>\n\n"
        f"⚡ <b>Базовый — 150 Stars/мес</b>\n"
        f"Llama 3.3 70B — умная и быстрая\n"
        f"✅ Безлимит ✅ Файлы (TXT, PDF)\n\n"
        f"🔥 <b>Стандарт — 250 Stars/мес</b>\n"
        f"Llama 3.2 11B Vision\n"
        f"✅ Безлимит ✅ Файлы ✅ Фото\n\n"
        f"👑 <b>Премиум — 450 Stars/мес</b>\n"
        f"Llama 3.2 90B Vision — топ модель\n"
        f"✅ Безлимит ✅ Файлы ✅ Фото ✅ Умнее всех",
        parse_mode="HTML",
        reply_markup=kb_plans()
    )
    await cb.answer()

@dp.callback_query(F.data.startswith("buy_"))
async def cb_buy(cb: CallbackQuery, bot: Bot):
    plan_key = cb.data.replace("buy_", "")
    plan = PLANS.get(plan_key)
    if not plan:
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
        parse_mode="HTML",
        reply_markup=kb_main(msg.from_user.id)
    )

@dp.callback_query(F.data == "status")
async def cb_status(cb: CallbackQuery):
    user_id = cb.from_user.id
    user = get_user(user_id)
    if not user:
        upsert_user(user_id, cb.from_user.username or "")
        user = get_user(user_id)

    today = str(datetime.now().date())
    used = user["req_today"] if user["req_date"] == today else 0
    plan = get_user_plan(user)

    if is_admin(user_id):
        status = "👑 Администратор\n♾️ Безлимит, все модели"
    elif is_subscribed(user):
        until = datetime.fromisoformat(user["sub_until"]).strftime("%d.%m.%Y")
        status = f"{plan['name']}\nДо: {until}\nМодель: {plan['model']}"
    else:
        status = f"🆓 Бесплатный план\nИспользовано: {used}/{FREE_REQUESTS_PER_DAY}"

    await cb.message.edit_text(
        f"📊 <b>Твой статус</b>\n\n{status}\n\n"
        f"👥 Приглашено друзей: <b>{user['referral_count']}</b>",
        parse_mode="HTML",
        reply_markup=kb_main(user_id)
    )
    await cb.answer()

@dp.callback_query(F.data == "referral")
async def cb_referral(cb: CallbackQuery):
    user_id = cb.from_user.id
    user = get_user(user_id)
    ref_link = f"https://t.me/{BOT_USERNAME}?start=ref_{user_id}"
    await cb.message.edit_text(
        f"👥 <b>Реферальная программа</b>\n\n"
        f"За каждого друга: <b>+{REFERRAL_BONUS_DAYS} дней</b>\n"
        f"👤 Приглашено: <b>{user['referral_count']} чел.</b>\n\n"
        f"Твоя ссылка:\n<code>{ref_link}</code>",
        parse_mode="HTML",
        reply_markup=kb_back()
    )
    await cb.answer()

@dp.callback_query(F.data == "back_main")
async def cb_back(cb: CallbackQuery):
    await cb.message.edit_text("Главное меню 👇", reply_markup=kb_main(cb.from_user.id))
    await cb.answer()

@dp.callback_query(F.data == "admin_panel")
async def cb_admin_panel(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        await cb.answer("⛔ Нет доступа", show_alert=True)
        return
    stats = get_stats()
    await cb.message.edit_text(
        f"👑 <b>Админ-панель</b>\n\n"
        f"👥 Всего пользователей: <b>{stats['total']}</b>\n"
        f"💎 Активных подписок: <b>{stats['subs']}</b>\n\n"
        f"⚡ Базовых: <b>{stats['by_plan']['basic']}</b>\n"
        f"🔥 Стандарт: <b>{stats['by_plan']['standard']}</b>\n"
        f"👑 Премиум: <b>{stats['by_plan']['premium']}</b>\n\n"
        f"Выдать подписку:\n<code>/give USER_ID план дней</code>\n"
        f"Планы: basic, standard, premium\n"
        f"Пример: <code>/give 123456789 premium 30</code>",
        parse_mode="HTML",
        reply_markup=kb_admin()
    )
    await cb.answer()

@dp.callback_query(F.data == "admin_stats")
async def cb_admin_stats(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        await cb.answer("⛔ Нет доступа", show_alert=True)
        return
    stats = get_stats()
    await cb.answer(
        f"👥 Пользователей: {stats['total']}\n💎 Подписок: {stats['subs']}",
        show_alert=True
    )

@dp.message(Command("give"))
async def cmd_give(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    args = msg.text.split()
    if len(args) != 4:
        await msg.answer("Использование: /give [user_id] [план] [дней]\nПример: /give 123456789 premium 30")
        return
    try:
        target_id = int(args[1])
        plan_key = args[2]
        days = int(args[3])
    except ValueError:
        await msg.answer("❌ Неверный формат.")
        return
    if plan_key not in PLANS or plan_key == "free":
        await msg.answer("❌ Неверный план. Доступны: basic, standard, premium")
        return
    user = get_user(target_id)
    if not user:
        await msg.answer(f"❌ Пользователь {target_id} не найден.")
        return
    until = activate_plan(target_id, plan_key, days)
    until_str = datetime.fromisoformat(until).strftime("%d.%m.%Y")
    plan = PLANS[plan_key]
    await msg.answer(f"✅ Пользователю {target_id} выдан тариф {plan['name']} до {until_str}")
    try:
        await msg.bot.send_message(target_id, f"🎁 Тебе выдан тариф {plan['name']} до {until_str}!")
    except:
        pass

# ─── Основной хендлер сообщений ──────────────────────────────────────────────
@dp.message(F.photo)
async def handle_photo(msg: Message, bot: Bot):
    user_id = msg.from_user.id
    upsert_user(user_id, msg.from_user.username or "")
    user = get_user(user_id)
    plan = get_user_plan(user) if not is_admin(user_id) else PLANS["premium"]

    if not is_admin(user_id) and not is_subscribed(user):
        await msg.answer(
            "📸 Анализ фото доступен только на платных тарифах.\n\n"
            "Оформи подписку 👇",
            reply_markup=kb_plans()
        )
        return

    if not plan["vision"] and not is_admin(user_id):
        await msg.answer(
            f"📸 Анализ фото недоступен на тарифе {plan['name']}.\n"
            f"Нужен тариф 🔥 Стандарт или 👑 Премиум.",
            reply_markup=kb_plans()
        )
        return

    await msg.bot.send_chat_action(msg.chat.id, "typing")
    photo = msg.photo[-1]
    file = await bot.get_file(photo.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, buf)
    image_b64 = base64.b64encode(buf.getvalue()).decode()
    caption = msg.caption or "Опиши что на изображении"
    model = PLANS["premium"]["model"] if is_admin(user_id) else plan["model"]
    response = await ask_ai_vision(caption, image_b64, model)
    await msg.answer(response)

@dp.message(F.document)
async def handle_document(msg: Message, bot: Bot):
    user_id = msg.from_user.id
    upsert_user(user_id, msg.from_user.username or "")
    user = get_user(user_id)
    plan = get_user_plan(user) if not is_admin(user_id) else PLANS["premium"]

    if not is_admin(user_id) and not is_subscribed(user):
        await msg.answer(
            "📄 Работа с файлами доступна только на платных тарифах.\n\n"
            "Оформи подписку 👇",
            reply_markup=kb_plans()
        )
        return

    if not plan["files"] and not is_admin(user_id):
        await msg.answer(
            f"📄 Файлы недоступны на тарифе {plan['name']}.",
            reply_markup=kb_plans()
        )
        return

    await msg.bot.send_chat_action(msg.chat.id, "typing")
    text = await extract_text_from_file(msg, bot)

    if text.startswith("⚠️"):
        await msg.answer(text)
        return

    caption = msg.caption or "Проанализируй этот текст"
    prompt = f"{caption}\n\n---\n{text}"
    model = PLANS["premium"]["model"] if is_admin(user_id) else plan["model"]
    response = await ask_ai(prompt, model)
    await msg.answer(f"📄 <b>Файл обработан:</b>\n\n{response}", parse_mode="HTML")

@dp.message(F.text & ~F.text.startswith("/"))
async def handle_message(msg: Message):
    user_id = msg.from_user.id
    upsert_user(user_id, msg.from_user.username or "")
    user = get_user(user_id)

    if is_admin(user_id):
        await msg.bot.send_chat_action(msg.chat.id, "typing")
        response = await ask_ai(msg.text, PLANS["premium"]["model"])
        await msg.answer(response)
        return

    plan = get_user_plan(user)

    if not is_subscribed(user):
        if not check_and_inc_free(user_id):
            await msg.answer(
                f"⛔ Бесплатный лимит исчерпан ({FREE_REQUESTS_PER_DAY} запроса/день).\n\n"
                f"Выбери тариф для продолжения 👇",
                reply_markup=kb_plans()
            )
            return

    await msg.bot.send_chat_action(msg.chat.id, "typing")
    response = await ask_ai(msg.text, plan["model"])
    await msg.answer(response)

# ─── Запуск ──────────────────────────────────────────────────────────────────
async def main():
    init_db()
    bot = Bot(token=BOT_TOKEN)
    log.info("Бот запущен ✅")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
