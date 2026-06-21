"""
AI Telegram Bot с подпиской и реферальной системой
Стек: Python 3.10+, aiogram 3, Groq API, SQLite
"""

import asyncio
import logging
import sqlite3
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
    FREE_REQUESTS_PER_DAY, SUBSCRIPTION_PRICE_STARS,
    SUBSCRIPTION_DAYS, REFERRAL_BONUS_DAYS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── База данных ─────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect("users.db")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id        INTEGER PRIMARY KEY,
            username       TEXT,
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
        "SELECT user_id, username, sub_until, req_today, req_date, referred_by, referral_count FROM users WHERE user_id=?",
        (user_id,)
    ).fetchone()
    con.close()
    if not row:
        return None
    return dict(zip(["user_id", "username", "sub_until", "req_today", "req_date", "referred_by", "referral_count"], row))

def upsert_user(user_id: int, username: str, referred_by: int = None):
    con = sqlite3.connect("users.db")
    is_new = con.execute("SELECT 1 FROM users WHERE user_id=?", (user_id,)).fetchone() is None
    con.execute(
        "INSERT OR IGNORE INTO users (user_id, username, req_today, req_date, referred_by) VALUES (?,?,0,?,?)",
        (user_id, username, str(datetime.now().date()), referred_by)
    )
    con.execute("UPDATE users SET username=? WHERE user_id=?", (username, user_id))
    con.commit()
    con.close()
    return is_new

def is_subscribed(user: dict) -> bool:
    if not user or not user["sub_until"]:
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

def activate_subscription(user_id: int, days: int) -> str:
    con = sqlite3.connect("users.db")
    user = con.execute("SELECT sub_until FROM users WHERE user_id=?", (user_id,)).fetchone()
    if user and user[0] and datetime.fromisoformat(user[0]) > datetime.now():
        until = (datetime.fromisoformat(user[0]) + timedelta(days=days)).isoformat()
    else:
        until = (datetime.now() + timedelta(days=days)).isoformat()
    con.execute("UPDATE users SET sub_until=? WHERE user_id=?", (until, user_id))
    con.commit()
    con.close()
    return until

def add_referral_bonus(referrer_id: int) -> str:
    con = sqlite3.connect("users.db")
    con.execute("UPDATE users SET referral_count = referral_count + 1 WHERE user_id=?", (referrer_id,))
    con.commit()
    con.close()
    return activate_subscription(referrer_id, REFERRAL_BONUS_DAYS)

def get_stats() -> dict:
    con = sqlite3.connect("users.db")
    total = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    subs = con.execute("SELECT COUNT(*) FROM users WHERE sub_until > ?", (datetime.now().isoformat(),)).fetchone()[0]
    con.close()
    return {"total": total, "subs": subs}

# ─── Проверка админа ─────────────────────────────────────────────────────────
def is_admin(user_id: int) -> bool:
    return user_id == ADMIN_ID

# ─── Клавиатуры ──────────────────────────────────────────────────────────────
def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💎 Оформить подписку", callback_data="subscribe")],
        [InlineKeyboardButton(text="👥 Реферальная программа", callback_data="referral")],
        [InlineKeyboardButton(text="📊 Мой статус", callback_data="status")],
    ])

def kb_main_admin() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="👑 Админ-панель", callback_data="admin_panel")],
        [InlineKeyboardButton(text="👥 Реферальная программа", callback_data="referral")],
        [InlineKeyboardButton(text="📊 Мой статус", callback_data="status")],
    ])

def kb_subscribe() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text=f"⭐ Оплатить {SUBSCRIPTION_PRICE_STARS} Stars ({SUBSCRIPTION_DAYS} дней)",
            callback_data="pay_stars"
        )],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="back_main")],
    ])

def kb_back() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Назад", callback_data="back_main")]
    ])

def kb_admin() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Статистика", callback_data="admin_stats")],
        [InlineKeyboardButton(text="🎁 Выдать подписку", callback_data="admin_give_prompt")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="back_main")],
    ])

# ─── AI запрос ───────────────────────────────────────────────────────────────
async def ask_ai(text: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": text}]
                }
            )
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"Groq error: {e}")
        return "⚠️ Ошибка при обращении к AI. Попробуй чуть позже."

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

    kb = kb_main_admin() if is_admin(user_id) else kb_main()
    await msg.answer(
        f"👋 Привет, <b>{msg.from_user.first_name}</b>!\n\n"
        f"Я — AI-ассистент, отвечу на любой вопрос 🤖\n\n"
        f"🆓 Бесплатно: <b>{FREE_REQUESTS_PER_DAY} запроса в день</b>\n"
        f"💎 Подписка: безлимит на {SUBSCRIPTION_DAYS} дней\n"
        f"👥 Приглашай друзей — получай дни подписки!\n\n"
        f"Просто напиши мне что-нибудь — я отвечу!",
        parse_mode="HTML",
        reply_markup=kb
    )

@dp.callback_query(F.data == "admin_panel")
async def cb_admin_panel(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        await cb.answer("⛔ Нет доступа", show_alert=True)
        return
    stats = get_stats()
    await cb.message.edit_text(
        f"👑 <b>Админ-панель</b>\n\n"
        f"👥 Всего пользователей: <b>{stats['total']}</b>\n"
        f"💎 Активных подписок: <b>{stats['subs']}</b>\n"
        f"🆓 Без подписки: <b>{stats['total'] - stats['subs']}</b>\n\n"
        f"Чтобы выдать подписку используй:\n"
        f"<code>/give USER_ID дней</code>\n"
        f"Пример: <code>/give 123456789 30</code>",
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

@dp.callback_query(F.data == "admin_give_prompt")
async def cb_admin_give_prompt(cb: CallbackQuery):
    if not is_admin(cb.from_user.id):
        await cb.answer("⛔ Нет доступа", show_alert=True)
        return
    await cb.message.edit_text(
        "🎁 <b>Выдать подписку</b>\n\n"
        "Отправь команду в формате:\n"
        "<code>/give USER_ID количество_дней</code>\n\n"
        "Пример:\n"
        "<code>/give 123456789 30</code>",
        parse_mode="HTML",
        reply_markup=kb_admin()
    )
    await cb.answer()

@dp.message(Command("give"))
async def cmd_give(msg: Message):
    if not is_admin(msg.from_user.id):
        return
    args = msg.text.split()
    if len(args) != 3:
        await msg.answer("Использование: /give [user_id] [дней]\nПример: /give 123456789 30")
        return
    try:
        target_id = int(args[1])
        days = int(args[2])
    except ValueError:
        await msg.answer("❌ Неверный формат. Пример: /give 123456789 30")
        return
    user = get_user(target_id)
    if not user:
        await msg.answer(f"❌ Пользователь {target_id} не найден в базе.")
        return
    until = activate_subscription(target_id, days)
    until_str = datetime.fromisoformat(until).strftime("%d.%m.%Y")
    await msg.answer(f"✅ Пользователю {target_id} выдана подписка до {until_str}")
    try:
        await msg.bot.send_message(target_id, f"🎁 Тебе выдана подписка до {until_str}!")
    except:
        pass

@dp.callback_query(F.data == "status")
async def cb_status(cb: CallbackQuery):
    user_id = cb.from_user.id
    user = get_user(user_id)
    if not user:
        upsert_user(user_id, cb.from_user.username or "")
        user = get_user(user_id)
    today = str(datetime.now().date())
    used = user["req_today"] if user["req_date"] == today else 0
    if is_admin(user_id):
        sub_text = "👑 Ты администратор\n♾️ Безлимитные запросы"
    elif is_subscribed(user):
        until = datetime.fromisoformat(user["sub_until"]).strftime("%d.%m.%Y")
        sub_text = f"✅ <b>Подписка активна</b> до {until}\n🔓 Безлимитные запросы"
    else:
        sub_text = f"🆓 Бесплатный план\n📊 Использовано сегодня: <b>{used}/{FREE_REQUESTS_PER_DAY}</b>"
    text = f"{sub_text}\n\n👥 Приглашено друзей: <b>{user['referral_count']}</b>"
    kb = kb_main_admin() if is_admin(user_id) else kb_main()
    await cb.message.edit_text(text, parse_mode="HTML", reply_markup=kb)
    await cb.answer()

@dp.callback_query(F.data == "referral")
async def cb_referral(cb: CallbackQuery):
    user_id = cb.from_user.id
    user = get_user(user_id)
    ref_link = f"https://t.me/{BOT_USERNAME}?start=ref_{user_id}"
    text = (
        f"👥 <b>Реферальная программа</b>\n\n"
        f"За каждого друга: <b>+{REFERRAL_BONUS_DAYS} дней</b> подписки\n"
        f"👤 Ты пригласил: <b>{user['referral_count']} чел.</b>\n\n"
        f"Твоя ссылка:\n<code>{ref_link}</code>"
    )
    await cb.message.edit_text(text, parse_mode="HTML", reply_markup=kb_back())
    await cb.answer()

@dp.callback_query(F.data == "subscribe")
async def cb_subscribe(cb: CallbackQuery):
    await cb.message.edit_text(
        f"💎 <b>Подписка на {SUBSCRIPTION_DAYS} дней</b>\n\n"
        f"✅ Безлимитные запросы к AI\n"
        f"✅ Приоритетная скорость\n\n"
        f"Оплата через Telegram Stars.",
        parse_mode="HTML",
        reply_markup=kb_subscribe()
    )
    await cb.answer()

@dp.callback_query(F.data == "back_main")
async def cb_back(cb: CallbackQuery):
    kb = kb_main_admin() if is_admin(cb.from_user.id) else kb_main()
    await cb.message.edit_text("Главное меню 👇", reply_markup=kb)
    await cb.answer()

@dp.callback_query(F.data == "pay_stars")
async def cb_pay_stars(cb: CallbackQuery, bot: Bot):
    await bot.send_invoice(
        chat_id=cb.from_user.id,
        title="💎 AI Подписка",
        description=f"Безлимитный доступ к AI на {SUBSCRIPTION_DAYS} дней",
        payload="sub_stars",
        currency="XTR",
        prices=[LabeledPrice(label="Подписка", amount=SUBSCRIPTION_PRICE_STARS)],
    )
    await cb.answer()

@dp.pre_checkout_query()
async def pre_checkout(query: PreCheckoutQuery, bot: Bot):
    await bot.answer_pre_checkout_query(query.id, ok=True)

@dp.message(F.successful_payment)
async def payment_success(msg: Message):
    until = activate_subscription(msg.from_user.id, SUBSCRIPTION_DAYS)
    until_str = datetime.fromisoformat(until).strftime("%d.%m.%Y")
    await msg.answer(
        f"🎉 <b>Подписка активирована!</b>\nДействует до {until_str}.",
        parse_mode="HTML"
    )

@dp.message(F.text & ~F.text.startswith("/"))
async def handle_message(msg: Message):
    user_id = msg.from_user.id
    upsert_user(user_id, msg.from_user.username or "")
    user = get_user(user_id)

    if is_admin(user_id):
        await msg.bot.send_chat_action(msg.chat.id, "typing")
        response = await ask_ai(msg.text)
        await msg.answer(response)
        return

    if not is_subscribed(user):
        if not check_and_inc_free(user_id):
            await msg.answer(
                f"⛔ Бесплатный лимит исчерпан ({FREE_REQUESTS_PER_DAY} запроса/день).\n\n"
                f"💡 Оформи подписку или пригласи друга 👇",
                reply_markup=kb_main()
            )
            return

    await msg.bot.send_chat_action(msg.chat.id, "typing")
    response = await ask_ai(msg.text)
    await msg.answer(response)

# ─── Запуск ──────────────────────────────────────────────────────────────────
async def main():
    init_db()
    bot = Bot(token=BOT_TOKEN)
    log.info("Бот запущен ✅")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
