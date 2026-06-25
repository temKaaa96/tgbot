import os

# ─── Telegram / Groq ─────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
BOT_USERNAME = os.environ.get("BOT_USERNAME", "")

# ─── Лимиты и тарифы ─────────────────────────────────────────────────────────
FREE_REQUESTS_PER_DAY = int(os.environ.get("FREE_REQUESTS_PER_DAY", 3))
SUBSCRIPTION_PRICE_STARS = int(os.environ.get("SUBSCRIPTION_PRICE_STARS", 250))
SUBSCRIPTION_DAYS = int(os.environ.get("SUBSCRIPTION_DAYS", 30))
REFERRAL_BONUS_DAYS = int(os.environ.get("REFERRAL_BONUS_DAYS", 7))
ADMIN_ID = int(os.environ.get("ADMIN_ID", "8926744054"))

# ─── Gemini (премиум-тариф) ──────────────────────────────────────────────────
# Один бесплатный ключ Google AI Studio (начинается с AIza). Получить без карты:
# aistudio.google.com → Get API key. Бесплатный тариф: ~1500 запросов/день, 15/мин.
# Один аккаунт — это легально, никаких банов (в отличие от фарма триал-ключей).
def _clean_key(k: str) -> str:
    k = k.strip().strip('"').strip("'").strip()
    if k.lower().startswith("bearer "):
        k = k[7:].strip()
    return k

GEMINI_API_KEY = _clean_key(os.environ.get("GEMINI_API_KEY", ""))
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

# Премиум-модели (обе бесплатные на free-тарифе):
#   PREMIUM_MODEL  — умнее (gemini-2.5-flash)
#   FAST_MODEL     — быстрее/легче (gemini-2.5-flash-lite)
GEMINI_PREMIUM_MODEL = os.environ.get("GEMINI_PREMIUM_MODEL", "gemini-2.5-flash")
GEMINI_FAST_MODEL = os.environ.get("GEMINI_FAST_MODEL", "gemini-2.5-flash-lite")

# Путь к базе. На Railway смонтируй volume и укажи путь внутри него,
# например DB_PATH="/data/users.db" — иначе база стирается при каждом редеплое.
DB_PATH = os.environ.get("DB_PATH", "users.db")
