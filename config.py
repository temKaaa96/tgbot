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

# ─── DeepSeek (премиум-тариф) ────────────────────────────────────────────────
# Несколько глобальных ключей через запятую: DEEPSEEK_KEYS="sk-aaa,sk-bbb"
# Они крутятся по кругу (ротация), чтобы не упираться в лимит одного ключа.
DEEPSEEK_KEYS = [k.strip() for k in os.environ.get("DEEPSEEK_KEYS", "").split(",") if k.strip()]

# Модель премиума по умолчанию (можно переключить на быструю в настройках).
DEEPSEEK_PREMIUM_MODEL = os.environ.get("DEEPSEEK_PREMIUM_MODEL", "deepseek-v4-pro")
DEEPSEEK_FAST_MODEL = os.environ.get("DEEPSEEK_FAST_MODEL", "deepseek-v4-flash")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Путь к базе. На Railway смонтируй volume и укажи сюда путь внутри него,
# например DB_PATH="/data/users.db" — иначе база стирается при каждом редеплое.
DB_PATH = os.environ.get("DB_PATH", "users.db")
