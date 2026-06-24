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

# ─── OpenModel (премиум-тариф) ───────────────────────────────────────────────
# Твои бесплатные ключи OpenModel (начинаются с om-), через запятую:
#   OPENMODEL_KEYS="om-aaa,om-bbb"
# Они крутятся по кругу (ротация), т.к. у бесплатного DeepSeek V4 Flash лимит
# 10 запросов/мин на ключ — несколько ключей расширяют этот потолок.
def _clean_key(k: str) -> str:
    k = k.strip().strip('"').strip("'").strip()
    if k.lower().startswith("bearer "):
        k = k[7:].strip()
    return k

OPENMODEL_KEYS = [_clean_key(k) for k in os.environ.get("OPENMODEL_KEYS", "").split(",") if _clean_key(k)]

# Базовый URL OpenModel (Anthropic-совместимый). Эндпоинт — /messages.
OPENMODEL_BASE_URL = os.environ.get("OPENMODEL_BASE_URL", "https://api.openmodel.ai/v1")

# Премиум-модели:
#   PREMIUM_MODEL      — бесплатная во время события (deepseek-v4-flash)
#   PREMIUM_MODEL_PRO  — точнее, но тратит кредиты аккаунта (deepseek-v4-pro)
PREMIUM_MODEL = os.environ.get("PREMIUM_MODEL", "deepseek-v4-flash")
PREMIUM_MODEL_PRO = os.environ.get("PREMIUM_MODEL_PRO", "deepseek-v4-pro")

# Путь к базе. На Railway смонтируй volume и укажи путь внутри него,
# например DB_PATH="/data/users.db" — иначе база стирается при каждом редеплое.
DB_PATH = os.environ.get("DB_PATH", "users.db")
