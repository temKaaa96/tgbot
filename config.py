import os

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
BOT_USERNAME = os.environ.get("BOT_USERNAME", "")
FREE_REQUESTS_PER_DAY = int(os.environ.get("FREE_REQUESTS_PER_DAY", 3))
SUBSCRIPTION_PRICE_STARS = int(os.environ.get("SUBSCRIPTION_PRICE_STARS", 250))
SUBSCRIPTION_DAYS = int(os.environ.get("SUBSCRIPTION_DAYS", 30))
REFERRAL_BONUS_DAYS = int(os.environ.get("REFERRAL_BONUS_DAYS", 7))
ADMIN_ID = 8926744054
