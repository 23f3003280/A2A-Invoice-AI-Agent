import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./invoice_agent.db")
    ai_api_key: str = os.getenv("AI_API_KEY", "")
    ai_base_url: str = os.getenv("AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    ai_model: str = os.getenv("AI_MODEL", "gpt-4o-mini")
    bearer_token: str = os.getenv("A2A_BEARER_TOKEN", "local-development-token")
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")
    ai_timeout_seconds: float = float(os.getenv("AI_TIMEOUT_SECONDS", "35"))

settings = Settings()
