import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BACKEND_DIR / ".env"

load_dotenv(ENV_PATH)


@dataclass(frozen=True)
class Settings:
    app_name: str = "Hugging Face Tutorial Demos"
    app_version: str = "0.1.0"
    hf_home: str = os.getenv("HF_HOME", "./hf_home")
    sentiment_model_id: str = os.getenv(
        "SENTIMENT_MODEL_ID",
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )


settings = Settings()

os.environ.setdefault("HF_HOME", str((BACKEND_DIR / settings.hf_home).resolve()))
os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "0"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "0"))

