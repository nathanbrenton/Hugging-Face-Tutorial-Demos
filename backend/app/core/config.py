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
    zero_shot_model_id: str = os.getenv(
        "ZERO_SHOT_MODEL_ID",
        "/home/mlops/Downloads/huggingface-models/repos/facebook/bart-large-mnli",
    )
    text_generation_model_id: str = os.getenv(
        "TEXT_GENERATION_MODEL_ID",
        "/home/mlops/Downloads/huggingface-models/repos/distilbert/distilgpt2",
    )
    fill_mask_model_id: str = os.getenv(
        "FILL_MASK_MODEL_ID",
        "/home/mlops/Downloads/huggingface-models/repos/distilbert/distilbert-base-uncased",
    )
    ner_model_id: str = os.getenv(
        "NER_MODEL_ID",
        "/home/mlops/Downloads/huggingface-models/repos/dslim/bert-base-NER",
    )
    question_answering_model_id: str = os.getenv(
        "QUESTION_ANSWERING_MODEL_ID",
        "/home/mlops/Downloads/huggingface-models/repos/deepset/roberta-base-squad2",
    )


settings = Settings()

os.environ.setdefault("HF_HOME", str((BACKEND_DIR / settings.hf_home).resolve()))
os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "0"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "0"))

