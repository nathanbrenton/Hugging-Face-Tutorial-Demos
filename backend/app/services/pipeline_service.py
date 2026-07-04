from functools import lru_cache
from typing import Any

from transformers import pipeline

from app.core.config import settings


@lru_cache(maxsize=1)
def get_sentiment_pipeline() -> Any:
    """
    Load the Hugging Face sentiment-analysis pipeline once
    and reuse it across requests.
    """
    return pipeline(
        task="sentiment-analysis",
        model=settings.sentiment_model_id,
    )

@lru_cache(maxsize=1)
def get_zero_shot_pipeline() -> Any:
    """
    Load the Hugging Face zero-shot-classification pipeline once
    and reuse it across requests.
    """
    return pipeline(
        task="zero-shot-classification",
        model=settings.zero_shot_model_id,
    )


def analyze_sentiment(texts: list[str]) -> list[dict[str, Any]]:
    """
    Run sentiment analysis on one or more texts.

    The pipeline handles:
    - preprocessing
    - tokenization
    - model inference
    - postprocessing
    """
    classifier = get_sentiment_pipeline()
    results = classifier(texts)

    return [
        {
            "text": text,
            "label": result["label"],
            "score": float(result["score"]),
        }
        for text, result in zip(texts, results)
    ]

def classify_zero_shot(text: str, candidate_labels: list[str]) -> dict[str, Any]:
    """
    Run zero-shot classification on one text using candidate labels.
    """
    classifier = get_zero_shot_pipeline()
    result = classifier(text, candidate_labels)

    return {
        "text": result["sequence"],
        "labels": result["labels"],
        "scores": [float(score) for score in result["scores"]],
    }
