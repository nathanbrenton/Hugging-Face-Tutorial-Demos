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

@lru_cache(maxsize=1)
def get_text_generation_pipeline() -> Any:
    """
    Load the Hugging Face text-generation pipeline once
    and reuse it across requests.
    """
    return pipeline(
        task="text-generation",
        model=settings.text_generation_model_id,
    )

@lru_cache(maxsize=1)
def get_fill_mask_pipeline() -> Any:
    """
    Load the Hugging Face fill-mask pipeline once
    and reuse it across requests.
    """
    return pipeline(
        task="fill-mask",
        model=settings.fill_mask_model_id,
    )

@lru_cache(maxsize=1)
def get_ner_pipeline() -> Any:
    """
    Load the Hugging Face named entity recognition pipeline once
    and reuse it across requests.
    """
    return pipeline(
        task="ner",
        model=settings.ner_model_id,
        aggregation_strategy="simple",
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

def generate_text(prompt: str, max_new_tokens: int = 50) -> list[dict[str, str]]:
    """
    Generate text from a prompt.
    """
    generator = get_text_generation_pipeline()
    results = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        pad_token_id=generator.tokenizer.eos_token_id,
    )

    return [
        {
            "generated_text": result["generated_text"],
        }
        for result in results
    ]

def fill_mask(text: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Predict likely replacements for the mask token in a sentence.
    """
    mask_pipeline = get_fill_mask_pipeline()
    results = mask_pipeline(text, top_k=top_k)

    return [
        {
            "sequence": result["sequence"],
            "score": float(result["score"]),
            "token": int(result["token"]),
            "token_str": result["token_str"],
        }
        for result in results
    ]

def extract_named_entities(text: str) -> list[dict[str, Any]]:
    """
    Extract named entities from text.
    """
    ner_pipeline = get_ner_pipeline()
    results = ner_pipeline(text)

    return [
        {
            "entity_group": result["entity_group"],
            "score": float(result["score"]),
            "word": result["word"],
            "start": result["start"],
            "end": result["end"],
        }
        for result in results
    ]
