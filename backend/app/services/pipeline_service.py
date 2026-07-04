from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

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

@lru_cache(maxsize=1)
def get_question_answering_components() -> tuple[Any, Any]:
    """
    Load the question-answering tokenizer and model once
    and reuse them across requests.
    """
    tokenizer = AutoTokenizer.from_pretrained(settings.question_answering_model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(settings.question_answering_model_id)
    model.eval()

    return tokenizer, model




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


def answer_question(question: str, context: str) -> dict[str, Any]:
    """
    Answer a question using the supplied context.

    This uses the model/tokenizer directly because this installed
    Transformers version does not register pipeline("question-answering").
    """
    tokenizer, model = get_question_answering_components()

    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        return_offsets_mapping=True,
    )

    sequence_ids = encoded.sequence_ids(0)
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**encoded)

    start_logits = outputs.start_logits[0].clone()
    end_logits = outputs.end_logits[0].clone()

    context_token_indexes = [
        index
        for index, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1
    ]

    for index, sequence_id in enumerate(sequence_ids):
        if sequence_id != 1:
            start_logits[index] = -float("inf")
            end_logits[index] = -float("inf")

    best_start = context_token_indexes[0]
    best_end = context_token_indexes[0]
    best_score = -float("inf")
    max_answer_tokens = 30

    for start_index in context_token_indexes:
        for end_index in context_token_indexes:
            if end_index < start_index:
                continue

            if end_index - start_index + 1 > max_answer_tokens:
                continue

            score = float(start_logits[start_index] + end_logits[end_index])

            if score > best_score:
                best_score = score
                best_start = start_index
                best_end = end_index

    start_probabilities = torch.softmax(start_logits, dim=0)
    end_probabilities = torch.softmax(end_logits, dim=0)
    answer_score = float(start_probabilities[best_start] * end_probabilities[best_end])

    start_char = int(offset_mapping[best_start][0])
    end_char = int(offset_mapping[best_end][1])
    answer = context[start_char:end_char].strip()

    if not answer:
        answer = tokenizer.decode(
            encoded["input_ids"][0][best_start : best_end + 1],
            skip_special_tokens=True,
        ).strip()

    return {
        "answer": answer,
        "score": answer_score,
        "start": start_char,
        "end": end_char,
    }




