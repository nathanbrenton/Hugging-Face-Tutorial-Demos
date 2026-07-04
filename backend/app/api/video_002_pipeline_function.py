from pydantic import BaseModel, Field
from fastapi import APIRouter

from app.core.config import settings
from app.services.pipeline_service import (
    analyze_sentiment,
    classify_zero_shot,
    extract_named_entities,
    fill_mask,
    generate_text,
)


router = APIRouter(
    prefix="/api/video-002/pipeline-function",
    tags=["Video 002 - The Pipeline function"],
)


class SentimentRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        description="One or more text inputs to analyze.",
    )

class SentimentResult(BaseModel):
    text: str
    label: str
    score: float

class SentimentResponse(BaseModel):
    video: str
    concept: str
    task: str
    input_count: int
    results: list[SentimentResult]

class ZeroShotRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Text to classify against candidate labels.",
    )
    candidate_labels: list[str] = Field(
        ...,
        min_length=2,
        description="Two or more labels to compare against the text.",
    )

class ZeroShotResponse(BaseModel):
    video: str
    concept: str
    task: str
    model: str
    text: str
    candidate_labels: list[str]
    labels: list[str]
    scores: list[float]

class TextGenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        description="Prompt used to start text generation.",
    )
    max_new_tokens: int = Field(
        50,
        ge=1,
        le=100,
        description="Maximum number of new tokens to generate.",
    )

class TextGenerationItem(BaseModel):
    generated_text: str

class TextGenerationResponse(BaseModel):
    video: str
    concept: str
    task: str
    model: str
    prompt: str
    max_new_tokens: int
    results: list[TextGenerationItem]

class FillMaskRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Text containing exactly one mask token.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=10,
        description="Number of mask predictions to return.",
    )

class FillMaskItem(BaseModel):
    sequence: str
    score: float
    token: int
    token_str: str

class FillMaskResponse(BaseModel):
    video: str
    concept: str
    task: str
    model: str
    text: str
    top_k: int
    results: list[FillMaskItem]


class NerRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Text from which named entities should be extracted.",
    )


class NerItem(BaseModel):
    entity_group: str
    score: float
    word: str
    start: int
    end: int


class NerResponse(BaseModel):
    video: str
    concept: str
    task: str
    model: str
    text: str
    results: list[NerItem]




@router.post("/sentiment", response_model=SentimentResponse)
def run_sentiment_analysis(payload: SentimentRequest) -> SentimentResponse:
    cleaned_texts = [text.strip() for text in payload.texts if text.strip()]

    results = analyze_sentiment(cleaned_texts)

    return SentimentResponse(
        video="02",
        concept="pipeline",
        task="sentiment-analysis",
        input_count=len(cleaned_texts),
        results=results,
    )

@router.post("/zero-shot-classification", response_model=ZeroShotResponse)
def run_zero_shot_classification(payload: ZeroShotRequest) -> ZeroShotResponse:
    cleaned_text = payload.text.strip()
    cleaned_labels = [
        label.strip()
        for label in payload.candidate_labels
        if label.strip()
    ]

    result = classify_zero_shot(cleaned_text, cleaned_labels)

    return ZeroShotResponse(
        video="002",
        concept="pipeline",
        task="zero-shot-classification",
        model=settings.zero_shot_model_id,
        text=result["text"],
        candidate_labels=cleaned_labels,
        labels=result["labels"],
        scores=result["scores"],
    )


@router.post("/text-generation", response_model=TextGenerationResponse)
def run_text_generation(payload: TextGenerationRequest) -> TextGenerationResponse:
    cleaned_prompt = payload.prompt.strip()

    results = generate_text(
        prompt=cleaned_prompt,
        max_new_tokens=payload.max_new_tokens,
    )

    return TextGenerationResponse(
        video="002",
        concept="pipeline",
        task="text-generation",
        model=settings.text_generation_model_id,
        prompt=cleaned_prompt,
        max_new_tokens=payload.max_new_tokens,
        results=results,
    )


@router.post("/fill-mask", response_model=FillMaskResponse)
def run_fill_mask(payload: FillMaskRequest) -> FillMaskResponse:
    cleaned_text = payload.text.strip()

    results = fill_mask(
        text=cleaned_text,
        top_k=payload.top_k,
    )

    return FillMaskResponse(
        video="002",
        concept="pipeline",
        task="fill-mask",
        model=settings.fill_mask_model_id,
        text=cleaned_text,
        top_k=payload.top_k,
        results=results,
    )


@router.post("/ner", response_model=NerResponse)
def run_named_entity_recognition(payload: NerRequest) -> NerResponse:
    cleaned_text = payload.text.strip()

    results = extract_named_entities(cleaned_text)

    return NerResponse(
        video="002",
        concept="pipeline",
        task="ner",
        model=settings.ner_model_id,
        text=cleaned_text,
        results=results,
    )



