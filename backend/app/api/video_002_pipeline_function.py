from pydantic import BaseModel, Field
from fastapi import APIRouter

from app.core.config import settings
from app.services.pipeline_service import analyze_sentiment, classify_zero_shot

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


