from pydantic import BaseModel, Field
from fastapi import APIRouter

from app.services.pipeline_service import analyze_sentiment


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
