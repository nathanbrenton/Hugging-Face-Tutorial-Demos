# Video 002 - The pipeline Function

## Demo

This demo uses Hugging Face's `pipeline()` function to perform sentiment analysis on one or more text inputs.

The user enters text in the browser. The frontend sends the text to the local FastAPI backend. The backend runs a Hugging Face sentiment-analysis pipeline and returns structured JSON results.

## Concept

The `pipeline()` function provides a high-level API for common machine learning tasks.

For this demo, the pipeline handles:

- preprocessing
- tokenization
- model inference
- postprocessing
- returning structured prediction objects

## Endpoint

POST /api/video-002/pipeline-function/sentiment

## Example Request

{
  "texts": [
    "I love learning Hugging Face.",
    "The setup was confusing."
  ]
}

## Example Response

{
  "video": "02",
  "concept": "pipeline",
  "task": "sentiment-analysis",
  "input_count": 2,
  "results": [
    {
      "text": "I love learning Hugging Face.",
      "label": "POSITIVE",
      "score": 0.9998
    },
    {
      "text": "The setup was confusing.",
      "label": "NEGATIVE",
      "score": 0.9989
    }
  ]
}
