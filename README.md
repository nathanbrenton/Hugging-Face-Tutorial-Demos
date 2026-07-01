
# Hugging Face Tutorial Demos

A collection of small, self-hosted AI/ML demos built while working through the Hugging Face YouTube playlist.

Each demo uses a simple local frontend and backend to demonstrate one Hugging Face concept at a time.

## Project Goals

* Practice Hugging Face concepts through working demos
* Maintain a consistent local frontend/backend architecture
* Build a visible GitHub commit history for AI/ML-related work
* Keep each video demo small, understandable, and portfolio-friendly

## Current Demo

* Video 002 - The pipeline function

## Architecture

Browser frontend
-> FastAPI backend
-> Hugging Face model or utility
-> structured JSON response
-> browser result display

## Project Structure

* backend/
* frontend/
* demos/
* docs/

## Backend Setup

From the project root:

1. Change into the backend directory:

   cd backend

2. Create a virtual environment:

   python3 -m venv .venv

3. Activate the virtual environment:

   source .venv/bin/activate

4. Upgrade pip:

   python -m pip install --upgrade pip

5. Install dependencies:

   python -m pip install -r requirements.txt

6. Start the backend:

   uvicorn app.main:app --reload

Backend URL:

http://127.0.0.1:8000

API docs:

http://127.0.0.1:8000/docs

## Frontend Setup

From the project root, in another terminal:

1. Change into the frontend directory:

   cd frontend

2. Start a simple static file server:

   python3 -m http.server 5500

Frontend URL:

http://127.0.0.1:5500

## 002 Demo

The first demo uses Hugging Face's pipeline() function for sentiment analysis.

Endpoint:

POST /api/video-002/pipeline-function/sentiment

Example request:

{
"texts": [
"I love learning Hugging Face.",
"The setup was confusing."
]
}

Example response:

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
