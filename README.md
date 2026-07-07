# Hugging Face Tutorial Demos

A collection of small, self-hosted AI/ML demos built while working through the Hugging Face YouTube playlist.

Each demo uses a simple local frontend and backend to demonstrate one Hugging Face concept at a time. The current implementation emphasizes local model assets, offline-friendly runtime behavior, and clear notes when the local implementation differs from the original course snippet.

## Screenshot Tour

![Local AI/ML Demo Platform header](docs/screenshots/02_00-Header.png)

## Video 002 - The pipeline function

Video 002 demonstrates the Hugging Face `pipeline()` concept across eight text-oriented tasks.

### 1. Sentiment Analysis

Course concept: `pipeline("sentiment-analysis")`

Local model: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`

![Sentiment analysis info panel](docs/screenshots/02_01-SentimentAnalysis_01-Info.png)

![Sentiment analysis example output](docs/screenshots/02_01-SentimentAnalysis_02-Example.png)

<details>
<summary>Terminal validation</summary>

![Sentiment analysis terminal validation](docs/screenshots/02_01-SentimentAnalysis_03-Terminal.png)

</details>

### 2. Zero-Shot Classification

Course concept: `pipeline("zero-shot-classification")`

Local model: `facebook/bart-large-mnli`

![Zero-shot classification info panel](docs/screenshots/02_02-ZeroShotClassification_01-Info.png)

![Zero-shot classification example output](docs/screenshots/02_02-ZeroShotClassification_02-Example.png)

<details>
<summary>Terminal validation</summary>

![Zero-shot classification terminal validation](docs/screenshots/02_02-ZeroShotClassification_03-Terminal.png)

</details>

### 3. Text Generation

Course concept: `pipeline("text-generation", model="distilgpt2")`

Local model: `distilbert/distilgpt2`

![Text generation info panel](docs/screenshots/02_03-TextGeneration_01-Info.png)

![Text generation example output](docs/screenshots/02_03-TextGeneration_02-Example.png)

<details>
<summary>Terminal validation</summary>

![Text generation terminal validation](docs/screenshots/02_03-TextGeneration_03-Terminal.png)

</details>

### 4. Fill-Mask

Course concept: `pipeline("fill-mask")`

Local model: `distilbert/distilbert-base-uncased`

![Fill-mask info panel](docs/screenshots/02_04-FillMask_01-Info.png)

![Fill-mask example output](docs/screenshots/02_04-FillMask_02-Example.png)

<details>
<summary>Terminal validation</summary>

![Fill-mask terminal validation](docs/screenshots/02_04-FillMask_03-Terminal.png)

</details>

### 5. Named Entity Recognition

Course concept: `pipeline("ner", grouped_entities=True)`

Local implementation: `pipeline("ner", aggregation_strategy="simple")`

Local model: `dslim/bert-base-NER`

![Named entity recognition info panel](docs/screenshots/02_05-NamedEntityRecognition_01-Info.png)

![Named entity recognition example output](docs/screenshots/02_05-NamedEntityRecognition_02-Example.png)

<details>
<summary>Terminal validation</summary>

![Named entity recognition terminal validation](docs/screenshots/02_05-NamedEntityRecognition_03-Terminal.png)

</details>

### 6. Question Answering

Course concept: `pipeline("question-answering")`

Local implementation: `AutoTokenizer` + `AutoModelForQuestionAnswering`

Local model: `deepset/roberta-base-squad2`

Implementation note: in this local environment, the installed Transformers pipeline registry does not register `question-answering`, so the backend uses the tokenizer and model directly.

![Question answering info panel](docs/screenshots/02_06-QuestionAnswering_01-Info.png)

![Question answering example output](docs/screenshots/02_06-QuestionAnswering_02-Example.png)

<details>
<summary>Terminal validation</summary>

![Question answering terminal validation](docs/screenshots/02_06-QuestionAnswering_03-Terminal.png)

</details>

### 7. Summarization

Course concept: `pipeline("summarization")`

Local implementation: `AutoTokenizer` + `AutoModelForSeq2SeqLM`

Local model: `sshleifer/distilbart-cnn-12-6`

Implementation note: in this local environment, the installed Transformers pipeline registry does not register `summarization`, so the backend uses the tokenizer and model directly.

![Summarization info panel](docs/screenshots/02_07-Summarization_01-Info.png)

![Summarization example output](docs/screenshots/02_07-Summarization_02-Example.png)

<details>
<summary>Terminal validation</summary>

![Summarization terminal validation](docs/screenshots/02_07-Summarization_03-Terminal.png)

</details>

### 8. Translation

Course concept: `pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")`

Local implementation: `MarianTokenizer` + `MarianMTModel`

Local model: `Helsinki-NLP/opus-mt-fr-en`

Implementation note: in this local environment, the installed Transformers pipeline registry does not register `translation`, so the backend uses the Marian tokenizer and model directly.

![Translation info panel](docs/screenshots/02_08-Translation_01-Info.png)

![Translation example output](docs/screenshots/02_08-Translation_02-Example.png)

<details>
<summary>Terminal validation</summary>

![Translation terminal validation](docs/screenshots/02_08-Translation_03-Terminal.png)

</details>

## Implemented Demos

| Video | Topic | Status |
|---|---|---|
| 002 | The pipeline function | Complete |

Video 002 includes:

| Demo | Course example | Local backend implementation |
|---|---|---|
| Sentiment Analysis | `pipeline("sentiment-analysis")` | `pipeline(task="sentiment-analysis")` |
| Zero-Shot Classification | `pipeline("zero-shot-classification")` | `pipeline(task="zero-shot-classification")` |
| Text Generation | `pipeline("text-generation", model="distilgpt2")` | `pipeline(task="text-generation")` |
| Fill-Mask | `pipeline("fill-mask")` | `pipeline(task="fill-mask")` |
| Named Entity Recognition | `pipeline("ner", grouped_entities=True)` | `pipeline(task="ner", aggregation_strategy="simple")` |
| Question Answering | `pipeline("question-answering")` | `AutoTokenizer` + `AutoModelForQuestionAnswering` |
| Summarization | `pipeline("summarization")` | `AutoTokenizer` + `AutoModelForSeq2SeqLM` |
| Translation | `pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")` | `MarianTokenizer` + `MarianMTModel` |

## Project Goals

* Practice Hugging Face concepts through working demos
* Maintain a consistent local frontend/backend architecture
* Build a visible GitHub commit history for AI/ML-related work
* Keep each video demo small, understandable, and portfolio-friendly
* Prefer local model assets and offline-friendly runtime behavior

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
* docs/screenshots/

## Local Model Assets

This project expects reusable Hugging Face model assets to live outside the Git repository.

Current model path convention:

`~/Downloads/huggingface-models/repos/`

Video 002 uses these local model directories:

* `~/Downloads/huggingface-models/repos/distilbert/distilbert-base-uncased-finetuned-sst-2-english`
* `~/Downloads/huggingface-models/repos/facebook/bart-large-mnli`
* `~/Downloads/huggingface-models/repos/distilbert/distilgpt2`
* `~/Downloads/huggingface-models/repos/distilbert/distilbert-base-uncased`
* `~/Downloads/huggingface-models/repos/dslim/bert-base-NER`
* `~/Downloads/huggingface-models/repos/deepset/roberta-base-squad2`
* `~/Downloads/huggingface-models/repos/sshleifer/distilbart-cnn-12-6`
* `~/Downloads/huggingface-models/repos/Helsinki-NLP/opus-mt-fr-en`

The backend `.env` file should point at these local paths. The `.env` file is intentionally ignored by Git.

Example `.env` values:

    HF_HOME=/home/mlops/Downloads/huggingface-models/cache
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1

    SENTIMENT_MODEL_ID=/home/mlops/Downloads/huggingface-models/repos/distilbert/distilbert-base-uncased-finetuned-sst-2-english
    ZERO_SHOT_MODEL_ID=/home/mlops/Downloads/huggingface-models/repos/facebook/bart-large-mnli
    TEXT_GENERATION_MODEL_ID=/home/mlops/Downloads/huggingface-models/repos/distilbert/distilgpt2
    FILL_MASK_MODEL_ID=/home/mlops/Downloads/huggingface-models/repos/distilbert/distilbert-base-uncased
    NER_MODEL_ID=/home/mlops/Downloads/huggingface-models/repos/dslim/bert-base-NER
    QUESTION_ANSWERING_MODEL_ID=/home/mlops/Downloads/huggingface-models/repos/deepset/roberta-base-squad2
    SUMMARIZATION_MODEL_ID=/home/mlops/Downloads/huggingface-models/repos/sshleifer/distilbart-cnn-12-6
    TRANSLATION_MODEL_ID=/home/mlops/Downloads/huggingface-models/repos/Helsinki-NLP/opus-mt-fr-en

## Backend Setup

From the project root:

1. Change into the backend directory:

       cd backend

2. Create a virtual environment:

       python3 -m venv .venv

3. Activate the virtual environment:

       source .venv/bin/activate

4. Install backend dependencies:

       python -m pip install --no-index --find-links ~/Downloads/python-wheelhouse/wheels-hf-cpu-clean -r requirements.txt

5. Install CPU PyTorch dependencies:

       python -m pip install --no-index --find-links ~/Downloads/python-wheelhouse/wheels-hf-cpu-clean -r requirements-torch-cpu.txt

6. Start the backend:

       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uvicorn app.main:app --reload

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

If frontend changes do not appear immediately, hard refresh the browser with `Ctrl+F5`.

## Current Backend Routes

* `GET /`
* `GET /health`
* `POST /api/video-002/pipeline-function/sentiment`
* `POST /api/video-002/pipeline-function/zero-shot-classification`
* `POST /api/video-002/pipeline-function/text-generation`
* `POST /api/video-002/pipeline-function/fill-mask`
* `POST /api/video-002/pipeline-function/ner`
* `POST /api/video-002/pipeline-function/question-answering`
* `POST /api/video-002/pipeline-function/summarization`
* `POST /api/video-002/pipeline-function/translation`

## Testing

From the backend directory:

    source .venv/bin/activate
    pytest -v

Expected result:

    10 passed

## Notes on Transformers Pipeline Compatibility

The Hugging Face course introduces several tasks through the high-level `pipeline()` function.

In this local environment, some task names from the course are not registered in the installed Transformers pipeline registry. For those cases, the backend keeps the same course concept visible in the frontend while using lower-level tokenizer/model classes directly.

This makes the demo useful in two ways:

* `pipeline()` examples show the beginner-friendly course abstraction.
* `AutoTokenizer`, `AutoModel`, and Marian model examples show the lower-level implementation path when pipeline support is unavailable or when more control is needed.

## Status

Video 002 is complete.

Next likely work:

1. Update `docs/playlist-progress.md`.
2. Add or refresh notes under `demos/video_002_pipeline_function/`.
3. Start the next practical Hugging Face course demo.
