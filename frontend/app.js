const API_BASE_URL = "http://127.0.0.1:8000";

const implementationNote = document.querySelector("#implementation-note");

const demoTabs = document.querySelectorAll(".demo-tab");
const sentimentDemo = document.querySelector("#sentiment-demo");
const placeholderDemo = document.querySelector("#placeholder-demo");
const placeholderText = document.querySelector("#placeholder-text");

const demoTitle = document.querySelector("#demo-title");
const pipelineCall = document.querySelector("#pipeline-call");
const modelUsed = document.querySelector("#model-used");
const modelRuntime = document.querySelector("#model-runtime");

const runDemoButton = document.querySelector("#run-demo-button");
const clearButton = document.querySelector("#clear-button");
const textInput = document.querySelector("#text-input");
const resultOutput = document.querySelector("#result-output");

const zeroShotDemo = document.querySelector("#zero-shot-demo");
const zeroShotTextInput = document.querySelector("#zero-shot-text-input");
const zeroShotLabelsInput = document.querySelector("#zero-shot-labels-input");
const runZeroShotButton = document.querySelector("#run-zero-shot-button");

const textGenerationDemo = document.querySelector("#text-generation-demo");
const textGenerationPromptInput = document.querySelector("#text-generation-prompt-input");
const textGenerationMaxTokensInput = document.querySelector("#text-generation-max-tokens-input");
const runTextGenerationButton = document.querySelector("#run-text-generation-button");

const fillMaskDemo = document.querySelector("#fill-mask-demo");
const fillMaskTextInput = document.querySelector("#fill-mask-text-input");
const fillMaskTopKInput = document.querySelector("#fill-mask-top-k-input");
const runFillMaskButton = document.querySelector("#run-fill-mask-button");

const nerDemo = document.querySelector("#ner-demo");
const nerTextInput = document.querySelector("#ner-text-input");
const runNerButton = document.querySelector("#run-ner-button");

const questionAnsweringDemo = document.querySelector("#question-answering-demo");
const questionAnsweringQuestionInput = document.querySelector("#question-answering-question-input");
const questionAnsweringContextInput = document.querySelector("#question-answering-context-input");
const runQuestionAnsweringButton = document.querySelector("#run-question-answering-button");

const summarizationDemo = document.querySelector("#summarization-demo");
const summarizationTextInput = document.querySelector("#summarization-text-input");
const summarizationMaxLengthInput = document.querySelector("#summarization-max-length-input");
const summarizationMinLengthInput = document.querySelector("#summarization-min-length-input");
const runSummarizationButton = document.querySelector("#run-summarization-button");

const translationDemo = document.querySelector("#translation-demo");
const translationTextInput = document.querySelector("#translation-text-input");
const translationMaxLengthInput = document.querySelector("#translation-max-length-input");
const runTranslationButton = document.querySelector("#run-translation-button");



const demoMetadata = {
  sentiment: {
    title: "Sentiment Analysis (aka Text Classification)",
    pipelineCall: 'pipeline("sentiment-analysis")',
    modelUsed: "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    runtime:
      "When the user clicks Analyze Text, the browser sends text to the local FastAPI backend. The backend runs the local sentiment model and returns labels and confidence scores.",
    implementationNote: "This local demo uses the standard Hugging Face pipeline abstraction, matching the course concept.",
    placeholder: "",
  },
  "zero-shot": {
    title: "Zero-Shot Classification",
    pipelineCall: 'pipeline("zero-shot-classification")',
    modelUsed: "facebook/bart-large-mnli",
    runtime:
      "When implemented, the user will submit text and candidate labels. The backend will use the local zero-shot model to rank which labels best match the text.",
    implementationNote: "This local demo uses the standard Hugging Face pipeline abstraction, matching the course concept.",
    placeholder:
      "Zero-shot classification is the next demo to implement. It will use text plus candidate labels such as education, politics, sports, or finance.",
  },
  "text-generation": {
    title: "Text Generation",
    pipelineCall: 'pipeline("text-generation", model="distilgpt2")',
    modelUsed: "distilbert/distilgpt2",
    runtime:
      "When implemented, the user will submit a prompt. The backend will use the local DistilGPT-2 model to generate a continuation.",
    implementationNote: "This local demo uses the standard Hugging Face pipeline abstraction, matching the course concept.",
    placeholder:
      "Text generation will be implemented after zero-shot classification. It will use a prompt and return generated text.",
  },
  "fill-mask": {
    title: "Fill-Mask",
    pipelineCall: 'pipeline("fill-mask")',
    modelUsed: "distilbert/distilbert-base-uncased",
    runtime: "Runs locally through the FastAPI backend using the downloaded DistilBERT base model.",
    implementationNote: "This local demo uses the standard Hugging Face pipeline abstraction, matching the course concept.",
    placeholder: "",
  },
  "ner": {
    title: "Named Entity Recognition",
    pipelineCall: 'pipeline("ner", grouped_entities=True)',
    modelUsed: "dslim/bert-base-NER",
    runtime:
      "When implemented, the user will submit text. The backend will identify entities such as people, organizations, and locations.",
    implementationNote: "This local demo uses the standard Hugging Face pipeline abstraction, matching the course concept.",
    placeholder:
      "Named entity recognition will be implemented after text generation. The backend will group detected entity tokens into readable spans.",
  },
  "question-answering": {
    title: "Question Answering",
    pipelineCall: 'pipeline("question-answering")',
    modelUsed: "deepset/roberta-base-squad2",
    runtime: "Runs locally through the FastAPI backend using the downloaded RoBERTa SQuAD2 model. The backend uses the tokenizer and model directly because this installed Transformers version does not register pipeline(\"question-answering\").",
    implementationNote: 'The course introduces this as pipeline("question-answering"). In this local environment, that task name is not registered in the installed Transformers pipeline registry, so the backend uses AutoTokenizer + AutoModelForQuestionAnswering directly.',
    placeholder: "",
  },
  summarization: {
    title: "Summarization",
    pipelineCall: 'pipeline("summarization")',
    modelUsed: "sshleifer/distilbart-cnn-12-6",
    runtime: 'Runs locally through the FastAPI backend using the downloaded DistilBART CNN summarization model. The backend uses the tokenizer and model directly because this installed Transformers version does not register pipeline("summarization").',
    implementationNote: 'The course introduces this as pipeline("summarization"). In this local environment, that task name is not registered in the installed Transformers pipeline registry, so the backend uses AutoTokenizer + AutoModelForSeq2SeqLM directly.',
    placeholder: "",
  },
  translation: {
    title: "Translation",
    pipelineCall: 'pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")',
    modelUsed: "Helsinki-NLP/opus-mt-fr-en",
    runtime: 'Runs locally through the FastAPI backend using the downloaded Helsinki-NLP Marian French-to-English model. The backend uses MarianTokenizer and MarianMTModel directly because this installed Transformers version does not register pipeline("translation").',
    implementationNote: 'The course introduces this as pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en"). In this local environment, that task name is not registered in the installed Transformers pipeline registry, so the backend uses MarianTokenizer + MarianMTModel directly.',
    placeholder: "",
  },
};

function parseTexts(rawText) {
  return rawText
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

function setActiveDemo(demoName) {
  const metadata = demoMetadata[demoName];

  demoTabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.demo === demoName);
  });

  demoTitle.textContent = metadata.title;
  pipelineCall.textContent = metadata.pipelineCall;
  modelUsed.textContent = metadata.modelUsed;
  modelRuntime.textContent = metadata.runtime;
  implementationNote.textContent = metadata.implementationNote;

  sentimentDemo.classList.remove("active");
  zeroShotDemo.classList.remove("active");
  textGenerationDemo.classList.remove("active");
  fillMaskDemo.classList.remove("active");
  nerDemo.classList.remove("active");
  questionAnsweringDemo.classList.remove("active");
  summarizationDemo.classList.remove("active");
  translationDemo.classList.remove("active");
  placeholderDemo.classList.remove("active");

  if (demoName === "sentiment") {
    sentimentDemo.classList.add("active");
  } else if (demoName === "zero-shot") {
    zeroShotDemo.classList.add("active");
  } else if (demoName === "text-generation") {
    textGenerationDemo.classList.add("active");
  } else if (demoName === "fill-mask") {
    fillMaskDemo.classList.add("active");
  } else if (demoName === "ner") {
    nerDemo.classList.add("active");
  } else if (demoName === "question-answering") {
    questionAnsweringDemo.classList.add("active");
  } else if (demoName === "summarization") {
    summarizationDemo.classList.add("active");
  } else if (demoName === "translation") {
    translationDemo.classList.add("active");
  } else {
    placeholderDemo.classList.add("active");
    placeholderText.textContent = metadata.placeholder;
    resultOutput.textContent = "This demo is documented in the UI and will be wired to the backend next.";
  }
}

async function runVideo002SentimentDemo() {
  const texts = parseTexts(textInput.value);

  if (texts.length === 0) {
    resultOutput.textContent = "Please enter at least one text input.";
    return;
  }

  runDemoButton.disabled = true;
  resultOutput.textContent = "Running sentiment analysis...";

  try {
    const response = await fetch(`${API_BASE_URL}/api/video-002/pipeline-function/sentiment`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ texts }),
    });

    const data = await response.json();

    resultOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultOutput.textContent = `Request failed: ${error}`;
  } finally {
    runDemoButton.disabled = false;
  }
}

demoTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    setActiveDemo(tab.dataset.demo);
  });
});

function parseLabels(rawText) {
  return rawText
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

async function runVideo002ZeroShotDemo() {
  const text = zeroShotTextInput.value.trim();
  const candidateLabels = parseLabels(zeroShotLabelsInput.value);

  if (text.length === 0) {
    resultOutput.textContent = "Please enter text to classify.";
    return;
  }

  if (candidateLabels.length < 2) {
    resultOutput.textContent = "Please enter at least two candidate labels.";
    return;
  }

  runZeroShotButton.disabled = true;
  resultOutput.textContent = "Running zero-shot classification...";

  try {
    const response = await fetch(`${API_BASE_URL}/api/video-002/pipeline-function/zero-shot-classification`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        candidate_labels: candidateLabels,
      }),
    });

    const data = await response.json();

    resultOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultOutput.textContent = `Request failed: ${error}`;
  } finally {
    runZeroShotButton.disabled = false;
  }
}

async function runVideo002TextGenerationDemo() {
  const prompt = textGenerationPromptInput.value.trim();
  const maxNewTokens = Number(textGenerationMaxTokensInput.value);

  if (prompt.length === 0) {
    resultOutput.textContent = "Please enter a prompt.";
    return;
  }

  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 100) {
    resultOutput.textContent = "Max new tokens must be a whole number between 1 and 100.";
    return;
  }

  runTextGenerationButton.disabled = true;
  resultOutput.textContent = "Generating text...";

  try {
    const response = await fetch(`${API_BASE_URL}/api/video-002/pipeline-function/text-generation`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt,
        max_new_tokens: maxNewTokens,
      }),
    });

    const data = await response.json();

    resultOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultOutput.textContent = `Request failed: ${error}`;
  } finally {
    runTextGenerationButton.disabled = false;
  }
}

async function runVideo002FillMaskDemo() {
  const text = fillMaskTextInput.value.trim();
  const topK = Number(fillMaskTopKInput.value);

  if (text.length === 0) {
    resultOutput.textContent = "Please enter text containing a mask token.";
    return;
  }

  if (!text.includes("[MASK]")) {
    resultOutput.textContent = "Please include the [MASK] token in the text.";
    return;
  }

  if (!Number.isInteger(topK) || topK < 1 || topK > 10) {
    resultOutput.textContent = "Top predictions must be a whole number between 1 and 10.";
    return;
  }

  runFillMaskButton.disabled = true;
  resultOutput.textContent = "Filling mask...";

  try {
    const response = await fetch(`${API_BASE_URL}/api/video-002/pipeline-function/fill-mask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        top_k: topK,
      }),
    });

    const data = await response.json();

    resultOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultOutput.textContent = `Request failed: ${error}`;
  } finally {
    runFillMaskButton.disabled = false;
  }
}


async function runVideo002NerDemo() {
  const text = nerTextInput.value.trim();

  if (text.length === 0) {
    resultOutput.textContent = "Please enter text for named entity recognition.";
    return;
  }

  runNerButton.disabled = true;
  resultOutput.textContent = "Extracting named entities...";

  try {
    const response = await fetch(`${API_BASE_URL}/api/video-002/pipeline-function/ner`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
      }),
    });

    const data = await response.json();

    resultOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultOutput.textContent = `Request failed: ${error}`;
  } finally {
    runNerButton.disabled = false;
  }
}



async function runVideo002QuestionAnsweringDemo() {
  const question = questionAnsweringQuestionInput.value.trim();
  const context = questionAnsweringContextInput.value.trim();

  if (question.length === 0) {
    resultOutput.textContent = "Please enter a question.";
    return;
  }

  if (context.length === 0) {
    resultOutput.textContent = "Please enter context.";
    return;
  }

  runQuestionAnsweringButton.disabled = true;
  resultOutput.textContent = "Answering question...";

  try {
    const response = await fetch(`${API_BASE_URL}/api/video-002/pipeline-function/question-answering`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        context,
      }),
    });

    const data = await response.json();

    resultOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultOutput.textContent = `Request failed: ${error}`;
  } finally {
    runQuestionAnsweringButton.disabled = false;
  }
}




async function runVideo002SummarizationDemo() {
  const text = summarizationTextInput.value.trim();
  const maxLength = Number(summarizationMaxLengthInput.value);
  const minLength = Number(summarizationMinLengthInput.value);

  if (text.length === 0) {
    resultOutput.textContent = "Please enter text to summarize.";
    return;
  }

  if (!Number.isInteger(maxLength) || maxLength < 20 || maxLength > 150) {
    resultOutput.textContent = "Max summary length must be a whole number between 20 and 150.";
    return;
  }

  if (!Number.isInteger(minLength) || minLength < 5 || minLength > 100) {
    resultOutput.textContent = "Min summary length must be a whole number between 5 and 100.";
    return;
  }

  if (minLength > maxLength) {
    resultOutput.textContent = "Min summary length cannot be greater than max summary length.";
    return;
  }

  runSummarizationButton.disabled = true;
  resultOutput.textContent = "Summarizing text...";

  try {
    const response = await fetch(`${API_BASE_URL}/api/video-002/pipeline-function/summarization`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        max_length: maxLength,
        min_length: minLength,
      }),
    });

    const data = await response.json();

    resultOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultOutput.textContent = `Request failed: ${error}`;
  } finally {
    runSummarizationButton.disabled = false;
  }
}



async function runVideo002TranslationDemo() {
  const text = translationTextInput.value.trim();
  const maxLength = Number(translationMaxLengthInput.value);

  if (text.length === 0) {
    resultOutput.textContent = "Please enter French text to translate.";
    return;
  }

  if (!Number.isInteger(maxLength) || maxLength < 10 || maxLength > 200) {
    resultOutput.textContent = "Max translation length must be a whole number between 10 and 200.";
    return;
  }

  runTranslationButton.disabled = true;
  resultOutput.textContent = "Translating text...";

  try {
    const response = await fetch(`${API_BASE_URL}/api/video-002/pipeline-function/translation`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        max_length: maxLength,
      }),
    });

    const data = await response.json();

    resultOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultOutput.textContent = `Request failed: ${error}`;
  } finally {
    runTranslationButton.disabled = false;
  }
}




runDemoButton.addEventListener("click", runVideo002SentimentDemo);

clearButton.addEventListener("click", () => {
  resultOutput.textContent = "Run the demo to see results here.";
});

runZeroShotButton.addEventListener("click", runVideo002ZeroShotDemo);

runTextGenerationButton.addEventListener("click", runVideo002TextGenerationDemo);

runFillMaskButton.addEventListener("click", runVideo002FillMaskDemo);

runNerButton.addEventListener("click", runVideo002NerDemo);

runQuestionAnsweringButton.addEventListener("click", runVideo002QuestionAnsweringDemo);

runSummarizationButton.addEventListener("click", runVideo002SummarizationDemo);

runTranslationButton.addEventListener("click", runVideo002TranslationDemo);





setActiveDemo("sentiment");
