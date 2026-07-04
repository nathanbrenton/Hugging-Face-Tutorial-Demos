const API_BASE_URL = "http://127.0.0.1:8000";

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


const demoMetadata = {
  sentiment: {
    title: "Sentiment Analysis",
    pipelineCall: 'pipeline("sentiment-analysis")',
    modelUsed: "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    runtime:
      "When the user clicks Analyze Text, the browser sends text to the local FastAPI backend. The backend runs the local sentiment model and returns labels and confidence scores.",
    placeholder: "",
  },
  "zero-shot": {
    title: "Zero-Shot Classification",
    pipelineCall: 'pipeline("zero-shot-classification")',
    modelUsed: "facebook/bart-large-mnli",
    runtime:
      "When implemented, the user will submit text and candidate labels. The backend will use the local zero-shot model to rank which labels best match the text.",
    placeholder:
      "Zero-shot classification is the next demo to implement. It will use text plus candidate labels such as education, politics, sports, or finance.",
  },
  "text-generation": {
    title: "Text Generation",
    pipelineCall: 'pipeline("text-generation", model="distilgpt2")',
    modelUsed: "distilbert/distilgpt2",
    runtime:
      "When implemented, the user will submit a prompt. The backend will use the local DistilGPT-2 model to generate a continuation.",
    placeholder:
      "Text generation will be implemented after zero-shot classification. It will use a prompt and return generated text.",
  },
  "fill-mask": {
    title: "Fill-Mask",
    pipelineCall: 'pipeline("fill-mask")',
    modelUsed: "distilbert/distilbert-base-uncased",
    runtime: "Runs locally through the FastAPI backend using the downloaded DistilBERT base model.",
    placeholder: "",
  },
  ner: {
    title: "Named Entity Recognition",
    pipelineCall: 'pipeline("ner", grouped_entities=True)',
    modelUsed: "dslim/bert-base-NER",
    runtime:
      "When implemented, the user will submit text. The backend will identify entities such as people, organizations, and locations.",
    placeholder:
      "Named entity recognition will be implemented after text generation. The backend will group detected entity tokens into readable spans.",
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

  sentimentDemo.classList.remove("active");
  zeroShotDemo.classList.remove("active");
  textGenerationDemo.classList.remove("active");
  fillMaskDemo.classList.remove("active");
  placeholderDemo.classList.remove("active");

  if (demoName === "sentiment") {
    sentimentDemo.classList.add("active");
  } else if (demoName === "zero-shot") {
    zeroShotDemo.classList.add("active");
  } else if (demoName === "text-generation") {
    textGenerationDemo.classList.add("active");
  } else if (demoName === "fill-mask") {
    fillMaskDemo.classList.add("active");
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


runDemoButton.addEventListener("click", runVideo002SentimentDemo);

clearButton.addEventListener("click", () => {
  resultOutput.textContent = "Run the demo to see results here.";
});

runZeroShotButton.addEventListener("click", runVideo002ZeroShotDemo);

runTextGenerationButton.addEventListener("click", runVideo002TextGenerationDemo);

runFillMaskButton.addEventListener("click", runVideo002FillMaskDemo);

setActiveDemo("sentiment");
