const API_BASE_URL = "http://127.0.0.1:8000";

const runDemoButton = document.querySelector("#run-demo-button");
const clearButton = document.querySelector("#clear-button");
const textInput = document.querySelector("#text-input");
const resultOutput = document.querySelector("#result-output");

function parseTexts(rawText) {
  return rawText
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

async function runVideo02PipelineDemo() {
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

    if (!response.ok) {
      resultOutput.textContent = JSON.stringify(data, null, 2);
      return;
    }

    resultOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultOutput.textContent = `Request failed: ${error}`;
  } finally {
    runDemoButton.disabled = false;
  }
}

runDemoButton.addEventListener("click", runVideo02PipelineDemo);

clearButton.addEventListener("click", () => {
  resultOutput.textContent = "Run the demo to see results here.";
});
