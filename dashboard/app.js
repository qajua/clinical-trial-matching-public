const $ = (selector) => document.querySelector(selector);
const API_BASE = window.API_BASE_URL || "http://127.0.0.1:8766";

const EXAMPLE_QUESTIONS = [
  "Does the note describe the patient as having heart failure?",
  "Does the note describe the patient as having arterial hypertension on treatment?",
  "Does the note describe the patient as ever having a stroke or transient ischemic attack?",
  "What is the highest serum creatinine mentioned in the note?",
  "What is the lowest hemoglobin mentioned in the note?",
];

let currentAnswer = "";
let notes = [];

function setStatus(text, mode = "idle") {
  const shell = document.body;
  $("#statusLine").textContent = text;
  shell.dataset.mode = mode;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function loadNotes() {
  const response = await fetch(`${API_BASE}/api/notes`);
  const payload = await response.json();
  notes = payload.notes || [];
  $("#noteSelect").innerHTML = notes
    .map((note) => {
      const label = `${note.note_id} | HADM ${note.hadm_id}`;
      return `<option value="${escapeHtml(note.note_id)}">${escapeHtml(label)}</option>`;
    })
    .join("");
  renderNotePreview();
}

function renderExamples() {
  $("#exampleQuestions").innerHTML = EXAMPLE_QUESTIONS.map(
    (question) => `<button class="example-chip" type="button">${escapeHtml(question)}</button>`,
  ).join("");
  document.querySelectorAll(".example-chip").forEach((button) => {
    button.addEventListener("click", () => {
      $("#questionInput").value = button.textContent;
      $("#questionInput").focus();
    });
  });
}

function selectedNote() {
  const noteId = $("#noteSelect").value;
  return notes.find((note) => note.note_id === noteId);
}

function renderNotePreview() {
  const note = selectedNote();
  $("#notePreview").innerHTML = note
    ? `<strong>${escapeHtml(note.note_id)}</strong><span>HADM ${escapeHtml(note.hadm_id)}</span><p>${escapeHtml(note.preview || "")}</p>`
    : "No note selected.";
}

function setSteps(mode) {
  const steps = Array.from(document.querySelectorAll("#runSteps span"));
  steps.forEach((step, index) => {
    step.className = "";
    if (mode === "busy") {
      step.classList.add(index === 0 ? "active" : "pending");
    } else if (mode === "done") {
      step.classList.add("done");
    } else if (mode === "error") {
      step.classList.add(index === 0 ? "error" : "pending");
    }
  });
}

function renderQueryOptimization(qopt) {
  const subQueries = (qopt.sub_queries || [])
    .map((item, index) => `<li><span>${index + 1}</span>${escapeHtml(item)}</li>`)
    .join("");

  $("#queryFlow").innerHTML = `
    <article class="flow-step primary">
      <div class="step-label">Rewritten Query</div>
      <p>${escapeHtml(qopt.rewritten_query || "-")}</p>
    </article>
    <article class="flow-step">
      <div class="step-label">Sub-Queries</div>
      <ol>${subQueries}</ol>
    </article>
  `;
}

function renderChunks(chunks) {
  $("#chunkCount").textContent = `${chunks.length} chunks`;
  $("#chunkList").innerHTML = chunks.length
    ? chunks
        .map(
          (chunk) => {
            const text = String(chunk.text || "").trim();
            const preview = text.length > 520 ? `${text.slice(0, 520).trim()}...` : text;
            const needsExpansion = text.length > 520;
            return `
            <article class="chunk-card">
              <div class="chunk-topline">
                <span class="rank">#${chunk.rank}</span>
                <span class="chunk-length">${text.length.toLocaleString()} chars</span>
              </div>
              <p class="chunk-preview">${escapeHtml(preview)}</p>
              ${
                needsExpansion
                  ? `<details class="chunk-details">
                      <summary>Expand full chunk</summary>
                      <p>${escapeHtml(text)}</p>
                    </details>`
                  : ""
              }
            </article>
          `;
          },
        )
        .join("")
    : '<div class="empty-state">No retrieved chunks returned.</div>';
}

function renderPrimeKg(result) {
  const primekg = result.primekg || {};
  const evidence = String(primekg.evidence || "").trim();
  const panel = $("#primekgPanel");
  if (!evidence) {
    panel.classList.add("hidden");
    $("#primekgBox").innerHTML = "";
    return;
  }

  panel.classList.remove("hidden");
  const entities = [...(primekg.diseases || []), ...(primekg.drugs || [])];
  const tags = entities.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("");

  $("#primekgBox").innerHTML = `
    ${tags ? `<div class="tag-row">${tags}</div>` : ""}
    <p>${escapeHtml(evidence)}</p>
  `;
}

function renderResult(result) {
  currentAnswer = result.answer || "-";
  $("#answerText").textContent = currentAnswer;
  $("#answerText").classList.remove("pending");
  $("#copyAnswerButton").disabled = false;
  const questionType = result.question_type === "yes" ? "Boolean" : "Numeric";
  const answerMethod = {
    llm_json: "LLM JSON",
    rule_numeric_extraction: "Rule extraction",
    llm: "LLM",
  }[result.answer_source] || result.answer_source || "Unknown";
  $("#answerMeta").innerHTML = `
    <span class="pill">Question type: ${escapeHtml(questionType)}</span>
    <span class="pill">Method: ${escapeHtml(answerMethod)}</span>
    <span class="pill">Matched note: ${escapeHtml(result.note?.note_id || "-")}</span>
    <span class="pill">HADM: ${escapeHtml(result.note?.hadm_id || "-")}</span>
  `;
  $("#rawPrediction").textContent = result.raw_prediction || "Answered without an LLM call.";
  renderQueryOptimization(result.query_optimization || {});
  renderChunks(result.top_chunks || []);
  renderPrimeKg(result);
}

async function runMatch() {
  const question = $("#questionInput").value.trim();
  if (!question) {
    setStatus("Enter a question before running matching.", "error");
    setSteps("error");
    return;
  }

  $("#runButton").disabled = true;
  $("#runButton").textContent = "Running...";
  $("#answerText").textContent = "Running";
  $("#answerText").classList.add("pending");
  $("#copyAnswerButton").disabled = true;
  setStatus("Running match pipeline...", "busy");
  setSteps("busy");

  try {
    const response = await fetch(`${API_BASE}/api/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        note_id: $("#noteSelect").value,
        include_primekg: true,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Match request failed.");
    }
    renderResult(payload);
    setStatus("Match complete", "done");
    setSteps("done");
  } catch (error) {
    setStatus(error.message, "error");
    setSteps("error");
  } finally {
    $("#runButton").disabled = false;
    $("#runButton").textContent = "Run Matching";
  }
}

$("#runButton").addEventListener("click", runMatch);
$("#noteSelect").addEventListener("change", renderNotePreview);
$("#clearQuestionButton").addEventListener("click", () => {
  $("#questionInput").value = "";
  $("#questionInput").focus();
});
$("#randomNoteButton").addEventListener("click", () => {
  if (!notes.length) return;
  const note = notes[Math.floor(Math.random() * notes.length)];
  $("#noteSelect").value = note.note_id;
  renderNotePreview();
});
$("#copyAnswerButton").addEventListener("click", async () => {
  if (!currentAnswer) return;
  await navigator.clipboard.writeText(currentAnswer);
  setStatus("Answer copied", "done");
});
$("#questionInput").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    runMatch();
  }
});

renderExamples();
loadNotes()
  .then(() => {
    setStatus("Ready");
    setSteps("idle");
  })
  .catch((error) => setStatus(error.message, "error"));