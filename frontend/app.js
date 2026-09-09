// Plain vanilla JS -- no build step, no framework. A linear wizard:
// pick entry context -> answer one question at a time -> final result.

const startScreen = document.getElementById("start-screen");
const questionScreen = document.getElementById("question-screen");
const resultScreen = document.getElementById("result-screen");

let caseId = null;
let currentQuestion = null;

async function api(path, options) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `request failed (${res.status})`);
  }
  return res.status === 204 ? null : res.json();
}

function showScreen(screen) {
  for (const s of [startScreen, questionScreen, resultScreen]) s.hidden = s !== screen;
}

function renderSafetyAlerts(container, alerts) {
  container.innerHTML = "";
  for (const a of alerts) {
    const div = document.createElement("div");
    div.className = `alert ${a.severity}`;
    div.innerHTML = `<strong>${a.message}</strong><br/>${a.escalation_text}`;
    container.appendChild(div);
  }
}

function renderRecommendationGroup(container, title, items) {
  if (!items || items.length === 0) return;
  const group = document.createElement("div");
  group.className = "rec-group";
  const h3 = document.createElement("h3");
  h3.textContent = title;
  const ul = document.createElement("ul");
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    ul.appendChild(li);
  }
  group.appendChild(h3);
  group.appendChild(ul);
  container.appendChild(group);
}

function renderStatus(status) {
  if (status.should_stop) {
    renderResult(status);
    return;
  }

  showScreen(questionScreen);
  document.getElementById("uncertainty-label").textContent =
    `Uncertainty: ${Math.round(status.uncertainty * 100)}%`;
  renderSafetyAlerts(document.getElementById("safety-alerts"), status.safety_alerts);

  currentQuestion = status.next_question;
  document.getElementById("question-text").textContent = currentQuestion.text;

  const optionsDiv = document.getElementById("answer-options");
  const numericForm = document.getElementById("numeric-form");
  optionsDiv.innerHTML = "";

  if (currentQuestion.data_type === "numeric") {
    numericForm.hidden = false;
    optionsDiv.hidden = true;
    const input = document.getElementById("numeric-input");
    input.value = "";
    input.placeholder = currentQuestion.unit || "";
  } else {
    numericForm.hidden = true;
    optionsDiv.hidden = false;
    for (const opt of currentQuestion.options) {
      const btn = document.createElement("button");
      btn.textContent = opt.label;
      btn.onclick = () => submitAnswer({ state: opt.state });
      optionsDiv.appendChild(btn);
    }
  }
}

function renderResult(status) {
  showScreen(resultScreen);
  renderSafetyAlerts(document.getElementById("safety-alerts-final"), status.safety_alerts);

  const top = status.ranked_candidates[0];
  document.getElementById("result-title").textContent = top ? top.problem_name : "No clear answer yet";
  document.getElementById("result-probability").textContent = top
    ? `Relative plausibility: ${Math.round(top.probability * 100)}% (not a certain diagnosis)`
    : "";

  const recDiv = document.getElementById("recommendations");
  recDiv.innerHTML = "";
  if (status.recommendations) {
    const r = status.recommendations;
    renderRecommendationGroup(recDiv, "Confirm this by checking", r.confirmation_checks);
    renderRecommendationGroup(recDiv, "Do now", r.immediate_actions);
    renderRecommendationGroup(recDiv, "Follow up", r.follow_up_actions);
    renderRecommendationGroup(recDiv, "Avoid", r.avoid);
    renderRecommendationGroup(recDiv, "Get professional help if", r.escalation_criteria);
  }

  resetFeedbackUI();
}

function resetFeedbackUI() {
  document.getElementById("feedback-comment").hidden = true;
  document.getElementById("feedback-submit").hidden = true;
  document.getElementById("feedback-thanks").hidden = true;
  document.getElementById("feedback-yes").hidden = false;
  document.getElementById("feedback-no").hidden = false;
}

async function submitAnswer(payload) {
  const status = await api(`/api/cases/${caseId}/answers`, {
    method: "POST",
    body: JSON.stringify({
      question_id: currentQuestion.question_id,
      evidence_id: currentQuestion.evidence_id,
      ...payload,
    }),
  });
  renderStatus(status);
}

async function startCase(entryContext) {
  const { case_id } = await api("/api/cases", {
    method: "POST",
    body: JSON.stringify({ entry_context: entryContext }),
  });
  caseId = case_id;
  const status = await api(`/api/cases/${caseId}/status`);
  document.getElementById("disclaimer").textContent = status.disclaimer;
  renderStatus(status);
}

async function loadEntryContexts() {
  const contexts = await api("/api/entry-contexts");
  const container = document.getElementById("entry-contexts");
  container.innerHTML = "";
  for (const ec of contexts) {
    const btn = document.createElement("button");
    btn.textContent = ec.name;
    btn.title = ec.description;
    btn.onclick = () => startCase(ec.id);
    container.appendChild(btn);
  }
  const unsureBtn = document.createElement("button");
  unsureBtn.textContent = "Not sure";
  unsureBtn.onclick = () => startCase(null);
  container.appendChild(unsureBtn);
}

document.getElementById("numeric-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const value = parseFloat(document.getElementById("numeric-input").value);
  if (!Number.isNaN(value)) submitAnswer({ raw_value: value });
});

document.getElementById("feedback-yes").onclick = () => showCommentBox(true);
document.getElementById("feedback-no").onclick = () => showCommentBox(false);

let pendingFeedbackHelpful = null;

function showCommentBox(helpful) {
  pendingFeedbackHelpful = helpful;
  document.getElementById("feedback-yes").hidden = true;
  document.getElementById("feedback-no").hidden = true;
  document.getElementById("feedback-comment").hidden = false;
  document.getElementById("feedback-submit").hidden = false;
}

document.getElementById("feedback-submit").onclick = async () => {
  const comment = document.getElementById("feedback-comment").value || null;
  await api(`/api/cases/${caseId}/feedback`, {
    method: "POST",
    body: JSON.stringify({ helpful: pendingFeedbackHelpful, comment }),
  });
  document.getElementById("feedback-comment").hidden = true;
  document.getElementById("feedback-submit").hidden = true;
  document.getElementById("feedback-thanks").hidden = false;
};

document.getElementById("restart").onclick = () => {
  caseId = null;
  currentQuestion = null;
  showScreen(startScreen);
};

loadEntryContexts();
