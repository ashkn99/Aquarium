// Plain vanilla JS -- no build step, no framework. A linear wizard:
// pick entry context -> answer one question at a time -> final result.

const startScreen = document.getElementById("start-screen");
const concernScreen = document.getElementById("concern-screen");
const questionScreen = document.getElementById("question-screen");
const resultScreen = document.getElementById("result-screen");

let caseId = null;
let currentQuestion = null;
let entryContextsById = {};
let questionIndex = 0;

// Coarse, human framing for the uncertainty score -- a bare "72%" doesn't
// tell a first-time user whether that's good or bad or how much is left.
function uncertaintyPhase(uncertainty) {
  if (uncertainty >= 0.75) return "just getting started";
  if (uncertainty >= 0.4) return "narrowing it down";
  return "close to an answer";
}

async function api(path, options) {
  document.getElementById("error-banner").hidden = true;
  let res;
  try {
    res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new Error("Couldn't reach the server. Check your connection and try again.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Something went wrong (${res.status}). Please try again.`);
  }
  return res.status === 204 ? null : res.json();
}

// Every click handler below is an unawaited async call (`onclick = () =>
// submitAnswer(...)`), so a thrown error has nowhere to be caught except
// here -- without this, a failed request (cold start, rate limit, a
// network blip) left the user staring at a button that silently did
// nothing.
window.addEventListener("unhandledrejection", (event) => {
  const banner = document.getElementById("error-banner");
  banner.textContent = event.reason?.message || "Something went wrong. Please try again.";
  banner.hidden = false;
  event.preventDefault();
});

function showScreen(screen) {
  for (const s of [startScreen, concernScreen, questionScreen, resultScreen]) s.hidden = s !== screen;
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
  questionIndex += 1;
  document.getElementById("uncertainty-label").textContent =
    `Question ${questionIndex} · ${uncertaintyPhase(status.uncertainty)}`;
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

  const concerns = entryContext ? entryContextsById[entryContext]?.concerns || [] : [];
  if (concerns.length > 0) {
    renderConcernScreen(concerns);
    return;
  }
  await proceedPastConcern();
}

function renderConcernScreen(concerns) {
  showScreen(concernScreen);
  const container = document.getElementById("concern-options");
  container.innerHTML = "";
  for (const concern of concerns) {
    const btn = document.createElement("button");
    btn.textContent = concern.name;
    btn.title = concern.description;
    btn.onclick = () => pickConcern(concern.id);
    container.appendChild(btn);
  }
  const skipBtn = document.createElement("button");
  skipBtn.textContent = "Not sure / something else";
  skipBtn.onclick = () => pickConcern(null);
  container.appendChild(skipBtn);
}

async function pickConcern(concernId) {
  const status = await api(`/api/cases/${caseId}/concern`, {
    method: "POST",
    body: JSON.stringify({ concern_id: concernId }),
  });
  document.getElementById("disclaimer").textContent = status.disclaimer;
  renderStatus(status);
}

async function proceedPastConcern() {
  const status = await api(`/api/cases/${caseId}/status`);
  document.getElementById("disclaimer").textContent = status.disclaimer;
  renderStatus(status);
}

// The KB defines more entry contexts than are worth surfacing as
// top-level starting points (e.g. "aquarium_environment" overlaps
// heavily with "water" and adds a 5th choice to an onboarding screen
// that should stay simple) -- this is a presentation-layer trim, not a
// KB change: the other contexts still work if ever addressed directly,
// they're just not offered as a button here. "unsure" already IS the
// KB's own no-particular-guess context (named "I'm not sure"), so it's
// used directly rather than duplicating it with a separate null-context
// button.
const VISIBLE_ENTRY_CONTEXTS = ["fish", "water", "plants", "unsure"];

async function loadEntryContexts() {
  const contexts = await api("/api/entry-contexts");
  entryContextsById = Object.fromEntries(contexts.map((ec) => [ec.id, ec]));
  const container = document.getElementById("entry-contexts");
  container.innerHTML = "";
  for (const id of VISIBLE_ENTRY_CONTEXTS) {
    const ec = entryContextsById[id];
    if (!ec) continue;
    const btn = document.createElement("button");
    btn.textContent = ec.name;
    btn.title = ec.description;
    btn.onclick = () => startCase(ec.id);
    container.appendChild(btn);
  }
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
  questionIndex = 0;
  showScreen(startScreen);
};

loadEntryContexts();
