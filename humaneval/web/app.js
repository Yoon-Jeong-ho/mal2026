const USERS = ["명훈", "찬희", "정호", "지민"];
const AXES = ["content", "organization", "expression"];
const LABELS = { content: "내용", organization: "구성", expression: "표현" };
const SCORE_REASON_PLACEHOLDERS = {
  content: "주장과 근거가 어느 정도 충실한지 간단히 적어 주세요",
  organization: "서론·본론·결론과 논리 전개를 어떻게 판단했는지 적어 주세요",
  expression: "문장, 어휘, 맞춤법과 가독성을 어떻게 판단했는지 적어 주세요",
};
let state = null;

const $ = (selector) => document.querySelector(selector);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: options.body ? { "Content-Type": "application/json" } : {},
    body: options.body ? JSON.stringify(options.body) : undefined,
    credentials: "same-origin",
  });
  const payload = await response.json().catch(() => ({ error: "응답을 읽을 수 없습니다." }));
  if (!response.ok) throw new Error(payload.error || "요청을 처리하지 못했습니다.");
  return payload;
}

let toastTimer;
function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("visible"), 3200);
}

function setBusy(form, busy) {
  form.querySelectorAll("button, input, textarea").forEach((node) => { node.disabled = busy; });
}

function showLogin() {
  state = null;
  $("#login-view").classList.remove("hidden");
  $("#study-view").classList.add("hidden");
  $("#user-area").classList.add("hidden");
}

function renderRubric(rubric) {
  const root = $("#rubric-content");
  root.replaceChildren();
  const criteria = el("div", "criteria-grid");
  AXES.forEach((axis) => {
    const card = el("div", "criterion");
    card.append(el("strong", "", rubric.criteria[axis].label));
    const list = el("ul");
    rubric.criteria[axis].points.forEach((point) => list.append(el("li", "", point)));
    card.append(list);
    criteria.append(card);
  });
  root.append(criteria);
  const scale = el("div", "scale-list");
  [1, 2, 3, 4, 5].forEach((score) => {
    const item = el("div");
    item.append(el("strong", "", `${score}점`), document.createTextNode(rubric.score_scale[String(score)]));
    scale.append(item);
  });
  root.append(scale);
}

function renderScoreForm() {
  const root = $("#score-fields");
  root.replaceChildren();
  AXES.forEach((axis) => {
    const field = el("section", "score-field");
    const copy = el("div");
    copy.append(el("h4", "", LABELS[axis]));
    copy.append(el("p", "", state.rubric.criteria[axis].points.join(" · ")));
    const options = el("div", "score-options");
    [1, 2, 3, 4, 5].forEach((score) => {
      const label = el("label");
      const input = el("input");
      input.type = "radio";
      input.name = `score-${axis}`;
      input.value = String(score);
      input.required = true;
      label.append(input, el("span", "", String(score)));
      options.append(label);
    });
    const reason = el("label", "reason-field axis-reason");
    const reasonTitle = el("span", "", `${LABELS[axis]} 판단 이유 `);
    reasonTitle.append(el("small", "", "선택 · 특히 1점과 2점의 구분 근거를 적어 주세요"));
    const textarea = el("textarea");
    textarea.id = `score-reason-${axis}`;
    textarea.maxLength = 2000;
    textarea.rows = 2;
    textarea.placeholder = SCORE_REASON_PLACEHOLDERS[axis];
    reason.append(reasonTitle, textarea);
    field.append(copy, options, reason);
    root.append(field);
  });
}

function renderRationale() {
  $("#rationale-label").textContent = state.rationale.label;
  $("#rationale-submit").textContent = state.rationale.label === "A" ? "확인하고 설명 B 보기" : "확인하고 다음 글로";
  $("#judge-guide-intro").textContent = state.judge_guide.intro;
  const guideChecks = $("#judge-guide-checks");
  guideChecks.replaceChildren();
  state.judge_guide.checks.forEach((check) => {
    const item = el("div", "judge-check");
    item.append(el("strong", "", check.title), el("p", "", check.description));
    guideChecks.append(item);
  });
  const recap = $("#score-recap");
  recap.replaceChildren(el("strong", "", "내가 준 점수"));
  AXES.forEach((axis) => recap.append(el("span", "", `${LABELS[axis]} ${state.submitted_scores[axis]}점`)));
  const fields = $("#rationale-fields");
  fields.replaceChildren();
  AXES.forEach((axis) => {
    const card = el("section", "rationale-axis-review");
    const explanation = el("div", "rationale-axis");
    explanation.append(el("strong", "", `${LABELS[axis]} 평가 설명`), el("p", "", state.rationale.texts[axis]));
    const verdict = el("fieldset", "verdict-field");
    verdict.append(el("legend", "", `${LABELS[axis]} 설명은 적절한가요?`));
    const options = el("div", "verdict-grid");
    [
      ["appropriate", "적절함", "근거와 판단이 타당함"],
      ["partial", "일부 적절함", "맞는 부분과 아쉬운 부분이 함께 있음"],
      ["inappropriate", "적절하지 않음", "핵심 판단이나 근거가 타당하지 않음"],
    ].forEach(([value, title, note]) => {
      const label = el("label");
      const input = el("input");
      input.type = "radio";
      input.name = `verdict-${axis}`;
      input.value = value;
      const choice = el("span");
      choice.append(el("strong", "", title), el("small", "", note));
      label.append(input, choice);
      options.append(label);
    });
    verdict.append(options);
    const reason = el("label", "reason-field");
    const title = el("span", "", `${LABELS[axis]} 설명 판단 이유 `);
    title.append(el("small", "", "선택"));
    const textarea = el("textarea");
    textarea.id = `rationale-reason-${axis}`;
    textarea.maxLength = 2000;
    textarea.rows = 2;
    textarea.placeholder = `${LABELS[axis]} 설명의 어떤 부분이 적절하거나 부적절한지 적어 주세요`;
    reason.append(title, textarea);
    card.append(explanation, verdict, reason);
    fields.append(card);
  });
}

function render(nextState, { scroll = true } = {}) {
  state = nextState;
  $("#login-view").classList.add("hidden");
  $("#study-view").classList.remove("hidden");
  $("#user-area").classList.remove("hidden");
  $("#user-name").textContent = `${state.user} 평가자`;
  $("#common-notice").textContent = state.common_notice;
  renderRubric(state.rubric);
  const { completed, total } = state.progress;
  $("#progress-label").textContent = `${completed} / ${total} 완료`;
  $("#progress-bar").style.width = `${(completed / total) * 100}%`;

  if (state.phase === "finished") {
    $("#phase-label").textContent = "평가 완료";
    $("#completion-view").classList.remove("hidden");
    $("#item-view").classList.add("hidden");
    return;
  }
  $("#completion-view").classList.add("hidden");
  $("#item-view").classList.remove("hidden");
  $("#item-number").textContent = `${state.item.number} / ${total}`;
  $("#topic-prompt").textContent = state.item.topic_prompt;
  $("#essay").textContent = state.item.essay;

  const scoring = state.phase === "score";
  const stage = scoring ? "독립 채점" : `평가 설명 ${state.rationale.label} 검토`;
  $("#phase-label").textContent = stage;
  $("#stage-pill").textContent = stage;
  $("#score-form").classList.toggle("hidden", !scoring);
  $("#rationale-form").classList.toggle("hidden", scoring);
  if (scoring) renderScoreForm(); else renderRationale();
  if (scroll) $("#item-view").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function login(name) {
  try {
    render(await api("/api/login", { method: "POST", body: { name } }), { scroll: false });
  } catch (error) { toast(error.message); }
}

function buildLogin() {
  const grid = $("#name-grid");
  USERS.forEach((name) => {
    const button = el("button", "name-button", name);
    button.type = "button";
    button.addEventListener("click", () => login(name));
    grid.append(button);
  });
}

$("#score-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const scores = {};
  const reasons = {};
  for (const axis of AXES) {
    const checked = document.querySelector(`input[name="score-${axis}"]:checked`);
    if (!checked) { toast("세 영역의 점수를 모두 선택해 주세요."); return; }
    scores[axis] = Number(checked.value);
    reasons[axis] = $(`#score-reason-${axis}`).value;
  }
  const form = event.currentTarget;
  setBusy(form, true);
  try {
    const next = await api("/api/score", {
      method: "POST",
      body: { item_index: state.item.index, scores, reasons },
    });
    render(next);
  } catch (error) { toast(error.message); }
  finally { setBusy(form, false); }
});

$("#rationale-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const verdicts = {};
  const reasons = {};
  for (const axis of AXES) {
    const checked = document.querySelector(`input[name="verdict-${axis}"]:checked`);
    if (!checked) { toast("내용, 구성, 표현 설명의 적절성을 모두 선택해 주세요."); return; }
    verdicts[axis] = checked.value;
    reasons[axis] = $(`#rationale-reason-${axis}`).value;
  }
  const form = event.currentTarget;
  setBusy(form, true);
  try {
    const next = await api("/api/rationale", {
      method: "POST",
      body: { item_index: state.item.index, verdicts, reasons },
    });
    render(next, { scroll: state.rationale.label === "B" });
  } catch (error) { toast(error.message); }
  finally { setBusy(form, false); }
});

$("#logout").addEventListener("click", async () => {
  try { await api("/api/logout", { method: "POST", body: {} }); }
  catch (_) { /* A local logout should still clear the visible state. */ }
  showLogin();
});

async function bootstrap() {
  buildLogin();
  try { render(await api("/api/state"), { scroll: false }); }
  catch (_) { showLogin(); }
}

bootstrap();
