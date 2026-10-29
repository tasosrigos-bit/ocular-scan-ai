// Ocular Scan AI front-end. Talks to the FastAPI service in app/api.py: the scan
// goes to /api/predict, the conversation (kept here, not on the server) to /api/chat.

const CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL'];
// The full name next to the class code, left empty where it would repeat the code.
const FULL = {
  CNV: 'Choroidal neovascularisation',
  DME: 'Diabetic macular oedema',
  DRUSEN: '',
  NORMAL: 'No pathology detected',
};
const SPOKEN = { CNV: 'CNV', DME: 'DME', DRUSEN: 'drusen', NORMAL: 'a normal scan' };
// A prediction below this confidence, or this close to the next class, is shown as uncertain.
const UNCERTAIN_BELOW = 0.7;
const UNCERTAIN_MARGIN = 0.3;

const state = {
  label: null,
  images: { orig: null, input: null, cam: null },
  view: 'cam',
  messages: [],
  waiting: false,
  // One conversation per page load; MLflow groups its questions under this id.
  sessionId: crypto.randomUUID(),
};

const $ = (id) => document.getElementById(id);
const els = {
  file: $('file'), drop: $('drop'), replace: $('replace'), viewer: $('viewer'),
  img: $('viewer-img'), busy: $('busy'), scanError: $('scan-error'),
  views: document.querySelector('.views'), prediction: $('prediction'),
  predLabel: $('pred-label'), predFull: $('pred-full'), predPct: $('pred-pct'), bars: $('bars'),
  predEmpty: $('pred-empty'), predEmptyTitle: $('pred-empty-title'), predEmptyText: $('pred-empty-text'),
  predResult: $('pred-result'), predBadge: $('pred-badge'), predUncertain: $('pred-uncertain'),
  messages: $('messages'), empty: $('empty'),
  suggestions: $('suggestions'), composer: $('composer'), ask: $('ask'),
  send: document.querySelector('.send'),
};

function pct(p) { return Math.round(p * 100); }

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// Just enough Markdown for the model's answers: headings, paragraphs, bullet and
// numbered lists (with bullets indented under a numbered item kept inside it), bold
// and italics. Works line by line, since the model often puts a heading and its list
// in one block without a blank line between them. With ``cite``, each citation such
// as [2, 4] becomes buttons that open the passage it points at.
function renderMarkdown(text, { cite = false } = {}) {
  // Punctuation right after a citation stays with it, so a line never starts with it.
  const citations = (s) => s.replace(/\[(\d+(?:\s*,\s*\d+)*)\]([.,;:)]?)/g, (_, nums, punct) => '<span class="cites">'
    + nums.split(/\s*,\s*/)
      .map((n) => `<button type="button" class="cite" data-n="${n}" aria-label="Show source ${n}">${n}</button>`)
      .join('') + punct + '</span>');
  const inline = (s) => {
    const html = escapeHtml(s)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*(?!\s)(.+?)\*/g, '$1<em>$2</em>');
    return cite ? citations(html) : html;
  };
  const out = [];
  let para = [];
  let items = [];   // {text, subs}
  let ordered = false;
  let start = 1;
  const flush = () => {
    if (para.length) out.push('<p>' + para.map(inline).join('<br>') + '</p>');
    if (items.length) {
      const li = items.map((it) => '<li>' + inline(it.text)
        + (it.subs.length ? '<ul>' + it.subs.map((s) => '<li>' + inline(s) + '</li>').join('') + '</ul>' : '')
        + '</li>').join('');
      out.push(ordered ? `<ol start="${start}">${li}</ol>` : `<ul>${li}</ul>`);
    }
    para = [];
    items = [];
  };
  for (const line of text.trim().split('\n')) {
    const heading = line.match(/^\s*#{1,6}\s+(.*)$/);
    const number = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    const bullet = line.match(/^(\s*)[-*•]\s+(.*)$/);
    if (!line.trim()) {
      // A blank line between numbered items keeps them in one list.
      if (!(items.length && ordered)) flush();
    } else if (heading) {
      flush();
      out.push('<h4>' + inline(heading[1]) + '</h4>');
    } else if (number) {
      if (para.length || (items.length && !ordered)) flush();
      if (!items.length) { ordered = true; start = Number(number[1]); }
      items.push({ text: number[2], subs: [] });
    } else if (bullet) {
      if (items.length && ordered && bullet[1].length >= 2) {
        items[items.length - 1].subs.push(bullet[2]);
      } else {
        if (para.length || (items.length && ordered)) flush();
        ordered = false;
        items.push({ text: bullet[2], subs: [] });
      }
    } else {
      if (items.length) flush();
      para.push(line);
    }
  }
  flush();
  return out.join('');
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

function showView(view) {
  state.view = view;
  for (const b of els.views.querySelectorAll('button')) {
    b.setAttribute('aria-pressed', String(b.dataset.view === view));
  }
  const src = state.images[view] || state.images.orig;
  if (src) els.img.src = src;
  els.img.alt = {
    orig: 'The OCT B-scan as uploaded',
    input: 'The cropped and resized B-scan the model classified',
    cam: 'Grad-CAM heat map over the model input',
  }[view];
}

// The prediction card stays on screen before and between results, showing a short
// message in place of the class and its probabilities.
function showPredictionMessage(title, text) {
  els.predEmptyTitle.textContent = title;
  els.predEmptyText.textContent = text;
  els.predEmpty.hidden = false;
  els.predResult.hidden = true;
}

function enableViews(on) {
  for (const b of els.views.querySelectorAll('button')) b.disabled = !on;
}

function renderPrediction(res) {
  els.predLabel.textContent = res.label;
  els.predFull.textContent = FULL[res.label];
  els.predPct.textContent = pct(res.confidence) + '%';
  els.bars.replaceChildren(...CLASSES.map((c) => {
    const row = document.createElement('div');
    row.className = 'bar' + (c === res.label ? ' top' : '');
    row.innerHTML = `<span class="bar-name">${c}</span>
      <span class="bar-track"><span class="bar-fill" style="width:0%"></span></span>
      <span class="bar-pct">${pct(res.probs[c])}%</span>`;
    requestAnimationFrame(() => { row.querySelector('.bar-fill').style.width = pct(res.probs[c]) + '%'; });
    return row;
  }));
  // The runner-up class, to say what the model nearly chose.
  const second = CLASSES.filter((c) => c !== res.label)
    .map((c) => [c, res.probs[c]])
    .sort((a, b) => b[1] - a[1])[0];
  const uncertain = res.confidence < UNCERTAIN_BELOW || res.confidence - second[1] < UNCERTAIN_MARGIN;
  els.predBadge.hidden = !uncertain;
  els.predUncertain.hidden = !uncertain;
  els.predUncertain.textContent = uncertain
    ? `Next most likely: ${second[0]} ${pct(second[1])}%. Confirm with clinical examination before relying on it.`
    : '';
  els.predEmpty.hidden = true;
  els.predResult.hidden = false;
}

async function handleFile(file) {
  if (!file) return;
  if (!/^image\/(png|jpeg)$/.test(file.type)) {
    showScanError('Upload a PNG or JPEG B-scan.');
    return;
  }
  els.scanError.hidden = true;
  if (state.images.orig) URL.revokeObjectURL(state.images.orig);
  state.images = { orig: URL.createObjectURL(file), input: null, cam: null };
  state.label = null;

  els.drop.hidden = true;
  els.viewer.hidden = false;
  els.replace.hidden = false;
  enableViews(false);
  showPredictionMessage('Analysing scan…', 'The classifier and Grad-CAM are running on the uploaded B-scan.');
  els.busy.hidden = false;
  els.img.src = state.images.orig;
  renderSuggestions();

  const body = new FormData();
  body.append('file', file);
  body.append('session_id', state.sessionId);
  try {
    const res = await fetch('/api/predict', { method: 'POST', body });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'The scan could not be analysed.');
    const data = await res.json();
    state.label = data.label;
    state.images.input = data.model_input;
    state.images.cam = data.gradcam;
    enableViews(true);
    showView('cam');
    renderPrediction(data);
  } catch (err) {
    showScanError(err.message);
    showView('orig');
    showPredictionMessage('No prediction', 'This file could not be analysed. Upload another OCT B-scan.');
  } finally {
    els.busy.hidden = true;
    renderSuggestions();
  }
}

function showScanError(message) {
  els.scanError.textContent = message;
  els.scanError.hidden = false;
}

els.drop.addEventListener('click', () => els.file.click());
els.replace.addEventListener('click', () => els.file.click());
els.file.addEventListener('change', () => { handleFile(els.file.files[0]); els.file.value = ''; });
els.views.addEventListener('click', (e) => {
  const b = e.target.closest('button[data-view]');
  if (b) showView(b.dataset.view);
});
for (const target of [els.drop, els.viewer]) {
  target.addEventListener('dragover', (e) => { e.preventDefault(); els.drop.classList.add('over'); });
  target.addEventListener('dragleave', () => els.drop.classList.remove('over'));
  target.addEventListener('drop', (e) => {
    e.preventDefault();
    els.drop.classList.remove('over');
    handleFile(e.dataTransfer.files[0]);
  });
}

// ---------------------------------------------------------------------------
// Assistant
// ---------------------------------------------------------------------------

// General questions only, since the assistant does not see the scan. After a
// prediction they turn to the predicted condition in general terms.
function renderSuggestions() {
  let qs;
  if (state.label === 'NORMAL') {
    qs = ['What does a healthy macula look like on OCT?', 'What are the earliest OCT signs of AMD?'];
  } else if (state.label) {
    const name = SPOKEN[state.label];
    qs = [`What are the signs of ${name}?`, `How is ${name} usually managed?`];
  } else {
    qs = ['How is DME distinguished from CNV on OCT?', 'What are the treatment options for neovascular AMD?'];
  }
  els.suggestions.replaceChildren(...qs.map((q) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = q;
    b.addEventListener('click', () => ask(q));
    return b;
  }));
}

// The sources sit in one collapsed dropdown under the answer, so a long list does
// not push the conversation apart. Its summary line says how many passages, from how
// many articles, and the best relevance among them; each passage opens on its own.
function sourcesBlock(sources) {
  if (!sources.length) {
    const none = document.createElement('div');
    none.className = 'sources-label';
    none.textContent = 'No literature search on this turn.';
    return none;
  }
  const wrap = document.createElement('details');
  wrap.className = 'sources';
  const articles = new Set(sources.map((s) => s.title)).size;
  const best = Math.round(Math.max(...sources.map((s) => s.score)) * 100);
  wrap.innerHTML = `<summary class="sources-summary">
      <svg class="chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>
      <span>Sources · ${sources.length} passage${sources.length === 1 ? '' : 's'} from ${articles} article${articles === 1 ? '' : 's'}</span>
      <span class="sources-best">best match ${best}%</span>
    </summary>
    <div class="sources-list"></div>`;
  const list = wrap.querySelector('.sources-list');
  sources.forEach((s, i) => {
    const d = document.createElement('details');
    d.className = 'source';
    const title = s.section ? `${s.title} - ${s.section}` : s.title;
    const meta = [s.doi ? `doi:${s.doi}` : null, s.license].filter(Boolean);
    // The reranker's score, a 0-1 relevance of the passage to the search query.
    const rel = Math.round(s.score * 100);
    const level = s.score >= 0.8 ? 'high' : s.score >= 0.5 ? 'mid' : 'low';
    d.innerHTML = `<summary><span class="n">[${i + 1}]</span><span class="source-title"></span>
      <span class="rel ${level}" title="Reranker relevance of this passage to the search query">
        <span class="rel-track"><span class="rel-fill" style="width:${rel}%"></span></span>${rel}%</span></summary>
      <div class="source-body"><div class="source-meta"></div><div class="source-text"></div></div>`;
    d.querySelector('.source-title').textContent = title;
    d.querySelector('.source-meta').textContent = meta.join(' · ');
    d.querySelector('.source-text').textContent = s.text;
    list.append(d);
  });
  return wrap;
}

function addMessage(role, content, sources) {
  els.empty.hidden = true;
  let node;
  if (role === 'user') {
    node = document.createElement('div');
    node.className = 'msg-user';
    node.textContent = content;
  } else {
    node = document.createElement('div');
    node.className = 'msg-bot';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = renderMarkdown(content, { cite: Boolean(sources && sources.length) });
    node.append(bubble);
    if (sources) node.append(sourcesBlock(sources));
  }
  els.messages.append(node);
  els.messages.scrollTop = els.messages.scrollHeight;
  return node;
}

function setWaiting(on) {
  state.waiting = on;
  els.send.disabled = on;
  for (const b of els.suggestions.querySelectorAll('button')) b.disabled = on;
}

// Read a newline-delimited JSON response one event at a time, as it arrives.
async function* readEvents(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line) yield JSON.parse(line);
    }
  }
  if (buffer.trim()) yield JSON.parse(buffer);
}

// The bubble shown while the assistant works: each step the server reports, the
// current one with a spinner and the finished ones ticked, plus the time elapsed.
function progressBubble() {
  const node = document.createElement('div');
  node.className = 'msg-bot';
  node.innerHTML = `<div class="bubble progress">
      <ol class="steps"></ol>
      <div class="elapsed" aria-hidden="true">0 s</div>
    </div>`;
  const list = node.querySelector('.steps');
  const elapsed = node.querySelector('.elapsed');
  const started = Date.now();
  const timer = setInterval(() => {
    elapsed.textContent = `${Math.floor((Date.now() - started) / 1000)} s`;
  }, 1000);
  const step = (text) => {
    const prev = list.querySelector('li.active');
    if (prev) prev.className = 'done';
    const li = document.createElement('li');
    li.className = 'active';
    li.textContent = text;
    list.append(li);
    els.messages.scrollTop = els.messages.scrollHeight;
  };
  step('Sending the question');
  return {
    node,
    step,
    done() { clearInterval(timer); node.remove(); },  // safe to call more than once
  };
}

async function ask(text) {
  text = text.trim();
  if (!text || state.waiting) return;
  state.messages.push({ role: 'user', content: text });
  addMessage('user', text);
  els.ask.value = '';
  setWaiting(true);

  let progress = progressBubble();
  els.messages.append(progress.node);
  els.messages.scrollTop = els.messages.scrollHeight;
  // The answer as it is being written, shown in place of the step list. The model
  // sends its text in a few large pieces, so the page reveals it a few words per
  // frame, catching up faster the further behind it is.
  let draft = null;
  let draftText = '';
  let shown = 0;
  let queued = false;
  const paint = () => {
    const behind = draftText.length - shown;
    if (!draft || behind <= 0) { queued = false; return; }
    const target = shown + Math.max(3, Math.ceil(behind / 25));
    const wordEnd = draftText.indexOf(' ', target);
    shown = wordEnd === -1 ? draftText.length : wordEnd;
    draft.innerHTML = renderMarkdown(draftText.slice(0, shown));
    els.messages.scrollTop = els.messages.scrollHeight;
    requestAnimationFrame(paint);
  };
  const revealed = () => new Promise((resolve) => {
    const check = () => (!draft || shown >= draftText.length ? resolve() : requestAnimationFrame(check));
    check();
  });

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: state.messages, session_id: state.sessionId }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'The assistant could not answer.');
    let data = null;
    for await (const event of readEvents(res)) {
      if (event.type === 'status') {
        // A new step after some text means that text was not the answer after all
        // (the model went on to call a tool), so the draft gives way to the steps.
        if (draft) {
          draft.parentElement.remove();
          draft = null;
          draftText = '';
          shown = 0;
          progress = progressBubble();
          els.messages.append(progress.node);
        }
        progress.step(event.text);
      } else if (event.type === 'text') {
        if (!draft) {
          progress.done();
          const node = document.createElement('div');
          node.className = 'msg-bot';
          node.innerHTML = '<div class="bubble streaming"></div>';
          els.messages.append(node);
          draft = node.firstChild;
        }
        draftText += event.text;
        if (!queued) { queued = true; requestAnimationFrame(paint); }
      } else if (event.type === 'error') {
        throw new Error(event.detail);
      } else if (event.type === 'answer') {
        data = event;
      }
    }
    if (!data) throw new Error('The assistant stopped before answering.');
    await revealed();
    progress.done();
    if (draft) draft.parentElement.remove();
    state.messages.push({ role: 'assistant', content: data.reply });
    addMessage('assistant', data.reply, data.sources);
  } catch (err) {
    progress.done();
    if (draft) draft.parentElement.remove();
    // The failed question stays out of the history, so a retry sends a clean conversation.
    state.messages.pop();
    const node = addMessage('assistant', `${err.message} Please try again.`);
    node.classList.add('failed');
  } finally {
    setWaiting(false);
    els.ask.focus();
  }
}

els.composer.addEventListener('submit', (e) => { e.preventDefault(); ask(els.ask.value); });

// A citation opens the answer's sources and the passage it points at.
els.messages.addEventListener('click', (e) => {
  const button = e.target.closest('.cite');
  if (!button) return;
  const sources = button.closest('.msg-bot').querySelector('details.sources');
  const passage = sources && sources.querySelectorAll('details.source')[Number(button.dataset.n) - 1];
  if (!passage) return;
  sources.open = true;
  passage.open = true;
  passage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  passage.classList.remove('flash');
  void passage.offsetWidth;  // restart the highlight if the same passage is clicked again
  passage.classList.add('flash');
});

renderSuggestions();
