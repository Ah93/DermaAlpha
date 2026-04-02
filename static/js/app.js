/* ═══════════════════════════════════════════════════════════════════════════
   DermaAlpha — Frontend Application
   Vanilla JS single-page app. No frameworks.
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Application State ─────────────────────────────────────────────────────────
const AppState = {
  currentScreen: 1,
  patientId: null,
  isNewPatient: false,
  isFirstEncounter: true,
  selectedImageFile: null,
  lastAnalysis: null,
  lastEncounterId: null,
  historyLoaded: false,
  historyOpen: false,
  trendChart: null,   // Chart.js instance — destroyed on re-render
};

// ── Utility Helpers ───────────────────────────────────────────────────────────

function $(id) { return document.getElementById(id); }
function truncate(str, n) { return str.length > n ? str.slice(0, n) + '…' : str; }
function formatDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso + 'T00:00:00');
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
}
function getSeverityClass(score) {
  if (score <= 3) return 'low';
  if (score <= 6) return 'moderate';
  if (score <= 8) return 'high';
  return 'urgent';
}
function getSeverityLabel(score) {
  if (score <= 3) return 'Mild';
  if (score <= 6) return 'Moderate';
  if (score <= 8) return 'Significant';
  return 'Urgent';
}
function getSeverityColor(score) {
  const map = { low: '#10b981', moderate: '#f59e0b', high: '#ef4444', urgent: '#dc2626' };
  return map[getSeverityClass(score)];
}

// ── Screen Navigation ─────────────────────────────────────────────────────────

function showScreen(n) {
  const current = document.querySelector('.screen.active');
  const next = $('screen-' + n);
  if (!next) return;

  if (current) {
    current.classList.remove('visible');
    setTimeout(() => { current.classList.remove('active'); }, 300);
  }

  AppState.currentScreen = n;
  next.classList.add('active');
  // Trigger transition on next frame
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { next.classList.add('visible'); });
  });

  window.scrollTo(0, 0);
}

// ── Toast Notifications ───────────────────────────────────────────────────────

function showToast(message, type = 'info') {
  const icons = { error: '⚠', success: '✓', info: 'ℹ' };
  const container = $('toast-container');

  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${escapeHtml(message)}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('toast--fade-out');
    setTimeout(() => toast.remove(), 300);
  }, 5000);
}

function escapeHtml(str) {
  const d = document.createElement('div');
  d.appendChild(document.createTextNode(str));
  return d.innerHTML;
}

// ── Loading Button State ──────────────────────────────────────────────────────

function setLoading(btn, isLoading, label = null) {
  if (isLoading) {
    btn.disabled = true;
    btn._origText = btn.innerHTML;
    btn.innerHTML = `<span class="spinner"></span>${label || 'Processing…'}`;
  } else {
    btn.disabled = false;
    if (btn._origText) btn.innerHTML = btn._origText;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SCREEN 1 — Welcome / Patient ID
// ─────────────────────────────────────────────────────────────────────────────

function initScreen1() {
  const returningOpt = $('opt-returning');
  const newOpt       = $('opt-new');
  const idWrapper    = $('id-input-wrapper');

  function selectRadio(opt) {
    returningOpt.classList.remove('selected');
    newOpt.classList.remove('selected');
    opt.classList.add('selected');
    idWrapper.style.display = (opt === returningOpt) ? 'flex' : 'none';
    if (opt === newOpt) $('patient-id-input').value = '';
  }

  returningOpt.addEventListener('click', () => selectRadio(returningOpt));
  newOpt.addEventListener('click',       () => selectRadio(newOpt));

  // Auto-uppercase patient ID
  $('patient-id-input').addEventListener('input', e => {
    const pos = e.target.selectionStart;
    e.target.value = e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, '');
    e.target.setSelectionRange(pos, pos);
  });

  $('patient-id-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') handleBeginBtn();
  });

  $('begin-btn').addEventListener('click', handleBeginBtn);

  // Default: new patient selected
  selectRadio(newOpt);
}

async function handleBeginBtn() {
  const isNew = $('opt-new').classList.contains('selected');
  const rawId = isNew ? 'new' : $('patient-id-input').value.trim();

  if (!isNew && !rawId) {
    showToast('Please enter your Patient ID or select "New Patient"', 'error');
    return;
  }

  const btn = $('begin-btn');
  setLoading(btn, true, 'Connecting…');

  try {
    const res = await fetch('/api/check-patient', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_id: rawId }),
    });

    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();

    // Reject returning patients whose ID doesn't exist in the database
    if (!isNew && !data.exists) {
      showToast('Patient ID not found. Please check your ID and try again.', 'error');
      setLoading(btn, false);
      return;
    }

    AppState.patientId        = data.patient_id;
    AppState.isNewPatient     = data.is_new;
    AppState.isFirstEncounter = !data.exists;

    // Update header badge
    const badge = $('header-patient-badge');
    badge.textContent = data.patient_id;
    badge.classList.add('visible');

    // Show generated ID to new patients
    if (data.is_new && !data.exists) {
      const idBox = $('generated-id-box');
      const idVal = $('generated-id-value');
      idVal.textContent = data.patient_id;
      idBox.style.display = 'block';

      // Give user time to note the ID
      await new Promise(r => setTimeout(r, 800));
    }

    showScreen(2);
    updateConsultScreenForPatient();

  } catch (err) {
    showToast('Could not connect: ' + err.message, 'error');
  } finally {
    setLoading(btn, false);
  }
}

function updateConsultScreenForPatient() {
  const subtitle    = $('consult-subtitle');
  const historyWrap = $('history-accordion-wrapper');

  if (AppState.isFirstEncounter) {
    subtitle.textContent = 'New Patient — Describe your condition and complete the intake below for a more accurate assessment.';
    historyWrap.style.display = 'block';
  } else {
    subtitle.textContent = `Welcome back, ${AppState.patientId} — describe your current condition and we will compare it with your previous visits.`;
    historyWrap.style.display = 'none';
    $('medical-history-textarea').value = '';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCTURED INTAKE — chip interactions + data builder
// ─────────────────────────────────────────────────────────────────────────────

function initIntakeChips() {
  // Toggle .selected class on chip labels when inputs change
  document.querySelectorAll('.intake-chip input').forEach(input => {
    input.addEventListener('change', () => {
      if (input.type === 'radio') {
        document.querySelectorAll(`input[name="${input.name}"]`).forEach(sib => {
          sib.closest('.intake-chip').classList.remove('selected');
        });
      }
      input.closest('.intake-chip').classList.toggle('selected', input.checked);
    });
  });

  // Fitzpatrick selector
  document.querySelectorAll('.fitz-btn input').forEach(input => {
    input.addEventListener('change', () => {
      document.querySelectorAll('.fitz-btn').forEach(btn => btn.classList.remove('selected'));
      if (input.checked) input.closest('.fitz-btn').classList.add('selected');
    });
  });

  // Show/hide treatment detail text box
  document.querySelectorAll('input[name="treated"]').forEach(input => {
    input.addEventListener('change', () => {
      const wrapper = $('treatment-detail-wrapper');
      if (wrapper) wrapper.style.display = input.value === 'Yes' ? 'block' : 'none';
    });
  });
}

function buildStructuredMedHistory() {
  const lines = [];

  const radio  = (name) => document.querySelector(`input[name="${name}"]:checked`);
  const checks = (name) => [...document.querySelectorAll(`input[name="${name}"]:checked`)].map(i => i.value);
  const text   = (id)   => { const el = $(id); return el ? el.value.trim() : ''; };

  const duration = radio('duration');
  if (duration) lines.push(`Duration: ${duration.value}`);

  const symptoms = checks('symptom');
  if (symptoms.length) lines.push(`Symptoms: ${symptoms.join(', ')}`);

  const spreading = radio('spreading');
  if (spreading) lines.push(`Spreading: ${spreading.value}`);

  const triggers = checks('trigger');
  if (triggers.length) lines.push(`Possible triggers: ${triggers.join(', ')}`);

  const fitz = radio('fitzpatrick');
  if (fitz) lines.push(`Fitzpatrick skin type: ${fitz.value}`);

  const meds = text('current-meds-input');
  if (meds) lines.push(`Current medications: ${meds}`);

  const treated = radio('treated');
  if (treated) {
    let t = `Prior treatment tried: ${treated.value}`;
    if (treated.value === 'Yes') {
      const detail = text('treatment-detail-input');
      if (detail) t += ` — ${detail}`;
    }
    lines.push(t);
  }

  const knownCond = text('known-conditions-input');
  if (knownCond) lines.push(`Known skin conditions: ${knownCond}`);

  const allergies = text('allergies-input');
  if (allergies) lines.push(`Allergies: ${allergies}`);

  const famHist = radio('family-history');
  if (famHist) lines.push(`Family history of skin cancer / autoimmune: ${famHist.value}`);

  return lines.join('\n');
}

function resetIntake() {
  document.querySelectorAll('.intake-chip input, .fitz-btn input').forEach(input => {
    input.checked = false;
    const parent = input.closest('.intake-chip, .fitz-btn');
    if (parent) parent.classList.remove('selected');
  });
  ['current-meds-input', 'treatment-detail-input', 'known-conditions-input', 'allergies-input'].forEach(id => {
    const el = $(id);
    if (el) el.value = '';
  });
  const wrapper = $('treatment-detail-wrapper');
  if (wrapper) wrapper.style.display = 'none';
}

// ─────────────────────────────────────────────────────────────────────────────
// SCREEN 2 — Consultation Form
// ─────────────────────────────────────────────────────────────────────────────

function initScreen2() {
  initDropZone();
  initIntakeChips();

  $('complaint-textarea').addEventListener('input', () => {
    const len = $('complaint-textarea').value.length;
    $('complaint-char-count').textContent = `${len} / 1000`;
  });

  $('analyze-btn').addEventListener('click', handleAnalyzeBtn);
  $('back-btn').addEventListener('click', () => showScreen(1));
}

// Drop Zone
function initDropZone() {
  const zone   = $('drop-zone');
  const input  = $('image-input');

  zone.addEventListener('click', () => { if (!AppState.selectedImageFile) input.click(); });
  input.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

  ['dragenter', 'dragover'].forEach(ev => {
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('dragover'); });
  });
  ['dragleave', 'dragend'].forEach(ev => {
    zone.addEventListener(ev, () => zone.classList.remove('dragover'));
  });
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  $('image-remove-btn').addEventListener('click', e => { e.stopPropagation(); clearImage(); });
}

function handleFile(file) {
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/gif'];
  if (!allowedTypes.includes(file.type)) {
    showToast('Unsupported file type. Please use JPG, PNG, WEBP, or BMP.', 'error');
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    showToast('File is too large. Maximum size is 15 MB.', 'error');
    return;
  }

  AppState.selectedImageFile = file;
  const url = URL.createObjectURL(file);

  $('drop-zone-content').style.display = 'none';
  $('image-preview-wrapper').style.display = 'flex';
  $('image-preview').src = url;
  $('image-filename').textContent = truncate(file.name, 40);
  $('drop-zone').classList.add('has-file');
}

function clearImage() {
  AppState.selectedImageFile = null;
  const preview = $('image-preview');
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = '';
  $('image-input').value = '';
  $('drop-zone-content').style.display = 'block';
  $('image-preview-wrapper').style.display = 'none';
  $('drop-zone').classList.remove('has-file');
}

async function handleAnalyzeBtn() {
  const complaint = $('complaint-textarea').value.trim();
  if (!complaint) {
    showToast('Please describe your skin condition before continuing.', 'error');
    $('complaint-textarea').focus();
    return;
  }
  if (complaint.length < 10) {
    showToast('Please provide more detail about your condition.', 'error');
    return;
  }
  await submitAnalysis();
}

// ─────────────────────────────────────────────────────────────────────────────
// SCREEN 3 — Loading
// ─────────────────────────────────────────────────────────────────────────────

const LOADING_STEPS = [
  { id: 'step-0', label: 'Uploading image & data' },
  { id: 'step-1', label: 'Consulting Gemini AI' },
  { id: 'step-2', label: 'Generating treatment plan' },
  { id: 'step-3', label: 'Saving to secure memory' },
];

let _stepTimers = [];

function startLoadingAnimation() {
  _stepTimers.forEach(clearTimeout);
  _stepTimers = [];
  LOADING_STEPS.forEach(s => {
    const el = $(s.id);
    el.classList.remove('done', 'active');
    el.querySelector('.step-icon').innerHTML = '';
  });
}

function setStepActive(i) {
  if (i >= LOADING_STEPS.length) return;
  const el = $(LOADING_STEPS[i].id);
  el.classList.add('active');
  el.querySelector('.step-icon').innerHTML = '<span class="spin-small"></span>';
}

function setStepDone(i) {
  if (i >= LOADING_STEPS.length) return;
  const el = $(LOADING_STEPS[i].id);
  el.classList.remove('active');
  el.classList.add('done');
  el.querySelector('.step-icon').textContent = '✓';
}

function simulateLoadingProgress(totalMs) {
  const ticks = [0, totalMs * 0.05, totalMs * 0.6, totalMs * 0.85];
  startLoadingAnimation();
  setStepActive(0);
  ticks.forEach((delay, i) => {
    if (i === 0) return; // step 0 starts immediately
    _stepTimers.push(setTimeout(() => { setStepDone(i - 1); setStepActive(i); }, delay));
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// API: Submit Analysis
// ─────────────────────────────────────────────────────────────────────────────

async function submitAnalysis() {
  showScreen(3);
  simulateLoadingProgress(15000); // Gemini API is fast (~5-15s)

  // For new patients, build structured history from intake form
  const medHistory = AppState.isFirstEncounter
    ? buildStructuredMedHistory()
    : $('medical-history-textarea').value.trim();

  const formData = new FormData();
  formData.append('patient_id',      AppState.patientId);
  formData.append('complaint',       $('complaint-textarea').value.trim());
  formData.append('medical_history', medHistory);
  if (AppState.selectedImageFile) {
    formData.append('image', AppState.selectedImageFile);
  }

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: formData });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || err.error || 'Analysis failed');
    }

    const data = await res.json();
    AppState.lastAnalysis = data;
    AppState.isFirstEncounter = data.is_first_encounter;

    // Finish loading steps
    _stepTimers.forEach(clearTimeout);
    setStepDone(0); setStepDone(1); setStepDone(2); setStepDone(3);

    await new Promise(r => setTimeout(r, 500)); // brief pause to show completion

    populateResults(data);
    showScreen(4);
    AppState.historyLoaded = false; // reset so history reloads for next open

  } catch (err) {
    _stepTimers.forEach(clearTimeout);
    showToast('Analysis error: ' + err.message, 'error');
    showScreen(2);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SCREEN 4 — Results Population
// ─────────────────────────────────────────────────────────────────────────────

function populateResults(data) {
  // Store encounter/patient IDs for download button
  AppState.lastEncounterId = data.encounter_id;

  // Encounter info bar
  $('enc-id-display').textContent   = data.encounter_id;
  $('enc-date-display').textContent = formatDate(data.encounter_date);
  $('enc-type-display').textContent = data.is_first_encounter ? 'First Visit' : 'Follow-up';

  // Urgent banner
  renderUrgentBanner(data.urgent);

  // Diagnosis
  $('diagnosis-title').textContent = data.diagnosis || 'See full assessment';

  // Differential
  renderDifferential(data.differential);

  // Severity gauge
  updateSeverityGauge(data.severity_score);
  $('gauge-label-text').textContent  = `${data.severity_score.toFixed(1)} / 10`;
  $('gauge-severity-level').textContent = getSeverityLabel(data.severity_score);

  // Classifier predictions (new)
  renderClassifierPredictions(data.lesion_classification);

  // Narrative
  renderNarrative($('narrative-text'), data.narrative);

  // Condition tags
  renderTags(data.condition_tags, data.body_region);

  // Medications (new)
  renderMedications(data.medications);

  // Treatment plan
  renderTreatmentPlan(data.treatment_plan);

  // Progress section
  renderProgressSection(data);

  // Severity trend chart (new)
  renderTrendChart(data.condition_chain);

  // Download report button (new)
  const dlBtn = $('download-report-btn');
  if (dlBtn) {
    dlBtn.style.display = 'inline-flex';
    dlBtn.onclick = () => {
      const url = `/api/report/${AppState.patientId}/${data.encounter_id}`;
      window.open(url, '_blank');
    };
  }
}

function renderUrgentBanner(urgent) {
  const banner = $('urgent-banner');
  if (urgent) {
    banner.classList.add('visible');
  } else {
    banner.classList.remove('visible');
  }
}

function renderNarrative(container, rawText) {
  container.innerHTML = '';
  const clean = (rawText || '').split('```')[0].trim();
  const paragraphs = clean.split(/\n{2,}/).filter(p => p.trim());
  paragraphs.forEach(para => {
    const p = document.createElement('p');
    p.textContent = para.trim();
    container.appendChild(p);
  });
  if (!paragraphs.length) {
    const p = document.createElement('p');
    p.textContent = 'Analysis complete. See clinical summary for details.';
    container.appendChild(p);
  }
}

function renderDifferential(list) {
  const ul = $('differential-list');
  ul.innerHTML = '';
  const items = Array.isArray(list) ? list.slice(0, 4) : [];
  if (!items.length) {
    const li = document.createElement('li');
    li.textContent = 'No alternative diagnoses noted';
    li.style.color = 'var(--text-muted)';
    ul.appendChild(li);
    return;
  }
  items.forEach(item => {
    const li = document.createElement('li');
    li.textContent = item;
    ul.appendChild(li);
  });
}

function renderTags(tags, bodyRegion) {
  const container = $('condition-tags');
  container.innerHTML = '';

  if (bodyRegion && bodyRegion !== 'unspecified') {
    const span = document.createElement('span');
    span.className = 'tag tag--region';
    span.textContent = bodyRegion.replace(/_/g, ' ');
    container.appendChild(span);
  }

  const tagList = Array.isArray(tags) ? tags : [];
  tagList.forEach(t => {
    const span = document.createElement('span');
    span.className = 'tag';
    span.textContent = t.replace(/_/g, ' ');
    container.appendChild(span);
  });

  if (!tagList.length && !bodyRegion) {
    const span = document.createElement('span');
    span.className = 'tag';
    span.textContent = 'skin condition';
    container.appendChild(span);
  }
}

function renderTreatmentPlan(text) {
  const container = $('treatment-text');
  container.innerHTML = '';
  if (!text) {
    const p = document.createElement('p');
    p.textContent = 'Please consult with a dermatologist for a personalised treatment plan.';
    p.className = 'text-muted';
    container.appendChild(p);
    return;
  }

  // Detect numbered list
  const numberedRegex = /^\d+[\.\)]/m;
  if (numberedRegex.test(text)) {
    const lines = text.split('\n');
    const ol = document.createElement('ol');
    lines.forEach(line => {
      const clean = line.replace(/^\d+[\.\)]\s*/, '').trim();
      if (clean) {
        const li = document.createElement('li');
        li.textContent = clean;
        ol.appendChild(li);
      }
    });
    if (ol.children.length) { container.appendChild(ol); return; }
  }

  // Plain paragraphs
  text.split(/\n{2,}/).filter(p => p.trim()).forEach(para => {
    const p = document.createElement('p');
    p.textContent = para.trim();
    container.appendChild(p);
  });
}

function renderProgressSection(data) {
  const section = $('progress-section');
  if (data.is_first_encounter) {
    section.classList.remove('visible');
    return;
  }
  section.classList.add('visible');

  // Badge
  const badge = $('progress-badge');
  badge.className = `badge badge--${data.progress || 'stable'}`;
  const icons = { improving: '↑', worsening: '↓', stable: '→' };
  badge.textContent = (icons[data.progress] || '→') + ' ' + (data.progress || 'stable').toUpperCase();

  // Severity delta
  if (data.previous_severity != null) {
    const prevVal = $('severity-prev');
    const currVal = $('severity-curr');
    prevVal.textContent = parseFloat(data.previous_severity).toFixed(1);
    currVal.textContent = parseFloat(data.severity_score).toFixed(1);
    prevVal.className = 'sev-value';
    currVal.className = 'sev-value ' + (data.progress || 'stable');
    $('severity-delta-container').style.display = 'flex';
  } else {
    $('severity-delta-container').style.display = 'none';
  }

  // Progress narrative
  renderNarrative($('progress-narrative'), data.progress_note);

  // Visual similarity panel
  renderVisualSimilarity(data.visual_similarity);
}

// ── Visual Similarity Panel ───────────────────────────────────────────────────

function renderVisualSimilarity(visualSimilarity) {
  const panel = $('visual-similarity-panel');
  if (!panel) return;

  if (!visualSimilarity || visualSimilarity.score == null) {
    panel.style.display = 'none';
    return;
  }

  panel.style.display = 'block';

  const score = visualSimilarity.score;           // 0.0 → 1.0
  const change = visualSimilarity.change;         // "LOW" | "MODERATE" | "HIGH"
  const contradiction = visualSimilarity.contradiction;

  // Score display
  $('visual-sim-score').textContent = `${(score * 100).toFixed(1)}%`;

  // Progress bar (similarity = bar fill)
  const bar = $('visual-sim-bar');
  bar.style.width = `${(score * 100).toFixed(1)}%`;

  // Color: high similarity = green, low = red
  const barColor = score >= 0.80 ? '#10b981'
                 : score >= 0.55 ? '#f59e0b'
                 : '#ef4444';
  bar.style.background = barColor;

  // Change label badge
  const changeTag = $('visual-sim-change-label');
  const changeMap = {
    LOW:      { text: 'LOW CHANGE',      cls: 'visual-change--low'      },
    MODERATE: { text: 'MODERATE CHANGE', cls: 'visual-change--moderate' },
    HIGH:     { text: 'HIGH CHANGE',     cls: 'visual-change--high'     },
  };
  const cm = changeMap[change] || { text: change, cls: '' };
  changeTag.textContent = cm.text;
  changeTag.className = `visual-sim-change-tag ${cm.cls}`;

  // Contradiction box
  const contBox = $('visual-contradiction-box');
  if (contradiction) {
    contBox.style.display = 'block';
    contBox.textContent = '⚠ ' + contradiction;
  } else {
    contBox.style.display = 'none';
  }
}

// ── Classifier Predictions ────────────────────────────────────────────────────

function renderClassifierPredictions(lc) {
  const card = $('classifier-card');
  const container = $('classifier-predictions');
  if (!card || !container) return;

  if (!lc || !lc.predictions || !lc.predictions.length) {
    card.style.display = 'none';
    return;
  }

  card.style.display = 'block';
  container.innerHTML = '';

  lc.predictions.forEach((pred, i) => {
    const bar = document.createElement('div');
    bar.className = 'classifier-bar';

    const row = document.createElement('div');
    row.className = 'classifier-bar-row';

    const label = document.createElement('span');
    label.className = 'classifier-label' + (i === 0 ? ' classifier-label--top' : '');
    label.textContent = pred.label;

    const score = document.createElement('span');
    score.className = 'classifier-score';
    score.textContent = `${pred.score}%`;

    row.appendChild(label);
    row.appendChild(score);

    const track = document.createElement('div');
    track.className = 'classifier-track';
    const fill = document.createElement('div');
    fill.className = 'classifier-fill' + (i === 0 ? ' classifier-fill--top' : '');
    fill.style.width = `${Math.min(pred.score, 100)}%`;
    track.appendChild(fill);

    bar.appendChild(row);
    bar.appendChild(track);
    container.appendChild(bar);
  });
}

// ── Medications ───────────────────────────────────────────────────────────────

function renderMedications(medications) {
  const card = $('medications-card');
  const container = $('medications-tags');
  if (!card || !container) return;

  const list = Array.isArray(medications) ? medications : [];
  if (!list.length) {
    card.style.display = 'none';
    return;
  }

  card.style.display = 'block';
  container.innerHTML = '';

  list.forEach(med => {
    const span = document.createElement('span');
    span.className = 'tag tag--drug';
    span.textContent = med.replace(/_/g, ' ');
    container.appendChild(span);
  });
}

// ── Severity Trend Chart ──────────────────────────────────────────────────────

function renderTrendChart(conditionChain) {
  const section = $('trend-chart-section');
  const canvas  = $('trend-chart');
  if (!section || !canvas) return;

  // Destroy previous chart instance before re-rendering
  if (AppState.trendChart) {
    AppState.trendChart.destroy();
    AppState.trendChart = null;
  }

  const chain = Array.isArray(conditionChain) ? conditionChain : [];
  if (chain.length < 2) {
    section.style.display = 'none';
    return;
  }

  section.style.display = 'block';

  const labels   = chain.map(p => p.date ? p.date : '');
  const scores   = chain.map(p => parseFloat(p.severity || 5));
  const pointColors = scores.map(s => getSeverityColor(s));

  AppState.trendChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Severity',
        data: scores,
        borderColor: '#0ea5e9',
        backgroundColor: 'rgba(14,165,233,0.08)',
        pointBackgroundColor: pointColors,
        pointBorderColor: pointColors,
        pointRadius: 6,
        pointHoverRadius: 8,
        borderWidth: 2,
        fill: true,
        tension: 0.35,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0c1a2e',
          borderColor: '#1e3a5f',
          borderWidth: 1,
          titleColor: '#7dd3fc',
          bodyColor: '#e2e8f0',
          callbacks: {
            title: ctx => `Encounter: ${ctx[0].label}`,
            label: ctx => `Severity: ${ctx.parsed.y.toFixed(1)} / 10 — ${getSeverityLabel(ctx.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: '#1e3a5f' },
          ticks: { color: '#64748b', font: { size: 11 } },
        },
        y: {
          min: 0, max: 10,
          grid: { color: '#1e3a5f' },
          ticks: {
            color: '#64748b',
            font: { size: 11 },
            callback: v => `${v}`,
            stepSize: 2,
          },
        },
      },
    },
  });
}

// ── Severity Gauge SVG ────────────────────────────────────────────────────────

function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(cx, cy, r, startAngle, endAngle) {
  const s = polarToCartesian(cx, cy, r, startAngle);
  const e = polarToCartesian(cx, cy, r, endAngle);
  const large = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
}

function updateSeverityGauge(score) {
  const gauge = $('severity-gauge');
  if (!gauge) return;

  // Arc geometry — cx/cy is the visual centre of the arc hole
  const cx = 100, cy = 100, r = 72;
  const startAngle = 135;  // lower-left endpoint
  const endAngle   = 405;  // lower-right endpoint (= 45°, 270° sweep)
  const fillEnd    = startAngle + ((score - 1) / 9) * (endAngle - startAngle);
  const color      = getSeverityColor(score);

  function svgEl(tag) {
    return document.createElementNS('http://www.w3.org/2000/svg', tag);
  }

  // ── Track (full grey arc) ────────────────────────────────────────────────
  let trackPath = gauge.querySelector('.gauge-track');
  if (!trackPath) {
    trackPath = svgEl('path');
    trackPath.setAttribute('class', 'gauge-track');
    trackPath.setAttribute('fill', 'none');
    trackPath.setAttribute('stroke', '#1e3a5f');
    trackPath.setAttribute('stroke-width', '12');
    trackPath.setAttribute('stroke-linecap', 'round');
    gauge.insertBefore(trackPath, gauge.firstChild);
  }
  trackPath.setAttribute('d', describeArc(cx, cy, r, startAngle, endAngle));

  // ── Coloured fill arc ────────────────────────────────────────────────────
  let fillPath = gauge.querySelector('.gauge-fill');
  if (!fillPath) {
    fillPath = svgEl('path');
    fillPath.setAttribute('class', 'gauge-fill');
    fillPath.setAttribute('fill', 'none');
    fillPath.setAttribute('stroke-width', '12');
    fillPath.setAttribute('stroke-linecap', 'round');
    gauge.appendChild(fillPath);
  }
  fillPath.setAttribute('d', describeArc(cx, cy, r, startAngle, Math.max(startAngle + 0.01, fillEnd)));
  fillPath.setAttribute('stroke', color);

  // ── Needle dot ───────────────────────────────────────────────────────────
  const needlePt = polarToCartesian(cx, cy, r, fillEnd);
  let needle = gauge.querySelector('.gauge-needle');
  if (!needle) {
    needle = svgEl('circle');
    needle.setAttribute('class', 'gauge-needle');
    needle.setAttribute('r', '7');
    needle.setAttribute('stroke', '#060d1a');
    needle.setAttribute('stroke-width', '2');
    gauge.appendChild(needle);
  }
  needle.setAttribute('cx', needlePt.x.toFixed(2));
  needle.setAttribute('cy', needlePt.y.toFixed(2));
  needle.setAttribute('fill', color);

  // ── Score number (large, centred inside the arc) ─────────────────────────
  // Vertical centre of the arc interior sits above cy; y=88 keeps it inside.
  let scoreText = gauge.querySelector('.gauge-score');
  if (!scoreText) {
    scoreText = svgEl('text');
    scoreText.setAttribute('class', 'gauge-score');
    scoreText.setAttribute('x', cx);
    scoreText.setAttribute('text-anchor', 'middle');
    scoreText.setAttribute('dominant-baseline', 'middle');
    scoreText.setAttribute('font-family', "'JetBrains Mono', monospace");
    scoreText.setAttribute('font-size', '26');
    scoreText.setAttribute('font-weight', '700');
    gauge.appendChild(scoreText);
  }
  scoreText.setAttribute('y', cy - 8);   // sits comfortably in arc interior
  scoreText.setAttribute('fill', color);
  scoreText.textContent = score.toFixed(1) + ' / 10';

  // ── Severity label ("Mild", "Moderate" …) below the score ────────────────
  let labelText = gauge.querySelector('.gauge-label-inner');
  if (!labelText) {
    labelText = svgEl('text');
    labelText.setAttribute('class', 'gauge-label-inner');
    labelText.setAttribute('x', cx);
    labelText.setAttribute('text-anchor', 'middle');
    labelText.setAttribute('dominant-baseline', 'middle');
    labelText.setAttribute('font-family', "'Inter', sans-serif");
    labelText.setAttribute('font-size', '13');
    labelText.setAttribute('font-weight', '600');
    gauge.appendChild(labelText);
  }
  labelText.setAttribute('y', cy + 16);  // below the score, still inside arc
  labelText.setAttribute('fill', color);
  labelText.textContent = getSeverityLabel(score);
}

// ─────────────────────────────────────────────────────────────────────────────
// HISTORY SIDEBAR
// ─────────────────────────────────────────────────────────────────────────────

function initHistorySidebar() {
  $('history-toggle-btn').addEventListener('click', toggleHistorySidebar);
  $('history-close-btn').addEventListener('click', closeHistorySidebar);

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && AppState.historyOpen) closeHistorySidebar();
  });
}

function toggleHistorySidebar() {
  if (AppState.historyOpen) {
    closeHistorySidebar();
  } else {
    openHistorySidebar();
  }
}

function openHistorySidebar() {
  const sidebar = $('history-sidebar');
  sidebar.classList.remove('collapsed');
  AppState.historyOpen = true;
  $('history-toggle-btn').setAttribute('aria-expanded', 'true');

  if (!AppState.historyLoaded) loadHistory();
}

function closeHistorySidebar() {
  const sidebar = $('history-sidebar');
  sidebar.classList.add('collapsed');
  AppState.historyOpen = false;
  $('history-toggle-btn').setAttribute('aria-expanded', 'false');
}

async function loadHistory() {
  const list = $('history-list');
  list.innerHTML = `
    <div class="history-loading">
      <div class="loading-dot"></div>
      <span>Loading history…</span>
    </div>`;

  try {
    const res = await fetch(`/api/history/${AppState.patientId}`);
    if (!res.ok) throw new Error('Failed to load history');
    const data = await res.json();

    AppState.historyLoaded = true;
    list.innerHTML = '';

    if (!data.encounters || data.encounters.length === 0) {
      list.innerHTML = `<div class="history-empty">No past encounters found for this patient.</div>`;
      return;
    }

    data.encounters.forEach(enc => {
      list.appendChild(createEncounterCard(enc));
    });
  } catch (err) {
    list.innerHTML = `<div class="history-empty">Could not load history: ${escapeHtml(err.message)}</div>`;
  }
}

function createEncounterCard(enc) {
  const card = document.createElement('li');
  card.className = 'encounter-card';

  const sevClass = getSeverityClass(enc.severity || 5);
  const hasImg   = enc.has_image ? ' 📷' : '';
  const type     = enc.is_first_encounter ? '🩺 First Visit' : '🔄 Follow-up';

  const date   = document.createElement('div');
  date.className = 'card-date';
  date.textContent = `${formatDate(enc.date)}  •  ${type}${hasImg}`;

  const diag   = document.createElement('div');
  diag.className = 'card-diagnosis';
  diag.textContent = truncate(enc.diagnosis || 'Unknown', 80);

  const meta   = document.createElement('div');
  meta.className = 'card-meta';

  const sevPill = document.createElement('span');
  sevPill.className = `sev-pill ${sevClass}`;
  sevPill.textContent = `${parseFloat(enc.severity || 5).toFixed(1)}`;

  const regionSpan = document.createElement('span');
  regionSpan.className = 'text-xs text-muted';
  regionSpan.textContent = (enc.body_region || 'unspecified').replace(/_/g, ' ');

  meta.appendChild(sevPill);
  meta.appendChild(regionSpan);

  // Condition tags
  const tags = Array.isArray(enc.tags) ? enc.tags.slice(0, 3) : [];
  tags.forEach(t => {
    const span = document.createElement('span');
    span.className = 'tag text-xs';
    span.style.fontSize = '11px';
    span.textContent = t.replace(/_/g, ' ');
    meta.appendChild(span);
  });

  // Medication tags
  const meds = Array.isArray(enc.medications) ? enc.medications.slice(0, 3) : [];
  meds.forEach(m => {
    const span = document.createElement('span');
    span.className = 'tag tag--drug text-xs';
    span.style.fontSize = '11px';
    span.textContent = m;
    meta.appendChild(span);
  });

  card.appendChild(date);
  card.appendChild(diag);
  card.appendChild(meta);
  return card;
}

// ─────────────────────────────────────────────────────────────────────────────
// SCREEN 4 Nav
// ─────────────────────────────────────────────────────────────────────────────

function initScreen4() {
  $('new-consult-btn').addEventListener('click', () => {
    // Reset form
    $('complaint-textarea').value = '';
    $('medical-history-textarea').value = '';
    clearImage();
    closeHistorySidebar();
    AppState.historyLoaded = false;
    // Reset new-feature UI
    const dlBtn = $('download-report-btn');
    if (dlBtn) dlBtn.style.display = 'none';
    if (AppState.trendChart) { AppState.trendChart.destroy(); AppState.trendChart = null; }
    resetIntake();
    showScreen(2);
    updateConsultScreenForPatient();
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initScreen1();
  initScreen2();
  initHistorySidebar();
  initScreen4();
  showScreen(1);
});
