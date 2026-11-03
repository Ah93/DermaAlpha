"""
agent/prompts.py — All prompt templates for the DermaAlpha agent.
Also contains ClinicalDiff — a structured pre-computed diff object injected
into the progress comparison prompt so the LLM explains numbers rather than guessing.
"""

from dataclasses import dataclass, field


@dataclass
class VisualDiff:
    """Pre-computed visual similarity metrics between two encounter images."""
    similarity: float           # 0.0 (completely different) → 1.0 (identical)
    visual_change: float        # 1.0 - similarity
    change_label: str           # "LOW" | "MODERATE" | "HIGH"
    contradiction: str | None   # warning string or None


@dataclass
class ClinicalDiff:
    """Pre-computed clinical metrics comparing a past encounter to the current analysis."""
    severity_delta: float           # current - previous (negative = improving)
    severity_pct_change: float      # percentage change, e.g. -35.2
    severity_direction: str         # "improving" | "worsening" | "stable"
    tags_resolved: list[str]        # conditions in previous but not in current
    tags_new: list[str]             # conditions newly appeared in current
    tags_persistent: list[str]      # conditions present in both
    region_changed: bool            # True if body region is different
    days_since_last: int            # days between previous and today
    encounter_count: int            # total condition-matched encounters on record


def compute_clinical_diff(
    previous_encounter,     # EncounterRecord — local import avoids circular dependency
    current_analysis,       # AnalysisResult
    encounter_count: int,
) -> ClinicalDiff:
    """Compute a structured ClinicalDiff from a past encounter and current analysis result."""
    from datetime import date

    delta = current_analysis.severity_score - previous_encounter.severity_score
    pct = round((delta / previous_encounter.severity_score) * 100, 1) if previous_encounter.severity_score else 0.0

    if abs(delta) < 0.5:
        direction = "stable"
    elif delta < 0:
        direction = "improving"
    else:
        direction = "worsening"

    prev_tags = set(t.lower() for t in (previous_encounter.condition_tags or []))
    curr_tags = set(t.lower() for t in (current_analysis.condition_tags or []))

    try:
        prev_date = date.fromisoformat(previous_encounter.encounter_date)
        days_since = (date.today() - prev_date).days
    except (ValueError, TypeError):
        days_since = 0

    return ClinicalDiff(
        severity_delta=round(delta, 2),
        severity_pct_change=pct,
        severity_direction=direction,
        tags_resolved=sorted(prev_tags - curr_tags),
        tags_new=sorted(curr_tags - prev_tags),
        tags_persistent=sorted(prev_tags & curr_tags),
        region_changed=(previous_encounter.body_region != current_analysis.body_region),
        days_since_last=max(0, days_since),
        encounter_count=encounter_count,
    )

# ─────────────────────────────────────────────────────────────────────────────
# System prompt injected into every MedGemma call
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are DermaAlpha, an expert AI dermatology assistant.
Your role is to analyze skin conditions from patient-provided images and descriptions,
provide thorough clinical assessments, and track patient progress over time.

IMPORTANT GUIDELINES:
- Begin your response directly with the clinical assessment — no greetings, no "Hello", no "Thank you", no preamble
- Always be empathetic, clear, and informative
- You are NOT a replacement for a licensed dermatologist; always recommend professional consultation
- Flag urgent conditions (potential melanoma, severe infections, spreading cellulitis) prominently
- Base your assessment on both the image (if provided) and the patient's history

RESPONSE FORMAT:
1. Provide a thorough narrative response to the patient (2-4 paragraphs)
2. End every response with a JSON block in the following exact format:

```json
{
  "diagnosis": "primary diagnosis as a concise string",
  "differential": ["alternative diagnosis 1", "alternative diagnosis 2"],
  "severity_score": 5.0,
  "body_region": "specific body area e.g. left_forearm, face, scalp",
  "condition_tags": ["tag1", "tag2", "tag3"],
  "treatment_plan": "concise treatment recommendations",
  "urgent": false,
  "progress": "first_encounter"
}
```

SEVERITY SCORE GUIDE (1.0 – 10.0):
  1-2: Minimal, cosmetic concern only
  3-4: Mild, self-care sufficient
  5-6: Moderate, OTC treatment recommended
  7-8: Significant, prescription treatment recommended
  9-10: Severe/urgent, immediate medical attention required

PROGRESS VALUES:
  "first_encounter" — no prior history to compare
  "improving"       — condition is better than last visit
  "stable"          — condition is unchanged
  "worsening"       — condition has deteriorated
"""

# ─────────────────────────────────────────────────────────────────────────────
# First encounter — no prior history
# ─────────────────────────────────────────────────────────────────────────────

FIRST_ENCOUNTER_PROMPT = """This is the patient's FIRST visit. No prior history is on file.

PATIENT COMPLAINT:
{complaint}

PATIENT MEDICAL HISTORY (provided by patient):
{medical_history}

{image_note}

Please:
1. Analyze the skin condition thoroughly
2. Provide a diagnosis with differential diagnoses
3. Assign a severity score (1-10)
4. Recommend a treatment plan
5. Note any red flags or reasons to seek urgent care
6. End with the required JSON block (set "progress" to "first_encounter")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Follow-up encounter — past history available
# ─────────────────────────────────────────────────────────────────────────────

FOLLOW_UP_PROMPT = """This is a FOLLOW-UP visit. The patient has prior medical history on file.

PATIENT COMPLAINT (current visit):
{complaint}

{image_note}

── LAST PRESCRIBED TREATMENT (most recent visit) ────────────────────────────
{latest_treatment}
─────────────────────────────────────────────────────────────────────────────

── PAST ENCOUNTER HISTORY ───────────────────────────────────────────────────
{history_block}
─────────────────────────────────────────────────────────────────────────────

Please:
1. Analyze the current skin condition from the image and complaint
2. Explicitly evaluate whether the last prescribed treatment worked:
   - If severity improved → confirm it is working, continue or taper as appropriate
   - If severity worsened or unchanged → state the treatment failed and explain why
   - Adjust or escalate the treatment plan accordingly
3. Compare to the most recent encounter and assess progress direction
4. End with the required JSON block (set "progress" to one of: improving/worsening/stable)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Progress comparison — generates a focused progress note
# ─────────────────────────────────────────────────────────────────────────────

PROGRESS_COMPARISON_PROMPT = """Generate a clear, patient-friendly progress note comparing two dermatology encounters.

PREVIOUS ENCOUNTER ({previous_date}):
Diagnosis: {previous_diagnosis}
Severity Score: {previous_severity}/10
Treatment Given: {previous_treatment}
Summary: {previous_summary}

CURRENT ENCOUNTER ({current_date}):
Diagnosis: {current_diagnosis}
Severity Score: {current_severity}/10
Current Summary: {current_summary}

Write a 2-3 sentence progress note that:
1. States clearly whether the condition has improved, worsened, or remained stable
2. Attributes the change to treatment effectiveness (or lack thereof) if applicable
3. Uses plain, empathetic language the patient can understand

Examples of good progress notes:
- "Your eczema has improved significantly since your last visit — the redness and scaling have reduced from a severity of 7 to 4. This suggests the prescribed topical corticosteroid is working well. Continue the current regimen and return if symptoms flare."
- "Unfortunately, your psoriasis appears to have worsened since we last met, with plaques spreading from your elbows to your forearms and severity increasing from 5 to 7. This may indicate the current treatment is insufficient — I'd recommend an urgent review to consider escalating therapy."
"""

# ─────────────────────────────────────────────────────────────────────────────
# Progress comparison WITH pre-computed clinical diff (preferred — more accurate)
# ─────────────────────────────────────────────────────────────────────────────

PROGRESS_COMPARISON_WITH_DIFF_PROMPT = """Generate a clear, patient-friendly progress note comparing two dermatology encounters.

PREVIOUS ENCOUNTER ({previous_date}):
Diagnosis: {previous_diagnosis}
Severity Score: {previous_severity}/10
Treatment Given: {previous_treatment}
Summary: {previous_summary}

CURRENT ENCOUNTER ({current_date}):
Diagnosis: {current_diagnosis}
Severity Score: {current_severity}/10
Summary: {current_summary}

── PRE-COMPUTED CLINICAL METRICS — use these exact numbers, do not recalculate ──
Severity change        : {severity_delta:+.2f} points ({severity_pct_change:+.1f}%)  →  {severity_direction}
Days since last visit  : {days_since_last} days
Total encounters       : {encounter_count}
Conditions resolved    : {tags_resolved}
Conditions new         : {tags_new}
Conditions persistent  : {tags_persistent}
Body region changed    : {region_changed}
─────────────────────────────────────────────────────────────────────────────────

{visual_evidence_block}

── TREATMENT EFFECTIVENESS ASSESSMENT ───────────────────────────────────────
Previous treatment: {previous_treatment}
Severity went from {previous_severity}/10 → {current_severity}/10 ({severity_direction}, {severity_delta:+.2f} points)

Explicitly assess:
- Did the previous treatment work? (yes / partially / no — based on severity change)
- If it failed or partially worked, what is the likely reason?
- What is the updated recommendation? (continue / modify / escalate / switch)
─────────────────────────────────────────────────────────────────────────────────

Write a 4-5 sentence progress note that:
1. States clearly whether the condition is {severity_direction} — cite the exact severity numbers
2. Explicitly states whether the previous treatment worked or failed and why
3. Gives a clear updated treatment recommendation based on effectiveness
4. References the visual similarity finding if an image was available
5. If a CONTRADICTION was detected, flag it clearly and suggest the patient verify with a clinician
6. Uses plain, empathetic language a patient can understand

Do NOT guess or re-derive numbers. Use only the pre-computed metrics above.
"""

# ─────────────────────────────────────────────────────────────────────────────
# History block template — formats a single past encounter for context injection
# ─────────────────────────────────────────────────────────────────────────────

HISTORY_ENCOUNTER_BLOCK = """[Encounter {index} — {date}]
Diagnosis: {diagnosis}
Severity: {severity}/10
Body Region: {body_region}
Conditions: {condition_tags}
Treatment: {treatment_plan}
Summary: {summary}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Medical history intake prompt (asked to patient on first encounter)
# ─────────────────────────────────────────────────────────────────────────────

MEDICAL_HISTORY_REQUEST = """To provide you with the best assessment, I need a brief medical history.
Please answer the following (you can skip any that don't apply):

  1. Do you have any known skin conditions (e.g., eczema, psoriasis, rosacea)?
  2. Are you currently taking any medications (oral or topical)?
  3. Do you have any allergies (medications, materials, foods)?
  4. Do you have any chronic conditions (diabetes, autoimmune disorders)?
  5. Has anyone in your family had similar skin problems or skin cancer?
  6. How long have you had the current symptom?
  7. Have you tried any treatments already? If so, did they help?

You can type your answers in free text — no need to follow the numbered format exactly.
(Type 'skip' to proceed without providing history.)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions to build populated prompts
# ─────────────────────────────────────────────────────────────────────────────

def _classifier_note(classifier_result: dict | None) -> str:
    """Format a classifier result block to inject into prompts."""
    if not classifier_result:
        return ""
    preds = classifier_result.get("predictions", [])
    if not preds:
        return ""
    lines = ", ".join(
        f"{p['label']} ({p['score']}%)" for p in preds[:3]
    )
    return (
        f"\nAI PRE-SCREENING (automated image classifier — treat as hints, "
        f"not final diagnoses): {lines}\n"
    )


def build_first_encounter_prompt(
    complaint: str,
    medical_history: str,
    has_image: bool,
    classifier_result: dict | None = None,
) -> str:
    image_note = (
        "An image of the affected area has been provided. Please analyze it carefully."
        if has_image
        else "No image was provided. Please base your assessment on the description alone."
    )
    classifier_note = _classifier_note(classifier_result)
    return FIRST_ENCOUNTER_PROMPT.format(
        complaint=complaint + classifier_note,
        medical_history=medical_history or "Not provided.",
        image_note=image_note,
    )


def build_follow_up_prompt(
    complaint: str,
    past_encounters: list,
    has_image: bool,
    classifier_result: dict | None = None,
    latest_treatment: str | None = None,
) -> str:
    image_note = (
        "An image of the current condition has been provided. Please analyze it carefully."
        if has_image
        else "No image was provided for this visit."
    )

    history_blocks = []
    for i, enc in enumerate(past_encounters, start=1):
        tags = ", ".join(enc.condition_tags) if enc.condition_tags else "N/A"
        block = HISTORY_ENCOUNTER_BLOCK.format(
            index=i,
            date=enc.encounter_date,
            diagnosis=enc.diagnosis_summary[:300],
            severity=enc.severity_score,
            body_region=enc.body_region,
            condition_tags=tags,
            treatment_plan=enc.treatment_plan[:200] if enc.treatment_plan else "None documented",
            summary=enc.diagnosis_summary[:400],
        )
        history_blocks.append(block)

    history_block = "\n".join(history_blocks) if history_blocks else "No relevant past encounters found."
    classifier_note = _classifier_note(classifier_result)
    treatment_text = latest_treatment.strip()[:400] if latest_treatment else "None documented for previous visit."

    return FOLLOW_UP_PROMPT.format(
        complaint=complaint + classifier_note,
        image_note=image_note,
        history_block=history_block,
        latest_treatment=treatment_text,
    )


def _build_visual_evidence_block(visual_diff) -> str:
    """Format the visual similarity section for injection into the prompt."""
    if visual_diff is None:
        return "── VISUAL COMPARISON ─────────────────────────────────────────────────────────\nNo image available for one or both visits — visual comparison skipped.\n─────────────────────────────────────────────────────────────────────────────────"

    lines = [
        "── OBJECTIVE VISUAL EVIDENCE (BioMedCLIP image analysis) ────────────────────",
        f"Visual similarity score  : {visual_diff.similarity:.3f} / 1.000",
        f"Visual change magnitude  : {visual_diff.change_label}  ({visual_diff.visual_change:.1%} change)",
        "  (1.0 = images identical, 0.0 = completely different)",
    ]

    if visual_diff.contradiction:
        lines.append("")
        lines.append(f"⚠  {visual_diff.contradiction}")

    lines.append("─────────────────────────────────────────────────────────────────────────────────")
    return "\n".join(lines)


def build_progress_comparison_prompt(
    previous_encounter,
    current_analysis,
    clinical_diff=None,     # ClinicalDiff | None
    visual_diff=None,       # VisualDiff | None
) -> str:
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")

    base_args = dict(
        previous_date=previous_encounter.encounter_date,
        previous_diagnosis=previous_encounter.diagnosis_summary[:300],
        previous_severity=previous_encounter.severity_score,
        previous_treatment=previous_encounter.treatment_plan[:200] if previous_encounter.treatment_plan else "None",
        previous_summary=previous_encounter.diagnosis_summary[:400],
        current_date=current_date,
        current_diagnosis=current_analysis.diagnosis,
        current_severity=current_analysis.severity_score,
        current_summary=current_analysis.raw_text[:400],
    )

    if clinical_diff is None:
        return PROGRESS_COMPARISON_PROMPT.format(**base_args)

    visual_evidence_block = _build_visual_evidence_block(visual_diff)

    return PROGRESS_COMPARISON_WITH_DIFF_PROMPT.format(
        **base_args,
        severity_delta=clinical_diff.severity_delta,
        severity_pct_change=clinical_diff.severity_pct_change,
        severity_direction=clinical_diff.severity_direction,
        days_since_last=clinical_diff.days_since_last,
        encounter_count=clinical_diff.encounter_count,
        tags_resolved=", ".join(clinical_diff.tags_resolved) or "none",
        tags_new=", ".join(clinical_diff.tags_new) or "none",
        tags_persistent=", ".join(clinical_diff.tags_persistent) or "none",
        region_changed="yes" if clinical_diff.region_changed else "no",
        visual_evidence_block=visual_evidence_block,
    )
