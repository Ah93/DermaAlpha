"""
app.py — DermaAlpha Flask web application.
Serves the SPA frontend and exposes JSON API endpoints for the agent.
"""

import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

# ── Flask app setup ───────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ── Lazy-initialised singletons ───────────────────────────────────────────────
_analyzer = None
_memory = None
_classifier = None    # LesionClassifier — None if model unavailable
_visual_embedder = None  # VisualEmbedder (BioMedCLIP) — None if unavailable


def get_services():
    """Return (analyzer, memory), initialising them on first call."""
    global _analyzer, _memory
    from config import settings
    from agent.medical_analyzer import MedicalAnalyzer
    from agent.memory_manager import MemoryManager

    if _analyzer is None:
        _analyzer = MedicalAnalyzer(settings)
    if _memory is None:
        _memory = MemoryManager(settings)
    return _analyzer, _memory


def get_classifier():
    """Return LesionClassifier singleton (lazy, never raises)."""
    global _classifier
    if _classifier is None:
        from agent.lesion_classifier import LesionClassifier
        _classifier = LesionClassifier()
    return _classifier


def get_visual_embedder():
    """Return VisualEmbedder singleton (lazy, never raises)."""
    global _visual_embedder
    if _visual_embedder is None:
        from agent.visual_embedder import VisualEmbedder
        _visual_embedder = VisualEmbedder()
    return _visual_embedder


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/check-patient", methods=["POST"])
def check_patient():
    """
    Validate or generate a patient ID.
    Body: { "patient_id": "PT-ABC123" | "new" }
    Returns: { "patient_id", "is_new", "exists" }
    """
    data = request.get_json(silent=True) or {}
    raw_id = (data.get("patient_id") or "").strip().upper()

    try:
        _, memory = get_services()
    except Exception as e:
        return jsonify({"error": "Service initialisation failed", "detail": str(e)}), 503

    if not raw_id or raw_id in {"NEW", "N"}:
        patient_id = f"PT-{uuid.uuid4().hex[:6].upper()}"
        return jsonify({"patient_id": patient_id, "is_new": True, "exists": False})

    # Normalize: UI shows "PT-" as a visual prefix so user types just the suffix.
    # Handle both "ABC123" and "PT-ABC123" inputs.
    if not raw_id.startswith("PT-"):
        raw_id = f"PT-{raw_id}"

    exists = memory.patient_exists(raw_id)
    return jsonify({"patient_id": raw_id, "is_new": not exists, "exists": exists})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Run a full dermatology analysis and save the encounter to Pinecone.
    Input: multipart/form-data  (patient_id, complaint, medical_history?, image?)
    Returns: full analysis JSON
    """
    from config import settings
    from agent.memory_manager import EncounterRecord, MemoryManager
    from agent.prompts import (
        build_first_encounter_prompt,
        build_follow_up_prompt,
        build_progress_comparison_prompt,
        compute_clinical_diff,
        VisualDiff,
    )
    from utils.image_utils import load_and_encode_image

    patient_id = (request.form.get("patient_id") or "").strip().upper()
    complaint = (request.form.get("complaint") or "").strip()
    medical_history = (request.form.get("medical_history") or "").strip()

    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400
    if not complaint:
        return jsonify({"error": "complaint is required"}), 400

    try:
        analyzer, memory = get_services()
    except Exception as e:
        return jsonify({"error": "Service initialisation failed", "detail": str(e)}), 503

    # ── Handle image upload ───────────────────────────────────────────────────
    image_data_uri = None
    tmp_path = None

    try:
        if "image" in request.files and request.files["image"].filename:
            img_file = request.files["image"]
            suffix = Path(secure_filename(img_file.filename)).suffix.lower() or ".jpg"
            os.makedirs(settings.image_cache_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                dir=settings.image_cache_dir,
                delete=False,
            ) as tmp:
                img_file.save(tmp)
                tmp_path = tmp.name

            _, image_data_uri = load_and_encode_image(
                tmp_path, max_dim=settings.max_image_dim
            )

        # ── Run lesion classifier (best-effort, non-blocking) ─────────────────
        lesion_classification = None
        if image_data_uri:
            try:
                lesion_classification = get_classifier().classify(image_data_uri)
            except Exception as e:
                logger.warning("Lesion classifier skipped: %s", e)

        # ── Generate BioMedCLIP visual fingerprint (best-effort) ──────────────
        current_image_vector = None
        if image_data_uri:
            try:
                current_image_vector = get_visual_embedder().embed_image(image_data_uri)
            except Exception as e:
                logger.warning("Visual embedding skipped: %s", e)

        # ── Determine encounter type ──────────────────────────────────────────
        is_first_encounter = not memory.patient_exists(patient_id)
        past_encounters = []
        latest_encounter = None

        if not is_first_encounter:
            past_encounters = memory.get_patient_history(
                patient_id=patient_id,
                current_complaint=complaint,
                top_k=settings.max_retrieved_encounters,
            )
            latest_encounter = memory.get_latest_encounter(patient_id)

        # ── Build prompt and analyse ──────────────────────────────────────────
        if is_first_encounter:
            prompt = build_first_encounter_prompt(
                complaint=complaint,
                medical_history=medical_history or "Not provided.",
                has_image=image_data_uri is not None,
                classifier_result=lesion_classification,
            )
        else:
            prompt = build_follow_up_prompt(
                complaint=complaint,
                past_encounters=past_encounters,
                has_image=image_data_uri is not None,
                classifier_result=lesion_classification,
                latest_treatment=latest_encounter.treatment_plan if latest_encounter else None,
            )

        analysis = analyzer.analyze(prompt=prompt, image_data_uri=image_data_uri, max_new_tokens=4096)

        # ── Progress comparison (follow-up only) ──────────────────────────────
        progress_note = ""
        previous_severity = None
        condition_chain = []
        visual_diff = None

        if not is_first_encounter and latest_encounter:
            previous_severity = latest_encounter.severity_score

            # Temporal chain: all encounters sharing tags with current condition
            condition_chain = memory.get_condition_chain(
                patient_id=patient_id,
                condition_tags=analysis.condition_tags,
            )

            # Structured clinical diff — computed before LLM call
            diff = compute_clinical_diff(
                previous_encounter=latest_encounter,
                current_analysis=analysis,
                encounter_count=len(condition_chain) if condition_chain else len(past_encounters),
            )

            # ── Visual similarity diff (BioMedCLIP) ──────────────────────────
            visual_diff = None
            if current_image_vector is not None and latest_encounter.has_image:
                try:
                    from agent.visual_embedder import VisualEmbedder
                    embedder = get_visual_embedder()

                    prev_image_vector = memory.get_image_vector(
                        patient_id=patient_id,
                        encounter_id=latest_encounter.encounter_id,
                    )

                    if prev_image_vector is not None:
                        similarity  = embedder.cosine_similarity(
                            current_image_vector, prev_image_vector
                        )
                        change_label = embedder.interpret_similarity(similarity)
                        contradiction = embedder.detect_contradiction(
                            similarity,
                            previous_severity=latest_encounter.severity_score,
                            current_severity=analysis.severity_score,
                        )
                        visual_diff = VisualDiff(
                            similarity=round(similarity, 4),
                            visual_change=round(1.0 - similarity, 4),
                            change_label=change_label,
                            contradiction=contradiction,
                        )
                        logger.info(
                            "Visual similarity: %.3f (%s) | contradiction: %s",
                            similarity, change_label, contradiction is not None,
                        )
                except Exception as e:
                    logger.warning("Visual diff skipped: %s", e)

            progress_prompt = build_progress_comparison_prompt(
                previous_encounter=latest_encounter,
                current_analysis=analysis,
                clinical_diff=diff,
                visual_diff=visual_diff,
            )
            prog_result = analyzer.analyze(
                prompt=progress_prompt,
                image_data_uri=None,
                max_new_tokens=1024,
            )
            progress_note = prog_result.raw_text.split("```")[0].strip()

        # ── Extract medications from treatment plan ───────────────────────────
        drug_tags = MemoryManager.extract_drug_tags(analysis.treatment_plan)

        # ── Save encounter ────────────────────────────────────────────────────
        now = datetime.now()
        encounter_id = f"ENC-{uuid.uuid4().hex[:8].upper()}"

        record = EncounterRecord(
            patient_id=patient_id,
            encounter_id=encounter_id,
            encounter_date=now.strftime("%Y-%m-%d"),
            encounter_timestamp=int(now.timestamp()),
            chief_complaint=complaint,
            body_region=analysis.body_region,
            condition_tags=analysis.condition_tags,
            severity_score=analysis.severity_score,
            has_image=image_data_uri is not None,
            diagnosis_summary=analysis.diagnosis + ". " + analysis.raw_text[:600],
            treatment_plan=analysis.treatment_plan,
            drug_tags=drug_tags,
            image_vector=current_image_vector,   # BioMedCLIP fingerprint
            is_first_encounter=is_first_encounter,
            follow_up_of=(
                latest_encounter.encounter_id if latest_encounter else None
            ),
        )
        memory.upsert_encounter(record)

        # ── Build response ────────────────────────────────────────────────────
        narrative = analysis.raw_text
        # Strip trailing ```json block from narrative shown to user
        if "```json" in narrative:
            narrative = narrative[: narrative.index("```json")].strip()
        elif "```" in narrative:
            narrative = narrative[: narrative.rindex("```")].strip()

        # Build condition chain + append current encounter so chart includes today.
        # If tag-matched chain is empty (different diagnosis), fall back to all past encounters
        chain_source = condition_chain if condition_chain else (past_encounters if not is_first_encounter else [])
        # Always include latest_encounter if not already in chain_source
        chain_ids = {enc.encounter_id for enc in chain_source}
        if not is_first_encounter and latest_encounter and latest_encounter.encounter_id not in chain_ids:
            chain_source = list(chain_source) + [latest_encounter]

        chain_for_chart = [
            {
                "encounter_id": enc.encounter_id,
                "date": enc.encounter_date,
                "severity": enc.severity_score,
                "tags": enc.condition_tags,
                "body_region": enc.body_region,
                "diagnosis": enc.diagnosis_summary[:150],
            }
            for enc in chain_source
        ]
        # Append current encounter at end (not yet in Pinecone at query time)
        chain_for_chart.append({
            "encounter_id": encounter_id,
            "date": now.strftime("%Y-%m-%d"),
            "severity": analysis.severity_score,
            "tags": analysis.condition_tags,
            "body_region": analysis.body_region,
            "diagnosis": analysis.diagnosis,
        })

        return jsonify(
            {
                "encounter_id": encounter_id,
                "patient_id": patient_id,
                "encounter_date": now.strftime("%Y-%m-%d"),
                "is_first_encounter": is_first_encounter,
                "diagnosis": analysis.diagnosis,
                "differential": analysis.differential,
                "severity_score": analysis.severity_score,
                "previous_severity": previous_severity,
                "body_region": analysis.body_region,
                "condition_tags": analysis.condition_tags,
                "treatment_plan": analysis.treatment_plan,
                "medications": drug_tags,
                "urgent": analysis.urgent,
                "progress": analysis.progress,
                "narrative": narrative,
                "progress_note": progress_note,
                "past_encounters_count": len(past_encounters),
                "condition_chain_count": len(condition_chain),
                "lesion_classification": lesion_classification,
                "visual_similarity": {
                    "score":        round(visual_diff.similarity, 3),
                    "change":       visual_diff.change_label,
                    "contradiction": visual_diff.contradiction,
                } if visual_diff else None,
                "past_encounters": [
                    {
                        "encounter_id": enc.encounter_id,
                        "date": enc.encounter_date,
                        "diagnosis": enc.diagnosis_summary[:200],
                        "severity": enc.severity_score,
                        "body_region": enc.body_region,
                        "tags": enc.condition_tags,
                    }
                    for enc in past_encounters
                ],
                # Chronological condition chain (oldest → current) for trend chart
                "condition_chain": chain_for_chart,
            }
        )

    except RuntimeError as e:
        logger.error("Analysis failed: %s", e)
        return (
            jsonify(
                {
                    "error": "Analysis service unavailable",
                    "detail": str(e),
                }
            ),
            503,
        )
    except Exception as e:
        logger.exception("Unexpected error in /api/analyze")
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/api/report/<patient_id>/<encounter_id>")
def download_report(patient_id, encounter_id):
    """
    Generate and stream a PDF report for a specific encounter.
    GET /api/report/PT-ABC123/ENC-12345678
    """
    patient_id  = patient_id.upper()
    encounter_id = encounter_id.upper()

    try:
        _, memory = get_services()
    except Exception as e:
        return jsonify({"error": "Service unavailable", "detail": str(e)}), 503

    enc = memory.get_encounter_by_id(patient_id, encounter_id)
    if not enc:
        return jsonify({"error": "Encounter not found"}), 404

    try:
        from utils.report_generator import generate_encounter_pdf
        import io as _io

        # Split stored diagnosis_summary back into clean diagnosis + narrative
        # Stored format: "Diagnosis Name. raw_text_excerpt..."
        diag_summary = enc.diagnosis_summary or ""
        dot_idx = diag_summary.find(". ")
        if dot_idx > 0 and dot_idx < 120:
            clean_diagnosis = diag_summary[:dot_idx]
            narrative_text  = diag_summary[dot_idx + 2:]
        else:
            clean_diagnosis = diag_summary[:120]
            narrative_text  = diag_summary

        # For follow-up visits fetch the previous encounter to derive progress
        previous_severity = None
        progress_label    = "stable"
        if not enc.is_first_encounter and enc.follow_up_of:
            prev_enc = memory.get_encounter_by_id(patient_id, enc.follow_up_of)
            if prev_enc:
                previous_severity = prev_enc.severity_score
                delta = enc.severity_score - prev_enc.severity_score
                if delta < -0.5:
                    progress_label = "improving"
                elif delta > 0.5:
                    progress_label = "worsening"

        # Build the data dict the generator expects
        report_data = {
            "patient_id":         enc.patient_id,
            "encounter_id":       enc.encounter_id,
            "encounter_date":     enc.encounter_date,
            "chief_complaint":    enc.chief_complaint,
            "diagnosis":          clean_diagnosis,
            "differential":       [],
            "severity_score":     enc.severity_score,
            "body_region":        enc.body_region,
            "condition_tags":     enc.condition_tags,
            "treatment_plan":     enc.treatment_plan,
            "medications":        enc.drug_tags,
            "narrative":          narrative_text,
            "is_first_encounter": enc.is_first_encounter,
            "progress":           progress_label,
            "progress_note":      narrative_text if not enc.is_first_encounter else "",
            "previous_severity":  previous_severity,
            "lesion_classification": None,
        }

        pdf_bytes = generate_encounter_pdf(report_data)
        filename  = f"DermaAlpha_{patient_id}_{encounter_id}.pdf"

        return send_file(
            _io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
        )
    except Exception as e:
        logger.exception("PDF generation failed")
        return jsonify({"error": "PDF generation failed", "detail": str(e)}), 500


@app.route("/api/history/<patient_id>")
def history(patient_id):
    """
    Retrieve all past encounters for a patient.
    Returns: { "patient_id", "count", "encounters": [...] }
    """
    patient_id = patient_id.upper()

    try:
        _, memory = get_services()
    except Exception as e:
        return jsonify({"error": "Service initialisation failed", "detail": str(e)}), 503

    if not memory.patient_exists(patient_id):
        return jsonify({"patient_id": patient_id, "count": 0, "encounters": []})

    encounters = memory.get_all_encounters(patient_id)
    medication_history = memory.get_medication_history(patient_id)
    return jsonify(
        {
            "patient_id": patient_id,
            "count": len(encounters),
            "medication_history": medication_history,
            "encounters": [
                {
                    "encounter_id": enc.encounter_id,
                    "date": enc.encounter_date,
                    "diagnosis": enc.diagnosis_summary[:300],
                    "severity": enc.severity_score,
                    "body_region": enc.body_region,
                    "tags": enc.condition_tags,
                    "medications": enc.drug_tags,
                    "treatment": enc.treatment_plan[:200],
                    "has_image": enc.has_image,
                    "chief_complaint": enc.chief_complaint[:200],
                    "is_first_encounter": enc.is_first_encounter,
                }
                for enc in encounters
            ],
        }
    )


# ── Edge TTS endpoint ─────────────────────────────────────────────────────────

@app.route("/api/tts", methods=["POST"])
def tts():
    """
    Generate speech audio using Edge TTS (Microsoft Neural voices).
    POST { "text": "...", "voice": "en-US-GuyNeural" }
    Returns: audio/mpeg stream
    """
    import asyncio
    import io
    import edge_tts

    body  = request.get_json(force=True, silent=True) or {}
    text  = (body.get("text") or "").strip()
    voice = body.get("voice") or "en-US-GuyNeural"

    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        async def _synthesise():
            communicate = edge_tts.Communicate(text, voice)
            buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            return buf

        audio_buf = asyncio.run(_synthesise())

        return send_file(
            audio_buf,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="speech.mp3",
        )
    except Exception as e:
        logger.exception("Edge TTS failed")
        return jsonify({"error": "TTS failed", "detail": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, port=5000, host="0.0.0.0")
