"""
utils/report_generator.py — PDF patient report generator using ReportLab.
Produces a clean, professional medical report for a single encounter.
"""

import io
from datetime import datetime


def generate_encounter_pdf(encounter_data: dict) -> bytes:
    """
    Generate a PDF report for one encounter and return it as bytes.

    encounter_data keys (all optional with safe defaults):
        patient_id, encounter_id, encounter_date, diagnosis, differential,
        severity_score, body_region, condition_tags, treatment_plan,
        narrative, progress, progress_note, previous_severity,
        is_first_encounter, lesion_classification
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    # ── Colours (matching app dark theme, but on white paper) ─────────────────
    TEAL    = colors.HexColor("#0d9488")
    NAVY    = colors.HexColor("#0c1a2e")
    BLUE    = colors.HexColor("#0ea5e9")
    MUTED   = colors.HexColor("#64748b")
    URGENT  = colors.HexColor("#ef4444")
    GREEN   = colors.HexColor("#10b981")
    AMBER   = colors.HexColor("#f59e0b")

    # ── Styles ─────────────────────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    def style(name, **kwargs):
        return ParagraphStyle(name, parent=styles["Normal"], **kwargs)

    H1   = style("H1",   fontSize=22, textColor=NAVY,  fontName="Helvetica-Bold",  spaceAfter=2)
    H2   = style("H2",   fontSize=14, textColor=TEAL,  fontName="Helvetica-Bold",  spaceAfter=6,  spaceBefore=12)
    H3   = style("H3",   fontSize=12, textColor=NAVY,  fontName="Helvetica-Bold",  spaceAfter=4)
    BODY = style("BODY", fontSize=10, textColor=colors.black, leading=15)
    SMALL= style("SMALL",fontSize=9,  textColor=MUTED,  leading=13)
    MONO = style("MONO", fontSize=10, textColor=NAVY,  fontName="Courier-Bold")
    CENT = style("CENT", fontSize=9,  textColor=MUTED,  alignment=TA_CENTER)
    DISC = style("DISC", fontSize=8,  textColor=MUTED,  leading=12, italic=True)

    # ── Document setup ─────────────────────────────────────────────────────────
    buf = io.BytesIO()
    PAGE_W, PAGE_H = A4
    MARGIN = 20 * mm

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
        title="DermaAlpha Clinical Report",
        author="DermaAlpha AI",
    )

    # ── Helpers ────────────────────────────────────────────────────────────────
    def HR(color=TEAL, thickness=1):
        return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=4)

    def to_hex(c) -> str:
        """Convert a reportlab Color to CSS hex string (without #)."""
        r = int(round(c.red   * 255))
        g = int(round(c.green * 255))
        b = int(round(c.blue  * 255))
        return f"{r:02x}{g:02x}{b:02x}"

    def sev_color(score):
        s = float(score or 5)
        if s <= 3:   return GREEN
        if s <= 6:   return AMBER
        if s <= 8:   return URGENT
        return colors.HexColor("#dc2626")

    def get(key, default="—"):
        v = encounter_data.get(key)
        return v if v not in (None, "", [], {}) else default

    # ── Content builder ────────────────────────────────────────────────────────
    elems = []

    # Header bar
    header_data = [[
        Paragraph("<b>DermaAlpha</b>", style("HL", fontSize=18, textColor=colors.white, fontName="Helvetica-Bold")),
        Paragraph("AI Dermatology Report", style("HR", fontSize=10, textColor=colors.HexColor("#b2f5ea"), alignment=TA_RIGHT)),
    ]]
    header_table = Table(header_data, colWidths=[PAGE_W - 2*MARGIN - 60*mm, 60*mm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), TEAL),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    elems.append(header_table)
    elems.append(Spacer(1, 8))

    # Patient info row
    enc_type = "First Visit" if get("is_first_encounter", True) else "Follow-up Visit"
    pid = get("patient_id")
    eid = get("encounter_id")
    edate = get("encounter_date")
    try:
        edate = datetime.strptime(edate, "%Y-%m-%d").strftime("%B %d, %Y")
    except Exception:
        pass

    info_data = [
        ["Patient ID", pid, "Encounter", eid],
        ["Date",       edate, "Type",    enc_type],
    ]
    info_table = Table(info_data, colWidths=[30*mm, 65*mm, 30*mm, 45*mm])
    info_table.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",    (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("TEXTCOLOR",   (0,0), (0,-1), MUTED),
        ("TEXTCOLOR",   (2,0), (2,-1), MUTED),
        ("TEXTCOLOR",   (1,0), (1,-1), NAVY),
        ("TEXTCOLOR",   (3,0), (3,-1), NAVY),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("BACKGROUND",  (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ("BOX",         (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    elems.append(info_table)
    elems.append(Spacer(1, 6))
    elems.append(HR())

    # ── Clinical Assessment ────────────────────────────────────────────────────
    elems.append(Paragraph("Clinical Assessment", H2))

    diagnosis = get("diagnosis")
    severity  = float(encounter_data.get("severity_score") or 5)
    body      = get("body_region", "unspecified").replace("_", " ").title()
    sev_col   = sev_color(severity)

    # Diagnosis + Severity side by side
    sev_label = (
        "Mild" if severity <= 3 else
        "Moderate" if severity <= 6 else
        "Significant" if severity <= 8 else "Urgent"
    )
    diag_data = [[
        Paragraph(f"<b>{diagnosis}</b>", H3),
        Paragraph(
            f'<font color="#{to_hex(sev_col)}"><b>{severity:.1f} / 10</b></font><br/>'
            f'<font color="#{to_hex(sev_col)}" size="9">{sev_label}</font>',
            style("SEV", fontSize=14, alignment=TA_CENTER)
        ),
    ]]
    diag_table = Table(diag_data, colWidths=[PAGE_W - 2*MARGIN - 45*mm, 45*mm])
    diag_table.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
    ]))
    elems.append(diag_table)
    elems.append(Spacer(1, 4))

    # Body region + tags
    tags = encounter_data.get("condition_tags") or []
    tags_str = " · ".join(t.replace("_", " ").title() for t in tags) if tags else "—"
    elems.append(Paragraph(f"<b>Region:</b> {body}    <b>Condition Tags:</b> {tags_str}", SMALL))
    elems.append(Spacer(1, 4))

    # Differential diagnoses
    diff = encounter_data.get("differential") or []
    if diff:
        diff_str = "  •  ".join(diff[:4])
        elems.append(Paragraph(f"<b>Differential:</b> {diff_str}", SMALL))
        elems.append(Spacer(1, 4))

    # Lesion classifier result (if available)
    lc = encounter_data.get("lesion_classification")
    if lc and isinstance(lc, dict):
        preds = lc.get("predictions", [])
        if preds:
            preds_str = "  |  ".join(f"{p['label']} ({p['score']}%)" for p in preds[:3])
            elems.append(Paragraph(f"<b>AI Pre-Classifier:</b> {preds_str}", SMALL))
            elems.append(Spacer(1, 4))

    elems.append(HR(color=colors.HexColor("#e2e8f0"), thickness=0.5))

    # ── Narrative ─────────────────────────────────────────────────────────────
    narrative = (encounter_data.get("narrative") or "").strip()
    if narrative:
        elems.append(Paragraph("Full Assessment", H2))
        for para in narrative.split("\n\n"):
            clean = para.strip()
            if clean:
                elems.append(Paragraph(clean, BODY))
                elems.append(Spacer(1, 4))
        elems.append(HR(color=colors.HexColor("#e2e8f0"), thickness=0.5))

    # ── Treatment Plan ────────────────────────────────────────────────────────
    treatment = (encounter_data.get("treatment_plan") or "").strip()
    if treatment:
        elems.append(Paragraph("Treatment Plan", H2))
        for line in treatment.split("\n"):
            clean = line.strip()
            if clean:
                elems.append(Paragraph(f"• {clean}", BODY))
                elems.append(Spacer(1, 2))
        elems.append(HR(color=colors.HexColor("#e2e8f0"), thickness=0.5))

    # ── Medications (drug memory) ─────────────────────────────────────────────
    medications = encounter_data.get("medications") or []
    if medications:
        meds_str = "  •  ".join(m.title() for m in medications)
        elems.append(Paragraph("Medications Mentioned", H2))
        elems.append(Paragraph(meds_str, BODY))
        elems.append(Spacer(1, 4))
        elems.append(HR(color=colors.HexColor("#e2e8f0"), thickness=0.5))

    # ── Progress Note (follow-up only) ────────────────────────────────────────
    progress_note = (encounter_data.get("progress_note") or "").strip()
    prev_sev = encounter_data.get("previous_severity")
    progress  = encounter_data.get("progress", "first_encounter")

    if progress_note and progress != "first_encounter":
        elems.append(Paragraph("Progress Assessment", H2))

        prog_color = GREEN if progress == "improving" else (URGENT if progress == "worsening" else MUTED)
        badge_label = {"improving": "↑ IMPROVING", "worsening": "↓ WORSENING", "stable": "→ STABLE"}.get(progress, progress.upper())
        elems.append(Paragraph(
            f'<font color="#{to_hex(prog_color)}"><b>{badge_label}</b></font>'
            + (f'   Previous severity: {float(prev_sev):.1f}  →  Current: {severity:.1f}' if prev_sev else ""),
            style("PROG", fontSize=11, spaceBefore=2, spaceAfter=6)
        ))

        for para in progress_note.split("\n\n"):
            clean = para.strip()
            if clean:
                elems.append(Paragraph(clean, BODY))
                elems.append(Spacer(1, 3))

        elems.append(HR(color=colors.HexColor("#e2e8f0"), thickness=0.5))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(
        "⚕ Medical Disclaimer: This report is generated by DermaAlpha AI for informational purposes only. "
        "It does not constitute a medical diagnosis or replace professional medical advice. "
        "Always consult a licensed dermatologist or healthcare provider for any skin concerns.",
        DISC,
    ))
    elems.append(Spacer(1, 4))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    elems.append(Paragraph(f"Generated by DermaAlpha  |  {generated_at}", CENT))

    # ── Build PDF ──────────────────────────────────────────────────────────────
    doc.build(elems)
    return buf.getvalue()
