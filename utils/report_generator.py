"""
utils/report_generator.py — PDF patient report generator using ReportLab.
Produces a clean, professional medical report for a single encounter.
Supports both New Patient (first visit) and Returning Patient (follow-up) reports.
"""

import io
from datetime import datetime


def generate_encounter_pdf(encounter_data: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable, KeepTogether, Paragraph,
        SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    # ── Palette ────────────────────────────────────────────────────────────────
    TEAL    = colors.HexColor("#0d9488")
    TEAL_LT = colors.HexColor("#ccfbf1")
    NAVY    = colors.HexColor("#0c1a2e")
    BLUE    = colors.HexColor("#0ea5e9")
    MUTED   = colors.HexColor("#64748b")
    URGENT  = colors.HexColor("#ef4444")
    URGENT_LT = colors.HexColor("#fee2e2")
    GREEN   = colors.HexColor("#10b981")
    GREEN_LT= colors.HexColor("#d1fae5")
    AMBER   = colors.HexColor("#f59e0b")
    AMBER_LT= colors.HexColor("#fef3c7")
    BORDER  = colors.HexColor("#e2e8f0")
    BG_LIGHT= colors.HexColor("#f8fafc")
    WHITE   = colors.white
    BLACK   = colors.black

    # ── Page geometry ──────────────────────────────────────────────────────────
    PAGE_W, PAGE_H = A4
    MARGIN = 18 * mm
    CONTENT_W = PAGE_W - 2 * MARGIN

    # ── Colour helpers ─────────────────────────────────────────────────────────
    def to_hex(c) -> str:
        r = int(round(c.red   * 255))
        g = int(round(c.green * 255))
        b = int(round(c.blue  * 255))
        return f"{r:02x}{g:02x}{b:02x}"

    def sev_colors(score: float):
        s = float(score or 5)
        if s <= 3:  return GREEN,  GREEN_LT,  "Mild"
        if s <= 6:  return AMBER,  AMBER_LT,  "Moderate"
        if s <= 8:  return URGENT, URGENT_LT, "Significant"
        return colors.HexColor("#dc2626"), colors.HexColor("#fecaca"), "Urgent"

    def prog_color(p: str):
        return {"improving": GREEN, "worsening": URGENT}.get(p, MUTED)

    # ── Data accessors ─────────────────────────────────────────────────────────
    def get(key, default="—"):
        v = encounter_data.get(key)
        return v if v not in (None, "", [], {}) else default

    # ── Styles ─────────────────────────────────────────────────────────────────
    def S(name, **kw):
        base = dict(fontName="Helvetica", fontSize=10, textColor=BLACK, leading=14)
        base.update(kw)
        return ParagraphStyle(name, **base)

    TITLE    = S("TITLE",  fontName="Helvetica-Bold", fontSize=20,
                 textColor=WHITE, leading=24)
    SUBTITLE = S("SUBTL",  fontSize=9, textColor=colors.HexColor("#b2f5ea"),
                 alignment=TA_RIGHT, leading=12)
    LABEL    = S("LBL",    fontName="Helvetica-Bold", fontSize=8,
                 textColor=MUTED, leading=10)
    VALUE    = S("VAL",    fontName="Helvetica-Bold", fontSize=10,
                 textColor=NAVY, leading=13)
    SEC_HDR  = S("SECHDR", fontName="Helvetica-Bold", fontSize=11,
                 textColor=TEAL, leading=14, spaceBefore=10, spaceAfter=2)
    DIAG_BIG = S("DIAGB",  fontName="Helvetica-Bold", fontSize=15,
                 textColor=NAVY, leading=18, spaceAfter=2)
    BODY     = S("BODY",   fontSize=10, textColor=colors.HexColor("#1e293b"),
                 leading=15, spaceAfter=2)
    BODY_J   = S("BODYJ",  fontSize=10, textColor=colors.HexColor("#1e293b"),
                 leading=15, alignment=TA_JUSTIFY, spaceAfter=2)
    SMALL    = S("SMALL",  fontSize=8,  textColor=MUTED, leading=11)
    DISC     = S("DISC",   fontSize=8,  textColor=MUTED, leading=12,
                 fontName="Helvetica-Oblique")
    CENT     = S("CENT",   fontSize=8,  textColor=MUTED, alignment=TA_CENTER,
                 leading=10)
    BULLET   = S("BULL",   fontSize=10, textColor=colors.HexColor("#1e293b"),
                 leading=15, leftIndent=10, spaceAfter=1)
    COMPLAINT= S("COMP",   fontSize=10, textColor=NAVY, leading=15,
                 leftIndent=8, fontName="Helvetica-Oblique")

    # ── Reusable flowables ─────────────────────────────────────────────────────
    def HR(color=BORDER, thickness=0.5, space_before=4, space_after=6):
        return HRFlowable(width="100%", thickness=thickness, color=color,
                          spaceBefore=space_before, spaceAfter=space_after)

    def section_header(number: int, title: str):
        """Numbered section header with teal background band."""
        hdr = Table(
            [[Paragraph(f"<b>{number}.  {title.upper()}</b>",
                        S("SH2", fontName="Helvetica-Bold", fontSize=10,
                          textColor=WHITE, leading=13))]],
            colWidths=[CONTENT_W])
        hdr.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), TEAL),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ]))
        return [Spacer(1, 8), hdr, Spacer(1, 6)]

    def render_paragraphs(text: str, style=BODY_J):
        """Split on double or single newlines; skip empty lines."""
        elems = []
        if not text or not text.strip():
            return elems
        # Normalise: collapse triple+ newlines to double
        import re
        text = re.sub(r'\n{3,}', '\n\n', text.strip())
        # Split on double newline first; if still one big block, split on single
        blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
        if len(blocks) == 1:
            blocks = [b.strip() for b in text.split('\n') if b.strip()]
        for block in blocks:
            elems.append(Paragraph(block, style))
            elems.append(Spacer(1, 3))
        return elems

    def render_treatment(text: str):
        """Smart treatment plan renderer: handles numbered lists and plain text."""
        import re
        elems = []
        if not text or not text.strip():
            return elems
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        numbered = re.compile(r'^(\d+[\.\)])\s*(.+)')
        bulleted = re.compile(r'^[•\-\*]\s*(.+)')
        for line in lines:
            m_num = numbered.match(line)
            m_bul = bulleted.match(line)
            if m_num:
                elems.append(Paragraph(
                    f"<b>{m_num.group(1)}</b>  {m_num.group(2)}", BULLET))
            elif m_bul:
                elems.append(Paragraph(f"•  {m_bul.group(1)}", BULLET))
            else:
                elems.append(Paragraph(f"•  {line}", BULLET))
            elems.append(Spacer(1, 2))
        return elems

    # ── Page footer callback ───────────────────────────────────────────────────
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, 10 * mm,
            "DermaAlpha AI Clinical Report  ·  Confidential  ·  "
            f"Generated {generated_at}")
        canvas.drawRightString(PAGE_W - MARGIN, 10 * mm,
            f"Page {doc.page}")
        canvas.restoreState()

    # ── Document ───────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=22 * mm,
        bottomMargin=22 * mm,
        title="DermaAlpha Clinical Report",
        author="DermaAlpha AI",
    )

    elems = []

    # ── ① Header banner ────────────────────────────────────────────────────────
    is_first = get("is_first_encounter", True)
    visit_label = "First Visit Report" if is_first else "Follow-up Visit Report"

    header_data = [[
        Paragraph("🔬  DermaAlpha", TITLE),
        Paragraph(
            f"AI Dermatology Assistant<br/><b>{visit_label}</b>",
            SUBTITLE),
    ]]
    header_tbl = Table(header_data,
                       colWidths=[CONTENT_W * 0.6, CONTENT_W * 0.4])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (0,0),   12),
        ("RIGHTPADDING",  (1,0), (1,0),   12),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("LINEBELOW",     (0,0), (-1,-1), 3, TEAL),
    ]))
    elems.append(header_tbl)
    elems.append(Spacer(1, 6))

    # ── ② Patient info grid ────────────────────────────────────────────────────
    pid    = get("patient_id")
    eid    = get("encounter_id")
    edate  = get("encounter_date")
    try:
        edate = datetime.strptime(edate, "%Y-%m-%d").strftime("%B %d, %Y")
    except Exception:
        pass

    def info_cell(label, value):
        return [Paragraph(label, LABEL), Paragraph(str(value), VALUE)]

    info_data = [
        [*info_cell("PATIENT ID", pid),   *info_cell("ENCOUNTER ID", eid)],
        [*info_cell("DATE",       edate), *info_cell("VISIT TYPE",   visit_label)],
    ]
    info_tbl = Table(info_data, colWidths=[28*mm, 60*mm, 28*mm, 54*mm])
    info_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), BG_LIGHT),
        ("BOX",           (0,0), (-1,-1), 0.5, BORDER),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    elems.append(info_tbl)
    elems.append(Spacer(1, 8))

    # ── ③ Clinical Assessment ──────────────────────────────────────────────────
    elems.extend(section_header(1, "Clinical Assessment"))

    diagnosis = get("diagnosis")
    severity  = float(encounter_data.get("severity_score") or 5)
    sev_col, sev_bg, sev_lbl = sev_colors(severity)
    body      = get("body_region", "unspecified").replace("_", " ").title()

    # Diagnosis name + severity pill side by side
    sev_pill = Table(
        [[Paragraph(
            f'<font color="#{to_hex(sev_col)}"><b>{severity:.1f} / 10</b></font><br/>'
            f'<font color="#{to_hex(sev_col)}" size="9">{sev_lbl}</font>',
            S("SEV", fontSize=14, alignment=TA_CENTER, leading=18))]],
        colWidths=[38*mm])
    sev_pill.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), sev_bg),
        ("BOX",           (0,0), (-1,-1), 1, sev_col),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("ROUNDEDCORNERS",[4]),
    ]))

    diag_row = Table(
        [[Paragraph(f"<b>{diagnosis}</b>", DIAG_BIG), sev_pill]],
        colWidths=[CONTENT_W - 44*mm, 44*mm])
    diag_row.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
    ]))
    elems.append(diag_row)
    elems.append(Spacer(1, 6))

    # Body region + condition tags
    tags = encounter_data.get("condition_tags") or []
    tags_str = "  ·  ".join(t.replace("_", " ").title() for t in tags) if tags else "—"
    elems.append(Paragraph(
        f'<b>Body Region:</b>  {body}     '
        f'<b>Condition Tags:</b>  {tags_str}', SMALL))
    elems.append(Spacer(1, 4))

    # Differential diagnoses
    diff = encounter_data.get("differential") or []
    if diff:
        diff_str = "   ·   ".join(diff[:4])
        elems.append(Paragraph(
            f'<b>Differential Diagnoses:</b>  {diff_str}', SMALL))
        elems.append(Spacer(1, 4))

    # Lesion classifier
    lc = encounter_data.get("lesion_classification")
    if lc and isinstance(lc, dict):
        preds = lc.get("predictions", [])
        if preds:
            preds_str = "   |   ".join(
                f"{p['label']} ({p['score']}%)" for p in preds[:3])
            elems.append(Paragraph(
                f'<b>AI Pre-Classifier:</b>  {preds_str}', SMALL))
            elems.append(Spacer(1, 4))

    elems.append(Spacer(1, 4))

    # ── ④ Chief Complaint ──────────────────────────────────────────────────────
    complaint = (encounter_data.get("chief_complaint") or "").strip()
    if complaint:
        elems.extend(section_header(2, "Chief Complaint"))
        elems.append(Paragraph(f'"{complaint}"', COMPLAINT))
        elems.append(Spacer(1, 4))

    # ── ⑤ Full Assessment (narrative) ─────────────────────────────────────────
    narrative = (encounter_data.get("narrative") or "").strip()
    if narrative:
        sec_num = 3 if complaint else 2
        elems.extend(section_header(sec_num, "Full Assessment"))
        elems.extend(render_paragraphs(narrative, BODY_J))
        elems.append(Spacer(1, 4))

    # ── ⑥ Treatment Plan ──────────────────────────────────────────────────────
    treatment = (encounter_data.get("treatment_plan") or "").strip()
    sec_num = (4 if complaint else 3)
    if treatment:
        elems.extend(section_header(sec_num, "Treatment Plan"))
        elems.extend(render_treatment(treatment))
        elems.append(Spacer(1, 4))
        sec_num += 1

    # ── ⑦ Medications ─────────────────────────────────────────────────────────
    medications = encounter_data.get("medications") or []
    if medications:
        elems.extend(section_header(sec_num, "Medications Mentioned"))
        # Render each med as a small pill-table row
        meds_cells = [[
            Paragraph(f"  {m.replace('_',' ').title()}  ",
                      S("MED", fontSize=9, textColor=TEAL,
                        fontName="Helvetica-Bold"))
        ] for m in medications]
        # Lay them out in a 3-column grid
        rows, row = [], []
        for i, cell in enumerate(meds_cells):
            row.append(cell[0])
            if (i + 1) % 3 == 0:
                rows.append(row); row = []
        if row:
            while len(row) < 3:
                row.append(Paragraph("", BODY))
            rows.append(row)
        if rows:
            med_tbl = Table(rows, colWidths=[CONTENT_W/3]*3)
            med_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), TEAL_LT),
                ("BOX",           (0,0), (-1,-1), 0.5, TEAL),
                ("INNERGRID",     (0,0), (-1,-1), 0.3, BORDER),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ]))
            elems.append(med_tbl)
        elems.append(Spacer(1, 4))
        sec_num += 1

    # ── ⑧ Progress Assessment (follow-up only) ────────────────────────────────
    progress_note = (encounter_data.get("progress_note") or "").strip()
    progress      = encounter_data.get("progress", "stable")
    prev_sev      = encounter_data.get("previous_severity")

    if not is_first and progress_note:
        elems.extend(section_header(sec_num, "Progress Assessment"))

        p_col    = prog_color(progress)
        p_bg     = {"improving": GREEN_LT, "worsening": URGENT_LT}.get(progress, BG_LIGHT)
        p_icon   = {"improving": "↑", "worsening": "↓", "stable": "→"}.get(progress, "→")
        p_label  = f"{p_icon}  {progress.upper()}"

        # Status + delta row
        status_parts = [
            Paragraph(
                f'<font color="#{to_hex(p_col)}"><b>{p_label}</b></font>',
                S("PS", fontSize=13)),
        ]
        if prev_sev is not None:
            delta     = severity - float(prev_sev)
            delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
            delta_col = URGENT if delta > 0 else (GREEN if delta < 0 else MUTED)
            status_parts.append(Paragraph(
                f'Severity:  <b>{float(prev_sev):.1f}</b>  →  <b>{severity:.1f}</b>  '
                f'<font color="#{to_hex(delta_col)}">({delta_str})</font>',
                S("SD", fontSize=10, textColor=NAVY)))

        status_tbl = Table([[p] for p in status_parts],
                           colWidths=[CONTENT_W])
        status_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), p_bg),
            ("BOX",           (0,0), (-1,-1), 0.5, p_col),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        elems.append(status_tbl)
        elems.append(Spacer(1, 6))
        elems.extend(render_paragraphs(progress_note, BODY_J))
        elems.append(Spacer(1, 4))

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    elems.append(Spacer(1, 10))
    elems.append(HR(color=BORDER, thickness=0.5))
    elems.append(Paragraph(
        "⚕  Medical Disclaimer:  This report is generated by DermaAlpha AI for "
        "informational purposes only. It does not constitute a medical diagnosis "
        "or replace professional medical advice. Always consult a licensed "
        "dermatologist or healthcare provider regarding any skin condition.",
        DISC))

    # ── Build ──────────────────────────────────────────────────────────────────
    doc.build(elems, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buf.getvalue()
