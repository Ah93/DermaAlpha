"""
agent/derma_agent.py — Main DermaAlpha agent orchestration.
Drives the PatientSession through the state machine, coordinating
MedicalAnalyzer (LLM) and MemoryManager (Pinecone).
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich import print as rprint

from agent.medical_analyzer import AnalysisResult, MedicalAnalyzer
from agent.memory_manager import EncounterRecord, MemoryManager
from agent.patient_session import PatientSession, SessionState
from agent.prompts import (
    MEDICAL_HISTORY_REQUEST,
    build_first_encounter_prompt,
    build_follow_up_prompt,
    build_progress_comparison_prompt,
    compute_clinical_diff,
)
from config import Settings
from utils.image_utils import load_and_encode_image, validate_image_path

logger = logging.getLogger(__name__)


class DermaAgent:
    """
    Orchestrates the full patient consultation flow.
    Call run() to start an interactive CLI session.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.console = Console()

        self.console.print(
            "[bold cyan]Initialising DermaAlpha...[/bold cyan]"
        )
        self.analyzer = MedicalAnalyzer(settings)
        self.memory = MemoryManager(settings)
        self.console.print("[bold green]✓ Ready.[/bold green]\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start and run a complete patient consultation session."""
        session = PatientSession()

        while session.state != SessionState.DONE:
            state = session.state
            if state == SessionState.GREETING:
                self._greeting(session)
            elif state == SessionState.COMPLAINT_INTAKE:
                self._complaint_intake(session)
            elif state == SessionState.HISTORY_INTAKE:
                self._history_intake(session)
            elif state == SessionState.IMAGE_INTAKE:
                self._image_intake(session)
            elif state == SessionState.FIRST_ENCOUNTER_CHECK:
                self._check_history(session)
            elif state == SessionState.HISTORY_DISPLAY:
                self._display_history(session)
            elif state == SessionState.ANALYSIS:
                self._run_analysis(session)
            elif state == SessionState.PROGRESS_COMPARE:
                self._compare_progress(session)
            elif state == SessionState.RESULT_DISPLAY:
                self._display_results(session)
            elif state == SessionState.SAVE_ENCOUNTER:
                self._save_encounter(session)
            elif state == SessionState.FOLLOW_UP_PROMPT:
                self._follow_up_prompt(session)

    # ─────────────────────────────────────────────────────────────────────────
    # State handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _greeting(self, session: PatientSession) -> None:
        """Welcome the patient and collect/assign a patient ID."""
        self.console.print(
            Panel(
                "[bold white]Welcome to [cyan]DermaAlpha[/cyan] — AI Dermatology Assistant[/bold white]\n\n"
                "This assistant analyzes skin conditions, tracks your medical history,\n"
                "and monitors your treatment progress over time.\n\n"
                "[dim]Note: DermaAlpha does not replace a licensed dermatologist.[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        self.console.print(
            "\n[bold]Patient Identification[/bold]\n"
            "Enter your Patient ID if you have one, or type [green]new[/green] to register.\n"
        )

        raw_id = Prompt.ask("[bold cyan]Patient ID[/bold cyan]").strip()

        if raw_id.lower() in {"new", "n", ""}:
            patient_id = f"PT-{uuid.uuid4().hex[:6].upper()}"
            session.is_new_patient = True
            self.console.print(
                f"\n[bold green]New patient registered![/bold green]\n"
                f"Your Patient ID is: [bold yellow]{patient_id}[/bold yellow]\n"
                f"[dim]Please save this ID — you will need it for follow-up visits.[/dim]\n"
            )
        else:
            patient_id = raw_id.upper()
            session.is_new_patient = False
            self.console.print(
                f"\n[bold]Welcome back![/bold] Patient ID: [bold yellow]{patient_id}[/bold yellow]\n"
            )

        session.patient_id = patient_id
        session.state = SessionState.COMPLAINT_INTAKE

    def _complaint_intake(self, session: PatientSession) -> None:
        """Collect the patient's current chief complaint."""
        self.console.print(Rule("[bold]Current Complaint[/bold]", style="cyan"))
        self.console.print(
            "\nPlease describe your skin concern in detail.\n"
            "Include: location, duration, symptoms (itching, pain, scaling), "
            "and any recent changes.\n"
        )

        complaint = Prompt.ask("[bold cyan]Describe your condition[/bold cyan]").strip()

        while not complaint:
            self.console.print("[yellow]Please describe your condition to continue.[/yellow]")
            complaint = Prompt.ask("[bold cyan]Describe your condition[/bold cyan]").strip()

        session.chief_complaint = complaint
        session.state = SessionState.IMAGE_INTAKE

    def _history_intake(self, session: PatientSession) -> None:
        """Collect patient's medical history (first encounter only)."""
        self.console.print(Rule("[bold]Medical History[/bold]", style="cyan"))
        self.console.print(Markdown(MEDICAL_HISTORY_REQUEST))

        history = Prompt.ask(
            "[bold cyan]Your medical history[/bold cyan]",
            default="skip",
        ).strip()

        if history.lower() == "skip":
            session.medical_history = "Not provided."
        else:
            session.medical_history = history

        session.state = SessionState.FIRST_ENCOUNTER_CHECK

    def _image_intake(self, session: PatientSession) -> None:
        """Optionally collect an image of the affected area."""
        self.console.print(Rule("[bold]Image Upload[/bold]", style="cyan"))
        self.console.print(
            "\nProvide the full path to an image of the affected area, "
            "or press [green]Enter[/green] to skip.\n"
            f"Supported formats: JPG, PNG, WEBP, BMP (max 15 MB)\n"
        )

        image_path = Prompt.ask(
            "[bold cyan]Image path[/bold cyan]",
            default="",
        ).strip().strip('"').strip("'")

        if image_path:
            is_valid, error_msg = validate_image_path(image_path)
            if not is_valid:
                self.console.print(f"[yellow]⚠ Image skipped: {error_msg}[/yellow]")
            else:
                try:
                    with self.console.status("[cyan]Processing image...[/cyan]"):
                        b64, data_uri = load_and_encode_image(
                            image_path,
                            max_dim=self.settings.max_image_dim,
                        )
                    session.image_path = image_path
                    session.image_b64 = b64
                    session.image_data_uri = data_uri
                    self.console.print("[green]✓ Image loaded successfully.[/green]")
                except Exception as e:
                    self.console.print(f"[yellow]⚠ Could not load image: {e}[/yellow]")

        # If first encounter, go collect medical history before checking Pinecone
        if session.is_new_patient or not session.patient_id:
            session.state = SessionState.HISTORY_INTAKE
        else:
            session.state = SessionState.FIRST_ENCOUNTER_CHECK

    def _check_history(self, session: PatientSession) -> None:
        """Query Pinecone to determine if this is a first encounter or follow-up."""
        with self.console.status(
            "[cyan]Checking patient history...[/cyan]", spinner="dots"
        ):
            exists = self.memory.patient_exists(session.patient_id)

            if exists:
                past = self.memory.get_patient_history(
                    patient_id=session.patient_id,
                    current_complaint=session.chief_complaint,
                    top_k=self.settings.max_retrieved_encounters,
                )
                latest = self.memory.get_latest_encounter(session.patient_id)
                session.past_encounters = past
                session.latest_encounter = latest
                session.is_first_encounter = False
            else:
                session.is_first_encounter = True

        if session.is_first_encounter:
            self.console.print(
                "[dim]No prior history found — treating as first encounter.[/dim]\n"
            )
            # If medical history wasn't collected yet (returning patient ID with no records)
            if not session.medical_history:
                session.state = SessionState.HISTORY_INTAKE
            else:
                session.state = SessionState.ANALYSIS
        else:
            n = len(session.past_encounters)
            self.console.print(
                f"[dim]Found {n} relevant past encounter(s) — loading history...[/dim]\n"
            )
            session.state = SessionState.HISTORY_DISPLAY

    def _display_history(self, session: PatientSession) -> None:
        """Show a summary of retrieved past encounters before analysis."""
        self.console.print(Rule("[bold]Past Encounter Summary[/bold]", style="yellow"))

        for i, enc in enumerate(session.past_encounters, start=1):
            tags = ", ".join(enc.condition_tags) if enc.condition_tags else "N/A"
            icon = "📸" if enc.has_image else "📝"
            self.console.print(
                Panel(
                    f"[bold]Date:[/bold] {enc.encounter_date}\n"
                    f"[bold]Diagnosis:[/bold] {enc.diagnosis_summary[:200]}\n"
                    f"[bold]Severity:[/bold] {enc.severity_score:.1f}/10\n"
                    f"[bold]Body Region:[/bold] {enc.body_region}\n"
                    f"[bold]Conditions:[/bold] {tags}\n"
                    f"[bold]Treatment:[/bold] {enc.treatment_plan[:150] or 'Not documented'}\n"
                    f"[bold]Chief Complaint:[/bold] {enc.chief_complaint[:150]}",
                    title=f"{icon} Encounter {i}  |  ID: {enc.encounter_id}",
                    border_style="yellow",
                    padding=(0, 1),
                )
            )

        self.console.print()
        session.state = SessionState.ANALYSIS

    def _run_analysis(self, session: PatientSession) -> None:
        """Call MedGemma to analyze the current condition."""
        self.console.print(Rule("[bold]Analyzing Condition[/bold]", style="cyan"))

        if session.is_first_encounter:
            prompt = build_first_encounter_prompt(
                complaint=session.chief_complaint,
                medical_history=session.medical_history,
                has_image=session.has_image,
            )
        else:
            prompt = build_follow_up_prompt(
                complaint=session.chief_complaint,
                past_encounters=session.past_encounters,
                has_image=session.has_image,
            )

        with self.console.status(
            "[cyan]MedGemma is analyzing your condition...[/cyan]", spinner="dots"
        ):
            analysis = self.analyzer.analyze(
                prompt=prompt,
                image_data_uri=session.image_data_uri,
            )

        session.current_analysis = analysis
        logger.info(
            "Analysis complete: %s | severity=%.1f | progress=%s",
            analysis.diagnosis,
            analysis.severity_score,
            analysis.progress,
        )

        if not session.is_first_encounter and session.latest_encounter:
            session.state = SessionState.PROGRESS_COMPARE
        else:
            session.state = SessionState.RESULT_DISPLAY

    def _compare_progress(self, session: PatientSession) -> None:
        """Generate a structured progress note using clinical diff + temporal chain."""
        if not session.latest_encounter or not session.current_analysis:
            session.state = SessionState.RESULT_DISPLAY
            return

        with self.console.status(
            "[cyan]Comparing with previous encounter...[/cyan]", spinner="dots"
        ):
            # Temporal chain — all encounters sharing tags with current condition
            condition_chain = self.memory.get_condition_chain(
                patient_id=session.patient_id,
                condition_tags=session.current_analysis.condition_tags,
            )

            # Pre-compute structured diff before LLM call
            diff = compute_clinical_diff(
                previous_encounter=session.latest_encounter,
                current_analysis=session.current_analysis,
                encounter_count=len(condition_chain) if condition_chain else len(session.past_encounters),
            )

            progress_prompt = build_progress_comparison_prompt(
                previous_encounter=session.latest_encounter,
                current_analysis=session.current_analysis,
                clinical_diff=diff,
            )
            progress_result = self.analyzer.analyze(
                prompt=progress_prompt,
                image_data_uri=None,
                max_new_tokens=500,
            )

        session.progress_note = progress_result.raw_text
        session.state = SessionState.RESULT_DISPLAY

    def _display_results(self, session: PatientSession) -> None:
        """Present the diagnosis and recommendations to the patient."""
        analysis = session.current_analysis
        if not analysis:
            session.state = SessionState.SAVE_ENCOUNTER
            return

        self.console.print(Rule("[bold]Diagnosis & Assessment[/bold]", style="green"))

        # Urgent warning banner
        if analysis.urgent:
            self.console.print(
                Panel(
                    "[bold red]⚠ URGENT: This condition may require immediate medical attention.[/bold red]\n"
                    "Please visit an emergency department or contact your doctor right away.",
                    border_style="red",
                    padding=(0, 1),
                )
            )

        # Severity badge
        severity = analysis.severity_score
        if severity <= 3:
            sev_color, sev_label = "green", "Mild"
        elif severity <= 6:
            sev_color, sev_label = "yellow", "Moderate"
        else:
            sev_color, sev_label = "red", "Significant"

        # Main narrative
        self.console.print(
            Panel(
                Markdown(analysis.raw_text.split("```json")[0].strip()),
                title=f"[bold green]Analysis for {session.display_patient_id}[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

        # Structured summary
        tags_str = "  ".join(f"[dim]{t}[/dim]" for t in analysis.condition_tags) or "[dim]None[/dim]"
        diff_str = "\n".join(f"  • {d}" for d in analysis.differential) or "  None"

        self.console.print(
            Panel(
                f"[bold]Primary Diagnosis:[/bold] {analysis.diagnosis}\n"
                f"[bold]Differential:[/bold]\n{diff_str}\n"
                f"[bold]Severity:[/bold] [{sev_color}]{severity:.1f}/10 ({sev_label})[/{sev_color}]\n"
                f"[bold]Body Region:[/bold] {analysis.body_region}\n"
                f"[bold]Condition Tags:[/bold] {tags_str}\n"
                f"[bold]Treatment Plan:[/bold] {analysis.treatment_plan or 'See narrative above'}",
                title="[bold]Clinical Summary[/bold]",
                border_style="blue",
                padding=(0, 1),
            )
        )

        # Progress note (follow-up only)
        if session.progress_note and not session.is_first_encounter:
            progress_icon = {
                "improving": "✅",
                "worsening": "⚠",
                "stable": "➡",
            }.get(analysis.progress, "ℹ")

            self.console.print(
                Panel(
                    Markdown(session.progress_note.split("```json")[0].strip()),
                    title=f"{progress_icon} [bold]Progress Since Last Visit[/bold]",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )

        session.state = SessionState.SAVE_ENCOUNTER

    def _save_encounter(self, session: PatientSession) -> None:
        """Build and persist the encounter record to Pinecone."""
        analysis = session.current_analysis
        if not analysis:
            session.state = SessionState.FOLLOW_UP_PROMPT
            return

        encounter_id = f"ENC-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now()

        record = EncounterRecord(
            patient_id=session.patient_id,
            encounter_id=encounter_id,
            encounter_date=now.strftime("%Y-%m-%d"),
            encounter_timestamp=int(now.timestamp()),
            chief_complaint=session.chief_complaint,
            body_region=analysis.body_region,
            condition_tags=analysis.condition_tags,
            severity_score=analysis.severity_score,
            has_image=session.has_image,
            diagnosis_summary=analysis.diagnosis + ". " + analysis.raw_text[:600],
            treatment_plan=analysis.treatment_plan,
            is_first_encounter=session.is_first_encounter,
            follow_up_of=(
                session.latest_encounter.encounter_id
                if session.latest_encounter else None
            ),
        )

        with self.console.status(
            "[cyan]Saving encounter to memory...[/cyan]", spinner="dots"
        ):
            self.memory.upsert_encounter(record)

        session.current_encounter_record = record
        self.console.print(
            f"[dim]✓ Encounter [bold]{encounter_id}[/bold] saved to memory.[/dim]\n"
        )
        session.state = SessionState.FOLLOW_UP_PROMPT

    def _follow_up_prompt(self, session: PatientSession) -> None:
        """Ask if the patient wants to start another consultation."""
        another = Confirm.ask(
            "\n[bold cyan]Would you like to start a new consultation?[/bold cyan]",
            default=False,
        )

        if another:
            # Reset for a new consultation, keeping patient_id and is_new_patient=False
            pid = session.patient_id
            session.__init__()              # type: ignore[misc]
            session.patient_id = pid
            session.is_new_patient = False
            session.state = SessionState.COMPLAINT_INTAKE
        else:
            self.console.print(
                "\n[bold cyan]Thank you for using DermaAlpha. Take care![/bold cyan]\n"
            )
            session.state = SessionState.DONE
