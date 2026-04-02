"""
agent/patient_session.py — Session state container for one patient interaction.
Acts as the data carrier passed between state machine stages in DermaAgent.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from agent.medical_analyzer import AnalysisResult
from agent.memory_manager import EncounterRecord


class SessionState(str, Enum):
    """States in the DermaAgent conversation flow."""
    GREETING              = "GREETING"
    COMPLAINT_INTAKE      = "COMPLAINT_INTAKE"
    HISTORY_INTAKE        = "HISTORY_INTAKE"       # First encounter only
    IMAGE_INTAKE          = "IMAGE_INTAKE"
    FIRST_ENCOUNTER_CHECK = "FIRST_ENCOUNTER_CHECK"
    HISTORY_DISPLAY       = "HISTORY_DISPLAY"       # Follow-up only
    ANALYSIS              = "ANALYSIS"
    PROGRESS_COMPARE      = "PROGRESS_COMPARE"      # Follow-up only
    RESULT_DISPLAY        = "RESULT_DISPLAY"
    SAVE_ENCOUNTER        = "SAVE_ENCOUNTER"
    FOLLOW_UP_PROMPT      = "FOLLOW_UP_PROMPT"
    DONE                  = "DONE"


@dataclass
class PatientSession:
    """
    Holds all mutable state for a single CLI session.
    Created fresh at the start of each run() call in DermaAgent.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    patient_id: str = ""
    is_new_patient: bool = True
    is_first_encounter: bool = True

    # ── Current encounter inputs ──────────────────────────────────────────────
    chief_complaint: str = ""
    medical_history: str = ""           # Collected on first encounter only
    image_path: Optional[str] = None
    image_b64: Optional[str] = None     # Base64 string (without data URI prefix)
    image_data_uri: Optional[str] = None  # Full "data:image/jpeg;base64,..." URI

    # ── Retrieved context (follow-up) ─────────────────────────────────────────
    past_encounters: list[EncounterRecord] = field(default_factory=list)
    latest_encounter: Optional[EncounterRecord] = None

    # ── Current encounter outputs ─────────────────────────────────────────────
    current_analysis: Optional[AnalysisResult] = None
    progress_note: str = ""
    current_encounter_record: Optional[EncounterRecord] = None

    # ── FSM control ───────────────────────────────────────────────────────────
    state: SessionState = SessionState.GREETING
    continue_session: bool = True       # Set to False to exit after DONE

    @property
    def has_image(self) -> bool:
        return self.image_data_uri is not None

    @property
    def display_patient_id(self) -> str:
        return self.patient_id if self.patient_id else "Unknown"
