"""
agent/memory_manager.py — Pinecone vector database operations.
Stores and retrieves patient encounter records. Pure data layer — no LLM calls.
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional

from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from config import Settings

logger = logging.getLogger(__name__)

# ── Drug/medication keyword extractor ─────────────────────────────────────────
_DRUG_RE = re.compile(
    r"\b("
    # Corticosteroids (topical/systemic)
    r"hydrocortisone|betamethasone|triamcinolone|clobetasol|fluocinonide|"
    r"mometasone|desonide|fluocinolone|halcinonide|desoximetasone|prednisone|prednisolone|"
    # Antifungals
    r"clotrimazole|miconazole|terbinafine|ketoconazole|fluconazole|nystatin|"
    r"ciclopirox|econazole|oxiconazole|itraconazole|griseofulvin|"
    # Antibiotics (topical/oral)
    r"doxycycline|minocycline|erythromycin|clindamycin|tetracycline|"
    r"metronidazole|mupirocin|azithromycin|amoxicillin|cephalexin|dicloxacillin|"
    r"trimethoprim|sulfamethoxazole|ciprofloxacin|"
    # Antivirals
    r"acyclovir|valacyclovir|famciclovir|"
    # Retinoids
    r"tretinoin|adapalene|isotretinoin|tazarotene|retinol|"
    # Antihistamines
    r"cetirizine|loratadine|fexofenadine|hydroxyzine|diphenhydramine|levocetirizine|"
    # Immunomodulators / biologics
    r"tacrolimus|pimecrolimus|dupilumab|secukinumab|ustekinumab|apremilast|"
    # Keratolytics / other topicals
    r"benzoyl peroxide|salicylic acid|azelaic acid|coal tar|"
    r"permethrin|ivermectin|calcipotriol|anthralin|zinc oxide|calamine"
    r")\b",
    re.IGNORECASE,
)

# How long to wait (seconds) for a new Pinecone index to become ready
INDEX_READY_TIMEOUT = 120
INDEX_READY_POLL_INTERVAL = 5


# ─────────────────────────────────────────────────────────────────────────────
# Data model for a single patient encounter
# ─────────────────────────────────────────────────────────────────────────────

class EncounterRecord(BaseModel):
    """Represents one patient visit, stored as a vector in Pinecone."""

    patient_id: str
    encounter_id: str
    encounter_date: str                     # ISO-8601: "2026-03-29"
    encounter_timestamp: int                # Unix epoch seconds (for sorting)
    chief_complaint: str
    body_region: str = "unspecified"
    condition_tags: list[str] = Field(default_factory=list)
    severity_score: float = 5.0
    has_image: bool = False
    diagnosis_summary: str = ""
    treatment_plan: str = ""
    drug_tags: list[str] = Field(default_factory=list)   # medications extracted from treatment
    image_vector: Optional[list[float]] = None           # BioMedCLIP 512-dim visual fingerprint
    is_first_encounter: bool = True
    follow_up_of: Optional[str] = None     # encounter_id of the prior visit

    def to_pinecone_metadata(self) -> dict:
        """Convert to a flat dict suitable for Pinecone metadata storage."""
        return {
            "patient_id": self.patient_id,
            "encounter_id": self.encounter_id,
            "encounter_date": self.encounter_date,
            "encounter_timestamp": self.encounter_timestamp,
            "chief_complaint": self.chief_complaint[:500],
            "body_region": self.body_region,
            "condition_tags": self.condition_tags,
            "severity_score": self.severity_score,
            "has_image": self.has_image,
            "has_image_vector": self.image_vector is not None,
            "diagnosis_summary": self.diagnosis_summary[:1000],
            "treatment_plan": self.treatment_plan[:500],
            "drug_tags": self.drug_tags,
            "is_first_encounter": self.is_first_encounter,
            "follow_up_of": self.follow_up_of or "",
        }

    @classmethod
    def from_pinecone_metadata(cls, metadata: dict) -> "EncounterRecord":
        """Reconstruct an EncounterRecord from Pinecone metadata."""
        return cls(
            patient_id=metadata.get("patient_id", ""),
            encounter_id=metadata.get("encounter_id", ""),
            encounter_date=metadata.get("encounter_date", ""),
            encounter_timestamp=int(metadata.get("encounter_timestamp", 0)),
            chief_complaint=metadata.get("chief_complaint", ""),
            body_region=metadata.get("body_region", "unspecified"),
            condition_tags=metadata.get("condition_tags", []),
            severity_score=float(metadata.get("severity_score", 5.0)),
            has_image=bool(metadata.get("has_image", False)),
            diagnosis_summary=metadata.get("diagnosis_summary", ""),
            treatment_plan=metadata.get("treatment_plan", ""),
            drug_tags=metadata.get("drug_tags", []),
            image_vector=None,   # loaded separately via get_image_vector()
            is_first_encounter=bool(metadata.get("is_first_encounter", True)),
            follow_up_of=metadata.get("follow_up_of") or None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Memory manager
# ─────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Manages all interactions with the Pinecone vector store.
    Uses sentence-transformers for generating embeddings locally (CPU-friendly).
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        logger.info("Loading embedding model: %s", settings.embedding_model)
        self.embedder = SentenceTransformer(settings.embedding_model)

        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self._get_or_create_index(pc, settings)
        logger.info("Pinecone index '%s' is ready.", settings.pinecone_index_name)

    # ── Index management ──────────────────────────────────────────────────────

    def _get_or_create_index(self, pc: Pinecone, settings: Settings):
        """Return the Pinecone index, creating it if it doesn't exist."""
        existing_indexes = [idx.name for idx in pc.list_indexes()]

        if settings.pinecone_index_name not in existing_indexes:
            logger.info(
                "Creating new Pinecone index '%s' (dim=%d, metric=cosine)...",
                settings.pinecone_index_name,
                settings.embedding_dim,
            )
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )
            self._wait_for_index_ready(pc, settings.pinecone_index_name)

        return pc.Index(settings.pinecone_index_name)

    def _wait_for_index_ready(self, pc: Pinecone, index_name: str) -> None:
        """Poll until the index reports ready status."""
        deadline = time.time() + INDEX_READY_TIMEOUT
        while time.time() < deadline:
            desc = pc.describe_index(index_name)
            if desc.status.get("ready", False):
                return
            logger.debug("Index not ready yet, waiting %ds...", INDEX_READY_POLL_INTERVAL)
            time.sleep(INDEX_READY_POLL_INTERVAL)
        raise TimeoutError(
            f"Pinecone index '{index_name}' did not become ready within "
            f"{INDEX_READY_TIMEOUT} seconds."
        )

    # ── Drug extraction ───────────────────────────────────────────────────────

    @staticmethod
    def extract_drug_tags(treatment_text: str) -> list[str]:
        """
        Scan a treatment plan for known dermatology drug names.
        Returns a sorted, deduplicated list (max 10 entries).
        """
        if not treatment_text:
            return []
        matches = {m.group(0).lower() for m in _DRUG_RE.finditer(treatment_text)}
        return sorted(matches)[:10]

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """Encode text using the biomedical embedding model (S-PubMedBert-MS-MARCO, 768-dim)."""
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    def _encounter_to_text(self, encounter: EncounterRecord) -> str:
        """Build the text string used to embed an encounter for semantic search."""
        tags = " ".join(encounter.condition_tags)
        return (
            f"{encounter.chief_complaint} "
            f"{encounter.diagnosis_summary} "
            f"{encounter.body_region} "
            f"{tags}"
        ).strip()

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_encounter(self, encounter: EncounterRecord) -> None:
        """Embed and store an encounter record in Pinecone."""
        text = self._encounter_to_text(encounter)
        embedding = self.embed_text(text)
        vector_id = f"{encounter.patient_id}#{encounter.encounter_id}"
        metadata = encounter.to_pinecone_metadata()

        vectors = [{"id": vector_id, "values": embedding, "metadata": metadata}]

        # Store image vector as a separate entry if available
        if encounter.image_vector is not None:
            img_vector_id = f"{vector_id}#img"
            img_metadata = {
                "patient_id":     encounter.patient_id,
                "encounter_id":   encounter.encounter_id,
                "encounter_date": encounter.encounter_date,
                "record_type":    "image_vector",
            }
            # Pad or trim to match index dimension (text dim=768, img dim=512)
            # We use a zero-padded vector so it lives in the same index
            img_vec_padded = encounter.image_vector + [0.0] * (
                self.settings.embedding_dim - len(encounter.image_vector)
            )
            vectors.append({
                "id":       img_vector_id,
                "values":   img_vec_padded,
                "metadata": img_metadata,
            })

        self.index.upsert(vectors=vectors)
        logger.info(
            "Upserted encounter %s for patient %s (image_vector=%s)",
            encounter.encounter_id,
            encounter.patient_id,
            encounter.image_vector is not None,
        )

    def get_image_vector(
        self,
        patient_id: str,
        encounter_id: str,
    ) -> Optional[list[float]]:
        """
        Retrieve the stored BioMedCLIP image vector for a specific encounter.
        Returns the original 512-dim slice (strips zero-padding).
        Returns None if no image was stored for that encounter.
        """
        vector_id = f"{patient_id}#{encounter_id}#img"
        try:
            result = self.index.fetch(ids=[vector_id])
            if vector_id not in result.vectors:
                return None
            full_vec = result.vectors[vector_id].values
            # Return only the first 512 dims (strip zero-padding)
            return list(full_vec[:512])
        except Exception as e:
            logger.warning("Failed to fetch image vector for %s: %s", encounter_id, e)
            return None

    # ── Query ─────────────────────────────────────────────────────────────────

    def patient_exists(self, patient_id: str) -> bool:
        """Return True if the patient has at least one encounter on record."""
        zero_vector = [0.0] * self.settings.embedding_dim
        results = self.index.query(
            vector=zero_vector,
            top_k=1,
            filter={"patient_id": {"$eq": patient_id}},
            include_metadata=False,
        )
        return len(results.matches) > 0

    def get_patient_history(
        self,
        patient_id: str,
        current_complaint: str,
        top_k: Optional[int] = None,
    ) -> list[EncounterRecord]:
        """
        Retrieve the most semantically relevant past encounters for a patient.
        Results are sorted by recency (newest first).

        Args:
            patient_id:        The patient's stable identifier.
            current_complaint: The patient's current complaint text (used for similarity).
            top_k:             Max encounters to return. Defaults to settings value.

        Returns:
            List of EncounterRecord objects, sorted by encounter_timestamp descending.
        """
        if top_k is None:
            top_k = self.settings.max_retrieved_encounters

        query_embedding = self.embed_text(current_complaint)

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter={"patient_id": {"$eq": patient_id}},
            include_metadata=True,
        )

        encounters = []
        for match in results.matches:
            if match.score >= self.settings.similarity_threshold:
                enc = EncounterRecord.from_pinecone_metadata(match.metadata)
                encounters.append(enc)

        # Sort by recency (newest first)
        encounters.sort(key=lambda e: e.encounter_timestamp, reverse=True)
        return encounters

    def get_latest_encounter(self, patient_id: str) -> Optional[EncounterRecord]:
        """
        Return the single most recent encounter for a patient, regardless of
        similarity to the current complaint.
        """
        # Fetch a generous batch sorted by recency
        zero_vector = [0.0] * self.settings.embedding_dim
        results = self.index.query(
            vector=zero_vector,
            top_k=50,
            filter={"patient_id": {"$eq": patient_id}},
            include_metadata=True,
        )

        if not results.matches:
            return None

        encounters = [
            EncounterRecord.from_pinecone_metadata(m.metadata)
            for m in results.matches
        ]
        encounters.sort(key=lambda e: e.encounter_timestamp, reverse=True)
        return encounters[0]

    def get_all_encounters(self, patient_id: str) -> list[EncounterRecord]:
        """Return all encounters for a patient, sorted by recency."""
        return self.get_patient_history(
            patient_id,
            current_complaint="",
            top_k=100,
        )

    def get_encounter_by_id(
        self,
        patient_id: str,
        encounter_id: str,
    ) -> Optional[EncounterRecord]:
        """Fetch a single encounter directly by its vector ID."""
        vector_id = f"{patient_id}#{encounter_id}"
        try:
            result = self.index.fetch(ids=[vector_id])
            vectors = result.vectors
            if vector_id not in vectors:
                return None
            return EncounterRecord.from_pinecone_metadata(vectors[vector_id].metadata)
        except Exception as e:
            logger.error("Failed to fetch encounter %s: %s", encounter_id, e)
            return None

    def get_medication_history(self, patient_id: str) -> list[str]:
        """
        Return a deduplicated list of all drug names ever prescribed to a patient,
        sorted alphabetically.
        """
        encounters = self.get_all_encounters(patient_id)
        all_drugs: set[str] = set()
        for enc in encounters:
            all_drugs.update(enc.drug_tags)
        return sorted(all_drugs)

    def get_condition_chain(
        self,
        patient_id: str,
        condition_tags: list[str],
        top_k: int = 100,
    ) -> list[EncounterRecord]:
        """
        Return all encounters for a patient that share at least one condition tag
        with the provided tags, sorted chronologically oldest-first.

        Used for temporal trend analysis — gives the LLM a full progression
        of a specific condition over time rather than just the nearest encounter.
        Unlike get_patient_history(), this does NOT apply similarity_threshold
        so early encounters with slightly different wording are never excluded.
        """
        zero_vector = [0.0] * self.settings.embedding_dim
        results = self.index.query(
            vector=zero_vector,
            top_k=top_k,
            filter={"patient_id": {"$eq": patient_id}},
            include_metadata=True,
        )

        if not results.matches:
            return []

        current_tag_set = set(t.lower() for t in condition_tags)
        chain = []
        for match in results.matches:
            enc = EncounterRecord.from_pinecone_metadata(match.metadata)
            past_tag_set = set(t.lower() for t in enc.condition_tags)
            if current_tag_set & past_tag_set:      # at least one tag in common
                chain.append(enc)

        # Oldest first — so the LLM reads the progression chronologically
        chain.sort(key=lambda e: e.encounter_timestamp)
        return chain
