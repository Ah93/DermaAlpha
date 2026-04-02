"""
agent/visual_embedder.py — BioMedCLIP visual embedding for skin lesion images.

Uses Microsoft's BiomedCLIP model trained on 15M biomedical image-text pairs
from PubMed to produce a 512-dim visual fingerprint for each uploaded image.

These fingerprints are stored in Pinecone alongside text embeddings.
On return visits, cosine similarity between fingerprints measures objective
visual change — independent of what the LLM or patient says.
"""

import base64
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# BioMedCLIP model identifier on HuggingFace / OpenCLIP hub
_BIOMEDCLIP_MODEL  = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_L_14"
_BIOMEDCLIP_DIM    = 512   # output embedding dimension

# Thresholds for interpreting cosine similarity
# (1.0 = identical, 0.0 = completely different)
SIMILARITY_HIGH   = 0.80   # visually similar   → likely same lesion, little change
SIMILARITY_MEDIUM = 0.55   # moderate change    → noticeable difference
# below 0.55 = HIGH visual change

# Contradiction: improvement in severity but high visual change
CONTRADICTION_SEVERITY_IMPROVEMENT = 0.5   # severity must drop by this fraction
CONTRADICTION_VISUAL_CHANGE        = 0.45  # visual change must exceed this


class VisualEmbedder:
    """
    Wraps BioMedCLIP to generate 512-dim visual fingerprints for skin images.
    Constructor never raises — sets _ready=False and logs a warning on failure.
    """

    def __init__(self):
        self._model     = None
        self._preprocess = None
        self._ready     = False
        self._load()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load(self):
        try:
            import open_clip
            import torch

            logger.info("Loading BioMedCLIP visual embedder…")
            model, _, preprocess = open_clip.create_model_and_transforms(
                _BIOMEDCLIP_MODEL
            )
            model.eval()

            self._model      = model
            self._preprocess = preprocess
            self._torch      = torch
            self._ready      = True
            logger.info("BioMedCLIP ready (dim=%d).", _BIOMEDCLIP_DIM)

        except Exception as e:
            logger.warning(
                "BioMedCLIP failed to load (%s). "
                "Visual similarity comparison will be skipped.",
                e,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def dim(self) -> int:
        return _BIOMEDCLIP_DIM

    def embed_image(self, image_data_uri: Optional[str]) -> Optional[list[float]]:
        """
        Convert a base64 image data URI into a 512-dim visual fingerprint.

        Returns:
            list of 512 floats (L2-normalised), or None if unavailable.
        """
        if not self._ready or not image_data_uri:
            return None

        try:
            from PIL import Image

            b64 = (
                image_data_uri.split(",", 1)[1]
                if "," in image_data_uri
                else image_data_uri
            )
            image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

            tensor = self._preprocess(image).unsqueeze(0)  # (1, C, H, W)

            with self._torch.no_grad():
                features = self._model.encode_image(tensor)
                # L2-normalise so cosine similarity = dot product
                features = features / features.norm(dim=-1, keepdim=True)

            return features[0].cpu().tolist()

        except Exception as e:
            logger.warning("Visual embedding failed: %s", e)
            return None

    # ── Similarity & interpretation ───────────────────────────────────────────

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Dot product of two L2-normalised vectors = cosine similarity."""
        return float(sum(a * b for a, b in zip(vec_a, vec_b)))

    @staticmethod
    def interpret_similarity(similarity: float) -> str:
        """Return a human-readable label for a similarity score."""
        if similarity >= SIMILARITY_HIGH:
            return "LOW"       # low visual change
        if similarity >= SIMILARITY_MEDIUM:
            return "MODERATE"
        return "HIGH"          # high visual change

    @staticmethod
    def detect_contradiction(
        similarity: float,
        previous_severity: float,
        current_severity: float,
    ) -> Optional[str]:
        """
        Return a contradiction warning string if severity and visual signals
        disagree, or None if they are consistent.

        Cases:
        - Severity improved significantly BUT visual change is high
          → possible different area photographed / actual worsening masked
        - Severity worsened BUT visual change is low
          → possible same lesion but patient reports worse symptoms
        """
        severity_delta = current_severity - previous_severity
        visual_change  = 1.0 - similarity

        # Case 1: severity improved but image looks very different
        if (severity_delta <= -CONTRADICTION_SEVERITY_IMPROVEMENT * previous_severity
                and visual_change >= CONTRADICTION_VISUAL_CHANGE):
            return (
                f"CONTRADICTION DETECTED: Severity score improved "
                f"({previous_severity:.1f} → {current_severity:.1f}) but visual "
                f"appearance changed significantly (similarity={similarity:.2f}). "
                f"Possible causes: different lesion area photographed, new lesion "
                f"development, or under-reported worsening. Investigate carefully."
            )

        # Case 2: severity worsened but image looks almost the same
        if (severity_delta >= CONTRADICTION_SEVERITY_IMPROVEMENT * previous_severity
                and visual_change < (1.0 - SIMILARITY_HIGH)):
            return (
                f"MILD CONTRADICTION: Severity score worsened "
                f"({previous_severity:.1f} → {current_severity:.1f}) but visual "
                f"appearance is largely unchanged (similarity={similarity:.2f}). "
                f"Possible causes: increased symptoms without visible change "
                f"(itch, pain) or early-stage progression not yet visible."
            )

        return None
