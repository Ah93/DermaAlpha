"""
agent/lesion_classifier.py — Lightweight skin lesion pre-classifier.

Runs a HuggingFace image-classification pipeline on the uploaded skin image
BEFORE the LLM call, producing top-3 predictions that are injected into the
prompt as structured hints.  Falls back silently if the model cannot load.
"""

import base64
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ViT model fine-tuned on 23-class skin disease dataset
_MODEL_ID = "Anwarkh1/Skin_Disease-Image_Classification"


class LesionClassifier:
    """
    Wraps a HuggingFace image-classification pipeline for skin conditions.
    Constructor never raises — sets _ready=False and logs a warning on failure.
    """

    def __init__(self):
        self._pipeline = None
        self._ready = False
        self._load()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load(self):
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading skin lesion classifier: %s", _MODEL_ID)
            self._pipeline = hf_pipeline(
                "image-classification",
                model=_MODEL_ID,
                top_k=3,
            )
            self._ready = True
            logger.info("Lesion classifier ready.")
        except Exception as e:
            logger.warning(
                "Lesion classifier failed to load (%s). "
                "Image pre-classification will be skipped.",
                e,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, image_data_uri: Optional[str]) -> Optional[dict]:
        """
        Classify a skin image and return a dict with top predictions.

        Returns:
            {
              "top_label": "Eczema",
              "top_score": 87.3,         # percentage
              "predictions": [
                {"label": "Eczema",  "score": 87.3},
                {"label": "Contact Dermatitis", "score": 7.1},
                {"label": "Psoriasis",          "score": 3.2},
              ]
            }
            or None if classifier unavailable / image decoding fails.
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
            results = self._pipeline(image)

            predictions = [
                {
                    "label": r["label"].replace("_", " ").replace("-", " ").title(),
                    "score": round(r["score"] * 100, 1),
                }
                for r in results
            ]

            return {
                "top_label": predictions[0]["label"],
                "top_score": predictions[0]["score"],
                "predictions": predictions,
            }

        except Exception as e:
            logger.warning("Lesion classification failed: %s", e)
            return None

    @property
    def ready(self) -> bool:
        return self._ready
