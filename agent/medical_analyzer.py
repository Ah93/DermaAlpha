"""
agent/medical_analyzer.py — Interface with MedGemma.
Supports three backends:
  - "ollama"  : Local Ollama server (recommended — free, handles GGUF quantisation)
  - "api"     : HuggingFace Serverless Inference API (MedGemma not supported on free tier)
  - "local"   : transformers pipeline (needs GPU / large RAM)
"""

import json
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from agent.prompts import SYSTEM_PROMPT
from config import Settings

logger = logging.getLogger(__name__)


# ── Result model ──────────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    raw_text: str = ""
    diagnosis: str = "Unknown condition"
    differential: list[str] = Field(default_factory=list)
    severity_score: float = 5.0
    body_region: str = "unspecified"
    condition_tags: list[str] = Field(default_factory=list)
    treatment_plan: str = ""
    urgent: bool = False
    progress: str = "first_encounter"

    @field_validator("severity_score")
    @classmethod
    def clamp_severity(cls, v: float) -> float:
        return max(1.0, min(10.0, float(v)))

    @field_validator("progress")
    @classmethod
    def validate_progress(cls, v: str) -> str:
        return v if v in {"first_encounter", "improving", "worsening", "stable"} else "first_encounter"


# ── Main analyzer ─────────────────────────────────────────────────────────────

class MedicalAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mode = settings.hf_inference_mode
        logger.info("MedicalAnalyzer backend: %s", self.mode)

        if self.mode == "ollama":
            self._init_ollama()
        elif self.mode == "api":
            self._init_hf_api()
        elif self.mode == "local":
            self._init_local()
        elif self.mode == "gemini":
            self._init_gemini()

    # ── Backend initialisers ──────────────────────────────────────────────────

    def _init_ollama(self):
        """Verify Ollama is reachable and the model is available."""
        import requests
        url = self.settings.ollama_base_url.rstrip("/")
        try:
            r = requests.get(f"{url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.settings.ollama_model in m for m in models):
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s. "
                    "Run: ollama pull %s",
                    self.settings.ollama_model, models, self.settings.ollama_model,
                )
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {url}.\n"
                "Make sure Ollama is installed and running:\n"
                "  1. Download from https://ollama.com\n"
                "  2. Run: ollama serve\n"
                f"  3. Run: ollama pull {self.settings.ollama_model}\n"
                f"Error: {e}"
            ) from e
        logger.info("Ollama is reachable. Using model: %s", self.settings.ollama_model)

    def _init_gemini(self):
        """Initialise Google Gemini API client (google-genai SDK)."""
        try:
            from google import genai
        except ImportError:
            raise RuntimeError(
                "google-genai package not installed.\n"
                "Run: pip install google-genai"
            )
        if not self.settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set in your .env file.\n"
                "Get a key at https://aistudio.google.com/app/apikey"
            )
        self._gemini_client = genai.Client(api_key=self.settings.gemini_api_key)
        logger.info("Gemini backend ready. Model: %s", self.settings.gemini_model)

    def _init_hf_api(self):
        from huggingface_hub import InferenceClient
        self._hf_client = InferenceClient(
            model=self.settings.hf_model_id,
            token=self.settings.hf_token,
        )

    def _init_local(self):
        """Load model weights locally via transformers (needs GPU or large RAM)."""
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        logger.info("Loading MedGemma locally — this may take several minutes…")
        self._processor = AutoProcessor.from_pretrained(
            self.settings.hf_model_id, token=self.settings.hf_token
        )
        self._local_model = AutoModelForImageTextToText.from_pretrained(
            self.settings.hf_model_id,
            token=self.settings.hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def analyze(
        self,
        prompt: str,
        image_data_uri: Optional[str] = None,
        max_new_tokens: int = 1500,
    ) -> AnalysisResult:
        if self.mode == "ollama":
            raw = self._call_ollama(prompt, image_data_uri, max_new_tokens)
        elif self.mode == "gemini":
            raw = self._call_gemini(prompt, image_data_uri, max_new_tokens)
        elif self.mode == "api":
            raw = self._call_hf_api(prompt, image_data_uri, max_new_tokens)
        elif self.mode == "local":
            raw = self._call_local(prompt, image_data_uri, max_new_tokens)
        else:
            raise RuntimeError(f"Unknown inference mode: {self.mode}")

        return self._parse_response(raw)

    # ── Ollama backend ────────────────────────────────────────────────────────

    def _call_ollama(
        self,
        prompt: str,
        image_data_uri: Optional[str],
        max_new_tokens: int,
    ) -> str:
        import requests

        base_url = self.settings.ollama_base_url.rstrip("/")

        # Merge system prompt into the user prompt (works with all GGUF models
        # regardless of whether they have a chat template)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        # Extract base64 image (strip data URI prefix if present)
        images = []
        if image_data_uri and "," in image_data_uri:
            images = [image_data_uri.split(",", 1)[1]]

        payload = {
            "model": self.settings.ollama_model,
            "prompt": full_prompt,
            "images": images,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": max_new_tokens,
            },
        }

        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=300,
            )
            if not response.ok:
                # Log the actual Ollama error body for easier debugging
                try:
                    err_body = response.json()
                except Exception:
                    err_body = response.text
                raise RuntimeError(
                    f"Ollama returned {response.status_code}: {err_body}"
                )
            return response.json()["response"]
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                "Ollama is not running. Open a terminal and run: ollama serve"
            ) from e
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

    # ── Gemini API backend ────────────────────────────────────────────────────

    def _call_gemini(
        self,
        prompt: str,
        image_data_uri: Optional[str],
        max_new_tokens: int,
    ) -> str:
        import base64
        from google import genai
        from google.genai import types

        parts = [types.Part(text=f"{SYSTEM_PROMPT}\n\n{prompt}")]

        if image_data_uri and "," in image_data_uri:
            header, b64 = image_data_uri.split(",", 1)
            mime_type = "image/jpeg"
            if "image/png"  in header: mime_type = "image/png"
            elif "image/webp" in header: mime_type = "image/webp"
            elif "image/bmp"  in header: mime_type = "image/bmp"
            parts.append(types.Part(
                inline_data=types.Blob(
                    mime_type=mime_type,
                    data=base64.b64decode(b64),
                )
            ))

        try:
            response = self._gemini_client.models.generate_content(
                model=self.settings.gemini_model,
                contents=[types.Content(parts=parts)],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=max_new_tokens,
                    # Disable thinking — saves token budget for actual response
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            # Extract text robustly — handle multi-part responses
            if response.text:
                return response.text
            # Fallback: concatenate all text parts from first candidate
            candidate = response.candidates[0]
            return "".join(
                p.text for p in candidate.content.parts if hasattr(p, "text") and p.text
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    # ── HuggingFace API backend ───────────────────────────────────────────────

    def _call_hf_api(
        self,
        prompt: str,
        image_data_uri: Optional[str],
        max_new_tokens: int,
    ) -> str:
        if image_data_uri:
            content = [
                {"type": "image_url", "image_url": {"url": image_data_uri}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        try:
            response = self._hf_client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=max_new_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(
                f"Failed to get response from MedGemma: {e}\n"
                "Note: HuggingFace Serverless Inference API does not support MedGemma.\n"
                "Switch to HF_INFERENCE_MODE=ollama in your .env file."
            ) from e

    # ── Local transformers backend ────────────────────────────────────────────

    def _call_local(
        self,
        prompt: str,
        image_data_uri: Optional[str],
        max_new_tokens: int,
    ) -> str:
        import base64, io, torch
        from PIL import Image

        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        if image_data_uri and "," in image_data_uri:
            b64 = image_data_uri.split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            inputs = self._processor(
                text=full_prompt, images=image, return_tensors="pt"
            ).to(self._local_model.device)
        else:
            inputs = self._processor(
                text=full_prompt, return_tensors="pt"
            ).to(self._local_model.device)

        with torch.no_grad():
            output_ids = self._local_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self._processor.decode(output_ids[0], skip_special_tokens=True)

    # ── Response parser ───────────────────────────────────────────────────────

    def _parse_response(self, raw_text: str) -> AnalysisResult:
        json_match = re.search(
            r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL | re.IGNORECASE
        )
        if not json_match:
            json_match = re.search(r'\{\s*"diagnosis"\s*:.*?\}', raw_text, re.DOTALL)

        structured: dict = {}
        if json_match:
            try:
                structured = json.loads(
                    json_match.group(1) if "```" in raw_text else json_match.group(0)
                )
            except json.JSONDecodeError as e:
                logger.warning("Could not parse JSON from model response: %s", e)

        return AnalysisResult(
            raw_text=raw_text,
            diagnosis=structured.get("diagnosis", "See full response"),
            differential=structured.get("differential", []),
            severity_score=structured.get("severity_score", 5.0),
            body_region=structured.get("body_region", "unspecified"),
            condition_tags=structured.get("condition_tags", []),
            treatment_plan=structured.get("treatment_plan", ""),
            urgent=structured.get("urgent", False),
            progress=structured.get("progress", "first_encounter"),
        )
